"""Detect data drift: is production data starting to look different from the reference data?

Two checks run for every feature in every production batch:

KS test (Kolmogorov-Smirnov)
    Compares the full shape of two distributions. The KS *statistic* is the
    largest gap between the two cumulative distributions (0 = identical,
    1 = completely separate). The *p-value* answers: "if nothing had changed,
    how likely is a gap this big by pure chance?" A p-value below KS_ALPHA
    (0.05) flags the feature as drifted.

PSI (Population Stability Index)
    Cuts the reference data into PSI_N_BINS bins at its quantiles (so each
    bin holds ~10% of the reference values). It then measures how much the
    share of production values in each bin has moved:
        PSI = sum over bins of (current% - reference%) * ln(current% / reference%)
    The bins are fitted on the reference data only. The outer bins are open
    ended, so production values beyond the reference range still land in a
    bin. Empty bins are raised to PSI_EPSILON to avoid dividing by zero.

Run from the project folder (after `python -m monitoring.simulate run`):
    python -m monitoring.drift run
    python -m monitoring.drift report
    python -m monitoring.drift report --batch-id 3
"""

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from monitoring import config, storage
from monitoring.data import validate_dataset
from monitoring.simulate import load_production_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DriftSettings:
    ks_alpha: float = config.KS_ALPHA
    psi_n_bins: int = config.PSI_N_BINS
    psi_warning_threshold: float = config.PSI_WARNING_THRESHOLD
    psi_drift_threshold: float = config.PSI_DRIFT_THRESHOLD
    psi_epsilon: float = config.PSI_EPSILON
    min_sample_size: int = config.MIN_DRIFT_SAMPLE_SIZE

    def __post_init__(self):
        if not 0 < self.ks_alpha < 1:
            raise ValueError("ks_alpha must be between 0 and 1.")
        if self.psi_n_bins < 2:
            raise ValueError("psi_n_bins must be at least 2.")
        if not 0 < self.psi_warning_threshold < self.psi_drift_threshold:
            raise ValueError("PSI thresholds must satisfy 0 < warning < drift.")
        if not 0 < self.psi_epsilon < 1:
            raise ValueError("psi_epsilon must be between 0 and 1.")
        if self.min_sample_size < 1:
            raise ValueError("min_sample_size must be at least 1.")


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------
def clean_values(values) -> tuple[np.ndarray, int]:
    """Convert to a float array and drop missing/infinite values.

    Returns (usable values, number of values dropped).
    """
    try:
        array = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Drift checks need numeric values: {exc}") from exc

    usable = np.isfinite(array)
    return array[usable], int((~usable).sum())


def fit_psi_bin_edges(reference: np.ndarray, n_bins: int = config.PSI_N_BINS) -> np.ndarray:
    """Return the inner bin edges, cut at quantiles of the reference data.

    With 10 bins the edges are the 10th, 20th, ..., 90th percentiles. The
    first bin is everything below the first edge and the last bin is
    everything above the last edge, so no value can fall outside the bins.
    Repeated edges (from many identical values) are merged.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if reference.size == 0:
        raise ValueError("Cannot fit PSI bins on an empty reference.")

    if reference.min() == reference.max():
        # Constant reference: use three bins, "below", "equal" and "above"
        # the constant, so any change is visible.
        value = reference[0]
        return np.array([value, np.nextafter(value, np.inf)])

    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(reference, quantiles))


def bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the share of values in each bin (bin i holds edges[i-1] <= x < edges[i])."""
    bin_index = np.searchsorted(edges, values, side="right")
    counts = np.bincount(bin_index, minlength=len(edges) + 1)
    return counts / counts.sum()


def population_stability_index(
    reference_proportions: np.ndarray,
    current_proportions: np.ndarray,
    epsilon: float = config.PSI_EPSILON,
) -> float:
    """PSI between two sets of bin proportions. 0 means identical; it is never negative."""
    reference_proportions = np.asarray(reference_proportions, dtype=float)
    current_proportions = np.asarray(current_proportions, dtype=float)
    if reference_proportions.shape != current_proportions.shape:
        raise ValueError("Both sets of proportions must have the same number of bins.")

    # Empty bins would give log(0) or division by zero, so raise them to epsilon.
    ref = np.clip(reference_proportions, epsilon, None)
    cur = np.clip(current_proportions, epsilon, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


# ---------------------------------------------------------------------------
# Reference profile (fitted once from the reference data)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureReference:
    name: str
    values: np.ndarray       # usable reference values, used by the KS test
    bin_edges: np.ndarray    # PSI bin edges fitted on the reference
    proportions: np.ndarray  # share of reference values in each PSI bin


def build_reference_profile(
    reference_df: pd.DataFrame,
    features: list[str] = config.DRIFT_FEATURES,
    settings: DriftSettings = DriftSettings(),
) -> dict[str, FeatureReference]:
    """Prepare everything needed to compare batches against the reference data."""
    missing = [f for f in features if f not in reference_df.columns]
    if missing:
        raise ValueError(f"Reference data is missing feature columns: {missing}")

    profile = {}
    for feature in features:
        values, n_dropped = clean_values(reference_df[feature])
        if values.size < settings.min_sample_size:
            raise ValueError(
                f"Reference feature '{feature}' has only {values.size} usable values "
                f"(need at least {settings.min_sample_size})."
            )
        if n_dropped:
            logger.warning("Reference feature '%s': ignored %d missing/infinite values.", feature, n_dropped)

        edges = fit_psi_bin_edges(values, settings.psi_n_bins)
        profile[feature] = FeatureReference(
            name=feature,
            values=values,
            bin_edges=edges,
            proportions=bin_proportions(values, edges),
        )
    return profile


# ---------------------------------------------------------------------------
# Comparing one feature / one batch
# ---------------------------------------------------------------------------
def compare_feature(reference: FeatureReference, current_values, settings: DriftSettings = DriftSettings()) -> list[dict]:
    """Run the KS test and PSI for one feature. Returns one result dict per test."""
    current, n_missing = clean_values(current_values)
    common = {
        "feature_name": reference.name,
        "n_reference": int(reference.values.size),
        "n_current": int(current.size),
        "n_missing": n_missing,
    }
    ks_result = {**common, "test_name": "ks", "threshold": settings.ks_alpha, "warning_threshold": None}
    psi_result = {
        **common,
        "test_name": "psi",
        "threshold": settings.psi_drift_threshold,
        "warning_threshold": settings.psi_warning_threshold,
        "p_value": None,
    }

    if current.size < settings.min_sample_size:
        detail = f"Only {current.size} usable values (need at least {settings.min_sample_size})."
        for result in (ks_result, psi_result):
            result.update(score=None, status="insufficient_data", detail=detail)
        ks_result["p_value"] = None
        return [ks_result, psi_result]

    detail = f"{n_missing} missing/infinite values ignored." if n_missing else None

    # KS test: deterministic (no random sampling involved).
    ks = ks_2samp(reference.values, current)
    p_value = float(ks.pvalue)
    ks_result.update(
        score=float(ks.statistic),
        p_value=p_value,
        status="drift" if p_value < settings.ks_alpha else "no_drift",
        detail=detail,
    )

    # PSI, using the bins fitted on the reference data.
    psi = population_stability_index(
        reference.proportions, bin_proportions(current, reference.bin_edges), settings.psi_epsilon
    )
    if psi >= settings.psi_drift_threshold:
        psi_status = "drift"
    elif psi >= settings.psi_warning_threshold:
        psi_status = "warning"
    else:
        psi_status = "no_drift"
    psi_result.update(score=psi, status=psi_status, detail=detail)

    return [ks_result, psi_result]


def detect_batch_drift(
    profile: dict[str, FeatureReference],
    batch_df: pd.DataFrame,
    settings: DriftSettings = DriftSettings(),
) -> list[dict]:
    """Compare every profiled feature of one batch against the reference."""
    missing = [f for f in profile if f not in batch_df.columns]
    if missing:
        raise ValueError(f"Batch is missing feature columns: {missing}")

    results = []
    for feature, reference in profile.items():
        results.extend(compare_feature(reference, batch_df[feature], settings))
    return results


# ---------------------------------------------------------------------------
# Running over the simulator's batches
# ---------------------------------------------------------------------------
def load_reference_data(path: Path | str = config.REFERENCE_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference data not found at {path}. Create it first with: python -m monitoring.train"
        )
    df = pd.read_csv(path)
    validate_dataset(df)
    return df


def run_drift_detection(
    db_path: Path | str = config.DB_PATH,
    reference_path: Path | str = config.REFERENCE_DATA_PATH,
    production_path: Path | str = config.PRODUCTION_DATA_PATH,
    batch_ids: list[int] | None = None,
    recompute: bool = False,
    settings: DriftSettings = DriftSettings(),
) -> dict:
    """Check every completed simulator batch for drift and store the results.

    Batches that already have results are skipped unless recompute=True.
    Recomputing replaces the old rows; it never creates duplicates.
    """
    profile = build_reference_profile(load_reference_data(reference_path), settings=settings)

    # Same rows and same transaction IDs the simulator used. The label
    # column is dropped: drift checks only look at model inputs.
    production = load_production_data(production_path)
    features = production.drop(columns=[config.TARGET_COLUMN]).set_index("transaction_id")

    processed, skipped = [], []
    with storage.open_db(db_path) as conn:
        completed = storage.get_completed_batch_ids(conn)
        if not completed:
            logger.info("No completed batches found. Run `python -m monitoring.simulate run` first.")

        if batch_ids is not None:
            unknown = sorted(set(batch_ids) - set(completed))
            if unknown:
                raise ValueError(f"These batch IDs are not completed batches: {unknown}")
            completed = [b for b in completed if b in set(batch_ids)]

        already_done = set() if recompute else storage.get_batch_ids_with_drift_results(conn)

        for batch_id in completed:
            if batch_id in already_done:
                skipped.append(batch_id)
                continue

            transaction_ids = storage.get_batch_transaction_ids(conn, batch_id)
            batch_df = features.reindex(transaction_ids)  # unknown IDs become missing values
            n_unmatched = int(batch_df.isna().all(axis=1).sum())
            if n_unmatched:
                logger.warning(
                    "Batch %d: %d transactions were not found in the production file "
                    "(was the model retrained?). They count as missing values.",
                    batch_id, n_unmatched,
                )

            results = detect_batch_drift(profile, batch_df, settings)
            storage.save_drift_results(conn, batch_id, results, datetime.now(timezone.utc).isoformat())
            processed.append(batch_id)
            log_batch_summary(batch_id, results)

    if skipped:
        logger.info("Skipped %d batches that already had drift results (use --recompute to redo).", len(skipped))
    return {"processed": processed, "skipped": skipped}


def log_batch_summary(batch_id: int, results: list[dict]) -> None:
    df = pd.DataFrame(results)
    ks = df[df["test_name"] == "ks"]
    psi = df[df["test_name"] == "psi"]
    logger.info(
        "Batch %d: KS drift in %d/%d features | PSI drift %d, warning %d | insufficient data %d",
        batch_id,
        (ks["status"] == "drift").sum(), len(ks),
        (psi["status"] == "drift").sum(), (psi["status"] == "warning").sum(),
        (df["status"] == "insufficient_data").sum() // 2,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize_drift(results: pd.DataFrame) -> pd.DataFrame:
    """One row per batch: how many features each test flagged."""
    def count(test, status):
        subset = results[(results["test_name"] == test) & (results["status"] == status)]
        return subset.groupby("batch_id").size()

    psi = results[results["test_name"] == "psi"]
    summary = pd.DataFrame(
        {
            "features": results[results["test_name"] == "ks"].groupby("batch_id").size(),
            "ks_drift": count("ks", "drift"),
            "psi_warning": count("psi", "warning"),
            "psi_drift": count("psi", "drift"),
            "max_psi": psi.groupby("batch_id")["score"].max(),
        }
    )
    count_columns = ["features", "ks_drift", "psi_warning", "psi_drift"]
    summary[count_columns] = summary[count_columns].fillna(0).astype(int)
    scored = psi.dropna(subset=["score"])
    top = scored.loc[scored.groupby("batch_id")["score"].idxmax()]
    summary["top_psi_feature"] = top.set_index("batch_id")["feature_name"]
    return summary


def print_report(db_path: Path | str, batch_id: int | None = None) -> None:
    with storage.open_db_readonly(db_path) as conn:
        results = storage.get_drift_results(conn, batch_id)

    if results.empty:
        print("No drift results yet. Run: python -m monitoring.drift run")
        return

    pd.set_option("display.width", 140)
    if batch_id is None:
        print("\n=== Drift summary per batch ===")
        print(f"(KS flags p-value < {config.KS_ALPHA}; PSI warning >= {config.PSI_WARNING_THRESHOLD}, "
              f"drift >= {config.PSI_DRIFT_THRESHOLD})\n")
        print(summarize_drift(results).to_string(float_format=lambda v: f"{v:.4f}"))
        return

    print(f"\n=== Drift details for batch {batch_id} ===\n")
    ks = results[results["test_name"] == "ks"].set_index("feature_name")
    psi = results[results["test_name"] == "psi"].set_index("feature_name")
    table = pd.DataFrame(
        {
            "ks_statistic": ks["score"],
            "ks_p_value": ks["p_value"],
            "ks_status": ks["status"],
            "psi": psi["score"],
            "psi_status": psi["status"],
            "n_current": ks["n_current"],
        }
    ).loc[[f for f in config.DRIFT_FEATURES if f in ks.index]]
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect data drift in production batches.")
    parser.add_argument("--db", type=Path, default=config.DB_PATH, help="SQLite database file")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="check completed batches for drift")
    run.add_argument("--batch-id", type=int, action="append", help="only this batch (can repeat)")
    run.add_argument("--recompute", action="store_true", help="redo batches that already have results")
    run.add_argument("--reference-data", type=Path, default=config.REFERENCE_DATA_PATH)
    run.add_argument("--production-data", type=Path, default=config.PRODUCTION_DATA_PATH)

    report = commands.add_parser("report", help="show stored drift results")
    report.add_argument("--batch-id", type=int, default=None, help="show per-feature details for one batch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        if args.command == "run":
            run_drift_detection(
                db_path=args.db,
                reference_path=args.reference_data,
                production_path=args.production_data,
                batch_ids=args.batch_id,
                recompute=args.recompute,
            )
            print_report(args.db)
        else:
            print_report(args.db, args.batch_id)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
