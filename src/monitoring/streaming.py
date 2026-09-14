"""Streaming drift detection with River's ADWIN and Page-Hinkley detectors.

Phase 3 (drift.py) compares each *batch* of input features with the
reference data. This module instead watches a *stream* of single numbers,
one per transaction in time order, and reports the moment its average
changes.

Signals (one value per transaction):
    fraud_probability  model score, 0-1                 available immediately
    flag_rate          1 if flagged as fraud, else 0    available immediately
    error_rate         1 if prediction was wrong, else 0  only after the true label arrives

Only error_rate says anything about model *performance*. A change in
fraud_probability or flag_rate means the model's outputs changed, which
can happen while accuracy stays the same.

Delayed labels: error_rate uses a batch only once every transaction in it
has its true label, and stops at the first batch that is not fully labeled,
so errors are always processed in time order.

Detector state is NOT saved between runs. Each run replays every logged
transaction from the beginning, in time order, through fresh detectors.
The detectors are deterministic and logged predictions/labels never
change, so a replay gives the same results for batches already stored;
those rows are skipped and only new batches are added.

Run from the project folder:
    python -m monitoring.streaming run
    python -m monitoring.streaming report
    python -m monitoring.streaming report --signal error_rate
"""

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from river import drift as river_drift

from monitoring import config, storage

logger = logging.getLogger(__name__)

DETECTORS = ("adwin", "page_hinkley")

SIGNALS = {
    "fraud_probability": {
        "requires_labels": False,
        "description": "Model fraud score per transaction (prediction drift, not performance).",
    },
    "flag_rate": {
        "requires_labels": False,
        "description": "1 if the transaction was flagged as fraud, else 0 (alert volume, not performance).",
    },
    "error_rate": {
        "requires_labels": True,
        "description": "1 if the prediction did not match the true label, else 0 (model performance).",
    },
}

MAX_STORED_DETECTIONS = 20  # detection details kept per batch row


# ---------------------------------------------------------------------------
# Settings and detectors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StreamSettings:
    adwin_delta: float = config.ADWIN_DELTA
    ph_min_instances: int = config.PAGE_HINKLEY_MIN_INSTANCES
    ph_delta: float = config.PAGE_HINKLEY_DELTA
    ph_threshold: float = config.PAGE_HINKLEY_THRESHOLD
    ph_alpha: float = config.PAGE_HINKLEY_ALPHA
    ph_mode: str = config.PAGE_HINKLEY_MODE

    def __post_init__(self):
        if not 0 < self.adwin_delta < 1:
            raise ValueError("adwin_delta must be between 0 and 1.")
        if self.ph_min_instances < 1:
            raise ValueError("ph_min_instances must be at least 1.")
        if self.ph_delta < 0:
            raise ValueError("ph_delta cannot be negative.")
        if self.ph_threshold <= 0:
            raise ValueError("ph_threshold must be positive.")
        if not 0 < self.ph_alpha <= 1:
            raise ValueError("ph_alpha must be in (0, 1].")
        if self.ph_mode not in {"up", "down", "both"}:
            raise ValueError("ph_mode must be 'up', 'down' or 'both'.")


def make_detector(name: str, settings: StreamSettings = StreamSettings()):
    """Create a fresh River detector."""
    if name == "adwin":
        return river_drift.ADWIN(delta=settings.adwin_delta)
    if name == "page_hinkley":
        return river_drift.PageHinkley(
            min_instances=settings.ph_min_instances,
            delta=settings.ph_delta,
            threshold=settings.ph_threshold,
            alpha=settings.ph_alpha,
            mode=settings.ph_mode,
        )
    raise ValueError(f"Unknown detector '{name}'. Choose from {DETECTORS}.")


def detector_settings(name: str, settings: StreamSettings) -> dict:
    prefix = "adwin_" if name == "adwin" else "ph_"
    return {k: v for k, v in asdict(settings).items() if k.startswith(prefix)}


# ---------------------------------------------------------------------------
# Turning logged predictions into signal values
# ---------------------------------------------------------------------------
def signal_values(predictions: pd.DataFrame, signal: str) -> pd.Series:
    """Return one float per row for the signal; NaN where it is missing or invalid."""
    if signal not in SIGNALS:
        raise ValueError(f"Unknown signal '{signal}'. Choose from {list(SIGNALS)}.")

    def numeric(column: str) -> pd.Series:
        if column not in predictions.columns:
            raise ValueError(f"Signal '{signal}' needs the column '{column}'.")
        return pd.to_numeric(predictions[column], errors="coerce").astype(float)

    if signal == "fraud_probability":
        values = numeric("fraud_probability")
        return values.where(values.between(0, 1))

    predicted = numeric("predicted_label")
    predicted = predicted.where(predicted.isin([0, 1]))
    if signal == "flag_rate":
        return predicted

    # error_rate: only defined once the true label is known.
    actual = numeric("actual_label")
    actual = actual.where(actual.isin([0, 1]))
    error = (predicted != actual).astype(float)
    return error.where(predicted.notna() & actual.notna())


# ---------------------------------------------------------------------------
# Feeding a stream through one detector
# ---------------------------------------------------------------------------
def monitor_stream(events: pd.DataFrame, detector) -> list[dict]:
    """Feed values to a detector in time order and summarise each batch.

    `events` needs the columns batch_id, transaction_id, event_time, value.
    Missing or non-finite values are skipped and counted as invalid.
    """
    required = {"batch_id", "transaction_id", "event_time", "value"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Stream events are missing columns: {sorted(missing)}")

    ordered = events.sort_values(["event_time", "transaction_id"], kind="stable")
    batch_order = ordered["batch_id"].to_numpy()
    if (np.diff(batch_order) < 0).any():
        raise ValueError("Batches overlap in time: a later batch has events before an earlier batch.")

    results = []
    for batch_id, group in ordered.groupby("batch_id", sort=False):
        values = pd.to_numeric(group["value"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)

        detections = []
        for transaction_id, event_time, value in zip(
            group["transaction_id"].to_numpy()[valid], group["event_time"].to_numpy()[valid], values[valid]
        ):
            detector.update(float(value))
            if detector.drift_detected:
                detections.append({"transaction_id": str(transaction_id), "event_time": float(event_time)})

        n_events = int(valid.sum())
        if n_events == 0:
            status = "no_data"
        else:
            status = "drift" if detections else "stable"

        info = {"detections": detections[:MAX_STORED_DETECTIONS]}
        if isinstance(detector, river_drift.ADWIN):
            info.update(window_mean=float(detector.estimation), window_width=int(detector.width))

        results.append(
            {
                "batch_id": int(batch_id),
                "status": status,
                "n_events": n_events,
                "n_invalid": int((~valid).sum()),
                "n_detections": len(detections),
                "signal_mean": float(values[valid].mean()) if n_events else None,
                "first_detection_transaction_id": detections[0]["transaction_id"] if detections else None,
                "first_detection_event_time": detections[0]["event_time"] if detections else None,
                "detector_info": info,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Running over the logged predictions
# ---------------------------------------------------------------------------
def label_complete_batches(predictions: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Split batches into (ready for error monitoring, still waiting for labels).

    Ready = every transaction in the batch has a true label, and so does
    every earlier batch. This keeps the error stream in time order.
    """
    ready, waiting = [], []
    for batch_id, group in predictions.groupby("batch_id", sort=True):
        if not waiting and group["actual_label"].notna().all():
            ready.append(int(batch_id))
        else:
            waiting.append(int(batch_id))
    return ready, waiting


def run_streaming_detection(
    db_path: Path | str = config.DB_PATH,
    recompute: bool = False,
    settings: StreamSettings = StreamSettings(),
) -> dict:
    """Replay logged predictions through both detectors for every signal and store the results."""
    computed_at = datetime.now(timezone.utc).isoformat()
    new_rows = 0

    with storage.open_db(db_path) as conn:
        if recompute:
            removed = storage.delete_stream_results(conn)
            logger.info("Removed %d old streaming results before recomputing.", removed)

        predictions = storage.get_predictions_with_labels(conn)
        completed = storage.get_completed_batch_ids(conn)
        predictions = predictions[predictions["batch_id"].isin(completed)]
        if predictions.empty:
            logger.info("No logged predictions found. Run `python -m monitoring.simulate run` first.")

        ready, waiting = label_complete_batches(predictions)
        if waiting:
            logger.info(
                "error_rate: %d batches are fully labeled; %d batches are waiting for labels (from batch %d).",
                len(ready), len(waiting), waiting[0],
            )

        for signal, spec in SIGNALS.items():
            rows = predictions[predictions["batch_id"].isin(ready)] if spec["requires_labels"] else predictions
            if rows.empty:
                continue

            events = rows[["batch_id", "transaction_id", "event_time"]].assign(value=signal_values(rows, signal))
            for detector_name in DETECTORS:
                results = monitor_stream(events, make_detector(detector_name, settings))
                for result in results:
                    result["detector_info"] = json.dumps(
                        {"settings": detector_settings(detector_name, settings), **result["detector_info"]}
                    )
                inserted = storage.save_stream_results(conn, detector_name, signal, results, computed_at)
                new_rows += inserted

                n_drift = sum(r["status"] == "drift" for r in results)
                logger.info(
                    "%-17s %-12s %d batches, %d with detections (%d new rows stored).",
                    signal, detector_name, len(results), n_drift, inserted,
                )

    return {"new_rows": new_rows, "label_ready_batches": ready, "waiting_for_labels": waiting}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(db_path: Path | str, signal: str | None = None) -> None:
    with storage.open_db_readonly(db_path) as conn:
        results = storage.get_stream_results(conn, signal=signal)

    if results.empty:
        print("No streaming results yet. Run: python -m monitoring.streaming run")
        return

    pd.set_option("display.width", 140)
    if signal is None:
        print("\n=== Streaming detectors: summary ===\n")
        summary = (
            results.groupby(["signal", "detector"])
            .agg(
                batches=("batch_id", "count"),
                last_batch=("batch_id", "max"),
                batches_with_change=("status", lambda s: int((s == "drift").sum())),
                detections=("n_detections", "sum"),
                first_change_batch=("batch_id", lambda b: _first_drift_batch(results.loc[b.index])),
            )
        )
        print(summary.to_string())
        print("\nfraud_probability / flag_rate: changes in model OUTPUTS (not performance).")
        print("error_rate: changes in model MISTAKES, using only batches whose true labels have arrived.")
        return

    print(f"\n=== Streaming detectors: {signal} ===")
    print(SIGNALS.get(signal, {}).get("description", ""), "\n")
    table = results.pivot(index="batch_id", columns="detector", values="status")
    table.insert(0, "signal_mean", results.groupby("batch_id")["signal_mean"].first())
    table.insert(1, "n_events", results.groupby("batch_id")["n_events"].first())
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))


def _first_drift_batch(group: pd.DataFrame):
    drifted = group.loc[group["status"] == "drift", "batch_id"]
    return int(drifted.min()) if not drifted.empty else "-"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming drift detection with ADWIN and Page-Hinkley.")
    parser.add_argument("--db", type=Path, default=config.DB_PATH, help="SQLite database file")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="replay logged predictions through the detectors")
    run.add_argument("--recompute", action="store_true", help="delete stored results and rebuild them")

    report = commands.add_parser("report", help="show stored detector results")
    report.add_argument("--signal", choices=list(SIGNALS), default=None, help="per-batch details for one signal")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        if args.command == "run":
            run_streaming_detection(db_path=args.db, recompute=args.recompute)
            print_report(args.db)
        else:
            print_report(args.db, args.signal)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
