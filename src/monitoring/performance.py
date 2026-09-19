"""Model performance monitoring using true labels that have arrived.

How it works:
  1. Read logged predictions together with their true labels (actual_labels table).
  2. Keep only batches whose labels have ALL arrived, in order (the same
     rule the streaming error monitor uses). Predictions are never used as labels.
  3. Group those batches, oldest first, into windows. A window closes once it
     holds at least PERFORMANCE_MIN_POSITIVES fraud cases and
     PERFORMANCE_MIN_NEGATIVES non-fraud cases. Batches after the last closed
     window form an "open" window that waits for more labeled data.
  4. Calculate precision, recall, F1, PR-AUC, false positive rate and the
     confusion matrix for each closed window, store them, and check the alert rules.

Run from the project folder:
    python -m monitoring.performance run
    python -m monitoring.performance report
    python -m monitoring.performance alerts
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from monitoring import config, storage
from monitoring.alerts import evaluate_rules, load_baseline, load_rules
from monitoring.streaming import label_complete_batches

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics for one set of labeled predictions
# ---------------------------------------------------------------------------
def compute_metrics(
    actual,
    predicted,
    scores=None,
    min_positives: int = config.PERFORMANCE_MIN_POSITIVES,
    min_negatives: int = config.PERFORMANCE_MIN_NEGATIVES,
) -> dict:
    """Calculate performance metrics. A metric is None when it cannot be calculated.

    `actual` are the true labels and `predicted` the model's 0/1 decisions;
    `scores` are fraud probabilities (needed for PR-AUC). The returned
    "notes" explain every missing or unreliable metric, and "status" is
    "insufficient_data" when there are too few fraud or non-fraud cases.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same length.")
    for name, values in [("actual", actual), ("predicted", predicted)]:
        if values.size and not np.isin(values, [0, 1]).all():
            raise ValueError(f"{name} labels must be 0 or 1.")
    actual = actual.astype(int)
    predicted = predicted.astype(int)

    tp = int(((predicted == 1) & (actual == 1)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())
    n_positive, n_negative = tp + fn, fp + tn
    notes = []

    precision = tp / (tp + fp) if tp + fp else None
    if precision is None:
        notes.append("precision: no transactions were flagged as fraud.")

    recall = tp / n_positive if n_positive else None
    if recall is None:
        notes.append("recall: no actual fraud cases in this window.")

    if precision is None or recall is None:
        f1 = None
        notes.append("f1: needs both precision and recall.")
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    false_positive_rate = fp / n_negative if n_negative else None
    if false_positive_rate is None:
        notes.append("false_positive_rate: no actual non-fraud cases in this window.")

    pr_auc = None
    if scores is None:
        notes.append("pr_auc: no probability scores provided.")
    else:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != actual.shape:
            raise ValueError("scores must have the same length as actual.")
        if not (np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()):
            notes.append("pr_auc: some scores are missing or outside 0-1.")
        elif n_positive == 0 or n_negative == 0:
            notes.append("pr_auc: needs both fraud and non-fraud cases (only one class present).")
        else:
            pr_auc = float(average_precision_score(actual, scores))

    status = "ok"
    if n_positive < min_positives:
        status = "insufficient_data"
        notes.append(
            f"only {n_positive} fraud cases (need {min_positives}): "
            "recall, precision, F1 and PR-AUC are unreliable."
        )
    if n_negative < min_negatives:
        status = "insufficient_data"
        notes.append(f"only {n_negative} non-fraud cases (need {min_negatives}): false positive rate is unreliable.")

    return {
        "status": status,
        "n_labeled": int(actual.size),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "false_positive_rate": false_positive_rate,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Chronological windows
# ---------------------------------------------------------------------------
def build_windows(
    labeled: pd.DataFrame,
    ready_batches: list[int],
    min_positives: int = config.PERFORMANCE_MIN_POSITIVES,
    min_negatives: int = config.PERFORMANCE_MIN_NEGATIVES,
) -> tuple[list[list[int]], list[int]]:
    """Group fully labeled batches (oldest first) into windows with enough data.

    Returns (closed windows as lists of batch IDs, batches in the open window).
    Only true labels decide where a window ends, never the predictions.
    """
    if min_positives < 1 or min_negatives < 1:
        raise ValueError("min_positives and min_negatives must be at least 1.")

    counts = labeled.groupby("batch_id")["actual_label"].agg(positives="sum", total="size")
    windows, current = [], []
    positives = negatives = 0
    for batch_id in sorted(ready_batches):
        current.append(batch_id)
        batch_positives = int(counts.loc[batch_id, "positives"]) if batch_id in counts.index else 0
        batch_total = int(counts.loc[batch_id, "total"]) if batch_id in counts.index else 0
        positives += batch_positives
        negatives += batch_total - batch_positives
        if positives >= min_positives and negatives >= min_negatives:
            windows.append(current)
            current, positives, negatives = [], 0, 0
    return windows, current


def evaluate_window(rows: pd.DataFrame, batch_ids: list[int], **thresholds) -> dict:
    """Metrics plus identifying details for one window."""
    metrics = compute_metrics(rows["actual_label"], rows["predicted_label"], rows["fraud_probability"], **thresholds)

    decision_thresholds = sorted(rows["threshold"].unique())
    if len(decision_thresholds) > 1:
        metrics["notes"].append(f"predictions used different decision thresholds: {decision_thresholds}")

    return {
        **metrics,
        "start_batch_id": int(batch_ids[0]),
        "end_batch_id": int(batch_ids[-1]),
        "first_event_time": float(rows["event_time"].min()),
        "last_event_time": float(rows["event_time"].max()),
        "decision_threshold": float(decision_thresholds[0]) if len(decision_thresholds) == 1 else None,
        "model_versions": ",".join(sorted(rows["model_version"].unique())),
    }


# ---------------------------------------------------------------------------
# Running over the database
# ---------------------------------------------------------------------------
def run_performance_monitoring(
    db_path: Path | str = config.DB_PATH,
    metrics_path: Path | str = config.METRICS_PATH,
    recompute: bool = False,
    min_positives: int = config.PERFORMANCE_MIN_POSITIVES,
    min_negatives: int = config.PERFORMANCE_MIN_NEGATIVES,
    rules: list[dict] = config.PERFORMANCE_ALERT_RULES,
) -> dict:
    """Evaluate every closed window, store its metrics and any alerts."""
    alert_rules = load_rules(rules)
    baseline = load_baseline(metrics_path)
    now = datetime.now(timezone.utc).isoformat()
    thresholds = {"min_positives": min_positives, "min_negatives": min_negatives}

    new_windows = new_alerts = 0
    with storage.open_db(db_path) as conn:
        if recompute:
            removed_windows, removed_alerts = storage.delete_performance_results(conn)
            logger.info("Removed %d windows and %d alerts before recomputing.", removed_windows, removed_alerts)

        predictions = storage.get_predictions_with_labels(conn)
        predictions = predictions[predictions["batch_id"].isin(storage.get_completed_batch_ids(conn))]

        ready, waiting = label_complete_batches(predictions)
        labeled = predictions[predictions["batch_id"].isin(ready) & predictions["actual_label"].notna()]
        windows, open_window = build_windows(labeled, ready, **thresholds)

        for batch_ids in windows:
            rows = labeled[labeled["batch_id"].isin(batch_ids)]
            window = evaluate_window(rows, batch_ids, **thresholds)
            window_id, created = storage.save_performance_window(
                conn, {**window, "notes": json.dumps(window["notes"])}, now
            )
            new_windows += created

            alerts = evaluate_rules({**window, "window_id": window_id}, alert_rules, baseline)
            stored = storage.save_alerts(conn, alerts, now)
            new_alerts += stored
            for alert in alerts:
                logger.warning("[%s] %s", alert["severity"].upper(), alert["description"])

        total_windows = len(storage.get_performance_windows(conn))

    if waiting:
        logger.info("%d batches are waiting for labels (from batch %d).", len(waiting), waiting[0])
    if open_window:
        logger.info(
            "Open window (batches %d-%d) does not have enough labeled fraud/non-fraud cases yet.",
            open_window[0], open_window[-1],
        )
    logger.info("Windows evaluated: %d (%d new). New alerts: %d.", len(windows), new_windows, new_alerts)

    return {
        "windows": windows,
        "new_windows": new_windows,
        "new_alerts": new_alerts,
        "total_windows": total_windows,
        "open_window": open_window,
        "waiting_for_labels": waiting,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def parse_notes(raw) -> list[str]:
    """Turn a stored `notes` value into a list of strings. Never raises.

    Windows are stored with a JSON list of notes. Anything else (edited or
    damaged rows) is shown as one note instead of crashing the reports.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        value = ...  # not valid JSON
    if value is None:
        return []
    if isinstance(value, list):
        return [str(note) for note in value]
    if isinstance(value, str):
        return [value]
    return [f"notes could not be read: {str(raw)[:200]}"]


def print_report(db_path: Path | str, metrics_path: Path | str = config.METRICS_PATH) -> None:
    with storage.open_db_readonly(db_path) as conn:
        windows = storage.get_performance_windows(conn)

    if windows.empty:
        print("No performance windows yet. Run: python -m monitoring.performance run")
        return

    table = windows.set_index("window_id")[
        ["start_batch_id", "end_batch_id", "n_labeled", "n_positive", "true_positives", "false_positives",
         "false_negatives", "precision", "recall", "f1", "pr_auc", "false_positive_rate", "status"]
    ].rename(columns={"start_batch_id": "from_batch", "end_batch_id": "to_batch", "n_positive": "fraud",
                      "true_positives": "TP", "false_positives": "FP", "false_negatives": "FN",
                      "false_positive_rate": "fpr"})

    pd.set_option("display.width", 160)
    print("\n=== Model performance per window (true labels only) ===\n")
    print(table.to_string(float_format=lambda v: f"{v:.4f}", na_rep="n/a"))

    baseline = load_baseline(metrics_path)
    if baseline:
        parts = [f"{k}={v:.4f}" for k, v in baseline.items() if v is not None]
        print("\nTraining baseline (validation set):", ", ".join(parts))

    for _, row in windows.iterrows():
        for note in parse_notes(row["notes"]):
            print(f"  note, batches {row['start_batch_id']}-{row['end_batch_id']}: {note}")


def print_alerts(db_path: Path | str) -> None:
    with storage.open_db_readonly(db_path) as conn:
        alerts = storage.get_alerts(conn)

    if alerts.empty:
        print("No alerts.")
        return

    print(f"\n=== Alerts ({len(alerts)}) ===\n")
    for _, alert in alerts.iterrows():
        print(f"#{alert['alert_id']} [{alert['severity'].upper()}] {alert['rule_name']}: {alert['description']}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor model performance with delayed true labels.")
    parser.add_argument("--db", type=Path, default=config.DB_PATH, help="SQLite database file")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="evaluate windows and raise alerts")
    run.add_argument("--recompute", action="store_true", help="delete stored windows/alerts and rebuild")
    run.add_argument("--baseline", type=Path, default=config.METRICS_PATH, help="training metrics JSON")

    report = commands.add_parser("report", help="show performance per window")
    report.add_argument("--baseline", type=Path, default=config.METRICS_PATH, help="training metrics JSON")

    commands.add_parser("alerts", help="list stored alerts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        if args.command == "run":
            run_performance_monitoring(db_path=args.db, metrics_path=args.baseline, recompute=args.recompute)
            print_report(args.db, args.baseline)
            print_alerts(args.db)
        elif args.command == "report":
            print_report(args.db, args.baseline)
        else:
            print_alerts(args.db)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
