"""Read-only data access for the Streamlit dashboard.

The dashboard opens SQLite in read-only mode, so viewing it can never create
tables, alerts or any other rows, and it never creates a missing database
file. All numbers come from results the monitoring modules already stored;
this module only fetches and reshapes them, reusing the existing query
functions in storage.py and the label/window rules in streaming.py and
performance.py.
"""

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring import config, storage
from monitoring.alerts import load_baseline
from monitoring.performance import build_windows, parse_notes
from monitoring.streaming import label_complete_batches

CORE_TABLES = {"monitoring_batches", "predictions", "actual_labels"}


@dataclass(frozen=True)
class DashboardPaths:
    """File locations the dashboard reads. Environment variables can override them."""

    db: Path = config.DB_PATH
    baseline: Path = config.METRICS_PATH
    reference_data: Path = config.REFERENCE_DATA_PATH
    production_data: Path = config.PRODUCTION_DATA_PATH

    @classmethod
    def from_env(cls, db: str | Path | None = None) -> "DashboardPaths":
        return cls(
            db=Path(db or os.environ.get("MONITORING_DB", config.DB_PATH)),
            baseline=Path(os.environ.get("MONITORING_BASELINE", config.METRICS_PATH)),
            reference_data=Path(os.environ.get("MONITORING_REFERENCE_DATA", config.REFERENCE_DATA_PATH)),
            production_data=Path(os.environ.get("MONITORING_PRODUCTION_DATA", config.PRODUCTION_DATA_PATH)),
        )


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
# Open the database read-only. Raises FileNotFoundError if it does not exist.
read_only = storage.open_db_readonly


def existing_tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def batch_source_expression(conn: sqlite3.Connection, alias: str = "") -> str:
    """SQL for a batch's source; databases made before the column existed hold only simulator batches."""
    prefix = f"{alias}." if alias else ""
    return f"{prefix}source" if has_column(conn, "monitoring_batches", "source") else "'simulator'"


def database_version(db_path: Path | str) -> float:
    """Last-modified time of the database file, used to refresh cached data. 0 if missing."""
    path = Path(db_path)
    return path.stat().st_mtime if path.is_file() else 0.0


def _read_table(db_path: Path | str, table: str, reader, empty_columns: list[str]) -> pd.DataFrame:
    """Run a storage reader if its table exists, otherwise return an empty frame."""
    with read_only(db_path) as conn:
        if table not in existing_tables(conn):
            return pd.DataFrame(columns=empty_columns)
        return reader(conn)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def load_overview(db_path: Path | str) -> dict:
    """Headline counts for the overview page."""
    with read_only(db_path) as conn:
        tables = existing_tables(conn)

        if CORE_TABLES <= tables:
            summary = storage.get_summary(conn)
        else:
            summary = {"batches": 0, "completed_batches": 0, "failed_batches": 0, "predictions": 0,
                       "flagged": 0, "labeled": 0, "pending_labels": 0}

        drift = {"ks_drift": 0, "psi_drift": 0, "psi_warning": 0, "drift_insufficient": 0, "drift_batches": 0}
        if "drift_results" in tables:
            for row in conn.execute("SELECT test_name, status, COUNT(*) FROM drift_results GROUP BY test_name, status"):
                test, status, count = row[0], row[1], row[2]
                if status == "drift":
                    drift[f"{test}_drift"] += count
                elif status == "warning" and test == "psi":
                    drift["psi_warning"] += count
                elif status == "insufficient_data":
                    drift["drift_insufficient"] += count
            drift["drift_batches"] = conn.execute("SELECT COUNT(DISTINCT batch_id) FROM drift_results").fetchone()[0]

        stream = {"stream_detections": 0, "stream_drift_rows": 0, "stream_batches": 0}
        if "stream_drift_results" in tables:
            row = conn.execute(
                "SELECT COALESCE(SUM(n_detections), 0), COALESCE(SUM(status = 'drift'), 0), "
                "COUNT(DISTINCT batch_id) FROM stream_drift_results"
            ).fetchone()
            stream = {"stream_detections": row[0], "stream_drift_rows": row[1], "stream_batches": row[2]}

        alerts = {"alerts_total": 0, "alerts_critical": 0, "alerts_warning": 0, "alerts_info": 0}
        if "alerts" in tables:
            for row in conn.execute(
                "SELECT severity, COUNT(*) FROM alerts WHERE source = 'performance' GROUP BY severity"
            ):
                alerts[f"alerts_{row[0]}"] = row[1]
                alerts["alerts_total"] += row[1]

        windows = 0
        if "performance_windows" in tables:
            windows = conn.execute("SELECT COUNT(*) FROM performance_windows").fetchone()[0]

        sources = {"simulator_batches": 0, "api_batches": 0, "api_predictions": 0}
        if CORE_TABLES <= tables:
            source = batch_source_expression(conn)
            for row in conn.execute(
                f"SELECT {source}, COUNT(*), COALESCE(SUM(n_logged), 0) FROM monitoring_batches "
                f"WHERE status = 'completed' GROUP BY {source}"
            ):
                sources[f"{row[0]}_batches"] = row[1]
                if row[0] == "api":
                    sources["api_predictions"] = row[2]

    return {
        **summary,
        **sources,
        **drift,
        "drift_events": drift["ks_drift"] + drift["psi_drift"],
        **stream,
        **alerts,
        "performance_windows": windows,
    }


def load_batches(db_path: Path | str) -> pd.DataFrame:
    """All monitoring batches as logged by the simulator, plus the share flagged."""
    columns = ["batch_id", "status", "model_version", "started_at", "finished_at", "n_transactions",
               "n_logged", "n_flagged", "mean_fraud_probability", "first_event_time", "last_event_time",
               "error_message"]

    def reader(conn):
        select = ", ".join(columns) + f", {batch_source_expression(conn)} AS source"
        rows = conn.execute(f"SELECT {select} FROM monitoring_batches ORDER BY batch_id").fetchall()
        return pd.DataFrame([tuple(r) for r in rows], columns=[*columns, "source"])

    df = _read_table(db_path, "monitoring_batches", reader, [*columns, "source"])
    df["flag_rate"] = df["n_flagged"] / df["n_transactions"].where(df["n_transactions"] > 0)
    return df


# ---------------------------------------------------------------------------
# Stored monitoring results (thin wrappers around storage.py readers)
# ---------------------------------------------------------------------------
def load_drift_results(db_path: Path | str) -> pd.DataFrame:
    columns = ["batch_id", *storage.DRIFT_RESULT_COLUMNS, "computed_at"]
    return _read_table(db_path, "drift_results", storage.get_drift_results, columns)


def load_stream_results(db_path: Path | str) -> pd.DataFrame:
    columns = ["detector", "signal", *storage.STREAM_RESULT_COLUMNS, "computed_at"]
    return _read_table(db_path, "stream_drift_results", storage.get_stream_results, columns)


def load_performance_windows(db_path: Path | str) -> pd.DataFrame:
    columns = ["window_id", *storage.PERFORMANCE_WINDOW_COLUMNS, "computed_at"]
    df = _read_table(db_path, "performance_windows", storage.get_performance_windows, columns)
    df["notes"] = [parse_notes(n) for n in df["notes"]]
    df["window_label"] = [f"{s}–{e}" if s != e else f"{s}" for s, e in zip(df["start_batch_id"], df["end_batch_id"])]
    return df


def load_alerts(db_path: Path | str) -> pd.DataFrame:
    columns = ["alert_id", *storage.ALERT_COLUMNS, "created_at"]
    return _read_table(db_path, "alerts", storage.get_alerts, columns)


def load_training_baseline(path: Path | str) -> dict | None:
    """Validation-set metrics saved at training time (reuses alerts.load_baseline)."""
    return load_baseline(path)


# ---------------------------------------------------------------------------
# Label arrival and performance-window progress
# ---------------------------------------------------------------------------
def load_label_status(
    db_path: Path | str,
    min_positives: int = config.PERFORMANCE_MIN_POSITIVES,
    min_negatives: int = config.PERFORMANCE_MIN_NEGATIVES,
) -> dict:
    """How far delayed labels have arrived, using the same rules as the monitors.

    Returns per-batch label counts, which batches are fully labeled (in order),
    which are still waiting, and how much labeled data the open performance
    window has collected so far.
    """
    empty = {
        "per_batch": pd.DataFrame(columns=["batch_id", "n_predictions", "n_labeled", "fraud_labeled",
                                           "labels_complete"]),
        "ready": [], "waiting": [], "closed_windows": [], "open_batches": [],
        "open_positives": 0, "open_negatives": 0,
    }
    with read_only(db_path) as conn:
        if not CORE_TABLES <= existing_tables(conn):
            return empty
        # Simulator batches only, matching storage.get_completed_batch_ids().
        rows = conn.execute(
            f"""
            SELECT p.batch_id, l.actual_label
            FROM predictions AS p
            JOIN monitoring_batches AS b ON b.batch_id = p.batch_id AND b.status = 'completed'
            LEFT JOIN actual_labels AS l ON l.transaction_id = p.transaction_id
            WHERE {batch_source_expression(conn, 'b')} = 'simulator'
            """
        ).fetchall()

    labels = pd.DataFrame([tuple(r) for r in rows], columns=["batch_id", "actual_label"])
    if labels.empty:
        return empty

    ready, waiting = label_complete_batches(labels)
    per_batch = (
        labels.groupby("batch_id")
        .agg(n_predictions=("actual_label", "size"), n_labeled=("actual_label", "count"),
             fraud_labeled=("actual_label", "sum"))
        .reset_index()
    )
    per_batch["fraud_labeled"] = per_batch["fraud_labeled"].astype(int)
    per_batch["labels_complete"] = per_batch["batch_id"].isin(ready)

    labeled = labels[labels["batch_id"].isin(ready)]
    closed, open_batches = build_windows(labeled, ready, min_positives, min_negatives)
    open_rows = labeled[labeled["batch_id"].isin(open_batches)]
    open_positives = int(open_rows["actual_label"].sum())

    return {
        "per_batch": per_batch,
        "ready": ready,
        "waiting": waiting,
        "closed_windows": closed,
        "open_batches": open_batches,
        "open_positives": open_positives,
        "open_negatives": int(len(open_rows) - open_positives),
    }


# ---------------------------------------------------------------------------
# Feature values for distribution charts
# ---------------------------------------------------------------------------
def batch_feature_values(
    db_path: Path | str, production: pd.DataFrame, batch_id: int, feature: str
) -> np.ndarray:
    """Values of one feature for the transactions the simulator put in a batch.

    `production` is the output of simulate.load_production_data (it carries the
    same transaction IDs the simulator logged).
    """
    if feature not in production.columns:
        raise ValueError(f"Unknown feature '{feature}'.")
    with read_only(db_path) as conn:
        # int(): sqlite3 would bind a numpy integer as a BLOB and silently match nothing.
        ids = storage.get_batch_transaction_ids(conn, int(batch_id))
    values = production.set_index("transaction_id").reindex(ids)[feature]
    return values.dropna().to_numpy(dtype=float)
