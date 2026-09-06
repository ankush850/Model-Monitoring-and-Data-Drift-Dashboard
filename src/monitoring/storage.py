"""Read and write monitoring records in a SQLite database.

Tables:
  monitoring_batches  one row per batch of transactions sent to the model
  predictions         one row per scored transaction (no true labels here)
  actual_labels       true fraud labels, added later when they become known

True labels live in their own table on purpose: when a prediction is made
the answer is not known yet, so it must never sit next to the prediction.
"""

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from monitoring import config

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS monitoring_batches (
    batch_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    status                 TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    model_version          TEXT NOT NULL,
    started_at             TEXT NOT NULL,
    finished_at            TEXT,
    n_transactions         INTEGER,
    n_logged               INTEGER,
    n_flagged              INTEGER,
    mean_fraud_probability REAL,
    first_event_time       REAL,
    last_event_time        REAL,
    error_message          TEXT,
    source                 TEXT NOT NULL DEFAULT 'simulator' CHECK (source IN ('simulator', 'api'))
);

CREATE TABLE IF NOT EXISTS predictions (
    transaction_id    TEXT PRIMARY KEY,
    batch_id          INTEGER NOT NULL REFERENCES monitoring_batches (batch_id),
    event_time        REAL NOT NULL,
    predicted_at      TEXT NOT NULL,
    fraud_probability REAL NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    predicted_label   INTEGER NOT NULL CHECK (predicted_label IN (0, 1)),
    threshold         REAL NOT NULL,
    model_version     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_batch_id ON predictions (batch_id);

CREATE TABLE IF NOT EXISTS actual_labels (
    transaction_id TEXT PRIMARY KEY REFERENCES predictions (transaction_id),
    actual_label   INTEGER NOT NULL CHECK (actual_label IN (0, 1)),
    labeled_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS drift_results (
    result_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id          INTEGER NOT NULL REFERENCES monitoring_batches (batch_id),
    feature_name      TEXT NOT NULL,
    test_name         TEXT NOT NULL CHECK (test_name IN ('ks', 'psi')),
    score             REAL,  -- KS statistic or PSI value; NULL if it could not be computed
    p_value           REAL,  -- KS test only
    threshold         REAL NOT NULL,  -- KS: significance level; PSI: drift threshold
    warning_threshold REAL,           -- PSI only
    status            TEXT NOT NULL
                      CHECK (status IN ('no_drift', 'warning', 'drift', 'insufficient_data')),
    n_reference       INTEGER NOT NULL,
    n_current         INTEGER NOT NULL,
    n_missing         INTEGER NOT NULL,
    detail            TEXT,
    computed_at       TEXT NOT NULL,
    UNIQUE (batch_id, feature_name, test_name)
);

CREATE TABLE IF NOT EXISTS stream_drift_results (
    result_id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id                       INTEGER NOT NULL REFERENCES monitoring_batches (batch_id),
    detector                       TEXT NOT NULL CHECK (detector IN ('adwin', 'page_hinkley')),
    signal                         TEXT NOT NULL,
    status                         TEXT NOT NULL CHECK (status IN ('stable', 'drift', 'no_data')),
    n_events                       INTEGER NOT NULL,  -- valid values fed to the detector
    n_invalid                      INTEGER NOT NULL,  -- missing/invalid values skipped
    n_detections                   INTEGER NOT NULL,
    signal_mean                    REAL,              -- average signal value in this batch
    first_detection_transaction_id TEXT,
    first_detection_event_time     REAL,
    detector_info                  TEXT,              -- JSON: settings, state, detection list
    computed_at                    TEXT NOT NULL,
    UNIQUE (batch_id, detector, signal)
);

CREATE TABLE IF NOT EXISTS performance_windows (
    window_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    start_batch_id       INTEGER NOT NULL REFERENCES monitoring_batches (batch_id),
    end_batch_id         INTEGER NOT NULL REFERENCES monitoring_batches (batch_id),
    first_event_time     REAL,
    last_event_time      REAL,
    status               TEXT NOT NULL CHECK (status IN ('ok', 'insufficient_data')),
    n_labeled            INTEGER NOT NULL,
    n_positive           INTEGER NOT NULL,  -- actual fraud cases
    n_negative           INTEGER NOT NULL,  -- actual non-fraud cases
    true_positives       INTEGER NOT NULL,
    false_positives      INTEGER NOT NULL,
    true_negatives       INTEGER NOT NULL,
    false_negatives      INTEGER NOT NULL,
    precision            REAL,  -- NULL when it cannot be calculated
    recall               REAL,
    f1                   REAL,
    pr_auc               REAL,
    false_positive_rate  REAL,
    decision_threshold   REAL,
    model_versions       TEXT,
    notes                TEXT,  -- JSON list of reasons a metric is missing or unreliable
    computed_at          TEXT NOT NULL,
    UNIQUE (start_batch_id, end_batch_id)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,  -- which monitor raised it, e.g. 'performance'
    rule_name       TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    observed_value  REAL NOT NULL,
    threshold       REAL NOT NULL,
    baseline_value  REAL,
    severity        TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    description     TEXT NOT NULL,
    window_id       INTEGER REFERENCES performance_windows (window_id),
    start_batch_id  INTEGER,
    end_batch_id    INTEGER,
    created_at      TEXT NOT NULL,
    UNIQUE (source, rule_name, start_batch_id, end_batch_id)
);
"""


@contextmanager
def open_db(db_path: Path | str = config.DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open the database (creating it and its tables if needed) and close it afterwards.

    Usage:
        with open_db() as conn:
            ...
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # rows behave like dictionaries
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        init_db(conn)
        yield conn
    finally:
        conn.close()


@contextmanager
def open_db_readonly(db_path: Path | str = config.DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open an existing database read-only and close it afterwards.

    For commands that only show stored results: nothing is ever created,
    migrated or written. Raises FileNotFoundError if the file does not exist.
    """
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"Monitoring database not found at {path}.")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    """Create the tables if they do not exist yet. Safe to call many times."""
    conn.executescript(SCHEMA)
    add_missing_columns(conn)
    conn.commit()


def add_missing_columns(conn: sqlite3.Connection) -> None:
    """Bring a database created by an earlier version up to date.

    Only adds columns (existing rows get the default value); nothing is
    removed or rewritten. `source` marks where a batch came from, so the
    batch monitors can keep API traffic apart from simulator batches.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(monitoring_batches)")}
    if "source" not in columns:
        conn.execute(
            "ALTER TABLE monitoring_batches ADD COLUMN source TEXT NOT NULL DEFAULT 'simulator' "
            "CHECK (source IN ('simulator', 'api'))"
        )
        logger.info("Added column monitoring_batches.source (existing batches marked 'simulator').")


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------
def start_batch(
    conn: sqlite3.Connection, model_version: str, started_at: str, source: str = "simulator"
) -> int:
    """Record that a batch has started and return its new batch_id.

    source is "simulator" (production replay) or "api" (live API requests).
    """
    with conn:
        cursor = conn.execute(
            "INSERT INTO monitoring_batches (status, model_version, started_at, source) "
            "VALUES ('running', ?, ?, ?)",
            (model_version, started_at, source),
        )
    return cursor.lastrowid


def finish_batch(conn: sqlite3.Connection, batch_id: int, finished_at: str, stats: dict) -> None:
    """Mark a batch as completed and save its summary numbers."""
    with conn:
        conn.execute(
            """
            UPDATE monitoring_batches
            SET status = 'completed', finished_at = :finished_at,
                n_transactions = :n_transactions, n_logged = :n_logged,
                n_flagged = :n_flagged, mean_fraud_probability = :mean_fraud_probability,
                first_event_time = :first_event_time, last_event_time = :last_event_time
            WHERE batch_id = :batch_id
            """,
            {**stats, "finished_at": finished_at, "batch_id": batch_id},
        )


def fail_batch(conn: sqlite3.Connection, batch_id: int, finished_at: str, error_message: str) -> None:
    """Mark a batch as failed and keep the error message for debugging."""
    with conn:
        conn.execute(
            "UPDATE monitoring_batches SET status = 'failed', finished_at = ?, error_message = ? "
            "WHERE batch_id = ?",
            (finished_at, error_message, batch_id),
        )


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def insert_predictions(
    conn: sqlite3.Connection,
    batch_id: int,
    predictions: pd.DataFrame,
    predicted_at: str,
    threshold: float,
    model_version: str,
) -> int:
    """Save predictions and return how many new rows were stored.

    `predictions` needs the columns transaction_id, event_time,
    fraud_probability and predicted_label. A transaction that is already
    in the table is skipped, so the same transaction is never logged twice.
    """
    rows = [
        (str(tid), batch_id, float(event_time), predicted_at, float(prob), int(label), threshold, model_version)
        for tid, event_time, prob, label in zip(
            predictions["transaction_id"],
            predictions["event_time"],
            predictions["fraud_probability"],
            predictions["predicted_label"],
        )
    ]
    before = conn.total_changes
    with conn:  # one transaction: either every row is saved or none are
        conn.executemany(
            """
            INSERT INTO predictions (transaction_id, batch_id, event_time, predicted_at,
                                     fraud_probability, predicted_label, threshold, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (transaction_id) DO NOTHING
            """,
            rows,
        )
    inserted = conn.total_changes - before

    skipped = len(rows) - inserted
    if skipped:
        logger.warning("Skipped %d transactions that were already logged.", skipped)
    return inserted


def get_logged_transaction_ids(conn: sqlite3.Connection) -> set[str]:
    """Return the IDs of every transaction that already has a prediction."""
    return {row[0] for row in conn.execute("SELECT transaction_id FROM predictions")}


# ---------------------------------------------------------------------------
# Actual labels (arrive later)
# ---------------------------------------------------------------------------
def insert_labels(conn: sqlite3.Connection, labels: pd.Series, labeled_at: str) -> int:
    """Save true labels (index = transaction_id, value = 0 or 1).

    Labels are only stored for transactions that have a prediction, and a
    label that was already received is not overwritten. Returns how many
    new labels were stored.
    """
    invalid = labels[~labels.isin([0, 1])]
    if not invalid.empty:
        raise ValueError(f"Labels must be 0 or 1. Invalid values: {invalid.head().to_dict()}")

    rows = [(str(tid), int(label), labeled_at, str(tid)) for tid, label in labels.items()]
    before = conn.total_changes
    with conn:
        conn.executemany(
            """
            INSERT INTO actual_labels (transaction_id, actual_label, labeled_at)
            SELECT ?, ?, ?
            WHERE EXISTS (SELECT 1 FROM predictions WHERE transaction_id = ?)
            ON CONFLICT (transaction_id) DO NOTHING
            """,
            rows,
        )
    return conn.total_changes - before


def get_unlabeled_transaction_ids(
    conn: sqlite3.Connection, up_to_batch_id: int | None = None
) -> list[str]:
    """Return IDs of predictions that have no true label yet.

    If up_to_batch_id is given, only batches with that ID or lower are included.
    """
    query = """
        SELECT p.transaction_id
        FROM predictions AS p
        LEFT JOIN actual_labels AS l ON l.transaction_id = p.transaction_id
        WHERE l.transaction_id IS NULL
    """
    params: tuple = ()
    if up_to_batch_id is not None:
        query += " AND p.batch_id <= ?"
        params = (up_to_batch_id,)
    query += " ORDER BY p.transaction_id"
    return [row[0] for row in conn.execute(query, params)]


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------
def get_predictions_with_labels(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return every prediction, with its true label if it has arrived (else empty)."""
    columns = [
        "transaction_id", "batch_id", "event_time", "predicted_at", "fraud_probability",
        "predicted_label", "threshold", "model_version", "actual_label", "labeled_at",
    ]
    rows = conn.execute(
        """
        SELECT p.transaction_id, p.batch_id, p.event_time, p.predicted_at, p.fraud_probability,
               p.predicted_label, p.threshold, p.model_version, l.actual_label, l.labeled_at
        FROM predictions AS p
        LEFT JOIN actual_labels AS l ON l.transaction_id = p.transaction_id
        ORDER BY p.event_time, p.transaction_id
        """
    ).fetchall()
    return pd.DataFrame([tuple(row) for row in rows], columns=columns)


def get_summary(conn: sqlite3.Connection) -> dict:
    """Return headline counts for the whole database."""
    row = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM monitoring_batches)                          AS batches,
            (SELECT COUNT(*) FROM monitoring_batches WHERE status = 'completed') AS completed_batches,
            (SELECT COUNT(*) FROM monitoring_batches WHERE status = 'failed')    AS failed_batches,
            (SELECT COUNT(*) FROM predictions)                                 AS predictions,
            (SELECT COALESCE(SUM(predicted_label), 0) FROM predictions)        AS flagged,
            (SELECT COUNT(*) FROM actual_labels)                               AS labeled
        """
    ).fetchone()
    summary = dict(row)
    # Only simulator predictions ever receive true labels; API predictions are
    # counted separately and would otherwise stay "pending" forever. Databases
    # made before the source column existed hold only simulator batches.
    columns = {r[1] for r in conn.execute("PRAGMA table_info(monitoring_batches)")}
    simulator_only = "b.source = 'simulator'" if "source" in columns else "1"
    summary["pending_labels"] = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM predictions AS p
        JOIN monitoring_batches AS b ON b.batch_id = p.batch_id
        LEFT JOIN actual_labels AS l ON l.transaction_id = p.transaction_id
        WHERE l.transaction_id IS NULL AND {simulator_only}
        """
    ).fetchone()[0]
    return summary


# ---------------------------------------------------------------------------
# Drift results
# ---------------------------------------------------------------------------
DRIFT_RESULT_COLUMNS = [
    "feature_name", "test_name", "score", "p_value", "threshold", "warning_threshold",
    "status", "n_reference", "n_current", "n_missing", "detail",
]


def get_completed_batch_ids(conn: sqlite3.Connection, source: str | None = "simulator") -> list[int]:
    """Return the IDs of successfully completed batches, oldest first.

    By default only simulator batches are returned: the drift, streaming and
    performance monitors work on the time-ordered production replay, and API
    requests (which never receive true labels) would otherwise block them.
    Pass source=None for every completed batch.
    """
    query = "SELECT batch_id FROM monitoring_batches WHERE status = 'completed'"
    params: tuple = ()
    if source is not None:
        query += " AND source = ?"
        params = (source,)
    rows = conn.execute(query + " ORDER BY batch_id", params)
    return [row[0] for row in rows]


def get_batch_transaction_ids(conn: sqlite3.Connection, batch_id: int) -> list[str]:
    """Return the transactions the simulator put in a batch, in time order."""
    rows = conn.execute(
        "SELECT transaction_id FROM predictions WHERE batch_id = ? ORDER BY event_time, transaction_id",
        (batch_id,),
    )
    return [row[0] for row in rows]


def get_batch_ids_with_drift_results(conn: sqlite3.Connection) -> set[int]:
    """Return the IDs of batches that already have drift results."""
    return {row[0] for row in conn.execute("SELECT DISTINCT batch_id FROM drift_results")}


def save_drift_results(
    conn: sqlite3.Connection, batch_id: int, results: list[dict], computed_at: str
) -> int:
    """Save drift results for one batch and return how many rows were written.

    Each (batch, feature, test) has exactly one row. Saving the same batch
    again replaces the old values instead of adding duplicates.
    """
    rows = [
        {**{col: result.get(col) for col in DRIFT_RESULT_COLUMNS}, "batch_id": batch_id, "computed_at": computed_at}
        for result in results
    ]
    with conn:  # all rows for the batch are saved together, or none are
        conn.executemany(
            """
            INSERT INTO drift_results (batch_id, feature_name, test_name, score, p_value, threshold,
                                       warning_threshold, status, n_reference, n_current, n_missing,
                                       detail, computed_at)
            VALUES (:batch_id, :feature_name, :test_name, :score, :p_value, :threshold,
                    :warning_threshold, :status, :n_reference, :n_current, :n_missing,
                    :detail, :computed_at)
            ON CONFLICT (batch_id, feature_name, test_name) DO UPDATE SET
                score = excluded.score, p_value = excluded.p_value, threshold = excluded.threshold,
                warning_threshold = excluded.warning_threshold, status = excluded.status,
                n_reference = excluded.n_reference, n_current = excluded.n_current,
                n_missing = excluded.n_missing, detail = excluded.detail,
                computed_at = excluded.computed_at
            """,
            rows,
        )
    return len(rows)


def get_drift_results(conn: sqlite3.Connection, batch_id: int | None = None) -> pd.DataFrame:
    """Return stored drift results (all batches, or just one)."""
    columns = ["batch_id", *DRIFT_RESULT_COLUMNS, "computed_at"]
    query = f"SELECT {', '.join(columns)} FROM drift_results"
    params: tuple = ()
    if batch_id is not None:
        query += " WHERE batch_id = ?"
        params = (batch_id,)
    query += " ORDER BY batch_id, feature_name, test_name"
    rows = conn.execute(query, params).fetchall()
    return pd.DataFrame([tuple(row) for row in rows], columns=columns)


# ---------------------------------------------------------------------------
# Streaming detector results
# ---------------------------------------------------------------------------
STREAM_RESULT_COLUMNS = [
    "batch_id", "status", "n_events", "n_invalid", "n_detections", "signal_mean",
    "first_detection_transaction_id", "first_detection_event_time", "detector_info",
]


def save_stream_results(
    conn: sqlite3.Connection, detector: str, signal: str, results: list[dict], computed_at: str
) -> int:
    """Save per-batch detector results and return how many new rows were stored.

    A (batch, detector, signal) row that already exists is left unchanged,
    so replaying the same stream never creates duplicates.
    """
    rows = [
        {**{col: result.get(col) for col in STREAM_RESULT_COLUMNS},
         "detector": detector, "signal": signal, "computed_at": computed_at}
        for result in results
    ]
    before = conn.total_changes
    with conn:
        conn.executemany(
            """
            INSERT INTO stream_drift_results (batch_id, detector, signal, status, n_events, n_invalid,
                                              n_detections, signal_mean, first_detection_transaction_id,
                                              first_detection_event_time, detector_info, computed_at)
            VALUES (:batch_id, :detector, :signal, :status, :n_events, :n_invalid,
                    :n_detections, :signal_mean, :first_detection_transaction_id,
                    :first_detection_event_time, :detector_info, :computed_at)
            ON CONFLICT (batch_id, detector, signal) DO NOTHING
            """,
            rows,
        )
    return conn.total_changes - before


def delete_stream_results(conn: sqlite3.Connection) -> int:
    """Remove all streaming detector results (used before a full recompute)."""
    with conn:
        cursor = conn.execute("DELETE FROM stream_drift_results")
    return cursor.rowcount


def get_stream_results(
    conn: sqlite3.Connection, signal: str | None = None, detector: str | None = None
) -> pd.DataFrame:
    """Return stored streaming detector results, optionally for one signal/detector."""
    columns = ["detector", "signal", *STREAM_RESULT_COLUMNS, "computed_at"]
    query = f"SELECT {', '.join(columns)} FROM stream_drift_results WHERE 1 = 1"
    params: list = []
    if signal is not None:
        query += " AND signal = ?"
        params.append(signal)
    if detector is not None:
        query += " AND detector = ?"
        params.append(detector)
    query += " ORDER BY signal, detector, batch_id"
    rows = conn.execute(query, params).fetchall()
    return pd.DataFrame([tuple(row) for row in rows], columns=columns)


# ---------------------------------------------------------------------------
# Performance windows and alerts
# ---------------------------------------------------------------------------
PERFORMANCE_WINDOW_COLUMNS = [
    "start_batch_id", "end_batch_id", "first_event_time", "last_event_time", "status",
    "n_labeled", "n_positive", "n_negative", "true_positives", "false_positives",
    "true_negatives", "false_negatives", "precision", "recall", "f1", "pr_auc",
    "false_positive_rate", "decision_threshold", "model_versions", "notes",
]

ALERT_COLUMNS = [
    "source", "rule_name", "metric_name", "observed_value", "threshold", "baseline_value",
    "severity", "description", "window_id", "start_batch_id", "end_batch_id",
]


def save_performance_window(conn: sqlite3.Connection, window: dict, computed_at: str) -> tuple[int, bool]:
    """Store a window's metrics. Returns (window_id, True if it was newly stored).

    A window is identified by its first and last batch. Evaluating the same
    window again keeps the row that is already stored.
    """
    row = {**{col: window.get(col) for col in PERFORMANCE_WINDOW_COLUMNS}, "computed_at": computed_at}
    with conn:
        cursor = conn.execute(
            f"""
            INSERT INTO performance_windows ({', '.join(row)})
            VALUES ({', '.join(':' + col for col in row)})
            ON CONFLICT (start_batch_id, end_batch_id) DO NOTHING
            """,
            row,
        )
    created = cursor.rowcount == 1
    window_id = conn.execute(
        "SELECT window_id FROM performance_windows WHERE start_batch_id = ? AND end_batch_id = ?",
        (window["start_batch_id"], window["end_batch_id"]),
    ).fetchone()[0]
    return window_id, created


def save_alerts(conn: sqlite3.Connection, alerts: list[dict], created_at: str) -> int:
    """Store alerts and return how many were new. The same rule never fires twice for the same window."""
    rows = [{**{col: alert.get(col) for col in ALERT_COLUMNS}, "created_at": created_at} for alert in alerts]
    before = conn.total_changes
    with conn:
        conn.executemany(
            f"""
            INSERT INTO alerts ({', '.join(ALERT_COLUMNS)}, created_at)
            VALUES ({', '.join(':' + col for col in ALERT_COLUMNS)}, :created_at)
            ON CONFLICT (source, rule_name, start_batch_id, end_batch_id) DO NOTHING
            """,
            rows,
        )
    return conn.total_changes - before


def delete_performance_results(conn: sqlite3.Connection) -> tuple[int, int]:
    """Remove performance alerts and windows (used before a full recompute)."""
    with conn:
        alerts = conn.execute("DELETE FROM alerts WHERE source = 'performance'").rowcount
        windows = conn.execute("DELETE FROM performance_windows").rowcount
    return windows, alerts


def get_performance_windows(conn: sqlite3.Connection) -> pd.DataFrame:
    columns = ["window_id", *PERFORMANCE_WINDOW_COLUMNS, "computed_at"]
    rows = conn.execute(
        f"SELECT {', '.join(columns)} FROM performance_windows ORDER BY start_batch_id"
    ).fetchall()
    return pd.DataFrame([tuple(row) for row in rows], columns=columns)


def get_alerts(conn: sqlite3.Connection, source: str | None = None) -> pd.DataFrame:
    columns = ["alert_id", *ALERT_COLUMNS, "created_at"]
    query = f"SELECT {', '.join(columns)} FROM alerts"
    params: tuple = ()
    if source is not None:
        query += " WHERE source = ?"
        params = (source,)
    query += " ORDER BY start_batch_id, alert_id"
    rows = conn.execute(query, params).fetchall()
    return pd.DataFrame([tuple(row) for row in rows], columns=columns)
