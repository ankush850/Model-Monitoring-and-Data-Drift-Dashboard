"""API batches must not disturb the batch monitors, and old databases must upgrade safely."""

import sqlite3

import pandas as pd

from monitoring import dashboard_data, storage
from monitoring.drift import run_drift_detection
from monitoring.performance import run_performance_monitoring
from monitoring.simulate import release_all_labels, run_simulation
from monitoring.streaming import run_streaming_detection

NOW = "2026-01-01T00:00:00+00:00"

OLD_BATCH_TABLE = """
CREATE TABLE monitoring_batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    model_version TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
    n_transactions INTEGER, n_logged INTEGER, n_flagged INTEGER, mean_fraud_probability REAL,
    first_event_time REAL, last_event_time REAL, error_message TEXT
);
"""


def log_api_prediction(conn, transaction_id: str, event_time: float) -> int:
    batch_id = storage.start_batch(conn, "v1", NOW, source="api")
    storage.insert_predictions(conn, batch_id, pd.DataFrame({
        "transaction_id": [transaction_id], "event_time": [event_time],
        "fraud_probability": [0.9], "predicted_label": [1]}), NOW, 0.5, "v1")
    storage.finish_batch(conn, batch_id, NOW, {
        "n_transactions": 1, "n_logged": 1, "n_flagged": 1, "mean_fraud_probability": 0.9,
        "first_event_time": event_time, "last_event_time": event_time})
    return batch_id


def test_old_database_gets_source_column_without_losing_data(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(OLD_BATCH_TABLE)
    conn.execute("INSERT INTO monitoring_batches (status, model_version, started_at) VALUES ('completed', 'v0', ?)",
                 (NOW,))
    conn.commit()
    conn.close()

    # Read-only access (dashboard) works on the old layout and treats it as simulator data.
    assert dashboard_data.load_batches(db)["source"].tolist() == ["simulator"]

    with storage.open_db(db) as conn:
        rows = [dict(r) for r in conn.execute("SELECT batch_id, model_version, source FROM monitoring_batches")]
        assert storage.get_completed_batch_ids(conn) == [1]
    assert rows == [{"batch_id": 1, "model_version": "v0", "source": "simulator"}]

    with storage.open_db(db):  # opening again does not fail or duplicate the column
        pass


def test_completed_batch_ids_exclude_api_batches_by_default(db_path):
    with storage.open_db(db_path) as conn:
        sim = storage.start_batch(conn, "v1", NOW)
        storage.finish_batch(conn, sim, NOW, {"n_transactions": 0, "n_logged": 0, "n_flagged": 0,
                                              "mean_fraud_probability": None, "first_event_time": None,
                                              "last_event_time": None})
        api = log_api_prediction(conn, "api-1", 1.0)
        assert storage.get_completed_batch_ids(conn) == [sim]
        assert storage.get_completed_batch_ids(conn, source="api") == [api]
        assert storage.get_completed_batch_ids(conn, source=None) == [sim, api]


def test_monitors_ignore_api_predictions(tmp_path, fake_df, trained_model_path):
    """API rows (no labels, out-of-order times) must not block or break the monitors."""
    db = tmp_path / "monitoring.db"
    reference, production = tmp_path / "reference.csv", tmp_path / "production.csv"
    fake_df.iloc[:1200].to_csv(reference, index=False)
    fake_df.to_csv(production, index=False)

    run_simulation(db_path=db, production_path=production, model_path=trained_model_path,
                   batch_size=400, max_batches=2, label_delay_batches=0)
    with storage.open_db(db) as conn:
        api_batch = log_api_prediction(conn, "api-early", event_time=-1.0)  # earlier than every simulator event
    run_simulation(db_path=db, production_path=production, model_path=trained_model_path,
                   batch_size=400, label_delay_batches=0)
    release_all_labels(db, production)

    drift = run_drift_detection(db_path=db, reference_path=reference, production_path=production)
    streaming = run_streaming_detection(db_path=db)
    performance = run_performance_monitoring(db_path=db, min_positives=10, min_negatives=100)

    assert api_batch not in drift["processed"]
    assert streaming["waiting_for_labels"] == []  # the unlabeled API batch does not block error_rate
    assert api_batch not in streaming["label_ready_batches"]
    assert performance["waiting_for_labels"] == []
    assert all(api_batch not in window for window in performance["windows"])

    with storage.open_db(db) as conn:
        assert api_batch not in set(storage.get_stream_results(conn)["batch_id"])

    status = dashboard_data.load_label_status(db)
    assert api_batch not in status["ready"] and api_batch not in status["waiting"]
    overview = dashboard_data.load_overview(db)
    assert (overview["simulator_batches"], overview["api_batches"], overview["api_predictions"]) == (5, 1, 1)
