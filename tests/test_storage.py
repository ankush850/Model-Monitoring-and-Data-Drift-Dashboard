import sqlite3

import pandas as pd
import pytest

from monitoring import storage

NOW = "2026-01-01T00:00:00+00:00"


def make_predictions(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ids,
            "event_time": [float(i) for i in range(len(ids))],
            "fraud_probability": [0.1] * len(ids),
            "predicted_label": [0] * len(ids),
        }
    )


def log_batch(conn, ids: list[str]) -> tuple[int, int]:
    batch_id = storage.start_batch(conn, "v1", NOW)
    inserted = storage.insert_predictions(conn, batch_id, make_predictions(ids), NOW, 0.5, "v1")
    return batch_id, inserted


def test_open_db_creates_file_and_tables(db_path):
    with storage.open_db(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}

    assert db_path.exists()
    assert {"monitoring_batches", "predictions", "actual_labels"} <= tables


def test_reopening_db_keeps_existing_data(db_path):
    with storage.open_db(db_path) as conn:
        log_batch(conn, ["a", "b"])
    with storage.open_db(db_path) as conn:
        assert storage.get_summary(conn)["predictions"] == 2


def test_predictions_table_has_no_actual_label_column(db_path):
    with storage.open_db(db_path) as conn:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(predictions)")]
    assert "actual_label" not in columns


def test_insert_predictions_skips_duplicates(db_path):
    with storage.open_db(db_path) as conn:
        _, first = log_batch(conn, ["a", "b", "c"])
        _, second = log_batch(conn, ["b", "c", "d"])
        total = storage.get_summary(conn)["predictions"]

    assert first == 3
    assert second == 1  # only "d" is new
    assert total == 4


def test_insert_predictions_rejects_invalid_probability(db_path):
    bad = make_predictions(["a"])
    bad["fraud_probability"] = 1.5
    with storage.open_db(db_path) as conn:
        batch_id = storage.start_batch(conn, "v1", NOW)
        with pytest.raises(sqlite3.IntegrityError):
            storage.insert_predictions(conn, batch_id, bad, NOW, 0.5, "v1")
        assert storage.get_summary(conn)["predictions"] == 0


def test_batch_can_be_completed_or_failed(db_path):
    stats = {
        "n_transactions": 3, "n_logged": 3, "n_flagged": 1,
        "mean_fraud_probability": 0.2, "first_event_time": 0.0, "last_event_time": 2.0,
    }
    with storage.open_db(db_path) as conn:
        ok_id = storage.start_batch(conn, "v1", NOW)
        status = conn.execute("SELECT status FROM monitoring_batches WHERE batch_id = ?", (ok_id,)).fetchone()
        assert status["status"] == "running"

        storage.finish_batch(conn, ok_id, NOW, stats)
        bad_id = storage.start_batch(conn, "v1", NOW)
        storage.fail_batch(conn, bad_id, NOW, "boom")

        ok = conn.execute("SELECT * FROM monitoring_batches WHERE batch_id = ?", (ok_id,)).fetchone()
        bad = conn.execute("SELECT * FROM monitoring_batches WHERE batch_id = ?", (bad_id,)).fetchone()

    assert ok["status"] == "completed"
    assert ok["n_flagged"] == 1
    assert bad["status"] == "failed"
    assert bad["error_message"] == "boom"


def test_labels_can_be_added_later(db_path):
    with storage.open_db(db_path) as conn:
        log_batch(conn, ["a", "b", "c"])
        assert storage.get_unlabeled_transaction_ids(conn) == ["a", "b", "c"]

        added = storage.insert_labels(conn, pd.Series({"a": 1, "b": 0}), NOW)

        assert added == 2
        assert storage.get_unlabeled_transaction_ids(conn) == ["c"]
        df = storage.get_predictions_with_labels(conn).set_index("transaction_id")

    assert df.loc["a", "actual_label"] == 1
    assert df.loc["b", "actual_label"] == 0
    assert pd.isna(df.loc["c", "actual_label"])


def test_insert_labels_skips_unknown_and_repeated_transactions(db_path):
    with storage.open_db(db_path) as conn:
        log_batch(conn, ["a"])
        assert storage.insert_labels(conn, pd.Series({"a": 1, "unknown": 0}), NOW) == 1
        assert storage.insert_labels(conn, pd.Series({"a": 0}), NOW) == 0  # not overwritten
        df = storage.get_predictions_with_labels(conn)

    assert df["actual_label"].tolist() == [1]


def test_insert_labels_rejects_values_other_than_0_or_1(db_path):
    with storage.open_db(db_path) as conn:
        log_batch(conn, ["a"])
        with pytest.raises(ValueError, match="0 or 1"):
            storage.insert_labels(conn, pd.Series({"a": 2}), NOW)


def test_unlabeled_ids_can_be_limited_to_older_batches(db_path):
    with storage.open_db(db_path) as conn:
        first_batch, _ = log_batch(conn, ["a"])
        log_batch(conn, ["b"])
        assert storage.get_unlabeled_transaction_ids(conn, up_to_batch_id=first_batch) == ["a"]


# --- read-only opening (report commands) ------------------------------------------------
def test_open_db_readonly_does_not_create_a_missing_database(tmp_path):
    missing = tmp_path / "no_folder" / "monitoring.db"
    with pytest.raises(FileNotFoundError, match="not found"):
        with storage.open_db_readonly(missing):
            pass
    assert not missing.parent.exists()


def test_open_db_readonly_refuses_writes(db_path):
    with storage.open_db(db_path) as conn:
        log_batch(conn, ["a"])
    with storage.open_db_readonly(db_path) as conn:
        assert storage.get_summary(conn)["predictions"] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM predictions")


# --- pending labels -----------------------------------------------------------------------
def test_pending_labels_count_only_simulator_predictions(db_path):
    with storage.open_db(db_path) as conn:
        log_batch(conn, ["sim-1", "sim-2"])
        storage.insert_labels(conn, pd.Series({"sim-1": 0}), NOW)
        api_batch = storage.start_batch(conn, "v1", NOW, source="api")
        storage.insert_predictions(conn, api_batch, make_predictions(["api-1"]), NOW, 0.5, "v1")
        summary = storage.get_summary(conn)

    assert summary["predictions"] == 3
    assert summary["labeled"] == 1
    assert summary["pending_labels"] == 1  # sim-2 only; api-1 never receives a label


def test_pending_labels_without_source_column(tmp_path):
    """A database made before the source column existed holds only simulator batches."""
    old = tmp_path / "old.db"
    with sqlite3.connect(old) as conn:
        conn.executescript(
            """
            CREATE TABLE monitoring_batches (batch_id INTEGER PRIMARY KEY, status TEXT);
            CREATE TABLE predictions (transaction_id TEXT PRIMARY KEY, batch_id INTEGER, predicted_label INTEGER);
            CREATE TABLE actual_labels (transaction_id TEXT PRIMARY KEY, actual_label INTEGER);
            INSERT INTO monitoring_batches VALUES (1, 'completed');
            INSERT INTO predictions VALUES ('a', 1, 0), ('b', 1, 1);
            INSERT INTO actual_labels VALUES ('a', 0);
            """
        )
    with storage.open_db_readonly(old) as conn:
        assert storage.get_summary(conn)["pending_labels"] == 1
