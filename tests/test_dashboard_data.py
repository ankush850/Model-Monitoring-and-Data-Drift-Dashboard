"""Tests for the dashboard's read-only data-access functions."""

import re
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard_fixtures import build_populated_project, file_hash
from monitoring import config, dashboard_data as dd, storage
from monitoring.simulate import load_production_data


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    return build_populated_project(tmp_path_factory.mktemp("dashboard_data"))


# --- missing and read-only database ------------------------------------------------------
def test_missing_database_is_reported_and_not_created(tmp_path):
    missing = tmp_path / "nothing.db"
    with pytest.raises(FileNotFoundError):
        dd.load_overview(missing)
    assert not missing.exists()
    assert dd.database_version(missing) == 0.0


def test_connection_is_read_only(project):
    with dd.read_only(project["db"]) as conn:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM alerts")


def test_loading_everything_does_not_change_the_database(project):
    before = file_hash(project["db"])
    dd.load_overview(project["db"])
    dd.load_batches(project["db"])
    dd.load_drift_results(project["db"])
    dd.load_stream_results(project["db"])
    dd.load_performance_windows(project["db"])
    dd.load_alerts(project["db"])
    dd.load_label_status(project["db"])
    assert file_hash(project["db"]) == before


def test_database_without_monitoring_tables_gives_empty_results(tmp_path):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE something_else (x INTEGER)")
    conn.close()

    overview = dd.load_overview(db)
    assert overview["predictions"] == 0 and overview["alerts_total"] == 0 and overview["drift_events"] == 0
    for loader in (dd.load_batches, dd.load_drift_results, dd.load_stream_results,
                   dd.load_performance_windows, dd.load_alerts):
        assert loader(db).empty
    assert dd.load_label_status(db)["ready"] == []


def test_empty_monitoring_database_gives_zero_counts(tmp_path):
    db = tmp_path / "empty.db"
    with storage.open_db(db):
        pass
    overview = dd.load_overview(db)
    assert overview["predictions"] == 0
    assert overview["completed_batches"] == 0
    assert dd.load_alerts(db).empty


# --- values match what the monitoring modules stored -----------------------------------------
def test_overview_counts_match_stored_results(project):
    overview = dd.load_overview(project["db"])
    with storage.open_db(project["db"]) as conn:
        summary = storage.get_summary(conn)
        drift = storage.get_drift_results(conn)
        stream = storage.get_stream_results(conn)
        alerts = storage.get_alerts(conn, source="performance")

    assert overview["predictions"] == summary["predictions"] == 2000
    assert overview["flagged"] == summary["flagged"]
    assert overview["completed_batches"] == 5
    assert overview["drift_events"] == int((drift["status"] == "drift").sum())
    assert overview["psi_warning"] == int(((drift["test_name"] == "psi") & (drift["status"] == "warning")).sum())
    assert overview["stream_detections"] == int(stream["n_detections"].sum())
    assert overview["alerts_total"] == len(alerts) > 0
    assert overview["alerts_info"] == int((alerts["severity"] == "info").sum())


def test_batches_include_flag_rate(project):
    batches = dd.load_batches(project["db"])
    assert list(batches["batch_id"]) == [1, 2, 3, 4, 5]
    expected = batches["n_flagged"] / batches["n_transactions"]
    pd.testing.assert_series_equal(batches["flag_rate"], expected, check_names=False)


def test_performance_windows_have_parsed_notes_and_labels(project):
    windows = dd.load_performance_windows(project["db"])
    assert not windows.empty
    assert all(isinstance(notes, list) for notes in windows["notes"])
    first = windows.iloc[0]
    assert first["window_label"] == f"{first['start_batch_id']}–{first['end_batch_id']}"


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("{not json", ["notes could not be read: {not json"]),
        ("null", []),
        ('"text"', ["text"]),
        ("5", ["notes could not be read: 5"]),
        ('["kept note"]', ["kept note"]),
    ],
)
def test_performance_windows_with_damaged_notes_still_load(tmp_path, project, stored, expected):
    db = tmp_path / "monitoring.db"
    shutil.copyfile(project["db"], db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE performance_windows SET notes = ?", (stored,))
    windows = dd.load_performance_windows(db)
    assert all(notes == expected for notes in windows["notes"])


def test_alerts_have_required_fields(project):
    alerts = dd.load_alerts(project["db"])
    for column in ["created_at", "severity", "metric_name", "observed_value", "threshold", "description",
                   "start_batch_id", "end_batch_id"]:
        assert alerts[column].notna().all(), column


def test_training_baseline_is_loaded(project, tmp_path):
    assert dd.load_training_baseline(project["baseline"])["pr_auc"] == 0.9
    assert dd.load_training_baseline(tmp_path / "missing.json") is None


# --- delayed labels and windows ------------------------------------------------------------------
def test_label_status_follows_delayed_labels(project):
    status = dd.load_label_status(project["db"])
    # 5 batches, labels 2 batches behind: batches 1-3 fully labeled, 4-5 waiting.
    assert status["ready"] == [1, 2, 3]
    assert status["waiting"] == [4, 5]
    per_batch = status["per_batch"].set_index("batch_id")
    assert per_batch.loc[1, "n_labeled"] == per_batch.loc[1, "n_predictions"] == 400
    assert per_batch.loc[5, "n_labeled"] == 0
    assert list(per_batch["labels_complete"]) == [True, True, True, False, False]


def test_label_status_open_window_counts(project):
    status = dd.load_label_status(project["db"], min_positives=10_000, min_negatives=1)
    assert status["closed_windows"] == []
    assert status["open_batches"] == [1, 2, 3]
    with dd.read_only(project["db"]) as conn:
        fraud = conn.execute(
            "SELECT SUM(l.actual_label) FROM actual_labels l JOIN predictions p USING (transaction_id) "
            "WHERE p.batch_id <= 3").fetchone()[0]
    assert status["open_positives"] == fraud
    assert status["open_positives"] + status["open_negatives"] == 1200


# --- feature values ---------------------------------------------------------------------------
def test_batch_feature_values_match_simulator_batch(project):
    production = load_production_data(project["production"])
    values = dd.batch_feature_values(project["db"], production, 2, "V1")
    expected = production.iloc[400:800]["V1"].to_numpy()
    np.testing.assert_allclose(values, expected)


def test_batch_feature_values_accepts_numpy_batch_ids(project):
    """Batch IDs taken from a DataFrame are numpy integers; they must still match."""
    production = load_production_data(project["production"])
    values = dd.batch_feature_values(project["db"], production, np.int64(2), "V1")
    assert values.size == 400


def test_batch_feature_values_rejects_unknown_feature(project):
    production = load_production_data(project["production"])
    with pytest.raises(ValueError, match="Unknown feature"):
        dd.batch_feature_values(project["db"], production, 1, "not_a_feature")


# --- overview keys used by the dashboard and the API (regression: KeyError 'simulator_batches') ---
ROOT = Path(__file__).resolve().parents[1]


def overview_keys_used() -> set[str]:
    """Every key app.py reads from `ov[...]` and api/main.py reads from `overview[...]`."""
    app = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    api = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
    keys = set(re.findall(r"""\bov\[["'](\w+)["']\]""", app))
    keys |= set(re.findall(r"""\boverview\[["'](\w+)["']\]""", api))
    return keys


def old_layout_db(tmp_path) -> Path:
    """A database created before Phase 7: monitoring_batches has no `source` column."""
    db = tmp_path / "pre_phase7.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE monitoring_batches (batch_id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL, "
        "model_version TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT, n_transactions INTEGER, "
        "n_logged INTEGER, n_flagged INTEGER, mean_fraud_probability REAL, first_event_time REAL, "
        "last_event_time REAL, error_message TEXT);"
        "INSERT INTO monitoring_batches (status, model_version, started_at, n_logged) "
        "VALUES ('completed', 'v0', '2026-01-01', 10);"
    )
    conn.executescript(storage.SCHEMA)  # the other tables; monitoring_batches already exists
    conn.close()
    return db


def test_scan_finds_the_phase7_overview_keys():
    keys = overview_keys_used()
    assert {"simulator_batches", "api_batches", "api_predictions", "predictions"} <= keys


def test_overview_has_every_key_the_dashboard_and_api_use(project, tmp_path):
    empty = tmp_path / "empty.db"
    with storage.open_db(empty):
        pass
    no_tables = tmp_path / "no_tables.db"
    sqlite3.connect(no_tables).close()

    required = overview_keys_used()
    for name, db in [("populated", project["db"]), ("empty", empty),
                     ("pre-Phase-7 layout", old_layout_db(tmp_path)), ("no tables", no_tables)]:
        missing = required - set(dd.load_overview(db))
        assert not missing, f"{name} database: load_overview() lacks {sorted(missing)}"


def test_overview_counts_old_layout_batches_as_simulator(tmp_path):
    overview = dd.load_overview(old_layout_db(tmp_path))
    assert (overview["simulator_batches"], overview["api_batches"], overview["api_predictions"]) == (1, 0, 0)


def test_paths_can_be_overridden_by_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MONITORING_DB", str(tmp_path / "x.db"))
    paths = dd.DashboardPaths.from_env()
    assert paths.db == tmp_path / "x.db"
    assert paths.baseline == config.METRICS_PATH
    assert dd.DashboardPaths.from_env(db=tmp_path / "y.db").db == tmp_path / "y.db"
