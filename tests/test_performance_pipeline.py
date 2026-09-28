"""Tests for performance monitoring over the database, with delayed labels and alerts."""

import json

import numpy as np
import pandas as pd
import pytest

from monitoring import storage
from monitoring.performance import compute_metrics, main, run_performance_monitoring
from monitoring.simulate import release_all_labels, run_simulation

NOW = "2026-01-01T00:00:00+00:00"
SETTINGS = {"min_positives": 30, "min_negatives": 500}


def log_batch(conn, batch_no, actual, recall_rate, rng):
    """Log one batch of 1,000 predictions; the 'model' catches `recall_rate` of the fraud."""
    actual = np.asarray(actual)
    caught = rng.random(actual.size) < recall_rate
    predicted = np.where(actual == 1, caught, rng.random(actual.size) < 0.02).astype(int)
    scores = np.where(predicted == 1, rng.uniform(0.6, 1.0, actual.size), rng.uniform(0.0, 0.4, actual.size))

    batch_id = storage.start_batch(conn, "v1", NOW)
    ids = [f"b{batch_no:02d}-{i:05d}" for i in range(actual.size)]
    storage.insert_predictions(conn, batch_id, pd.DataFrame({
        "transaction_id": ids,
        "event_time": batch_no * 10_000 + np.arange(actual.size, dtype=float),
        "fraud_probability": scores,
        "predicted_label": predicted,
    }), NOW, 0.5, "v1")
    storage.finish_batch(conn, batch_id, NOW, {
        "n_transactions": actual.size, "n_logged": actual.size, "n_flagged": int(predicted.sum()),
        "mean_fraud_probability": float(scores.mean()), "first_event_time": 0.0, "last_event_time": 0.0,
    })
    return pd.Series(actual, index=ids)


def build_db(db_path, recall_per_batch, label_batches=None, seed=0):
    """Each batch: 1,000 transactions, 40 of them fraud. Labels arrive for `label_batches`."""
    rng = np.random.default_rng(seed)
    with storage.open_db(db_path) as conn:
        for batch_no, recall_rate in enumerate(recall_per_batch, start=1):
            actual = np.zeros(1000, int)
            actual[rng.choice(1000, 40, replace=False)] = 1
            labels = log_batch(conn, batch_no, actual, recall_rate, rng)
            if label_batches is None or batch_no in label_batches:
                storage.insert_labels(conn, labels, NOW)
    return db_path


@pytest.fixture
def baseline_path(tmp_path):
    # Matches the synthetic "good model" in build_db (recall ~0.9 gives PR-AUC ~0.55).
    path = tmp_path / "training_metrics.json"
    path.write_text(json.dumps({"validation_metrics": {
        "pr_auc": 0.60, "recall": 0.92, "precision": 0.5, "f1": 0.6,
        "false_positives": 30, "true_negatives": 970,
    }}))
    return path


def run(db_path, baseline_path, **kwargs):
    return run_performance_monitoring(db_path=db_path, metrics_path=baseline_path, **{**SETTINGS, **kwargs})


def stored(db_path):
    with storage.open_db(db_path) as conn:
        return storage.get_performance_windows(conn), storage.get_alerts(conn)


# --- metrics stored correctly -------------------------------------------------------
def test_window_metrics_match_direct_calculation(db_path, baseline_path):
    build_db(db_path, [0.9, 0.9, 0.9])
    run(db_path, baseline_path)

    windows, _ = stored(db_path)
    assert len(windows) == 3  # 40 fraud per batch >= 30, so each batch is one window

    with storage.open_db(db_path) as conn:
        logged = storage.get_predictions_with_labels(conn)
    batch1 = logged[logged["batch_id"] == 1]
    expected = compute_metrics(batch1["actual_label"], batch1["predicted_label"], batch1["fraud_probability"])
    first = windows.iloc[0]
    for metric in ["precision", "recall", "f1", "pr_auc", "false_positive_rate"]:
        assert first[metric] == pytest.approx(expected[metric])
    assert first["true_positives"] + first["false_negatives"] == first["n_positive"] == 40
    assert first["status"] == "ok"


# --- delayed and missing labels -------------------------------------------------------
def test_only_batches_with_arrived_labels_are_evaluated(db_path, baseline_path):
    build_db(db_path, [0.9] * 4, label_batches={1, 2})
    outcome = run(db_path, baseline_path)

    assert outcome["windows"] == [[1], [2]]
    assert outcome["waiting_for_labels"] == [3, 4]
    windows, _ = stored(db_path)
    assert windows["n_labeled"].sum() == 2000  # predictions without labels are never counted


def test_missing_labels_in_a_batch_block_later_batches(db_path, baseline_path):
    build_db(db_path, [0.9] * 3, label_batches={1, 3})  # batch 2 still waiting
    outcome = run(db_path, baseline_path)
    assert outcome["windows"] == [[1]]
    assert outcome["waiting_for_labels"] == [2, 3]


def test_new_labels_are_picked_up_on_the_next_run(db_path, fake_csv, trained_model_path, baseline_path):
    run_simulation(db_path=db_path, production_path=fake_csv, model_path=trained_model_path,
                   batch_size=400, label_delay_batches=2)

    first = run(db_path, baseline_path, min_positives=5, min_negatives=100)
    assert all(batch <= 3 for window in first["windows"] for batch in window)

    release_all_labels(db_path, fake_csv)
    second = run(db_path, baseline_path, min_positives=5, min_negatives=100)
    assert second["waiting_for_labels"] == []
    assert second["total_windows"] >= first["total_windows"]

    windows, _ = stored(db_path)
    with storage.open_db(db_path) as conn:
        labeled = storage.get_predictions_with_labels(conn)
    covered = labeled[labeled["batch_id"] <= windows["end_batch_id"].max()]
    assert windows["n_labeled"].sum() == len(covered)


def test_window_boundaries_ignore_predictions(db_path, baseline_path):
    """Windows close on actual fraud counts; a model that flags nothing gives the same windows."""
    build_db(db_path, [0.9, 0.9], seed=1)
    good = run(db_path, baseline_path)["windows"]

    other_db = db_path.parent / "other.db"
    build_db(other_db, [0.0, 0.0], seed=1)
    assert run(other_db, baseline_path)["windows"] == good


# --- insufficient data -------------------------------------------------------------------
def test_insufficient_labeled_fraud_creates_no_window(db_path, baseline_path):
    build_db(db_path, [0.9, 0.9])
    outcome = run(db_path, baseline_path, min_positives=100)  # only 80 fraud cases in total

    assert outcome["windows"] == []
    assert outcome["open_window"] == [1, 2]
    windows, alerts = stored(db_path)
    assert windows.empty and alerts.empty


# --- alerts -----------------------------------------------------------------------------
def test_recall_drop_raises_critical_alert_for_that_window_only(db_path, baseline_path):
    build_db(db_path, [0.95, 0.95, 0.3])
    run(db_path, baseline_path)

    windows, alerts = stored(db_path)
    recall_alerts = alerts[alerts["rule_name"] == "recall_below_minimum"]
    assert list(recall_alerts["start_batch_id"]) == [3]

    alert = recall_alerts.iloc[0]
    window3 = windows[windows["start_batch_id"] == 3].iloc[0]
    assert alert["severity"] == "critical"
    assert alert["observed_value"] == pytest.approx(window3["recall"])
    assert alert["threshold"] == 0.80
    assert alert["window_id"] == window3["window_id"]
    assert alert["created_at"]
    assert "Recall" in alert["description"]


def test_pr_auc_drop_alert_uses_training_baseline(db_path, baseline_path):
    build_db(db_path, [0.95, 0.2])
    run(db_path, baseline_path)

    _, alerts = stored(db_path)
    drops = alerts[alerts["rule_name"] == "pr_auc_drop_from_baseline"]
    assert list(drops["start_batch_id"]) == [2]
    assert drops.iloc[0]["baseline_value"] == 0.60


def test_custom_rules_are_used(db_path, baseline_path):
    build_db(db_path, [0.9])
    rules = [{"name": "strict_recall", "metric": "recall", "condition": "below", "threshold": 0.99,
              "severity": "info"}]
    run(db_path, baseline_path, rules=rules)
    _, alerts = stored(db_path)
    assert list(alerts["rule_name"]) == ["strict_recall"]
    assert alerts.iloc[0]["severity"] == "info"


# --- duplicates and persistence ---------------------------------------------------------
def test_repeated_runs_create_no_duplicate_windows_or_alerts(db_path, baseline_path):
    build_db(db_path, [0.95, 0.3, 0.3])
    first = run(db_path, baseline_path)
    windows_before, alerts_before = stored(db_path)
    assert first["new_alerts"] == len(alerts_before) > 0

    second = run(db_path, baseline_path)
    windows_after, alerts_after = stored(db_path)

    assert second["new_windows"] == 0 and second["new_alerts"] == 0
    pd.testing.assert_frame_equal(windows_before, windows_after)
    pd.testing.assert_frame_equal(alerts_before, alerts_after)


def test_recompute_rebuilds_identical_results(db_path, baseline_path):
    build_db(db_path, [0.95, 0.3])
    run(db_path, baseline_path)
    windows_before, alerts_before = stored(db_path)

    run(db_path, baseline_path, recompute=True)
    windows_after, alerts_after = stored(db_path)

    ignore = ["window_id", "computed_at"]
    pd.testing.assert_frame_equal(windows_before.drop(columns=ignore), windows_after.drop(columns=ignore))
    ignore = ["alert_id", "window_id", "created_at"]
    pd.testing.assert_frame_equal(alerts_before.drop(columns=ignore), alerts_after.drop(columns=ignore))


def test_results_persist_after_reopening(db_path, baseline_path):
    build_db(db_path, [0.95, 0.3])
    run(db_path, baseline_path)
    first_windows, first_alerts = stored(db_path)
    again_windows, again_alerts = stored(db_path)
    pd.testing.assert_frame_equal(first_windows, again_windows)
    pd.testing.assert_frame_equal(first_alerts, again_alerts)
    assert not first_windows.empty and not first_alerts.empty


def test_empty_database(db_path, baseline_path):
    outcome = run(db_path, baseline_path)
    assert outcome["windows"] == [] and outcome["new_alerts"] == 0


# --- command line -----------------------------------------------------------------------------
def test_command_line_run_report_and_alerts(db_path, baseline_path, capsys):
    build_db(db_path, [0.95] * 30 + [0.3] * 30)  # enough data for the default window size
    common = ["--db", str(db_path)]
    assert main(common + ["run", "--baseline", str(baseline_path)]) == 0
    assert main(common + ["report", "--baseline", str(baseline_path)]) == 0
    assert main(common + ["alerts"]) == 0

    output = capsys.readouterr().out
    assert "Model performance per window" in output
    assert "Training baseline" in output
    assert "[CRITICAL] recall_below_minimum" in output
