"""Tests for running streaming detectors over the database, including delayed labels."""

import json

import numpy as np
import pandas as pd
import pytest

from monitoring import config, storage
from monitoring.simulate import release_all_labels, run_simulation
from monitoring.streaming import DETECTORS, SIGNALS, main, run_streaming_detection

NOW = "2026-01-01T00:00:00+00:00"
LABEL_FREE = ["fraud_probability", "flag_rate"]


def log_batch(conn, batch_no, probabilities, predicted, start_time):
    """Log one batch of predictions directly (bypassing the model)."""
    n = len(predicted)
    batch_id = storage.start_batch(conn, "v1", NOW)
    predictions = pd.DataFrame(
        {
            "transaction_id": [f"b{batch_no:02d}-{i:05d}" for i in range(n)],
            "event_time": start_time + np.arange(n, dtype=float),
            "fraud_probability": probabilities,
            "predicted_label": predicted,
        }
    )
    storage.insert_predictions(conn, batch_id, predictions, NOW, 0.5, "v1")
    storage.finish_batch(conn, batch_id, NOW, {
        "n_transactions": n, "n_logged": n, "n_flagged": int(np.sum(predicted)),
        "mean_fraud_probability": float(np.mean(probabilities)),
        "first_event_time": float(start_time), "last_event_time": float(start_time + n - 1),
    })
    return batch_id, predictions["transaction_id"]


def results(db_path, signal=None) -> pd.DataFrame:
    with storage.open_db(db_path) as conn:
        return storage.get_stream_results(conn, signal=signal)


def statuses(df, detector):
    rows = df[df["detector"] == detector]
    return dict(zip(rows["batch_id"], rows["status"]))


@pytest.fixture
def error_jump_db(db_path):
    """6 batches. The model always predicts 'not fraud' with the same score,
    but from batch 4 the true fraud rate jumps from 5% to 40%, so errors jump."""
    rng = np.random.default_rng(0)
    with storage.open_db(db_path) as conn:
        for batch_no in range(1, 7):
            _, ids = log_batch(conn, batch_no, np.full(1000, 0.01), np.zeros(1000, int), batch_no * 10_000)
            rate = 0.05 if batch_no <= 3 else 0.40
            storage.insert_labels(conn, pd.Series(rng.binomial(1, rate, 1000), index=ids), NOW)
    return db_path


@pytest.fixture
def simulated_db(db_path, fake_csv, trained_model_path):
    """Real simulator run on fake data: 5 batches, labels 2 batches behind."""
    run_simulation(db_path=db_path, production_path=fake_csv, model_path=trained_model_path,
                   batch_size=400, label_delay_batches=2)
    return db_path


# --- detecting changes --------------------------------------------------------------
def test_error_rate_change_is_detected_while_outputs_stay_stable(error_jump_db):
    run_streaming_detection(error_jump_db)
    errors = results(error_jump_db, "error_rate")

    for detector in DETECTORS:
        s = statuses(errors, detector)
        assert s[1] == s[2] == s[3] == "stable"
        assert s[4] == "drift"

    for signal in LABEL_FREE:  # model outputs never changed
        assert (results(error_jump_db, signal)["status"] == "stable").all()


def test_output_shift_without_more_errors_is_not_a_performance_change(db_path):
    """Scores rise but stay below the threshold and labels stay the same:
    fraud_probability changes, error_rate does not."""
    rng = np.random.default_rng(1)
    with storage.open_db(db_path) as conn:
        for batch_no in range(1, 7):
            score = 0.01 if batch_no <= 3 else 0.30
            _, ids = log_batch(conn, batch_no, np.full(1000, score), np.zeros(1000, int), batch_no * 10_000)
            storage.insert_labels(conn, pd.Series(rng.binomial(1, 0.05, 1000), index=ids), NOW)

    run_streaming_detection(db_path)

    probability = results(db_path, "fraud_probability")
    assert (probability.loc[probability["batch_id"] == 4, "status"] == "drift").all()
    assert (results(db_path, "error_rate")["status"] == "stable").all()


# --- delayed labels ---------------------------------------------------------------
def test_error_rate_waits_for_delayed_labels(simulated_db, fake_csv):
    outcome = run_streaming_detection(simulated_db)

    # Labels have arrived for batches 1-3 only.
    assert outcome["label_ready_batches"] == [1, 2, 3]
    assert outcome["waiting_for_labels"] == [4, 5]
    assert set(results(simulated_db, "error_rate")["batch_id"]) == {1, 2, 3}
    for signal in LABEL_FREE:
        assert set(results(simulated_db, signal)["batch_id"]) == {1, 2, 3, 4, 5}

    before = results(simulated_db)

    # The remaining labels arrive; the next run adds batches 4-5 for error_rate only.
    release_all_labels(simulated_db, fake_csv)
    second = run_streaming_detection(simulated_db)
    assert second["waiting_for_labels"] == []
    assert second["new_rows"] == 2 * len(DETECTORS)

    after = results(simulated_db)
    assert set(after.loc[after["signal"] == "error_rate", "batch_id"]) == {1, 2, 3, 4, 5}
    # Rows stored earlier are untouched.
    key = ["signal", "detector", "batch_id"]
    pd.testing.assert_frame_equal(
        before.set_index(key).sort_index(),
        after.set_index(key).loc[before.set_index(key).index].sort_index(),
    )


def test_error_rate_stops_at_first_batch_missing_labels(db_path):
    with storage.open_db(db_path) as conn:
        batch_ids = []
        for batch_no in range(1, 4):
            _, ids = log_batch(conn, batch_no, np.full(200, 0.01), np.zeros(200, int), batch_no * 1000)
            batch_ids.append(ids)
        storage.insert_labels(conn, pd.Series(0, index=batch_ids[0]), NOW)  # batch 1 labeled
        storage.insert_labels(conn, pd.Series(0, index=batch_ids[2]), NOW)  # batch 3 labeled, 2 not

    outcome = run_streaming_detection(db_path)
    assert outcome["label_ready_batches"] == [1]
    assert outcome["waiting_for_labels"] == [2, 3]
    assert set(results(db_path, "error_rate")["batch_id"]) == {1}


def test_label_free_signals_do_not_depend_on_labels(tmp_path, fake_df, trained_model_path):
    flipped = fake_df.copy()
    flipped[config.TARGET_COLUMN] = 1 - flipped[config.TARGET_COLUMN]

    outputs = []
    for name, df in [("a", fake_df), ("b", flipped)]:
        csv, db = tmp_path / f"{name}.csv", tmp_path / f"{name}.db"
        df.to_csv(csv, index=False)
        run_simulation(db_path=db, production_path=csv, model_path=trained_model_path, batch_size=400)
        run_streaming_detection(db)
        stored = results(db)
        outputs.append(stored[stored["signal"].isin(LABEL_FREE)].drop(columns="computed_at").reset_index(drop=True))

    pd.testing.assert_frame_equal(outputs[0], outputs[1])


# --- persistence and duplicates --------------------------------------------------------
def test_results_are_stored_with_required_fields(error_jump_db):
    run_streaming_detection(error_jump_db)
    stored = results(error_jump_db)

    assert len(stored) == 6 * len(SIGNALS) * len(DETECTORS)
    for column in ["batch_id", "detector", "signal", "status", "n_events", "computed_at", "detector_info"]:
        assert stored[column].notna().all(), column

    drift_row = stored[(stored["signal"] == "error_rate") & (stored["status"] == "drift")].iloc[0]
    assert drift_row["first_detection_transaction_id"].startswith("b04")
    info = json.loads(drift_row["detector_info"])
    assert "settings" in info and info["detections"]


def test_results_persist_after_reopening(error_jump_db):
    run_streaming_detection(error_jump_db)
    pd.testing.assert_frame_equal(results(error_jump_db), results(error_jump_db))
    assert not results(error_jump_db).empty


def test_running_again_creates_no_duplicates(error_jump_db):
    first = run_streaming_detection(error_jump_db)
    stored_first = results(error_jump_db)

    second = run_streaming_detection(error_jump_db)
    stored_second = results(error_jump_db)

    assert first["new_rows"] == len(stored_first)
    assert second["new_rows"] == 0
    pd.testing.assert_frame_equal(stored_first, stored_second)
    assert not stored_second.duplicated(["batch_id", "detector", "signal"]).any()


def test_recompute_rebuilds_identical_results(error_jump_db):
    run_streaming_detection(error_jump_db)
    before = results(error_jump_db).drop(columns="computed_at")
    run_streaming_detection(error_jump_db, recompute=True)
    after = results(error_jump_db).drop(columns="computed_at")
    pd.testing.assert_frame_equal(before, after)


def test_failed_batches_are_ignored(error_jump_db):
    with storage.open_db(error_jump_db) as conn:
        failed = storage.start_batch(conn, "v1", NOW)
        storage.fail_batch(conn, failed, NOW, "boom")
    run_streaming_detection(error_jump_db)
    assert failed not in set(results(error_jump_db)["batch_id"])


def test_empty_database_stores_nothing(db_path):
    outcome = run_streaming_detection(db_path)
    assert outcome["new_rows"] == 0
    assert results(db_path).empty


# --- command line -----------------------------------------------------------------
def test_command_line_run_and_report(error_jump_db, capsys):
    common = ["--db", str(error_jump_db)]
    assert main(common + ["run"]) == 0
    assert main(common + ["report", "--signal", "error_rate"]) == 0

    output = capsys.readouterr().out
    assert "Streaming detectors: summary" in output
    assert "Streaming detectors: error_rate" in output
    assert "page_hinkley" in output
