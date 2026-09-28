"""Tests for the streaming detectors and signals (no database involved)."""

import numpy as np
import pandas as pd
import pytest

from monitoring.streaming import (
    DETECTORS,
    StreamSettings,
    make_detector,
    monitor_stream,
    signal_values,
)

BATCH_SIZE = 1000


def make_events(values, batch_size=BATCH_SIZE) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame(
        {
            "batch_id": np.arange(n) // batch_size + 1,
            "transaction_id": [f"t{i:06d}" for i in range(n)],
            "event_time": np.arange(n, dtype=float),
            "value": values,
        }
    )


def first_detection_index(values, detector_name):
    detector = make_detector(detector_name)
    for i, value in enumerate(values):
        detector.update(float(value))
        if detector.drift_detected:
            return i
    return None


# --- stable, sudden and gradual streams ----------------------------------------
@pytest.mark.parametrize("detector_name", DETECTORS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stable_error_stream_has_no_detections(detector_name, seed):
    values = np.random.default_rng(seed).binomial(1, 0.05, 6000)
    results = monitor_stream(make_events(values), make_detector(detector_name))

    assert [r["status"] for r in results] == ["stable"] * 6
    assert sum(r["n_detections"] for r in results) == 0


@pytest.mark.parametrize("detector_name", DETECTORS)
def test_stable_continuous_scores_have_no_detections(detector_name):
    values = np.random.default_rng(7).beta(1, 20, 6000)  # fraud-probability-like values
    results = monitor_stream(make_events(values), make_detector(detector_name))
    assert all(r["status"] == "stable" for r in results)


@pytest.mark.parametrize("detector_name", DETECTORS)
def test_sudden_change_is_detected_soon_after_it_happens(detector_name):
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.binomial(1, 0.05, 3000), rng.binomial(1, 0.40, 3000)])

    index = first_detection_index(values, detector_name)
    assert index is not None
    assert 3000 <= index < 3500

    results = monitor_stream(make_events(values), make_detector(detector_name))
    statuses = {r["batch_id"]: r["status"] for r in results}
    assert statuses[1] == statuses[2] == statuses[3] == "stable"
    assert statuses[4] == "drift"


@pytest.mark.parametrize("detector_name", DETECTORS)
def test_gradual_change_is_detected_after_it_starts(detector_name):
    rng = np.random.default_rng(1)
    rate = np.concatenate([np.full(2000, 0.05), np.linspace(0.05, 0.40, 3000), np.full(1000, 0.40)])
    values = rng.binomial(1, rate)

    index = first_detection_index(values, detector_name)
    assert index is not None
    assert 2000 <= index < 5000


@pytest.mark.parametrize("detector_name", DETECTORS)
def test_detection_details_are_recorded(detector_name):
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.binomial(1, 0.05, 3000), rng.binomial(1, 0.40, 3000)])
    batch4 = monitor_stream(make_events(values), make_detector(detector_name))[3]

    assert batch4["n_detections"] >= 1
    assert batch4["first_detection_transaction_id"].startswith("t003")
    assert 3000 <= batch4["first_detection_event_time"] < 4000
    assert batch4["detector_info"]["detections"][0]["transaction_id"] == batch4["first_detection_transaction_id"]


# --- order and reproducibility ------------------------------------------------
def test_events_are_processed_in_time_order_whatever_the_input_order():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.binomial(1, 0.05, 3000), rng.binomial(1, 0.40, 3000)])
    events = make_events(values)
    shuffled = events.sample(frac=1, random_state=3)

    in_order = monitor_stream(events, make_detector("adwin"))
    from_shuffled = monitor_stream(shuffled, make_detector("adwin"))
    assert in_order == from_shuffled


def test_results_are_deterministic():
    values = np.random.default_rng(5).binomial(1, 0.1, 3000)
    events = make_events(values)
    for name in DETECTORS:
        assert monitor_stream(events, make_detector(name)) == monitor_stream(events, make_detector(name))


def test_overlapping_batches_are_rejected():
    events = make_events(np.zeros(2000))
    events.loc[events.index[:10], "batch_id"] = 2  # batch 2 events before batch 1 events
    with pytest.raises(ValueError, match="overlap in time"):
        monitor_stream(events, make_detector("adwin"))


# --- missing and invalid values -------------------------------------------------
def test_missing_values_are_skipped_and_counted():
    values = np.random.default_rng(0).binomial(1, 0.05, 2000).astype(float)
    values[:50] = np.nan
    values[50:55] = np.inf

    results = monitor_stream(make_events(values), make_detector("adwin"))
    assert results[0]["n_invalid"] == 55
    assert results[0]["n_events"] == 945
    assert results[1]["n_invalid"] == 0


def test_batch_with_no_valid_values_has_no_data_status():
    values = np.concatenate([np.full(1000, np.nan), np.zeros(1000)])
    results = monitor_stream(make_events(values), make_detector("page_hinkley"))
    assert results[0]["status"] == "no_data"
    assert results[0]["signal_mean"] is None
    assert results[1]["status"] == "stable"


def test_stream_missing_columns_is_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        monitor_stream(pd.DataFrame({"batch_id": [1], "value": [0.0]}), make_detector("adwin"))


def test_signal_values_mark_invalid_entries_as_missing():
    predictions = pd.DataFrame(
        {
            "fraud_probability": [0.2, 1.5, -0.1, None, 0.9],
            "predicted_label": [0, 1, 2, 1, 1],
            "actual_label": [0, 0, 1, None, 1],
        }
    )
    prob = signal_values(predictions, "fraud_probability")
    flag = signal_values(predictions, "flag_rate")
    error = signal_values(predictions, "error_rate")

    assert prob.tolist()[0] == 0.2 and prob.iloc[1:4].isna().all() and prob.iloc[4] == 0.9
    assert flag.tolist()[:2] == [0.0, 1.0] and np.isnan(flag.iloc[2])
    # error is only defined where both labels are valid and the true label is known
    assert error.iloc[0] == 0.0
    assert error.iloc[1] == 1.0
    assert error.iloc[2:4].isna().all()
    assert error.iloc[4] == 0.0


def test_error_rate_needs_actual_labels_column():
    with pytest.raises(ValueError, match="actual_label"):
        signal_values(pd.DataFrame({"predicted_label": [0, 1]}), "error_rate")


def test_unknown_signal_and_detector_are_rejected():
    with pytest.raises(ValueError, match="Unknown signal"):
        signal_values(pd.DataFrame({"x": [1]}), "accuracy")
    with pytest.raises(ValueError, match="Unknown detector"):
        make_detector("ddm")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"adwin_delta": 0},
        {"adwin_delta": 1},
        {"ph_min_instances": 0},
        {"ph_delta": -1},
        {"ph_threshold": 0},
        {"ph_alpha": 0},
        {"ph_mode": "sideways"},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        StreamSettings(**kwargs)
