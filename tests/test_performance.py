"""Tests for performance metric calculations and window building (no database)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from monitoring.performance import build_windows, compute_metrics, parse_notes

SMALL = {"min_positives": 1, "min_negatives": 1}


# --- correct calculations -------------------------------------------------------
def test_hand_calculated_example():
    actual = [1, 1, 0, 0, 0]
    predicted = [1, 0, 1, 0, 0]
    scores = [0.9, 0.4, 0.6, 0.2, 0.1]

    m = compute_metrics(actual, predicted, scores, **SMALL)

    assert (m["true_positives"], m["false_negatives"], m["false_positives"], m["true_negatives"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5)            # 1 of 2 flagged was fraud
    assert m["recall"] == pytest.approx(0.5)               # 1 of 2 frauds caught
    assert m["f1"] == pytest.approx(0.5)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)  # 1 of 3 normal flagged
    assert m["status"] == "ok"
    assert m["notes"] == []


def test_metrics_match_scikit_learn():
    rng = np.random.default_rng(0)
    actual = rng.binomial(1, 0.1, 5000)
    scores = np.clip(actual * 0.5 + rng.normal(0.25, 0.2, 5000), 0, 1)
    predicted = (scores >= 0.5).astype(int)

    m = compute_metrics(actual, predicted, scores)
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()

    assert (m["true_negatives"], m["false_positives"], m["false_negatives"], m["true_positives"]) == (tn, fp, fn, tp)
    assert m["precision"] == pytest.approx(precision_score(actual, predicted))
    assert m["recall"] == pytest.approx(recall_score(actual, predicted))
    assert m["f1"] == pytest.approx(f1_score(actual, predicted))
    assert m["pr_auc"] == pytest.approx(average_precision_score(actual, scores))
    assert m["false_positive_rate"] == pytest.approx(fp / (fp + tn))


# --- single-class and empty windows -----------------------------------------------
def test_window_without_fraud_cannot_measure_recall_or_pr_auc():
    m = compute_metrics([0] * 2000, [0] * 1990 + [1] * 10, [0.1] * 2000)

    assert m["recall"] is None
    assert m["pr_auc"] is None
    assert m["f1"] is None
    assert m["precision"] == 0.0
    assert m["false_positive_rate"] == pytest.approx(10 / 2000)
    assert m["status"] == "insufficient_data"
    assert any("no actual fraud" in note for note in m["notes"])
    assert any("only one class" in note for note in m["notes"])


def test_window_with_only_fraud_cannot_measure_false_positive_rate():
    m = compute_metrics([1] * 50, [1] * 40 + [0] * 10, [0.9] * 50, **SMALL)
    assert m["false_positive_rate"] is None
    assert m["pr_auc"] is None
    assert m["recall"] == pytest.approx(0.8)


def test_nothing_flagged_means_precision_is_undefined():
    m = compute_metrics([1, 0, 0, 0], [0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1], **SMALL)
    assert m["precision"] is None
    assert m["recall"] == 0.0
    assert m["f1"] is None
    assert any("no transactions were flagged" in note for note in m["notes"])


def test_zero_precision_and_recall_give_zero_f1():
    m = compute_metrics([1, 0], [0, 1], [0.2, 0.8], **SMALL)
    assert m["precision"] == 0.0 and m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_empty_window():
    m = compute_metrics([], [], [])
    assert m["n_labeled"] == 0
    assert all(m[k] is None for k in ["precision", "recall", "f1", "pr_auc", "false_positive_rate"])
    assert m["status"] == "insufficient_data"


# --- insufficient data and invalid inputs ---------------------------------------------
def test_too_few_fraud_cases_is_insufficient_data():
    actual = [1] * 5 + [0] * 5000
    m = compute_metrics(actual, actual, [0.9] * 5 + [0.1] * 5000, min_positives=30, min_negatives=1000)
    assert m["recall"] == 1.0  # calculated, but flagged as unreliable
    assert m["status"] == "insufficient_data"
    assert any("only 5 fraud cases" in note for note in m["notes"])


def test_missing_scores_skip_pr_auc_only():
    m = compute_metrics([1, 0], [1, 0], None, **SMALL)
    assert m["pr_auc"] is None
    assert m["recall"] == 1.0
    m = compute_metrics([1, 0], [1, 0], [np.nan, 0.1], **SMALL)
    assert m["pr_auc"] is None


@pytest.mark.parametrize(
    "actual, predicted, scores",
    [
        ([1, 0, 2], [1, 0, 0], None),     # invalid true label
        ([1, 0], [1, 0, 0], None),        # length mismatch
        ([1, 0], [1, -1], None),          # invalid prediction
        ([1, 0], [1, 0], [0.5]),          # scores length mismatch
    ],
)
def test_invalid_inputs_are_rejected(actual, predicted, scores):
    with pytest.raises(ValueError):
        compute_metrics(actual, predicted, scores)


# --- windows ---------------------------------------------------------------------------
def labeled_batches(fraud_per_batch, size=1000):
    rows = []
    for batch_id, n_fraud in enumerate(fraud_per_batch, start=1):
        rows += [{"batch_id": batch_id, "actual_label": int(i < n_fraud)} for i in range(size)]
    return pd.DataFrame(rows)


def test_windows_close_once_they_have_enough_fraud_cases():
    labeled = labeled_batches([10, 10, 15, 40, 5, 5])
    windows, open_window = build_windows(labeled, [1, 2, 3, 4, 5, 6], min_positives=30, min_negatives=900)

    assert windows == [[1, 2, 3], [4]]  # 35 and 40 fraud cases
    assert open_window == [5, 6]        # only 10 so far: waits for more labeled data


def test_window_stays_open_until_it_has_enough_non_fraud_cases():
    labeled = labeled_batches([40, 5])  # batch 1: 40 fraud but only 960 non-fraud
    windows, open_window = build_windows(labeled, [1, 2], min_positives=30, min_negatives=1000)
    assert windows == [[1, 2]]
    assert open_window == []


def test_windows_need_enough_non_fraud_cases_too():
    labeled = labeled_batches([30, 30], size=600)
    windows, open_window = build_windows(labeled, [1, 2], min_positives=30, min_negatives=1000)
    assert windows == [[1, 2]]
    assert open_window == []


def test_windows_only_use_ready_batches():
    labeled = labeled_batches([40, 40, 40])
    windows, open_window = build_windows(labeled, [1], min_positives=30, min_negatives=500)
    assert windows == [[1]]
    assert open_window == []


def test_window_settings_are_validated():
    labeled = labeled_batches([1])
    with pytest.raises(ValueError):
        build_windows(labeled, [1], min_positives=0)


# --- stored notes ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('["recall: no actual fraud cases in this window."]', ["recall: no actual fraud cases in this window."]),
        ('["a", "b"]', ["a", "b"]),
        ("[]", []),
        ("", []),
        (None, []),
        ("null", []),
        ('"one note"', ["one note"]),
        ("{not json", ["notes could not be read: {not json"]),
        ("5", ["notes could not be read: 5"]),
        ('{"a": 1}', ['notes could not be read: {"a": 1}']),
        ("true", ["notes could not be read: true"]),
    ],
)
def test_parse_notes(raw, expected):
    assert parse_notes(raw) == expected


def test_parse_notes_keeps_long_raw_values_short():
    (note,) = parse_notes("{" + "x" * 1000)
    assert len(note) < 250
