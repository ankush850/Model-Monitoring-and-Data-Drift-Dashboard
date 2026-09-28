"""Tests for alert rules (no database)."""

import json

import pytest

from monitoring import config
from monitoring.alerts import AlertRule, evaluate_rules, load_baseline, load_rules

RULES = load_rules([
    {"name": "low_recall", "metric": "recall", "condition": "below", "threshold": 0.80, "severity": "critical"},
    {"name": "high_fpr", "metric": "false_positive_rate", "condition": "above", "threshold": 0.05, "severity": "warning"},
    {"name": "pr_auc_drop", "metric": "pr_auc", "condition": "drop_from_baseline", "threshold": 0.10, "severity": "warning"},
])
BASELINE = {"pr_auc": 0.87, "recall": 0.92}


def window(**metrics):
    return {"status": "ok", "start_batch_id": 1, "end_batch_id": 3, "window_id": 7,
            "recall": 0.9, "false_positive_rate": 0.03, "pr_auc": 0.85, **metrics}


def fired(alerts):
    return sorted(a["rule_name"] for a in alerts)


def test_healthy_window_raises_no_alerts():
    assert evaluate_rules(window(), RULES, BASELINE) == []


def test_each_rule_fires_when_its_limit_is_crossed():
    assert fired(evaluate_rules(window(recall=0.79), RULES, BASELINE)) == ["low_recall"]
    assert fired(evaluate_rules(window(false_positive_rate=0.051), RULES, BASELINE)) == ["high_fpr"]
    assert fired(evaluate_rules(window(pr_auc=0.76), RULES, BASELINE)) == ["pr_auc_drop"]


def test_values_exactly_at_the_limit_do_not_fire():
    w = window(recall=0.80, false_positive_rate=0.05, pr_auc=0.77)
    assert evaluate_rules(w, RULES, BASELINE) == []


def test_alert_contains_required_fields():
    alert = evaluate_rules(window(recall=0.5), RULES, BASELINE)[0]
    assert alert["metric_name"] == "recall"
    assert alert["observed_value"] == 0.5
    assert alert["threshold"] == 0.80
    assert alert["severity"] == "critical"
    assert (alert["start_batch_id"], alert["end_batch_id"], alert["window_id"]) == (1, 3, 7)
    assert "below the minimum" in alert["description"] and "batches 1-3" in alert["description"]


def test_baseline_alert_records_baseline_value():
    alert = evaluate_rules(window(pr_auc=0.60), RULES, BASELINE)[0]
    assert alert["baseline_value"] == 0.87
    assert "0.2700 below the training baseline" in alert["description"]


def test_baseline_rule_is_skipped_without_baseline():
    assert evaluate_rules(window(pr_auc=0.10), RULES, baseline=None) == []


def test_missing_metric_is_skipped():
    assert evaluate_rules(window(recall=None, pr_auc=None), RULES, BASELINE) == []


def test_insufficient_data_window_raises_no_alerts():
    w = window(status="insufficient_data", recall=0.0, false_positive_rate=0.9, pr_auc=0.0)
    assert evaluate_rules(w, RULES, BASELINE) == []


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "x", "metric": "accuracy", "condition": "below", "threshold": 0.5, "severity": "warning"},
        {"name": "x", "metric": "recall", "condition": "equals", "threshold": 0.5, "severity": "warning"},
        {"name": "x", "metric": "recall", "condition": "below", "threshold": 0.5, "severity": "urgent"},
        {"name": "x", "metric": "recall", "condition": "below", "threshold": 1.5, "severity": "warning"},
        {"name": "x", "metric": "recall", "condition": "below", "severity": "warning"},
    ],
)
def test_invalid_rules_are_rejected(bad):
    with pytest.raises(ValueError):
        load_rules([bad])


def test_duplicate_rule_names_are_rejected():
    rule = {"name": "x", "metric": "recall", "condition": "below", "threshold": 0.5, "severity": "warning"}
    with pytest.raises(ValueError, match="unique"):
        load_rules([rule, rule])


def test_configured_rules_are_valid():
    rules = load_rules(config.PERFORMANCE_ALERT_RULES)
    assert all(isinstance(rule, AlertRule) for rule in rules)


def test_load_baseline_reads_validation_metrics(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"validation_metrics": {
        "pr_auc": 0.8, "recall": 0.9, "precision": 0.1, "f1": 0.18,
        "false_positives": 10, "true_negatives": 990,
    }}))
    baseline = load_baseline(path)
    assert baseline["pr_auc"] == 0.8
    assert baseline["false_positive_rate"] == pytest.approx(0.01)


def test_load_baseline_missing_file_returns_none(tmp_path):
    assert load_baseline(tmp_path / "missing.json") is None
