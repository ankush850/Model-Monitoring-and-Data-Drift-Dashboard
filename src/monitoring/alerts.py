"""Alert rules: turn performance metrics into alerts when they cross configured limits.

Rules are defined in config.PERFORMANCE_ALERT_RULES, for example:
    {"name": "recall_below_minimum", "metric": "recall",
     "condition": "below", "threshold": 0.80, "severity": "critical"}
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from monitoring import config

logger = logging.getLogger(__name__)

METRICS = ("precision", "recall", "f1", "pr_auc", "false_positive_rate")
CONDITIONS = ("below", "above", "drop_from_baseline")
SEVERITIES = ("info", "warning", "critical")

METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "pr_auc": "PR-AUC",
    "false_positive_rate": "False positive rate",
}


@dataclass(frozen=True)
class AlertRule:
    name: str
    metric: str
    condition: str
    threshold: float
    severity: str

    def __post_init__(self):
        if self.metric not in METRICS:
            raise ValueError(f"Rule '{self.name}': unknown metric '{self.metric}'. Choose from {METRICS}.")
        if self.condition not in CONDITIONS:
            raise ValueError(f"Rule '{self.name}': unknown condition '{self.condition}'. Choose from {CONDITIONS}.")
        if self.severity not in SEVERITIES:
            raise ValueError(f"Rule '{self.name}': unknown severity '{self.severity}'. Choose from {SEVERITIES}.")
        if not isinstance(self.threshold, (int, float)) or not 0 <= self.threshold <= 1:
            raise ValueError(f"Rule '{self.name}': threshold must be a number between 0 and 1.")


def load_rules(raw_rules: list[dict] = config.PERFORMANCE_ALERT_RULES) -> list[AlertRule]:
    """Build and validate the alert rules from the configuration."""
    try:
        rules = [AlertRule(**raw) for raw in raw_rules]
    except TypeError as exc:
        raise ValueError(f"Invalid alert rule definition: {exc}") from exc

    names = [rule.name for rule in rules]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Alert rule names must be unique. Duplicates: {duplicates}")
    return rules


def load_baseline(path: Path | str = config.METRICS_PATH) -> dict | None:
    """Return the validation-set metrics saved at training time, or None if unavailable."""
    path = Path(path)
    if not path.exists():
        logger.warning("No baseline metrics at %s; baseline rules will be skipped.", path)
        return None

    with open(path, encoding="utf-8") as f:
        validation = json.load(f).get("validation_metrics", {})

    baseline = {metric: validation.get(metric) for metric in METRICS}
    fp, tn = validation.get("false_positives"), validation.get("true_negatives")
    if baseline["false_positive_rate"] is None and fp is not None and tn is not None and fp + tn > 0:
        baseline["false_positive_rate"] = fp / (fp + tn)
    return baseline


def evaluate_rules(
    window: dict, rules: list[AlertRule], baseline: dict | None = None, source: str = "performance"
) -> list[dict]:
    """Return an alert dict for every rule the window breaks.

    Nothing is evaluated for windows marked "insufficient_data", and a rule
    is skipped when its metric could not be calculated (None).
    """
    if window.get("status") != "ok":
        return []

    batches = f"batches {window['start_batch_id']}-{window['end_batch_id']}"
    alerts = []
    for rule in rules:
        value = window.get(rule.metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            logger.info("Rule '%s' skipped for %s: %s not available.", rule.name, batches, rule.metric)
            continue

        label = METRIC_LABELS[rule.metric]
        baseline_value = None
        if rule.condition == "below":
            fired = value < rule.threshold
            description = f"{label} {value:.4f} is below the minimum {rule.threshold:.4f} ({batches})."
        elif rule.condition == "above":
            fired = value > rule.threshold
            description = f"{label} {value:.4f} is above the maximum {rule.threshold:.4f} ({batches})."
        else:  # drop_from_baseline
            baseline_value = (baseline or {}).get(rule.metric)
            if baseline_value is None:
                logger.info("Rule '%s' skipped: no baseline value for %s.", rule.name, rule.metric)
                continue
            drop = baseline_value - value
            fired = drop > rule.threshold
            description = (
                f"{label} {value:.4f} is {drop:.4f} below the training baseline {baseline_value:.4f} "
                f"(allowed drop {rule.threshold:.4f}, {batches})."
            )

        if fired:
            alerts.append(
                {
                    "source": source,
                    "rule_name": rule.name,
                    "metric_name": rule.metric,
                    "observed_value": float(value),
                    "threshold": rule.threshold,
                    "baseline_value": baseline_value,
                    "severity": rule.severity,
                    "description": description,
                    "window_id": window.get("window_id"),
                    "start_batch_id": window["start_batch_id"],
                    "end_batch_id": window["end_batch_id"],
                }
            )
    return alerts
