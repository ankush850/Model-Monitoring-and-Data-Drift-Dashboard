"""Builds a monitoring database populated by the real pipeline, for dashboard tests."""

import hashlib
from pathlib import Path

import joblib

from conftest import make_fake_transactions
from monitoring import config
from monitoring.data import split_features_target
from monitoring.drift import run_drift_detection
from monitoring.performance import run_performance_monitoring
from monitoring.simulate import release_all_labels, run_simulation
from monitoring.streaming import run_streaming_detection
from monitoring.train import build_model

# An extra rule that is sure to fire, so the alerts page has something to show.
TEST_RULES = config.PERFORMANCE_ALERT_RULES + [
    {"name": "test_precision_below_one", "metric": "precision", "condition": "below",
     "threshold": 1.0, "severity": "info"},
]


def build_populated_project(root: Path) -> dict:
    """Run every monitoring step on fake data. Returns the file paths used.

    2,000 transactions (5 batches of 400). Labels arrive 2 batches late; the
    drift, streaming and performance steps run, then the remaining labels arrive.
    """
    df = make_fake_transactions()
    paths = {
        "db": root / "monitoring.db",
        "reference": root / "reference.csv",
        "production": root / "production.csv",
        "model": root / "fraud_model.joblib",
        "baseline": root / "training_metrics.json",
    }
    df.iloc[:1200].to_csv(paths["reference"], index=False)
    df.to_csv(paths["production"], index=False)
    X, y = split_features_target(df)
    joblib.dump(build_model().fit(X, y), paths["model"])
    paths["baseline"].write_text(
        '{"validation_metrics": {"pr_auc": 0.9, "recall": 0.9, "precision": 0.5, "f1": 0.64, '
        '"false_positives": 20, "true_negatives": 980}}'
    )

    run_simulation(db_path=paths["db"], production_path=paths["production"], model_path=paths["model"],
                   batch_size=400, label_delay_batches=2)
    run_drift_detection(db_path=paths["db"], reference_path=paths["reference"],
                        production_path=paths["production"])
    run_streaming_detection(db_path=paths["db"])
    run_performance_monitoring(db_path=paths["db"], metrics_path=paths["baseline"], rules=TEST_RULES)
    return paths


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deliver_remaining_labels(paths: dict) -> None:
    release_all_labels(paths["db"], paths["production"])
