import json

import joblib
import pandas as pd
import pytest

from monitoring import config
from monitoring.data import split_features_target
from monitoring.train import build_model, evaluate_model, run_training


def test_evaluate_model_returns_valid_metrics(fake_df):
    X, y = split_features_target(fake_df)
    model = build_model().fit(X, y)

    metrics = evaluate_model(model, X, y)

    for name in ["pr_auc", "roc_auc", "precision", "recall", "f1"]:
        assert 0.0 <= metrics[name] <= 1.0
    counts = ["true_positives", "false_positives", "true_negatives", "false_negatives"]
    assert sum(metrics[c] for c in counts) == len(y)


def test_evaluate_model_needs_both_classes(fake_df):
    X, y = split_features_target(fake_df)
    model = build_model().fit(X, y)
    only_normal = y[y == 0]
    with pytest.raises(ValueError, match="both fraud and non-fraud"):
        evaluate_model(model, X.loc[only_normal.index], only_normal)


def test_run_training_saves_model_metrics_and_splits(tmp_path, fake_csv, fake_df):
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"

    report = run_training(fake_csv, processed_dir, models_dir)

    # Files exist where later phases expect them.
    model_file = models_dir / config.MODEL_PATH.name
    metrics_file = models_dir / config.METRICS_PATH.name
    reference_file = processed_dir / config.REFERENCE_DATA_PATH.name
    production_file = processed_dir / config.PRODUCTION_DATA_PATH.name
    for path in [model_file, metrics_file, reference_file, production_file]:
        assert path.exists(), f"{path.name} was not created"

    # Saved model can be loaded and used for predictions.
    model = joblib.load(model_file)
    X, _ = split_features_target(fake_df.head(5))
    assert model.predict_proba(X).shape == (5, 2)

    # Saved metrics match the returned report.
    with open(metrics_file, encoding="utf-8") as f:
        assert json.load(f)["validation_metrics"] == report["validation_metrics"]

    # Reference + production together are the whole dataset, with no overlap.
    reference = pd.read_csv(reference_file)
    production = pd.read_csv(production_file)
    assert len(reference) + len(production) == len(fake_df)
    assert reference[config.TIME_COLUMN].max() <= production[config.TIME_COLUMN].min()


def test_model_learns_better_than_guessing(tmp_path, fake_csv, fake_df):
    report = run_training(fake_csv, tmp_path / "processed", tmp_path / "models")
    fraud_rate = fake_df[config.TARGET_COLUMN].mean()
    # A random guesser gets PR-AUC close to the fraud rate.
    assert report["validation_metrics"]["pr_auc"] > fraud_rate * 3
