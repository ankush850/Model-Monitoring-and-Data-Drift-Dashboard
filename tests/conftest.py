"""Shared test helpers.

The tests use a small, made-up dataset with the same columns as the Kaggle
file. That way they run in seconds and work even before the real dataset
is downloaded.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from monitoring import config
from monitoring.data import split_features_target
from monitoring.train import build_model


def make_fake_transactions(n_rows: int = 2000, fraud_rate: float = 0.05, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(rng.normal(size=(n_rows, len(config.PCA_COLUMNS))), columns=config.PCA_COLUMNS)
    df.insert(0, config.TIME_COLUMN, np.sort(rng.uniform(0, 172_800, n_rows)))
    df["Amount"] = rng.exponential(scale=80, size=n_rows).round(2)
    df[config.TARGET_COLUMN] = (rng.random(n_rows) < fraud_rate).astype(int)

    # Give fraud rows a clear pattern so a model can learn something.
    is_fraud = df[config.TARGET_COLUMN] == 1
    df.loc[is_fraud, "V1"] -= 3
    df.loc[is_fraud, "V2"] += 3

    return df


@pytest.fixture
def fake_df() -> pd.DataFrame:
    return make_fake_transactions()


@pytest.fixture
def fake_csv(tmp_path, fake_df):
    path = tmp_path / "creditcard.csv"
    fake_df.to_csv(path, index=False)
    return path


@pytest.fixture
def trained_model_path(tmp_path, fake_df):
    """A model trained on the fake data and saved like the real one."""
    X, y = split_features_target(fake_df)
    path = tmp_path / "fraud_model.joblib"
    joblib.dump(build_model().fit(X, y), path)
    return path


@pytest.fixture
def db_path(tmp_path):
    """A fresh, empty database location for each test."""
    return tmp_path / "monitoring.db"
