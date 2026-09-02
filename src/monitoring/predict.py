"""Load the trained model and use it to score transactions."""

import hashlib
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from monitoring import config

logger = logging.getLogger(__name__)


def load_model(path: Path | str = config.MODEL_PATH) -> Pipeline:
    """Load the saved model.

    The saved file is the whole scikit-learn Pipeline, so the fitted
    StandardScaler (the preprocessing step) is loaded together with the
    classifier. New data is scaled exactly as the training data was.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train it first with: python -m monitoring.train"
        )

    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{path} does not contain a model with predict_proba().")

    logger.info("Loaded model from %s", path)
    return model


def get_model_version(path: Path | str = config.MODEL_PATH) -> str:
    """Return a short fingerprint of the model file.

    Retraining produces a different file and therefore a different version,
    so every logged prediction can be traced back to the model that made it.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def predict_batch(
    model: Pipeline,
    features: pd.DataFrame,
    threshold: float = config.DECISION_THRESHOLD,
) -> pd.DataFrame:
    """Return the fraud probability and the 0/1 decision for each row.

    The true label column is refused on purpose: a model in production
    never knows the answer when it makes a prediction.
    """
    if config.TARGET_COLUMN in features.columns:
        raise ValueError(
            f"The '{config.TARGET_COLUMN}' column holds the actual labels and must not "
            "be passed to the model. Remove it before predicting."
        )

    missing = [col for col in config.FEATURE_COLUMNS if col not in features.columns]
    if missing:
        raise ValueError(f"Cannot predict: missing feature columns {missing}")

    probabilities = model.predict_proba(features[config.FEATURE_COLUMNS])[:, 1]
    return pd.DataFrame(
        {
            "fraud_probability": probabilities,
            "predicted_label": (probabilities >= threshold).astype(int),
        },
        index=features.index,
    )
