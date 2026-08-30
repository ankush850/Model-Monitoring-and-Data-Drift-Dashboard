"""Load, check and split the credit card fraud dataset."""

from pathlib import Path

import pandas as pd

from monitoring import config


def load_dataset(path: Path | str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the CSV file and make sure it looks like the Kaggle dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'creditcard.csv' from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and place it at {config.RAW_DATA_PATH}."
        )

    df = pd.read_csv(path)
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise a ValueError with a clear message if the data has problems."""
    if df.empty:
        raise ValueError("The dataset is empty.")

    missing = [col for col in config.EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing columns: {missing}")

    null_counts = df[config.EXPECTED_COLUMNS].isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        raise ValueError(
            f"The dataset has missing values: {columns_with_nulls.to_dict()}"
        )

    labels = set(df[config.TARGET_COLUMN].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"'{config.TARGET_COLUMN}' must only contain 0 and 1, found {sorted(labels)}"
        )


def time_based_split(
    df: pd.DataFrame, first_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into an earlier part and a later part, ordered by time.

    We split by time instead of randomly because, in real life, a model is
    trained on the past and then used on the future.
    """
    if not 0 < first_fraction < 1:
        raise ValueError("first_fraction must be between 0 and 1.")

    df_sorted = df.sort_values(config.TIME_COLUMN, kind="stable").reset_index(drop=True)
    cut = int(len(df_sorted) * first_fraction)

    earlier = df_sorted.iloc[:cut].reset_index(drop=True)
    later = df_sorted.iloc[cut:].reset_index(drop=True)
    return earlier, later


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the model inputs (X) from the answer we want to predict (y)."""
    return df[config.FEATURE_COLUMNS], df[config.TARGET_COLUMN]


def summarize(df: pd.DataFrame) -> dict:
    """Return a few basic facts about a DataFrame of transactions."""
    fraud_count = int(df[config.TARGET_COLUMN].sum())
    return {
        "rows": len(df),
        "fraud_count": fraud_count,
        "fraud_rate": fraud_count / len(df) if len(df) else 0.0,
        "time_start": float(df[config.TIME_COLUMN].min()),
        "time_end": float(df[config.TIME_COLUMN].max()),
    }
