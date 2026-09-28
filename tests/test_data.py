import numpy as np
import pytest

from monitoring import config
from monitoring.data import load_dataset, split_features_target, summarize, time_based_split


def test_load_dataset_reads_valid_csv(fake_csv, fake_df):
    df = load_dataset(fake_csv)
    assert len(df) == len(fake_df)
    assert list(df.columns) == config.EXPECTED_COLUMNS


def test_load_dataset_missing_file_gives_download_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        load_dataset(tmp_path / "does_not_exist.csv")


def test_load_dataset_rejects_missing_column(tmp_path, fake_df):
    path = tmp_path / "bad.csv"
    fake_df.drop(columns=["Amount"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_dataset(path)


def test_load_dataset_rejects_missing_values(tmp_path, fake_df):
    fake_df.loc[5, "V3"] = np.nan
    path = tmp_path / "bad.csv"
    fake_df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing values"):
        load_dataset(path)


def test_load_dataset_rejects_bad_labels(tmp_path, fake_df):
    fake_df.loc[0, config.TARGET_COLUMN] = 2
    path = tmp_path / "bad.csv"
    fake_df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="only contain 0 and 1"):
        load_dataset(path)


def test_time_based_split_keeps_order_and_sizes(fake_df):
    shuffled = fake_df.sample(frac=1, random_state=1)
    earlier, later = time_based_split(shuffled, 0.6)

    assert len(earlier) == int(len(fake_df) * 0.6)
    assert len(earlier) + len(later) == len(fake_df)
    # Every "earlier" row happens no later than every "later" row.
    assert earlier[config.TIME_COLUMN].max() <= later[config.TIME_COLUMN].min()


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
def test_time_based_split_rejects_bad_fraction(fake_df, fraction):
    with pytest.raises(ValueError):
        time_based_split(fake_df, fraction)


def test_split_features_target_excludes_time_and_label(fake_df):
    X, y = split_features_target(fake_df)
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert config.TIME_COLUMN not in X.columns
    assert config.TARGET_COLUMN not in X.columns
    assert y.name == config.TARGET_COLUMN


def test_summarize_counts_fraud(fake_df):
    info = summarize(fake_df)
    assert info["rows"] == len(fake_df)
    assert info["fraud_count"] == int(fake_df[config.TARGET_COLUMN].sum())
    assert info["fraud_rate"] == pytest.approx(info["fraud_count"] / len(fake_df))
