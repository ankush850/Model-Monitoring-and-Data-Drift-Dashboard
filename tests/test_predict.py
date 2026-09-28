import pytest

from monitoring import config
from monitoring.predict import get_model_version, load_model, predict_batch


def test_load_model_missing_file_says_how_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="monitoring.train"):
        load_model(tmp_path / "missing.joblib")


def test_model_version_is_stable_for_same_file(trained_model_path):
    version = get_model_version(trained_model_path)
    assert len(version) == 12
    assert version == get_model_version(trained_model_path)


def test_predict_batch_returns_probabilities_and_labels(trained_model_path, fake_df):
    model = load_model(trained_model_path)
    features = fake_df.drop(columns=[config.TARGET_COLUMN])

    result = predict_batch(model, features, threshold=0.5)

    assert len(result) == len(fake_df)
    assert result["fraud_probability"].between(0, 1).all()
    assert (result["predicted_label"] == (result["fraud_probability"] >= 0.5)).all()


def test_predict_batch_refuses_actual_labels(trained_model_path, fake_df):
    model = load_model(trained_model_path)
    with pytest.raises(ValueError, match="actual labels"):
        predict_batch(model, fake_df)


def test_predict_batch_needs_all_features(trained_model_path, fake_df):
    model = load_model(trained_model_path)
    features = fake_df.drop(columns=[config.TARGET_COLUMN, "Amount"])
    with pytest.raises(ValueError, match="missing feature columns"):
        predict_batch(model, features)
