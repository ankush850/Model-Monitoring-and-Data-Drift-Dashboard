"""Tests for the FastAPI service. All tests use temporary files; the real database is never touched."""

import json
import math
import sqlite3

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import Settings, create_app
from dashboard_fixtures import file_hash
from monitoring import config, storage
from monitoring.performance import run_performance_monitoring
from monitoring.predict import get_model_version, load_model, predict_batch
from monitoring.simulate import release_all_labels, run_simulation

PROTECTED_FILES = [config.DB_PATH, config.MODEL_PATH, config.METRICS_PATH]


@pytest.fixture(autouse=True)
def real_files_are_not_modified():
    """Fail if any test changes the real database, model or training metrics."""
    before = {p: file_hash(p) for p in PROTECTED_FILES if p.is_file()}
    yield
    after = {p: file_hash(p) for p in PROTECTED_FILES if p.is_file()}
    assert before == after


@pytest.fixture
def ready_db(db_path):
    """An empty, initialised monitoring database in a temporary folder."""
    with storage.open_db(db_path):
        pass
    return db_path


@pytest.fixture
def baseline_path(tmp_path):
    path = tmp_path / "training_metrics.json"
    path.write_text(json.dumps({
        "trained_at": "2026-01-01T00:00:00+00:00", "model_type": "test", "sklearn_version": "x",
        "validation_metrics": {"pr_auc": 0.9, "recall": 0.9, "precision": 0.5, "f1": 0.6,
                               "false_positives": 20, "true_negatives": 980},
    }))
    return path


@pytest.fixture
def make_client(trained_model_path, baseline_path):
    clients = []

    def factory(db, model=None, raise_server_exceptions=True):
        settings = Settings(db_path=db, model_path=model or trained_model_path, metrics_path=baseline_path)
        client = TestClient(create_app(settings), raise_server_exceptions=raise_server_exceptions)
        client.__enter__()  # runs the startup (model loading)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def payload(fake_df):
    row = fake_df.iloc[0]
    return {name: float(row[name]) for name in config.FEATURE_COLUMNS}


def logged_predictions(db) -> pd.DataFrame:
    with storage.open_db(db) as conn:
        return storage.get_predictions_with_labels(conn)


def batches(db) -> list[dict]:
    with storage.open_db(db) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM monitoring_batches ORDER BY batch_id")]


# --- /health -------------------------------------------------------------------------------
def test_health_ok(make_client, ready_db, trained_model_path):
    response = make_client(ready_db).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == {"status": "ok", "version": get_model_version(trained_model_path)}
    assert body["database"]["status"] == "ok"


def test_health_reports_missing_database_without_creating_it(make_client, db_path):
    response = make_client(db_path).get("/health")
    assert response.status_code == 503
    assert response.json()["database"]["status"] == "missing"
    assert not db_path.exists()


def test_health_reports_missing_model(make_client, ready_db, tmp_path):
    response = make_client(ready_db, model=tmp_path / "missing.joblib").get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["model"]["status"] == "unavailable"
    assert "python -m monitoring.train" in body["model"]["detail"]


def test_health_reports_unreadable_database(make_client, tmp_path):
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"this is not a sqlite database" * 100)
    response = make_client(broken).get("/health")
    assert response.status_code == 503
    assert response.json()["database"]["status"] == "error"


# --- /model-info -------------------------------------------------------------------------------
def test_model_info(make_client, ready_db, trained_model_path):
    body = make_client(ready_db).get("/model-info").json()
    assert body["model_version"] == get_model_version(trained_model_path)
    assert body["features"] == config.FEATURE_COLUMNS
    assert body["n_features"] == 29
    assert config.TARGET_COLUMN not in body["features"] and config.TIME_COLUMN not in body["features"]
    assert body["label_accepted_as_input"] is False
    assert [s["type"] for s in body["pipeline_steps"]] == ["StandardScaler", "LogisticRegression"]
    assert body["training"]["validation_metrics"]["pr_auc"] == 0.9


def test_model_info_without_model(make_client, ready_db, tmp_path):
    response = make_client(ready_db, model=tmp_path / "missing.joblib").get("/model-info")
    assert response.status_code == 503


# --- /predict: valid requests ---------------------------------------------------------------------
def test_predict_returns_and_logs_prediction(make_client, ready_db, payload, trained_model_path):
    response = make_client(ready_db).post("/predict", json={"transaction_id": "client-001", **payload})
    assert response.status_code == 200
    body = response.json()

    # Same result as calling the existing pipeline directly.
    expected = predict_batch(load_model(trained_model_path), pd.DataFrame([payload]))
    assert body["fraud_probability"] == pytest.approx(expected["fraud_probability"].iloc[0])
    assert body["predicted_label"] == int(expected["predicted_label"].iloc[0])
    assert body["predicted_label"] == int(body["fraud_probability"] >= body["decision_threshold"])
    assert body["model_version"] == get_model_version(trained_model_path)
    assert body["transaction_id"] == "client-001"

    logged = logged_predictions(ready_db)
    assert list(logged["transaction_id"]) == ["client-001"]
    assert logged.iloc[0]["fraud_probability"] == pytest.approx(body["fraud_probability"])
    assert logged.iloc[0]["model_version"] == body["model_version"]
    assert pd.isna(logged.iloc[0]["actual_label"])  # no label is ever stored by the API

    (batch,) = batches(ready_db)
    assert batch["batch_id"] == body["batch_id"]
    assert (batch["source"], batch["status"], batch["n_logged"]) == ("api", "completed", 1)


def test_predict_generates_unique_ids(make_client, ready_db, payload):
    client = make_client(ready_db)
    first = client.post("/predict", json=payload).json()["transaction_id"]
    second = client.post("/predict", json=payload).json()["transaction_id"]
    assert first.startswith("api-") and second.startswith("api-") and first != second
    assert len(logged_predictions(ready_db)) == 2


def test_integer_feature_values_are_accepted(make_client, ready_db, payload):
    payload = {**payload, "Amount": 100, "V1": 0}
    assert make_client(ready_db).post("/predict", json=payload).status_code == 200


# --- /predict: duplicates ------------------------------------------------------------------------------
def test_duplicate_transaction_id_is_rejected(make_client, ready_db, payload):
    client = make_client(ready_db)
    assert client.post("/predict", json={"transaction_id": "dup-1", **payload}).status_code == 200
    response = client.post("/predict", json={"transaction_id": "dup-1", **payload})

    assert response.status_code == 409
    assert "already been scored" in response.json()["detail"]
    assert len(logged_predictions(ready_db)) == 1
    assert len(batches(ready_db)) == 1  # no extra batch for the rejected request


def test_simulator_id_prefix_is_reserved(make_client, ready_db, payload):
    response = make_client(ready_db).post("/predict", json={"transaction_id": "txn-0000001", **payload})
    assert response.status_code == 422
    assert "reserved for the simulator" in response.json()["detail"]
    assert logged_predictions(ready_db).empty


# --- /predict: invalid input -------------------------------------------------------------------------------
def error_fields(response) -> list:
    return [e["field"] for e in response.json()["errors"]]


def test_missing_feature_is_rejected(make_client, ready_db, payload):
    del payload["V7"]
    response = make_client(ready_db).post("/predict", json=payload)
    assert response.status_code == 422
    assert error_fields(response) == ["V7"]
    assert logged_predictions(ready_db).empty


def test_all_missing_features_are_listed(make_client, ready_db):
    response = make_client(ready_db).post("/predict", json={})
    assert response.status_code == 422
    assert sorted(error_fields(response)) == sorted(config.FEATURE_COLUMNS)


def test_true_label_is_rejected(make_client, ready_db, payload):
    response = make_client(ready_db).post("/predict", json={**payload, "Class": 1})
    assert response.status_code == 422
    (error,) = response.json()["errors"]
    assert error["field"] == "Class"
    assert "true fraud label" in error["message"]
    assert logged_predictions(ready_db).empty


@pytest.mark.parametrize("extra", [{"Time": 1234.0}, {"merchant": "shop"}])
def test_unexpected_fields_are_rejected(make_client, ready_db, payload, extra):
    response = make_client(ready_db).post("/predict", json={**payload, **extra})
    assert response.status_code == 422
    assert error_fields(response) == list(extra)


@pytest.mark.parametrize("bad_value", ["1.5", "abc", None, True, [1.0], {"x": 1}])
def test_non_numeric_values_are_rejected(make_client, ready_db, payload, bad_value):
    response = make_client(ready_db).post("/predict", json={**payload, "V3": bad_value})
    assert response.status_code == 422
    assert error_fields(response) == ["V3"]


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_are_rejected(make_client, ready_db, payload, token):
    body = json.dumps(payload).replace(str(payload["V2"]), token, 1)
    assert token in body
    response = make_client(ready_db).post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert error_fields(response) == ["V2"]
    assert logged_predictions(ready_db).empty


def test_negative_amount_is_rejected(make_client, ready_db, payload):
    response = make_client(ready_db).post("/predict", json={**payload, "Amount": -5.0})
    assert response.status_code == 422
    assert error_fields(response) == ["Amount"]


@pytest.mark.parametrize("bad_id", ["", "has space", "a" * 65, "semi;colon"])
def test_invalid_transaction_id_is_rejected(make_client, ready_db, payload, bad_id):
    response = make_client(ready_db).post("/predict", json={"transaction_id": bad_id, **payload})
    assert response.status_code == 422


def test_malformed_json_is_rejected(make_client, ready_db):
    response = make_client(ready_db).post("/predict", content="{not json",
                                          headers={"Content-Type": "application/json"})
    assert response.status_code == 422


# --- /predict: model and database failures --------------------------------------------------------------
def test_predict_without_database_returns_503_and_creates_nothing(make_client, db_path, payload):
    response = make_client(db_path).post("/predict", json=payload)
    assert response.status_code == 503
    assert "not found" in response.json()["detail"]
    assert not db_path.exists()


def test_predict_with_broken_database_returns_503_without_internals(make_client, tmp_path, payload):
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"this is not a sqlite database" * 100)
    response = make_client(broken).post("/predict", json=payload)
    assert response.status_code == 503
    assert "not logged" in response.json()["detail"]
    assert "Traceback" not in response.text and "sqlite3" not in response.text


def test_predict_without_model_returns_503(make_client, ready_db, payload, tmp_path):
    response = make_client(ready_db, model=tmp_path / "missing.joblib").post("/predict", json=payload)
    assert response.status_code == 503
    assert logged_predictions(ready_db).empty


def test_unexpected_error_returns_generic_500(make_client, ready_db, payload, monkeypatch):
    import api.main

    def explode(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(api.main, "predict_batch", explode)
    response = make_client(ready_db, raise_server_exceptions=False).post("/predict", json=payload)
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert batches(ready_db) == []  # the model failed before any batch was started


# --- /metrics -------------------------------------------------------------------------------------------
@pytest.fixture
def monitored_db(db_path, fake_csv, trained_model_path, baseline_path):
    """A database filled by the simulator, with labels and performance results."""
    run_simulation(db_path=db_path, production_path=fake_csv, model_path=trained_model_path,
                   batch_size=400, label_delay_batches=0)
    release_all_labels(db_path, fake_csv)
    run_performance_monitoring(db_path=db_path, metrics_path=baseline_path)
    return db_path


def test_metrics_report_stored_results(make_client, monitored_db):
    response = make_client(monitored_db).get("/metrics")
    assert response.status_code == 200
    body = response.json()

    with storage.open_db(monitored_db) as conn:
        summary = storage.get_summary(conn)
        windows = storage.get_performance_windows(conn)
    assert body["predictions"]["total_logged"] == summary["predictions"] == 2000
    assert body["predictions"]["flagged_as_fraud"] == summary["flagged"]
    assert body["batches"]["simulator_completed"] == 5
    assert body["performance"]["windows_evaluated"] == len(windows)
    latest = body["performance"]["latest_window"]
    assert latest["recall"] == pytest.approx(windows.iloc[-1]["recall"])
    assert "transaction_id" not in json.dumps(body)  # summary only, no transaction-level data


def test_metrics_count_api_predictions_separately(make_client, monitored_db, payload):
    client = make_client(monitored_db)
    client.post("/predict", json=payload)
    body = client.get("/metrics").json()
    assert body["predictions"]["from_api"] == 1
    assert body["predictions"]["total_logged"] == 2001
    assert body["batches"]["api_completed"] == 1
    assert body["batches"]["simulator_completed"] == 5


def test_metrics_do_not_change_the_database(make_client, monitored_db):
    before = file_hash(monitored_db)
    make_client(monitored_db).get("/metrics")
    assert file_hash(monitored_db) == before


def test_metrics_without_database(make_client, db_path):
    response = make_client(db_path).get("/metrics")
    assert response.status_code == 503
    assert not db_path.exists()


def test_metrics_values_are_json_safe(make_client, ready_db):
    body = make_client(ready_db).get("/metrics").json()
    assert body["performance"]["latest_window"] is None
    assert all(not (isinstance(v, float) and math.isnan(v)) for v in body["predictions"].values())


def test_openapi_docs_available(make_client, ready_db):
    client = make_client(ready_db)
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert {"/health", "/model-info", "/predict", "/metrics"} <= set(schema["paths"])
    transaction = schema["components"]["schemas"]["Transaction"]
    assert "Class" not in transaction["properties"]
    assert transaction["additionalProperties"] is False


# --- /predict never creates, initialises or migrates an unusable database file ------------------------
def test_predict_with_empty_file_returns_503_and_leaves_it_empty(make_client, tmp_path, payload):
    empty = tmp_path / "empty.db"
    empty.write_bytes(b"")
    response = make_client(empty).post("/predict", json=payload)
    assert response.status_code == 503
    assert "not logged" in response.json()["detail"]
    assert empty.stat().st_size == 0


def test_predict_with_unrelated_sqlite_file_does_not_change_it(make_client, tmp_path, payload):
    other = tmp_path / "other_app.db"
    with sqlite3.connect(other) as conn:
        conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO customers (name) VALUES ('x')")
    conn.close()
    before = file_hash(other)

    response = make_client(other).post("/predict", json=payload)
    assert response.status_code == 503
    assert "tables missing" in response.json()["detail"]
    assert file_hash(other) == before


def test_predict_with_broken_database_leaves_file_unchanged(make_client, tmp_path, payload):
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"this is not a sqlite database" * 100)
    before = file_hash(broken)
    assert make_client(broken).post("/predict", json=payload).status_code == 503
    assert file_hash(broken) == before


# --- /metrics: pending labels and stored notes -------------------------------------------------------
def test_metrics_pending_labels_exclude_api_predictions(make_client, monitored_db, payload):
    client = make_client(monitored_db)
    before = client.get("/metrics").json()["predictions"]
    assert client.post("/predict", json=payload).status_code == 200
    after = client.get("/metrics").json()["predictions"]

    assert before["true_labels_pending"] == after["true_labels_pending"] == 0  # every simulator label arrived
    assert after["from_api"] == before["from_api"] + 1
    assert set(after) == set(before)  # response schema unchanged


def test_metrics_pending_labels_count_unlabeled_simulator_predictions(make_client, db_path, fake_csv,
                                                                      trained_model_path):
    run_simulation(db_path=db_path, production_path=fake_csv, model_path=trained_model_path,
                   batch_size=400, label_delay_batches=2)  # labels of the last 2 batches have not arrived
    body = make_client(db_path).get("/metrics").json()
    assert body["predictions"]["true_labels_pending"] == 2 * 400
    assert body["predictions"]["from_api"] == 0


@pytest.mark.parametrize(
    ("stored", "expected"),
    [("{not json", ["notes could not be read: {not json"]), ("null", []), ('"text"', ["text"]),
     ("5", ["notes could not be read: 5"]), ('["kept note"]', ["kept note"])],
)
def test_metrics_with_damaged_notes(make_client, monitored_db, stored, expected):
    with sqlite3.connect(monitored_db) as conn:
        conn.execute("UPDATE performance_windows SET notes = ?", (stored,))
    conn.close()
    response = make_client(monitored_db, raise_server_exceptions=False).get("/metrics")
    assert response.status_code == 200
    assert response.json()["performance"]["latest_window"]["notes"] == expected
    assert "Traceback" not in response.text
