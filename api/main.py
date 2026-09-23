"""FastAPI service: score transactions with the trained model and read monitoring results.

Start from the project folder (PowerShell):
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
Interactive documentation: http://127.0.0.1:8000/docs

Design notes
  * The model is the saved scikit-learn Pipeline (scaler + classifier), loaded
    once at startup with monitoring.predict.load_model.
  * Every prediction is logged to the existing monitoring database through
    monitoring.storage, as its own batch with source = "api". The batch
    monitors (drift, streaming, performance) only read simulator batches, so
    API traffic - which never receives true labels - cannot block them.
  * The API never creates the monitoring database. If it is missing,
    /predict returns 503 instead of serving predictions that cannot be logged.
  * Request bodies must contain exactly the 29 model features (plus an
    optional transaction_id). The true label "Class" is rejected.
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from monitoring import config, dashboard_data, storage
from monitoring.predict import get_model_version, load_model, predict_batch

logger = logging.getLogger("monitoring.api")

API_SOURCE = "api"
API_ID_PREFIX = "api-"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """File locations. Environment variables override the config.py defaults."""

    db_path: Path = config.DB_PATH
    model_path: Path = config.MODEL_PATH
    metrics_path: Path = config.METRICS_PATH
    threshold: float = config.DECISION_THRESHOLD

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=Path(os.environ.get("MONITORING_DB", config.DB_PATH)),
            model_path=Path(os.environ.get("MONITORING_MODEL", config.MODEL_PATH)),
            metrics_path=Path(os.environ.get("MONITORING_BASELINE", config.METRICS_PATH)),
        )


# ---------------------------------------------------------------------------
# Request and response models
# ---------------------------------------------------------------------------
# Synthetic values (randomly generated, not a transaction from the dataset);
# the same values as api/example_transaction.json and the README example.
EXAMPLE_TRANSACTION = {
    "transaction_id": "demo-0001",
    "V1": -1.916241, "V2": -0.129308, "V3": -2.103322, "V4": 1.52672, "V5": 1.390709,
    "V6": -0.427701, "V7": 0.542761, "V8": -0.357223, "V9": 1.708838, "V10": -1.268508,
    "V11": -2.070155, "V12": -0.140557, "V13": -0.355262, "V14": 0.176057, "V15": 0.393515,
    "V16": 0.95262, "V17": 0.16204, "V18": -0.369887, "V19": 0.477325, "V20": 0.909136,
    "V21": -2.142523, "V22": -1.698248, "V23": -0.721636, "V24": 0.165312, "V25": 0.264867,
    "V26": -0.340791, "V27": 0.043588, "V28": 1.663705, "Amount": 42.5,
}

# strict=True: numbers only (no "1.5" strings or true/false); allow_inf_nan=False: no NaN/Infinity.
FeatureValue = Annotated[float, Field(strict=True, allow_inf_nan=False)]
AmountValue = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]
TransactionId = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$",
          description="Optional unique ID (letters, digits, _ . : -). Generated as 'api-<uuid>' if omitted. "
                      f"IDs starting with '{config.TRANSACTION_ID_PREFIX}' are reserved for the simulator."),
]

_feature_fields: dict[str, Any] = {
    name: (FeatureValue, Field(description=f"Anonymised PCA feature {name}")) for name in config.PCA_COLUMNS
}
_feature_fields["Amount"] = (AmountValue, Field(description="Transaction amount (0 or more)"))

Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_TRANSACTION]}),
    transaction_id=(TransactionId | None, None),
    **_feature_fields,
)


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(description="Model probability that the transaction is fraud (0-1)")
    predicted_label: int = Field(description="1 = flagged as fraud, 0 = not flagged")
    decision_threshold: float
    model_version: str
    batch_id: int = Field(description="Monitoring batch this prediction was logged under")
    predicted_at: str
    logged: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    """Make values JSON-safe: NaN/NA become null, numpy numbers become Python numbers."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def database_status(db_path: Path) -> dict:
    """Check the database read-only. Never creates the file."""
    if not db_path.is_file():
        return {"status": "missing",
                "detail": "Monitoring database not found. Create it with: python -m monitoring.simulate run"}
    try:
        with dashboard_data.read_only(db_path) as conn:
            missing = dashboard_data.CORE_TABLES - dashboard_data.existing_tables(conn)
    except sqlite3.Error:
        logger.exception("Database health check failed")
        return {"status": "error", "detail": "Monitoring database could not be read."}
    if missing:
        return {"status": "error", "detail": f"Monitoring tables missing: {sorted(missing)}"}
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model, app.state.model_version, app.state.model_error = None, None, None
        try:
            app.state.model = load_model(settings.model_path)
            app.state.model_version = get_model_version(settings.model_path)
            logger.info("Model %s loaded.", app.state.model_version)
        except Exception:  # keep running so /health can report the problem
            logger.exception("Model could not be loaded")
            app.state.model_error = "Model could not be loaded. Train it first with: python -m monitoring.train"
        yield

    app = FastAPI(
        title="Fraud Model Monitoring API",
        version="1.0.0",
        description="Scores credit card transactions with the trained fraud model, logs every prediction "
                    "to the monitoring database, and exposes stored monitoring results.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # --- error handling --------------------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        errors = []
        for error in exc.errors():
            field = error["loc"][-1] if error.get("loc") else None
            message = error["msg"]
            if error["type"] == "extra_forbidden" and field == config.TARGET_COLUMN:
                message = ("'Class' is the true fraud label. It must not be sent to the model; "
                           "true labels are recorded separately when they arrive.")
            elif error["type"] == "extra_forbidden" and field == config.TIME_COLUMN:
                message = "'Time' is not a model input. Send only V1-V28 and Amount."
            elif error["type"] == "extra_forbidden":
                message = "Unexpected field. Send only transaction_id, V1-V28 and Amount."
            errors.append({"field": field, "location": list(error.get("loc", [])), "message": message,
                           "type": error["type"]})
        return JSONResponse(status_code=422, content={"detail": "Invalid request.", "errors": errors})

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    def require_model():
        if app.state.model is None:
            raise HTTPException(503, detail=app.state.model_error or "Model is not loaded.")
        return app.state.model

    # --- endpoints ---------------------------------------------------------------------
    @app.get("/health", summary="Service health")
    def health():
        model = ({"status": "ok", "version": app.state.model_version} if app.state.model is not None
                 else {"status": "unavailable", "detail": app.state.model_error})
        database = database_status(settings.db_path)
        healthy = model["status"] == "ok" and database["status"] == "ok"
        body = {"status": "ok" if healthy else "degraded", "model": model, "database": database}
        return JSONResponse(status_code=200 if healthy else 503, content=body)

    @app.get("/model-info", summary="Model metadata and version")
    def model_info():
        model = require_model()
        info = {
            "model_version": app.state.model_version,
            "model_file": settings.model_path.name,
            "pipeline_steps": [{"name": name, "type": type(step).__name__} for name, step in model.steps]
            if hasattr(model, "steps") else [type(model).__name__],
            "features": config.FEATURE_COLUMNS,
            "n_features": len(config.FEATURE_COLUMNS),
            "decision_threshold": settings.threshold,
            "label_column": config.TARGET_COLUMN,
            "label_accepted_as_input": False,
            "runtime_sklearn_version": sklearn.__version__,
            "training": None,
        }
        report = None
        if settings.metrics_path.is_file():
            try:
                with open(settings.metrics_path, encoding="utf-8") as f:
                    report = json.load(f)
            except (OSError, ValueError):
                logger.warning("Could not read %s", settings.metrics_path)
        if report:
            info["training"] = {
                "trained_at": report.get("trained_at"),
                "model_type": report.get("model_type"),
                "sklearn_version": report.get("sklearn_version"),
                "validation_metrics": report.get("validation_metrics"),
                "note": "Saved at training time; describes the model trained by `python -m monitoring.train`.",
            }
        return info

    @app.post("/predict", response_model=PredictionResponse, summary="Score one transaction and log it")
    def predict(transaction: Transaction):
        model = require_model()
        version = app.state.model_version

        transaction_id = transaction.transaction_id or f"{API_ID_PREFIX}{uuid.uuid4().hex}"
        if transaction_id.startswith(config.TRANSACTION_ID_PREFIX):
            raise HTTPException(422, detail=f"transaction_id must not start with "
                                            f"'{config.TRANSACTION_ID_PREFIX}' (reserved for the simulator).")
        # Same read-only check as /health, before anything is written: a missing, empty,
        # unreadable or unrelated SQLite file is never created, initialised or migrated here.
        db_status = database_status(settings.db_path)
        if db_status["status"] != "ok":
            raise HTTPException(503, detail="Monitoring database unavailable, so the prediction was not logged: "
                                            f"{db_status['detail']}")

        # Only the model features are passed on; the request cannot contain the label.
        features = pd.DataFrame([{name: getattr(transaction, name) for name in config.FEATURE_COLUMNS}])

        try:
            with storage.open_db(settings.db_path) as conn:
                if conn.execute("SELECT 1 FROM predictions WHERE transaction_id = ?", (transaction_id,)).fetchone():
                    raise HTTPException(409, detail=f"transaction_id '{transaction_id}' has already been scored "
                                                    "and logged. Use a new transaction_id.")

                scores = predict_batch(model, features, settings.threshold)
                probability = float(scores["fraud_probability"].iloc[0])
                label = int(scores["predicted_label"].iloc[0])
                predicted_at = utc_now()
                event_time = time.time()

                batch_id = storage.start_batch(conn, version, predicted_at, source=API_SOURCE)
                try:
                    inserted = storage.insert_predictions(
                        conn, batch_id,
                        pd.DataFrame({"transaction_id": [transaction_id], "event_time": [event_time],
                                      "fraud_probability": [probability], "predicted_label": [label]}),
                        predicted_at, settings.threshold, version,
                    )
                    if inserted != 1:  # another request logged the same ID in the meantime
                        storage.fail_batch(conn, batch_id, utc_now(), "Duplicate transaction_id")
                        raise HTTPException(409, detail=f"transaction_id '{transaction_id}' has already been "
                                                        "scored and logged.")
                    storage.finish_batch(conn, batch_id, utc_now(), {
                        "n_transactions": 1, "n_logged": 1, "n_flagged": label,
                        "mean_fraud_probability": probability,
                        "first_event_time": event_time, "last_event_time": event_time,
                    })
                except HTTPException:
                    raise
                except Exception as exc:
                    storage.fail_batch(conn, batch_id, utc_now(), type(exc).__name__)
                    raise
        except HTTPException:
            raise
        except sqlite3.Error:
            logger.exception("Database error while logging a prediction")
            raise HTTPException(503, detail="Monitoring database error: the prediction was not logged, "
                                            "so no result is returned.")

        logger.info("Scored %s: probability=%.4f label=%d batch=%d", transaction_id, probability, label, batch_id)
        return PredictionResponse(
            transaction_id=transaction_id, fraud_probability=probability, predicted_label=label,
            decision_threshold=settings.threshold, model_version=version, batch_id=batch_id,
            predicted_at=predicted_at,
        )

    @app.get("/metrics", summary="Summary of stored monitoring results")
    def metrics():
        status = database_status(settings.db_path)
        if status["status"] != "ok":
            raise HTTPException(503, detail=status["detail"])
        try:
            overview = dashboard_data.load_overview(settings.db_path)
            windows = dashboard_data.load_performance_windows(settings.db_path)
        except sqlite3.Error:
            logger.exception("Database error while reading metrics")
            raise HTTPException(503, detail="Monitoring database could not be read.")

        latest = None
        if not windows.empty:
            row = windows.iloc[-1]
            latest = {
                "batches": row["window_label"],
                "status": row["status"],
                "n_labeled": clean(row["n_labeled"]),
                "fraud_cases": clean(row["n_positive"]),
                **{m: clean(row[m]) for m in ["precision", "recall", "f1", "pr_auc", "false_positive_rate"]},
                "notes": row["notes"],
            }

        return {
            "generated_at": utc_now(),
            "source": "Stored monitoring results (read-only; nothing is recomputed).",
            "predictions": {
                "total_logged": overview["predictions"],
                "flagged_as_fraud": overview["flagged"],
                "from_api": overview["api_predictions"],
                "true_labels_received": overview["labeled"],
                "true_labels_pending": overview["pending_labels"],
            },
            "batches": {
                "simulator_completed": overview["simulator_batches"],
                "api_completed": overview["api_batches"],
                "failed": overview["failed_batches"],
            },
            "data_drift": {
                "batches_checked": overview["drift_batches"],
                "ks_drift_results": overview["ks_drift"],
                "psi_drift_results": overview["psi_drift"],
                "psi_warning_results": overview["psi_warning"],
            },
            "streaming_drift": {"detections": overview["stream_detections"]},
            "performance": {"windows_evaluated": overview["performance_windows"], "latest_window": latest},
            "alerts": {
                "total": overview["alerts_total"],
                "critical": overview["alerts_critical"],
                "warning": overview["alerts_warning"],
                "info": overview["alerts_info"],
            },
        }

    return app


app = create_app()
