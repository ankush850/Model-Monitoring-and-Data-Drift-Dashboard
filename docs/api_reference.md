# FastAPI Service & Endpoint Reference

The system provides a production-ready **FastAPI** service (`api/main.py`) for online scoring, health observability, model metadata, and aggregated monitoring telemetry.

---

## 🌐 Server Execution

Start the server using `uvicorn`:

```bash
# Development (with auto-reload)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Production mode
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Interactive Documentation:
* **Swagger UI:** `http://127.0.0.1:8000/docs`
* **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## 📌 Endpoint Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness and readiness probe (checks model loading and database health). |
| `GET` | `/model-info` | Model metadata, fingerprint hash, features, decision threshold, and training metrics. |
| `POST` | `/predict` | Scores a single transaction, records the prediction in SQLite, and returns probability. |
| `GET` | `/metrics` | Read-only summary of logged predictions, batches, drift findings, performance, and alerts. |

---

## 📋 Endpoint Details

### 1. `GET /health`
Validates that the model pipeline is loaded and all required SQLite database tables exist and are readable.

#### Response: `200 OK` (Healthy)
```json
{
  "status": "ok",
  "model": {
    "status": "ok",
    "version": "a4f8d9b1e2c3"
  },
  "database": {
    "status": "ok"
  }
}
```

#### Response: `503 Service Unavailable` (Degraded)
```json
{
  "status": "degraded",
  "model": {
    "status": "unavailable",
    "detail": "Model could not be loaded. Train it first with: python -m monitoring.train"
  },
  "database": {
    "status": "missing",
    "detail": "Monitoring database not found. Create it with: python -m monitoring.simulate run"
  }
}
```

---

### 2. `GET /model-info`
Provides complete transparency into the runtime model pipeline and training validation baseline.

#### Response: `200 OK`
```json
{
  "model_version": "a4f8d9b1e2c3",
  "model_file": "fraud_model.joblib",
  "pipeline_steps": [
    {"name": "scaler", "type": "StandardScaler"},
    {"name": "classifier", "type": "LogisticRegression"}
  ],
  "features": ["V1", "V2", ..., "V28", "Amount"],
  "n_features": 29,
  "decision_threshold": 0.5,
  "label_column": "Class",
  "label_accepted_as_input": false,
  "runtime_sklearn_version": "1.5.0",
  "training": {
    "trained_at": "2026-10-29T10:15:30Z",
    "model_type": "StandardScaler + LogisticRegression(class_weight='balanced')",
    "validation_metrics": {
      "pr_auc": 0.8142,
      "roc_auc": 0.9712,
      "precision": 0.8521,
      "recall": 0.8190,
      "f1": 0.8352
    }
  }
}
```

---

### 3. `POST /predict`
Accepts a single transaction payload, scores it using the loaded pipeline, logs the prediction to `predictions` under an independent API batch, and returns the classification result.

#### Request Validation Rules
* **Strict Features:** Exactly 29 features (`V1` through `V28` as floats, and `Amount` $\ge 0$). Extra fields are strictly forbidden (`extra="forbid"`).
* **No `Class` Column:** The ground-truth label `Class` is explicitly rejected with `422 Unprocessable Entity` to prevent simulated leakage.
* **No `Time` Column:** `Time` is omitted because it is non-stationary and not an input feature.
* **Transaction ID:** Optional string matching regex `^[A-Za-z0-9_.:-]+$`. If omitted, an `api-<uuid>` identifier is generated. Reserved prefix `txn-` is rejected.

#### Request Body:
```json
{
  "transaction_id": "demo-0001",
  "V1": -1.916241, "V2": -0.129308, "V3": -2.103322, "V4": 1.526720,
  "V5": 1.390709,  "V6": -0.427701, "V7": 0.542761,  "V8": -0.357223,
  "V9": 1.708838,  "V10": -1.268508, "V11": -2.070155, "V12": -0.140557,
  "V13": -0.355262, "V14": 0.176057, "V15": 0.393515, "V16": 0.952620,
  "V17": 0.162040, "V18": -0.369887, "V19": 0.477325, "V20": 0.909136,
  "V21": -2.142523, "V22": -1.698248, "V23": -0.721636, "V24": 0.165312,
  "V25": 0.264867, "V26": -0.340791, "V27": 0.043588, "V28": 1.663705,
  "Amount": 42.50
}
```

#### Response: `200 OK`
```json
{
  "transaction_id": "demo-0001",
  "fraud_probability": 0.0241,
  "predicted_label": 0,
  "decision_threshold": 0.5,
  "model_version": "a4f8d9b1e2c3",
  "batch_id": 14,
  "predicted_at": "2026-10-29T10:45:00.123456Z",
  "logged": true
}
```

#### Error Responses:
* `409 Conflict`: The `transaction_id` has already been scored and logged.
* `422 Unprocessable Entity`: Invalid fields, missing features, or `Class` provided.
* `503 Service Unavailable`: Model is not loaded or database is unavailable.

---

### 4. `GET /metrics`
Returns an aggregate summary of all stored monitoring activity, drift test counts, and the most recent closed performance window.

#### Response: `200 OK`
```json
{
  "generated_at": "2026-10-29T10:46:12Z",
  "source": "Stored monitoring results (read-only; nothing is recomputed).",
  "predictions": {
    "total_logged": 113923,
    "flagged_as_fraud": 2145,
    "from_api": 12,
    "true_labels_received": 100000,
    "true_labels_pending": 13923
  },
  "batches": {
    "simulator_completed": 20,
    "api_completed": 12,
    "failed": 0
  },
  "data_drift": {
    "batches_checked": 20,
    "ks_drift_results": 4,
    "psi_drift_results": 2,
    "psi_warning_results": 7
  },
  "streaming_drift": {
    "detections": 3
  },
  "performance": {
    "windows_evaluated": 3,
    "latest_window": {
      "batches": "batches 15-18",
      "status": "ok",
      "n_labeled": 20000,
      "fraud_cases": 35,
      "precision": 0.8286,
      "recall": 0.8286,
      "f1": 0.8286,
      "pr_auc": 0.8015,
      "false_positive_rate": 0.0003,
      "notes": []
    }
  },
  "alerts": {
    "total": 1,
    "critical": 0,
    "warning": 1,
    "info": 0
  }
}
```

---

## 💻 Sample Client Calls

### Using PowerShell
```powershell
# Check health
Invoke-RestMethod http://127.0.0.1:8000/health

# Score a transaction
$body = Get-Content .\api\example_transaction.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body

# Retrieve monitoring metrics
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

### Using cURL
```bash
curl -X GET http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @api/example_transaction.json

curl -X GET http://127.0.0.1:8000/metrics
```

