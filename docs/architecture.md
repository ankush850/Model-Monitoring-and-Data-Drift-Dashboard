# System Architecture & Database Design

This document details the architectural blueprint, data flow mechanics, and database schema of the **ML Model Monitoring & Drift Detection System**.

---

## 🏗️ High-Level System Architecture

The architecture separates training, real-time prediction, batch simulation, asynchronous monitoring, and user visualization into dedicated modules connected through a single persistent SQLite database.

```mermaid
flowchart TB
    subgraph INGESTION["1. Data Ingestion & Training"]
        RAW["Raw Transactions (creditcard.csv)"]
        SPLIT["Time-Based Split (60% Ref / 40% Prod)"]
        TRAIN["Model Trainer (StandardScaler + LogisticRegression)"]
        ARTIFACT_M[["Model Pipeline (models/fraud_model.joblib)"]]
        ARTIFACT_B[["Baseline Metrics (models/training_metrics.json)"]]
        REF_CSV[("Reference Data (data/processed/reference.csv)")]
        PROD_CSV[("Production Data (data/processed/production.csv)")]
    end

    subgraph RUNTIME["2. Runtime Execution"]
        API["FastAPI Scoring Engine (api/main.py)"]
        SIM["Production Simulator (src/monitoring/simulate.py)"]
    end

    subgraph DATABASE["3. Central Storage (SQLite: db/monitoring.db)"]
        T_BATCHES[("monitoring_batches")]
        T_PRED[("predictions")]
        T_LABELS[("actual_labels")]
        T_DRIFT[("drift_results")]
        T_STREAM[("stream_drift_results")]
        T_PERF[("performance_windows")]
        T_ALERTS[("alerts")]
    end

    subgraph MONITORS["4. Monitoring Engine"]
        M_DRIFT["Batch Drift Monitor (KS-Test & PSI)"]
        M_STREAM["Streaming Drift Monitor (ADWIN & Page-Hinkley)"]
        M_PERF["Performance & Alerting Monitor"]
    end

    subgraph UI["5. Observability Interfaces"]
        DASH["Streamlit Dashboard (dashboard/app.py)"]
        DOCS["Swagger / OpenAPI UI (127.0.0.1:8000/docs)"]
    end

    RAW --> SPLIT
    SPLIT --> REF_CSV & PROD_CSV
    REF_CSV --> TRAIN
    TRAIN --> ARTIFACT_M & ARTIFACT_B

    ARTIFACT_M --> API & SIM
    PROD_CSV --> SIM

    API --> T_BATCHES & T_PRED
    SIM --> T_BATCHES & T_PRED & T_LABELS

    T_PRED & REF_CSV --> M_DRIFT
    T_PRED & T_LABELS --> M_STREAM
    T_PRED & T_LABELS --> M_PERF

    M_DRIFT --> T_DRIFT
    M_STREAM --> T_STREAM
    M_PERF --> T_PERF & T_ALERTS

    DATABASE --> DASH
    API --> DOCS
```

---

## 🗄️ Database Schema & Entity-Relationship Model

The central SQLite database (`db/monitoring.db`) stores transaction metadata, scoring outputs, delayed labels, and monitoring calculations. Foreign key enforcement (`PRAGMA foreign_keys = ON`) is enabled on all connections.

```mermaid
erDiagram
    monitoring_batches ||--o{ predictions : contains
    monitoring_batches ||--o{ drift_results : evaluates
    monitoring_batches ||--o{ stream_drift_results : tracks
    predictions ||--o| actual_labels : "receives late label"
    performance_windows ||--o{ alerts : triggers

    monitoring_batches {
        int batch_id PK
        string status "running | completed | failed"
        string model_version "SHA256 fingerprint"
        string started_at "ISO-8601 UTC"
        string finished_at
        int n_transactions
        int n_logged
        int n_flagged
        float mean_fraud_probability
        float first_event_time
        float last_event_time
        string error_message
        string source "simulator | api"
    }

    predictions {
        string transaction_id PK
        int batch_id FK
        float event_time "Timestamp relative to dataset"
        string predicted_at "ISO-8601 UTC"
        float fraud_probability "0.0 - 1.0"
        int predicted_label "0 or 1"
        float threshold "Decision boundary (0.5)"
        string model_version
    }

    actual_labels {
        string transaction_id PK, FK
        int actual_label "0 or 1"
        string labeled_at "ISO-8601 UTC"
    }

    drift_results {
        int result_id PK
        int batch_id FK
        string feature_name
        string test_name "ks | psi"
        float score "KS stat or PSI value"
        float p_value "KS test only"
        float threshold
        float warning_threshold
        string status "no_drift | warning | drift | insufficient_data"
        int n_reference
        int n_current
        int n_missing
        string detail
        string computed_at
    }

    stream_drift_results {
        int result_id PK
        int batch_id FK
        string detector "adwin | page_hinkley"
        string signal "fraud_probability | flag_rate | error_rate"
        string status "stable | drift | no_data"
        int n_events
        int n_invalid
        int n_detections
        float signal_mean
        string first_detection_transaction_id
        float first_detection_event_time
        string detector_info "JSON metadata"
        string computed_at
    }

    performance_windows {
        int window_id PK
        int start_batch_id FK
        int end_batch_id FK
        float first_event_time
        float last_event_time
        string status "ok | insufficient_data"
        int n_labeled
        int n_positive "Actual fraud cases"
        int n_negative "Actual non-fraud cases"
        int true_positives
        int false_positives
        int true_negatives
        int false_negatives
        float precision
        float recall
        float f1
        float pr_auc
        float false_positive_rate
        float decision_threshold
        string model_versions
        string notes "JSON array"
        string computed_at
    }

    alerts {
        int alert_id PK
        string source "performance"
        string rule_name
        string metric_name
        float observed_value
        float threshold
        float baseline_value
        string severity "info | warning | critical"
        string description
        int window_id FK
        int start_batch_id
        int end_batch_id
        string created_at
    }
```

---

## 🔒 Key Architectural Safeguards

### 1. Label Separation Pattern
Predictions and actual ground-truth labels are intentionally stored in separate tables (`predictions` vs `actual_labels`).
* **Why:** In production fraud detection, true labels arrive weeks or months later following chargebacks and dispute resolution. Storing labels directly on prediction records would tempt training or validation queries to leak labels into production inference steps.

### 2. Batch Source Isolation (`simulator` vs `api`)
Every batch created in `monitoring_batches` includes a `source` column (`simulator` or `api`).
* **Why:** API predictions are single transactions that never receive true labels. The performance, streaming, and drift batch monitors exclusively aggregate batches where `source = 'simulator'`, preventing unlabelled real-time API traffic from blocking batch evaluation pipelines.

### 3. Read-Only Database Checks
The FastAPI scoring service and Streamlit dashboard use read-only SQLite connections (`?mode=ro&immutable=1` or `PRAGMA query_only = ON`) for health and metric reads, ensuring analytical queries cannot corrupt or lock active transactional tables.

