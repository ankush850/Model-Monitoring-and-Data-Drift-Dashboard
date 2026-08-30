"""Central settings for the project.

Every path, column name and number that other modules need lives here,
so there is exactly one place to change them.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file is at <project>/src/monitoring/config.py, so the project root
# is two folders above the folder this file is in.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DATA_PATH = PROCESSED_DIR / "reference.csv"
PRODUCTION_DATA_PATH = PROCESSED_DIR / "production.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
METRICS_PATH = MODELS_DIR / "training_metrics.json"

# ---------------------------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------------------------
# "Time" = seconds since the first transaction in the dataset.
# "V1".."V28" = anonymised features (PCA components) provided by Kaggle.
# "Amount" = transaction amount. "Class" = 1 for fraud, 0 for normal.
TIME_COLUMN = "Time"
TARGET_COLUMN = "Class"
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]

# Columns the model learns from. "Time" is left out on purpose: it only
# ever increases, so it says nothing useful about fraud and would always
# look like "drift" to the monitoring system later.
FEATURE_COLUMNS = PCA_COLUMNS + ["Amount"]

# Every column the CSV file must contain.
EXPECTED_COLUMNS = [TIME_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

# ---------------------------------------------------------------------------
# Splitting and training
# ---------------------------------------------------------------------------
# The data is sorted by time. The first 60% is the "reference" period (what
# the model is built from). The last 40% is kept aside as "production" data
# that later phases replay batch by batch.
REFERENCE_FRACTION = 0.6

# Inside the reference period, the last 20% (by time) is the validation set
# used to measure how good the model is.
VALIDATION_FRACTION = 0.2

# Fixed seed so results are repeatable.
RANDOM_STATE = 42

# A transaction is flagged as fraud when the model's fraud probability is
# at least this value.
DECISION_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Monitoring database and production simulation
# ---------------------------------------------------------------------------
DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "monitoring.db"

# How many transactions the simulator sends to the model at once.
BATCH_SIZE = 5000

# Real fraud labels arrive late (after a customer disputes a charge). The
# simulator releases a batch's labels only after this many newer batches
# have been processed.
LABEL_DELAY_BATCHES = 2

# The dataset has no ID column, so each production row gets an ID built
# from its position in the time-sorted production file, e.g. "txn-0000042".
TRANSACTION_ID_PREFIX = "txn-"

# ---------------------------------------------------------------------------
# Data drift detection
# ---------------------------------------------------------------------------
# Features checked for drift: the same inputs the model uses. "Time" is
# excluded (it always increases) and "Class" is excluded (it is the label).
DRIFT_FEATURES = FEATURE_COLUMNS

# KS test: a feature is flagged when its p-value is below this value.
KS_ALPHA = 0.05

# PSI: number of bins, cut at reference-data quantiles (each bin holds about
# 10% of the reference data).
PSI_N_BINS = 10

# PSI rule of thumb used across the credit-risk industry:
#   PSI < 0.10         no meaningful change
#   0.10 <= PSI < 0.25 moderate change -> "warning"
#   PSI >= 0.25        significant change -> "drift"
PSI_WARNING_THRESHOLD = 0.10
PSI_DRIFT_THRESHOLD = 0.25

# Empty bins would make PSI divide by zero / take log(0). Proportions below
# this value are raised to it before PSI is calculated.
PSI_EPSILON = 1e-4

# With fewer usable values than this, results are too unreliable to judge,
# so the status is "insufficient_data" instead of a drift decision.
MIN_DRIFT_SAMPLE_SIZE = 50

# ---------------------------------------------------------------------------
# Streaming drift detection (River ADWIN and Page-Hinkley)
# ---------------------------------------------------------------------------
# ADWIN: smaller delta = more evidence needed before reporting a change
# (fewer false alarms, slower detection). 0.002 is River's default.
ADWIN_DELTA = 0.002

# Page-Hinkley (River defaults):
#   min_instances  values to see before any change can be reported
#   delta          size of change that is tolerated as normal wobble
#   threshold      how much accumulated evidence triggers a detection
#   alpha          forgetting factor for the running mean (close to 1 = long memory)
#   mode           "both" reports increases and decreases
PAGE_HINKLEY_MIN_INSTANCES = 30
PAGE_HINKLEY_DELTA = 0.005
PAGE_HINKLEY_THRESHOLD = 50.0
PAGE_HINKLEY_ALPHA = 0.9999
PAGE_HINKLEY_MODE = "both"

# ---------------------------------------------------------------------------
# Performance monitoring and alerts
# ---------------------------------------------------------------------------
# Performance is measured over windows of consecutive, fully labeled batches.
# A window is closed (and evaluated) once it holds at least this many actual
# fraud cases and actual non-fraud cases. Fraud is rare, so recall measured on
# just a handful of fraud cases would mostly be noise.
PERFORMANCE_MIN_POSITIVES = 30
PERFORMANCE_MIN_NEGATIVES = 1000

# Alert rules. Each rule checks one metric of every evaluated window.
#   condition "below":              alert when value <  threshold
#   condition "above":              alert when value >  threshold
#   condition "drop_from_baseline": alert when baseline - value > threshold,
#                                   where the baseline is the validation-set
#                                   result saved at training time (METRICS_PATH)
# severity: "info", "warning" or "critical"
PERFORMANCE_ALERT_RULES = [
    {
        "name": "recall_below_minimum",
        "metric": "recall",
        "condition": "below",
        "threshold": 0.80,
        "severity": "critical",
    },
    {
        "name": "false_positive_rate_above_maximum",
        "metric": "false_positive_rate",
        "condition": "above",
        "threshold": 0.05,
        "severity": "warning",
    },
    {
        "name": "pr_auc_drop_from_baseline",
        "metric": "pr_auc",
        "condition": "drop_from_baseline",
        "threshold": 0.10,
        "severity": "warning",
    },
]
