"""Train the fraud detection model, evaluate it, and save the results.

Run from the project folder with:
    python -m monitoring.train
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from monitoring import config
from monitoring.data import load_dataset, split_features_target, summarize, time_based_split


def build_model() -> Pipeline:
    """Create an untrained model.

    StandardScaler puts every feature on a similar scale (important because
    "Amount" is much larger than the V columns). LogisticRegression is a
    simple, fast, easy-to-explain classifier. class_weight="balanced" tells
    it to pay extra attention to the rare fraud cases.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = config.DECISION_THRESHOLD,
) -> dict:
    """Measure how well the model finds fraud.

    We do not report accuracy: with so few fraud cases, a model that always
    says "not fraud" would look almost perfect while catching nothing.
    """
    if y.nunique() < 2:
        raise ValueError("Evaluation data must contain both fraud and non-fraud rows.")

    fraud_scores = model.predict_proba(X)[:, 1]
    predictions = (fraud_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()

    return {
        "rows": len(y),
        "fraud_cases": int(y.sum()),
        "threshold": threshold,
        "pr_auc": float(average_precision_score(y, fraud_scores)),
        "roc_auc": float(roc_auc_score(y, fraud_scores)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def run_training(
    raw_data_path: Path | str = config.RAW_DATA_PATH,
    processed_dir: Path | str = config.PROCESSED_DIR,
    models_dir: Path | str = config.MODELS_DIR,
) -> dict:
    """Run the full training process and return the saved metrics report."""
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and check the data.
    df = load_dataset(raw_data_path)

    # 2. Keep the newest 40% aside as future "production" data.
    reference_df, production_df = time_based_split(df, config.REFERENCE_FRACTION)

    # 3. Split the reference period into training and validation (by time).
    train_df, validation_df = time_based_split(
        reference_df, 1 - config.VALIDATION_FRACTION
    )

    # 4. Train the model.
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(validation_df)
    model = build_model()
    model.fit(X_train, y_train)

    # 5. Evaluate on data the model has not seen.
    validation_metrics = evaluate_model(model, X_val, y_val)

    # 6. Save everything later phases need.
    reference_df.to_csv(processed_dir / config.REFERENCE_DATA_PATH.name, index=False)
    production_df.to_csv(processed_dir / config.PRODUCTION_DATA_PATH.name, index=False)
    joblib.dump(model, models_dir / config.MODEL_PATH.name)

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "StandardScaler + LogisticRegression(class_weight='balanced')",
        "sklearn_version": sklearn.__version__,
        "features": config.FEATURE_COLUMNS,
        "data": {
            "full": summarize(df),
            "train": summarize(train_df),
            "validation": summarize(validation_df),
            "production_holdout": summarize(production_df),
        },
        "validation_metrics": validation_metrics,
    }
    with open(models_dir / config.METRICS_PATH.name, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def print_report(report: dict) -> None:
    """Print the training results in a readable way."""
    print("\n=== Data ===")
    for name, info in report["data"].items():
        print(
            f"{name:<20} rows={info['rows']:>7,}  "
            f"fraud={info['fraud_count']:>4}  ({info['fraud_rate']:.3%})"
        )

    m = report["validation_metrics"]
    print("\n=== Validation metrics ===")
    print(f"PR-AUC     {m['pr_auc']:.4f}   (main metric; higher is better)")
    print(f"ROC-AUC    {m['roc_auc']:.4f}")
    print(f"Precision  {m['precision']:.4f}   (of flagged rows, share that were fraud)")
    print(f"Recall     {m['recall']:.4f}   (of real fraud, share we caught)")
    print(f"F1         {m['f1']:.4f}")
    print(
        f"\nCaught {m['true_positives']} of {m['fraud_cases']} fraud cases, "
        f"with {m['false_positives']:,} false alarms (threshold {m['threshold']})."
    )


def main() -> None:
    report = run_training()
    print_report(report)
    print(f"\nModel saved to:   {config.MODEL_PATH}")
    print(f"Metrics saved to: {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
