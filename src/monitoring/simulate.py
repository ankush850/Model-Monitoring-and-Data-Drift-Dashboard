"""Replay the saved production data through the model, batch by batch.

This pretends the newest 40% of the dataset is live traffic:
  * transactions arrive in time order, in batches,
  * the model scores each batch and the results are logged to SQLite,
  * the true fraud labels arrive later (a few batches behind), like real
    chargebacks, and are stored in a separate table.

Examples (run from the project folder):
    python -m monitoring.simulate run
    python -m monitoring.simulate run --batch-size 2000 --max-batches 5
    python -m monitoring.simulate labels
    python -m monitoring.simulate status
"""

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from monitoring import config, storage
from monitoring.data import validate_dataset
from monitoring.predict import get_model_version, load_model, predict_batch

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_transaction_id(position: int) -> str:
    return f"{config.TRANSACTION_ID_PREFIX}{position:07d}"


def load_production_data(path: Path | str = config.PRODUCTION_DATA_PATH) -> pd.DataFrame:
    """Read the production file, sort it by time and give each row an ID."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Production data not found at {path}. Create it first with: python -m monitoring.train"
        )

    df = pd.read_csv(path)
    validate_dataset(df)
    df = df.sort_values(config.TIME_COLUMN, kind="stable").reset_index(drop=True)
    df.insert(0, "transaction_id", [make_transaction_id(i) for i in range(len(df))])
    return df


def make_batches(df: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    """Cut the rows into consecutive chunks of batch_size (the last may be smaller)."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    return [df.iloc[start : start + batch_size] for start in range(0, len(df), batch_size)]


def process_batch(
    conn: sqlite3.Connection,
    model: Pipeline,
    model_version: str,
    batch_df: pd.DataFrame,
    threshold: float = config.DECISION_THRESHOLD,
) -> dict:
    """Score one batch and log the predictions. Actual labels are never used here."""
    batch_id = storage.start_batch(conn, model_version, utc_now())
    try:
        scores = predict_batch(model, batch_df[config.FEATURE_COLUMNS], threshold)
        predictions = pd.DataFrame(
            {
                "transaction_id": batch_df["transaction_id"],
                "event_time": batch_df[config.TIME_COLUMN],
                "fraud_probability": scores["fraud_probability"],
                "predicted_label": scores["predicted_label"],
            }
        )
        n_logged = storage.insert_predictions(
            conn, batch_id, predictions, utc_now(), threshold, model_version
        )
        stats = {
            "n_transactions": len(predictions),
            "n_logged": n_logged,
            "n_flagged": int(predictions["predicted_label"].sum()),
            "mean_fraud_probability": float(predictions["fraud_probability"].mean()),
            "first_event_time": float(predictions["event_time"].min()),
            "last_event_time": float(predictions["event_time"].max()),
        }
        storage.finish_batch(conn, batch_id, utc_now(), stats)
    except Exception as exc:
        logger.exception("Batch %d failed.", batch_id)
        storage.fail_batch(conn, batch_id, utc_now(), f"{type(exc).__name__}: {exc}")
        raise

    logger.info(
        "Batch %d: logged %d transactions, flagged %d as fraud.",
        batch_id, stats["n_logged"], stats["n_flagged"],
    )
    return {"batch_id": batch_id, **stats}


def release_labels(conn: sqlite3.Connection, true_labels: pd.Series, up_to_batch_id: int) -> int:
    """Store the true labels for every unlabeled prediction in batches <= up_to_batch_id."""
    pending = storage.get_unlabeled_transaction_ids(conn, up_to_batch_id)
    if not pending:
        return 0

    found = true_labels.reindex(pending).dropna().astype(int)
    if len(found) < len(pending):
        logger.warning("No true label found for %d transactions.", len(pending) - len(found))
    return storage.insert_labels(conn, found, utc_now())


def run_simulation(
    db_path: Path | str = config.DB_PATH,
    production_path: Path | str = config.PRODUCTION_DATA_PATH,
    model_path: Path | str = config.MODEL_PATH,
    batch_size: int = config.BATCH_SIZE,
    max_batches: int | None = None,
    label_delay_batches: int = config.LABEL_DELAY_BATCHES,
    threshold: float = config.DECISION_THRESHOLD,
) -> dict:
    """Send not-yet-logged production transactions to the model in batches.

    Running it again continues where the last run stopped; transactions that
    are already in the database are not processed twice.
    """
    if label_delay_batches < 0:
        raise ValueError("label_delay_batches cannot be negative.")

    model = load_model(model_path)
    model_version = get_model_version(model_path)
    production = load_production_data(production_path)

    # Keep the answers apart from the inputs. Only `true_labels` holds them,
    # and it is used only when labels are released after the delay.
    true_labels = production.set_index("transaction_id")[config.TARGET_COLUMN]
    transactions = production.drop(columns=[config.TARGET_COLUMN])

    results = []
    with storage.open_db(db_path) as conn:
        already_logged = storage.get_logged_transaction_ids(conn)
        remaining = transactions[~transactions["transaction_id"].isin(already_logged)]
        if already_logged:
            logger.info("%d transactions were already logged and will be skipped.", len(already_logged))

        batches = make_batches(remaining, batch_size)
        if max_batches is not None:
            batches = batches[:max_batches]
        if not batches:
            logger.info("Nothing new to process: every production transaction is already logged.")

        for batch_df in batches:
            result = process_batch(conn, model, model_version, batch_df, threshold)
            results.append(result)

            n_labels = release_labels(conn, true_labels, result["batch_id"] - label_delay_batches)
            if n_labels:
                logger.info("Labels arrived for %d earlier transactions.", n_labels)

        summary = storage.get_summary(conn)

    return {"batches": results, "summary": summary}


def release_all_labels(
    db_path: Path | str = config.DB_PATH,
    production_path: Path | str = config.PRODUCTION_DATA_PATH,
    up_to_batch_id: int | None = None,
) -> int:
    """Store true labels for every pending prediction (optionally only up to a batch)."""
    production = load_production_data(production_path)
    true_labels = production.set_index("transaction_id")[config.TARGET_COLUMN]

    with storage.open_db(db_path) as conn:
        if up_to_batch_id is None:
            up_to_batch_id = conn.execute(
                "SELECT COALESCE(MAX(batch_id), 0) FROM monitoring_batches"
            ).fetchone()[0]
        n_labels = release_labels(conn, true_labels, up_to_batch_id)

    logger.info("Stored %d new labels.", n_labels)
    return n_labels


def print_summary(summary: dict, db_path: Path | str) -> None:
    print(f"\n=== Monitoring database: {db_path} ===")
    print(
        f"Batches:            {summary['batches']:,} "
        f"(completed {summary['completed_batches']:,}, failed {summary['failed_batches']:,})"
    )
    print(f"Predictions logged: {summary['predictions']:,}")
    print(f"Flagged as fraud:   {summary['flagged']:,}")
    print(f"Labels received:    {summary['labeled']:,}")
    print(f"Labels pending:     {summary['pending_labels']:,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate production traffic for the fraud model.")
    parser.add_argument("--db", type=Path, default=config.DB_PATH, help="SQLite database file")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="score production transactions in batches")
    run.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    run.add_argument("--max-batches", type=int, default=None, help="stop after this many batches")
    run.add_argument(
        "--label-delay", type=int, default=config.LABEL_DELAY_BATCHES,
        help="batches to wait before true labels arrive",
    )
    run.add_argument("--production-data", type=Path, default=config.PRODUCTION_DATA_PATH)
    run.add_argument("--model", type=Path, default=config.MODEL_PATH)

    labels = commands.add_parser("labels", help="add true labels for logged predictions")
    labels.add_argument("--up-to-batch", type=int, default=None, help="only batches up to this ID")
    labels.add_argument("--production-data", type=Path, default=config.PRODUCTION_DATA_PATH)

    commands.add_parser("status", help="show what is in the database")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        if args.command == "run":
            run_simulation(
                db_path=args.db,
                production_path=args.production_data,
                model_path=args.model,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                label_delay_batches=args.label_delay,
            )
        elif args.command == "labels":
            release_all_labels(args.db, args.production_data, args.up_to_batch)

        with storage.open_db_readonly(args.db) as conn:
            print_summary(storage.get_summary(conn), args.db)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
