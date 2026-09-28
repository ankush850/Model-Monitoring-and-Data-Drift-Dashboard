import pandas as pd
import pytest

from monitoring import config, simulate, storage
from monitoring.simulate import (
    load_production_data,
    main,
    make_batches,
    release_all_labels,
    run_simulation,
)

BATCH_SIZE = 400  # the fake dataset has 2,000 rows, so 5 batches


def run(db_path, production_path, model_path, **kwargs):
    kwargs.setdefault("batch_size", BATCH_SIZE)
    return run_simulation(
        db_path=db_path, production_path=production_path, model_path=model_path, **kwargs
    )


def read_logged(db_path) -> pd.DataFrame:
    with storage.open_db(db_path) as conn:
        return storage.get_predictions_with_labels(conn)


def test_make_batches_splits_into_chunks(fake_df):
    sizes = [len(batch) for batch in make_batches(fake_df, 600)]
    assert sizes == [600, 600, 600, 200]
    with pytest.raises(ValueError):
        make_batches(fake_df, 0)


def test_load_production_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="monitoring.train"):
        load_production_data(tmp_path / "missing.csv")


def test_load_production_data_adds_unique_ids_in_time_order(fake_csv):
    df = load_production_data(fake_csv)
    assert df["transaction_id"].is_unique
    assert df["transaction_id"].iloc[0] == f"{config.TRANSACTION_ID_PREFIX}0000000"
    assert df[config.TIME_COLUMN].is_monotonic_increasing


def test_simulation_logs_every_transaction_in_chronological_batches(db_path, fake_csv, trained_model_path, fake_df):
    result = run(db_path, fake_csv, trained_model_path, label_delay_batches=0)

    assert len(result["batches"]) == 5
    assert result["summary"]["completed_batches"] == 5
    assert result["summary"]["predictions"] == len(fake_df)

    # Each batch is entirely later in time than the batch before it.
    per_batch = read_logged(db_path).groupby("batch_id")["event_time"].agg(["min", "max"]).sort_index()
    assert (per_batch["max"].to_numpy()[:-1] <= per_batch["min"].to_numpy()[1:]).all()


def test_running_twice_does_not_duplicate(db_path, fake_csv, trained_model_path, fake_df):
    run(db_path, fake_csv, trained_model_path)
    second = run(db_path, fake_csv, trained_model_path)

    assert second["batches"] == []
    assert second["summary"]["predictions"] == len(fake_df)
    assert second["summary"]["batches"] == 5


def test_simulation_resumes_where_it_stopped(db_path, fake_csv, trained_model_path, fake_df):
    first = run(db_path, fake_csv, trained_model_path, max_batches=2)
    assert first["summary"]["predictions"] == 2 * BATCH_SIZE

    second = run(db_path, fake_csv, trained_model_path)
    assert len(second["batches"]) == 3
    assert second["summary"]["predictions"] == len(fake_df)


def test_labels_arrive_after_a_delay(db_path, fake_csv, trained_model_path, fake_df):
    result = run(db_path, fake_csv, trained_model_path, label_delay_batches=2)

    # 5 batches with a 2-batch delay: labels have arrived for batches 1-3 only.
    logged = read_logged(db_path)
    labeled_batches = set(logged.loc[logged["actual_label"].notna(), "batch_id"])
    batch_ids = [b["batch_id"] for b in result["batches"]]
    assert labeled_batches == set(batch_ids[:3])
    assert result["summary"]["pending_labels"] == 2 * BATCH_SIZE

    # Later, the rest of the labels come in.
    assert release_all_labels(db_path, fake_csv) == 2 * BATCH_SIZE

    logged = read_logged(db_path)
    assert logged["actual_label"].notna().all()

    # The stored labels are the real ones from the production file.
    truth = load_production_data(fake_csv).set_index("transaction_id")[config.TARGET_COLUMN]
    stored = logged.set_index("transaction_id")["actual_label"].astype(int)
    assert stored.sort_index().equals(truth.sort_index())


def test_predictions_do_not_depend_on_actual_labels(tmp_path, fake_df, trained_model_path):
    original_csv = tmp_path / "original.csv"
    flipped_csv = tmp_path / "flipped.csv"
    fake_df.to_csv(original_csv, index=False)
    flipped = fake_df.copy()
    flipped[config.TARGET_COLUMN] = 1 - flipped[config.TARGET_COLUMN]
    flipped.to_csv(flipped_csv, index=False)

    run(tmp_path / "a.db", original_csv, trained_model_path)
    run(tmp_path / "b.db", flipped_csv, trained_model_path)

    a = read_logged(tmp_path / "a.db").set_index("transaction_id")["fraud_probability"]
    b = read_logged(tmp_path / "b.db").set_index("transaction_id")["fraud_probability"]
    pd.testing.assert_series_equal(a, b)


def test_failed_batch_is_recorded(db_path, fake_csv, trained_model_path, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(simulate, "predict_batch", broken_predict)

    with pytest.raises(RuntimeError, match="model crashed"):
        run(db_path, fake_csv, trained_model_path)

    with storage.open_db(db_path) as conn:
        batches = conn.execute("SELECT status, error_message FROM monitoring_batches").fetchall()
        assert storage.get_summary(conn)["predictions"] == 0
    assert [row["status"] for row in batches] == ["failed"]
    assert "model crashed" in batches[0]["error_message"]


def test_command_line_run_labels_and_status(db_path, fake_csv, trained_model_path, capsys):
    common = ["--db", str(db_path)]
    assert main(common + [
        "run", "--production-data", str(fake_csv), "--model", str(trained_model_path),
        "--batch-size", "500", "--label-delay", "1",
    ]) == 0
    assert main(common + ["labels", "--production-data", str(fake_csv)]) == 0
    assert main(common + ["status"]) == 0

    output = capsys.readouterr().out
    assert "Predictions logged: 2,000" in output
    assert "Labels pending:     0" in output


def test_command_line_reports_missing_model(db_path, fake_csv, tmp_path):
    exit_code = main([
        "--db", str(db_path), "run",
        "--production-data", str(fake_csv), "--model", str(tmp_path / "missing.joblib"),
    ])
    assert exit_code == 1
