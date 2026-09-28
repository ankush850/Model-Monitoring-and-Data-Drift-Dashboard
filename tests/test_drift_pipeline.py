"""Tests for running drift detection over the simulator's batches and storing results."""

import pandas as pd
import pytest

from monitoring import config, storage
from monitoring.data import time_based_split
from monitoring.drift import main, run_drift_detection
from monitoring.simulate import run_simulation

BATCH_SIZE = 200  # fake production data has 800 rows, so 4 batches
N_FEATURES = len(config.DRIFT_FEATURES)


@pytest.fixture
def split_files(tmp_path, fake_df):
    """Reference and production CSVs made with the same split as training."""
    reference, production = time_based_split(fake_df, config.REFERENCE_FRACTION)
    reference_path = tmp_path / "reference.csv"
    production_path = tmp_path / "production.csv"
    reference.to_csv(reference_path, index=False)
    production.to_csv(production_path, index=False)
    return reference_path, production_path


@pytest.fixture
def simulated_db(db_path, split_files, trained_model_path):
    """A database where the simulator has already logged all production batches."""
    _, production_path = split_files
    run_simulation(
        db_path=db_path, production_path=production_path, model_path=trained_model_path,
        batch_size=BATCH_SIZE, label_delay_batches=0,
    )
    return db_path


def run_drift(db_path, split_files, **kwargs):
    reference_path, production_path = split_files
    return run_drift_detection(
        db_path=db_path, reference_path=reference_path, production_path=production_path, **kwargs
    )


def stored_results(db_path) -> pd.DataFrame:
    with storage.open_db(db_path) as conn:
        return storage.get_drift_results(conn)


def test_results_are_stored_for_every_batch_feature_and_test(simulated_db, split_files):
    outcome = run_drift(simulated_db, split_files)
    results = stored_results(simulated_db)

    assert outcome["processed"] == [1, 2, 3, 4]
    assert len(results) == 4 * N_FEATURES * 2
    assert set(results["test_name"]) == {"ks", "psi"}
    assert set(results["feature_name"]) == set(config.DRIFT_FEATURES)
    for column in ["batch_id", "feature_name", "test_name", "score", "threshold", "status", "computed_at"]:
        assert results[column].notna().all(), column
    assert (results.loc[results["test_name"] == "ks", "threshold"] == config.KS_ALPHA).all()
    assert (results.loc[results["test_name"] == "psi", "threshold"] == config.PSI_DRIFT_THRESHOLD).all()


def test_results_persist_after_reopening_database(simulated_db, split_files):
    run_drift(simulated_db, split_files)
    first = stored_results(simulated_db)
    second = stored_results(simulated_db)  # opens a fresh connection
    pd.testing.assert_frame_equal(first, second)
    assert len(second) > 0


def test_batches_match_the_simulator_batches(simulated_db, split_files):
    run_drift(simulated_db, split_files)
    results = stored_results(simulated_db)
    with storage.open_db(simulated_db) as conn:
        sizes = dict(conn.execute("SELECT batch_id, n_transactions FROM monitoring_batches").fetchall())

    for batch_id, n_current in results.groupby("batch_id")["n_current"].max().items():
        assert n_current == sizes[batch_id]


def test_repeated_processing_does_not_create_duplicates(simulated_db, split_files):
    run_drift(simulated_db, split_files)
    count_before = len(stored_results(simulated_db))

    second = run_drift(simulated_db, split_files)
    assert second["processed"] == []
    assert second["skipped"] == [1, 2, 3, 4]

    third = run_drift(simulated_db, split_files, recompute=True, batch_ids=[2])
    assert third["processed"] == [2]

    results = stored_results(simulated_db)
    assert len(results) == count_before
    assert not results.duplicated(["batch_id", "feature_name", "test_name"]).any()


def test_recompute_gives_identical_scores(simulated_db, split_files):
    run_drift(simulated_db, split_files)
    before = stored_results(simulated_db).drop(columns="computed_at")
    run_drift(simulated_db, split_files, recompute=True)
    after = stored_results(simulated_db).drop(columns="computed_at")
    pd.testing.assert_frame_equal(before, after)


def test_shifted_production_feature_is_flagged(tmp_path, db_path, split_files, trained_model_path):
    reference_path, production_path = split_files
    shifted = pd.read_csv(production_path)
    shifted["V3"] = shifted["V3"] + 3
    shifted_path = tmp_path / "shifted.csv"
    shifted.to_csv(shifted_path, index=False)

    run_simulation(db_path=db_path, production_path=shifted_path, model_path=trained_model_path,
                   batch_size=BATCH_SIZE)
    run_drift(db_path, (reference_path, shifted_path))

    v3 = stored_results(db_path).query("feature_name == 'V3'")
    assert (v3["status"] == "drift").all()


def test_drift_results_do_not_depend_on_labels(tmp_path, split_files, trained_model_path):
    reference_path, production_path = split_files
    flipped = pd.read_csv(production_path)
    flipped[config.TARGET_COLUMN] = 1 - flipped[config.TARGET_COLUMN]
    flipped_path = tmp_path / "flipped.csv"
    flipped.to_csv(flipped_path, index=False)

    outputs = []
    for name, path in [("a.db", production_path), ("b.db", flipped_path)]:
        db = tmp_path / name
        run_simulation(db_path=db, production_path=path, model_path=trained_model_path, batch_size=BATCH_SIZE)
        run_drift(db, (reference_path, path))
        outputs.append(stored_results(db).drop(columns="computed_at"))

    pd.testing.assert_frame_equal(outputs[0], outputs[1])


def test_failed_batches_are_not_checked(simulated_db, split_files):
    with storage.open_db(simulated_db) as conn:
        failed_id = storage.start_batch(conn, "v1", "2026-01-01T00:00:00+00:00")
        storage.fail_batch(conn, failed_id, "2026-01-01T00:00:00+00:00", "boom")

    outcome = run_drift(simulated_db, split_files)
    assert failed_id not in outcome["processed"]


def test_unknown_batch_id_is_rejected(simulated_db, split_files):
    with pytest.raises(ValueError, match="not completed batches"):
        run_drift(simulated_db, split_files, batch_ids=[999])


def test_empty_database_processes_nothing(db_path, split_files):
    assert run_drift(db_path, split_files) == {"processed": [], "skipped": []}


def test_command_line_run_and_report(simulated_db, split_files, capsys):
    reference_path, production_path = split_files
    common = ["--db", str(simulated_db)]
    assert main(common + ["run", "--reference-data", str(reference_path),
                          "--production-data", str(production_path)]) == 0
    assert main(common + ["report", "--batch-id", "1"]) == 0

    output = capsys.readouterr().out
    assert "Drift summary per batch" in output
    assert "Drift details for batch 1" in output
    assert "ks_p_value" in output


def test_command_line_reports_missing_reference(simulated_db, tmp_path):
    assert main(["--db", str(simulated_db), "run", "--reference-data", str(tmp_path / "missing.csv")]) == 1
