"""Render every dashboard page with Streamlit's AppTest and check for errors."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard_fixtures import build_populated_project, deliver_remaining_labels, file_hash

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
PAGES = ["Overview", "Data drift", "Streaming monitoring", "Model performance", "Alerts"]


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    return build_populated_project(tmp_path_factory.mktemp("dashboard_app"))


def use_project(monkeypatch, paths):
    monkeypatch.setenv("MONITORING_DB", str(paths["db"]))
    monkeypatch.setenv("MONITORING_BASELINE", str(paths["baseline"]))
    monkeypatch.setenv("MONITORING_REFERENCE_DATA", str(paths["reference"]))
    monkeypatch.setenv("MONITORING_PRODUCTION_DATA", str(paths["production"]))


def render(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    if page != "Overview":
        at.radio(key="page").set_value(page).run()
    return at


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [w.value for w in at.warning] + [s.value for s in at.success] + [t.value for t in at.title]
    parts += [h.value for h in at.subheader]
    return "\n".join(str(p) for p in parts)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(monkeypatch, project, page):
    use_project(monkeypatch, project)
    at = render(page)
    assert not at.exception, at.exception
    assert not at.error
    assert at.title[0].value == page


def test_data_drift_page_draws_distribution_chart(monkeypatch, project):
    use_project(monkeypatch, project)
    at = render("Data drift")
    distribution_tab = at.tabs[3]
    assert not distribution_tab.info, [i.value for i in distribution_tab.info]
    # heatmap + PSI trend + KS trend + distribution histogram
    assert len(at.get("plotly_chart")) == 4
    assert any("KS statistic **" in m.value for m in distribution_tab.markdown)


def test_overview_shows_stored_counts(monkeypatch, project):
    use_project(monkeypatch, project)
    at = render("Overview")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Transactions monitored"] == "2,000"
    assert metrics["Monitoring batches"] == "5"
    assert int(metrics["Performance alerts"]) > 0


def test_streaming_page_explains_waiting_labels(monkeypatch, project):
    use_project(monkeypatch, project)
    text = all_text(render("Streaming monitoring"))
    assert "waiting" in text and "true labels" in text


def test_performance_page_shows_baseline_and_windows(monkeypatch, project):
    use_project(monkeypatch, project)
    at = render("Model performance")
    labels = [m.label for m in at.metric]
    assert labels == ["Precision", "Recall", "F1-score", "PR-AUC", "False positive rate"]
    assert "Baseline = validation-set results" in all_text(at)


def test_alerts_page_lists_stored_alerts(monkeypatch, project):
    use_project(monkeypatch, project)
    at = render("Alerts")
    assert len(at.dataframe) == 1
    assert len(at.dataframe[0].value) > 0


def test_viewing_pages_does_not_change_the_database(monkeypatch, project):
    use_project(monkeypatch, project)
    before = file_hash(project["db"])
    for page in PAGES:
        render(page)
    assert file_hash(project["db"]) == before


def test_missing_database_shows_setup_instructions(monkeypatch, tmp_path):
    missing = tmp_path / "missing.db"
    monkeypatch.setenv("MONITORING_DB", str(missing))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert "No monitoring database found" in at.warning[0].value
    assert "python -m monitoring.simulate run" in at.code[0].value
    assert not missing.exists()


def test_empty_database_pages_show_guidance(monkeypatch, tmp_path):
    from monitoring import storage

    db = tmp_path / "empty.db"
    with storage.open_db(db):
        pass
    monkeypatch.setenv("MONITORING_DB", str(db))
    for page in PAGES:
        at = render(page)
        assert not at.exception, (page, at.exception)


def test_outdated_overview_data_shows_restart_message_instead_of_traceback(monkeypatch, tmp_path):
    """Regression: a dashboard started before Phase 7 kept the old dashboard_data module in memory,
    so the overview dictionary lacked 'simulator_batches' and the page crashed with a KeyError."""
    from monitoring import dashboard_data, storage

    db = tmp_path / "stale.db"
    with storage.open_db(db):
        pass
    monkeypatch.setenv("MONITORING_DB", str(db))

    real_load_overview = dashboard_data.load_overview

    def pre_phase7_load_overview(db_path):
        overview = real_load_overview(db_path)
        for key in ("simulator_batches", "api_batches", "api_predictions"):
            overview.pop(key)
        return overview

    monkeypatch.setattr(dashboard_data, "load_overview", pre_phase7_load_overview)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()

    assert not at.exception, at.exception
    (error,) = at.error
    assert "simulator_batches" in error.value
    assert "restart" in error.value.lower()


def test_error_rate_appears_after_labels_arrive(monkeypatch, tmp_path):
    from monitoring.streaming import run_streaming_detection

    paths = build_populated_project(tmp_path)
    deliver_remaining_labels(paths)
    run_streaming_detection(db_path=paths["db"])
    use_project(monkeypatch, paths)
    text = all_text(render("Streaming monitoring"))
    assert "every logged batch is fully labeled" in text


@pytest.mark.parametrize("notes", ["{not json", "null", "5", '"text"'])
def test_performance_page_survives_damaged_notes(monkeypatch, tmp_path, project, notes):
    import shutil
    import sqlite3

    db = tmp_path / "monitoring.db"
    shutil.copyfile(project["db"], db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE performance_windows SET notes = ?", (notes,))
    use_project(monkeypatch, {**project, "db": db})
    at = render("Model performance")
    assert not at.exception, at.exception
    assert not at.error
    assert "- t\n- e\n- x\n- t" not in all_text(at)  # a string note is one note, not one per letter
