"""Report and status commands only read the database: they never create or change it."""

import shutil
import sqlite3

import pytest

from dashboard_fixtures import build_populated_project, file_hash
from monitoring import drift, performance, simulate, streaming

# (command main, arguments after --db); "{baseline}" is replaced by the test baseline file.
REPORT_COMMANDS = [
    (simulate.main, ["status"]),
    (drift.main, ["report"]),
    (drift.main, ["report", "--batch-id", "1"]),
    (streaming.main, ["report"]),
    (streaming.main, ["report", "--signal", "error_rate"]),
    (performance.main, ["report", "--baseline", "{baseline}"]),
    (performance.main, ["alerts"]),
]
IDS = [f"{main.__module__.split('.')[-1]} {' '.join(args[:3])}" for main, args in REPORT_COMMANDS]


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    return build_populated_project(tmp_path_factory.mktemp("cli_reports"))


def arguments(args, paths):
    return [str(paths["baseline"]) if a == "{baseline}" else a for a in args]


@pytest.mark.parametrize(("main", "args"), REPORT_COMMANDS, ids=IDS)
def test_report_on_missing_database_fails_and_creates_nothing(tmp_path, project, caplog, main, args):
    missing = tmp_path / "no_folder" / "monitoring.db"
    assert main(["--db", str(missing), *arguments(args, project)]) == 1
    assert "not found" in caplog.text
    assert not missing.parent.exists()


@pytest.mark.parametrize(("main", "args"), REPORT_COMMANDS, ids=IDS)
def test_report_does_not_change_an_existing_database(project, main, args):
    before = file_hash(project["db"])
    assert main(["--db", str(project["db"]), *arguments(args, project)]) == 0
    assert file_hash(project["db"]) == before


@pytest.mark.parametrize("notes", ["{not json", "5", '{"a": 1}'])
def test_performance_report_survives_malformed_notes(tmp_path, project, capsys, notes):
    db = tmp_path / "monitoring.db"
    shutil.copyfile(project["db"], db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE performance_windows SET notes = ?", (notes,))

    assert performance.main(["--db", str(db), "report", "--baseline", str(project["baseline"])]) == 0
    assert "notes could not be read" in capsys.readouterr().out
