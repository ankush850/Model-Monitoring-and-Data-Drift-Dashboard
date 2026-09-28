"""Checks for the Docker setup: the database import script and the Docker/Compose configuration."""

import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from monitoring import storage

ROOT = Path(__file__).resolve().parents[1]

# Load docker/import_db.py by path (the folder name "docker" is not a Python package).
_spec = importlib.util.spec_from_file_location("import_db", ROOT / "docker" / "import_db.py")
import_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(import_db)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source_db(tmp_path):
    path = tmp_path / "host" / "monitoring.db"
    path.parent.mkdir()
    with storage.open_db(path) as conn:
        storage.start_batch(conn, "v1", "2026-01-01T00:00:00+00:00")
    return path


# --- database import script ---------------------------------------------------------------
def test_import_copies_database_and_leaves_source_unchanged(source_db, tmp_path):
    before = sha256(source_db)
    target = tmp_path / "volume" / "monitoring.db"

    counts = import_db.import_database(source_db, target)

    assert counts["monitoring_batches"] == 1
    assert sha256(source_db) == before
    with sqlite3.connect(target) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT model_version FROM monitoring_batches").fetchone()[0] == "v1"
    assert not list(target.parent.glob("*.importing"))


def test_import_never_overwrites_existing_database(source_db, tmp_path):
    target = tmp_path / "existing.db"
    target.write_bytes(b"existing data")

    with pytest.raises(FileExistsError):
        import_db.import_database(source_db, target)
    assert target.read_bytes() == b"existing data"
    assert import_db.main([str(source_db), str(target)]) == 1
    assert target.read_bytes() == b"existing data"


def test_import_reports_missing_source(tmp_path):
    target = tmp_path / "volume" / "monitoring.db"
    assert import_db.main([str(tmp_path / "nothing.db"), str(target)]) == 1
    assert not target.exists()


def test_import_rejects_a_file_that_is_not_a_database(tmp_path):
    source = tmp_path / "broken.db"
    source.write_bytes(b"not sqlite" * 100)
    target = tmp_path / "volume" / "monitoring.db"
    assert import_db.main([str(source), str(target)]) == 1
    assert not target.exists()
    assert not list(target.parent.glob("*.importing"))


# --- Dockerfile and .dockerignore -------------------------------------------------------------
# .dockerignore keeps these files out of the image, so these checks only run on the host.
needs_docker_files = pytest.mark.skipif(
    not (ROOT / "Dockerfile").is_file() or not (ROOT / ".dockerignore").is_file(),
    reason="Dockerfile/.dockerignore not present (running inside the image)",
)


@needs_docker_files
def test_dockerfile_runs_as_non_root_and_copies_only_code():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    instructions = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]

    assert instructions[0].startswith("FROM python:3.12")
    assert "USER app" in instructions
    copies = [line for line in instructions if line.startswith("COPY")]
    for forbidden in ("data", "db/", "models", ".venv", ".env"):
        assert not any(forbidden in line.split()[1] for line in copies), forbidden
    assert any("requirements.txt" in line for line in copies)
    assert "C:\\" not in text  # no Windows paths inside the Linux image


@needs_docker_files
def test_dockerignore_excludes_data_models_databases_and_secrets():
    entries = {line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines()
               if line.strip() and not line.startswith("#")}
    for required in (".venv/", ".git/", "data/", "db/", "models/", "**/*.db", "**/*.csv", "**/*.joblib", ".env"):
        assert required in entries, required


# --- docker-compose.yml (needs the Docker CLI) -------------------------------------------------
OVERRIDE_VARIABLES = ("MONITORING_VOLUME", "API_PORT", "DASHBOARD_PORT")


def compose_config() -> dict:
    """The resolved compose file with default settings (override variables in the shell are ignored)."""
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI not available")
    env = {k: v for k, v in os.environ.items() if k not in OVERRIDE_VARIABLES}
    result = subprocess.run(
        ["docker", "compose", "--profile", "tools", "config", "--format", "json"],
        cwd=ROOT, capture_output=True, text=True, timeout=60, env=env,
    )
    if result.returncode != 0:
        pytest.skip(f"docker compose config failed: {result.stderr.strip()[:200]}")
    return json.loads(result.stdout)


def mounts(service: dict) -> dict[str, dict]:
    return {m["target"]: m for m in service.get("volumes", [])}


def test_compose_services_share_one_database_volume():
    config = compose_config()
    services = config["services"]
    assert {"api", "dashboard", "monitor", "db-import"} <= set(services)

    for name in ("api", "dashboard", "monitor", "db-import"):
        db_mount = mounts(services[name])["/app/db"]
        assert db_mount["type"] == "volume" and db_mount["source"] == "monitoring-db", name
    assert config["volumes"]["monitoring-db"]["name"] == "ml-monitoring-db"


def test_compose_paths_are_linux_paths_and_consistent():
    services = compose_config()["services"]
    for name in ("api", "dashboard", "monitor"):
        env = services[name]["environment"]
        assert env["MONITORING_DB"] == "/app/db/monitoring.db"
        assert env["MONITORING_MODEL"] == "/app/models/fraud_model.joblib"
        assert all(value.startswith("/app/") for key, value in env.items() if key.startswith("MONITORING_"))


def test_compose_model_and_data_are_read_only():
    services = compose_config()["services"]
    for name in ("api", "dashboard", "monitor"):
        m = mounts(services[name])
        assert m["/app/models"]["read_only"] is True, name
        assert m["/app/data/processed"]["read_only"] is True, name
    assert mounts(services["db-import"])["/import"]["read_only"] is True


def test_compose_ports_healthchecks_and_restart_policies():
    services = compose_config()["services"]
    for name, port in (("api", 8000), ("dashboard", 8502)):
        service = services[name]
        (published,) = service["ports"]
        assert published["target"] == port
        assert published["host_ip"] == "127.0.0.1"  # not exposed to the network
        assert service["restart"] == "unless-stopped"
        assert service["healthcheck"]["test"][0] == "CMD"
    for name in ("monitor", "db-import"):
        assert services[name]["profiles"] == ["tools"]
