"""Copy an existing monitoring database into the Docker volume - without ever overwriting.

Used by the `db-import` service in docker-compose.yml:
    docker compose --profile tools run --rm db-import

    source  /import/monitoring.db   (the host's db folder, mounted READ-ONLY)
    target  /app/db/monitoring.db   (the persistent Docker volume)

Safety rules
  * The source is opened read-only and is never changed.
  * If the target already exists, nothing happens (exit code 1).
  * The copy is made with SQLite's backup API (a consistent snapshot), written to
    a temporary file, checked with PRAGMA integrity_check, and only then renamed
    into place. A failed import leaves no partial database behind.
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

TABLES_TO_REPORT = ("monitoring_batches", "predictions", "actual_labels", "drift_results",
                    "stream_drift_results", "performance_windows", "alerts")


def row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    existing = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES_TO_REPORT if t in existing}


def import_database(source: Path, target: Path) -> dict[str, int]:
    """Copy source to target. Raises FileExistsError / FileNotFoundError / RuntimeError on problems."""
    if target.exists():
        raise FileExistsError(f"{target} already exists - refusing to overwrite it.")
    if not source.is_file():
        raise FileNotFoundError(f"No database to import at {source}.")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".importing")
    temporary.unlink(missing_ok=True)

    # immutable=1: the read-only mount cannot hold lock files; stop host writers before importing.
    src = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        source_counts = row_counts(src)
        dst = sqlite3.connect(temporary)
        try:
            src.backup(dst)
            check = dst.execute("PRAGMA integrity_check").fetchone()[0]
            copied_counts = row_counts(dst)
        finally:
            dst.close()
    finally:
        src.close()

    if check != "ok" or copied_counts != source_counts:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Copy failed verification (integrity: {check}).")

    if target.exists():  # created by someone else while we were copying
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"{target} appeared during the import - left untouched.")
    os.replace(temporary, target)
    return copied_counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a monitoring database without overwriting.")
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args(argv)

    try:
        counts = import_database(args.source, args.target)
    except (FileExistsError, FileNotFoundError, RuntimeError, sqlite3.Error) as exc:
        print(f"Import not done: {exc}", file=sys.stderr)
        return 1

    print(f"Imported {args.source} -> {args.target}")
    for table, count in counts.items():
        print(f"  {table:<22} {count:>8,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
