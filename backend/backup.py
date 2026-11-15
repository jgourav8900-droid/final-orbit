import argparse
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.engine import make_url

from backend.database import DATABASE_URL


def backup_sqlite_database(
    destination: str | Path | None = None,
    database_url: str | None = None,
) -> Path:
    """Create and verify a consistent SQLite snapshot using SQLite's backup API."""
    url = make_url(database_url or DATABASE_URL)
    if not url.drivername.startswith("sqlite"):
        raise ValueError("This backup command supports SQLite only; use the Compose PostgreSQL backup service.")
    if not url.database or url.database == ":memory:":
        raise ValueError("A file-backed SQLite database is required.")

    source_path = Path(url.database).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Database file does not exist: {source_path}")

    if destination is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination_path = Path("backups") / f"orbit-{timestamp}.sqlite3"
    else:
        destination_path = Path(destination)
    destination_path = destination_path.resolve()
    if destination_path == source_path:
        raise ValueError("Backup destination must be different from the source database.")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(destination_path.name + ".tmp")
    try:
        with closing(sqlite3.connect(source_path)) as source, closing(
            sqlite3.connect(temporary_path)
        ) as backup:
            source.backup(backup)
            result = backup.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                raise RuntimeError("SQLite backup integrity check failed.")
        os.replace(temporary_path, destination_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a verified SQLite backup.")
    parser.add_argument(
        "destination",
        nargs="?",
        help="Backup output path (default: backups/orbit-<UTC timestamp>.sqlite3)",
    )
    args = parser.parse_args()
    print(backup_sqlite_database(args.destination))


if __name__ == "__main__":
    main()
