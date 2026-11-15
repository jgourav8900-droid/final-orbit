import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from backend import models  # noqa: F401
from backend.database import Base

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_migrations(database_url: str) -> None:
    environment = os.environ.copy()
    environment["DATABASE_URL"] = database_url
    subprocess.run(
        [sys.executable, "-m", "backend.migrations"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_migrations_create_a_fresh_database(tmp_path):
    database_path = tmp_path / "fresh.sqlite3"
    database_url = f"sqlite:///{database_path.as_posix()}"

    _run_migrations(database_url)

    migrated_engine = create_engine(database_url)
    try:
        tables = set(inspect(migrated_engine).get_table_names())
        assert "alembic_version" in tables
        assert set(Base.metadata.tables).issubset(tables)
    finally:
        migrated_engine.dispose()


def test_migrations_adopt_the_legacy_schema_without_losing_data(tmp_path):
    database_path = tmp_path / "legacy.sqlite3"
    database_url = f"sqlite:///{database_path.as_posix()}"
    legacy_engine = create_engine(database_url)
    legacy_tables = [
        table
        for name, table in Base.metadata.tables.items()
        if name != "ai_quiz_issued_attempts"
    ]
    try:
        Base.metadata.create_all(bind=legacy_engine, tables=legacy_tables)
        with legacy_engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO users (username, email, hashed_password, xp, level) "
                    "VALUES ('legacy-user', 'legacy@example.test', 'hash', 10, 2)"
                )
            )
    finally:
        legacy_engine.dispose()

    _run_migrations(database_url)

    migrated_engine = create_engine(database_url)
    try:
        inspector = inspect(migrated_engine)
        assert "ai_quiz_issued_attempts" in inspector.get_table_names()
        with migrated_engine.connect() as connection:
            username = connection.execute(
                text("SELECT username FROM users WHERE email = 'legacy@example.test'")
            ).scalar_one()
            revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        assert username == "legacy-user"
        assert revision == "0001_initial"
    finally:
        migrated_engine.dispose()


def test_migrations_refuse_to_stamp_an_unrecognized_database(tmp_path):
    database_path = tmp_path / "unrecognized.sqlite3"
    database_url = f"sqlite:///{database_path.as_posix()}"
    with create_engine(database_url).begin() as connection:
        connection.execute(text("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)"))

    with pytest.raises(subprocess.CalledProcessError):
        _run_migrations(database_url)

    migrated_engine = create_engine(database_url)
    try:
        tables = set(inspect(migrated_engine).get_table_names())
        assert tables == {"unrelated"}
    finally:
        migrated_engine.dispose()
