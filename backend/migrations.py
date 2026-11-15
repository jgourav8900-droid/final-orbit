from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from backend import models  # noqa: F401
from backend.database import DATABASE_URL, Base, engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_REVISION = "0001_initial"
LEGACY_TABLES = set(Base.metadata.tables) - {"ai_quiz_issued_attempts"}


def _alembic_config() -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))
    return config


def upgrade_database() -> None:
    """Upgrade an empty database or safely adopt the pre-Alembic app schema."""
    config = _alembic_config()
    existing_tables = set(inspect(engine).get_table_names())

    if "alembic_version" not in existing_tables and existing_tables:
        missing_tables = LEGACY_TABLES - existing_tables
        if missing_tables:
            missing = ", ".join(sorted(missing_tables))
            raise RuntimeError(
                "The existing database has an incomplete unversioned schema "
                f"(missing: {missing}); restore a backup or reconcile it before migrating."
            )
        for table_name in sorted(LEGACY_TABLES):
            expected_columns = {column.name for column in Base.metadata.tables[table_name].columns}
            actual_columns = {
                column["name"] for column in inspect(engine).get_columns(table_name)
            }
            missing_columns = expected_columns - actual_columns
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise RuntimeError(
                    f"The existing unversioned table {table_name!r} is missing "
                    f"columns ({missing}); no schema changes were applied."
                )

        # Existing releases created their tables with SQLAlchemy. Preserve those
        # tables, add any model tables introduced since, and mark the baseline.
        Base.metadata.create_all(bind=engine)
        command.stamp(config, BASELINE_REVISION)

    command.upgrade(config, "head")


if __name__ == "__main__":
    upgrade_database()
    print("Database schema is up to date.")
