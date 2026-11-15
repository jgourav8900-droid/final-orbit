import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

import backend.main as main
from backend.routers import health
from backend.backup import backup_sqlite_database
from backend.main import app


client = TestClient(app)


def test_health_endpoints_report_readiness_and_liveness():
    assert client.get("/api/health/live").json() == {"status": "alive"}
    response = client.get("/api/health/ready")
    assert response.status_code == 200
    assert response.json()["subsystems"]["database"] == "sqlite_ready"


def test_readiness_fails_when_database_is_unavailable(monkeypatch):
    class UnavailableConnection:
        def __enter__(self):
            raise SQLAlchemyError("database down")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(health.engine, "connect", lambda: UnavailableConnection())

    assert client.get("/api/health/live").status_code == 200
    response = client.get("/api/health/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "Database is unavailable."


def test_readiness_fails_before_migrations(monkeypatch):
    unmigrated_engine = create_engine("sqlite:///:memory:")
    monkeypatch.setattr(health, "engine", unmigrated_engine)
    try:
        response = client.get("/api/health/ready")
        assert response.status_code == 503
    finally:
        unmigrated_engine.dispose()


def test_untrusted_host_is_rejected():
    response = client.get("/", headers={"host": "attacker.example"})
    assert response.status_code == 400


def test_metrics_endpoint_requires_a_bearer_token(monkeypatch):
    monkeypatch.setattr(main, "METRICS_TOKEN", "")
    assert client.get("/metrics").status_code == 503

    monkeypatch.setattr(main, "METRICS_TOKEN", "test-metrics-token")
    assert client.get("/metrics").status_code == 403
    response = client.get(
        "/metrics",
        headers={"Authorization": "Bearer test-metrics-token"},
    )
    assert response.status_code == 200
    assert "orbit_http_requests_total" in response.text


def test_sqlite_backup_is_a_verified_snapshot(tmp_path):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "snapshot.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        connection.execute("INSERT INTO sample (value) VALUES ('preserved')")

    backup_path = backup_sqlite_database(
        destination,
        database_url=f"sqlite:///{source.as_posix()}",
    )

    assert backup_path == destination.resolve()
    with sqlite3.connect(backup_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT value FROM sample").fetchone()[0] == "preserved"


def test_sqlite_backup_rejects_non_sqlite_urls(tmp_path):
    with pytest.raises(ValueError, match="supports SQLite only"):
        backup_sqlite_database(
            tmp_path / "backup.dump",
            database_url="postgresql+psycopg://user:password@localhost/orbit",
        )
