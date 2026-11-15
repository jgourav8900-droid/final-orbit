# Production operations

## Database migrations

The app now uses Alembic migrations instead of creating tables during import. The supported upgrade command is:

```powershell
python -m backend.migrations
```

On a new, empty database this applies the migration history. On the existing pre-Alembic schema it first verifies that all legacy tables are present, preserves their data, creates tables introduced since that release, and records the baseline. An incomplete or unfamiliar unversioned schema fails without being stamped. Back up the database before the first upgrade. `python run_server.py` runs the same upgrade automatically before serving traffic; run this on one instance only during a rolling deployment, then start the other instances.

To create a new migration after changing SQLAlchemy models:

```powershell
python -m alembic revision --autogenerate -m "describe the schema change"
```

Review and edit generated revisions before applying them. Apply with `python -m backend.migrations`.

## Backups and restore

For SQLite, create a transactionally consistent snapshot and run an integrity check:

```powershell
python -m backend.backup
```

The default output is `backups/orbit-<UTC timestamp>.sqlite3`; pass a destination path to choose another location. Backups are ignored by Git and Docker build context. Schedule this command daily using Windows Task Scheduler or an equivalent scheduler. Copy backups to encrypted off-host storage, restrict access, and periodically rehearse restoring one.

To restore a SQLite snapshot, stop the app first and copy the chosen snapshot over the file named by `DATABASE_URL` in `.env`, then restart the app. Keep an additional copy of the current database before replacing it.

Compose uses PostgreSQL with a persistent named volume. Create an on-demand compressed custom-format dump with 14-day local retention:

```powershell
docker compose --profile backup run --rm db-backup
```

Dumps are written to `backups/postgres`. To restore a dump, first stop app writes and take a separate copy of the current database. Then restore into the configured database (this replaces matching objects):

```powershell
docker compose --profile backup run --rm --entrypoint pg_restore db-backup --clean --if-exists --no-owner /backups/orbit-<timestamp>.dump
```

Backups stored on the same host are not disaster recovery. Replicate them to encrypted off-host storage and test restores.

## Docker deployment

Copy `.env.example` to `.env`, then set a unique `JWT_SECRET` and URL-safe `POSTGRES_PASSWORD` (generate with `python -c "import secrets; print(secrets.token_urlsafe(48))"`). Set `METRICS_TOKEN` if enabling monitoring; provider API keys are optional. Never commit `.env` or backup files.

Start the app and PostgreSQL:

```powershell
docker compose up --build -d
```

The app port binds to `127.0.0.1:8000` by default, PostgreSQL is not published to the host, and the app runs as a non-root user with a read-only container filesystem. The model-artifact volume preserves trained model output. Put a TLS-terminating reverse proxy in front of the app, set `ALLOWED_HOSTS` to the actual host names and `CORS_ORIGINS` to the browser origins, and do not proxy `/metrics` publicly. To bind the app directly to another interface, set `APP_BIND_ADDRESS` intentionally and configure a firewall.

For local development, `python run_server.py` uses SQLite and defaults to loopback. `HOST=0.0.0.0` is appropriate inside the container, not as a default for a development workstation.

## Health and monitoring

- `GET /api/health/live` reports process liveness and does not depend on the database.
- `GET /api/health/ready` and the compatibility route `GET /api/health` check the database and return `503` if it is unavailable.
- `/metrics` requires `Authorization: Bearer <METRICS_TOKEN>` and returns `503` when no token is configured.

To run Prometheus locally, set `METRICS_TOKEN` in `.env`, put the same value in the ignored `monitoring/metrics-token` file (copy `monitoring/metrics-token.example` as a reminder), and start the monitoring profile:

```powershell
docker compose --profile monitoring up -d
```

Prometheus is available at `http://localhost:9090` and is bound to loopback. Its scrape configuration and token file are mounted read-only.

## Browser smoke check

After startup, open `http://localhost:8000/` and check the home page, navigation to Solar System, AI Guide, ML Lab, NASA Live, Planets, Quiz, and Research, plus browser developer-console errors. Exercise account creation and quiz submission with a test account. NASA and Gemini pages may use their documented fallback modes when provider keys are not configured; automated tests mock these external services and do not replace a real-provider QA pass.
