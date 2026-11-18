# Orbit Overload

Orbit Overload is an educational space-science and astronomy platform. It combines an interactive browser interface with a Python/FastAPI backend, NASA data feeds, optional Gemini-powered assistance, quizzes, and machine-learning demonstrations.

- **Repository:** [jgourav8900-droid/final-orbit](https://github.com/jgourav8900-droid/final-orbit) (private)
- **Local URL:** <http://127.0.0.1:8000/>
- **API documentation:** <http://127.0.0.1:8000/docs> when the local server is running

## Features

- **Space exploration:** Solar System views, planet information, and an interactive 3D experience.
- **NASA data:** Astronomy Picture of the Day (APOD), near-Earth objects, Mars rover imagery, EPIC imagery, and DONKI solar-flare information.
- **AI assistance:** A retrieval-augmented astronomy guide and research/agent endpoints, with Gemini as an optional external provider and a local corpus-based fallback.
- **Quizzes:** Standard quizzes and optional AI-generated quizzes, with submission, history, and leaderboard features.
- **Machine-learning lab:** Model status, training, prediction, classification, explanations, simulation, and anomaly-related endpoints.
- **Accounts:** Registration, login, token refresh, and user profile endpoints.
- **Operations:** Alembic database migrations, health/readiness endpoints, SQLite snapshots, PostgreSQL dump support, and protected Prometheus metrics.
- **Deployment checks:** GitHub Actions starts the PostgreSQL-backed Compose stack and checks readiness and metrics authorization.

## Technology

- Python, FastAPI, Uvicorn, Pydantic, and SQLAlchemy
- Alembic for database migrations
- SQLite for local development; PostgreSQL 16 for the Docker Compose deployment
- HTML, CSS, JavaScript, and Three.js-based browser experiences
- NASA APIs and the Google Gemini API (both keys are optional; provider access depends on valid credentials)
- Prometheus client instrumentation and optional Prometheus service

## Run locally (Windows)

Use Python 3.11. From the project directory, create and activate a virtual environment, install dependencies, and start the application:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` before starting:

1. Set `JWT_SECRET` to a unique, random value of at least 32 characters. Generate one with:
   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```
2. Optionally add provider credentials to `GEMINI_API_KEY` and `NASA_API_KEY`. Gemini keys are available from [Google AI Studio](https://aistudio.google.com/apikey); NASA offers API keys at [api.nasa.gov](https://api.nasa.gov/). Keep keys private and never paste them into source code, issues, or chat.
3. Keep the local `DATABASE_URL` on SQLite unless you intentionally configure another database.

Start the app:

```powershell
python run_server.py
```

The startup command applies database migrations before serving. By default, the server binds to `127.0.0.1:8000`. Open <http://127.0.0.1:8000/>. Stop it with **Ctrl+C**.

If PowerShell prevents virtual-environment activation, use the Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_server.py
```

The repository also includes `START_LOCALHOST.bat` as a Windows launcher.

## Configuration

`.env.example` documents the supported settings. Copy it to `.env` for local work; `.env` is ignored by Git.

| Variable | Purpose |
| --- | --- |
| `HOST`, `PORT` | Local server bind address and port; defaults to `127.0.0.1` and `8000`. |
| `DEBUG` | Debug-mode setting for local development; keep it disabled in production. |
| `DATABASE_URL` | SQLAlchemy database URL. Local development defaults to SQLite. |
| `JWT_SECRET` | Required, unique secret used to sign authentication tokens. Use a random value of at least 32 characters. |
| `JWT_EXPIRATION_SECONDS` | Authentication token lifetime; defaults to 3600 seconds. |
| `NASA_API_KEY` | Optional NASA API credential. Provider rate limits and data availability still apply. |
| `GEMINI_API_KEY` | Optional Gemini credential used for external generative responses and AI quiz generation. A configured value does not guarantee that Google accepts the key or that a model is available. |
| `ML_TRAINING_TOKEN` | Optional token used to protect model-training operations. |
| `METRICS_TOKEN` | Enables `/metrics` when set; send it as a bearer token. |
| `CORS_ORIGINS` | Comma-separated browser origins permitted by CORS. Use explicit origins, not `*`. |
| `ALLOWED_HOSTS` | Comma-separated hostnames accepted by the application. Set these to the actual hostnames when deployed. |
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | PostgreSQL settings for Docker Compose. Use a strong URL-safe password. |
| `APP_BIND_ADDRESS`, `APP_PORT` | Host interface and port used to publish the Compose app port. Compose defaults to loopback. |
| `PROMETHEUS_PORT` | Host port for the optional Prometheus UI, bound to loopback. |

Never commit `.env`, API keys, database files, backups, or metrics-token files. Rotate any credential that has been exposed.

## Docker Compose deployment

Install Docker with the Compose plugin and ensure its engine is running. Then copy `.env.example` to `.env` and set at least a strong, unique `JWT_SECRET` and URL-safe `POSTGRES_PASSWORD`. Start the application and PostgreSQL:

```powershell
docker compose up --build -d
docker compose ps
```

The app is published on `127.0.0.1:8000` by default; PostgreSQL is not exposed on the host. To stop the services without deleting their data:

```powershell
docker compose down
```

Do not use `docker compose down --volumes` for normal shutdown: it deletes persistent database and model/monitoring data. The CI workflow uses disposable volumes and removes them after its integration check.

### Production deployment checklist

- Set unique production secrets and valid provider keys in the deployment environment; do not reuse development credentials.
- Use a TLS-terminating reverse proxy, set `ALLOWED_HOSTS` and `CORS_ORIGINS` to the actual host/origin, and configure firewall rules.
- Keep PostgreSQL private and do not expose `/metrics` publicly.
- Configure and verify scheduled backups, encrypted off-host replication, access controls, retention, and restore drills.
- Run the migration command on one application instance before bringing up other instances during a rolling deployment.
- Configure external uptime/alerting and review logs and Prometheus metrics.

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed migration, backup/restore, Docker, monitoring, and browser-smoke instructions.

## Database migrations and backups

Apply database migrations explicitly with:

```powershell
python -m backend.migrations
```

`python run_server.py` also applies migrations on startup. When changing SQLAlchemy models, create a revision and review it before applying:

```powershell
python -m alembic revision --autogenerate -m "describe the schema change"
python -m backend.migrations
```

For local SQLite, create a consistent, integrity-checked snapshot:

```powershell
python -m backend.backup
```

For the Compose PostgreSQL deployment, create an on-demand dump:

```powershell
docker compose --profile backup run --rm db-backup
```

These commands create local backups; they do not configure a scheduler or off-site storage automatically. Follow [DEPLOYMENT.md](DEPLOYMENT.md) for restore procedures and operational precautions.

## API overview

FastAPI's interactive OpenAPI UI is available at `/docs`; the generated schema is at `/openapi.json`.

| Prefix | Purpose |
| --- | --- |
| `/api/health` | Liveness and database/migration readiness. |
| `/api/planets` | Planet catalogue. |
| `/api/nasa` | NASA data and alert preferences. |
| `/api/ai` | AI guide, research, reports, conversation history, and RAG search. |
| `/api/agent` | Astronomy agent, paper search, explanations, and comparisons. |
| `/api/ai-quiz` | AI quiz generation, submission, history, and leaderboard. |
| `/api/quiz` | Standard quiz and submission endpoints. |
| `/api/auth` | Registration, login, refresh, and current-user endpoints. |
| `/api/ml` | ML status, training, prediction, explanation, and related operations. |
| `/metrics` | Prometheus metrics; requires `Authorization: Bearer <METRICS_TOKEN>`. |
| `/ws/assistant` | WebSocket assistant and telemetry stream. |

Useful checks after startup:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health/live
Invoke-RestMethod http://127.0.0.1:8000/api/health/ready
Invoke-RestMethod http://127.0.0.1:8000/api/nasa/status
Invoke-RestMethod http://127.0.0.1:8000/api/ai/models
```

Health reports whether a provider key is configured; it does not validate the key with the provider. A live provider request is needed to verify credentials and account access.

## Monitoring

Set `METRICS_TOKEN` in `.env`, then copy `monitoring/metrics-token.example` to `monitoring/metrics-token` and put the same token in that ignored local file. Start Prometheus:

```powershell
docker compose --profile monitoring up -d
```

Prometheus is available at <http://127.0.0.1:9090>. The metrics endpoint is disabled when no token is configured and rejects requests without the matching bearer token.

## Tests and CI

Install the requirements, then run the test suite from the repository root:

```powershell
python -m pytest
```

Tests cover API behavior, migrations, and production operations. Tests mock external providers; they do not prove that NASA or Gemini credentials work.

The GitHub Actions workflow runs on relevant pushes and pull requests. It builds and starts the Compose app with PostgreSQL, checks database readiness, confirms metrics are blocked without a token and available with the token, and then removes its disposable stack. The workflow does not deploy the application.

## Troubleshooting

- **The app exits during startup:** Check `.env`, ensure `JWT_SECRET` is present and at least 32 characters, then review the terminal error.
- **Port 8000 is occupied:** Change `PORT` for local execution or `APP_PORT` for Compose.
- **Health is not ready:** Check the database URL, migration output, and (for Docker) `docker compose logs`.
- **Gemini is configured but AI uses a fallback:** Configuration status only checks that a value exists. Verify the key in Google AI Studio, API/project access, restrictions, billing/quota where applicable, and model availability. Do not share the secret when asking for help.
- **NASA data is unavailable:** Check `NASA_API_KEY`, provider status, rate limits, and network access. Some endpoints can use a documented fallback when provider data is unavailable.
- **Docker commands cannot connect:** Start Docker Desktop/the Docker engine and retry `docker compose ps`.

## Project layout

```text
backend/       FastAPI app, routers, database, migrations, and provider integrations
migrations/    Alembic migration environment and revisions
pages/         Browser page templates served by FastAPI
static/        CSS, JavaScript, images, and other static assets
tests/         Automated API and operations tests
monitoring/    Prometheus configuration and token example
.github/       GitHub Actions workflows
DEPLOYMENT.md  Detailed database and production operations guide
```

## Security and project status

This project is not automatically production-deployed by the CI workflow. Public deployment still requires environment-specific secrets, TLS and host/origin configuration, scheduled encrypted off-site backups, alerting, and production browser/provider validation. Never include secret values in issues, logs, screenshots, commits, or support requests.
