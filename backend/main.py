import os
import json
import asyncio
import secrets
import time
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from backend.routers import health, planets, ai, ml, nasa, auth, quiz, ai_quiz, agent
from backend.rag_engine import rag_engine
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

load_dotenv()

app = FastAPI(
    title="Orbit Overload",
    description="Space Research & Exploration Platform - FastAPI Backend & Educational Astrophysics Services",
    version="2.4.0"
)

CORS_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]
ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(",")
    if host.strip()
]
if "*" in CORS_ORIGINS:
    raise RuntimeError("CORS_ORIGINS must list explicit origins when credentials are enabled.")
METRICS_TOKEN = os.getenv("METRICS_TOKEN", "").strip()

HTTP_REQUESTS = Counter(
    "orbit_http_requests_total",
    "HTTP requests served by Orbit Overload.",
    ("method", "route", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "orbit_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "route"),
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    started_at = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response
    finally:
        route = request.scope.get("route")
        route_name = getattr(route, "path", "unmatched")
        HTTP_REQUESTS.labels(request.method, route_name, str(status_code)).inc()
        HTTP_REQUEST_DURATION.labels(request.method, route_name).observe(
            time.perf_counter() - started_at
        )


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics(request: Request):
    if not METRICS_TOKEN:
        raise HTTPException(status_code=503, detail="Metrics endpoint is disabled.")
    authorization = request.headers.get("authorization", "")
    if not secrets.compare_digest(authorization, f"Bearer {METRICS_TOKEN}"):
        raise HTTPException(status_code=403, detail="Forbidden.")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Register API routers
app.include_router(health.router)
app.include_router(planets.router)
app.include_router(ai.router)
app.include_router(ml.router)
app.include_router(nasa.router)
app.include_router(auth.router)
app.include_router(quiz.router)
app.include_router(ai_quiz.router)
app.include_router(agent.router)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES_DIR = os.path.join(BASE_DIR, "pages")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Unconditionally ensure and mount static directory
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Real-time WebSocket Assistant & Telemetry as documented in Section 5
@app.websocket("/ws/assistant")
async def websocket_assistant(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": "Connected to Orbit Overload Telemetry & Assistant Stream",
            "status": "ready"
        }))
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                query = msg.get("query", "")
                if query:
                    # Run RAG query
                    result = rag_engine.ask(query)
                    await websocket.send_text(json.dumps({
                        "type": "assistant_response",
                        "answer": result["answer"],
                        "source": result["source"],
                        "latency_ms": result["latency_ms"]
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "telemetry",
                        "orbits_active": 8,
                        "leo_debris_tracked": 1420,
                        "timestamp": asyncio.get_event_loop().time()
                    }))
            except Exception:
                await websocket.send_text(json.dumps({
                    "type": "telemetry_ping",
                    "status": "active"
                }))
    except WebSocketDisconnect:
        pass

# Route mappings specified in Section 4 of Project Dossier
@app.get("/", response_class=FileResponse)
def route_home():
    return FileResponse(os.path.join(PAGES_DIR, "index.html"))

@app.get("/solar", response_class=FileResponse)
def route_solar():
    return FileResponse(os.path.join(PAGES_DIR, "solar.html"))

@app.get("/ai-guide", response_class=FileResponse)
def route_ai_guide():
    return FileResponse(os.path.join(PAGES_DIR, "ai-guide.html"))

@app.get("/orbit-ai", response_class=FileResponse)
def route_orbit_ai():
    return FileResponse(os.path.join(PAGES_DIR, "ai-guide.html"))

@app.get("/ml-lab", response_class=FileResponse)
def route_ml_lab():
    return FileResponse(os.path.join(PAGES_DIR, "ml-lab.html"))

@app.get("/nasa-live", response_class=FileResponse)
def route_nasa_live():
    return FileResponse(os.path.join(PAGES_DIR, "nasa-live.html"))

@app.get("/planets", response_class=FileResponse)
def route_planets():
    return FileResponse(os.path.join(PAGES_DIR, "planets.html"))

@app.get("/quiz", response_class=FileResponse)
def route_quiz():
    return FileResponse(os.path.join(PAGES_DIR, "quiz.html"))

@app.get("/ai-quiz", response_class=FileResponse)
def route_ai_quiz():
    return FileResponse(os.path.join(PAGES_DIR, "ai-quiz.html"))

@app.get("/3d", response_class=FileResponse)
def route_3d():
    return FileResponse(os.path.join(PAGES_DIR, "3d.html"))

@app.get("/research", response_class=FileResponse)
def route_research():
    return FileResponse(os.path.join(PAGES_DIR, "research.html"))

# Legacy alias map ensuring zero broken links or 404s for historical paths
LEGACY_HTML_MAP = {
    "ml": "ml-lab.html",
    "nasa": "nasa-live.html",
    "3d-experience": "3d.html",
    "universe": "solar.html",
    "orbit-ai": "ai-guide.html",
    "mission-control": "index.html",
    "discovery": "quiz.html",
}

# Catch-all route returning genuine 404 for missing HTML files
@app.get("/{filename}.html", response_class=FileResponse)
def route_html_files(filename: str):
    if filename in LEGACY_HTML_MAP:
        alias_target = os.path.join(PAGES_DIR, LEGACY_HTML_MAP[filename])
        if os.path.exists(alias_target):
            return FileResponse(alias_target)

    page_target = os.path.join(PAGES_DIR, f"{filename}.html")
    if os.path.exists(page_target):
        return FileResponse(page_target)
    root_target = os.path.join(BASE_DIR, f"{filename}.html")
    if os.path.exists(root_target):
        return FileResponse(root_target)
    raise HTTPException(status_code=404, detail=f"Page '{filename}.html' not found")
