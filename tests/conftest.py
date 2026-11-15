import os
import secrets
import tempfile
from pathlib import Path

import pytest


_TEST_DIRECTORY = tempfile.TemporaryDirectory(prefix="orbit-overload-tests-")
_TEST_DATABASE = Path(_TEST_DIRECTORY.name, "test.sqlite3")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DATABASE.as_posix()}"
os.environ["JWT_SECRET"] = secrets.token_urlsafe(48)
os.environ["NASA_API_KEY"] = "test-nasa-api-key"
os.environ["GEMINI_API_KEY"] = "test-gemini-api-key"
os.environ["ML_TRAINING_TOKEN"] = ""
os.environ["METRICS_TOKEN"] = ""
os.environ["CORS_ORIGINS"] = "http://localhost:8000,http://127.0.0.1:8000"
os.environ["ALLOWED_HOSTS"] = "localhost,127.0.0.1,testserver"

from backend.migrations import upgrade_database

upgrade_database()


@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch):
    from backend.routers import agent, nasa
    from backend.rag_engine import rag_engine

    nasa.NASA_CACHE.clear()
    agent._PAPER_CACHE.clear()

    def fake_nasa_fetch(url, timeout=14):
        if "/planetary/apod" in url and "count=" in url:
            data = [{
                "title": "Test Astronomy Image",
                "url": "https://example.test/apod.jpg",
                "date": "2024-01-01",
                "explanation": "A test image.",
                "media_type": "image",
            }]
        elif "/planetary/apod" in url:
            data = {
                "title": "Test Astronomy Image",
                "url": "https://example.test/apod.jpg",
                "date": "2024-01-01",
                "explanation": "A test image.",
                "media_type": "image",
            }
        elif "/neo/rest/v1/feed" in url:
            data = {"near_earth_objects": {}}
        elif "/mars-photos/" in url:
            data = {"rover": "curiosity", "sol": 1000, "photos": [{"id": 1}]}
        elif "/EPIC/api/natural" in url:
            data = [{
                "date": "2024-01-01 00:00:00",
                "image": "test-image",
                "identifier": "test-identifier",
                "caption": "Test Earth image",
                "centroid_coordinates": {},
            }]
        elif "/DONKI/" in url:
            data = []
        else:
            data = {}
        return data, 200, None

    def fake_ask(query, **kwargs):
        answer = f"Deterministic test astronomy answer for: {query}"
        return {
            "answer": answer,
            "source": "test-rag",
            "model": "test-model",
            "latency_ms": 1,
            "sources": ["test-corpus"],
            "suggested_followups": ["What should I learn next?"],
        }

    def fake_research(query):
        answer = f"Deterministic test research summary for: {query}"
        return {"query": query, "summary": answer, "answer": answer, "sources": ["test-corpus"]}

    def fake_generate_report(topic, context_body=None):
        return {
            "topic": topic,
            "dossier": f"Deterministic test research dossier for {topic}.",
            "citations": [],
            "dois": [],
        }

    def fake_paper_search(query, max_results=8):
        return [{
            "title": f"Test paper about {query}",
            "authors": ["Test Author"],
            "abstract": "A deterministic paper abstract.",
            "arxiv_id": "0000.00000",
            "url": "https://example.test/paper",
            "doi_url": None,
            "published": "2024-01-01",
            "categories": ["astro-ph"],
            "type": "preprint",
            "source": "arXiv",
        }][:max_results]

    monkeypatch.setattr(nasa, "fetch_nasa_api", fake_nasa_fetch)
    monkeypatch.setattr(rag_engine, "ask", fake_ask)
    monkeypatch.setattr(rag_engine, "research", fake_research)
    monkeypatch.setattr(rag_engine, "generate_report", fake_generate_report)
    monkeypatch.setattr(agent, "_search_arxiv", fake_paper_search)
    monkeypatch.setattr(agent, "_search_openalex", fake_paper_search)


def pytest_unconfigure(config):
    from backend.database import engine

    engine.dispose()
    _TEST_DIRECTORY.cleanup()
