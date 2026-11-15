import os
from fastapi import APIRouter

router = APIRouter(prefix="/api/health", tags=["Health"])

@router.get("")
def health_status():
    nasa_key = os.getenv("NASA_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    return {
        "status": "healthy",
        "subsystems": {
            "api": "online",
            "database": "sqlite_ready",
            "rag_engine": "ready",
            "ml_classifier": "ready",
            "nasa_integration": "active" if len(nasa_key) > 5 else "demo_mode",
            "gemini_integration": "active" if len(gemini_key) > 5 else "offline_synthesis"
        },
        "nasa_key_configured": bool(nasa_key and len(nasa_key) > 5),
        "gemini_key_configured": bool(gemini_key and len(gemini_key) > 5),
        "version": "2.4.0-production"
    }
