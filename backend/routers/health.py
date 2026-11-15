import os
import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.database import engine

router = APIRouter(prefix="/api/health", tags=["Health"])
logger = logging.getLogger(__name__)


@router.get("/live")
def liveness():
    return {"status": "alive"}


@router.get("")
@router.get("/ready")
def readiness():
    nasa_key = os.getenv("NASA_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            revision = connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one_or_none()
            if not revision:
                raise SQLAlchemyError("No Alembic migration is recorded.")
    except SQLAlchemyError as exc:
        logger.exception("Database readiness check failed")
        raise HTTPException(status_code=503, detail="Database is unavailable.") from exc

    return {
        "status": "healthy",
        "subsystems": {
            "api": "online",
            "database": f"{engine.dialect.name}_ready",
            "rag_engine": "ready",
            "ml_classifier": "ready",
            "nasa_integration": "active" if len(nasa_key) > 5 else "demo_mode",
            "gemini_integration": "active" if len(gemini_key) > 5 else "offline_synthesis"
        },
        "nasa_key_configured": bool(nasa_key and len(nasa_key) > 5),
        "gemini_key_configured": bool(gemini_key and len(gemini_key) > 5),
        "version": "2.4.0-production"
    }
