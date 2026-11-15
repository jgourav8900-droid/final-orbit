import os
import json
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Conversation
from backend.routers.auth import get_current_user_optional, get_current_user
from backend.rag_engine import rag_engine
from backend.routers.planets import PLANET_CATALOGUE
from backend.routers.nasa import NASA_CACHE

router = APIRouter(prefix="/api/ai", tags=["AI & RAG"])

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    context_planet: Optional[str] = None
    mode: Optional[str] = "standard"  # "standard" or "research"
    extra_context: Optional[str] = None

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)

class ReportRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200)
    context_body: Optional[str] = None

@router.post("/ask")
def ai_ask(
    req: AskRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Intelligent space research assistant query.
    Enriched with RAG, celestial context, NASA live status, and conversational persistence.
    """
    # Context enrichment if planet is selected
    planet_details = None
    if req.context_planet:
        for p in PLANET_CATALOGUE:
            if p["name"].lower() == req.context_planet.lower():
                planet_details = f"Body: {p['name']}, Type: {p['type']}, Diameter: {p['diameter_km']:,} km, Distance from Sun: {p['distance_from_sun_au']} AU, Orbit: {p['orbital_period_days']} days, Moons: {p['moons']}"
                break
    
    full_extra = req.extra_context or ""
    if planet_details:
        full_extra = f"{full_extra}\n{planet_details}".strip()

    result = rag_engine.ask(
        query=req.query,
        context_planet=req.context_planet,
        mode=req.mode or "standard",
        extra_context=full_extra or None
    )

    # Persist conversation to database if user is authenticated
    conv_id = None
    if current_user:
        conv = Conversation(
            user_id=current_user.id,
            query=req.query,
            response=result["answer"],
            source=result.get("source", "rag-gemini"),
            model_used=result.get("model", rag_engine.model)
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conv_id = conv.id

    result["conversation_id"] = conv_id
    result["authenticated"] = bool(current_user)
    return result

@router.post("/research")
def ai_research(req: ResearchRequest):
    """Deep astrophysical research brief with DOIs and citations."""
    return rag_engine.research(req.query)

@router.post("/report")
def ai_report(req: ReportRequest):
    """Generates comprehensive, publication-ready research dossier with PDF export support."""
    return rag_engine.generate_report(req.topic, context_body=req.context_body)

@router.get("/conversations")
def get_user_conversations(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves protected conversation history for authenticated user."""
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
        .limit(min(limit, 50))
        .all()
    )
    return {
        "count": len(convs),
        "conversations": [
            {
                "id": c.id,
                "query": c.query,
                "snippet": (c.response[:140] + "...") if len(c.response) > 140 else c.response,
                "response": c.response,
                "source": c.source,
                "model_used": c.model_used,
                "created_at": c.created_at.isoformat() if c.created_at else None
            }
            for c in convs
        ]
    }

@router.delete("/conversations")
def clear_user_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enforces ownership and securely clears authenticated user's conversation history."""
    deleted_count = db.query(Conversation).filter(Conversation.user_id == current_user.id).delete()
    db.commit()
    return {
        "status": "success",
        "deleted_count": deleted_count,
        "message": "All conversations cleared."
    }

@router.get("/context/system")
def get_system_context():
    """Returns active snapshot of all connected modules for the AI Guide."""
    # NASA cached data summary
    nasa_summary = {
        "has_cached_apod": "apod_latest" in NASA_CACHE,
        "active_cache_entries": len(NASA_CACHE),
    }
    
    planets_list = [p["name"] for p in PLANET_CATALOGUE]
    return {
        "platform": "Orbit Overload AI Research System",
        "active_features": ["RAG", "Gemini 3.5 Flash", "NASA Feeds", "3D Orrery", "ML Lab", "Quiz Engine"],
        "planets_available": planets_list,
        "catalog_planets": planets_list,
        "nasa_telemetry": nasa_summary,
        "primary_model": rag_engine.model,
        "gemini_active": bool(rag_engine.gemini_key and len(rag_engine.gemini_key) > 10),
        "rag_corpus_docs": len(rag_engine.corpus)
    }

@router.get("/rag-search")
def rag_search(q: str, top_k: int = 3):
    results = rag_engine.hybrid_search(q, top_k=top_k)
    return {
        "query": q,
        "count": len(results),
        "results": results
    }

@router.get("/models")
def ai_models():
    return {
        "primary_model": rag_engine.model,
        "fallback_model": "rag-local-lexical-corpus",
        "gemini_active": bool(rag_engine.gemini_key and len(rag_engine.gemini_key) > 10),
        "temperature": 0.3,
        "rag_corpus_documents": len(rag_engine.corpus)
    }
