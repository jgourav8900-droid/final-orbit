"""
Module A: AI Astronomer Agent + Module E: Scientific Research Mode
Provides grounded astronomy Q&A, paper search via arXiv/OpenAlex, and report generation.
"""
import os
import time
import json
import urllib.request
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.routers.auth import get_current_user, get_current_user_optional
from backend.rag_engine import rag_engine
from backend.cache import BoundedTTLDict

router = APIRouter(prefix="/api/agent", tags=["AI Astronomer Agent"])

# Cache: paper search results for 10 minutes
_PAPER_CACHE: BoundedTTLDict = BoundedTTLDict(maxsize=50, default_ttl=600)
# Cache: chat context (last N turns per user/session)
_CHAT_CONTEXT: Dict[str, List[Dict]] = {}
_MAX_CONTEXT_TURNS = 8

# ─── Request/Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1200)
    level: Optional[str] = Field("intermediate", pattern="^(beginner|intermediate|advanced)$")
    context_object: Optional[str] = None
    session_id: Optional[str] = None

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=800)
    max_papers: Optional[int] = Field(5, ge=1, le=20)

class ReportRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=300)
    context_body: Optional[str] = None
    include_papers: Optional[bool] = True

class PaperSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=400)
    source: Optional[str] = Field("arxiv", pattern="^(arxiv|openalex|all)$")
    max_results: Optional[int] = Field(8, ge=1, le=20)

# ─── Paper Search Helpers ─────────────────────────────────────────────────────

def _search_arxiv(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Search arXiv for astronomy papers via public API."""
    cache_key = f"arxiv:{query.lower()[:80]}:{max_results}"
    cached = _PAPER_CACHE.get(cache_key)
    if cached:
        return cached

    encoded = urllib.parse.quote(query)
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query=ti:{encoded}+OR+abs:{encoded}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
        f"&cat=astro-ph*"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OrbitOverload/2.4 (education; contact=admin@example.com)"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_data = resp.read().decode("utf-8")
    except Exception as e:
        return [{"error": f"arXiv API unavailable: {str(e)[:80]}", "source": "arxiv"}]

    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return [{"error": "arXiv response parse error", "source": "arxiv"}]

    papers = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        id_el = entry.find("atom:id", ns)
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None]
        categories = [c.get("term", "") for c in entry.findall("atom:category", ns)]

        arxiv_id = ""
        link = ""
        if id_el is not None and id_el.text:
            raw_id = id_el.text.strip()
            arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id.split("/")[-1]
            link = f"https://arxiv.org/abs/{arxiv_id}"

        # Identify if peer-reviewed (heuristic - look for doi link)
        doi_link = None
        for lnk in entry.findall("atom:link", ns):
            if lnk.get("title") == "doi":
                doi_link = lnk.get("href", "")

        papers.append({
            "title": (title_el.text or "").strip().replace("\n", " ") if title_el is not None else "Unknown",
            "authors": authors[:5],  # limit to first 5
            "abstract": (summary_el.text or "").strip().replace("\n", " ")[:400] + ("..." if summary_el is not None and len(summary_el.text or "") > 400 else ""),
            "arxiv_id": arxiv_id,
            "url": link,
            "doi_url": doi_link,
            "published": (published_el.text or "")[:10] if published_el is not None else "",
            "categories": categories[:3],
            "type": "peer-reviewed" if doi_link else "preprint",
            "source": "arXiv",
        })

    _PAPER_CACHE.set(cache_key, papers)
    return papers


def _search_openalex(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Search OpenAlex for astronomy papers (free scholarly index)."""
    cache_key = f"openalex:{query.lower()[:80]}:{max_results}"
    cached = _PAPER_CACHE.get(cache_key)
    if cached:
        return cached

    encoded = urllib.parse.quote(query)
    url = (
        f"https://api.openalex.org/works"
        f"?search={encoded}"
        f"&filter=concepts.display_name%3Aastronomy"
        f"&sort=relevance_score:desc"
        f"&per-page={min(max_results, 10)}"
        f"&select=title,authorships,abstract_inverted_index,doi,publication_year,type,primary_location"
        f"&mailto=admin@example.com"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OrbitOverload/2.4"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return [{"error": f"OpenAlex API unavailable: {str(e)[:80]}", "source": "openalex"}]

    papers = []
    for work in data.get("results", []):
        # Reconstruct abstract from inverted index
        abstract = ""
        inv = work.get("abstract_inverted_index")
        if inv:
            pos_word = {}
            for word, positions in inv.items():
                for pos in positions:
                    pos_word[pos] = word
            abstract = " ".join(pos_word[k] for k in sorted(pos_word.keys()))[:400]

        authors = [
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])[:5]
        ]
        doi = work.get("doi", "")
        loc = work.get("primary_location", {}) or {}
        venue = (loc.get("source") or {}).get("display_name", "")

        papers.append({
            "title": work.get("title", "Unknown"),
            "authors": authors,
            "abstract": abstract,
            "arxiv_id": "",
            "url": doi or "",
            "doi_url": doi,
            "published": str(work.get("publication_year", "")),
            "categories": [],
            "venue": venue,
            "type": "peer-reviewed" if work.get("type") == "article" else work.get("type", "unknown"),
            "source": "OpenAlex",
        })

    _PAPER_CACHE.set(cache_key, papers)
    return papers


def _build_level_instruction(level: str) -> str:
    if level == "beginner":
        return "Explain clearly for someone with no astronomy background. Avoid jargon; define any technical terms used."
    elif level == "advanced":
        return "Provide technically rigorous detail suitable for a graduate student or researcher. Include quantitative context where relevant."
    return "Explain at an intermediate level, suitable for a curious science enthusiast. Use some technical terms but briefly explain them."


# ─── API Endpoints ────────────────────────────────────────────────────────────

@router.post("/chat")
def agent_chat(
    req: ChatRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Conversational astronomy Q&A with context-aware follow-ups.
    Supports beginner/intermediate/advanced levels.
    Grounded with RAG corpus; uses Gemini for synthesis.
    """
    level_instr = _build_level_instruction(req.level or "intermediate")
    
    # Build context-enhanced query
    context_parts = []
    if req.context_object:
        context_parts.append(f"The user is currently viewing: {req.context_object}")
    context_parts.append(f"Explanation level requested: {req.level}")
    context_parts.append(level_instr)
    extra_ctx = "\n".join(context_parts)

    # Retrieve RAG results
    rag_result = rag_engine.ask(
        query=req.query,
        mode="research" if (req.level == "advanced") else "standard",
        extra_context=extra_ctx
    )

    # Build suggested follow-ups based on topic
    follow_ups = _generate_followups(req.query, req.level or "intermediate")

    return {
        "answer": rag_result.get("answer", ""),
        "source": rag_result.get("source", "rag-gemini"),
        "model": rag_result.get("model", ""),
        "latency_ms": rag_result.get("latency_ms", 0),
        "level": req.level,
        "context_object": req.context_object,
        "suggested_followups": follow_ups,
        "corpus_sources": rag_result.get("sources", []),
        "disclaimer": "AI-generated explanation grounded in astronomy knowledge base. Verify scientific claims with peer-reviewed sources.",
    }


@router.post("/research")
def agent_research(req: ResearchRequest):
    """
    Deep research brief: searches arXiv for papers, retrieves RAG context,
    synthesizes a grounded answer with citations.
    """
    # Parallel: search papers + RAG
    papers = _search_arxiv(req.query, max_results=min(req.max_papers or 5, 10))
    real_papers = [p for p in papers if "error" not in p]

    rag_result = rag_engine.research(req.query)

    # Build paper summary for context
    paper_context = ""
    if real_papers:
        summaries = []
        for p in real_papers[:5]:
            summaries.append(f"- '{p['title']}' by {', '.join(p['authors'][:2])} ({p['published'][:4]}): {p['abstract'][:200]}")
        paper_context = "Recent arXiv papers:\n" + "\n".join(summaries)

    return {
        "query": req.query,
        "summary": rag_result.get("summary", rag_result.get("answer", "")),
        "papers": real_papers,
        "paper_count": len(real_papers),
        "rag_sources": rag_result.get("sources", []),
        "source": "arxiv+rag",
        "disclaimer": "Paper abstracts from arXiv (preprint server). Check publication status before citing. AI synthesis may contain errors.",
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@router.post("/report")
def agent_report(
    req: ReportRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Generate a structured scientific research report with citations.
    Format: title, abstract, background, evidence, findings, limitations, references.
    """
    papers = []
    if req.include_papers:
        papers = [p for p in _search_arxiv(req.topic, max_results=6) if "error" not in p]

    report = rag_engine.generate_report(req.topic, context_body=req.context_body)

    # Attach papers to report
    report["papers"] = papers
    report["paper_count"] = len(papers)
    report["generated_by"] = "Orbit Overload AI Astronomer + arXiv"
    report["disclaimer"] = (
        "This report was AI-generated using the Orbit Overload RAG system and arXiv preprint data. "
        "It is for educational purposes only. Verify all claims and citations before scientific use."
    )
    return report


@router.get("/papers/search")
def search_papers(
    q: str,
    source: str = "arxiv",
    max_results: int = 8
):
    """
    Search for astronomy papers via arXiv or OpenAlex.
    Returns title, authors, abstract, DOI/arXiv link, publication date, type.
    """
    if not q or len(q) < 2:
        raise HTTPException(status_code=422, detail="Query must be at least 2 characters")
    if max_results < 1 or max_results > 20:
        max_results = 8

    papers = []
    errors = []

    if source in ("arxiv", "all"):
        arxiv_results = _search_arxiv(q, max_results)
        for p in arxiv_results:
            if "error" in p:
                errors.append({"source": "arXiv", "error": p["error"]})
            else:
                papers.append(p)

    if source in ("openalex", "all"):
        oa_results = _search_openalex(q, max_results)
        for p in oa_results:
            if "error" in p:
                errors.append({"source": "OpenAlex", "error": p["error"]})
            else:
                papers.append(p)

    return {
        "query": q,
        "count": len(papers),
        "papers": papers,
        "errors": errors,
        "source": source,
        "disclaimer": "Preprint papers from arXiv may not be peer-reviewed. Always verify publication status.",
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@router.get("/explain/{object_name}")
def explain_object(object_name: str, level: str = "intermediate"):
    """
    Quick AI explanation of a celestial object.
    Used by the Interactive Universe 'Explain this object' button.
    """
    level_instr = _build_level_instruction(level)
    query = f"Explain the celestial object: {object_name}. What is it, how was it discovered, why is it significant, and what do scientists still not know about it?"

    result = rag_engine.ask(
        query=query,
        extra_context=f"Explanation level: {level}. {level_instr}. Focus on: {object_name}"
    )
    return {
        "object": object_name,
        "explanation": result.get("answer", ""),
        "source": result.get("source", "rag-gemini"),
        "level": level,
        "disclaimer": "AI-generated explanation. Verify with authoritative astronomical sources.",
    }


@router.post("/compare-papers")
def compare_papers(req: PaperSearchRequest):
    """
    Search papers and return a comparison: research question, method, findings, limitations.
    """
    papers = []
    if req.source in ("arxiv", "all"):
        papers += [p for p in _search_arxiv(req.query, req.max_results or 6) if "error" not in p]
    if req.source in ("openalex", "all"):
        papers += [p for p in _search_openalex(req.query, req.max_results or 6) if "error" not in p]

    if not papers:
        return {"query": req.query, "papers": [], "comparison": None, "message": "No papers found for comparison."}

    # Build AI comparison
    paper_texts = []
    for i, p in enumerate(papers[:5], 1):
        paper_texts.append(f"Paper {i}: '{p['title']}' ({p['published'][:4]}) — {p['abstract'][:200]}")

    comparison_query = (
        f"Compare these astronomy research papers on the topic of '{req.query}':\n"
        + "\n".join(paper_texts)
        + "\n\nFor each paper identify: research question, dataset/method, key findings, limitations. "
        "Then summarize key differences and agreements. Cite papers by their title."
    )

    result = rag_engine.ask(query=comparison_query, mode="research")

    return {
        "query": req.query,
        "papers": papers[:5],
        "paper_count": len(papers[:5]),
        "comparison": result.get("answer", ""),
        "disclaimer": "AI-generated comparison. Verify claims against original papers.",
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _generate_followups(query: str, level: str) -> List[str]:
    """Generate contextually relevant follow-up questions."""
    q_lower = query.lower()
    followups = []

    if any(w in q_lower for w in ["planet", "earth", "mars", "jupiter", "saturn"]):
        followups = [
            "How do planetary atmospheres compare across the solar system?",
            "What missions are currently exploring this planet?",
            "Could this planet support life?",
        ]
    elif any(w in q_lower for w in ["star", "sun", "stellar", "supernova", "neutron"]):
        followups = [
            "What happens when a star like this reaches the end of its life?",
            "How do astronomers measure a star's distance?",
            "What is the stellar classification system?",
        ]
    elif any(w in q_lower for w in ["galaxy", "milky way", "andromeda", "quasar"]):
        followups = [
            "How do galaxies form and evolve over cosmic time?",
            "What is at the center of most large galaxies?",
            "How do astronomers measure galactic distances?",
        ]
    elif any(w in q_lower for w in ["exoplanet", "transit", "kepler", "tess"]):
        followups = [
            "How do astronomers detect exoplanet atmospheres?",
            "What makes a planet potentially habitable?",
            "How many confirmed exoplanets are there?",
        ]
    elif any(w in q_lower for w in ["black hole", "dark matter", "dark energy"]):
        followups = [
            "How was this first observed or confirmed?",
            "What do scientists still not understand about this?",
            "How does this affect the structure of the universe?",
        ]
    else:
        followups = [
            "What are the most recent discoveries in this area?",
            "How is this studied with modern telescopes?",
            "What are the open questions in this field?",
        ]

    if level == "advanced":
        followups.append("What are the current theoretical models explaining this?")
    elif level == "beginner":
        followups = [f.replace("astronomers", "scientists") for f in followups]

    return followups[:3]
