"""
Orbit Overload – Unlimited AI Quiz Router
Generates fresh quiz questions on any topic using Gemini AI.
API key is strictly server-side; never exposed to the client.
"""

import os
import re
import json
import hashlib
import secrets
import time
import datetime
import logging
import urllib.request
import urllib.error
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import AIQuizAttempt, AIQuizIssuedAttempt, AIQuestionRecord, User
from backend.routers.auth import get_current_user_optional
from backend.cache import BoundedTTLDict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai-quiz", tags=["AI Quiz"])

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
]
DIFFICULTY_LABELS = {"beginner", "intermediate", "advanced", "expert"}
MAX_QUESTIONS = 20
MIN_QUESTIONS = 3

# Bounded in-memory caches to prevent memory leaks and farming
_QCACHE = BoundedTTLDict(maxsize=3000, default_ttl=7200)               # hash -> question dict (2h TTL)
_LAST_GEN = BoundedTTLDict(maxsize=2000, default_ttl=3600)             # identifier -> last request time
_LAST_SUBMIT = BoundedTTLDict(maxsize=2000, default_ttl=3600)          # identifier -> last submit time
ISSUED_ATTEMPT_TTL = datetime.timedelta(hours=2)

def get_client_ip(request: Request) -> str:
    if "x-forwarded-for" in request.headers:
        return request.headers["x-forwarded-for"].split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=120)
    difficulty: str = Field("intermediate")
    count: int = Field(5, ge=MIN_QUESTIONS, le=MAX_QUESTIONS)


class SubmitRequest(BaseModel):
    session_id: str = Field(..., min_length=32, max_length=64)
    topic: str = Field(..., min_length=2, max_length=120)
    difficulty: str = Field(..., min_length=1, max_length=20)
    questions: List[dict] = Field(..., min_length=1, max_length=MAX_QUESTIONS)
    answers: List[int] = Field(..., min_length=1, max_length=MAX_QUESTIONS)
    time_taken_seconds: int = Field(0, ge=0, le=86400)


class HistoryParams(BaseModel):
    limit: int = Field(10, ge=1, le=50)


# ---------------------------------------------------------------------------
# Gemini helper
# ---------------------------------------------------------------------------
def _get_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key or len(key) < 10:
        raise HTTPException(status_code=503, detail="Gemini API key not configured. Add GEMINI_API_KEY to .env")
    return key


def _call_gemini(prompt: str, timeout: int = 10) -> str:
    """Call Gemini with multi-model fallback. Sanitizes errors and keeps secrets internal."""
    key = _get_gemini_key()
    last_error_code = None

    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": 4096
            }
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            last_error_code = e.code
            logger.warning("Gemini AI Quiz HTTP %s on model %s", e.code, model)
            if e.code == 404 or e.code == 400:
                continue   # Try next model
            if e.code == 429 or e.code >= 500:
                time.sleep(0.3)
                continue
        except Exception as ex:
            logger.warning("Gemini AI Quiz network/parse error on model %s: %s", model, type(ex).__name__)
            continue

    status_code = 429 if last_error_code == 429 else 503
    raise HTTPException(status_code=status_code, detail="AI Quiz generation temporarily unavailable from upstream provider. Please try again.")


def _make_quiz_prompt(topic: str, difficulty: str, count: int) -> str:
    difficulty_map = {
        "beginner":     "simple, factual, single-concept questions suitable for beginners with no prior knowledge",
        "intermediate": "moderately challenging questions that require understanding of core concepts",
        "advanced":     "in-depth questions requiring strong domain knowledge and analytical reasoning",
        "expert":       "expert-level questions involving nuanced technical depth, edge cases, and expert analysis",
    }
    desc = difficulty_map.get(difficulty, difficulty_map["intermediate"])

    return f"""Generate exactly {count} multiple-choice quiz questions about the topic: "{topic}".

Difficulty level: {difficulty.upper()} — {desc}

Requirements:
- Each question must have exactly 4 answer options (A, B, C, D)
- Questions must be UNIQUE, varied, and cover different aspects of the topic
- Explanations must be clear, educational, and 1-3 sentences
- Do NOT repeat concepts across questions
- For expert/advanced: include quantitative, edge-case, or deeply technical questions

Output ONLY valid JSON — no markdown, no explanation text — in this exact structure:
{{
  "questions": [
    {{
      "question": "...",
      "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
      "correct_index": 0,
      "explanation": "..."
    }}
  ]
}}

The correct_index is 0-based (0=A, 1=B, 2=C, 3=D).
Topic: {topic}
Count: {count}"""


def _parse_quiz_json(raw: str) -> list:
    """Extract and parse JSON from Gemini response, with cleaning."""
    # Strip markdown code fences
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```")[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```")[0]

    # Try to find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        data = json.loads(text)
        questions = data.get("questions", [])
        if not questions:
            raise ValueError("No questions found in response")
        # Validate structure
        validated = []
        for q in questions:
            if not all(k in q for k in ("question", "options", "correct_index", "explanation")):
                continue
            if len(q["options"]) != 4:
                continue
            if not isinstance(q["correct_index"], int) or not (0 <= q["correct_index"] <= 3):
                continue
            validated.append({
                "question": str(q["question"]).strip(),
                "options": [str(o).strip() for o in q["options"]],
                "correct_index": int(q["correct_index"]),
                "explanation": str(q["explanation"]).strip()
            })
        return validated
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to parse Gemini quiz response: {e}. Raw snippet: {raw[:200]}")


def _question_hash(topic: str, question: str) -> str:
    return hashlib.sha256(f"{topic.lower()}::{question.lower()}".encode()).hexdigest()[:48]


# ---------------------------------------------------------------------------
# XP / scoring rules
# ---------------------------------------------------------------------------
XP_PER_CORRECT = {"beginner": 20, "intermediate": 35, "advanced": 55, "expert": 80}
TIME_BONUS_THRESHOLD = 15  # seconds per question on average → bonus


def _calc_xp(score: int, total: int, difficulty: str, time_taken: int) -> int:
    per = XP_PER_CORRECT.get(difficulty, 35)
    base = score * per
    pct = score / max(total, 1)
    # Perfect bonus
    if score == total:
        base += total * 20
    # Time bonus: fast finisher
    avg_time = time_taken / max(total, 1)
    if avg_time <= TIME_BONUS_THRESHOLD:
        base += int(score * 10)
    # Accuracy bonus tiers
    if pct >= 0.9:
        base = int(base * 1.15)
    elif pct >= 0.7:
        base = int(base * 1.05)
    return base


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
def ai_quiz_status():
    """Service health and capability status."""
    key = os.getenv("GEMINI_API_KEY", "")
    return {
        "service": "Orbit Overload AI Quiz Engine",
        "gemini_configured": len(key) > 10,
        "models": GEMINI_MODELS,
        "difficulty_levels": list(DIFFICULTY_LABELS),
        "question_range": {"min": MIN_QUESTIONS, "max": MAX_QUESTIONS},
        "question_cache_size": len(_QCACHE),
        "xp_per_correct": XP_PER_CORRECT,
    }


@router.post("/generate")
def generate_quiz(
    req: GenerateRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Generate fresh quiz questions using Gemini AI.
    Rate-limited: 1 generation per IP or authenticated user per 15 seconds.
    Gemini API key is kept server-side only.
    """
    # Validate difficulty
    topic = req.topic.strip()
    if len(topic) < 2:
        raise HTTPException(status_code=422, detail="topic must contain at least 2 non-whitespace characters")

    diff = req.difficulty.lower()
    if diff not in DIFFICULTY_LABELS:
        raise HTTPException(status_code=422, detail=f"difficulty must be one of: {sorted(DIFFICULTY_LABELS)}")

    # Rate-limit by authenticated user or client IP; session identifiers are server-issued.
    now = time.time()
    ip_addr = get_client_ip(request)
    identity_key = f"user_{current_user.id}" if current_user else f"ip_{ip_addr}"
    
    last_identity = _LAST_GEN.get(identity_key, 0)

    if now - last_identity < 15:
        wait = round(15 - (now - last_identity), 1)
        raise HTTPException(status_code=429, detail=f"Please wait {wait}s before generating a new quiz.")
    
    _LAST_GEN[identity_key] = now

    prompt = _make_quiz_prompt(topic, diff, req.count)
    raw = _call_gemini(prompt)
    questions = _parse_quiz_json(raw)

    if not questions:
        raise HTTPException(status_code=502, detail="Gemini returned an empty quiz. Please try again.")

    # Create unique server-owned question data and hashes for this issued attempt.
    issued_questions = []
    public_questions = []
    seen_hashes = set()
    for q in questions:
        h = _question_hash(topic, q["question"])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        if len(issued_questions) == req.count:
            break
        stored_question = {"hash": h, **q}
        issued_questions.append(stored_question)
        _QCACHE[h] = q  # cache full question (with answer)
        public_questions.append({
            "hash": h,
            "question": q["question"],
            "options": q["options"],
            # correct_index deliberately omitted — evaluated server-side on submit
        })

    if len(issued_questions) != req.count:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned fewer unique questions than requested. Please try again.",
        )

    now = datetime.datetime.now(datetime.timezone.utc)
    db.query(AIQuizIssuedAttempt).filter(AIQuizIssuedAttempt.expires_at <= now).delete(synchronize_session=False)
    session_id = secrets.token_urlsafe(32)
    issued_attempt = AIQuizIssuedAttempt(
        session_id=session_id,
        user_id=current_user.id if current_user else None,
        topic=topic,
        difficulty=diff,
        questions_json=json.dumps(issued_questions),
        created_at=now,
        expires_at=now + ISSUED_ATTEMPT_TTL,
    )
    db.add(issued_attempt)
    db.commit()

    return {
        "topic": topic,
        "difficulty": diff,
        "count": len(public_questions),
        "session_id": session_id,
        "questions": public_questions,
        "_generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@router.post("/submit")
def submit_ai_quiz(
    req: SubmitRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Evaluate quiz answers server-side.
    Answers are checked against the exact questions in a persisted server-issued attempt.
    Attempts are atomically consumed to prevent replay and duplicate scoring.
    """
    if not req.questions:
        raise HTTPException(status_code=422, detail="No questions provided.")
    if len(req.answers) != len(req.questions):
        raise HTTPException(status_code=422, detail="Answer count must match question count.")

    now = datetime.datetime.now(datetime.timezone.utc)
    now_timestamp = time.time()
    attempt = db.query(AIQuizIssuedAttempt).filter(
        AIQuizIssuedAttempt.session_id == req.session_id,
        AIQuizIssuedAttempt.expires_at > now,
    ).first()
    if not attempt or attempt.consumed_at is not None:
        raise HTTPException(status_code=400, detail="This quiz attempt is missing, expired, or already submitted.")

    user_id = current_user.id if current_user else None
    if attempt.user_id != user_id:
        raise HTTPException(status_code=403, detail="This quiz attempt belongs to a different user session.")

    expected_questions = json.loads(attempt.questions_json)
    expected_hashes = [question["hash"] for question in expected_questions]
    submitted_hashes = [question.get("hash") for question in req.questions]
    if (
        len(set(expected_hashes)) != len(expected_hashes)
        or any(not isinstance(question_hash, str) for question_hash in submitted_hashes)
        or submitted_hashes != expected_hashes
        or len(set(submitted_hashes)) != len(submitted_hashes)
    ):
        raise HTTPException(status_code=400, detail="Submitted questions do not match the issued quiz attempt.")
    if req.topic.strip() != attempt.topic or req.difficulty.lower() != attempt.difficulty:
        raise HTTPException(status_code=400, detail="Submitted quiz metadata does not match the issued attempt.")
    if len(req.answers) != len(expected_questions):
        raise HTTPException(status_code=422, detail="Answer count must match the issued question count.")
    if any(answer < -1 or answer > 3 for answer in req.answers):
        raise HTTPException(status_code=422, detail="Answers must be -1 (unanswered) or an option index from 0 to 3.")

    diff = attempt.difficulty

    # Submission rate-limiting per user / IP
    ip_addr = get_client_ip(request)
    submit_key = f"sub_user_{current_user.id}" if current_user else f"sub_ip_{ip_addr}"
    last_sub = _LAST_SUBMIT.get(submit_key, 0)
    if now_timestamp - last_sub < 5.0:
        raise HTTPException(status_code=429, detail="Submission cooldown active. Please wait 5 seconds between quiz submissions.")
    _LAST_SUBMIT[submit_key] = now_timestamp

    owner_filter = AIQuizIssuedAttempt.user_id == user_id
    claimed = db.query(AIQuizIssuedAttempt).filter(
        AIQuizIssuedAttempt.session_id == req.session_id,
        AIQuizIssuedAttempt.consumed_at.is_(None),
        AIQuizIssuedAttempt.expires_at > now,
        owner_filter,
    ).update(
        {AIQuizIssuedAttempt.consumed_at: now},
        synchronize_session=False,
    )
    if claimed != 1:
        db.rollback()
        raise HTTPException(status_code=400, detail="This quiz attempt has already been submitted.")

    score = 0
    feedback = []
    for i, (question, user_ans) in enumerate(zip(expected_questions, req.answers)):
        correct_idx = question["correct_index"]
        explanation = question["explanation"]

        is_correct = (user_ans == correct_idx)
        if is_correct:
            score += 1

        feedback.append({
            "question_number": i + 1,
            "question": question["question"],
            "user_answer_index": user_ans,
            "correct_index": correct_idx,
            "user_answer": question["options"][user_ans] if 0 <= user_ans < len(question["options"]) else "No answer",
            "correct_answer": question["options"][correct_idx] if 0 <= correct_idx < len(question["options"]) else "",
            "is_correct": is_correct,
            "explanation": explanation,
        })

    total = len(expected_questions)
    percentage = round(score / total * 100, 1)
    xp = _calc_xp(score, total, diff, req.time_taken_seconds)

    # Save to DB
    record = AIQuizAttempt(
        user_id=current_user.id if current_user else None,
        session_id=req.session_id,
        topic=attempt.topic,
        difficulty=diff,
        question_count=total,
        score=score,
        total=total,
        percentage=percentage,
        xp_awarded=xp,
        time_taken_seconds=req.time_taken_seconds,
        review_data=json.dumps(feedback),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )
    db.add(record)

    user_summary = None
    if current_user:
        current_user.xp = (current_user.xp or 0) + xp
        current_user.level = max(1, current_user.xp // 250 + 1)
        db.commit()
        db.refresh(current_user)
        user_summary = {
            "id": current_user.id,
            "username": current_user.username,
            "xp": current_user.xp,
            "level": current_user.level,
        }
    else:
        db.commit()

    grade = (
        "🏆 Perfect!" if score == total
        else "🌟 Excellent!" if percentage >= 90
        else "⭐ Great!" if percentage >= 75
        else "✅ Good!" if percentage >= 60
        else "📚 Keep studying!" if percentage >= 40
        else "💡 Review needed"
    )

    return {
        "score": score,
        "total": total,
        "percentage": percentage,
        "xp_awarded": xp,
        "grade": grade,
        "feedback": feedback,
        "user": user_summary,
        "attempt_id": record.id,
    }


@router.get("/history")
def ai_quiz_history(
    limit: int = 10,
    topic: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Retrieve quiz attempt history for the authenticated user (or guest by session).
    """
    if not current_user:
        return {"history": [], "message": "Sign in to view your quiz history."}

    q = db.query(AIQuizAttempt).filter(AIQuizAttempt.user_id == current_user.id)
    if topic:
        q = q.filter(AIQuizAttempt.topic.ilike(f"%{topic}%"))
    q = q.order_by(AIQuizAttempt.created_at.desc()).limit(min(limit, 50))

    rows = q.all()
    history = []
    for r in rows:
        history.append({
            "id": r.id,
            "topic": r.topic,
            "difficulty": r.difficulty,
            "score": r.score,
            "total": r.total,
            "percentage": r.percentage,
            "xp_awarded": r.xp_awarded,
            "time_taken_seconds": r.time_taken_seconds,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })

    return {
        "user": current_user.username,
        "total_attempts": len(history),
        "history": history,
    }


@router.get("/leaderboard")
def ai_quiz_leaderboard(db: Session = Depends(get_db)):
    """Top 10 users by total AI Quiz XP earned."""
    from sqlalchemy import func
    rows = (
        db.query(
            AIQuizAttempt.user_id,
            func.sum(AIQuizAttempt.xp_awarded).label("total_xp"),
            func.count(AIQuizAttempt.id).label("attempts"),
            func.avg(AIQuizAttempt.percentage).label("avg_pct"),
        )
        .filter(AIQuizAttempt.user_id.isnot(None))
        .group_by(AIQuizAttempt.user_id)
        .order_by(func.sum(AIQuizAttempt.xp_awarded).desc())
        .limit(10)
        .all()
    )

    board = []
    for r in rows:
        user = db.query(User).filter(User.id == r.user_id).first()
        board.append({
            "rank": len(board) + 1,
            "username": user.username if user else "Unknown",
            "total_xp": int(r.total_xp or 0),
            "attempts": int(r.attempts or 0),
            "avg_score_pct": round(float(r.avg_pct or 0), 1),
        })

    return {"leaderboard": board}
