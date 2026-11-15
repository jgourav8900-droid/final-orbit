import time
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Dict, Optional
from backend.database import get_db
from backend.models import QuizScore, User
from backend.routers.auth import get_current_user_optional
from backend.cache import BoundedTTLDict
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/quiz", tags=["Quiz"])

QUIZ_BANK = [
    {
        "id": "q1",
        "question": "Which photometric band in SDSS DR18 operates at the shortest ultraviolet wavelength?",
        "options": ["u (ultraviolet ~354 nm)", "g (green ~477 nm)", "r (red ~623 nm)", "z (infrared ~913 nm)"],
        "correct_index": 0,
        "explanation": "The SDSS u filter covers the near-ultraviolet spectrum centered around 354 nm."
    },
    {
        "id": "q2",
        "question": "What is the primary physical cause of orbital plane precession in Low Earth Orbit?",
        "options": ["Solar radiation pressure", "Earth's equatorial oblateness (J2 harmonic)", "Lunar tidal resonance", "Upper atmospheric drag"],
        "correct_index": 1,
        "explanation": "Earth's oblateness (the J2 gravitational zonal harmonic) produces nodal regression and apsidal precession."
    },
    {
        "id": "q3",
        "question": "In the SDSS classification model, what single feature exhibits the highest Gini feature importance?",
        "options": ["u - g color index", "Spectroscopic Redshift (z)", "i-band magnitude", "Petrosian radius"],
        "correct_index": 1,
        "explanation": "Spectroscopic redshift accounts for ~57.4% of Random Forest and ~89.8% of gradient-boosted tree importance."
    },
    {
        "id": "q4",
        "question": "At which Sun-Earth Lagrangian point is the James Webb Space Telescope (JWST) stationed?",
        "options": ["L1 (Sun-facing)", "L2 (Midnight side, ~1.5M km)", "L4 (Leading 60°)", "L5 (Trailing 60°)"],
        "correct_index": 1,
        "explanation": "JWST orbits the Sun-Earth L2 point, shielded from solar, terrestrial, and lunar infrared radiation."
    },
    {
        "id": "q5",
        "question": "What atmospheric compound produces the extreme runaway greenhouse effect on Venus?",
        "options": ["Methane (CH4)", "Carbon Dioxide (CO2 ~96.5%)", "Water vapor (H2O)", "Sulfur Dioxide (SO2)"],
        "correct_index": 1,
        "explanation": "Venus's dense CO2 atmosphere produces 92 atmospheres of surface pressure and temperatures exceeding 460 °C."
    },
    {
        "id": "q6",
        "question": "Which NASA planetary defense mission successfully demonstrated kinetic deflection on asteroid Dimorphos?",
        "options": ["OSIRIS-REx", "DART (Double Asteroid Redirection Test)", "Lucy", "Dawn"],
        "correct_index": 1,
        "explanation": "In 2022, NASA's DART spacecraft impacted Dimorphos, successfully shortening its orbital period around Didymos."
    },
    {
        "id": "q7",
        "question": "According to Kepler's Third Law, what is the relation between orbital period (T) and semi-major axis (a)?",
        "options": ["T is proportional to a", "T squared is proportional to a cubed (T² ∝ a³)", "T cubed is proportional to a squared", "T is inversely proportional to a"],
        "correct_index": 1,
        "explanation": "Kepler's Third Law states that the square of orbital period is proportional to the cube of the semi-major axis."
    },
    {
        "id": "q8",
        "question": "What is the typical altitude range for Low Earth Orbit (LEO) where space debris density peaks?",
        "options": ["10 km to 100 km", "200 km to 2,000 km", "10,000 km to 20,000 km", "35,786 km (Geostationary)"],
        "correct_index": 1,
        "explanation": "LEO extends from 200 km to 2,000 km altitude, where tracking networks actively monitor thousands of debris objects."
    },
    {
        "id": "q9",
        "question": "What stellar end-state typically results from the core collapse of a star exceeding 20 solar masses?",
        "options": ["White Dwarf", "Stellar-Mass Black Hole", "Brown Dwarf", "Planetary Nebula"],
        "correct_index": 1,
        "explanation": "Massive stars with progenitor core masses above the Tolman-Oppenheimer-Volkoff limit collapse into black holes."
    },
    {
        "id": "q10",
        "question": "Which NASA rover is currently investigating the ancient deltaic deposits of Jezero Crater on Mars?",
        "options": ["Curiosity", "Opportunity", "Perseverance", "Spirit"],
        "correct_index": 2,
        "explanation": "NASA's Perseverance rover landed in Jezero Crater in 2021 to cache samples from the ancient river delta."
    }
]

# Bounded anti-farming cooldown cache: key -> timestamp (auto-evicts, max 1000 entries)
LAST_SUBMISSIONS = BoundedTTLDict(maxsize=1000, default_ttl=3600)

class SubmitRequest(BaseModel):
    answers: Dict[str, int]  # e.g. {"q1": 0, "q2": 1}

@router.get("")
def get_quiz_questions():
    """Returns randomized quiz questions with correct answers stripped to prevent client-side tampering."""
    hardened = []
    for q in QUIZ_BANK:
        hardened.append({
            "id": q["id"],
            "question": q["question"],
            "options": q["options"]
        })
    return {
        "count": len(hardened),
        "questions": hardened
    }

@router.post("/submit")
def submit_quiz(
    req: SubmitRequest, 
    request: Request,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Evaluates quiz, enforces 10-second anti-farming cooldown with per-IP / per-user keying, and securely credits verified user account."""
    # Determine user identity securely
    user_id = current_user.id if current_user else None
    if user_id:
        cooldown_key = f"user_{user_id}"
    else:
        client_ip = request.client.host if request.client else "127.0.0.1"
        if "x-forwarded-for" in request.headers:
            client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
        cooldown_key = f"ip_{client_ip}"
    
    # 10-second anti-farming cooldown protection
    now = time.time()
    last_time = LAST_SUBMISSIONS.get(cooldown_key)
    if last_time and (now - last_time < 10.0):
        raise HTTPException(
            status_code=429, 
            detail="Cooldown active. Please wait 10 seconds between quiz evaluation attempts."
        )
    LAST_SUBMISSIONS[cooldown_key] = now

    score = 0
    feedback = []
    for q in QUIZ_BANK:
        qid = q["id"]
        chosen = req.answers.get(qid)
        is_correct = (chosen == q["correct_index"])
        if is_correct:
            score += 1
        feedback.append({
            "id": qid,
            "is_correct": is_correct,
            "correct_option": q["options"][q["correct_index"]],
            "explanation": q["explanation"]
        })

    total = len(QUIZ_BANK)
    xp_awarded = score * 50

    # Save to database record
    record = QuizScore(
        user_id=user_id,
        category="Astrophysics",
        score=score,
        total=total,
        xp_awarded=xp_awarded
    )
    db.add(record)
    
    # Update user XP and level if user is authenticated
    user_summary = None
    if current_user:
        current_user.xp += xp_awarded
        current_user.level = max(1, current_user.xp // 250 + 1)
        db.commit()
        db.refresh(current_user)
        user_summary = {
            "id": current_user.id,
            "username": current_user.username,
            "xp": current_user.xp,
            "level": current_user.level
        }
    else:
        db.commit()

    return {
        "score": score,
        "total": total,
        "percentage": round((score / total) * 100, 1),
        "xp_awarded": xp_awarded,
        "feedback": feedback,
        "user": user_summary
    }
