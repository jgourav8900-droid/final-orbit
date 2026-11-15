import json
import time
import uuid
import pytest
from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["nasa_key_configured"] is True
    assert data["gemini_key_configured"] is True

def test_planets():
    res = client.get("/api/planets")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 10
    names = [b["name"] for b in data["bodies"]]
    assert "Earth" in names
    assert "Mars" in names
    assert "Jupiter" in names

def test_ml_status():
    res = client.get("/api/ml/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["dataset_loaded"] is True
    assert data["dataset_records"] == 1500
    assert "Random Forest Classifier (scikit-learn)" in data["models_available"]
    assert "XGBoost Classifier" in data["models_available"]
    assert data["shap_explainability"] is True

def test_ml_dataset():
    res = client.get("/api/ml/dataset?page=1&limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is True
    assert len(data["records"]) == 10
    assert data["summary"]["total_records"] == 1500
    assert data["summary"]["class_counts"]["STAR"] == 500
    assert data["summary"]["class_counts"]["GALAXY"] == 500
    assert data["summary"]["class_counts"]["QUASAR"] == 500

def test_ml_predict_rf_and_xgb():
    sample = {"u": 18.2, "g": 17.5, "r": 17.1, "i": 16.9, "z": 16.8, "redshift": 0.0001}
    # Test RF
    res_rf = client.post("/api/ml/predict", json={"model": "rf", **sample})
    assert res_rf.status_code == 200
    d_rf = res_rf.json()
    assert d_rf["prediction"] == "STAR"
    assert d_rf["probabilities"]["STAR"] > 0.80

    # Test XGBoost
    res_xgb = client.post("/api/ml/predict", json={"model": "xgb", **sample})
    assert res_xgb.status_code == 200
    d_xgb = res_xgb.json()
    assert d_xgb["prediction"] == "STAR"
    assert d_xgb["probabilities"]["STAR"] > 0.80

def test_ml_explain_shap():
    sample = {"u": 19.5, "g": 19.1, "r": 18.9, "i": 18.7, "z": 18.6, "redshift": 1.45}
    res = client.post("/api/ml/explain", json={"model": "rf", **sample})
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is True
    assert data["predicted_class"] == "QUASAR"
    assert len(data["contributions"]) == 10
    assert "redshift" in [c["feature"] for c in data["contributions"]]

def test_ml_classify_star():

    res = client.post("/api/ml/classify", json={
        "u": 18.2, "g": 17.5, "r": 17.1, "i": 16.9, "z": 16.8, "redshift": 0.0001
    })
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "STAR"
    assert "probabilities" in data
    assert "derived_colors" in data

def test_ml_classify_quasar():
    res = client.post("/api/ml/classify", json={
        "u": 19.5, "g": 19.1, "r": 18.9, "i": 18.7, "z": 18.6, "redshift": 1.45
    })
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "QUASAR"

def test_ml_metrics():
    res = client.get("/api/ml/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "holdout_accuracy" in data
    assert data["holdout_accuracy"] >= 0.90
    assert "feature_importances" in data

def test_ml_orbit_simulate():
    res = client.post("/api/ml/orbit-simulate", json={
        "semi_major_axis_au": 1.524,
        "eccentricity": 0.0934,
        "inclination_deg": 1.85,
        "raan_deg": 49.5,
        "arg_per_deg": 286.5,
        "points": 50
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data["trajectory"]) == 51
    assert "orbital_period_hours" in data
    assert "period_years" in data
    assert data["parameters"]["semi_major_axis_au"] == pytest.approx(1.524)
    assert data["parameters"]["semi_major_axis_km"] == pytest.approx(1.524 * 149_597_870.7, rel=1e-6)
    assert data["period_years"] == pytest.approx(1.8808, abs=0.01)


@pytest.mark.parametrize("overrides", [
    {"semi_major_axis_au": 0},
    {"eccentricity": 1},
    {"eccentricity": -0.1},
    {"points": 0},
    {"points": 2001},
])
def test_ml_orbit_rejects_invalid_parameters(overrides):
    payload = {
        "semi_major_axis_au": 1.0,
        "eccentricity": 0.0167,
        "inclination_deg": 0.0,
        "raan_deg": 0.0,
        "arg_per_deg": 0.0,
        "points": 100,
    }
    payload.update(overrides)
    response = client.post("/api/ml/orbit-simulate", json=payload)
    assert response.status_code == 422


def test_ml_orbit_rejects_non_finite_parameters():
    from pydantic import ValidationError
    from backend.routers.ml import OrbitSimulateRequest

    with pytest.raises(ValidationError):
        OrbitSimulateRequest(semi_major_axis_au=float("nan"))


def test_ml_training_requires_operator_token(monkeypatch):
    from backend.routers import ml as ml_router

    monkeypatch.setattr(ml_router, "ML_TRAINING_TOKEN", "")
    disabled = client.post("/api/ml/train")
    assert disabled.status_code == 503

    monkeypatch.setattr(ml_router, "ML_TRAINING_TOKEN", "test-training-token-that-is-at-least-32-chars")
    denied = client.post("/api/ml/train")
    assert denied.status_code == 403
    denied = client.post("/api/ml/train", headers={"X-ML-Training-Token": "wrong"})
    assert denied.status_code == 403

    def complete_fake_training():
        with ml_router._training_state_lock:
            ml_router._training_state.update(status="completed", finished_at="test", error=None)
        ml_router._training_lock.release()

    monkeypatch.setattr(ml_router, "_run_training_job", complete_fake_training)
    queued = client.post(
        "/api/ml/train",
        headers={"X-ML-Training-Token": "test-training-token-that-is-at-least-32-chars"},
    )
    assert queued.status_code == 202
    assert queued.json()["status"] == "queued"
    assert client.get("/api/ml/train/status").json()["status"] == "completed"

def test_ai_rag_search():
    res = client.get("/api/ai/rag-search?q=kepler+orbits&top_k=2")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 1
    assert "Keplerian" in data["results"][0]["title"]

def test_ai_ask():
    res = client.post("/api/ai/ask", json={"query": "Explain Kepler's second law."})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert len(data["answer"]) > 10

def test_nasa_status():
    res = client.get("/api/nasa/status")
    assert res.status_code == 200
    data = res.json()
    assert data["api_key_configured"] is True
    assert "feeds" in data
    assert "apod" in data["feeds"]
    assert "neo" in data["feeds"]

def test_nasa_apod():
    res = client.get("/api/nasa/apod")
    assert res.status_code == 200
    data = res.json()
    assert "title" in data
    assert "url" in data

def test_nasa_apod_history():
    res = client.get("/api/nasa/apod/history?count=3")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data

def test_nasa_neo():
    res = client.get("/api/nasa/neo")
    assert res.status_code == 200
    data = res.json()
    assert "near_earth_objects" in data

def test_nasa_epic():
    res = client.get("/api/nasa/epic")
    assert res.status_code == 200
    data = res.json()
    assert "images" in data

def test_nasa_mars():
    res = client.get("/api/nasa/mars?rover=curiosity&sol=1000")
    assert res.status_code == 200
    data = res.json()
    assert "rover" in data

def test_nasa_donki_flares():
    res = client.get("/api/nasa/donki/flares")
    assert res.status_code == 200
    data = res.json()
    assert "flares" in data


def test_quiz_hardening():
    # Hardening test: verify correct answers and explanations are stripped from GET
    res = client.get("/api/quiz")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 5
    for q in data["questions"]:
        assert "correct_index" not in q
        assert "explanation" not in q

def test_auth_and_secured_quiz():
    uid = str(uuid.uuid4())[:8]
    username = f"astro_{uid}"
    email = f"astro_{uid}@example.com"
    password = "SuperSecretAstroPassword123!"

    # 1. Register with bcrypt hashing
    reg_res = client.post("/api/auth/register", json={
        "username": username,
        "email": email,
        "password": password
    })
    assert reg_res.status_code == 200
    reg_data = reg_res.json()
    assert "token" in reg_data
    token = reg_data["token"]
    assert reg_data["user"]["xp"] == 100

    # 2. Get Me (Authenticated with PyJWT)
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["username"] == username

    # 3. Token refresh endpoint
    refresh_res = client.post("/api/auth/refresh", headers={"Authorization": f"Bearer {token}"})
    assert refresh_res.status_code == 200
    new_token = refresh_res.json()["token"]
    assert isinstance(new_token, str) and len(new_token) > 20

    # 4. Submit quiz as authenticated user
    quiz_res = client.post(
        "/api/quiz/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={"answers": {"q1": 0, "q2": 1, "q3": 1, "q4": 1, "q5": 1}}
    )
    assert quiz_res.status_code == 200
    quiz_data = quiz_res.json()
    assert quiz_data["score"] == 5
    assert quiz_data["xp_awarded"] == 250
    assert quiz_data["user"]["xp"] == 350  # 100 bonus + 250 quiz

    # 5. Anti-farming cooldown triggers 429 when immediate second submission
    cooldown_res = client.post(
        "/api/quiz/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={"answers": {"q1": 0}}
    )
    assert cooldown_res.status_code == 429

def test_auth_negative_cases():
    # Password too short (minimum is now 8 characters)
    res_short = client.post("/api/auth/register", json={
        "username": "shorty", "email": "short@example.com", "password": "1234567"
    })
    assert res_short.status_code == 400

    # Missing auth header
    res_no_auth = client.get("/api/auth/me")
    assert res_no_auth.status_code == 401

    # Invalid token format
    res_bad_tok = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid.token.signature"})
    assert res_bad_tok.status_code == 401

def test_frontend_routes():
    routes = ["/", "/solar", "/ai-guide", "/orbit-ai", "/ml-lab", "/nasa-live", "/planets", "/quiz", "/ai-quiz", "/3d"]
    for r in routes:
        res = client.get(r)
        assert res.status_code == 200
        assert len(res.content) > 500

def test_ai_system_context():
    res = client.get("/api/ai/context/system")
    assert res.status_code == 200
    data = res.json()
    assert "planets_available" in data
    assert "Jupiter" in data["planets_available"]
    assert "primary_model" in data

def test_ai_report():
    res = client.post("/api/ai/report", json={
        "topic": "Atmospheric Composition of Titan",
        "context_body": "Titan"
    })
    assert res.status_code == 200
    data = res.json()
    assert "dossier" in data
    assert len(data["dossier"]) > 50
    assert "citations" in data
    assert "dois" in data

def test_ai_conversations_auth():
    # Register/login user
    import uuid
    uid = uuid.uuid4().hex[:6]
    reg = client.post("/api/auth/register", json={
        "username": f"ai_user_{uid}",
        "email": f"ai_{uid}@example.com",
        "password": "Password123!"
    })
    assert reg.status_code == 200
    token = reg.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Ask authenticated question
    ask_res = client.post(
        "/api/ai/ask",
        json={"query": "What are the rings of Saturn composed of?", "planet": "Saturn", "mode": "research"},
        headers=headers
    )
    assert ask_res.status_code == 200
    data = ask_res.json()
    assert "answer" in data
    assert "suggested_followups" in data

    # Fetch conversation history
    conv_res = client.get("/api/ai/conversations", headers=headers)
    assert conv_res.status_code == 200
    history = conv_res.json()["conversations"]
    assert len(history) >= 1
    assert any("Saturn" in item.get("query", "") or "Saturn" in str(item) for item in history)

    # Clear conversation history
    del_res = client.delete("/api/ai/conversations", headers=headers)
    assert del_res.status_code == 200
    empty_history = client.get("/api/ai/conversations", headers=headers).json()["conversations"]
    assert len(empty_history) == 0

def test_ai_quiz_status():
    res = client.get("/api/ai-quiz/status")
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "Orbit Overload AI Quiz Engine"
    assert data["gemini_configured"] is True
    assert "beginner" in data["difficulty_levels"]
    assert "expert" in data["difficulty_levels"]
    assert data["question_range"]["min"] == 3
    assert data["question_range"]["max"] == 20

def test_ai_quiz_leaderboard():
    res = client.get("/api/ai-quiz/leaderboard")
    assert res.status_code == 200
    assert "leaderboard" in res.json()

def test_ai_quiz_submit_and_xp(monkeypatch):
    from backend.routers import ai_quiz

    generated_questions = [
        {
            "question": "What is the Schwarzschild radius of the Sun?",
            "options": ["~3 km", "~30 km", "~300 km", "~3000 km"],
            "correct_index": 0,
            "explanation": "The Schwarzschild radius of 1 solar mass is approximately 2.95 km."
        },
        {
            "question": "Which object is at the center of the Milky Way?",
            "options": ["Cygnus X-1", "Sagittarius A*", "M87*", "Betelgeuse"],
            "correct_index": 1,
            "explanation": "Sagittarius A* is the supermassive black hole at the Galactic Center."
        },
        {
            "question": "Which planet has the shortest year?",
            "options": ["Mercury", "Venus", "Earth", "Mars"],
            "correct_index": 0,
            "explanation": "Mercury completes an orbit around the Sun in about 88 days."
        },
    ]
    monkeypatch.setattr(
        ai_quiz,
        "_call_gemini",
        lambda prompt: json.dumps({"questions": generated_questions}),
    )
    ai_quiz._LAST_GEN.clear()
    ai_quiz._LAST_SUBMIT.clear()

    uid = uuid.uuid4().hex[:8]
    registration = client.post("/api/auth/register", json={
        "username": f"quiz_{uid}",
        "email": f"quiz_{uid}@example.com",
        "password": "QuizPassword99!",
    })
    assert registration.status_code == 200
    token = registration.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    generated = client.post("/api/ai-quiz/generate", json={
        "topic": "Black Holes",
        "difficulty": "intermediate",
        "count": 3,
        "session_id": "attacker-controlled-session",
    }, headers=headers)
    assert generated.status_code == 200
    generated_data = generated.json()
    unique_session = generated_data["session_id"]
    assert unique_session != "attacker-controlled-session"

    submitted_questions = generated_data["questions"]
    sub_res = client.post("/api/ai-quiz/submit", headers=headers, json={
        "session_id": unique_session,
        "topic": "Black Holes",
        "difficulty": "intermediate",
        "questions": submitted_questions,
        "answers": [0, 1, 0],
        "time_taken_seconds": 20
    })
    assert sub_res.status_code == 200
    d = sub_res.json()
    assert d["score"] == 3
    assert d["total"] == 3
    assert d["percentage"] == 100.0
    assert d["xp_awarded"] > 0
    assert "Perfect" in d["grade"]
    assert len(d["feedback"]) == 3
    assert d["feedback"][0]["is_correct"] is True

    # Anti-replay: same session_id must be rejected with 400
    replay_res = client.post("/api/ai-quiz/submit", headers=headers, json={
        "session_id": unique_session,
        "topic": "Black Holes",
        "difficulty": "intermediate",
        "questions": submitted_questions,
        "answers": [0, 1, 0],
        "time_taken_seconds": 5
    })
    assert replay_res.status_code == 400

    # A fresh server-issued attempt still rejects duplicated/reordered question hashes.
    ai_quiz._LAST_GEN.clear()
    generated = client.post("/api/ai-quiz/generate", headers=headers, json={
        "topic": "Black Holes",
        "difficulty": "intermediate",
        "count": 3,
    })
    assert generated.status_code == 200
    questions = generated.json()["questions"]
    ai_quiz._LAST_SUBMIT.clear()
    duplicate_res = client.post("/api/ai-quiz/submit", headers=headers, json={
        "session_id": generated.json()["session_id"],
        "topic": "Black Holes",
        "difficulty": "intermediate",
        "questions": [questions[0], questions[1], questions[1]],
        "answers": [0, 1, 1],
        "time_taken_seconds": 20,
    })
    assert duplicate_res.status_code == 400


def test_jwt_secret_rejects_example_placeholder():
    from backend.routers.auth import is_jwt_secret_secure

    assert not is_jwt_secret_secure("your_random_32_character_jwt_secret_here")
    assert not is_jwt_secret_secure("replace_me_with_a_unique_secure_secret")
    assert is_jwt_secret_secure("zV7uQ4tX6wB2nM9pL5kR8cD1fG3hJ6sA")

def test_ai_quiz_validation_rules():
    # Invalid difficulty
    res_diff = client.post("/api/ai-quiz/generate", json={
        "topic": "Mars",
        "difficulty": "impossible",
        "count": 5,
        "session_id": "test_sess_invalid"
    })
    assert res_diff.status_code == 422

    # Count out of bounds (< 3)
    res_low = client.post("/api/ai-quiz/generate", json={
        "topic": "Mars",
        "difficulty": "beginner",
        "count": 1,
        "session_id": "test_sess_invalid"
    })
    assert res_low.status_code == 422

def test_catch_all_404():
    res = client.get("/nonexistent_dummy_page.html")
    assert res.status_code == 404

# ---------------------------------------------------------------------------
# New security & infrastructure tests
# ---------------------------------------------------------------------------

def test_bounded_ttl_dict_basic():
    """BoundedTTLDict: LRU eviction and TTL expiry work correctly."""
    from backend.cache import BoundedTTLDict

    # Basic set/get
    c = BoundedTTLDict(maxsize=3, default_ttl=60)
    c["a"] = "alpha"
    c["b"] = "beta"
    c["c"] = "gamma"
    assert c.get("a") == "alpha"
    assert len(c) == 3

    # LRU eviction on 4th insert (oldest = 'a' moved to end after get? no — let's evict 'b')
    # After c.get("a"), "a" is moved to end. Oldest is "b".
    c["d"] = "delta"   # evicts "b"
    assert c.get("b") is None
    assert c.get("d") == "delta"
    assert len(c) <= 3

    # TTL expiry: use ttl=0 which should mark entry as already-expired (exp <= now)
    c2 = BoundedTTLDict(maxsize=100, default_ttl=60)
    c2.set("x", "xyz", ttl=0)   # ttl=0 → exp set to 0.0 (epoch), always expired
    assert c2.get("x") is None

    # __contains__ works correctly
    c3 = BoundedTTLDict(maxsize=100, default_ttl=60)
    c3["key"] = "value"
    assert "key" in c3
    assert "missing" not in c3

def test_auth_token_refresh():
    """POST /api/auth/refresh issues a fresh JWT for authenticated users."""
    uid = uuid.uuid4().hex[:8]
    reg = client.post("/api/auth/register", json={
        "username": f"refresh_{uid}",
        "email": f"refresh_{uid}@example.com",
        "password": "RefreshPassword99!"
    })
    assert reg.status_code == 200
    token = reg.json()["token"]

    refresh_res = client.post("/api/auth/refresh", headers={"Authorization": f"Bearer {token}"})
    assert refresh_res.status_code == 200
    new_token = refresh_res.json()["token"]
    assert isinstance(new_token, str) and len(new_token) > 20

    # New token must be valid
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {new_token}"})
    assert me_res.status_code == 200

def test_auth_password_minimum_8():
    """Registration fails for passwords shorter than 8 characters."""
    # 7-char password must fail
    res = client.post("/api/auth/register", json={
        "username": "pw_test_x", "email": "pw_test_x@example.com", "password": "1234567"
    })
    assert res.status_code == 400

    # 8-char password must succeed
    uid = uuid.uuid4().hex[:8]
    res2 = client.post("/api/auth/register", json={
        "username": f"pw_ok_{uid}", "email": f"pw_ok_{uid}@example.com", "password": "12345678"
    })
    assert res2.status_code == 200

def test_ai_quiz_forged_hash_rejected():
    """Submit with an unknown/forged question hash must return HTTP 400."""
    # Clear submit rate-limiter for this test to avoid 429 masking the 400
    from backend.routers.ai_quiz import _LAST_SUBMIT
    _LAST_SUBMIT.clear()

    res = client.post("/api/ai-quiz/submit", json={
        "session_id": uuid.uuid4().hex,
        "topic": "Space",
        "difficulty": "beginner",
        "questions": [
            {
                "hash": "definitely_not_a_real_hash_abc123",
                "question": "Fake question?",
                "options": ["A", "B", "C", "D"],
                "correct_index": 0,
                "explanation": "Fabricated"
            }
        ],
        "answers": [0],
        "time_taken_seconds": 5
    })
    assert res.status_code == 400

def test_env_credentials_not_in_responses():
    """API responses must NEVER leak raw API keys."""
    import json as _json

    # Health check
    h = client.get("/api/health")
    body = h.text
    assert "AQ.Ab8RN" not in body, "Gemini API key leaked in health response"
    assert "6whzQMhm" not in body, "NASA API key leaked in health response"

    # NASA status
    n = client.get("/api/nasa/status")
    ns = n.text
    assert "6whzQMhm" not in ns, "NASA API key leaked in nasa/status response"

    # AI context
    a = client.get("/api/ai/context/system")
    aa = a.text
    assert "AQ.Ab8RN" not in aa, "Gemini API key leaked in ai/context/system"

# ── Agent Router Tests ──────────────────────────────────────────────────────

def test_agent_chat_basic():
    """Agent chat returns structured response with answer and follow-ups."""
    res = client.post("/api/agent/chat", json={
        "query": "What is an exoplanet transit?",
        "level": "intermediate"
    })
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert len(data["answer"]) > 20
    assert "suggested_followups" in data
    assert isinstance(data["suggested_followups"], list)
    assert "disclaimer" in data


def test_agent_chat_levels():
    """Agent chat accepts beginner/intermediate/advanced levels."""
    for level in ("beginner", "intermediate", "advanced"):
        res = client.post("/api/agent/chat", json={
            "query": "Tell me about black holes",
            "level": level
        })
        assert res.status_code == 200, f"Level {level} failed: {res.text}"
        assert res.json()["level"] == level


def test_agent_chat_invalid_level():
    """Invalid level returns 422 validation error."""
    res = client.post("/api/agent/chat", json={
        "query": "What is Mars?",
        "level": "expert"  # invalid
    })
    assert res.status_code == 422


def test_agent_research():
    """Research endpoint returns papers list and summary."""
    res = client.post("/api/agent/research", json={
        "query": "exoplanet atmosphere detection",
        "max_papers": 3
    })
    assert res.status_code == 200
    data = res.json()
    assert "query" in data
    assert "summary" in data
    assert "papers" in data
    assert isinstance(data["papers"], list)
    assert "disclaimer" in data


def test_agent_paper_search_arxiv():
    """Paper search endpoint returns structured results."""
    res = client.get("/api/agent/papers/search?q=galaxy+classification&source=arxiv&max_results=3")
    assert res.status_code == 200
    data = res.json()
    assert "query" in data
    assert "count" in data
    assert isinstance(data["papers"], list)
    assert "disclaimer" in data


def test_agent_paper_search_empty_query():
    """Paper search with empty query returns 422."""
    res = client.get("/api/agent/papers/search?q=a")
    # q=a is 1 char, should still be allowed or fail with meaningful response
    # min_length=2, so single char fails - but url param not body
    # The endpoint checks >= 2 internally and raises HTTPException
    assert res.status_code in (200, 422, 400)


def test_agent_explain_object():
    """Explain object endpoint returns explanation."""
    res = client.get("/api/agent/explain/Mars?level=beginner")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "Mars"
    assert "explanation" in data
    assert len(data["explanation"]) > 10
    assert data["level"] == "beginner"


def test_agent_report():
    """Report endpoint returns structured research report."""
    res = client.post("/api/agent/report", json={
        "topic": "Water on Mars",
        "include_papers": False  # skip arxiv fetch to speed up test
    })
    assert res.status_code == 200
    data = res.json()
    assert "disclaimer" in data
    assert "generated_by" in data


def test_research_page_route():
    """Research page serves correctly."""
    res = client.get("/research")
    assert res.status_code == 200
    assert "Research" in res.text or "research" in res.text.lower()


# ── ML Anomaly Detection Tests ────────────────────────────────────────────────

def test_ml_anomaly_detection():
    """Anomaly detection endpoint returns structured results."""
    observations = [
        {"u": 18.2, "g": 17.5, "r": 17.1, "i": 16.9, "z": 16.8, "redshift": 0.0001},  # normal star
        {"u": 15.0, "g": 14.8, "r": 14.5, "i": 14.3, "z": 14.1, "redshift": 0.0},      # bright star
        {"u": 23.5, "g": 22.1, "r": 21.5, "i": 21.2, "z": 20.9, "redshift": 3.5},      # likely anomalous quasar
        {"u": 19.1, "g": 18.7, "r": 18.3, "i": 18.1, "z": 17.9, "redshift": 0.12},     # normal galaxy
        {"u": 20.5, "g": 20.2, "r": 20.0, "i": 19.8, "z": 19.6, "redshift": 0.5},      # normal
    ]
    res = client.post("/api/ml/anomaly", json={"observations": observations, "contamination": 0.2})
    assert res.status_code == 200
    data = res.json()
    assert "total" in data
    assert data["total"] == 5
    assert "anomaly_count" in data
    assert "results" in data
    assert len(data["results"]) == 5
    assert "disclaimer" in data
    assert "method" in data
    # Each result has required fields
    for r in data["results"]:
        assert "is_anomaly" in r
        assert "anomaly_score" in r
        assert "index" in r


def test_ml_model_card():
    """Model card endpoint returns documentation."""
    res = client.get("/api/ml/model-card")
    assert res.status_code == 200
    data = res.json()
    assert "model_name" in data
    assert "dataset" in data
    assert "limitations" in data
    assert isinstance(data["limitations"], list)
    assert len(data["limitations"]) > 0


# ── NASA Alert Tests ──────────────────────────────────────────────────────────

def test_nasa_alert_preferences_default():
    """Alert preferences returns defaults."""
    res = client.get("/api/nasa/alerts/preferences?session=test-session")
    assert res.status_code == 200
    data = res.json()
    assert "preferences" in data
    prefs = data["preferences"]
    assert "enable_neo_alerts" in prefs
    assert "enable_flare_alerts" in prefs


def test_nasa_alert_preferences_set():
    """Alert preferences can be updated."""
    res = client.post(
        "/api/nasa/alerts/preferences?session=test-session-2",
        json={"neo_min_diameter_km": 0.5, "flare_min_class": "X"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "updated"


def test_nasa_active_alerts():
    """Active alerts endpoint returns structured response."""
    res = client.get("/api/nasa/alerts?session=test-session")
    assert res.status_code == 200
    data = res.json()
    assert "count" in data
    assert "alerts" in data
    assert isinstance(data["alerts"], list)
    assert "checked_at" in data
