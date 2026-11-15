import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from backend.database import Base

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    xp = Column(Integer, default=0)
    level = Column(Integer, default=1)
    created_at = Column(DateTime, default=utc_now)

    scores = relationship("QuizScore", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")
    history = relationship("ExplorationHistory", back_populates="user")
    ai_quiz_attempts = relationship("AIQuizAttempt", back_populates="user")

class QuizScore(Base):
    __tablename__ = "quiz_scores"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    category = Column(String(50), default="Astrophysics")
    score = Column(Integer, nullable=False)
    total = Column(Integer, nullable=False)
    xp_awarded = Column(Integer, default=0)
    created_at = Column(DateTime, default=utc_now)

    user = relationship("User", back_populates="scores")

class AIQuizAttempt(Base):
    __tablename__ = "ai_quiz_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    session_id = Column(String(64), index=True, nullable=False)
    topic = Column(String(100), index=True, nullable=False)
    difficulty = Column(String(20), index=True, nullable=False)
    question_count = Column(Integer, nullable=False)
    score = Column(Integer, nullable=False)
    total = Column(Integer, nullable=False)
    percentage = Column(Float, nullable=False)
    xp_awarded = Column(Integer, default=0)
    time_taken_seconds = Column(Integer, default=0)
    review_data = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    user = relationship("User", back_populates="ai_quiz_attempts")

class AIQuestionRecord(Base):
    __tablename__ = "ai_question_records"

    id = Column(Integer, primary_key=True, index=True)
    question_hash = Column(String(64), unique=True, index=True, nullable=False)
    topic = Column(String(100), index=True, nullable=False)
    difficulty = Column(String(20), index=True, nullable=False)
    question = Column(Text, nullable=False)
    options = Column(Text, nullable=False)  # JSON-encoded array of 4 options
    correct_index = Column(Integer, nullable=False)
    explanation = Column(Text, nullable=False)
    times_served = Column(Integer, default=1)
    created_at = Column(DateTime, default=utc_now)

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    source = Column(String(50), default="rag-gemini")
    model_used = Column(String(50), default="gemini-2.5-flash")
    created_at = Column(DateTime, default=utc_now)

    user = relationship("User", back_populates="conversations")

class ExplorationHistory(Base):
    __tablename__ = "exploration_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    target_body = Column(String(50), nullable=False)
    action = Column(String(50), default="view")
    created_at = Column(DateTime, default=utc_now)


    user = relationship("User", back_populates="history")
