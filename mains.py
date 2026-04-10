import os
import uvicorn
import logging
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Any

from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy.orm import Session

from database import engine, SessionLocal, Base
from models import Doctor, Appointment, CallSession

from redis_client import get_history, append_message, r
from rag import create_vectorstore, get_llm, ask_question

from dotenv import load_dotenv

# ---------------- ENV ---------------- #
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is not set")

# ---------------- DB ---------------- #
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_doctors():
    db = SessionLocal()

    if db.query(Doctor).count() == 0:
        doctors = [
            Doctor(name="Dr. Arjun Mehta", specialization="Cardiologist",
                   available_days="Monday, Wednesday, Friday", available_time="10:00 AM - 2:00 PM"),
            Doctor(name="Dr. Priya Sharma", specialization="Pediatrician",
                   available_days="Tuesday, Thursday, Saturday", available_time="11:00 AM - 4:00 PM"),
            Doctor(name="Dr. Rahul Desai", specialization="Orthopedic",
                   available_days="Monday - Saturday", available_time="5:00 PM - 8:00 PM"),
            Doctor(name="Dr. Sana Khan", specialization="Dermatologist",
                   available_days="Wednesday - Sunday", available_time="12:00 PM - 6:00 PM"),
        ]
        db.add_all(doctors)
        db.commit()

    db.close()


Base.metadata.create_all(bind=engine)
seed_doctors()

# ---------------- LOGGING ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- STARTUP ---------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    app.state.vectorstore = create_vectorstore()
    app.state.llm = get_llm()

    yield

    logger.info("Shutting down...")

# ---------------- APP ---------------- #
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HELPERS ---------------- #

def extract_user_message(data: dict):
    """
    Works with VAPI + Make + raw webhook payloads
    """
    try:
        # VAPI format
        messages = data.get("message", {}).get("messages", [])
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("message") or m.get("content")
    except:
        pass

    # fallback formats
    return (
        data.get("message")
        or data.get("input")
        or data.get("text")
    )


def safe_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("message") or x.get("content") or ""
    return ""


def generate_followup_ai(message):
    msg = safe_text(message).lower()

    if not msg:
        return "I didn't understand that."

    if any(x in msg for x in ["thank", "thanks", "bye"]):
        return "You're welcome! Have a great day 😊"

    if "doctor" in msg or "available" in msg:
        return "We have doctors available. Which specialization do you need?"

    if "book" in msg or "appointment" in msg:
        return "Sure, which doctor would you like to book?"

    return "Could you please clarify?"
# ---------------- CHAT ENDPOINT ---------------- #

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.body()

        if not body:
            return JSONResponse({"result": "empty request"})

        try:
            data = json.loads(body)
        except Exception:
            return JSONResponse({"result": "invalid json"})

        user_text = extract_user_message(data)

        if not user_text:
            return JSONResponse({"result": "no user message detected"})

        print("USER:", user_text)

        # SIMPLE AI LOGIC FIRST (stable mode)
        reply = generate_followup_ai(user_text)

        return JSONResponse({
            "result": reply
        })

    except Exception as e:
        print("CHAT ERROR:", e)
        return JSONResponse({
            "result": "server error"
        })

# ---------------- RETELL WEBHOOK ---------------- #

@app.post("/retell-webhook")
async def retell_webhook(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()

        if data.get("event") != "call_analyzed":
            return {"status": "ignored"}

        call = data.get("call", {})
        call_id = call.get("id")
        transcript = call.get("transcript", "")

        session = db.query(CallSession).filter_by(call_id=call_id).first()

        if not session:
            session = CallSession(
                call_id=call_id,
                status="completed",
                conversation_history=[]
            )
            db.add(session)
            db.commit()
            db.refresh(session)

        history = session.conversation_history or []

        history.append({
            "transcript": transcript,
            "timestamp": datetime.utcnow().isoformat()
        })

        session.conversation_history = history
        db.commit()

        return {"status": "saved"}

    except Exception as e:
        print("❌ ERROR:", e)
        return {"status": "error"}

# ---------------- TEST ---------------- #

@app.get("/test-db")
def test_db(db: Session = Depends(get_db)):
    return {"count": db.query(Doctor).count()}

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    uvicorn.run("mains:app", host="0.0.0.0", port=8000)