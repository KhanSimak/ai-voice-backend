import os
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

from fastapi import Request, FastAPI, Depends
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

# ---------------- SCHEMAS ---------------- #
class QuestionRequest(BaseModel):
    text: Optional[str] = None
    question: Optional[str] = None

# ---------------- SIMPLE AI ---------------- #
def generate_followup_ai(message: str):
    msg = message.lower()

    if any(x in msg for x in ["thank", "thanks", "bye"]):
        return "You're welcome! Have a great day 😊"

    if "doctor" in msg or "available" in msg:
        return "We have doctors available today. Which specialization do you need?"

    if "book" in msg or "appointment" in msg:
        return "Sure, which doctor would you like to book?"

    return "Could you please clarify?"

# ---------------- RAG ---------------- #
def ask_question_for_voice(vectorstore, llm, query, history=None):
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs]) or "No relevant info"

        prompt = f"""
You are a medical voice assistant.
Answer shortly (2 lines max).

Context:
{context}

User:
{query}
"""

        res = llm.invoke(prompt)
        return res.content if hasattr(res, "content") else str(res)

    except Exception as e:
        print("RAG ERROR:", e)
        return None

# ---------------- CHAT ---------------#

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.body()

        # If empty request
        if not body:
            return {"reply": "No input received"}

        text = body.decode("utf-8")

        # Try JSON parsing safely
        try:
            data = await request.json()
            user_message = data.get("message", "")
        except:
            user_message = text  # fallback raw text

    except Exception as e:
        return {"reply": f"Error: {str(e)}"}

    print("USER:", user_message)

    return {
        "reply": f"You said: {user_message}"
    }

def get_doctors_from_db(db):
    doctors = db.query(Doctor).all()
    return "\n".join([f"{d.name} - {d.specialization}" for d in doctors])

def clean_user_input(text: str) -> str:
    text = text.lower().strip()

    # fix common speech mistakes
    replacements = {
        "team doctors": "doctors",
        "doctor name": "doctors",
        "name doctor": "doctors",
        "available doctor": "doctors",
        "octavoids": "orthopedics",
        "steelock": "doctor",
    }

    for k, v in replacements.items():
        if k in text:
            return v

    return text


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