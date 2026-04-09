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
from rag import create_vectorstore, get_llm

from dotenv import load_dotenv

# ---------------- ENV ---------------- #
load_dotenv()
load_dotenv(override=True)

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

# ---------------- INTENT DETECTION ---------------- #
def detect_intent(message: str):
    msg = message.lower()

    if any(x in msg for x in ["bye", "thank", "thanks"]):
        return "exit"
    if "book" in msg or "appointment" in msg:
        return "booking"
    if "doctor" in msg or "available" in msg:
        return "doctor_query"

    return "unknown"

# ---------------- DOCTOR HANDLER ---------------- #
def get_doctors_response(db: Session):
    doctors = db.query(Doctor).all()

    if not doctors:
        return "No doctors available today."

    response = "Here are available doctors:\n"
    for d in doctors:
        response += f"{d.name}, {d.specialization}. "

    return response

# ---------------- BOOKING HANDLER ---------------- #
def handle_booking(message, history):
    return "Sure, please tell me the doctor's name you want to book."

# ---------------- RAG ---------------- #
def ask_question_for_voice(vectorstore, llm, query, history=None):
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        context = "\n".join([d.page_content for d in docs]) or ""

        prompt = f"""
You are a smart medical voice assistant.

Answer shortly (max 2 lines).
Use context if relevant.

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

# ---------------- CHAT ---------------- #
@app.post("/chat")
async def chat(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()
        print("📩 VAPI:", data)

        messages = data.get("message", {}).get("artifact", {}).get("messages", [])

        if not messages:
            return JSONResponse({"response": "Sorry, I didn’t understand that."})

        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("message")
                break

        if not last_user_message:
            return JSONResponse({"response": "Please repeat."})

        print("👤 USER:", last_user_message)

        # Redis memory
        call_id = data.get("message", {}).get("call", {}).get("id", "default")
        history = get_history(call_id)

        append_message(call_id, "user", last_user_message)

        # Intent detection
        intent = detect_intent(last_user_message)

        # Intent routing
        if intent == "exit":
            reply = "You're welcome! Have a great day 😊"

        elif intent == "doctor_query":
            reply = get_doctors_response(db)

        elif intent == "booking":
            reply = handle_booking(last_user_message, history)

        else:
            # RAG fallback
            reply = ask_question_for_voice(
                app.state.vectorstore,
                app.state.llm,
                last_user_message,
                history
            )

            if not reply:
                reply = "Could you please clarify?"

        append_message(call_id, "assistant", reply)

        print("🤖 AI:", reply)

        return JSONResponse({"response": reply})

    except Exception as e:
        print("❌ ERROR:", str(e))
        return JSONResponse({"response": "Something went wrong"})

# ---------------- RETELL WEBHOOK ---------------- #
@app.post("/retell-webhook")
async def retell_webhook(request: Request, db: Session = Depends(get_db)):
    body = await request.body()
    if not body:
        return {"status": "ignored"}

    try:
        data = await request.json()
    except:
        return {"status": "ignored"}

    if data.get("event") == "call_analyzed":
        call = data.get("call", {})
        call_id = call.get("id")
        transcript = call.get("transcript", "")
        summary = call.get("call_analysis", {}).get("call_summary", "")

        session = db.query(CallSession).filter_by(call_id=call_id).first()

        if not session:
            session = CallSession(call_id=call_id, status="completed", conversation_history=[])
            db.add(session)
            db.commit()
            db.refresh(session)

        history = session.conversation_history or []

        history.append({
            "transcript": transcript,
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        })

        session.conversation_history = history
        session.status = "completed"
        db.commit()

    return {"status": "ok"}

# ---------------- TEST ---------------- #
@app.get("/test-db")
def test_db(db: Session = Depends(get_db)):
    return {"doctors": db.query(Doctor).count()}

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    uvicorn.run("mains:app", host="0.0.0.0", port=8000)