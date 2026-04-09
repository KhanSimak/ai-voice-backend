import os
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import Request, FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import engine, SessionLocal, Base
from models import Doctor, Appointment, CallSession

from redis_client import get_history, append_message, r
from rag import create_vectorstore, get_llm, ask_question

from dotenv import load_dotenv

# ---------------- ENV ---------------- #
load_dotenv(dotenv_path="C:/ai voice/.env")
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

# ---------------- SCHEMAS ---------------- #
class ChatRequest(BaseModel):
    text: str

class QuestionRequest(BaseModel):
    text: Optional[str] = None
    question: Optional[str] = None

# ---------------- RAG ---------------- #
def ask_question_for_voice(vectorstore, llm, query, history=None):
    try:
        if not query:
            return "I didn't understand that."

        if history is None:
            history = []

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs]) or "No relevant info"

        chat_history = ""
        for h in history[-4:]:
            chat_history += f"{h.get('role')}: {h.get('content')}\n"

        prompt = f"""
You are a real-time voice assistant.

Rules:
- Short answers (2-3 lines)
- Use context only

History:
{chat_history}

Context:
{context}

User:
{query}

Answer:
"""

        res = llm.invoke(prompt)
        return res.content if hasattr(res, "content") else str(res)

    except Exception as e:
        print("RAG ERROR:", e)
        return "Something went wrong."

# ---------------- ROUTES ---------------- #

@app.get("/")
def root():
    return {"status": "ok"}

# ✅ SAFE CHAT ENDPOINT
@app.post("/chat")
async def chat(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()
        print("📩 VAPI DATA:", data)
    except Exception as e:
        print("❌ JSON ERROR:", e)
        return {"output": "Error reading input"}

    messages = data.get("messages", [])

    if not messages:
        print("⚠️ No messages received")
        return {"output": "Hello, how can I help you?"}

    last_message = messages[-1].get("content", "").lower()
    print("🧠 USER SAID:", last_message)

    if not last_message:
        return {"output": "Can you repeat that?"}

    # Doctor logic
    if "doctor" in last_message:
        doctors = db.query(Doctor).all()

        if not doctors:
            return {"output": "No doctors available right now"}

        response = "\n".join([
            f"{d.name} - {d.specialization}"
            for d in doctors
        ])

        print("✅ RESPONSE:", response)
        return {"output": response}

    # Fallback
    response = ask_question_for_voice(
        app.state.vectorstore,
        app.state.llm,
        last_message
    )

    print("🤖 AI RESPONSE:", response)

    return {"output": response or "Sorry, I didn’t understand"}

# ✅ RETELL WEBHOOK (FIXEDsss
@app.post("/retell-webhook")
async def retell_webhook(request: Request, db: Session = Depends(get_db)):

    # ✅ Handle empty requests (Retell sends them)
    body = await request.body()
    if not body:
        return {"status": "ignored"}

    # ✅ Safe JSON parsing
    try:
        data = await request.json()
    except Exception as e:
        print("❌ JSON ERROR:", e)
        return {"status": "ignored"}

    print("🔥 FULL DATA:", data)

    # ✅ Only handle analyzed calls
    if data.get("event") == "call_analyzed":
        call = data.get("call", {})

        call_id = call.get("call_id")
        transcript = call.get("transcript", "")
        summary = call.get("call_analysis", {}).get("call_summary", "")

        print("📞 Call ID:", call_id)

        # ✅ STEP 1: Check if session exists
        session = db.query(CallSession).filter_by(call_id=call_id).first()

        if not session:
            # ✅ Create new session
            session = CallSession(
                call_id=call_id,
                status="completed",
                conversation_history=[]
            )
            db.add(session)
            db.commit()
            db.refresh(session)

        # ✅ STEP 2: Save transcript
        try:
            history = session.conversation_history or []

            history.append({
                "transcript": transcript,
                "summary": summary,
                "timestamp": datetime.utcnow().isoformat()
            })

            session.conversation_history = history
            session.status = "completed"

            db.commit()

            print("✅ Conversation saved")

        except Exception as db_error:
            print("❌ DB ERROR:", db_error)
            db.rollback()

        # ✅ STEP 3: (Optional) detect booking
        try:
            if "appointment is confirmed" in transcript.lower():
                session.booking_stage = "confirmed"
                session.appointment_time = "5-6 PM"  # later dynamic
                db.commit()

                print("✅ Booking updated")

        except Exception as e:
            print("❌ Booking ERROR:", e)
            db.rollback()

    return {"status": "ok"}
@app.post("/ask")
async def ask(body: QuestionRequest):
    query = body.text or body.question

    if not query:
        return {"answer": "No input provided"}

    answer = ask_question(
        app.state.vectorstore,
        app.state.llm,
        query
    )

    return {"answer": answer}

@app.get("/test-db")
def test_db(db: Session = Depends(get_db)):
    doctors = db.query(Doctor).all()
    return {"count": len(doctors)}

# ---------------- DEBUG ---------------- #

@app.get("/test-all")
def test_all(db: Session = Depends(get_db)):
    result = {}

    try:
        result["db"] = f"OK ({db.query(Doctor).count()} doctors)"
    except Exception as e:
        result["db"] = str(e)

    try:
        append_message("test", "user", "hello")
        result["redis"] = f"OK ({len(get_history('test'))})"
    except Exception as e:
        result["redis"] = str(e)

    try:
        ask_question(app.state.vectorstore, app.state.llm, "hello")
        result["ai"] = "OK"
    except Exception as e:
        result["ai"] = str(e)

    return result

# ---------------- RUN -------------43--- #
if __name__ == "__main__":
    uvicorn.run("mains:app", host="0.0.0.0", port=8000)
