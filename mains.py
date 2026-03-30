import os
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import Request
from database import engine, SessionLocal, Base
from models import Doctor, Patient, Appointment
from models import CallSession
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from fastapi import Depends
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Appointment
from schemas import AppointmentCreate

from rag import create_vectorstore, get_llm, ask_question





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
            Doctor(
                name="Dr. Arjun Mehta",
                specialization="Cardiologist",
                available_days="Monday, Wednesday, Friday",
                available_time="10:00 AM - 2:00 PM"
            ),
            Doctor(
                name="Dr. Priya Sharma",
                specialization="Pediatrician",
                available_days="Tuesday, Thursday, Saturday",
                available_time="11:00 AM - 4:00 PM"
            ),
            Doctor(
                name="Dr. Rahul Desai",
                specialization="Orthopedic",
                available_days="Monday - Saturday",
                available_time="5:00 PM - 8:00 PM"
            ),
            Doctor(
                name="Dr. Sana Khan",
                specialization="Dermatologist",
                available_days="Wednesday - Sunday",
                available_time="12:00 PM - 6:00 PM"
            )
        ]

        db.add_all(doctors)
        db.commit()

    db.close()
Base.metadata.create_all(bind=engine)
seed_doctors()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initializing vectorstore and LLM...")
    try:
        app.state.vectorstore = create_vectorstore()
        app.state.llm = get_llm()
        logger.info("Startup complete.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG + Retell AI API",
    description="PDF knowledge base QA and Retell AI voice webhook.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# ── Schemas — existing /ask endpoint ─────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str


# ── Schemas — Retell webhook ──────────────────────────────────────────────────

class RetellResponse(BaseModel):
    response: str


# ── Voice-optimised RAG answer ────────────────────────────────────────────────

def ask_question_for_voice(vectorstore, llm, question: str, history: list) -> str:
    """
    Same FAISS + LLM pipeline as /ask, tuned for voice:
    - Shorter answers (2-3 sentences)
    - No markdown or bullet points
    - Conversation history included for multi-turn context
    """

    question = question.strip()
    if not question:
        return "I didn't catch that. Could you please repeat your question?"

    docs = vectorstore.similarity_search(question, k=3)
    print("QUESTION:", question)
    print("DOCS FOUND:", len(docs))

    if docs:
        context = "\n\n".join(
            f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content.strip()}"
            for doc in docs
        )

        system_content = (
    "You are a helpful, friendly voice assistant for our clinic. "
    "You are speaking on a phone call, so keep responses short and natural, 2 to 3 sentences maximum. "
    "Use the information from the knowledge base below to answer the caller’s question clearly. "
    "Be specific when possible, such as naming doctors, services, or availability. "
    "Do not mention the knowledge base or say you are reading from context. "
    "If the information truly is not available, say you can connect them with the team."

    f"\n\nKNOWLEDGE BASE:\n{context}"
)
    else:
        system_content = (
            "You are a helpful, friendly voice assistant. "
            "You are speaking aloud on a phone call — keep answers SHORT (2-3 sentences max). "
            "Do NOT use bullet points, numbered lists, or markdown. "
            "If you don't know something, offer to connect the caller with the team."
        )

    messages = [SystemMessage(content=system_content)]

    # Include last 6 turns of conversation history
    for msg in history[-6:]:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Add current user question
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)

    if hasattr(response, "content"):
        return response.content.strip()

    return str(response).strip()

# ── Existing endpoint: /ask ───────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ok",
        "endpoints": {
            "POST /ask": "Text Q&A over PDF knowledge base",
            "POST /retell-webhook": "Retell AI voice call webhook",
            "GET /health": "Health check",
        },
    }


@app.post("/book-appointment")
def book_appointment(data: AppointmentCreate, db: Session = Depends(get_db)):

    # check if doctor already booked at that time
    existing = db.query(Appointment).filter(
    Appointment.doctor_id == data.doctor_id,
    Appointment.appointment_time == data.appointment_time,
).first()

    if existing:
        return {
            "message": "This time slot is already booked. Please choose another time."
        }

    appointment = Appointment(
    doctor_id=data.doctor_id,
    patient_id=data.patient_id,
    appointment_time=data.appointment_time
)

    db.add(appointment)
    db.commit()
    db.refresh(appointment)

    return {
        "message": "Appointment booked successfully",
        "appointment_id": appointment.id
    }

def get_doctor_by_name(db, name: str):
    doctors = db.query(Doctor).all()

    name = name.lower()

    for doctor in doctors:
        if doctor.name.lower() in name or name in doctor.name.lower():
            return doctor

    return None


def handle_booking(db, call_id, user_text):

    session = db.query(CallSession).filter_by(call_id=call_id).first()

    print("HANDLE BOOKING CALLED")

    if not session:
        session = CallSession(
            call_id=call_id,
            booking_stage="ask_doctor",
            status="in_progress"
        )
        db.add(session)
        db.commit()

        print("NEW SESSION CREATED")
        return "Sure. Which doctor would you like to see?"

    if session.booking_stage == "ask_doctor":

     print("STEP: ASK DOCTOR")

     doctor = get_doctor_by_name(db, user_text)

    # ❌ if not found
     if not doctor:
      return "Sorry, I couldn't find that doctor. Please say the full name."

    # ✅ if found
     session.doctor_name = doctor.name
     session.doctor_id = doctor.id
     session.booking_stage = "ask_date"
     db.commit()
     return f"Okay. What date would you like to see {doctor.name}?"
    


    if session.booking_stage == "ask_date":
        print("STEP: ASK DATE")

        session.appointment_date = user_text
        session.booking_stage = "ask_time"
        db.commit()

        return "What time would you prefer?"

    if session.booking_stage == "ask_time":
        print("STEP: ASK TIME")

        session.appointment_time = user_text
        session.booking_stage = "ask_name"
        db.commit()

        return "May I have your name please?"

    if session.booking_stage == "ask_name":
        print("STEP: FINAL BOOKING")

        session.patient_name = user_text
        session.booking_stage = "complete"
        session.status = "booked"
        db.commit()

        appointment = Appointment(
            doctor_id=session.doctor_id,
            patient_id=1,
            appointment_time=session.appointment_time
        )

        db.add(appointment)
        db.commit()

        save_lead(db, session)

        return (
            f"Great, {user_text}! You're all set. "
            f"I’ve booked your appointment with {session.doctor_name} on "
            f"{session.appointment_date} at {session.appointment_time}. "
            f"We look forward to seeing you!"
        )

    return "Let me help you with that. Could you repeat?"

@app.get("/leads")
def get_leads(db: Session = Depends(get_db)):

    sessions = db.query(CallSession).filter(
        CallSession.status == "booked"
    ).all()

    leads = []

    for s in sessions:
        leads.append({
            "name": s.patient_name,
            "doctor": s.doctor_name,
            "date": s.appointment_date,
            "time": s.appointment_time
        })

    return leads



@app.get("/health")
async def health():
    ready = (
        hasattr(app.state, "vectorstore") and app.state.vectorstore is not None
        and hasattr(app.state, "llm") and app.state.llm is not None
    )
    return {"status": "ready" if ready else "initializing"}
@app.get("/debug")
async def debug():
    return {
        "vectorstore_loaded": hasattr(app.state, "vectorstore") and app.state.vectorstore is not None
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask(body: QuestionRequest):
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if not hasattr(app.state, "vectorstore") or app.state.vectorstore is None:
        raise HTTPException(status_code=503, detail="Service not ready yet. Please retry shortly.")
    try:
        answer = ask_question(app.state.vectorstore, app.state.llm, body.question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error in /ask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


def save_lead(db, session):
    lead_data = {
        "name": session.patient_name,
        "doctor": session.doctor_name,
        "date": session.appointment_date,
        "time": session.appointment_time,
    }

    print("LEAD SAVED:", lead_data)

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):

    total_calls = db.query(CallSession).count()

    booked_calls = db.query(CallSession).filter(
        CallSession.status == "booked"
    ).count()

    conversion_rate = 0

    if total_calls > 0:
        conversion_rate = (booked_calls / total_calls) * 100

    return {
        "total_calls": total_calls,
        "booked_calls": booked_calls,
        "conversion_rate": f"{round(conversion_rate, 2)}%"
    }



@app.post("/retell-webhook")
async def retell_webhook(request: Request):

    try:
        body = await request.json()
    except:
        return {"response": "Invalid request. No JSON received."}

    print("BODY:", body)

    call = body.get("call", {})
    call_id = call.get("call_id")

    # ✅ THIS IS THE FIX (YOU WERE MISSING THIS)
    transcript = call.get("transcript_object", [])

    latest_user_message = ""

    for msg in reversed(transcript):
        if msg.get("role") == "user":
            latest_user_message = msg.get("content", "").strip()
            break

    if not latest_user_message:
        return {"response": "Sorry, could you repeat that?"}

    db = SessionLocal()

    def is_booking_intent(text: str):
        keywords = ["appointment", "book", "visit", "see doctor", "checkup"]
        return any(word in text.lower() for word in keywords)

    if is_booking_intent(latest_user_message):
        return {"response": handle_booking(db, call_id, latest_user_message)}

    session = db.query(CallSession).filter_by(call_id=call_id).first()

    if session and session.booking_stage != "complete":
        return {"response": handle_booking(db, call_id, latest_user_message)}

    return {"response": "How can I help you today?"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("mains:app", host="0.0.0.0", port=port, reload=False)

