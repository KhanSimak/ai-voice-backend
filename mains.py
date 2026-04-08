import os
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import Request, FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from twilio.rest import Client

from database import engine, SessionLocal, Base
from models import Doctor, Patient, Appointment, CallSession
from schemas import AppointmentCreate
from redis_client import get_history, append_message
from rag import create_vectorstore, get_llm, ask_question
from redis_client import r
from dotenv import load_dotenv
# ---------------- DB ---------------- #
load_dotenv(dotenv_path="C:/ai voice/.env")
load_dotenv(override=True)

import os




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

# ---------------- Logging ---------------- #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Startup ---------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    from dotenv import load_dotenv
    load_dotenv()

    logger.info("Starting up...")

    app.state.vectorstore = create_vectorstore()
    app.state.llm = get_llm()

    yield

    logger.info("Shutting down...")

# ---------------- App ---------------- #

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Schemas ---------------- #

class QuestionRequest(BaseModel):
    question: str


# ---------------- Helpers ---------------- #

def get_doctor_by_name(db, name: str):
    doctors = db.query(Doctor).all()
    name = name.lower()

    for doctor in doctors:
        if doctor.name.lower() in name or name in doctor.name.lower():
            return doctor
    return None


def is_booking_intent(text: str):
    keywords = ["appointment", "book", "visit", "see doctor", "checkup"]
    return any(k in text.lower() for k in keywords)



def notify_human(call_id, message):
    try:
        client = Client(
            os.getenv("TWILIO_SID"),
            os.getenv("TWILIO_TOKEN")
        )

        client.messages.create(
            body=f"🚨 Human needed!\nCall ID: {call_id}\nUser said: {message}",
            from_="whatsapp:+14155238886",
            to=os.getenv("HUMAN_NUMBER")
        )

    except Exception as e:
        print("Twilio error:", e)


# ---------------- Booking Flow ---------------- #


@app.get("/test-db")
def test_db(db: Session = Depends(get_db)):
    try:
        doctors = db.query(Doctor).all()
        appointments = db.query(Appointment).all()
        sessions = db.query(CallSession).all()

        return {
            "status": "success",
            "doctors_count": len(doctors),
            "appointments_count": len(appointments),
            "call_sessions_count": len(sessions)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    

@app.get("/test-redis")
def test_redis():
    try:
        test_call_id = "debug123"

        # write test data
        append_message(test_call_id, "user", "hello")
        append_message(test_call_id, "assistant", "hi there")

        # read it back
        history = get_history(test_call_id)

        return {
            "status": "success",
            "history": history
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    

@app.get("/clear-redis/{call_id}")
def clear_redis(call_id: str):
    try:
        r.delete(call_id)
        return {"status": "cleared", "call_id": call_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/doctors")
def get_doctors(db: Session = Depends(get_db)):
    doctors = db.query(Doctor).all()

    return [
        {
            "id": d.id,
            "name": d.name,
            "specialization": d.specialization,
            "days": d.available_days,
            "time": d.available_time
        }
        for d in doctors
    ]


@app.get("/appointments")
def get_appointments(db: Session = Depends(get_db)):
    appts = db.query(Appointment).all()

    return [
        {
            "id": a.id,
            "doctor_id": a.doctor_id,
            "time": a.appointment_time
        }
        for a in appts
    ]


@app.get("/sessions")
def get_sessions(db: Session = Depends(get_db)):
    sessions = db.query(CallSession).all()

    return [
        {
            "call_id": s.call_id,
            "status": s.status,
            "doctor": s.doctor_name,
            "stage": s.booking_stage
        }
        for s in sessions
    ]

def handle_booking(db, call_id, user_text):

    session = db.query(CallSession).filter_by(call_id=call_id).first()

    if not session:
        session = CallSession(
            call_id=call_id,
            booking_stage="ask_doctor",
            status="in_progress"
        )
        db.add(session)
        db.commit()
        return "Sure, I can help with that. Which doctor would you like to see?"

    if session.booking_stage == "ask_doctor":
        doctor = get_doctor_by_name(db, user_text)

        if not doctor:
            return "Sorry, I couldn't find that doctor. Could you say the full name again?"

        session.doctor_name = doctor.name
        session.doctor_id = doctor.id
        session.booking_stage = "ask_date"
        db.commit()

        return f"Got it. What date would you like to see {doctor.name}?"

    if session.booking_stage == "ask_date":
        session.appointment_date = user_text
        session.booking_stage = "ask_time"
        db.commit()
        return "And what time works best for you?"

    if session.booking_stage == "ask_time":
        session.appointment_time = user_text
        session.booking_stage = "ask_name"
        db.commit()
        return "May I have your name, please?"

    if session.booking_stage == "ask_name":

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

        # 🔥 WHATSAPP
        send_whatsapp_reminder(
            name=session.patient_name,
            doctor=session.doctor_name,
            date=session.appointment_date,
            time=session.appointment_time
        )

        return (
            f"Great {user_text}! Your appointment with {session.doctor_name} "
            f"is confirmed on {session.appointment_date} at {session.appointment_time}. "
            f"You will receive a WhatsApp confirmation shortly."
        )





def send_whatsapp_reminder(name, doctor, date, time):
    try:
        from twilio.rest import Client
        import os

        client = Client(
            os.getenv("TWILIO_SID"),
            os.getenv("TWILIO_TOKEN")
        )

        message = f"""
Hello {name},

Your appointment is confirmed ✅

Doctor: {doctor}
Date: {date}
Time: {time}

Please arrive 10 minutes early.
        """

        client.messages.create(
            body=message,
            from_="whatsapp:+14155238886",
            to=os.getenv("PATIENT_PHONE_NUMBER")
        )

    except Exception as e:
        print("WhatsApp Error:", e)

# ---------------- RAG ---------------- #

def ask_question_for_voice(vectorstore, llm, query, history=None):
    """
    Voice-optimized RAG function for PDF QA (low latency, short answers)
    """

    try:
        if history is None:
            history = []

        # -----------------------------
        # 1. RETRIEVE CONTEXT FROM PDF
        # -----------------------------
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        if not context.strip():
            context = "No relevant context found in the document."

        # -----------------------------
        # 2. BUILD CONVERSATION HISTORY (SHORT)
        # -----------------------------
        chat_history = ""
        for h in history[-4:]:  # keep last 4 turns only
            role = h.get("role")
            content = h.get("content")
            chat_history += f"{role}: {content}\n"

        # -----------------------------
        # 3. SYSTEM PROMPT (VOICE OPTIMIZED)
        # -----------------------------
        system_prompt = """
You are a real-time voice assistant.

Rules:
- Answer ONLY using the given context
- If answer is not in context, say: "I don't have that information in the document"
- Keep response VERY short (1-3 sentences max)
- Speak naturally like a human assistant
- Do NOT explain your thinking
"""

        # -----------------------------
        # 4. FINAL PROMPT
        # -----------------------------
        prompt = f"""
{system_prompt}

CHAT HISTORY:
{chat_history}

CONTEXT FROM PDF:
{context}

USER QUESTION:
{query}

Answer:
"""

        # -----------------------------
        # 5. CALL LLM
        # -----------------------------
        response = llm.invoke(prompt)

        # if LangChain response object
        if hasattr(response, "content"):
            response = response.content

        return response.strip()

    except Exception as e:
        print("RAG error:", e)
        return "Sorry, I had trouble finding that information."

# ---------------- Routes ---------------- #

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/ask")
async def ask(body: QuestionRequest):
    answer = ask_question(app.state.vectorstore, app.state.llm, body.question)
    return {"answer": answer}
   


@app.post("/retell-webhook")
async def retell_webhook(request: Request):

    body = await request.json()
    print("EVENT:", body.get("event"))

    event = body.get("event")
    call = body.get("call", {})
    call_id = call.get("call_id")

    # -------------------------------
    # 1. CALL START
    # -------------------------------
    if event == "call_started":
        return {
            "response": "Hello! I am your assistant. How can I help you today?"
        }

    # -------------------------------
    # 2. ONLY PROCESS LIVE USER INPUT
    # -------------------------------
    if event not in ["transcript_updated", "user_message", "call_analyzed"]:
        return {"response": ""}

    # -------------------------------
    # 3. GET LAST USER MESSAGE (REAL-TIME SAFE)
    # -------------------------------
    transcript = call.get("transcript", "")

    latest_user_message = ""
    for line in reversed(transcript.split("\n")):
        if "User:" in line:
            latest_user_message = line.replace("User:", "").strip()
            break

    if not latest_user_message:
        return {"response": ""}

    # -------------------------------
    # 4. DB SESSION
    # -------------------------------
    db = SessionLocal()

    try:
        session = db.query(CallSession).filter_by(call_id=call_id).first()

        if not session:
            session = CallSession(call_id=call_id)
            db.add(session)
            db.commit()

        # -------------------------------
        # 5. HUMAN HANDOFF
        # -------------------------------
        if is_human_request(latest_user_message):
            return {
                "response": "Do you want me to transfer you to a human agent?"
            }

        # -------------------------------
        # 6. BOOKING FLOW
        # -------------------------------
        if is_booking_intent(latest_user_message):
            response = handle_booking(db, call_id, latest_user_message)

        # -------------------------------
        # 7. RAG (PDF QA)
        # -------------------------------
        else:
            history = get_history(call_id)

            response = ask_question_for_voice(
                app.state.vectorstore,
                app.state.llm,
                latest_user_message,
                history
            )

        # -------------------------------
        # 8. MEMORY
        # -------------------------------
        append_message(call_id, "user", latest_user_message)
        append_message(call_id, "assistant", response)

        return {"response": response}

    finally:
        db.close()
def is_human_request(text: str):
    keywords = ["human", "agent", "real person", "representative", "talk to someone"]
    return any(k in text.lower() for k in keywords)


def is_confirmation(text: str):
    confirm_words = ["yes", "yeah", "yep", "sure", "okay", "please"]
    return any(word in text.lower() for word in confirm_words)

def transfer_call():
    return {
        "response": "Connecting you to a human agent now. Please hold.",
        "actions": [
            {
                "type": "transfer_call",
                "to": os.getenv("HUMAN_PHONE_NUMBER")  # your number
            }
        ]
    }

# Step 2 → User confirms


@app.get("/test-all")
def test_all(db: Session = Depends(get_db)):
    result = {}

    # DB TEST
    try:
        doctors = db.query(Doctor).count()
        result["db"] = f"OK ({doctors} doctors)"
    except Exception as e:
        result["db"] = f"ERROR: {str(e)}"

    # REDIS TEST
    try:
        append_message("test", "user", "hello")
        history = get_history("test")
        result["redis"] = f"OK ({len(history)} messages)"
    except Exception as e:
        result["redis"] = f"ERROR: {str(e)}"

    # AI TEST
    try:
        response = ask_question(
            app.state.vectorstore,
            app.state.llm,
            "hello"
        )
        result["ai"] = "OK"
    except Exception as e:
        result["ai"] = f"ERROR: {str(e)}"

    return result
# ---------------- Run ---------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("mains:app", host="0.0.0.0", port=port)