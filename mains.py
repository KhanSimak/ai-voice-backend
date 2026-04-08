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
from fastapi.responses import JSONResponse
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

from pydantic import BaseModel

class ChatRequest(BaseModel):
    text: str





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
    text: str = None
    question: str = None

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

    try:
        if not query:
            return "I didn't understand that."

        if history is None:
            history = []

        # ---------------- RETRIEVE ----------------
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs])

        if not context:
            context = "No relevant information found."

        # ---------------- HISTORY ----------------
        chat_history = ""
        for h in history[-4:]:
            chat_history += f"{h.get('role')}: {h.get('content')}\n"

        # ---------------- PROMPT ----------------
        prompt = f"""
You are a real-time voice AI assistant.

RULES:
- Use ONLY context
- If not found say "I don't have that information"
- Keep answer under 2-3 sentences

CHAT HISTORY:
{chat_history}

CONTEXT:
{context}

USER:
{query}

ANSWER:
"""

        res = llm.invoke(prompt)

        return res.content if hasattr(res, "content") else str(res)

    except Exception as e:
        print("RAG ERROR:", e)
        return "Sorry, I had trouble processing that."
# ---------------- Routes ---------------- #

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/ask")
async def ask(body: QuestionRequest):
    answer = ask_question(app.state.vectorstore, app.state.llm, body.question)
    return {"answer": answer}
   




def generate_response(text: str):
    return f"You said: {text}"



@app.post("/retell-webhook")
async def retell_webhook(request: Request):
    data = await request.json()

    print("🔥 FULL DATA:", data)

    # 🔥 THIS is what you need
    messages = data.get("messages", [])

    if not messages:
        return {
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you?"
                    }
                }
            ]
        }

    call_id = data.get("call", {}).get("call_id")

# 🔥 get history
    history = get_history(call_id)

# 🔥 AI response
    response = ask_question_for_voice(
      app.state.vectorstore,
      app.state.llm,
      user_text,
      history
    )

# 🔥 save history
    append_message(call_id, "user", user_text)
    append_message(call_id, "assistant", response)

    return {
        "choices": [
            {
                "message": {
                    "content": response
                }
            }
        ]
    }
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

@app.post("/chat")
async def chat(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    text = data.get("text", "").lower()

    # 🔥 doctor query
    if "doctor" in text:
        doctors = db.query(Doctor).all()

        return {
            "response": "\n".join([
                f"{d.name} - {d.specialization}"
                for d in doctors
            ])
        }

    # fallback AI
    response = ask_question_for_voice(
        app.state.vectorstore,
        app.state.llm,
        text
    )

    return {"response": response}




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
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "mains:app",
        host="0.0.0.0",
        port=port
    )