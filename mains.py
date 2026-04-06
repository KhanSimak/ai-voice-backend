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

# ---------------- Logging ---------------- #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Startup ---------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    
    from redis_client import r

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

def ask_question_for_voice(vectorstore, llm, question, history):

    docs = vectorstore.similarity_search(question, k=3)

    if docs:
        context = "\n\n".join([doc.page_content for doc in docs])
        system = f"""
You are a friendly clinic assistant speaking on a call.
Keep answers short (2-3 sentences), natural, and conversational.
Use this info if helpful:

{context}
"""
    else:
        system = "You are a friendly assistant. Keep answers short and natural."

    messages = [SystemMessage(content=system)]

    for msg in history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    res = llm.invoke(messages)
    return res.content if hasattr(res, "content") else str(res)


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

    try:
        raw_body = await request.body()
        print("RAW BODY:", raw_body)

        body = await request.json()
    except Exception:
        return {"response": "Invalid request format"}

    call = body.get("call", {})
    call_id = call.get("call_id")

    transcript = call.get("transcript_object", [])

    latest_user_message = ""
    for msg in reversed(transcript):
        if msg.get("role") == "user":
            latest_user_message = msg.get("content", "").strip()
            break

    if not latest_user_message:
        return {"response": "Sorry, I didn’t catch that. Could you repeat?"}

    db = SessionLocal()

    try:
        # ---------------- SESSION ---------------- #
        session = db.query(CallSession).filter_by(call_id=call_id).first()

        if not session:
            session = CallSession(
                call_id=call_id,
                booking_stage=None,
                status="in_progress",
                human_requested=False
            )
            db.add(session)
            db.commit()

        # ---------------- HUMAN HANDOFF ---------------- #

        # STEP 1: Ask confirmation
        if is_human_request(latest_user_message) and not session.human_requested:
            session.human_requested = True
            db.commit()

            return {
                "response": "I can connect you to a human agent. Do you want me to transfer your call?"
            }

        # STEP 2: Confirm + transfer
        if session.human_requested:
            if is_confirmation(latest_user_message):

                session.human_requested = False
                db.commit()

                return {
                    "response": "Connecting you to a human agent now. Please hold.",
                    "actions": [
                        {
                            "type": "transfer_call",
                            "to": os.getenv("HUMAN_PHONE_NUMBER")
                        }
                    ]
                }

            else:
                session.human_requested = False
                db.commit()

                return {
                    "response": "Alright, I’ll continue assisting you."
                }

        # ---------------- BOOKING FLOW ---------------- #
        if is_booking_intent(latest_user_message) or session.booking_stage:
            response = handle_booking(db, call_id, latest_user_message)

        # ---------------- RAG FLOW ---------------- #
        else:
            history = get_history(call_id)

            response = ask_question_for_voice(
                app.state.vectorstore,
                app.state.llm,
                latest_user_message,
                history
            )

        # ---------------- SAVE MEMORY ---------------- #
        append_message(call_id, "user", latest_user_message)
        append_message(call_id, "assistant", response)

        return {"response": response}

    except Exception as e:
        print("Webhook error:", e)
        return {"response": "Sorry, something went wrong. Please try again."}

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