# main.py
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, Appointment, Conversation
from memry import save_message, load_memory
from rag import ask_clinic_bot
from datetime import datetime

# Initialize app
app = FastAPI()

# Create tables
Base.metadata.create_all(bind=engine)

# -----------------------------
# DB session dependency
# -----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------
# Conversation engine
# -----------------------------
def conversation_engine(user_id: str, message: str):
    db = SessionLocal()
    convo = db.query(Conversation).filter_by(user_id=user_id).first()
    
    if not convo:
        convo = Conversation(user_id=user_id, state="greeting")
        db.add(convo)
        db.commit()

    if convo.state == "greeting":
        convo.state = "collecting_name"
        db.commit()
        return "Hello! May I know your name?"

    if convo.state == "collecting_name":
        convo.name = message
        convo.state = "collecting_purpose"
        db.commit()
        return "How can I help you today?"

    return None  # conversation complete

# -----------------------------
# Retell Webhook
# -----------------------------# -----------------------------
# RETELL WEBHOOK
# -----------------------------
@app.post("/retell-webhook")
async def retell_webhook(request: Request):
    data = await request.json()

    # Retell usually sends call_id + transcript
    user_id = data.get("call_id")
    message = data.get("transcript")

    if not user_id or not message:
        return {"response": "Sorry, I didn't receive your message properly."}

    # 1️⃣ Save user message in memory
    save_message(user_id, role="user", message=message)

    # 2️⃣ Check conversation state
    db = SessionLocal()
    convo = db.query(Conversation).filter_by(user_id=user_id).first()

    if not convo:
        convo = Conversation(user_id=user_id, state="greeting")
        db.add(convo)
        db.commit()

    # -----------------------------
    # Conversation Flow
    # -----------------------------

    if convo.state == "greeting":
        convo.state = "collecting_name"
        db.commit()
        db.close()
        reply = "Hello! May I know your name?"

    elif convo.state == "collecting_name":
        convo.name = message
        convo.state = "normal"
        db.commit()
        db.close()
        reply = f"Nice to meet you {message}. How can I help you today?"

    else:
        db.close()

        # 3️⃣ Load memory for better context
        history = load_memory(user_id, limit=10)
        context = "\n".join(
            [f"{h['role']}: {h['message']}" for h in history]
        )

        full_prompt = f"""
        Previous conversation:
        {context}

        User question:
        {message}
        """

        # 4️⃣ Call RAG
        reply = ask_clinic_bot(full_prompt)

    # 5️⃣ Save bot reply
    save_message(user_id, role="bot", message=reply)

    # 6️⃣ Return response to Retell
    return {"response": reply}

    # 3️⃣ If conversation complete, fallback to RAG + memory
    if not response:
        # Load last 10 messages as context
        history = load_memory(user_id, limit=10)
        context = "\n".join([h["message"] for h in history])
        combined_message = f"{context}\nUser: {message}"
        response = ask_clinic_bot(combined_message)

    # 4️⃣ Save bot reply
    save_message(user_id, role="bot", message=response)

    return {"response": response}

# -----------------------------
# Appointments endpoints
# -----------------------------
@app.get("/appointments")
def get_appointments(db: Session = Depends(get_db)):
    appointments = db.query(Appointment).all()
    result = []
    for a in appointments:
        result.append({
            "id": a.id,
            "patient_name": a.patient_name,
            "phone_number": a.phone_number,
            "doctor_name": a.doctor_name,
            "appointment_time": a.appointment_time.isoformat()
        })
    return result

@app.post("/book-appointment")
def book_appointment(
    patient_name: str,
    phone_number: str,
    doctor_name: str,
    appointment_time: str,
    db: Session = Depends(get_db)
):
    dt = datetime.fromisoformat(appointment_time)
    appointment = Appointment(
        patient_name=patient_name,
        phone_number=phone_number,
        doctor_name=doctor_name,
        appointment_time=dt
    )
    db.add(appointment)
    db.commit()
    db.refresh(appointment)
    return {
        "message": "Appointment booked successfully ✅",
        "appointment_id": appointment.id
    }

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def home():
    return {"message": "Server is running 🚀"}

@app.get("/check-db")
def check_db():
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
            return {"database": "Connected successfully ✅"}
    except Exception as e:
        return {"database": "Connection failed ❌", "error": str(e)}