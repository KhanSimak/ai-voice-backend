from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import engine, SessionLocal
from models import Base, Appointment
from datetime import datetime
from sqlalchemy import func
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from auth import authenticate_user, create_access_token
from jose import jwt, JWTError
from auth import SECRET_KEY, ALGORITHM
from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask_clinic_bot
from memry import load_memory
from database import SessionLocal
from models import ChatHistory
from sqlalchemy import distinct
from fastapi import APIRouter, Depends
from auth import verify_token

app = FastAPI()


# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def home():
    return {"message": "Server is running 🚀"}

@app.get("/check-db")
def check_db():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            return {"database": "Connected successfully ✅"}
    except Exception as e:
        return {"database": "Connection failed ❌", "error": str(e)}

# 👇 NEW ENDPOINT

@app.get("/appointments")
def get_appointments(db: Session = Depends(get_db)):
    appointments = db.query(Appointment).all()

    result = []
    for a in appointments:
        result.append({
            "id": a.id,
            "patient_name": a.patient_name,
            "phone_number": a.phone_number,
            "doctor_name": a.doctor_name
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
    
class ChatRequest(BaseModel):
    phone_number: str
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    answer = ask_clinic_bot(request.phone_number, request.message)
    return {"response": answer}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token({"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
admin_router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(verify_token)]
)

@app.get("/admin/analytics/total-users")
def total_users():
    db = SessionLocal()
    count = db.query(distinct(ChatHistory.phone_number)).count()
    db.close()
    return {"total_users": count}
admin_router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(verify_token)]
)

@admin_router.get("/users")
def get_users():
    db = SessionLocal()
    users = db.query(distinct(ChatHistory.phone_number)).all()
    db.close()

    return {"users": [u[0] for u in users]}
@app.get("/admin/analytics/top-questions")
def top_questions():
    db = SessionLocal()

    results = (
        db.query(ChatHistory.message, func.count(ChatHistory.message).label("count"))
        .filter(ChatHistory.role == "user")
        .group_by(ChatHistory.message)
        .order_by(func.count(ChatHistory.message).desc())
        .limit(5)
        .all()
    )

    db.close()

    return [{"question": r[0], "count": r[1]} for r in results]

@admin_router.get("/chats/{phone_number}")

def get_chat(phone_number: str):
    db = SessionLocal()
    messages = (
        db.query(ChatHistory)
        .filter(ChatHistory.phone_number == phone_number)
        .order_by(ChatHistory.created_at.asc())
        .all()
    )
    db.close()

    return messages
  



app.include_router(admin_router)

@app.post("/retell-webhook")
async def retell_webhook(request: Request):
    data = await request.json()
    print("Incoming from Retell:", data)

    return {
        "response": "Hello, how can I help you today?"
    }