# main.py
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, Appointment, Conversation
from memry import save_message, load_memory
from rag import ask_rag
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import uvicorn



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
class AskRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(payload: AskRequest):
    answer = ask_rag(payload.question)
    return {"answer": answer}
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


embeddings = OpenAIEmbeddings()

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

def ask_rag(question: str):
    result = qa_chain.invoke({"query": question})
    return result["result"]


@app.post("/retell-webhook")
async def retell_webhook(request: Request):
    try:
        data = await request.json()
        print("Received:", data)
        return {"status": "received", "data": data}
    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}
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
        

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)