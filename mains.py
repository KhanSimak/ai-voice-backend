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
        body = await request.body()

        # ✅ handle empty body
        if not body:
            return {"response": "I didn't catch that. Can you repeat?"}

        # ✅ parse JSON safely
        try:
            data = await request.json()
        except Exception as e:
            print("JSON ERROR:", e)
            return {"response": "Invalid input"}

        text = data.get("text", "").lower().strip()

        if not text:
            return {"response": "Can you say that again?"}

        # ✅ doctor query
        if "doctor" in text:
            doctors = db.query(Doctor).all()

            return {
                "response": "\n".join([
                    f"{d.name} - {d.specialization}"
                    for d in doctors
                ])
            }

        # ✅ fallback AI
        response = ask_question_for_voice(
            app.state.vectorstore,
            app.state.llm,
            text
        )

        return {"response": response}

    except Exception as e:
        print("CHAT ERROR:", e)
        return {"response": "Something went wrong"}
# ✅ RETELL WEBHOOK (FIXED)
@app.post("/retell-webhook")
async def retell_webhook(request: Request, db: Session = Depends(get_db)):
    try:
        body = await request.body()

        # 🛑 Ignore empty requests (Retell sends these)
        if not body:
            return {"status": "ignored"}

        try:
            data = await request.json()
        except Exception:
            return {"status": "not_json"}

        print("🔥 FULL DATA:", data)

        event = data.get("event")

        # 🛑 Ignore events that don't need response
        if event in ["call_started", "call_ended", "call_analyzed"]:
            return {"status": f"ignored_{event}"}

        # ✅ Extract messages
        messages = data.get("messages", [])

        if not messages:
            return {
                "choices": [
                    {"message": {"content": "Hello, how can I help you?"}}
                ]
            }

        # ✅ Get last user message
        user_text = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content")
                break

        if not user_text:
            return {
                "choices": [
                    {"message": {"content": "Can you repeat that?"}}
                ]
            }

        user_text = user_text.lower().strip()

        call_id = data.get("call", {}).get("call_id", "default")

        # 🔥 HISTORY (Redis)
        history = get_history(call_id)

        # -------------------------------
        # ✅ HANDLE DOCTOR QUERY DIRECTLY
        # -------------------------------
        if "doctor" in user_text:
            doctors = db.query(Doctor).all()

            response = "Here are available doctors:\n" + "\n".join([
                f"{d.name} - {d.specialization}"
                for d in doctors
            ])

        # -------------------------------
        # ✅ AI RESPONSE
        # -------------------------------
        else:
            response = ask_question_for_voice(
                app.state.vectorstore,
                app.state.llm,
                user_text,
                history
            )

        # 🔥 SAVE HISTORY
        append_message(call_id, "user", user_text)
        append_message(call_id, "assistant", response)

        # ✅ RETELL FORMAT (VERY IMPORTANT)
        return {
            "choices": [
                {
                    "message": {
                        "content": response
                    }
                }
            ]
        }

    except Exception as e:
        print("❌ WEBHOOK ERROR:", str(e))

        return {
            "choices": [
                {
                    "message": {
                        "content": "Sorry, something went wrong."
                    }
                }
            ]
        }
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

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    uvicorn.run("mains:app", host="0.0.0.0", port=8000)