# memory_db.py
from sqlalchemy import desc

from sqlalchemy.orm import Session
from models import ChatHistory
from database import SessionLocal

def save_message(phone_number: str, role: str, message: str):
    db: Session = SessionLocal()
    chat = ChatHistory(
        phone_number=phone_number,
        role=role,
        message=message
    )
    db.add(chat)
    db.commit()
    db.close()

def load_memory(phone_number: str, limit: int = 10):
    db: Session = SessionLocal()

    messages = (
        db.query(ChatHistory)
        .filter(ChatHistory.phone_number == phone_number)
        .order_by(desc(ChatHistory.created_at))
        .limit(limit)
        .all()
    )

    db.close()


    return messages