# memory_db.py
from sqlalchemy import desc
from sqlalchemy.orm import Session
from models import ChatHistory
from database import SessionLocal

def save_message(phone_number: str, role: str, message: str):
    with SessionLocal() as db:
        chat = ChatHistory(
            phone_number=phone_number,
            role=role,
            message=message
        )
        db.add(chat)
        db.commit()


def load_memory(phone_number: str, limit: int = 10):
    """
    Load last 'limit' messages for a phone number,
    returns a list of dicts instead of raw ORM objects.
    """
    with SessionLocal() as db:
        messages = (
            db.query(ChatHistory)
            .filter(ChatHistory.phone_number == phone_number)
            .order_by(desc(ChatHistory.created_at))
            .limit(limit)
            .all()
        )
        # convert to simple dicts
        return [
            {
                "role": m.role,
                "message": m.message,
                "created_at": m.created_at.isoformat()
            }
            for m in reversed(messages)  # oldest first
        ]