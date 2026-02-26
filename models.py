from sqlalchemy import Column, Integer, String, DateTime
from database import Base
from datetime import datetime

class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String, nullable=False)
    phone_number = Column(String, nullable=False)
    doctor_name = Column(String, nullable=False)
    appointment_time = Column(DateTime, nullable=False, default=datetime.utcnow)

from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from database import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String)
    role = Column(String)
    message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())