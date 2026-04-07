from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON,ForeignKey
from datetime import datetime
from database import Base


class CallSession(Base):
    __tablename__ = "call_sessions"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(String)

    booking_stage = Column(String)

    # ✅ ADD ALL THESE
    status = Column(String, default="in_progress")

    doctor_name = Column(String)
    doctor_id = Column(Integer)

    patient_name = Column(String)

    appointment_date = Column(String)
    appointment_time = Column(String)
    conversation_history = Column(JSON, default=list)

class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    specialization = Column(String)
    available_days = Column(String)
    available_time = Column(String)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    phone = Column(String)

class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    patient_id = Column(Integer, ForeignKey("patients.id"))
    appointment_time = Column(DateTime)
    human_requested = Column(Boolean, default=False)