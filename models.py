from sqlalchemy import Column, Integer, String, Date, DateTime, ForeignKey
from datetime import datetime
from database import Base


class CallSession(Base):
    __tablename__ = "call_sessions"
    status = Column(String, default="in_progress")

    id = Column(Integer, primary_key=True)
    call_id = Column(String, unique=True)

    patient_name = Column(String)
    doctor_name = Column(String)

    appointment_date = Column(String)
    appointment_time = Column(String)
    booking_stage = Column(String)
    status = Column(String, default="in_progress")

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