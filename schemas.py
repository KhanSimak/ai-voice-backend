from pydantic import BaseModel
from datetime import datetime

class QueryRequest(BaseModel):
    question: str
    phone_number: str

class AppointmentCreate(BaseModel):
    patient_name: str
    phone_number: str
    doctor_name: str
    appointment_time: datetime
