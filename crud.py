from sqlalchemy.orm import Session
from models import Appointment, CallLog

def create_appointment(db: Session, data):
    appointment = Appointment(**data.dict())
    db.add(appointment)
    db.commit()
    db.refresh(appointment)
    return appointment

def log_call(db: Session, phone, question, response):
    log = CallLog(
        phone_number=phone,
        question=question,
        response=response
    )
    db.add(log)
    db.commit()
