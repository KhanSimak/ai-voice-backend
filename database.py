from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

print("DB PATH:", os.path.abspath("test.db"))
 
DATABASE_URL = os.getenv(
    "DATABASE_URL",
     "postgresql://postgres:simak007@localhost:5432/clinicdb"
)
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
