from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise Exception("DATABASE_URL missing")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

print("✅ USING DB:", DATABASE_URL)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()