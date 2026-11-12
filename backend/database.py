"""
database.py — Base de datos SQLite con SQLAlchemy.
Se crea automáticamente en backend/agua_app.db al arrancar.
"""

from sqlalchemy import create_engine, Column, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime, uuid

DATABASE_URL = "sqlite:///./agua_app.db"

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class User(Base):
    __tablename__ = "users"

    id                   = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email                = Column(String, unique=True, index=True, nullable=False)
    username             = Column(String, unique=True, nullable=False)
    hashed_password      = Column(String, nullable=False)
    role                 = Column(String, default="user")   # "user" | "admin"
    is_active            = Column(Boolean, default=True)
    created_at           = Column(DateTime, default=datetime.datetime.utcnow)
    reset_code           = Column(String, nullable=True)
    reset_code_expires   = Column(DateTime, nullable=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def crear_tablas():
    Base.metadata.create_all(bind=engine)
