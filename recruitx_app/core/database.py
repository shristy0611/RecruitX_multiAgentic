import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from recruitx_app.core.config import settings

# We will update this to come from settings later, but for now use a SQLite database for development
SQLALCHEMY_DATABASE_URL = "sqlite:///./recruitx.db"  # SQLite is easier for local dev

# When you're ready for PostgreSQL, uncomment the following:
# SQLALCHEMY_DATABASE_URL = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}  # only needed for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 