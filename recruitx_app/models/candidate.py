from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.sql import func
from recruitx_app.core.database import Base

class Candidate(Base):
    __tablename__ = "candidates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    resume_raw = Column(Text, nullable=False)  # Raw text from the CV
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Store the structured data extracted by the CV Analysis Agent
    analysis = Column(JSON, nullable=True)  # Will store skills, work experience, education, etc.

    def __repr__(self):
        return f"<Candidate {self.id}: {self.name}>" 