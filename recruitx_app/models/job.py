from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.sql import func
from recruitx_app.core.database import Base

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    company = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    description_raw = Column(Text, nullable=False)  # Raw text from the JD
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Store the structured data extracted by the JD Analysis Agent
    analysis = Column(JSON, nullable=True)  # Will store extracted skills, requirements, etc.

    def __repr__(self):
        return f"<Job {self.id}: {self.title} at {self.company}>" 