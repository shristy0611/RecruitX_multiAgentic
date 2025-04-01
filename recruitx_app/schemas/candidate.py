from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

# Base candidate schema with common attributes
class CandidateBase(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None

# Schema for creating a candidate
class CandidateCreate(CandidateBase):
    resume_raw: str
    
# Schema for candidate data that will be returned to clients
class Candidate(CandidateBase):
    id: int
    resume_raw: str
    analysis: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True  # For ORM compatibility

# Schema for candidate analysis results
class CandidateAnalysis(BaseModel):
    candidate_id: int
    contact_info: Dict[str, Any] = Field(..., description="Contact information extracted from the CV")
    summary: Optional[str] = Field(None, description="Professional summary/objective")
    skills: List[str] = Field([], description="Technical and soft skills")
    work_experience: List[Dict[str, Any]] = Field([], description="Work experience history")
    education: List[Dict[str, Any]] = Field([], description="Educational background")
    certifications: List[str] = Field([], description="Professional certifications")
    projects: List[Dict[str, Any]] = Field([], description="Notable projects")
    languages: List[str] = Field([], description="Languages the candidate knows")
    overall_profile: Optional[str] = Field(None, description="A brief assessment of the candidate's profile") 