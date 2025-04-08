from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr, ConfigDict

# Base candidate schema with common attributes
class CandidateBase(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    location: Optional[str] = None
    resume_raw: Optional[str] = None

# Schema for creating a candidate
class CandidateCreate(CandidateBase):
    pass

# Schema for candidate data that will be returned to clients
class Candidate(CandidateBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

# Schema for candidate analysis results
class CandidateAnalysis(BaseModel):
    candidate_id: int
    contact_info: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    skills: List[str] = []
    work_experience: List[Dict[str, Any]] = []
    education: List[Dict[str, Any]] = []
    certifications: List[str] = []
    projects: List[Dict[str, Any]] = []
    languages: List[str] = []
    overall_profile: Optional[str] = None
    cv_text: Optional[str] = None

# Schemas related to Candidate data

# Schema for contact information (can be nested)
class ContactInfo(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Optional[str] = None

# Schemas for CV Analysis structured data

class WorkExperience(BaseModel):
    company: str
    title: str
    dates: Optional[str] = None
    responsibilities: Optional[List[str]] = None

class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    graduation_date: Optional[str] = None

class Project(BaseModel):
    name: str
    description: Optional[str] = None
    technologies: Optional[List[str]] = None 