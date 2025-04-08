from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

# Base score schema with common attributes
class ScoreBase(BaseModel):
    job_id: int = Field(..., description="ID of the job the score is for")
    candidate_id: int = Field(..., description="ID of the candidate the score is for")
    overall_score: int = Field(..., ge=0, le=100, description="Overall match score (0-100)")
    explanation: str = Field(..., description="Human-readable explanation of the score")
    details: Dict[str, Any] = Field({}, description="Detailed breakdown of the score components")

# Schema for creating a score (internal use)
class ScoreCreate(ScoreBase):
    pass

# Schema for score data that will be returned to clients
class Score(ScoreBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

# Additional schemas for specific score components

class SkillsMatch(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Match score for skills (0-100)")
    matching_skills: List[str] = Field([], description="Skills that match between the job and candidate")
    missing_skills: List[str] = Field([], description="Required skills the candidate is missing")
    additional_skills: List[str] = Field([], description="Additional skills the candidate has")

class ExperienceMatch(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Match score for experience (0-100)")
    evaluation: str = Field(..., description="Explanation of the experience match")

class EducationMatch(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Match score for education (0-100)")
    evaluation: str = Field(..., description="Explanation of the education match")

class ResponsibilityMatch(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Match score for responsibilities (0-100)")
    matching_responsibilities: List[Dict[str, str]] = Field([], description="Matched responsibilities between job and candidate")

class ScoreDetails(BaseModel):
    """Detailed breakdown of score components - used for structured score details"""
    skills_match: SkillsMatch
    experience_match: ExperienceMatch
    education_match: Optional[EducationMatch] = None
    responsibility_match: Optional[ResponsibilityMatch] = None 