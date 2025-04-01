from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

# Base job schema with common attributes
class JobBase(BaseModel):
    title: str
    company: Optional[str] = None
    location: Optional[str] = None

# Schema for creating a job (used when uploading a JD)
class JobCreate(JobBase):
    description_raw: str
    
# Schema for job data that will be returned to clients
class Job(JobBase):
    id: int
    description_raw: str
    analysis: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True  # For ORM compatibility

# Schema for skill demand in market insights
class SkillDemand(BaseModel):
    high_demand_skills: List[str] = Field([], description="Skills from the job description that are in high demand")
    trending_skills: List[str] = Field([], description="Skills that are trending in this industry")

# Schema for market insights
class MarketInsights(BaseModel):
    skill_demand: Optional[SkillDemand] = Field(None, description="Information about the demand for key skills")
    salary_insights: Optional[str] = Field(None, description="Factual information about salary ranges for this role in the market")
    industry_outlook: Optional[str] = Field(None, description="Brief outlook for the industry this job is in")

# Schema for job analysis results
class JobAnalysis(BaseModel):
    job_id: int
    required_skills: List[str] = Field(..., description="List of required skills extracted from the JD")
    preferred_skills: List[str] = Field([], description="List of preferred/nice-to-have skills")
    minimum_experience: Optional[str] = Field(None, description="Minimum years of experience required")
    education: Optional[str] = Field(None, description="Required education level")
    responsibilities: List[str] = Field([], description="Job responsibilities")
    job_type: Optional[str] = Field(None, description="Type of job (full-time, part-time, contract)")
    salary_range: Optional[str] = Field(None, description="Salary range if mentioned")
    company_culture: Optional[str] = Field(None, description="Company culture information")
    benefits: List[str] = Field([], description="Benefits offered by the company")
    industry: Optional[str] = Field(None, description="Industry or sector of the job")
    seniority_level: Optional[str] = Field(None, description="Seniority level (junior, mid, senior)")
    market_insights: Optional[MarketInsights] = Field(None, description="Grounded market insights about this job and industry")
    reasoning: Optional[str] = Field(None, description="Explanation of the analysis process and key insights")
    analysis_process: Optional[str] = Field(None, description="Detailed thinking process from Gemini's analysis") 