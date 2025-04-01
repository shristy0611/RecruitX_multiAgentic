from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import json

from recruitx_app.models.job import Job
from recruitx_app.schemas.job import JobCreate, JobAnalysis
from recruitx_app.agents.jd_analysis_agent import JDAnalysisAgent

class JobService:
    def __init__(self):
        self.jd_analysis_agent = JDAnalysisAgent()
    
    def get_job(self, db: Session, job_id: int) -> Optional[Job]:
        """Get a job by ID."""
        return db.query(Job).filter(Job.id == job_id).first()
    
    def get_jobs(self, db: Session, skip: int = 0, limit: int = 100) -> List[Job]:
        """Get a list of jobs with pagination."""
        return db.query(Job).offset(skip).limit(limit).all()
    
    def create_job(self, db: Session, job_data: JobCreate) -> Job:
        """Create a new job in the database."""
        db_job = Job(**job_data.model_dump())
        db.add(db_job)
        db.commit()
        db.refresh(db_job)
        return db_job
    
    async def analyze_job(self, db: Session, job_id: int) -> Optional[Dict[str, Any]]:
        """Analyze a job description and update the database with the analysis results."""
        # Get the job from the database
        job = self.get_job(db, job_id)
        if not job:
            return None
        
        # Use the JD Analysis Agent to analyze the job description
        analysis_result = await self.jd_analysis_agent.analyze_job_description(
            job_id=job.id, 
            job_description=job.description_raw
        )
        
        if analysis_result:
            # Convert the Pydantic model to a dictionary
            analysis_dict = analysis_result.model_dump()
            
            # Update the job with the analysis results
            job.analysis = analysis_dict
            db.commit()
            db.refresh(job)
            
            return analysis_dict
        
        return None 