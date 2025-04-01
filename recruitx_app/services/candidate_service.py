from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import logging

from recruitx_app.models.candidate import Candidate
from recruitx_app.agents.cv_analysis_agent import CVAnalysisAgent

logger = logging.getLogger(__name__)

class CandidateService:
    """
    Service for handling candidate-related operations.
    """
    
    def __init__(self):
        """Initialize the candidate service with the CV analysis agent."""
        self.cv_agent = CVAnalysisAgent()
    
    def get_candidates(self, db: Session, skip: int = 0, limit: int = 100) -> List[Candidate]:
        """Get a list of candidates with pagination."""
        return db.query(Candidate).offset(skip).limit(limit).all()
    
    def get_candidate(self, db: Session, candidate_id: int) -> Optional[Candidate]:
        """Get a specific candidate by ID."""
        return db.query(Candidate).filter(Candidate.id == candidate_id).first()
    
    def create_candidate(self, db: Session, candidate_data: Dict[str, Any]) -> Candidate:
        """
        Create a new candidate.
        
        Args:
            db: Database session
            candidate_data: Dictionary containing candidate information
            
        Returns:
            The created candidate
        """
        db_candidate = Candidate(**candidate_data.model_dump())
        db.add(db_candidate)
        db.commit()
        db.refresh(db_candidate)
        return db_candidate
    
    async def analyze_cv(self, db: Session, candidate_id: int) -> Optional[Dict[str, Any]]:
        """
        Analyze a candidate's CV using the CV Analysis Agent.
        
        Args:
            db: Database session
            candidate_id: ID of the candidate to analyze
            
        Returns:
            Dictionary containing the analysis result or None if failed
        """
        # Get the candidate
        candidate = self.get_candidate(db, candidate_id)
        if not candidate:
            logger.error(f"Candidate with ID {candidate_id} not found")
            return None
            
        # Check if we have the raw resume text
        if not candidate.resume_raw:
            logger.error(f"Candidate with ID {candidate_id} has no resume text")
            return None
            
        # Analyze the CV
        analysis_result = await self.cv_agent.analyze_cv(candidate.resume_raw)
        
        if analysis_result:
            # Update the candidate with the analysis
            candidate.analysis = analysis_result
            db.commit()
            db.refresh(candidate)
            
        return analysis_result 