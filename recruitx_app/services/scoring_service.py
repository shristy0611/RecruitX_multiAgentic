from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging
import asyncio

from recruitx_app.models.job import Job
from recruitx_app.models.candidate import Candidate
from recruitx_app.models.score import Score
# Rename the import
from recruitx_app.agents.simple_scoring_agent import OrchestrationAgent 

# Set up logging
logger = logging.getLogger(__name__)

class ScoringService:
    """
    Service for generating match scores using a manual multi-step orchestration.
    Step 1: Extract skills using LLM.
    Step 2: Synthesize final score using LLM based on texts and extracted skills.
    """
    
    def __init__(self):
        # Use the orchestration agent
        self.orchestration_agent = OrchestrationAgent()
    
    async def generate_score( 
        self, 
        db: Session, 
        job_id: int,
        candidate_id: int
    ) -> Optional[Score]:
        """
        Generates a match score using a two-step manual orchestration flow.
        
        Args:
            db: The database session
            job_id: The ID of the job
            candidate_id: The ID of the candidate
            
        Returns:
            The created Score object or None if an error occurred
        """
        job = db.query(Job).filter(Job.id == job_id).first()
        candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()

        if not job or not candidate:
            logger.warning(f"Cannot generate score: Job ID {job_id} or Candidate ID {candidate_id} not found.")
            return None
            
        logger.info(f"Starting two-step score generation for Job {job_id}, Candidate {candidate_id}.")
        try:
            # --- Step 1: Extract Skills --- 
            logger.info(f"Step 1: Extracting skills for Job {job_id}, Candidate {candidate_id}.")
            skill_extraction_result = await self.orchestration_agent.extract_skills(
                job_description=job.description_raw,
                candidate_resume=candidate.resume_raw
            )

            db_score: Optional[Score] = None # Define db_score variable here

            if "error" in skill_extraction_result:
                logger.error(f"Skill extraction failed: {skill_extraction_result['error']}")
                # Create a score record indicating the failure
                db_score = Score(
                    job_id=job_id,
                    candidate_id=candidate_id,
                    overall_score=0.0, 
                    explanation=f"Failed during skill extraction: {skill_extraction_result.get('error', 'Unknown error')}",
                    details=skill_extraction_result
                )
            else:
                job_skills = skill_extraction_result.get("job_skills", [])
                candidate_skills = skill_extraction_result.get("candidate_skills", [])
                logger.info(f"Step 1 successful. Job Skills: {len(job_skills)}, Candidate Skills: {len(candidate_skills)}.")
                
                # --- Step 2: Synthesize Score --- 
                logger.info(f"Step 2: Synthesizing score for Job {job_id}, Candidate {candidate_id}.")
                # Add a small delay between API calls, might help slightly with ultra-strict limits
                await asyncio.sleep(1) # Reduced delay slightly
                
                score_synthesis_result = await self.orchestration_agent.synthesize_score(
                     job_description=job.description_raw,
                     candidate_resume=candidate.resume_raw,
                     job_skills=job_skills,
                     candidate_skills=candidate_skills
                )

                overall_score = 0.0 # Default score
                explanation = "Score synthesis failed."
                details = score_synthesis_result # Store synthesis result by default
                
                if "error" in score_synthesis_result:
                     logger.error(f"Score synthesis failed: {score_synthesis_result.get('error', 'Unknown error')}")
                     explanation = f"Failed during score synthesis: {score_synthesis_result.get('error', 'Unknown error')}"
                     # Keep overall_score at 0.0
                else:
                     overall_score = score_synthesis_result.get("overall_score", 0.0)
                     explanation = score_synthesis_result.get("explanation", "No explanation provided.")
                     # Combine extraction and synthesis results for details
                     details = { 
                         "synthesis_result": score_synthesis_result,
                         "extracted_skills": skill_extraction_result
                     }
                     logger.info(f"Step 2 successful. Score: {overall_score}")

                # Create the score record with the final result (or failure info from step 2)
                db_score = Score(
                    job_id=job_id,
                    candidate_id=candidate_id,
                    overall_score=overall_score, 
                    explanation=explanation,
                    details=details
                )

            # --- Save Score Record (if created) --- 
            if db_score:
                db.add(db_score)
                db.commit()
                db.refresh(db_score)
                logger.info(f"Saved score {db_score.id} for Job {job_id}, Candidate {candidate_id}. Final Score: {db_score.overall_score}")
                return db_score
            else:
                 logger.error(f"Failed to create a score object for Job {job_id}, Candidate {candidate_id}.")
                 return None

        except Exception as e:
            logger.error(f"Exception during orchestrated score generation for Job {job_id}, Candidate {candidate_id}: {e}", exc_info=True)
            db.rollback()
            return None # Return None on failure
            
    def get_score(self, db: Session, score_id: int) -> Optional[Score]:
        """Get a score by ID."""
        return db.query(Score).filter(Score.id == score_id).first()
    
    def get_scores_for_job(self, db: Session, job_id: int) -> List[Score]:
        """Get all scores for a specific job."""
        return db.query(Score).filter(Score.job_id == job_id).all()
    
    def get_scores_for_candidate(self, db: Session, candidate_id: int) -> List[Score]:
        """Get all scores for a specific candidate."""
        return db.query(Score).filter(Score.candidate_id == candidate_id).all()

    # Removed batch_generate_scores as the synchronous version is in the API endpoint now
    # A more robust batch implementation would modify generate_score or use a different approach 