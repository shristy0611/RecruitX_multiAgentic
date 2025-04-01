from fastapi import APIRouter, Depends, HTTPException, Body, Query, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging

from recruitx_app.core.database import get_db 
from recruitx_app.services.scoring_service import ScoringService # Import service
from recruitx_app.models.score import Score
from pydantic import BaseModel

router = APIRouter()
scoring_service = ScoringService() # Reinstate global instance

# Set up logging
logger = logging.getLogger(__name__)

class ScoreCreate(BaseModel):
    job_id: int
    candidate_id: int
    # include_visualizations can be added back if needed later

class BatchScoreCreate(BaseModel):
    job_id: int
    candidate_ids: List[int]

# Restore original ScoreResponse
class ScoreResponse(BaseModel):
    id: int
    job_id: int
    candidate_id: int
    overall_score: float # Make non-nullable again
    explanation: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    # status field removed
    
    class Config:
        from_attributes = True

# Revert create_score endpoint to synchronous operation
@router.post("/", response_model=ScoreResponse)
async def create_score(
    score_data: ScoreCreate,
    db: Session = Depends(get_db)
):
    """
    Generate a match score between a job and a candidate.
    (Now operates synchronously, potentially using multiple steps internally).
    """
    # Call the (soon to be updated) synchronous scoring service method
    score = await scoring_service.generate_score(
        db=db,
        job_id=score_data.job_id,
        candidate_id=score_data.candidate_id
    )
    
    if not score:
        # Handle cases where job/candidate not found or scoring failed internally
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job or Candidate not found, or scoring failed."
        )
        
    return score # Return the generated score directly

# GET endpoints remain mostly the same, just return the reverted ScoreResponse
@router.get("/{score_id}", response_model=ScoreResponse)
def get_score(
    score_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific score by ID.
    """
    score = scoring_service.get_score(db, score_id=score_id)
    
    if not score:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Score with ID {score_id} not found"
        )
    
    return score 

@router.get("/job/{job_id}", response_model=List[ScoreResponse])
def get_scores_for_job(
    job_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all scores for a specific job.
    """
    scores = scoring_service.get_scores_for_job(db, job_id=job_id)
    return scores

@router.get("/candidate/{candidate_id}", response_model=List[ScoreResponse])
def get_scores_for_candidate(
    candidate_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all scores for a specific candidate.
    """
    scores = scoring_service.get_scores_for_candidate(db, candidate_id=candidate_id)
    return scores

# Revert batch endpoint to synchronous (or keep async but call synchronous generate_score)
# For simplicity, let's make it call generate_score repeatedly, which might be slow/hit limits
# A better batch approach would need redesigning generate_score for batching.
@router.post("/batch", response_model=Dict[str, Any])
async def batch_create_scores(
    batch_data: BatchScoreCreate,
    db: Session = Depends(get_db)
):
    """
    Generate scores for a job against multiple candidates synchronously.
    WARNING: This might be slow and could hit rate limits on the free tier.
    """
    results = {}
    successful_count = 0
    
    for candidate_id in batch_data.candidate_ids:
        try:
            # Call generate_score for each candidate
            score = await scoring_service.generate_score(
                db=db, 
                job_id=batch_data.job_id, 
                candidate_id=candidate_id
            )
            if score:
                results[str(candidate_id)] = {
                    "score_id": score.id,
                    "overall_score": score.overall_score,
                    "status": "success"
                }
                successful_count += 1
            else:
                 results[str(candidate_id)] = {"status": "error", "message": "Score generation returned None (Job/Candidate not found?)"}
        except Exception as e:
            logger.error(f"Error processing candidate {candidate_id} in batch for job {batch_data.job_id}: {e}")
            results[str(candidate_id)] = {"status": "error", "message": str(e)}

    return {
        "job_id": batch_data.job_id,
        "results": results,
        "total_processed": len(batch_data.candidate_ids),
        "successful": successful_count
    } 