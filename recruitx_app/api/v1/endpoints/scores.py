from fastapi import APIRouter, Depends, HTTPException, Body, Query, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
import asyncio # Import asyncio
from uuid import UUID
from pydantic import BaseModel, ConfigDict  # Added ConfigDict

from recruitx_app.core.database import get_db 
from recruitx_app.services.scoring_service import ScoringService # Import service
from recruitx_app.models.score import Score
from recruitx_app.schemas.score import ScoreCreate

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
    
    model_config = ConfigDict(from_attributes=True)

# Define a response model for generate_score to avoid leaking internal details
class ScoreGenerationResponse(BaseModel):
    """Response model for the score generation endpoint."""
    job_id: UUID
    candidate_id: UUID
    message: str
    score_id: UUID | None = None  # Score ID might not be available immediately if async

    model_config = ConfigDict(from_attributes=True)  # Replaced Config class

# Pydantic model for the request body of the generate endpoint
class GenerateScoreRequest(BaseModel):
    """Request body for initiating score generation."""
    job_id: UUID
    candidate_id: UUID

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
    min_score: Optional[float] = Query(None, ge=0, le=100, description="Minimum overall score to include"),
    sort_by: Optional[str] = Query("overall_score", pattern="^(overall_score|created_at)$", description="Field to sort by"),
    sort_order: Optional[str] = Query("desc", pattern="^(asc|desc)$", description="Sort order (asc or desc)"),
    db: Session = Depends(get_db)
):
    """
    Get all scores for a specific job, with optional filtering and sorting.
    """
    scores = scoring_service.get_scores_for_job(
        db,
        job_id=job_id,
        min_score=min_score,
        sort_by=sort_by,
        sort_order=sort_order
    )
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

# Revert batch endpoint to use asyncio.gather for concurrency
@router.post("/batch", response_model=Dict[str, Any])
async def batch_create_scores(
    batch_data: BatchScoreCreate,
    db: Session = Depends(get_db)
):
    """
    Generate scores for a job against multiple candidates concurrently.
    WARNING: Still makes individual API calls per candidate, could hit rate limits.
    """
    results = {}
    successful_count = 0
    tasks = []

    # Create tasks for each candidate scoring operation
    for candidate_id in batch_data.candidate_ids:
        tasks.append(scoring_service.generate_score(
            db=db, 
            job_id=batch_data.job_id, 
            candidate_id=candidate_id
        ))
        
    # Run scoring tasks concurrently
    logger.info(f"Starting concurrent batch scoring for job {batch_data.job_id} and {len(tasks)} candidates.")
    score_results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"Finished concurrent batch scoring for job {batch_data.job_id}.")

    # Process results
    for i, result in enumerate(score_results):
        candidate_id = batch_data.candidate_ids[i]
        if isinstance(result, Exception):
            logger.error(f"Error processing candidate {candidate_id} in batch for job {batch_data.job_id}: {result}")
            results[str(candidate_id)] = {"status": "error", "message": str(result)}
        elif result: # Check if score object is not None
            results[str(candidate_id)] = {
                "score_id": result.id,
                "overall_score": result.overall_score,
                "status": "success"
            }
            successful_count += 1
        else:
             # Handle case where generate_score returned None (e.g., job/candidate not found initially)
             logger.warning(f"Score generation returned None for candidate {candidate_id} in batch for job {batch_data.job_id} (Job/Candidate not found?).")
             results[str(candidate_id)] = {"status": "error", "message": "Score generation returned None (Job/Candidate not found?)"}

    return {
        "job_id": batch_data.job_id,
        "results": results,
        "total_processed": len(batch_data.candidate_ids),
        "successful": successful_count
    } 

@router.post("/generate", response_model=ScoreGenerationResponse, status_code=202)
async def generate_score(
    *,
    db: Session = Depends(get_db),
    request_body: GenerateScoreRequest,
    scoring_service: ScoringService = Depends(ScoringService),
):
    """
    Initiates the scoring process for a given job and candidate.
    This endpoint is asynchronous and will return immediately.
    The actual scoring happens in the background.
    """
    try:
        # TODO: Implement background task processing (e.g., using Celery or FastAPI's BackgroundTasks)
        # For now, call synchronously for demonstration
        score = await scoring_service.generate_score(
            db=db,
            job_id=request_body.job_id,
            candidate_id=request_body.candidate_id
        )
        if score:
            return ScoreGenerationResponse(
                job_id=request_body.job_id,
                candidate_id=request_body.candidate_id,
                message="Score generation initiated successfully.",
                score_id=score.id  # Return score_id if available
            )
        else:
             # Handle cases where score generation might fail synchronously (e.g., invalid IDs)
             # Although the main processing is async, initial checks might fail.
            return ScoreGenerationResponse(
                 job_id=request_body.job_id,
                 candidate_id=request_body.candidate_id,
                 message="Score generation failed (e.g., invalid IDs or initial error)."
            )

    except Exception as e:
        # Log the exception details
        print(f"Error initiating score generation: {e}") # Basic logging
        raise HTTPException(status_code=500, detail=f"Failed to initiate score generation: {e}") 