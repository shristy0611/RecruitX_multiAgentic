from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

from recruitx_app.core.database import get_db
from recruitx_app.services.candidate_service import CandidateService
from recruitx_app.schemas.candidate import Candidate, CandidateCreate, CandidateAnalysis
from recruitx_app.utils.file_parser import extract_text_from_file

router = APIRouter()
candidate_service = CandidateService()

@router.post("/upload", response_model=Candidate, status_code=status.HTTP_201_CREATED)
async def upload_candidate_cv(
    name: str = Form(...),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a candidate CV file (PDF, DOCX, or TXT) and create a new candidate record.
    Analysis is not triggered automatically by this endpoint.
    """
    # Read the file content
    file_content = await file.read()
    
    # Extract text from the file
    resume_raw = extract_text_from_file(file_content, file.filename)
    
    if not resume_raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract text from the provided file. Supported formats: PDF, DOCX, TXT"
        )
    
    # Create candidate data Pydantic model
    candidate_data = CandidateCreate(
        name=name,
        email=email,
        phone=phone,
        resume_raw=resume_raw
    )
    
    # Create the candidate record using the service
    try:
        created_candidate = candidate_service.create_candidate(db=db, candidate_data=candidate_data)
        return created_candidate
    except Exception as e:
        # Catch potential database errors or other issues
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create candidate: {str(e)}"
        )

@router.get("/", response_model=List[Candidate])
def get_candidates(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """
    Get a list of all candidates with pagination.
    """
    candidates = candidate_service.get_candidates(db, skip=skip, limit=limit)
    return candidates

@router.get("/{candidate_id}", response_model=Candidate)
def get_candidate(
    candidate_id: int, 
    db: Session = Depends(get_db)
):
    """
    Get a specific candidate by ID.
    """
    candidate = candidate_service.get_candidate(db, candidate_id=candidate_id)
    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Candidate with ID {candidate_id} not found"
        )
    return candidate

@router.post("/{candidate_id}/analyze", response_model=Dict[str, Any])
async def analyze_candidate_cv(
    candidate_id: int, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Trigger CV analysis for a specific candidate.
    The analysis runs in the background.
    """
    # Check if candidate exists
    candidate = candidate_service.get_candidate(db, candidate_id=candidate_id)
    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Candidate with ID {candidate_id} not found"
        )
    
    # Check if analysis already exists to avoid re-running unnecessarily
    if candidate.analysis:
        return {"message": "Analysis already exists for this candidate.", "analysis": candidate.analysis}
        
    # Add the analysis task to the background
    background_tasks.add_task(candidate_service.analyze_cv, db, candidate_id)
    
    return {"message": f"CV analysis triggered in the background for candidate ID {candidate_id}."} 