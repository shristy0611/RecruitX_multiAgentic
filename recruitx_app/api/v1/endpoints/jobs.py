from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from typing import List, Optional

from recruitx_app.core.database import get_db
from recruitx_app.services.job_service import JobService
from recruitx_app.schemas.job import Job, JobCreate, JobAnalysis
from recruitx_app.utils.file_parser import extract_text_from_file

router = APIRouter()
job_service = JobService()

@router.get("/", response_model=List[Job])
def get_jobs(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """
    Get a list of all jobs with pagination.
    """
    jobs = job_service.get_jobs(db, skip=skip, limit=limit)
    return jobs

@router.get("/{job_id}", response_model=Job)
def get_job(
    job_id: int, 
    db: Session = Depends(get_db)
):
    """
    Get a specific job by ID.
    """
    job = job_service.get_job(db, job_id=job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Job with ID {job_id} not found"
        )
    return job

@router.post("/", response_model=Job)
def create_job(
    job_data: JobCreate, 
    db: Session = Depends(get_db)
):
    """
    Create a new job (manual entry).
    """
    return job_service.create_job(db, job_data=job_data)

@router.post("/upload", response_model=Job)
async def upload_job_description(
    title: str = Form(...),
    company: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a job description file (PDF, DOCX, or TXT) and create a new job.
    """
    # Read the file content
    file_content = await file.read()
    
    # Extract text from the file
    description_raw = extract_text_from_file(file_content, file.filename)
    
    if not description_raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract text from the provided file. Supported formats: PDF, DOCX, TXT"
        )
    
    # Create job data
    job_data = JobCreate(
        title=title,
        company=company,
        location=location,
        description_raw=description_raw
    )
    
    # Create the job
    return job_service.create_job(db, job_data=job_data)

@router.post("/{job_id}/analyze", response_model=JobAnalysis)
async def analyze_job(
    job_id: int, 
    db: Session = Depends(get_db)
):
    """
    Analyze a job description using AI to extract structured information.
    """
    # Check if job exists
    job = job_service.get_job(db, job_id=job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Job with ID {job_id} not found"
        )
    
    # Analyze the job
    analysis_result = await job_service.analyze_job(db, job_id=job_id)
    
    if not analysis_result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze the job description"
        )
    
    return JobAnalysis(**analysis_result) 