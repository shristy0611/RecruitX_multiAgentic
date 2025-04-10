from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import json
import logging # Import logging

from recruitx_app.models.job import Job
from recruitx_app.schemas.job import JobCreate, JobAnalysis
from recruitx_app.agents.jd_analysis_agent import JDAnalysisAgent
# Import the vector DB service
from recruitx_app.services.vector_db_service import vector_db_service
# Import the text splitter utility
from recruitx_app.utils.text_utils import split_text

logger = logging.getLogger(__name__) # Setup logger

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
        """
        Analyze a job description, update the database, and index the content for RAG.
        """
        job = self.get_job(db, job_id)
        if not job or not job.description_raw:
            logger.warning(f"Job {job_id} not found or has no raw description for analysis.")
            return None
        
        # Step 1: Analyze the job description using the agent
        logger.info(f"Starting JD analysis for job ID: {job_id}")
        analysis_result = await self.jd_analysis_agent.analyze_job_description(
            job_id=job.id, 
            job_description=job.description_raw
        )
        
        if not analysis_result:
            logger.error(f"JD analysis failed for job ID: {job_id}")
            return None

        # Step 2: Update the job record with analysis results
        try:
            analysis_dict = analysis_result.model_dump()
            job.analysis = analysis_dict
            db.commit()
            db.refresh(job)
            logger.info(f"Successfully updated job {job_id} with analysis results.")
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save analysis results for job {job_id}: {e}", exc_info=True)
            return None
            
        # Step 3: Chunk and index the raw description for RAG using the refined splitter
        try:
            logger.info(f"Starting chunking and indexing for job ID: {job_id}")
            
            # Use the refined split_text function
            # Using default chunk_size=1000, chunk_overlap=100 for now
            # These could be made configurable if needed
            chunks = split_text(job.description_raw)
            
            if not chunks:
                logger.warning(f"No text chunks generated by split_text for job ID: {job_id}")
                return analysis_dict # Return analysis even if indexing fails

            metadatas = []
            ids = []
            for i, chunk in enumerate(chunks):
                doc_id = f"job_{job_id}"
                chunk_id = f"{doc_id}_chunk_{i}"
                metadatas.append({
                    "doc_id": doc_id,
                    "doc_type": "job",
                    "job_id": job_id,
                    "chunk_index": i
                    # Add other relevant metadata? e.g., job title? Maybe later.
                })
                ids.append(chunk_id)
            
            # Add chunks to vector store
            logger.info(f"Attempting to index {len(chunks)} chunks for job ID: {job_id}")
            success = await vector_db_service.add_document_chunks(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            if success:
                logger.info(f"Successfully indexed {len(chunks)} chunks for job ID: {job_id}")
            else:
                logger.error(f"Vector DB service reported failure indexing chunks for job ID: {job_id}")
                # Don't fail the whole process, just log the error
                
        except Exception as e:
            logger.error(f"Error during chunking/indexing for job {job_id}: {e}", exc_info=True)
            # Log error but proceed, returning the analysis dict

        return analysis_dict # Return the analysis result 