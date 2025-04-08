from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging
import asyncio
from sqlalchemy import desc, asc # Import asc/desc

from recruitx_app.models.job import Job
from recruitx_app.models.candidate import Candidate
from recruitx_app.models.score import Score
# Rename the import
from recruitx_app.agents.simple_scoring_agent import OrchestrationAgent 
from recruitx_app.agents.jd_analysis_agent import JDAnalysisAgent # Import JD agent
from recruitx_app.services.agentic_rag_service import agentic_rag_service # Import Agentic RAG service

# Set up logging
logger = logging.getLogger(__name__)

class ScoringService:
    """
    Service for generating match scores using Agentic RAG principles.
    Steps:
    1. Decompose JD into requirement facets (LLM).
    2. Dynamically retrieve evidence for each facet from candidate docs (RAG).
    3. Calculate overall semantic similarity (Embeddings).
    4. Synthesize final score, providing facets, evidence, and similarity (LLM).
    """
    
    def __init__(self):
        # Use the orchestration agent for synthesis and JD agent for decomposition
        self.orchestration_agent = OrchestrationAgent()
        self.jd_analysis_agent = JDAnalysisAgent() # Add JD agent instance
    
    async def generate_score( 
        self, 
        db: Session, 
        job_id: int,
        candidate_id: int
    ) -> Optional[Score]:
        """
        Generates a match score using an enhanced three-step orchestration flow:
        1. Skill extraction (LLM)
        2. Semantic similarity (embeddings)
        3. Score synthesis (LLM with similarity input)
        
        Args:
            db: The database session
            job_id: The ID of the job
            candidate_id: The ID of the candidate
            
        Returns:
            The created Score object or None if an error occurred
        """
        job = db.query(Job).filter(Job.id == job_id).first()
        candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()

        if not job or not candidate or not job.description_raw or not candidate.resume_raw:
            logger.warning(f"Cannot generate score: Job {job_id} or Candidate {candidate_id} not found, or missing raw text.")
            return None
            
        logger.info(f"Starting Agentic RAG scoring process for Job {job_id}, Candidate {candidate_id}.")
        try:
            db_score: Optional[Score] = None # Define db_score variable here

            # --- Step 1: Decompose JD into Facets --- 
            logger.info(f"Step 1: Decomposing JD {job_id} into requirement facets.")
            job_facets = await self.jd_analysis_agent.decompose_job_description(
                job_id=job_id, 
                job_description=job.description_raw
            )

            if not job_facets:
                logger.error(f"JD decomposition failed for Job {job_id}. Cannot proceed with scoring.")
                # Create a score record indicating the failure
                db_score = Score(
                    job_id=job_id,
                    candidate_id=candidate_id,
                    overall_score=0.0, 
                    explanation="Failed during job description decomposition.",
                    details={"error": "JD decomposition failed"}
                )
                # Skip to saving the error score
            else:
                logger.info(f"Step 1 successful. Decomposed JD {job_id} into {len(job_facets)} facets.")
                
                # --- Step 2: Iterative Retrieval & Validation --- 
                logger.info(f"Step 2: Retrieving and validating evidence for {len(job_facets)} facets from Candidate {candidate_id} with refinement.")
                # Use the Agentic RAG service's new iterative method
                validated_evidence_results = await agentic_rag_service.iterative_retrieve_and_validate(
                    candidate_id=candidate_id,
                    facets=job_facets,
                    max_attempts_per_facet=2,  # Try up to 2 times for required facets with insufficient evidence
                    min_evidence_chunks=1,     # At least 1 relevant chunk per facet
                    n_results_per_facet=3,     # Retrieve 3 chunks per query
                    relevance_threshold=0.5    # Keep chunks with similarity >= 0.5
                )
                
                # Calculate metrics for logging
                total_facets = len(job_facets)
                required_facets = sum(1 for f in job_facets if f.is_required)
                facets_with_evidence = sum(1 for v in validated_evidence_results.values() if v is not None)
                required_facets_with_evidence = sum(1 for i, f in enumerate(job_facets) 
                                                 if f.is_required and validated_evidence_results.get(i) is not None)
                
                logger.info(f"Step 2 successful. Evidence found for {facets_with_evidence}/{total_facets} facets " +
                           f"({required_facets_with_evidence}/{required_facets} required facets).")

                # --- Step 3: Tool Integration - Enrich with External Data ---
                logger.info(f"Step 3: Enriching evidence with external market data for Job {job_id}.")
                
                # Get job title and location from the job object
                job_title = job.title if hasattr(job, 'title') and job.title else "Unknown Position"
                job_location = job.location if hasattr(job, 'location') and job.location else None
                
                # Call the AgenticRAG service to integrate external tool data
                enriched_data = await agentic_rag_service.enrich_evidence_with_external_data(
                    facets=job_facets,
                    validated_evidence=validated_evidence_results,
                    job_title=job_title,
                    location=job_location
                )
                
                external_data_success = False
                if enriched_data and "external_data" in enriched_data:
                    if enriched_data["external_data"].get("error") is None:
                        # Log successful data points retrieved
                        data_points = []
                        if enriched_data["external_data"].get("salary_benchmark"):
                            data_points.append("salary benchmarks")
                        if enriched_data["external_data"].get("market_insights"):
                            data_points.append("market demand insights")
                        if enriched_data["external_data"].get("skill_trends"):
                            data_points.append("skill trends")
                            
                        if data_points:
                            logger.info(f"Step 3 successful. Retrieved external data: {', '.join(data_points)}")
                            external_data_success = True
                        else:
                            logger.warning("Step 3 partial success. No specific data points retrieved.")
                    else:
                        logger.warning(f"Step 3 failed: {enriched_data['external_data'].get('error')}")
                else:
                    logger.warning("Step 3 failed: Invalid response from external data enrichment")

                # Count facets enriched with external data
                facets_enriched = 0
                if "facet_external_data" in enriched_data:
                    facets_enriched = len(enriched_data["facet_external_data"])
                
                if external_data_success:
                    logger.info(f"Enhanced {facets_enriched} facets with external market data")
                
                # --- Step 4 & 5: Calculate Similarity and Synthesize Score --- 
                logger.info(f"Steps 4+5: Synthesizing final score for Job {job_id}, Candidate {candidate_id}.")
                # Add a small delay between API calls
                await asyncio.sleep(1) 
                
                # Update to include the external data enrichment
                score_synthesis_result = await self.orchestration_agent.synthesize_score(
                     job_description=job.description_raw,
                     candidate_resume=candidate.resume_raw,
                     job_facets=job_facets,
                     retrieved_evidence=validated_evidence_results,
                     candidate_id=candidate_id,
                     external_data=enriched_data if external_data_success else None
                )

                overall_score = 0.0 # Default score
                explanation = "Score synthesis failed."
                details = score_synthesis_result # Store synthesis result by default
                
                if "error" in score_synthesis_result:
                     logger.error(f"Score synthesis failed: {score_synthesis_result.get('error', 'Unknown error')}")
                     explanation = f"Failed during score synthesis: {score_synthesis_result.get('error', 'Unknown error')}"
                     details = { 
                         "synthesis_error": score_synthesis_result,
                         "job_facets": [f.model_dump() for f in job_facets], # Include facets in error details
                         "retrieved_evidence_summary": {k: (v is not None) for k, v in validated_evidence_results.items()}, # Summarize evidence retrieval status
                         "external_data_status": "success" if external_data_success else "failed"
                     }
                else:
                     overall_score = score_synthesis_result.get("overall_score", 0.0)
                     explanation = score_synthesis_result.get("explanation", "No explanation provided.")
                     # Include facets, evidence summary, and external data status in success details
                     details = { 
                         "synthesis_result": score_synthesis_result,
                         "job_facets": [f.model_dump() for f in job_facets], 
                         "retrieved_evidence_summary": {k: (v is not None) for k, v in validated_evidence_results.items()},
                         "external_data_status": "success" if external_data_success else "failed",
                         "facets_enriched": facets_enriched
                     }
                     logger.info(f"Steps 4+5 successful. Final Score: {overall_score}")

                # Create the score record 
                db_score = Score(
                    job_id=job_id,
                    candidate_id=candidate_id,
                    overall_score=overall_score, 
                    explanation=explanation,
                    details=details # Store comprehensive details
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
    
    def get_scores_for_job(self, db: Session, job_id: int, min_score: Optional[float] = None, sort_by: str = "overall_score", sort_order: str = "desc") -> List[Score]:
        """Get all scores for a specific job, with filtering and sorting."""
        query = db.query(Score).filter(Score.job_id == job_id)
        
        # Apply minimum score filter if provided
        if min_score is not None:
            query = query.filter(Score.overall_score >= min_score)
            
        # Determine sorting column and direction
        sort_column = Score.overall_score if sort_by == "overall_score" else Score.created_at
        order_func = desc if sort_order == "desc" else asc
        
        query = query.order_by(order_func(sort_column))
        
        return query.all()
    
    def get_scores_for_candidate(self, db: Session, candidate_id: int) -> List[Score]:
        """Get all scores for a specific candidate."""
        return db.query(Score).filter(Score.candidate_id == candidate_id).all()

    # Removed batch_generate_scores as the synchronous version is in the API endpoint now
    # A more robust batch implementation would modify generate_score or use a different approach 