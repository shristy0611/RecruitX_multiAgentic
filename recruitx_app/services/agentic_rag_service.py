import logging
from typing import List, Dict, Optional, Any

from recruitx_app.schemas.job import JobRequirementFacet
from recruitx_app.services.vector_db_service import vector_db_service
from recruitx_app.services.external_tool_service import external_tool_service
from recruitx_app.utils.text_utils import cosine_similarity

logger = logging.getLogger(__name__)

class AgenticRAGService:
    """
    Service responsible for implementing Agentic RAG principles,
    starting with dynamic retrieval based on decomposed job facets.
    """

    async def retrieve_evidence_for_facets(
        self,
        candidate_id: int,
        facets: List[JobRequirementFacet],
        n_results_per_facet: int = 3
    ) -> Dict[int, Dict[str, Any]]: # Return Dict mapping facet index to results
        """
        Retrieves relevant text chunks from a candidate's documents for each job requirement facet.

        Args:
            candidate_id: The ID of the candidate whose documents to search.
            facets: A list of JobRequirementFacet objects from the decomposed JD.
            n_results_per_facet: The number of document chunks to retrieve for each facet.

        Returns:
            A dictionary where keys are the *indices* of the input facets and values are
            the ChromaDB query result dictionaries for that facet (or None if query fails).
            Example: {0: {'ids': [...], 'documents': [...], ...}, 1: None, 2: {...}}
        """
        logger.info(f"Starting dynamic evidence retrieval for Candidate ID: {candidate_id} across {len(facets)} facets.")
        evidence: Dict[int, Optional[Dict[str, Any]]] = {}

        # Define the base 'where' filter for candidate documents
        where_filter = {
            "doc_type": "candidate",
            "candidate_id": candidate_id
        }

        for i, facet in enumerate(facets):
            # --- 1. Formulate Query --- 
            # Simple strategy for now: Combine detail and context.
            # Future: Could use more sophisticated query generation based on facet_type.
            query_text = f"{facet.detail} {facet.context or ''}".strip()
            logger.debug(f"Facet {i} ({facet.facet_type}: '{facet.detail}') - Query: '{query_text[:100]}...'" )

            if not query_text:
                logger.warning(f"Skipping empty query for facet index {i}")
                evidence[i] = None
                continue

            # --- 2. Query Vector Store --- 
            try:
                facet_results = await vector_db_service.query_collection(
                    query_texts=[query_text],
                    n_results=n_results_per_facet,
                    where=where_filter,
                    # include=['metadatas', 'documents', 'distances'] # Already included by default in service
                )
                
                if facet_results and facet_results.get('ids') and facet_results['ids'][0]:
                    num_found = len(facet_results['ids'][0])
                    logger.debug(f"Facet {i} query found {num_found} results.")
                    evidence[i] = facet_results
                else:
                    logger.debug(f"Facet {i} query found no results.")
                    evidence[i] = None # Store None if no results found

            except Exception as e:
                logger.error(f"Error querying vector store for facet index {i} ('{query_text[:50]}...'): {e}", exc_info=True)
                evidence[i] = None # Store None on error
        
        logger.info(f"Finished dynamic evidence retrieval for Candidate ID: {candidate_id}. Found evidence for {sum(1 for r in evidence.values() if r is not None)} facets.")
        return evidence

    async def validate_evidence_relevance(
        self,
        facets: List[JobRequirementFacet],
        retrieved_evidence: Dict[int, Optional[Dict[str, Any]]],
        relevance_threshold: float = 0.5 # Configurable threshold
    ) -> Dict[int, Optional[Dict[str, Any]]]:
        """
        Validates the relevance of retrieved evidence chunks against the original facet detail
        using embedding similarity.

        Args:
            facets: The original list of JobRequirementFacet objects.
            retrieved_evidence: The dictionary returned by retrieve_evidence_for_facets.
            relevance_threshold: The minimum cosine similarity score for an evidence chunk to be considered relevant.

        Returns:
            A dictionary similar to retrieved_evidence, but with irrelevant chunks filtered out
            from the 'documents', 'ids', 'metadatas', and 'distances' lists within each facet's results.
        """
        logger.info(f"Starting evidence relevance validation for {len(facets)} facets.")
        validated_evidence: Dict[int, Optional[Dict[str, Any]]] = {}

        # Prepare lists for batch embedding generation
        facet_details_to_embed = []
        evidence_chunks_to_embed = []
        facet_indices_map = [] # To map results back to original facet index
        chunk_indices_map = [] # To map results back to chunk index within a facet

        # Collect all text that needs embedding
        for i, facet in enumerate(facets):
            if i in retrieved_evidence and retrieved_evidence[i] is not None:
                evidence_data = retrieved_evidence[i]
                if evidence_data.get('documents') and evidence_data['documents'][0]:
                    facet_details_to_embed.append(facet.detail)
                    facet_indices_map.append(i)
                    
                    chunks = evidence_data['documents'][0]
                    current_chunk_indices = []
                    for j, chunk in enumerate(chunks):
                        evidence_chunks_to_embed.append(chunk)
                        current_chunk_indices.append(j)
                    chunk_indices_map.append(current_chunk_indices)
                else:
                     validated_evidence[i] = None # No evidence to validate
            else:
                validated_evidence[i] = None # No evidence retrieved initially
                
        if not facet_details_to_embed: # No evidence found for any facet
            logger.info("No evidence found to validate.")
            return validated_evidence

        # Perform batch embedding generation
        logger.info(f"Generating embeddings for {len(facet_details_to_embed)} facet details and {len(evidence_chunks_to_embed)} evidence chunks.")
        try:
            all_texts_to_embed = facet_details_to_embed + evidence_chunks_to_embed
            all_embeddings = await vector_db_service.generate_embeddings(all_texts_to_embed)
            
            if not all_embeddings or len(all_embeddings) != len(all_texts_to_embed):
                logger.error("Failed to generate embeddings for validation. Skipping relevance check.")
                # Return the original evidence if embedding fails
                return retrieved_evidence 

            # Split embeddings back
            num_facets = len(facet_details_to_embed)
            facet_embeddings = all_embeddings[:num_facets]
            evidence_embeddings = all_embeddings[num_facets:]
            
        except Exception as e:
            logger.error(f"Error during embedding generation for validation: {e}. Skipping relevance check.", exc_info=True)
            return retrieved_evidence

        # Validate relevance using cosine similarity
        current_evidence_embedding_index = 0
        for idx, (facet_index, facet_embedding) in enumerate(zip(facet_indices_map, facet_embeddings)):
            original_evidence_data = retrieved_evidence[facet_index]
            if original_evidence_data is None: continue # Should not happen based on logic above, but safety check

            relevant_indices = []
            original_chunk_count = len(original_evidence_data['documents'][0])
            
            # Get the embeddings for the chunks of this specific facet
            facet_chunk_embeddings = evidence_embeddings[current_evidence_embedding_index : current_evidence_embedding_index + original_chunk_count]
            
            for chunk_idx, chunk_embedding in enumerate(facet_chunk_embeddings):
                similarity = cosine_similarity(facet_embedding, chunk_embedding)
                if similarity is not None and similarity >= relevance_threshold:
                    relevant_indices.append(chunk_idx)
                # else: logger.debug(f"Chunk {chunk_idx} for facet {facet_index} is irrelevant (Similarity: {similarity:.4f})")
            
            # Filter the original evidence data based on relevant indices
            if relevant_indices:
                filtered_data = {
                    'ids': [[original_evidence_data['ids'][0][k] for k in relevant_indices]],
                    'documents': [[original_evidence_data['documents'][0][k] for k in relevant_indices]],
                    'metadatas': [[original_evidence_data['metadatas'][0][k] for k in relevant_indices]],
                    'distances': [[original_evidence_data['distances'][0][k] for k in relevant_indices]]
                    # Add other included fields if necessary
                }
                validated_evidence[facet_index] = filtered_data
                logger.debug(f"Facet {facet_index}: Kept {len(relevant_indices)} of {original_chunk_count} evidence chunks after relevance check.")
            else:
                 validated_evidence[facet_index] = None # No relevant chunks found
                 logger.debug(f"Facet {facet_index}: No relevant evidence chunks found after check.")
            
            current_evidence_embedding_index += original_chunk_count # Move to the next set of chunk embeddings

        logger.info(f"Finished evidence relevance validation. Kept evidence for {sum(1 for r in validated_evidence.values() if r is not None)} facets.")
        return validated_evidence

    async def iterative_retrieve_and_validate(
        self,
        candidate_id: int,
        facets: List[JobRequirementFacet],
        max_attempts_per_facet: int = 2,
        min_evidence_chunks: int = 1,
        n_results_per_facet: int = 3,
        relevance_threshold: float = 0.5
    ) -> Dict[int, Optional[Dict[str, Any]]]:
        """
        Implements the iterative refinement loop for evidence retrieval and validation.
        Makes multiple attempts for required facets that have insufficient evidence.
        
        Args:
            candidate_id: The ID of the candidate whose documents to search.
            facets: A list of JobRequirementFacet objects from the decomposed JD.
            max_attempts_per_facet: Maximum number of query refinement attempts per facet.
            min_evidence_chunks: Minimum number of evidence chunks needed to consider a facet covered.
            n_results_per_facet: Number of chunks to retrieve per query.
            relevance_threshold: Threshold for the relevance validation.
            
        Returns:
            A dictionary with evidence for each facet, including any results from refinement attempts.
        """
        logger.info(f"Starting iterative evidence retrieval and validation for Candidate {candidate_id} across {len(facets)} facets.")
        
        # Initial retrieval and validation
        evidence_results = await self.retrieve_evidence_for_facets(
            candidate_id=candidate_id,
            facets=facets,
            n_results_per_facet=n_results_per_facet
        )
        
        validated_results = await self.validate_evidence_relevance(
            facets=facets,
            retrieved_evidence=evidence_results,
            relevance_threshold=relevance_threshold
        )
        
        # Track attempts per facet
        attempt_count = {i: 1 for i in range(len(facets))}
        final_results = validated_results.copy()
        
        # Identify required facets that need more evidence
        for i, facet in enumerate(facets):
            # Only retry for required facets with insufficient evidence
            if (facet.is_required and 
                (i not in validated_results or 
                 validated_results[i] is None or 
                 len(validated_results[i]['documents'][0]) < min_evidence_chunks)):
                
                logger.info(f"Required facet {i} ({facet.facet_type}: '{facet.detail}') has insufficient evidence. Will attempt refinement.")
                
                # Attempt refinements
                while attempt_count[i] < max_attempts_per_facet:
                    attempt_count[i] += 1
                    current_attempt = attempt_count[i]
                    
                    # Refine the query based on attempt number and facet type
                    refined_query = self._refine_query(facet, current_attempt)
                    if not refined_query:
                        logger.warning(f"Could not generate refined query for facet {i} on attempt {current_attempt}.")
                        break
                    
                    logger.info(f"Attempt {current_attempt} for facet {i}: Using refined query: '{refined_query[:100]}...'")
                    
                    # Query using the refined query
                    where_filter = {
                        "doc_type": "candidate",
                        "candidate_id": candidate_id
                    }
                    
                    try:
                        refined_results = await vector_db_service.query_collection(
                            query_texts=[refined_query],
                            n_results=n_results_per_facet,
                            where=where_filter
                        )
                        
                        # Validate the new results
                        if refined_results and refined_results.get('documents') and refined_results['documents'][0]:
                            # Embedding comparison between facet and each chunk for relevance check
                            facet_embedding = await vector_db_service.generate_embeddings([facet.detail])
                            if not facet_embedding:
                                logger.warning(f"Failed to generate embedding for facet {i}. Skipping relevance check.")
                                continue
                                
                            chunk_embeddings = await vector_db_service.generate_embeddings(refined_results['documents'][0])
                            if not chunk_embeddings:
                                logger.warning(f"Failed to generate embeddings for refined results of facet {i}. Skipping relevance check.")
                                continue
                            
                            # Identify relevant chunks (using same threshold as validate_evidence_relevance)
                            relevant_indices = []
                            for j, chunk_embedding in enumerate(chunk_embeddings):
                                similarity = cosine_similarity(facet_embedding[0], chunk_embedding)
                                if similarity is not None and similarity >= relevance_threshold:
                                    relevant_indices.append(j)
                            
                            # If we found relevant chunks, add them to final results
                            if relevant_indices:
                                filtered_data = {
                                    'ids': [[refined_results['ids'][0][k] for k in relevant_indices]],
                                    'documents': [[refined_results['documents'][0][k] for k in relevant_indices]],
                                    'metadatas': [[refined_results['metadatas'][0][k] for k in relevant_indices]],
                                    'distances': [[refined_results['distances'][0][k] for k in relevant_indices]]
                                }
                                
                                # Merge with existing results if any
                                if i in final_results and final_results[i] is not None:
                                    # Extend each list with the new results
                                    for key in ['ids', 'documents', 'metadatas', 'distances']:
                                        final_results[i][key][0].extend(filtered_data[key][0])
                                else:
                                    final_results[i] = filtered_data
                                
                                logger.info(f"Attempt {current_attempt} for facet {i} found {len(relevant_indices)} relevant chunks.")
                                
                                # If we have enough evidence now, break the loop
                                if len(final_results[i]['documents'][0]) >= min_evidence_chunks:
                                    logger.info(f"Found sufficient evidence for facet {i} after {current_attempt} attempts.")
                                    break
                            else:
                                logger.info(f"Attempt {current_attempt} for facet {i} found no relevant chunks.")
                        else:
                            logger.info(f"Attempt {current_attempt} for facet {i} returned no results.")
                    
                    except Exception as e:
                        logger.error(f"Error during refinement attempt {current_attempt} for facet {i}: {e}", exc_info=True)
                        continue
                
                # After all attempts, if still no evidence, log it
                if i not in final_results or final_results[i] is None or len(final_results[i]['documents'][0]) < min_evidence_chunks:
                    logger.warning(f"Failed to find sufficient evidence for required facet {i} after {attempt_count[i]} attempts.")
        
        evidence_count = sum(1 for r in final_results.values() if r is not None)
        chunk_count = sum(len(r['documents'][0]) for r in final_results.values() if r is not None)
        logger.info(f"Completed iterative retrieval. Found evidence for {evidence_count} facets, total of {chunk_count} relevant chunks.")
        
        return final_results
    
    def _refine_query(self, facet: JobRequirementFacet, attempt: int) -> Optional[str]:
        """
        Refines a query for a facet based on the attempt number and facet type.
        Uses different strategies for different facet types and attempts.
        
        Args:
            facet: The JobRequirementFacet to refine the query for.
            attempt: The current attempt number (1-based, where 1 is the original query).
            
        Returns:
            A refined query string or None if no refinement is possible.
        """
        detail = facet.detail.strip()
        context = facet.context.strip() if facet.context else ""
        
        # Base case - should not reach here normally since attempt 1 uses the original query
        if attempt <= 1 or not detail:
            return f"{detail} {context}".strip()
        
        # Different refinement strategies based on facet type and attempt number
        if facet.facet_type == 'skill':
            # For skills, try broader or related terms
            if attempt == 2:
                # For second attempt, try finding variations or removing qualifiers
                # Example: "Python 3+ years" -> "Python programming"
                detail_parts = detail.split()
                if len(detail_parts) > 1:
                    # Keep only the first part (likely the core skill name)
                    core_skill = detail_parts[0]
                    # Add common generalizations for skills
                    return f"{core_skill} experience programming development"
                else:
                    # Add common skill-related terms
                    return f"{detail} experience programming development"
        
        elif facet.facet_type == 'experience':
            # For experience, try more generic experience queries
            if attempt == 2:
                # Remove years and specific qualifiers
                detail_lowered = detail.lower()
                # Extract what the experience is about by removing year qualifiers
                for year_pattern in ['years', 'year', 'yr', 'yrs', '+', '-']:
                    detail_lowered = detail_lowered.replace(year_pattern, ' ')
                # Clean up multiple spaces
                detail_lowered = ' '.join(detail_lowered.split())
                return f"experience with {detail_lowered} background"
        
        elif facet.facet_type == 'education':
            # For education, try alternative credentials or related fields
            if attempt == 2:
                detail_lowered = detail.lower()
                if 'degree' in detail_lowered or 'bachelor' in detail_lowered or 'master' in detail_lowered:
                    # Try to extract the field of study
                    # Example: "Bachelor's Degree in Computer Science" -> "Computer Science education"
                    if 'in ' in detail_lowered:
                        field = detail_lowered.split('in ', 1)[1].strip()
                        return f"{field} education qualification training"
                    else:
                        return f"{detail} education qualification training"
                else:
                    return f"{detail} education qualification training"
        
        elif facet.facet_type == 'certification':
            # For certifications, try more general certification terms
            if attempt == 2:
                return f"{detail} certified qualification"
        
        elif facet.facet_type == 'responsibility':
            # For responsibilities, try related activities/roles
            if attempt == 2:
                return f"{detail} responsibility role task"
        
        # Default fallback for other facet types or if specific strategies fail
        if attempt == 2:
            # Just add some general terms for the second attempt
            return f"{detail} {context} related similar"
        
        # If no refinement strategy worked
        return None

    async def get_external_data_for_job_market_fit(
        self,
        job_title: str,
        location: Optional[str],
        skills: List[str],
        experience_years: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Tool integration function that gathers external market and salary data
        to enrich job-candidate matching with market insights.
        
        Args:
            job_title: The job title being evaluated
            location: Optional job location
            skills: List of skills mentioned in the job description
            experience_years: Optional experience requirement
            
        Returns:
            Dictionary containing enriched data from external sources
        """
        logger.info(f"Gathering external market data for job '{job_title}' with {len(skills)} skills")
        
        # Initialize the results dictionary
        external_data = {
            "salary_benchmark": None,
            "market_insights": None,
            "skill_trends": None,
            "error": None
        }
        
        try:
            # Get salary benchmark data (async call)
            salary_response = await external_tool_service.get_salary_benchmark(
                job_title=job_title,
                location=location,
                experience_years=experience_years,
                skills=skills[:10] if skills else None  # Limit to top 10 skills
            )
            
            if salary_response["success"]:
                external_data["salary_benchmark"] = salary_response["data"]
                logger.debug(f"Retrieved salary data for {job_title}: {salary_response['data']['salary_data']['median']} USD median")
            else:
                logger.warning(f"Failed to retrieve salary data: {salary_response['error']}")
            
            # Get job market insights (async call)
            market_response = await external_tool_service.get_job_market_insights(
                job_title=job_title,
                skills=skills[:10] if skills else None,  # Limit to top 10 skills
                location=location,
                time_period="6months"  # Default to 6 months of data
            )
            
            if market_response["success"]:
                external_data["market_insights"] = market_response["data"]
                logger.debug(f"Retrieved market insights for {job_title}: Growth rate {market_response['data']['market_data']['demand_growth_rate']}")
            else:
                logger.warning(f"Failed to retrieve market insights: {market_response['error']}")
            
            # Get skill-specific demand trends (async call)
            # Only request if we have skills and they're not too many
            if skills and len(skills) <= 15:
                skill_response = await external_tool_service.get_skill_demand_trends(
                    skills=skills,
                    location=location,
                    time_period="6months"  # Default to 6 months of data
                )
                
                if skill_response["success"]:
                    external_data["skill_trends"] = skill_response["data"]
                    logger.debug(f"Retrieved skill trends for {len(skills)} skills")
                else:
                    logger.warning(f"Failed to retrieve skill trends: {skill_response['error']}")
            
            # Basic error handling
            if not any([external_data["salary_benchmark"], external_data["market_insights"], external_data["skill_trends"]]):
                external_data["error"] = "Failed to retrieve any external data"
                logger.error("All external data retrieval attempts failed")
                
        except Exception as e:
            error_msg = f"Error gathering external market data: {str(e)}"
            external_data["error"] = error_msg
            logger.error(error_msg, exc_info=True)
        
        return external_data
    
    async def enrich_evidence_with_external_data(
        self,
        facets: List[JobRequirementFacet],
        validated_evidence: Dict[int, Optional[Dict[str, Any]]],
        job_title: str,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhances evidence from vector retrieval with external data for facets
        that can benefit from market insights.
        
        Args:
            facets: List of job requirement facets
            validated_evidence: The validated evidence for each facet
            job_title: Job title to use for external data lookup
            location: Optional job location
            
        Returns:
            Dictionary containing enriched evidence and external data
        """
        # Extract skills from all facets of type "skill"
        skills = []
        experience_years = None
        
        for facet in facets:
            if facet.facet_type.lower() == "skill":
                # Add the skill detail to our list
                skills.append(facet.detail)
            
            # Try to extract years of experience from experience facets
            if facet.facet_type.lower() == "experience" and not experience_years:
                # Simple extraction of years from text like "5+ years of experience"
                import re
                exp_match = re.search(r'(\d+)[\+]?\s*years?', facet.detail.lower())
                if exp_match:
                    try:
                        experience_years = int(exp_match.group(1))
                        logger.debug(f"Extracted {experience_years} years of experience requirement")
                    except ValueError:
                        pass
        
        logger.info(f"Getting external data for {job_title} with {len(skills)} skills and {experience_years if experience_years else 'unknown'} years experience")
        
        # Get external data from our service
        external_data = await self.get_external_data_for_job_market_fit(
            job_title=job_title,
            location=location,
            skills=skills,
            experience_years=experience_years
        )
        
        # Map external data to relevant facets
        facet_external_data = {}
        
        for i, facet in enumerate(facets):
            # Skip facets with no evidence (nothing to enrich)
            if i not in validated_evidence or validated_evidence[i] is None:
                continue
                
            facet_type = facet.facet_type.lower()
            facet_external_data[i] = {}
            
            # Add salary data for compensation facets
            if facet_type == "compensation" and external_data["salary_benchmark"]:
                facet_external_data[i]["salary_benchmark"] = external_data["salary_benchmark"]["salary_data"]
                logger.debug(f"Added salary data to compensation facet {i}")
                
            # Add job growth data for job title
            if facet_type in ["experience", "responsibilities"] and external_data["market_insights"]:
                growth_data = {
                    "demand_growth": external_data["market_insights"]["market_data"]["demand_growth_rate"],
                    "competition_level": external_data["market_insights"]["market_data"]["competition_level"]
                }
                facet_external_data[i]["market_growth"] = growth_data
                logger.debug(f"Added market growth data to {facet_type} facet {i}")
                
            # Add skill-specific data for skill facets
            if facet_type == "skill" and external_data["skill_trends"]:
                # Find this specific skill in the skill trends data
                facet_skill = facet.detail.lower()
                
                for skill_trend in external_data["skill_trends"]["skill_trends"]:
                    if facet_skill in skill_trend["skill"].lower():
                        facet_external_data[i]["skill_trend"] = {
                            "demand_growth_rate": skill_trend["demand_growth_rate"],
                            "popularity_rank": skill_trend["popularity_rank"],
                            "is_emerging": skill_trend.get("is_emerging", False),
                            "salary_impact": skill_trend.get("average_salary_impact", "Unknown")
                        }
                        logger.debug(f"Added skill trend data to skill facet {i} ({facet_skill})")
                        break
        
        return {
            "external_data": external_data,
            "facet_external_data": facet_external_data
        }

# Instantiate the service for easy import
agentic_rag_service = AgenticRAGService() 