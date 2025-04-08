import google.generativeai as genai
import json
import logging
from typing import Dict, Any, Optional, List
import asyncio
import numpy as np

from recruitx_app.core.config import settings
from recruitx_app.utils.retry_utils import call_gemini_with_backoff
# Import the vector DB service
from recruitx_app.services.vector_db_service import vector_db_service
# Import text utilities for cosine similarity
from recruitx_app.utils.text_utils import cosine_similarity
# Import JobRequirementFacet for type hints
from recruitx_app.schemas.job import JobRequirementFacet

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt 1: Extract Key Skills
SKILL_EXTRACTION_PROMPT = """
You are an AI assistant specializing in analyzing recruitment documents.
Review the job description and candidate resume provided below.

JOB DESCRIPTION:
--------------------
{job_description}
--------------------

CANDIDATE RESUME:
--------------------
{candidate_resume}
--------------------

Your primary task is to identify and extract the key skills mentioned in both texts.
Prioritize technical skills, required software/tools, programming languages, methodologies, and certifications explicitly mentioned as requirements in the job description.
Also, extract relevant domain experience and directly applicable soft skills from both texts. Analyze the context of skills mentioned in the resume (e.g., years of experience, project relevance) to inform your extraction.
Ignore skills listed under purely personal sections like 'Hobbies' or 'Interests' unless they are directly relevant to the job requirements (e.g., a design hobby for a design role).

Return your response ONLY as a valid JSON object in the following format, with no introductory text or explanations:
{{
    "job_skills": ["skill1", "skill2", ...],
    "candidate_skills": ["skillA", "skillB", ...]
}}
"""

# Prompt 2: Synthesize Score - Updated for Agentic RAG with External Data
SCORE_SYNTHESIS_PROMPT = """
You are an AI-powered recruitment assistant evaluating a candidate's suitability for a specific job role.
Using Agentic RAG techniques, you'll analyze requirement facets with direct evidence from the candidate's resume, plus overall semantic similarity and external market data.

JOB DESCRIPTION:
--------------------
{job_description}
--------------------

CANDIDATE RESUME:
--------------------
{candidate_resume}
--------------------

SEMANTIC SIMILARITY SCORE: {semantic_similarity}
(This is a cosine similarity score between the JD and Resume embeddings, ranging from -1 to 1, where 1 means perfect similarity, 0 means no correlation, and -1 means opposite)

REQUIREMENT FACETS WITH EVIDENCE:
{facets_with_evidence}

{external_data_section}

Your primary task is to evaluate how well the candidate matches each specific requirement facet based on the retrieved evidence and available external data.

For each requirement facet:
1. Assess if there's strong evidence (direct matches), weak evidence (partial/implied matches), or no evidence in the candidate's resume
2. For mandatory requirements (is_required=true), weigh these more heavily in your assessment
3. Consider the quality and specificity of the evidence - e.g., does it show the right experience level or context?
4. When external market data is available for a facet (especially for skills/compensation), incorporate these insights into your evaluation

Based on this comprehensive facet-by-facet analysis:
1. Calculate an overall match score: An integer between 0 (no fit) and 100 (perfect fit)
   * Weight mandatory facets more heavily
   * Consider both presence of evidence and quality/relevance of evidence
   * Use the semantic similarity score as a general signal about overall fit
   * Factor in market-relevant data (e.g., high-demand skills should boost score slightly)
   
2. Provide an explanation with:
   * Clear assessment of how well the candidate meets key MANDATORY requirements (at least 2-3 examples)
   * Any notable OPTIONAL requirements they meet well
   * 1-2 significant gaps or weaknesses in their match
   * Brief mention of how the semantic similarity score relates to your findings
   * When available, include insights from external data (e.g., "Candidate has experience with Python, which has seen 23% demand growth in the job market")

Return your response ONLY as a valid JSON object in the following format, with no introductory text or explanations:
{{
    "overall_score": <integer score 0-100>,
    "explanation": "<explanation with assessment of key requirements, strengths, and gaps>"
}}
"""

class OrchestrationAgent: # Renamed from SimpleScoringAgent
    """
    An agent that orchestrates scoring using the Agentic RAG approach.
    Uses API key rotation for reliability.
    """
    
    def __init__(self):
        self.model_name = settings.GEMINI_PRO_MODEL
        # Key rotation is handled by settings.get_next_api_key()
        # Initial configuration happens once here, but _get_gemini_model can reconfigure if needed
        genai.configure(api_key=settings.get_next_api_key())
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    
    def _get_gemini_model(self, purpose="general"): # Added purpose for potential future config tweaks
        """Get the Gemini model, explicitly rotating API keys via settings for every call."""
        # --- Force rotation on every call --- 
        api_key = settings.get_next_api_key() # Get the strictly next key
        logger.info(f"Rotating to API Key ending in: ...{api_key[-4:]} for {purpose}")
        # -------------------------------------
        
        genai.configure(api_key=api_key) # Configure with the new key
        try:
            # Configure for JSON output
            generation_config = genai.GenerationConfig(
                temperature=0.1, # Low temp for structured output
                top_p=0.95,
                top_k=40,
                response_mime_type="application/json", 
            )
            model = genai.GenerativeModel(
                self.model_name,
                safety_settings=self.safety_settings,
                generation_config=generation_config
            )
            return model
        except Exception as e:
            logger.error(f"Fatal error initializing Gemini model even after forced key rotation: {e}")
            raise e # Re-raise the exception if configuration fails
    
    async def extract_skills(self, job_description: str, candidate_resume: str) -> Dict[str, Any]:
        """
        First step: Extracts job and candidate skills using an LLM call.
        
        Returns:
            Dictionary containing lists of job_skills and candidate_skills, or an error structure.
        """
        # logger.info("--- SIMULATING Skill Extraction --- ")
        # await asyncio.sleep(0.5) # Simulate processing time
        
        # # --- START SIMULATION ---
        # # Hardcoded dummy data - replace or remove when using real API
        # dummy_skills = {
        #     "job_skills": ["UX/UI Design", "Figma", "Sketch", "Wireframing", "Prototyping", "User Research", "Mobile Design", "Web Application Design"],
        #     "candidate_skills": ["Data Analysis", "Healthcare Analytics", "SQL", "Python", "Pandas", "PowerBI", "EMR Data", "HIPAA Compliance"]
        # }
        # logger.info(f"Simulation returning dummy skills: {dummy_skills}")
        # return dummy_skills
        # # --- END SIMULATION ---

        # --- Original API Call Logic --- 
        try:
            prompt = SKILL_EXTRACTION_PROMPT.format(
                job_description=job_description,
                candidate_resume=candidate_resume
            )
            model = self._get_gemini_model(purpose="skill_extraction")
            
            response = await call_gemini_with_backoff(
                model.generate_content,
                prompt,
                stream=False
            )
            
            response_text = response.text
            logger.debug(f"Skill extraction raw response: {response_text[:200]}...")
            result = json.loads(response_text)
            # Basic validation
            if "job_skills" in result and "candidate_skills" in result:
                 return result
            else:
                 logger.warning(f"Skill extraction response missing expected keys: {result}")
                 return {"error": "Invalid format from skill extraction", "details": result}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from skill extraction: {e}")
            # Attempt to access potentially blocked content details if available
            try: 
                error_details = response.prompt_feedback
            except Exception: 
                 error_details = "No further details available."
            return {"error": "Failed to parse skill extraction JSON", "details": error_details}        
        except Exception as e:
            logger.error(f"Error in skill extraction agent step: {e}", exc_info=True)
            return {"error": str(e)}
        # --- End Original Logic ---

    async def synthesize_score(
        self, 
        job_description: str, 
        candidate_resume: str, 
        job_facets: List[JobRequirementFacet], 
        retrieved_evidence: Dict[int, Dict], 
        candidate_id: int,
        external_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates the final score and explanation based on structured job facets, 
        retrieved evidence for each facet, semantic similarity, and external data.
        
        Args:
            job_description: Raw text of the job description
            candidate_resume: Raw text of the candidate's resume
            job_facets: List of JobRequirementFacet objects from decomposition
            retrieved_evidence: Dictionary mapping facet indices to retrieval results
            candidate_id: ID of the candidate (for logging)
            external_data: Optional dictionary containing external market data
            
        Returns:
            Dictionary with overall_score, explanation, and optional metadata
        """
        # Default values
        semantic_similarity_score = 0.0
        
        try:
            # --- Format Facets with Evidence for the Prompt ---
            facets_with_evidence_text = self._format_facets_with_evidence(job_facets, retrieved_evidence, external_data)
            logger.debug(f"Formatted {len(job_facets)} facets with evidence and external data")
            
            # --- Semantic Similarity Calculation Step ---
            logger.info(f"Generating embeddings for JD and CV to calculate semantic similarity")
            jd_cv_embeddings = await vector_db_service.generate_embeddings(
                texts=[job_description, candidate_resume]
            )
            
            if jd_cv_embeddings and len(jd_cv_embeddings) >= 2:
                jd_embedding = jd_cv_embeddings[0]
                cv_embedding = jd_cv_embeddings[1]
                
                similarity = cosine_similarity(jd_embedding, cv_embedding)
                # Ensure similarity is a float before formatting
                if isinstance(similarity, (list, np.ndarray)) and len(similarity) > 0 and isinstance(similarity[0], (list, np.ndarray)) and len(similarity[0]) > 0:
                    # Extract the float value (assuming nested list/array like [[score]])
                    semantic_similarity_score = float(similarity[0][0])
                elif isinstance(similarity, (float, int)):
                    semantic_similarity_score = float(similarity)
                else:
                    # Handle unexpected format or None
                    logger.warning(f"Failed to calculate or extract valid cosine similarity (received: {similarity}), using default value 0.0")
                    semantic_similarity_score = 0.0 # Reset to default if extraction failed

                logger.info(f"Calculated semantic similarity: {semantic_similarity_score:.4f}") # Log the final value
            else:
                logger.warning("Failed to generate embeddings for similarity calculation")
            
            # --- Format External Data Section ---
            external_data_section = self._format_external_data_section(external_data)
            
            # --- Score Synthesis API Call ---
            prompt = SCORE_SYNTHESIS_PROMPT.format(
                job_description=job_description,
                candidate_resume=candidate_resume,
                facets_with_evidence=facets_with_evidence_text,
                semantic_similarity=f"{semantic_similarity_score:.4f}",
                external_data_section=external_data_section
            )
            
            model = self._get_gemini_model(purpose="agentic_rag_synthesis")
            response = await call_gemini_with_backoff(
                model.generate_content,
                prompt,
                stream=False
            )
            
            response_text = response.text
            logger.debug(f"Agentic RAG synthesis raw response: {response_text[:200]}...")
            result = json.loads(response_text)

            if "overall_score" in result and "explanation" in result:
                try:
                    result["overall_score"] = float(result["overall_score"])
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert overall_score '{result['overall_score']}' to float. Defaulting to 0.0")
                    result["overall_score"] = 0.0
                return result
            else:
                logger.warning(f"Agentic RAG synthesis response missing expected keys: {result}")
                return {
                    "error": "Invalid format from score synthesis", 
                    "overall_score": 0.0, 
                    "explanation": "Failed to generate score.", 
                    "details": result
                }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Agentic RAG synthesis: {e}")
            try: 
                error_details = response.prompt_feedback
            except Exception: 
                error_details = "No further details available."
            return {
                "error": "Failed to parse score synthesis JSON", 
                "overall_score": 0.0, 
                "explanation": "Failed to generate score.", 
                "details": error_details
            }
        except Exception as e:
            logger.error(f"Error in Agentic RAG synthesis step: {e}", exc_info=True)
            return {
                "error": str(e), 
                "overall_score": 0.0, 
                "explanation": "Failed to generate score."
            }
    
    def _format_facets_with_evidence(
        self, 
        facets: List[JobRequirementFacet], 
        evidence: Dict[int, Dict],
        external_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Formats requirement facets with their evidence and external data for inclusion in the prompt.
        
        Args:
            facets: List of requirement facets
            evidence: Dictionary mapping facet index to retrieved evidence
            external_data: Optional dictionary containing external market data
            
        Returns:
            Formatted string representation of facets with evidence and external data
        """
        formatted_sections = []
        
        # Check if we have facet-specific external data
        facet_external_data = {}
        if external_data and "facet_external_data" in external_data:
            facet_external_data = external_data["facet_external_data"]
        
        for i, facet in enumerate(facets):
            # Start with the facet details
            requirement_status = "REQUIRED" if facet.is_required else "PREFERRED"
            facet_header = f"FACET #{i+1}: [{requirement_status}] {facet.facet_type.upper()}: {facet.detail}"
            if facet.context:
                facet_header += f" (Context: {facet.context})"
                
            # Format evidence if available
            evidence_text = "NO EVIDENCE FOUND"
            if i in evidence and evidence[i] is not None and evidence[i].get('documents') and evidence[i]['documents'][0]:
                chunks = evidence[i]['documents'][0]
                distances = evidence[i].get('distances', [[]])[0]
                
                # Format each chunk with its distance (if available)
                evidence_chunks = []
                for j, chunk in enumerate(chunks):
                    distance_info = ""
                    if distances and j < len(distances):
                        # Convert distance to similarity score (1 - distance) for easier interpretation
                        # Higher value = more similar
                        similarity = 1.0 - distances[j]
                        distance_info = f" [Relevance: {similarity:.2f}]"
                        
                    evidence_chunks.append(f"Evidence #{j+1}{distance_info}: {chunk.strip()}")
                
                if evidence_chunks:
                    evidence_text = "\n".join(evidence_chunks)
            
            # Format external data for this facet if available
            external_data_text = ""
            if i in facet_external_data and facet_external_data[i]:
                data_points = []
                
                # Add salary benchmark data
                if "salary_benchmark" in facet_external_data[i]:
                    sb = facet_external_data[i]["salary_benchmark"]
                    data_points.append(f"Salary Benchmark: ${sb['median']:,} median (range: ${sb['min']:,} - ${sb['max']:,})")
                
                # Add market growth data
                if "market_growth" in facet_external_data[i]:
                    mg = facet_external_data[i]["market_growth"]
                    data_points.append(f"Market Growth: {mg['demand_growth']*100:.1f}% growth rate, Competition: {mg['competition_level']}")
                
                # Add skill trend data
                if "skill_trend" in facet_external_data[i]:
                    st = facet_external_data[i]["skill_trend"]
                    trend_status = "emerging" if st.get('is_emerging') else "established"
                    data_points.append(f"Skill Trend: {st['demand_growth_rate']*100:.1f}% growth, {trend_status} skill (rank: #{st['popularity_rank']}), Salary Impact: {st.get('salary_impact', 'N/A')}")
                
                if data_points:
                    external_data_text = "\nEXTERNAL DATA:\n" + "\n".join(data_points)
            
            # Combine facet, evidence, and external data
            section = f"{facet_header}\n{evidence_text}{external_data_text}"
            formatted_sections.append(section)
        
        return "\n\n".join(formatted_sections)
    
    def _format_external_data_section(self, external_data: Optional[Dict[str, Any]]) -> str:
        """
        Formats the general external data section for the prompt.
        
        Args:
            external_data: The external data dictionary from AgenticRAGService
            
        Returns:
            Formatted string with market insights and trends
        """
        if not external_data or "external_data" not in external_data:
            return "EXTERNAL MARKET DATA: None available"
        
        data = external_data["external_data"]
        sections = ["EXTERNAL MARKET DATA:"]
        
        # Add salary benchmark summary
        if data.get("salary_benchmark"):
            sb = data["salary_benchmark"]["salary_data"]
            sections.append(f"SALARY BENCHMARK: ${sb['median']:,} median salary (range: ${sb['min']:,} - ${sb['max']:,})")
            
            # Add factors that influence salary
            if "market_factors" in data["salary_benchmark"]:
                factors = data["salary_benchmark"]["market_factors"]
                factor_text = "Salary factors: "
                if factors.get("location_factor") and factors["location_factor"] != 1.0:
                    direction = "higher" if factors["location_factor"] > 1.0 else "lower"
                    factor_text += f"Location ({abs(factors['location_factor']-1.0)*100:.0f}% {direction} than average), "
                if factors.get("experience_factor") and factors["experience_factor"] != 1.0:
                    factor_text += f"Experience level ({factors['experience_factor']:.1f}x multiplier), "
                if factors.get("skills_factor") and factors["skills_factor"] != 1.0:
                    factor_text += f"Skill premium ({(factors['skills_factor']-1.0)*100:.0f}%), "
                sections.append(factor_text.rstrip(", "))
        
        # Add market insights summary
        if data.get("market_insights"):
            mi = data["market_insights"]["market_data"]
            sections.append(f"MARKET INSIGHTS: {mi['demand_growth_rate']*100:.1f}% demand growth, {mi['job_postings_last_period']:,} recent job postings")
            sections.append(f"COMPETITION LEVEL: {mi['competition_level'].title()}, avg. {mi['average_time_to_fill']} days to fill positions")
            
            # Add top locations
            if "regional_insights" in data["market_insights"]:
                ri = data["market_insights"]["regional_insights"]
                if ri.get("top_locations"):
                    sections.append(f"TOP LOCATIONS: {', '.join(ri['top_locations'][:3])}")
                if ri.get("remote_percentage"):
                    sections.append(f"REMOTE WORK: {ri['remote_percentage']}% of positions offer remote options")
        
        # Add skill trends summary
        if data.get("skill_trends") and "skill_trends" in data["skill_trends"]:
            # Find top growing skills
            skill_trends = sorted(
                data["skill_trends"]["skill_trends"], 
                key=lambda x: x.get("demand_growth_rate", 0), 
                reverse=True
            )
            
            if skill_trends:
                top_skills = skill_trends[:3]  # Take top 3
                trend_details = []
                for skill in top_skills:
                    growth = skill.get("demand_growth_rate", 0) * 100
                    emerging = " (emerging)" if skill.get("is_emerging") else ""
                    trend_details.append(f"{skill['skill']}: +{growth:.1f}%{emerging}")
                
                sections.append(f"TOP GROWING SKILLS: {', '.join(trend_details)}")
            
            # Add industry insights
            if "industry_insights" in data["skill_trends"]:
                ii = data["skill_trends"]["industry_insights"]
                if ii.get("fastest_growing_industries"):
                    sections.append(f"GROWING INDUSTRIES: {', '.join(ii['fastest_growing_industries'][:3])}")
        
        return "\n".join(sections)