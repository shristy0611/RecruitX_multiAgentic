import google.generativeai as genai
import json
import logging
from typing import Dict, Any, Optional, List
import asyncio

from recruitx_app.core.config import settings
from recruitx_app.utils.retry_utils import call_gemini_with_backoff

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt 1: Extract Key Skills
SKILL_EXTRACTION_PROMPT = """
You are an AI assistant analyzing recruitment documents.
Review the job description and candidate resume below.

JOB DESCRIPTION:
--------------------
{job_description}
--------------------

CANDIDATE RESUME:
--------------------
{candidate_resume}
--------------------

Your task is to identify and extract the key skills mentioned in both texts.
Focus on technical skills, software, methodologies, and relevant soft skills.

Return your response ONLY as a valid JSON object in the following format:
{{
    "job_skills": ["skill1", "skill2", ...],
    "candidate_skills": ["skillA", "skillB", ...]
}}

Do not include any explanation or introductory text outside the JSON structure.
"""

# Prompt 2: Synthesize Score based on Texts and Extracted Skills
SCORE_SYNTHESIS_PROMPT = """
You are an AI-powered recruitment assistant evaluating a candidate's fit for a job.
You have the original job description and candidate resume, along with pre-extracted lists of key skills for both.

JOB DESCRIPTION:
--------------------
{job_description}
--------------------

CANDIDATE RESUME:
--------------------
{candidate_resume}
--------------------

EXTRACTED JOB SKILLS:
{job_skills}

EXTRACTED CANDIDATE SKILLS:
{candidate_skills}

Based on ALL the provided information (especially comparing the skill lists and considering experience/education context from the full texts), provide a comprehensive evaluation.

Calculate an overall match score (0-100) reflecting the candidate's suitability.
Provide a concise explanation justifying the score, highlighting key strengths and weaknesses based on skill overlap, experience relevance, and other factors from the texts.

Return your response ONLY as a valid JSON object in the following format:
{{
    "overall_score": <score 0-100>,
    "explanation": "<concise explanation for the score>"
}}

Do not include any introductory text outside the JSON structure.
"""

class OrchestrationAgent: # Renamed from SimpleScoringAgent
    """
    An agent that orchestrates scoring by first extracting structured data (skills)
    and then synthesizing a score based on the original text and extracted data.
    Uses API key rotation.
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

    async def synthesize_score(self, job_description: str, candidate_resume: str, job_skills: List[str], candidate_skills: List[str]) -> Dict[str, Any]:
        """
        Second step: Generates the final score and explanation based on texts and skills.
        
        Returns:
            Dictionary containing overall_score and explanation, or an error structure.
        """
        # logger.info("--- SIMULATING Score Synthesis --- ")
        # await asyncio.sleep(0.8) # Simulate processing time

        # # --- START SIMULATION ---
        # # Hardcoded dummy data - replace or remove when using real API
        # dummy_score = {
        #     "overall_score": 45.0, # Simulate the score we saw in logs
        #     "explanation": "SIMULATED: Candidate shows strong data analysis skills (SQL, Python, PowerBI) but lacks the core UX/UI design skills (Figma, Sketch, Prototyping) required by the job description. Experience is in healthcare analytics, not design. Low match based on simulated skill comparison."
        # }
        # logger.info(f"Simulation returning dummy score: {dummy_score}")
        # return dummy_score
        # # --- END SIMULATION ---

        # --- Original API Call Logic --- 
        try:
            prompt = SCORE_SYNTHESIS_PROMPT.format(
                job_description=job_description,
                candidate_resume=candidate_resume,
                job_skills=json.dumps(job_skills), # Format lists as JSON strings for the prompt
                candidate_skills=json.dumps(candidate_skills)
            )
            model = self._get_gemini_model(purpose="score_synthesis")
            
            response = await call_gemini_with_backoff(
                model.generate_content,
                prompt,
                stream=False
            )
            
            response_text = response.text
            logger.debug(f"Score synthesis raw response: {response_text[:200]}...")
            result = json.loads(response_text)
            # Basic validation
            if "overall_score" in result and "explanation" in result:
                # Convert score to float if possible
                try:
                    result["overall_score"] = float(result["overall_score"])
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert overall_score '{result['overall_score']}' to float. Defaulting to 0.0")
                    result["overall_score"] = 0.0
                return result
            else:
                logger.warning(f"Score synthesis response missing expected keys: {result}")
                return {"error": "Invalid format from score synthesis", "overall_score": 0.0, "explanation": "Failed to generate score.", "details": result}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from score synthesis: {e}")
            try: 
                error_details = response.prompt_feedback
            except Exception: 
                 error_details = "No further details available."
            return {"error": "Failed to parse score synthesis JSON", "overall_score": 0.0, "explanation": "Failed to generate score.", "details": error_details}
        except Exception as e:
            logger.error(f"Error in score synthesis agent step: {e}", exc_info=True)
            return {"error": str(e), "overall_score": 0.0, "explanation": "Failed to generate score."}
        # --- End Original Logic ---