import google.generativeai as genai
import json
import logging
from typing import Dict, Any, Optional, List, Union
import asyncio # Added for sleep

from recruitx_app.core.config import settings
from recruitx_app.utils.retry_utils import call_gemini_with_backoff # Import the retry helper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeExecutionAgent:
    """
    An agent that leverages Gemini 2.5 Pro's code execution capabilities 
    to dynamically generate and execute code for advanced analysis tasks.
    """
    
    def __init__(self):
        self.model_name = settings.GEMINI_PRO_MODEL
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
    
    def _get_gemini_model(self):
        """Get the Gemini model, rotating API keys if necessary."""
        try:
            model = genai.GenerativeModel(
                self.model_name,
                safety_settings=self.safety_settings,
                generation_config={
                    "temperature": 0.2,  # Slightly higher for creative code solutions
                    "top_p": 0.95,
                    "top_k": 40,
                }
            )
            return model
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            genai.configure(api_key=settings.get_next_api_key())
            logger.info("Rotated to next API key")
            try:
                return genai.GenerativeModel(
                    self.model_name,
                    safety_settings=self.safety_settings,
                    generation_config={
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "top_k": 40,
                    }
                )
            except Exception as e2:
                logger.error(f"Second error initializing Gemini model: {e2}")
                raise e2
    
    async def generate_and_execute_skill_matcher(
        self, 
        job_skills: List[str], 
        candidate_skills: List[str]
    ) -> Dict[str, Any]:
        """
        Generate and execute custom Python code to match job skills with candidate skills.
        
        Args:
            job_skills: List of skills required for the job
            candidate_skills: List of skills possessed by the candidate
            
        Returns:
            Dictionary with match results and explanations
        """
        try:
            model = self._get_gemini_model()
            
            # Create a prompt that instructs Gemini to write and execute code
            prompt = f"""
            Write Python code to match these job skills:
            {job_skills}
            
            With these candidate skills:
            {candidate_skills}
            
            Your code must:
            1. Use NLP techniques to measure semantic similarity between skills
            2. Account for variations in skill descriptions and acronyms
            3. Calculate a match score (0-100) for each job skill
            4. Provide an overall match percentage
            5. Generate an explanation for each skill match
            6. Identify skills gaps and recommend learning resources
            
            Use libraries like spaCy, NLTK, scikit-learn, or sentence-transformers if needed.
            Make sure to import all necessary dependencies.
            
            Write code that will run successfully, execute it, and return the results as a JSON object.
            The result should include the overall match percentage, individual skill matches, and explanations.
            """
            
            # Use the retry helper for the API call
            response = await call_gemini_with_backoff(
                model.generate_content,
                prompt,
                allow_code_execution=True,  # Enable code execution
                code_execution_config={"isolate": True}  # Run in an isolated environment
            )
            
            # Parse and return the results
            results = {}
            
            # Extract the code execution results
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'code_execution_result'):
                                results["code_execution"] = part.code_execution_result
                            if hasattr(part, 'text') and part.text:
                                try:
                                    json_result = json.loads(part.text)
                                    results.update(json_result)
                                except json.JSONDecodeError:
                                    if "text" not in results:
                                        results["text"] = part.text
                                except Exception as json_e:
                                    logger.error(f"Error parsing skill matcher JSON: {json_e}")
                                    if "error_parsing" not in results:
                                        results["error_parsing"] = part.text
            
            # If we couldn't get structured data, include the raw response
            if not results and hasattr(response, 'text') and response.text:
                try:
                    return json.loads(response.text)
                except:
                    return {"matching_results": response.text}
            
            return results
            
        except Exception as e:
            logger.error(f"Error in code execution agent: {e}")
            return {"error": str(e)}
    
    async def generate_skill_visualization(
        self, 
        job_skills: List[str], 
        candidate_skills: List[str], 
        match_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate and execute code to create visualizations of skill matches.
        
        Args:
            job_skills: List of skills required for the job
            candidate_skills: List of skills possessed by the candidate
            match_results: Results from the skill matcher
            
        Returns:
            Dictionary with visualization data (Base64 encoded images)
        """
        try:
            model = self._get_gemini_model()
            
            # Create a prompt that instructs Gemini to write visualization code
            prompt = f"""
            Write Python code to visualize the match between job skills and candidate skills.
            
            Job skills: {job_skills}
            Candidate skills: {candidate_skills}
            Match results: {json.dumps(match_results)}
            
            Your code must:
            1. Create at least two informative visualizations:
               - A radar chart or bar chart showing skill match percentages
               - A heatmap or network graph showing relationships between skills
            2. Use matplotlib, seaborn, plotly, or any suitable visualization libraries
            3. Convert the visualizations to base64 encoded strings to include in the response
            4. Return the images in a JSON format
            
            Ensure your code is efficient, well-commented, and handles edge cases gracefully.
            Execute the code and return the base64 encoded images in your response.
            """
            
            # Use the retry helper for the API call
            response = await call_gemini_with_backoff(
                model.generate_content,
                prompt,
                allow_code_execution=True,
                code_execution_config={"isolate": True}
            )
            
            # Extract the visualization results
            results = {"visualizations": {}}
            
            # Extract the code execution results
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'code_execution_result'):
                                results["code_execution"] = part.code_execution_result
                            if hasattr(part, 'text') and part.text:
                                try:
                                    json_data = json.loads(part.text)
                                    if "visualizations" in results:
                                        results["visualizations"].update(json_data)
                                    else:
                                        results["visualizations"] = json_data
                                except json.JSONDecodeError:
                                    results["explanation"] = part.text
                                except Exception as json_e:
                                    logger.error(f"Error parsing visualization JSON: {json_e}")
                                    if "error_parsing" not in results:
                                        results["error_parsing"] = part.text
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return {"error": str(e)} 