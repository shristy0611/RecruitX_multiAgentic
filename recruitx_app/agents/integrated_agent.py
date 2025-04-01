import google.generativeai as genai
import json
import logging
from typing import Dict, Any, Optional, List, Union, Callable
import asyncio # Added for sleep
import time # Added for sleep

from recruitx_app.core.config import settings
from recruitx_app.agents.jd_analysis_agent import JDAnalysisAgent
from recruitx_app.agents.code_execution_agent import CodeExecutionAgent
from recruitx_app.agents.tool_use_agent import ToolUseAgent
from recruitx_app.agents.multimodal_agent import MultimodalAgent
from recruitx_app.utils.retry_utils import call_gemini_with_backoff # Import the retry helper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define longer delay time
INTER_STEP_DELAY = 15 # seconds

class IntegratedAgent:
    """
    An integrated agent that combines all Gemini 2.5 Pro capabilities:
    - Function calling
    - Search grounding
    - Code execution
    - Native tool use
    - Thinking
    - Multimodal processing
    
    This agent orchestrates the specialized agents to provide a unified experience.
    """
    
    def __init__(self):
        self.model_name = settings.GEMINI_PRO_MODEL
        genai.configure(api_key=settings.get_next_api_key())
        
        # Initialize specialized agents
        self.jd_analysis_agent = JDAnalysisAgent()
        self.code_execution_agent = CodeExecutionAgent()
        self.tool_use_agent = ToolUseAgent()
        self.multimodal_agent = MultimodalAgent()
        
        # Configure the main Gemini model
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
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    # "enable_thinking": True, # Temporarily remove thinking capability
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
                        # "enable_thinking": True,
                    }
                )
            except Exception as e2:
                logger.error(f"Second error initializing Gemini model: {e2}")
                raise e2
    
    async def comprehensive_job_candidate_analysis(
        self, 
        job_id: int, 
        candidate_id: int,
        include_visualizations: bool = True,
        use_search_grounding: bool = True
    ) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of a job and candidate match using all capabilities.
        
        Args:
            job_id: The ID of the job
            candidate_id: The ID of the candidate
            include_visualizations: Whether to include visualizations in the results
            use_search_grounding: Whether to use search grounding for market insights
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        results = {
            "job_id": job_id,
            "candidate_id": candidate_id,
            "analysis_components": {},
            "integrated_analysis": {},
            "visualizations": {},
            "thinking_process": None
        }
        
        try:
            # Step 1: Get job analysis
            logger.info(f"Getting job analysis for job_id: {job_id}")
            job_analysis = await self.jd_analysis_agent.get_industry_insights(
                industry="Technology", 
                skills=["Python", "FastAPI", "SQL", "Machine Learning"] 
            )
            results["analysis_components"]["job_analysis"] = job_analysis
            logger.info(f"Waiting {INTER_STEP_DELAY}s before next step...")
            await asyncio.sleep(INTER_STEP_DELAY) # Increase delay

            # Step 2: Use tool agent to analyze job-candidate match
            logger.info(f"Analyzing match between job {job_id} and candidate {candidate_id}")
            match_analysis = await self.tool_use_agent.analyze_job_candidate_match(
                job_id=job_id,
                candidate_id=candidate_id
            )
            results["analysis_components"]["match_analysis"] = match_analysis
            logger.info(f"Waiting {INTER_STEP_DELAY}s before next step...")
            await asyncio.sleep(INTER_STEP_DELAY) # Increase delay
            
            # Step 3: Use code execution to generate custom skill matching
            if "analysis" in match_analysis and isinstance(match_analysis.get("analysis"), dict):
                # Safely extract skills, providing defaults if keys are missing
                job_reqs = next((step["result"] for step in match_analysis.get("steps", []) if step.get("action") == "Called fetch_job_requirements"), {})
                candidate_data = next((step["result"] for step in match_analysis.get("steps", []) if step.get("action") == "Called get_candidate_skills"), {})
                
                job_skills = job_reqs.get("required_skills", [])
                candidate_skills = candidate_data.get("technical_skills", [])
                
                if job_skills and candidate_skills:
                    logger.info("Generating custom skill matching code")
                    skill_match_results = await self.code_execution_agent.generate_and_execute_skill_matcher(
                        job_skills=job_skills,
                        candidate_skills=candidate_skills
                    )
                    results["analysis_components"]["skill_match"] = skill_match_results
                    logger.info(f"Waiting {INTER_STEP_DELAY}s before visualization...")
                    await asyncio.sleep(INTER_STEP_DELAY) # Increase delay
                    
                    # Generate visualizations if requested
                    if include_visualizations:
                        logger.info("Generating skill match visualizations")
                        visualization_results = await self.code_execution_agent.generate_skill_visualization(
                            job_skills=job_skills,
                            candidate_skills=candidate_skills,
                            match_results=skill_match_results
                        )
                        results["visualizations"] = visualization_results.get("visualizations", {})
                        logger.info(f"Waiting {INTER_STEP_DELAY}s before final integration...")
                        await asyncio.sleep(INTER_STEP_DELAY) # Increase delay
                else:
                     logger.warning("Could not extract job or candidate skills for code execution step.")
            else:
                 logger.warning("Match analysis step did not produce expected dictionary output.")

            # Step 4: Use the main model to generate an integrated analysis
            logger.info("Generating final integrated analysis...")
            model = self._get_gemini_model()
            integration_prompt = f"""
            I need you to integrate the following analyses into a comprehensive report on the match between Job {job_id} and Candidate {candidate_id}.
            
            Job Analysis:
            {json.dumps(results["analysis_components"]["job_analysis"], indent=2)}
            
            Match Analysis:
            {json.dumps(results["analysis_components"]["match_analysis"], indent=2)}
            
            Skill Matching:
            {json.dumps(results["analysis_components"].get("skill_match", {}), indent=2)}
            
            Think step by step through the integration of these analyses. Consider areas of agreement and disagreement.
            Provide a unified assessment that covers:
            
            1. Overall Match Assessment (0-100% with explanation)
            2. Key Strengths (with confidence level for each)
            3. Areas for Development (with specific recommendations)
            4. Market Context (how this candidate compares to market expectations)
            5. Hiring Recommendation (with reasoning)
            
            Show your reasoning process clearly, explaining how you combined the different analyses to reach your conclusions.
            """
            
            # Use the retry helper for the API call
            response = await call_gemini_with_backoff(
                model.generate_content,
                integration_prompt
            )
            
            # Extract thinking process if available
            thinking_output = None
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'thinking'):
                            thinking_output = candidate.content.thinking
                            results["thinking_process"] = thinking_output
            
            # Parse the response for the integrated analysis
            if hasattr(response, 'text'):
                try:
                    # Try to parse as JSON
                    integrated_analysis = json.loads(response.text)
                    results["integrated_analysis"] = integrated_analysis
                except:
                    # If not JSON, use as text
                    results["integrated_analysis"] = {"text": response.text}
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            results["error"] = str(e)
            return results
    
    async def analyze_resume_with_visuals(
        self, 
        resume_text: str, 
        image_data_list: List[bytes] = None
    ) -> Dict[str, Any]:
        """
        Analyze a resume including any visual elements using multimodal capabilities.
        
        Args:
            resume_text: The text content of the resume
            image_data_list: List of image data from the resume (charts, etc.)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Use the multimodal agent - assuming it uses the retry helper internally if needed
            multimodal_analysis = await self.multimodal_agent.analyze_document_with_images(
                text_content=resume_text,
                image_data_list=image_data_list
            )
            logger.info(f"Waiting {INTER_STEP_DELAY}s before grounding...")
            await asyncio.sleep(INTER_STEP_DELAY) # Increase delay
            
            # Get detailed insights
            model = self._get_gemini_model()
            grounding_prompt = f"""
            Based on this resume analysis:
            {multimodal_analysis.get("analysis", "")}
            
            1. Extract a list of all skills mentioned
            2. Identify the candidate's experience level (entry, mid, senior)
            3. Determine their primary domain of expertise
            
            Then, use search to find:
            - Current market demand for these skills
            - Average salary range for this profile
            - Industry trends relevant to this candidate
            
            Provide a comprehensive market context for this candidate based on factual information.
            """
            
            # Use the retry helper for the API call
            insights_response = await call_gemini_with_backoff(
                model.generate_content,
                grounding_prompt
            )
            
            # Combine the results
            results = {
                "multimodal_analysis": multimodal_analysis,
                "market_insights": insights_response.text if hasattr(insights_response, 'text') else None
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in resume analysis with visuals: {e}")
            return {"error": str(e)} 