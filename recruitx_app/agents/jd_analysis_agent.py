import google.generativeai as genai
import json
from typing import Dict, Any, Optional, List, Union
import logging
import asyncio # Added for sleep

from recruitx_app.core.config import settings
from recruitx_app.schemas.job import JobAnalysis
from recruitx_app.utils.retry_utils import call_gemini_with_backoff # Import the retry helper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the structured output schema for function calling
ANALYSIS_SCHEMA = {
    "name": "analyze_job_description",
    "description": "Analyze a job description to extract structured information about requirements and details",
    "parameters": {
        "type": "object",
        "properties": {
            "required_skills": {
                "type": "array",
                "description": "List of technical and soft skills explicitly required for the position",
                "items": {"type": "string"}
            },
            "preferred_skills": {
                "type": "array",
                "description": "List of skills mentioned as 'nice to have' or preferred",
                "items": {"type": "string"}
            },
            "minimum_experience": {
                "type": "string",
                "description": "The minimum years of experience required for the role"
            },
            "education": {
                "type": "string",
                "description": "The required education level or degree"
            },
            "responsibilities": {
                "type": "array",
                "description": "Major responsibilities of the role",
                "items": {"type": "string"}
            },
            "job_type": {
                "type": "string",
                "description": "Whether it's full-time, part-time, contract, etc."
            },
            "salary_range": {
                "type": "string",
                "description": "If mentioned, extract the salary range"
            },
            "company_culture": {
                "type": "string",
                "description": "Insights about the company culture or work environment"
            },
            "benefits": {
                "type": "array",
                "description": "Any benefits offered by the company",
                "items": {"type": "string"}
            },
            "industry": {
                "type": "string",
                "description": "The industry or sector of the job"
            },
            "seniority_level": {
                "type": "string",
                "description": "The level of the position (junior, mid, senior)"
            },
            "market_insights": {
                "type": "object",
                "description": "Grounded market insights about this job and industry",
                "properties": {
                    "skill_demand": {
                        "type": "object",
                        "description": "Information about the demand for key skills",
                        "properties": {
                            "high_demand_skills": {
                                "type": "array",
                                "description": "Skills from the job description that are in high demand",
                                "items": {"type": "string"}
                            },
                            "trending_skills": {
                                "type": "array",
                                "description": "Skills that are trending in this industry",
                                "items": {"type": "string"}
                            }
                        }
                    },
                    "salary_insights": {
                        "type": "string",
                        "description": "Factual information about salary ranges for this role in the market"
                    },
                    "industry_outlook": {
                        "type": "string",
                        "description": "Brief outlook for the industry this job is in"
                    }
                }
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation of the analysis process and key insights derived from the job description"
            }
        },
        "required": ["required_skills", "responsibilities", "market_insights", "reasoning"]
    }
}

# JD Analysis prompt that instructs Gemini how to extract information - enhanced for Gemini 2.5 with search grounding
JD_ANALYSIS_PROMPT = """
You are an advanced Job Description Analysis Agent powered by Gemini 2.5 Pro, specifically designed to extract comprehensive structured information from job descriptions.

Job Description:
```
{job_description}
```

Your task is to carefully analyze this job description and extract key information with detailed context. Use your enhanced reasoning capabilities to:

1. Understand the explicit and implicit requirements
2. Recognize industry-specific terminology and skills
3. Infer seniority level from responsibilities and requirements 
4. Distinguish between mandatory and preferred qualifications
5. Identify cultural indicators and work environment details

Additionally, use search to ground your analysis in factual information about:
- Current market demand for the identified skills
- Typical salary ranges for this type of position
- Industry trends and outlook
- Related emerging skills that might be relevant

Show your thinking process as you analyze the document, explaining your reasoning step by step before making conclusions.

Extract this information according to the function schema you've been provided.

Remember to be thorough and use your large context window to fully understand the entire job description. If information is genuinely missing, indicate that instead of making assumptions.
"""

class JDAnalysisAgent:
    def __init__(self):
        # Use the Gemini model from settings
        self.model_name = settings.GEMINI_PRO_MODEL  
        # Initialize the counter for API key rotation
        self._api_key_index = 0
        # Configure Gemini API with the first API key for now
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
        # Better error handling: if we get an error, try to rotate to the next API key
        try:
            model = genai.GenerativeModel(
                self.model_name,
                safety_settings=self.safety_settings,
                generation_config={
                    "temperature": 0,  # Lower temperature for more deterministic results
                    "top_p": 0.95,
                    "top_k": 40,
                    # "response_mime_type": "application/json", # Removed as per Gemini API guidelines for function calling
                    # "enable_thinking": True,  # Temporarily remove thinking capability
                }
            )
            return model
        except Exception as e:
            # Log the error
            logger.error(f"Error initializing Gemini model: {e}")
            # Try with a new API key
            genai.configure(api_key=settings.get_next_api_key())
            logger.info("Rotated to next API key")
            try:
                return genai.GenerativeModel(
                    self.model_name,
                    safety_settings=self.safety_settings,
                    generation_config={
                        "temperature": 0,
                        "top_p": 0.95,
                        "top_k": 40,
                        # "response_mime_type": "application/json", # Removed as per Gemini API guidelines for function calling
                        # "enable_thinking": True,
                    }
                )
            except Exception as e2:
                logger.error(f"Second error initializing Gemini model: {e2}")
                raise e2
    
    async def analyze_job_description(self, job_id: int, job_description: str) -> Optional[JobAnalysis]:
        """
        Analyze a job description with Gemini 2.5 Pro and extract structured information 
        using function calling, search grounding, and explicit thinking.
        
        Args:
            job_id: The ID of the job
            job_description: The full text of the job description
            
        Returns:
            JobAnalysis object containing the extracted information
        """
        try:
            # Fill in the job description in the prompt
            prompt = JD_ANALYSIS_PROMPT.format(job_description=job_description)
            
            # Get the model (with key rotation if needed)
            model = self._get_gemini_model()
            
            # Set up function calling with our schema
            tools = [{"function_declarations": [ANALYSIS_SCHEMA]}]
            
            # Use the retry helper for the API call
            response = await call_gemini_with_backoff(
                model.generate_content,
                prompt,
                tools=tools, 
                tool_config={"function_calling_config": {"mode": "any"}},
                # search_args={"enable": True},  # Temporarily remove search_args
                stream=False
            )
            
            result_json = None
            thinking_output = None
            
            # Extract the thinking process if available
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        # Get thinking output if available
                        if hasattr(candidate.content, 'thinking'):
                            thinking_output = candidate.content.thinking
                            logger.info("Captured thinking process from Gemini")
                            
                        # Extract function call result
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call'):
                                function_call = part.function_call
                                if function_call.name == "analyze_job_description":
                                    # Directly use the arguments from the function call object
                                    result_json = function_call.args # No need for json.loads here
                                    break
                    if result_json:
                        break
            
            # Fallback to standard text processing if function calling didn't work
            if result_json is None and hasattr(response, 'text') and response.text:
                try:
                    result_json = json.loads(response.text)
                except json.JSONDecodeError:
                    logger.warning("Fallback: Response text is not valid JSON.")
                except Exception as json_e:
                    logger.error(f"Error parsing fallback text as JSON: {json_e}")
                
            if result_json:
                # Add the job_id to the result
                result_json["job_id"] = job_id
                
                # Store the thinking process if available
                if thinking_output:
                    result_json["analysis_process"] = thinking_output
                
                # Validate and convert the dict result to a JobAnalysis model
                try:
                    # Use pydantic's parse_obj for direct dict -> model conversion
                    return JobAnalysis.model_validate(result_json)
                except Exception as pydantic_e:
                    logger.error(f"Pydantic validation error for JobAnalysis: {pydantic_e}")
                    logger.error(f"Data received: {result_json}")
                    return None
            else:
                logger.error("Failed to extract structured data from response after fallback.")
                return None
                
        except Exception as e:
            # Log the error
            logger.error(f"Error analyzing job description: {e}")
            return None
            
    async def get_industry_insights(self, industry: str, skills: List[str]) -> Dict[str, Any]:
        """
        Get grounded insights about an industry and skills using search capabilities.
        
        Args:
            industry: The industry to research
            skills: List of skills to check market demand for
            
        Returns:
            Dictionary with industry insights
        """
        try:
            model = self._get_gemini_model()
            
            # Create a detailed prompt for industry research
            prompt = f"""
            Provide factual, search-grounded insights about the {industry} industry and these skills:
            {', '.join(skills)}
            
            Include:
            1. Current industry growth trends and outlook
            2. Market demand for each skill (high, medium, low)
            3. Average salary ranges for professionals with these skills
            4. Emerging technologies or skills in this field
            5. Major companies hiring for these skills
            
            Base all your analysis on factual information you find through search.
            Provide a detailed, objective assessment with specific facts and figures where available.
            Format your response as detailed JSON.
            """
            
            # Use the retry helper for the API call
            response = await call_gemini_with_backoff(
                model.generate_content,
                prompt,
                # search_args={"enable": True}, # Temporarily remove search_args
                stream=False
            )
            
            # Extract and parse the response
            if hasattr(response, 'text') and response.text:
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                     logger.warning("Industry insights response is not valid JSON.")
                     return {"insights": response.text} # Return raw text if not JSON
                except Exception as json_e:
                     logger.error(f"Error parsing industry insights JSON: {json_e}")
                     return {"error": f"Failed to parse response: {json_e}"}
            else:
                logger.warning("No text content received for industry insights.")
                # Check if there's content in parts as a fallback
                if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'content'):
                     text_content = " ".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                     if text_content:
                         try:
                             return json.loads(text_content)
                         except:
                             return {"insights": text_content}
                return {"error": "No content received"}
                    
        except Exception as e:
            logger.error(f"Error getting industry insights: {e}")
            return {"error": str(e)} 