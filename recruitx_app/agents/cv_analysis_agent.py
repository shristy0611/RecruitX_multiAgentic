import google.generativeai as genai
import json
import logging
from typing import Dict, Any, List, Optional

from recruitx_app.core.config import settings

logger = logging.getLogger(__name__)

class CVAnalysisAgent:
    """
    Agent responsible for analyzing candidate CVs using Gemini.
    """
    
    def __init__(self):
        """Initialize the CV Analysis Agent."""
        self.model_name = settings.GEMINI_PRO_MODEL
        
    async def analyze_cv(self, cv_text: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a CV using Gemini to extract structured information.
        
        Args:
            cv_text: The raw text of the CV to analyze
            
        Returns:
            Dictionary containing structured CV information or None if analysis fails
        """
        try:
            # Get an API key using round-robin
            api_key = settings.get_next_api_key()
            genai.configure(api_key=api_key)
            
            # Create a model instance
            model = genai.GenerativeModel(self.model_name)
            
            # Define the prompt for CV analysis
            prompt = f"""
You are an expert CV analyzer for a recruitment system. Analyze the following CV/resume text and extract structured information. 
Focus on the candidate's skills, work experience, education, certifications, and any other relevant details.

CV/Resume to analyze:
{cv_text}

Provide a detailed analysis and structure your response in JSON format with the following sections:
- contact_info: Basic contact information if available (name, email, phone, location)
- summary: Brief professional summary/objective
- skills: Technical and soft skills (as a list)
- work_experience: List of work experiences with company, title, dates, and responsibilities
- education: Educational background with institution, degree, field, graduation date
- certifications: Any professional certifications
- projects: Notable projects mentioned
- languages: Languages the candidate knows
- overall_profile: A brief assessment of the candidate's profile

Format your response as valid JSON. Do not include any explanations, only the JSON output.
"""
            
            # Generate the analysis
            response = model.generate_content(prompt)
            
            # Extract and parse the response
            if hasattr(response, 'text'):
                result_text = response.text
                # Clean up the text to ensure it's valid JSON
                # Sometimes the model includes markdown code blocks
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0].strip()
                
                # Parse the JSON
                try:
                    analysis_result = json.loads(result_text)
                    return analysis_result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse CV analysis JSON: {e}")
                    logger.error(f"Received text: {result_text}")
                    return None
            else:
                logger.error("Unexpected response format from Gemini")
                return None
                
        except Exception as e:
            logger.error(f"Error in CV analysis: {e}")
            return None 