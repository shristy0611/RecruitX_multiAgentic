import google.generativeai as genai
import json
import logging
from typing import Dict, Any, Optional, List, Union, Callable
import asyncio # Added for sleep

from recruitx_app.core.config import settings
from recruitx_app.utils.retry_utils import call_gemini_with_backoff # Import the retry helper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolUseAgent:
    """
    An agent that leverages Gemini 2.5 Pro's native tool use capabilities
    to interact with databases, external APIs, and custom functions.
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
        
        # Register available tools
        self.available_tools = {
            "fetch_job_requirements": self._fetch_job_requirements,
            "get_candidate_skills": self._get_candidate_skills,
            "check_skill_database": self._check_skill_database,
            "search_learning_resources": self._search_learning_resources,
            "get_market_salary_data": self._get_market_salary_data
        }
        
        # Define tool schemas
        self.tool_schemas = [
            {
                "name": "fetch_job_requirements",
                "description": "Fetch detailed requirements for a specific job from the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "integer",
                            "description": "The ID of the job to fetch"
                        }
                    },
                    "required": ["job_id"]
                }
            },
            {
                "name": "get_candidate_skills",
                "description": "Get the skills of a specific candidate from the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {
                            "type": "integer",
                            "description": "The ID of the candidate to fetch"
                        }
                    },
                    "required": ["candidate_id"]
                }
            },
            {
                "name": "check_skill_database",
                "description": "Check if a skill exists in our database and get related skills",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "The name of the skill to check"
                        }
                    },
                    "required": ["skill_name"]
                }
            },
            {
                "name": "search_learning_resources",
                "description": "Search for learning resources for a specific skill",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "The skill to find learning resources for"
                        },
                        "resource_type": {
                            "type": "string",
                            "description": "Type of resources (courses, books, tutorials, etc.)",
                            "enum": ["courses", "books", "tutorials", "videos", "all"]
                        }
                    },
                    "required": ["skill_name"]
                }
            },
            {
                "name": "get_market_salary_data",
                "description": "Get market salary data for a job title and location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_title": {
                            "type": "string",
                            "description": "The job title to look up"
                        },
                        "location": {
                            "type": "string",
                            "description": "Location (city or country)"
                        },
                        "experience_level": {
                            "type": "string",
                            "description": "Experience level (entry, mid, senior)",
                            "enum": ["entry", "mid", "senior"]
                        }
                    },
                    "required": ["job_title"]
                }
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
    
    # Tool implementation methods - These would connect to real databases and APIs in production
    
    def _fetch_job_requirements(self, args):
        """Mock implementation of fetch_job_requirements tool."""
        job_id = args.get("job_id")
        # In production, this would query the database
        logger.info(f"Fetching job requirements for job_id: {job_id}")
        
        # Mock data for demonstration
        requirements = {
            "required_skills": ["Python", "FastAPI", "SQL", "Machine Learning"],
            "preferred_skills": ["Docker", "AWS", "PyTorch"],
            "minimum_experience": "3 years",
            "education": "Bachelor's in Computer Science or related field",
            "job_type": "Full-time"
        }
        
        return requirements
    
    def _get_candidate_skills(self, args):
        """Mock implementation of get_candidate_skills tool."""
        candidate_id = args.get("candidate_id")
        # In production, this would query the database
        logger.info(f"Fetching skills for candidate_id: {candidate_id}")
        
        # Mock data for demonstration
        skills = {
            "technical_skills": ["Python", "Django", "PostgreSQL", "TensorFlow"],
            "soft_skills": ["Communication", "Team Leadership", "Problem Solving"],
            "certifications": ["AWS Certified Developer", "MongoDB Certified Developer"],
            "languages": ["English (Fluent)", "Spanish (Intermediate)"]
        }
        
        return skills
    
    def _check_skill_database(self, args):
        """Mock implementation of check_skill_database tool."""
        skill_name = args.get("skill_name")
        # In production, this would query a skill database or ontology
        logger.info(f"Checking skill database for: {skill_name}")
        
        # Mock data for demonstration
        skill_data = {
            "exists": True,
            "canonical_name": skill_name.title(),
            "related_skills": ["Data Science", "Deep Learning", "Neural Networks", "Computer Vision"],
            "domain": "Artificial Intelligence",
            "popularity_score": 85
        }
        
        return skill_data
    
    def _search_learning_resources(self, args):
        """Mock implementation of search_learning_resources tool."""
        skill_name = args.get("skill_name")
        resource_type = args.get("resource_type", "all")
        # In production, this would call an external API or search engine
        logger.info(f"Searching learning resources for {skill_name}, type: {resource_type}")
        
        # Mock data for demonstration
        resources = {
            "courses": [
                {"title": f"{skill_name} Masterclass", "platform": "Udemy", "url": "https://udemy.com/example"},
                {"title": f"Advanced {skill_name}", "platform": "Coursera", "url": "https://coursera.org/example"}
            ],
            "books": [
                {"title": f"{skill_name} in Practice", "author": "John Smith", "year": 2023},
                {"title": f"Mastering {skill_name}", "author": "Jane Doe", "year": 2022}
            ],
            "tutorials": [
                {"title": f"{skill_name} for Beginners", "source": "Medium", "url": "https://medium.com/example"},
                {"title": f"Building Projects with {skill_name}", "source": "Dev.to", "url": "https://dev.to/example"}
            ]
        }
        
        if resource_type != "all":
            return {resource_type: resources.get(resource_type, [])}
        return resources
    
    def _get_market_salary_data(self, args):
        """Mock implementation of get_market_salary_data tool."""
        job_title = args.get("job_title")
        location = args.get("location", "United States")
        experience_level = args.get("experience_level", "mid")
        # In production, this would call a salary database API
        logger.info(f"Getting market salary data for {job_title} in {location}, level: {experience_level}")
        
        # Mock data for demonstration
        salary_data = {
            "job_title": job_title,
            "location": location,
            "experience_level": experience_level,
            "salary_range": {
                "min": 85000,
                "median": 110000,
                "max": 135000
            },
            "currency": "USD",
            "data_source": "Industry Salary Survey 2025",
            "last_updated": "2025-03-15"
        }
        
        return salary_data
    
    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with the given arguments."""
        if tool_name not in self.available_tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            result = self.available_tools[tool_name](args)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}
    
    async def analyze_job_candidate_match(
        self, 
        job_id: int, 
        candidate_id: int
    ) -> Dict[str, Any]:
        """
        Analyze the match between a job and a candidate using native tool use.
        
        Args:
            job_id: The ID of the job
            candidate_id: The ID of the candidate
            
        Returns:
            Dictionary with match analysis and recommendations
        """
        try:
            model = self._get_gemini_model()
            
            # Create a detailed prompt
            prompt = f"""
            Analyze the match between Job ID {job_id} and Candidate ID {candidate_id}.
            
            You have access to several tools that can fetch information from our database.
            First, get the job requirements and candidate skills.
            Then, analyze how well the candidate's skills match the job requirements.
            For any skill gaps, search for appropriate learning resources.
            Finally, check current market salary data for this position.
            
            Provide a comprehensive analysis of the match, including:
            1. Overall match percentage
            2. Strengths and weaknesses
            3. Skill gaps and learning recommendations
            4. Salary expectations versus market rates
            
            Use a step-by-step approach and be thorough in your analysis.
            """
            
            # Initial call to get tool requests from Gemini
            # Use the retry helper
            response = await call_gemini_with_backoff(
                model.generate_content,
                prompt,
                tools=[{"function_declarations": self.tool_schemas}],
                tool_config={
                    "function_calling_config": {
                        "mode": "auto"
                    }
                }
            )
            
            # Process the response and extract tool calls
            analysis_results = {"steps": []}
            tool_calls = []
            
            # Track which tools were called by the model
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call'):
                                function_call = part.function_call
                                tool_name = function_call.name
                                args = json.loads(function_call.args)
                                
                                # Execute the tool
                                logger.info(f"Executing tool: {tool_name} with args: {args}")
                                tool_result = await self.execute_tool(tool_name, args)
                                
                                # Record the tool call and result
                                tool_calls.append({
                                    "tool": tool_name,
                                    "args": args,
                                    "result": tool_result
                                })
                                
                                analysis_results["steps"].append({
                                    "action": f"Called {tool_name}",
                                    "result": tool_result
                                })
            
            # Now we have the tool call results, get Gemini to generate the final analysis
            if tool_calls:
                # Create a new prompt with the tool results
                analysis_prompt = f"""
                Based on the following data retrieved from our tools:
                
                {json.dumps(tool_calls, indent=2)}
                
                Provide a comprehensive analysis of the match between Job ID {job_id} and Candidate ID {candidate_id}.
                
                Include:
                1. Overall match percentage with justification
                2. Strengths (skills that match well)
                3. Weaknesses (skills that are missing or underdeveloped)
                4. Learning recommendations with specific resources
                5. Salary analysis compared to market rates
                
                Format your response as a structured JSON object with these sections.
                """
                
                # Get the final analysis - Use the retry helper
                final_analysis_response = await call_gemini_with_backoff(
                    model.generate_content,
                    analysis_prompt
                )
                
                # Parse the response
                final_analysis_text = final_analysis_response.text if hasattr(final_analysis_response, 'text') else None
                if final_analysis_text:
                    try:
                        final_results = json.loads(final_analysis_text)
                        analysis_results["analysis"] = final_results
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse final analysis as JSON, using raw text.")
                        analysis_results["analysis"] = final_analysis_text
                else:
                    logger.warning("No text content received for final analysis after tool calls.")
                    analysis_results["analysis"] = "No analysis text generated after tool execution."
                
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in tool use agent: {e}")
            return {"error": str(e)} 