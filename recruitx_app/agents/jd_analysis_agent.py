import google.generativeai as genai
import json
from typing import Dict, Any, Optional, List, Union
import logging
import asyncio # Added for sleep

from recruitx_app.core.config import settings
from recruitx_app.schemas.job import JobAnalysis, JobRequirementFacet
from recruitx_app.utils.retry_utils import call_gemini_with_backoff # Import the retry helper
from recruitx_app.services.vector_db_service import vector_db_service # Import the vector db service

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
                "description": "List of technical and soft skills explicitly required (e.g., Python, React, Communication)",
                "items": {"type": "string"}
            },
            "preferred_skills": {
                "type": "array",
                "description": "List of skills mentioned as 'nice to have' or preferred (e.g., AWS, Docker)",
                "items": {"type": "string"}
            },
            "minimum_experience": {
                "type": "string",
                "description": "Minimum years of experience required (e.g., '3 years', '5+ years', 'Entry-level')"
            },
            "education": {
                "type": "string",
                "description": "Required education level or degree (e.g., 'Bachelor's Degree in CS', 'Master's preferred')"
            },
            "responsibilities": {
                "type": "array",
                "description": "Major responsibilities or duties of the role",
                "items": {"type": "string"}
            },
            "job_type": {
                "type": "string",
                "description": "Employment type (e.g., 'Full-time', 'Part-time', 'Contract')"
            },
            "salary_range": {
                "type": "string",
                "description": "Salary range if mentioned (e.g., '$100k - $120k', 'Competitive')"
            },
            "company_culture": {
                "type": "string",
                "description": "Insights about company culture or work environment mentioned"
            },
            "benefits": {
                "type": "array",
                "description": "Benefits offered (e.g., 'Health Insurance', '401k Match')",
                "items": {"type": "string"}
            },
            "industry": {
                "type": "string",
                "description": "Industry or sector of the job (e.g., 'Technology', 'Healthcare')"
            },
            "seniority_level": {
                "type": "string",
                "description": "Seniority level (e.g., 'Junior', 'Mid-level', 'Senior', 'Lead')"
            },
            "market_insights": {
                "type": "object",
                "description": "Grounded market insights (use search if necessary)",
                "properties": {
                    "skill_demand": {
                        "type": "object",
                        "description": "Demand for key skills mentioned in the JD",
                        "properties": {
                            "high_demand_skills": {
                                "type": "array",
                                "description": "Required skills that are currently in high demand",
                                "items": {"type": "string"}
                            },
                            "trending_skills": {
                                "type": "array",
                                "description": "Skills relevant to this role/industry that are trending",
                                "items": {"type": "string"}
                            }
                        }
                    },
                    "salary_insights": {
                        "type": "string",
                        "description": "Typical market salary range for this type of role (cite source/basis if possible)"
                    },
                    "industry_outlook": {
                        "type": "string",
                        "description": "Brief factual outlook for the job's industry"
                    }
                }
            },
            "reasoning": {
                "type": "string",
                "description": "Step-by-step explanation of the analysis process and key insights"
            }
        },
        "required": ["required_skills", "responsibilities", "market_insights", "reasoning"]
    }
}

# JD Analysis prompt - Enhanced
JD_ANALYSIS_PROMPT = """
You are an advanced Job Description Analysis Agent powered by Gemini, specifically designed to extract comprehensive structured information.

Job Description:
```
{job_description}
```

Your task is to meticulously analyze this job description and extract key information with detailed context. Use your enhanced reasoning capabilities to:

1. Understand explicit and implicit requirements.
2. Recognize industry-specific terminology and skills.
3. Infer seniority level from responsibilities and requirements.
4. Distinguish between mandatory and preferred qualifications.
5. Identify cultural indicators and work environment details.

Additionally, **use search grounding** to provide factual, up-to-date information for the `market_insights` section concerning:
- Current market demand for the identified skills.
- Typical salary ranges for this type of position.
- Industry trends and outlook.

Show your thinking process step-by-step in the `reasoning` field before concluding.

Extract this information strictly according to the `analyze_job_description` function schema provided. 
**If specific information for a non-required field (like `salary_range` or `benefits`) is not present in the text, explicitly use `null` or an empty list/string as appropriate for that field in your function call response. Do not invent information.**

Be thorough and utilize the full context of the job description.
"""

# Add a new RAG-enhanced prompt
JD_RAG_ANALYSIS_PROMPT = """
You are an advanced Job Description Analysis Agent powered by Gemini, specifically designed to extract comprehensive structured information.

Job Description:
```
{job_description}
```

Here is relevant context from similar job descriptions and industry data that may help your analysis:
```
{retrieved_context}
```

Your task is to meticulously analyze this job description and extract key information with detailed context. Use your enhanced reasoning capabilities to:

1. Understand explicit and implicit requirements.
2. Recognize industry-specific terminology and skills.
3. Infer seniority level from responsibilities and requirements.
4. Distinguish between mandatory and preferred qualifications.
5. Identify cultural indicators and work environment details.
6. Leverage the relevant context provided to enhance your analysis, especially for understanding industry standards, common skills, and typical requirements for this type of role.

Additionally, **use search grounding** to provide factual, up-to-date information for the `market_insights` section concerning:
- Current market demand for the identified skills.
- Typical salary ranges for this type of position.
- Industry trends and outlook.

Show your thinking process step-by-step in the `reasoning` field before concluding.

Extract this information strictly according to the `analyze_job_description` function schema provided. 
**If specific information for a non-required field (like `salary_range` or `benefits`) is not present in the text, explicitly use `null` or an empty list/string as appropriate for that field in your function call response. Do not invent information.**

Be thorough and utilize the full context of the job description.
"""

# --- NEW: Schema and Prompt for JD Decomposition --- 

# Define the structured output schema for function calling for DECOMPOSITION
# This schema aims to extract individual, verifiable requirements.
DECOMPOSE_JD_SCHEMA = {
    "name": "extract_job_requirements",
    "description": "Extract individual, verifiable requirements (facets) from a job description for later verification against a candidate profile.",
    "parameters": {
        "type": "object",
        "properties": {
            "requirements": {
                "type": "array",
                "description": "A list of distinct requirements extracted from the job description.",
                "items": {
                    "type": "object",
                    "properties": {
                        "facet_type": {
                            "type": "string",
                            "description": "The category of the requirement (e.g., 'skill', 'experience', 'education', 'certification', 'responsibility', 'language')."
                        },
                        "detail": {
                            "type": "string",
                            "description": "The specific detail of the requirement (e.g., 'Python', '5+ years in backend development', 'Bachelor\'s Degree in CS', 'Develop REST APIs')."
                        },
                        "is_required": {
                            "type": "boolean",
                            "description": "Whether this requirement is explicitly stated as mandatory (true) or preferred/optional (false)."
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional: Any specific context provided in the JD about this requirement (e.g., 'for data analysis tasks', 'on AWS cloud platform'). Use null if no specific context."
                        }
                    },
                    "required": ["facet_type", "detail", "is_required"]
                }
            }
        },
        "required": ["requirements"]
    }
}

# Prompt for JD Decomposition
DECOMPOSE_JD_PROMPT = """
You are an AI assistant specialized in breaking down Job Descriptions (JDs) into individual, verifiable requirements (facets).

Job Description:
```
{job_description}
```

Your task is to meticulously analyze this job description and extract distinct requirements that can later be checked against a candidate's profile. For each requirement, identify:
1.  **facet_type**: The category (e.g., 'skill', 'experience', 'education', 'certification', 'responsibility', 'language').
2.  **detail**: The specific requirement (e.g., 'Python', '5+ years in backend development', 'AWS Certified Solutions Architect', 'Develop REST APIs'). Be precise.
3.  **is_required**: Whether the requirement is mandatory (true) or preferred/optional (false).
4.  **context**: Any brief, directly relevant context mentioned alongside the requirement (e.g., 'for data analysis tasks', 'using Agile methodologies'). If no specific context is given, use null.

Focus on extracting concrete, checkable items. Avoid vague entries. Combine closely related phrases into a single requirement where appropriate (e.g., '3-5 years of experience' becomes one 'experience' facet).

Use the `extract_job_requirements` function call schema provided to structure your response. Return *only* the function call.
"""

# --- End NEW --- 

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

    def _get_gemini_model(self, purpose="general_analysis"): # Updated default purpose
        """Get the Gemini model, rotating API keys if necessary."""
        # Ensure API key is configured before creating the model
        current_key = settings.get_next_api_key() # Force rotation/get next key
        logger.info(f"Using API Key ending in: ...{current_key[-4:]} for JD Analysis ({purpose})") # Added purpose to log
        genai.configure(api_key=current_key)
        
        try:
            # Adjust generation config based on purpose if needed in the future
            # For now, using a consistent config, but function calling won't use response_mime_type
            generation_config={ 
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
            }
            model = genai.GenerativeModel(
                self.model_name,
                safety_settings=self.safety_settings,
                generation_config=generation_config 
            )
            return model
        except Exception as e:
            logger.error(f"Fatal error initializing Gemini model: {e}")
            raise e # Re-raise after logging
    
    async def get_relevant_context(self, job_description: str, max_chunks: int = 5) -> str:
        """
        Retrieves relevant context from the vector store based on the job description.
        
        Args:
            job_description: The job description to find context for
            max_chunks: Maximum number of context chunks to retrieve
            
        Returns:
            String containing relevant context or empty string if no context found
        """
        logger.info("Retrieving relevant context for job description using RAG")
        
        try:
            # Extract key terms from the job description to use as query
            # Start with the first 2000 characters to get the core description
            core_desc = job_description[:2000]
            
            # Query for similar job descriptions in the vector DB
            where_filter = {"doc_type": "job"}  # Only get job documents
            
            results = await vector_db_service.query_collection(
                query_texts=[core_desc],
                n_results=max_chunks,
                where=where_filter
            )
            
            if not results or not results.get('documents') or not results['documents'][0]:
                logger.info("No relevant context found in vector store")
                return ""
            
            # Format the retrieved chunks with their relevance score
            context_chunks = []
            documents = results['documents'][0]
            distances = results.get('distances', [[]])[0]
            
            for i, doc in enumerate(documents):
                # Convert distance to similarity score (1 - distance) for easier interpretation
                similarity = 1.0 - distances[i] if i < len(distances) else 0.0
                relevance = f"[Relevance: {similarity:.2f}]"
                context_chunks.append(f"Context {i+1} {relevance}:\n{doc.strip()}")
            
            # Combine all context chunks into a single string
            return "\n\n".join(context_chunks)
            
        except Exception as e:
            logger.error(f"Error retrieving context from vector DB: {e}", exc_info=True)
            return ""
    
    async def analyze_job_description(self, job_id: int, job_description: str) -> Optional[JobAnalysis]:
        """
        Analyzes a job description using Gemini function calling to extract structured information.
        Now enhanced with RAG to retrieve and incorporate relevant context for improved analysis.

        Args:
            job_id: The ID of the job being analyzed.
            job_description: The raw text of the job description to analyze.

        Returns:
            JobAnalysis object containing structured JD information or None if analysis fails.
        """
        try:
            # Get relevant context from the vector database using RAG
            retrieved_context = await self.get_relevant_context(job_description)
            
            # Choose prompt based on whether we found relevant context
            if retrieved_context:
                logger.info(f"Using RAG-enhanced prompt with context for Job ID: {job_id}")
                prompt = JD_RAG_ANALYSIS_PROMPT.format(
                    job_description=job_description,
                    retrieved_context=retrieved_context
                )
            else:
                logger.info(f"Using standard prompt (no relevant context found) for Job ID: {job_id}")
                prompt = JD_ANALYSIS_PROMPT.format(job_description=job_description)
            
            # Get the model with appropriate purpose
            model = self._get_gemini_model(purpose="rag_jd_analysis" if retrieved_context else "jd_analysis")
            
            # Use the tools list for function calling
            tools = [{"function_declarations": [ANALYSIS_SCHEMA]}]

            # The rest of the method remains the same
            logger.info(f"Calling Gemini for JD analysis for job ID: {job_id}.")
            response = await call_gemini_with_backoff(
                model.generate_content,
                prompt,
                tools=tools,
                tool_config={"function_calling_config": {"mode": "any"}}, # Or "any" if text fallback desired
                stream=False
            )

            # Extract function call arguments
            function_call_args = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        # Confirm it's the expected function
                        if function_call.name == "analyze_job_description":
                            function_call_args = json.loads(function_call.args)
                            logger.info(f"Successfully received function call: analyze_job_description")
                            break

            if not function_call_args:
                logger.error(f"Failed to get function call arguments from response.")
                return None

            # Convert to JobAnalysis object:
            job_analysis = JobAnalysis.model_validate(function_call_args)
            logger.info(f"Successfully analyzed job ID: {job_id}, identified {len(job_analysis.required_skills)} required skills.")
            
            return job_analysis

        except Exception as e:
            logger.error(f"Error in JD analysis for job ID {job_id}: {e}", exc_info=True)
            return None

    # --- NEW Method: Decompose JD --- 
    async def decompose_job_description(self, job_id: int, job_description: str) -> Optional[List[JobRequirementFacet]]:
        """
        Decomposes a job description into a list of verifiable requirement facets
        using Gemini function calling.
        
        Args:
            job_id: The ID of the job (for logging).
            job_description: The raw text of the job description.
            
        Returns:
            A list of JobRequirementFacet objects or None if decomposition fails.
        """
        try:
            prompt = DECOMPOSE_JD_PROMPT.format(job_description=job_description)
            model = self._get_gemini_model(purpose="decomposition") # Specific purpose
            tools = [{"function_declarations": [DECOMPOSE_JD_SCHEMA]}]
            
            logger.info(f"Calling Gemini for JD decomposition (Job ID: {job_id}) with function calling.")
            response = await call_gemini_with_backoff(
                model.generate_content,
                prompt,
                tools=tools,
                tool_config={"function_calling_config": {"mode": "any"}}, # Force function call ideally
                # No search needed for decomposition - based solely on the text
                stream=False
            )

            # Extract function call arguments
            function_call_args = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                     # Check for the specific function call name
                    if part.function_call and part.function_call.name == DECOMPOSE_JD_SCHEMA['name']:
                        function_call_args = type(part.function_call).to_dict(part.function_call).get('args', None)
                        logger.info(f"Extracted function call arguments for decomposition (Job ID: {job_id})")
                        break # Found the function call

            if not function_call_args or 'requirements' not in function_call_args:
                response_text = getattr(response, 'text', 'No text content')
                logger.warning(f"Gemini did not return a valid '{DECOMPOSE_JD_SCHEMA['name']}' function call with 'requirements' for decomposition (Job ID: {job_id}). Response text: {response_text[:500]}...")
                return None # Indicate failure

            # Validate each requirement facet using the Pydantic model
            try:
                requirement_list_data = function_call_args['requirements']
                validated_facets = [JobRequirementFacet(**facet_data) for facet_data in requirement_list_data]
                logger.info(f"Successfully validated {len(validated_facets)} requirement facets for Job ID: {job_id}")
                return validated_facets
            except Exception as e: # Catches Pydantic validation errors during list comprehension
                logger.error(f"Failed to validate requirement facets against JobRequirementFacet schema for Job ID: {job_id}: {e}", exc_info=True)
                logger.error(f"Received requirements data: {requirement_list_data}")
                return None

        except Exception as e:
            logger.error(f"Error during JD decomposition agent execution for Job ID: {job_id}: {e}", exc_info=True)
            return None
    # --- End NEW Method --- 
            
    def _validate_job_facets(self, facets_data: Optional[Dict[str, Any]], job_id: str) -> List[JobRequirementFacet]:
        """
        Validate and convert raw facet data to JobRequirementFacet objects.
        
        Args:
            facets_data: Dictionary containing requirements data
            job_id: Job ID for logging purposes
            
        Returns:
            List of validated JobRequirementFacet objects or empty list if validation fails
        """
        if not facets_data:
            return []
            
        requirements = facets_data.get('requirements', [])
        if not requirements:
            return []
            
        validated_facets = []
        for facet_data in requirements:
            try:
                facet = JobRequirementFacet(**facet_data)
                validated_facets.append(facet)
            except Exception as e:
                logger.error(f"Failed to validate facet for Job ID {job_id}: {e}")
                # Continue with other facets
                
        return validated_facets
            
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