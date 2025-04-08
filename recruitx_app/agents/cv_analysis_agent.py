import google.generativeai as genai
import json
import logging
from typing import Dict, Any, List, Optional

from recruitx_app.core.config import settings
from recruitx_app.schemas.candidate import CandidateAnalysis # Import the schema
from recruitx_app.utils.retry_utils import call_gemini_with_backoff # Import retry helper
from recruitx_app.services.vector_db_service import vector_db_service # Import the vector db service

logger = logging.getLogger(__name__)

# Define the structured output schema for function calling
# Aligned with schemas.candidate.CandidateAnalysis, excluding candidate_id which is added later
CV_ANALYSIS_SCHEMA = {
    "name": "analyze_cv",
    "description": "Analyze a candidate CV/Resume to extract structured information.",
    "parameters": {
        "type": "object",
        "properties": {
            "contact_info": {
                "type": "object",
                "description": "Contact details (name, email, phone, location). Use null for missing fields.",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "location": {"type": "string"}
                },
                 # Make the object itself optional, or require specific sub-fields?
                 # Let's make sub-fields optional by not listing them in 'required'.
            },
            "summary": {
                "type": "string",
                "description": "Professional summary or objective statement. Use null if not present."
            },
            "skills": {
                "type": "array",
                "description": "List of technical and soft skills identified.",
                "items": {"type": "string"}
            },
            "work_experience": {
                "type": "array",
                "description": "List of work experiences.",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "title": {"type": "string"},
                        "dates": {"type": "string", "description": "Employment dates (e.g., 'Jan 2020 - Present')"},
                        "responsibilities": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["company", "title"] # Core experience details
                }
            },
            "education": {
                "type": "array",
                "description": "List of educational qualifications.",
                "items": {
                    "type": "object",
                    "properties": {
                        "institution": {"type": "string"},
                        "degree": {"type": "string"},
                        "field_of_study": {"type": "string"},
                        "graduation_date": {"type": "string"}
                    },
                    "required": ["institution"] # Core education details
                }
            },
            "certifications": {
                "type": "array",
                "description": "List of professional certifications or licenses. Use empty list if none found.",
                "items": {"type": "string"}
            },
            "projects": {
                "type": "array",
                "description": "List of notable projects mentioned. Use empty list if none found.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "technologies": {"type": "array", "items": {"type": "string"}}
                    }
                    # Can add 'required' for project name if desired
                }
            },
            "languages": {
                "type": "array",
                "description": "Languages spoken by the candidate. Use empty list if none found.",
                "items": {"type": "string"}
            },
            "overall_profile": {
                "type": "string",
                "description": "A brief overall assessment based purely on the CV content. Use null if unable to assess."
            }
        },
        # Define core required fields for a minimally useful CV analysis
        "required": ["skills", "work_experience", "education", "contact_info"]
    }
}

# Updated prompt for function calling
CV_ANALYSIS_PROMPT = """
You are an expert CV/Resume Analyzer. Your task is to meticulously parse the following candidate document and extract structured information using the provided `analyze_cv` function schema.

CV/Resume Text:
{cv_text}

Focus on extracting details for all fields defined in the `analyze_cv` function schema, including:
- Contact Information (name, email, phone, location)
- Professional Summary/Objective
- Technical and Soft Skills
- Work Experience (company, title, dates, responsibilities)
- Education (institution, degree, field, dates)
- Certifications
- Projects
- Languages
- Provide a brief Overall Profile assessment based *only* on the CV content.

**Important:** Adhere strictly to the `analyze_cv` function schema.
- For optional top-level fields (`summary`, `certifications`, `projects`, `languages`, `overall_profile`), use `null` if no information is found.
- For list fields (`skills`, `work_experience`, `education`, `certifications`, `projects`, `languages`), use an empty list `[]` if no relevant items are found.
- For the optional fields within `contact_info` (email, phone, location), use `null` if that specific piece of contact info is missing.
- Do **not** invent or assume information not present in the text. Your response MUST be a call to the `analyze_cv` function.
"""

# New RAG-enhanced prompt for CV analysis
CV_RAG_ANALYSIS_PROMPT = """
You are an expert CV/Resume Analyzer. Your task is to meticulously parse the following candidate document and extract structured information using the provided `analyze_cv` function schema.

CV/Resume Text:
{cv_text}

Here is relevant context from similar profiles and industry information that may help your analysis:
```
{retrieved_context}
```

Focus on extracting details for all fields defined in the `analyze_cv` function schema, including:
- Contact Information (name, email, phone, location)
- Professional Summary/Objective
- Technical and Soft Skills - use the relevant context to better identify and categorize skills
- Work Experience (company, title, dates, responsibilities)
- Education (institution, degree, field, dates)
- Certifications
- Projects
- Languages
- Provide a brief Overall Profile assessment based *only* on the CV content and the relevant context.

Use the retrieved context to:
1. Better understand industry-specific terminology and skillsets
2. Identify skills that might be implied but not explicitly stated
3. Standardize how you name and categorize skills to align with industry norms
4. Enrich your understanding of the candidate's experience and qualifications

**Important:** Adhere strictly to the `analyze_cv` function schema.
- For optional top-level fields (`summary`, `certifications`, `projects`, `languages`, `overall_profile`), use `null` if no information is found.
- For list fields (`skills`, `work_experience`, `education`, `certifications`, `projects`, `languages`), use an empty list `[]` if no relevant items are found.
- For the optional fields within `contact_info` (email, phone, location), use `null` if that specific piece of contact info is missing.
- Do **not** invent or assume information not present in the text. Your response MUST be a call to the `analyze_cv` function.
"""

class CVAnalysisAgent:
    """
    Agent responsible for analyzing candidate CVs using Gemini function calling.
    """

    def __init__(self):
        """Initialize the CV Analysis Agent."""
        self.model_name = settings.GEMINI_PRO_MODEL
        # Safety settings can be defined once
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        # Initial configuration, will be re-configured in _get_gemini_model
        genai.configure(api_key=settings.get_next_api_key())

    def _get_gemini_model(self, purpose="cv_analysis"):
        """Get the Gemini model, managing API key rotation."""
        current_key = settings.get_next_api_key() # Force rotation/get next key
        logger.info(f"Using API Key ending in: ...{current_key[-4:]} for CV Analysis ({purpose})")
        genai.configure(api_key=current_key)

        try:
            # Define generation config for consistency
            generation_config = genai.GenerationConfig(
                temperature=0.1, # Low temp for structured output consistency
                top_p=0.95,
                top_k=40
            )
            model = genai.GenerativeModel(
                self.model_name,
                safety_settings=self.safety_settings,
                generation_config=generation_config
                # response_mime_type is NOT set for function calling
            )
            return model
        except Exception as e:
            logger.error(f"Fatal error initializing Gemini model for CV Analysis: {e}")
            raise e # Re-raise after logging
            
    async def get_relevant_context(self, cv_text: str, max_chunks: int = 5) -> str:
        """
        Retrieves relevant context from the vector store based on the CV content.
        
        Args:
            cv_text: The CV text to find context for
            max_chunks: Maximum number of context chunks to retrieve
            
        Returns:
            String containing relevant context or empty string if no context found
        """
        logger.info("Retrieving relevant context for CV analysis using RAG")
        
        try:
            # Extract key terms from the CV to use as query
            # Start with the first 2000 characters to get the core content
            core_text = cv_text[:2000]
            
            # Prepare two queries:
            # 1. For similar candidate profiles
            # 2. For relevant job descriptions that match the profile
            
            # Query for similar candidate profiles
            candidate_filter = {"doc_type": "candidate"}
            candidate_results = await vector_db_service.query_collection(
                query_texts=[core_text],
                n_results=max(2, max_chunks // 2),  # Allocate some chunks for candidate profiles
                where=candidate_filter
            )
            
            # Query for relevant job descriptions that match the profile
            job_filter = {"doc_type": "job"}
            job_results = await vector_db_service.query_collection(
                query_texts=[core_text],
                n_results=max(2, max_chunks // 2),  # Allocate some chunks for job descriptions
                where=job_filter
            )
            
            # Format the context chunks from both queries
            context_chunks = []
            
            # Process candidate results
            if candidate_results and candidate_results.get('documents') and candidate_results['documents'][0]:
                documents = candidate_results['documents'][0]
                distances = candidate_results.get('distances', [[]])[0]
                
                for i, doc in enumerate(documents):
                    similarity = 1.0 - distances[i] if i < len(distances) else 0.0
                    relevance = f"[Relevance: {similarity:.2f}]"
                    context_chunks.append(f"Similar Candidate Profile {i+1} {relevance}:\n{doc.strip()}")
            
            # Process job results
            if job_results and job_results.get('documents') and job_results['documents'][0]:
                documents = job_results['documents'][0]
                distances = job_results.get('distances', [[]])[0]
                
                for i, doc in enumerate(documents):
                    similarity = 1.0 - distances[i] if i < len(distances) else 0.0
                    relevance = f"[Relevance: {similarity:.2f}]"
                    context_chunks.append(f"Relevant Job Description {i+1} {relevance}:\n{doc.strip()}")
            
            if not context_chunks:
                logger.info("No relevant context found in vector store for CV analysis")
                return ""
                
            # Combine all context chunks into a single string
            return "\n\n".join(context_chunks)
            
        except Exception as e:
            logger.error(f"Error retrieving context from vector DB for CV analysis: {e}", exc_info=True)
            return ""

    async def analyze_cv(self, cv_text: str, candidate_id: int) -> Optional[CandidateAnalysis]:
        """
        Analyze a CV using Gemini function calling to extract structured information.
        Now enhanced with RAG to retrieve and incorporate relevant context for improved analysis.

        Args:
            cv_text: The raw text of the CV to analyze.
            candidate_id: The ID of the candidate being analyzed.

        Returns:
            CandidateAnalysis object containing structured CV information or None if analysis fails.
        """
        try:
            # Get relevant context from the vector database using RAG
            retrieved_context = await self.get_relevant_context(cv_text)
            
            # Choose prompt based on whether we found relevant context
            if retrieved_context:
                logger.info(f"Using RAG-enhanced prompt with context for Candidate ID: {candidate_id}")
                prompt = CV_RAG_ANALYSIS_PROMPT.format(
                    cv_text=cv_text,
                    retrieved_context=retrieved_context
                )
            else:
                logger.info(f"Using standard prompt (no relevant context found) for Candidate ID: {candidate_id}")
                prompt = CV_ANALYSIS_PROMPT.format(cv_text=cv_text)
            
            # Get the model with appropriate purpose
            model = self._get_gemini_model(purpose="rag_cv_analysis" if retrieved_context else "cv_analysis")
            
            tools = [{"function_declarations": [CV_ANALYSIS_SCHEMA]}]

            logger.info(f"Calling Gemini for CV analysis (Candidate ID: {candidate_id}) with function calling.")
            response = await call_gemini_with_backoff(
                model.generate_content,
                prompt,
                tools=tools,
                # Force model to call the function
                tool_config={"function_calling_config": {"mode": "any"}}, # Or "any" if text fallback desired
                stream=False
            )

            # Extract function call arguments
            function_call_args = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    # Check if it's the specific function we expect
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        if function_call.name == "analyze_cv":
                            function_call_args = json.loads(function_call.args)
                            logger.info(f"Successfully received function call: analyze_cv")
                            break

            if not function_call_args:
                logger.error(f"Failed to get function call arguments from response for Candidate ID: {candidate_id}")
                return None

            # Create a CandidateAnalysis object from the arguments
            # Add the candidate_id, which isn't part of the function call args
            function_call_args["candidate_id"] = candidate_id
            
            # Validate and create the CandidateAnalysis object
            candidate_analysis = CandidateAnalysis.model_validate(function_call_args)
            
            logger.info(f"Successfully analyzed Candidate ID: {candidate_id}")
            return candidate_analysis

        except Exception as e:
            logger.error(f"Error in CV analysis for Candidate ID {candidate_id}: {e}", exc_info=True)
            return None
