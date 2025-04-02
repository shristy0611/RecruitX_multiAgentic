import pytest
import os
import sys
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from recruitx_app.agents.jd_analysis_agent import JDAnalysisAgent
from recruitx_app.schemas.job import JobAnalysis

# Sample test data
TEST_JOB_ID = 1
TEST_JD = """
Job Title: Senior Backend Developer

About Us:
We are a growing tech company in the finance space.

Responsibilities:
- Design and implement APIs
- Work with SQL databases
- Develop scalable systems

Requirements:
- 5+ years of Python experience
- Knowledge of FastAPI or Django
- SQL database experience
"""

# Mock results for vector DB queries
MOCK_VECTOR_RESULTS = {
    'documents': [
        [
            "Job Title: Python Backend Developer\n\nResponsibilities:\n- Build REST APIs\n- Work with PostgreSQL\n- Implement data processing pipelines",
            "Job Title: Senior Software Engineer\n\nRequirements:\n- 7+ years experience\n- Python, Django\n- Microservices architecture"
        ]
    ],
    'distances': [[0.2, 0.3]]
}

# Mock for function call response
MOCK_FUNCTION_ARGS = {
    "required_skills": ["Python", "FastAPI", "SQL"],
    "preferred_skills": ["Docker", "AWS"],
    "minimum_experience": "5+ years",
    "education": "Bachelor's in Computer Science",
    "responsibilities": ["Design APIs", "Work with databases", "Develop scalable systems"],
    "job_type": "Full-time",
    "seniority_level": "Senior",
    "market_insights": {
        "skill_demand": {
            "high_demand_skills": ["Python", "FastAPI"],
            "trending_skills": ["GraphQL", "Kubernetes"]
        },
        "salary_insights": "$120K - $150K",
        "industry_outlook": "Growing demand in fintech"
    },
    "reasoning": "Analysis conducted based on requirements and industry standards"
}

class TestJDAnalysisAgent:

    @pytest.fixture
    def jd_agent(self):
        """Create a JDAnalysisAgent instance for testing."""
        return JDAnalysisAgent()

    @pytest.mark.asyncio
    @patch('recruitx_app.services.vector_db_service.vector_db_service.query_collection')
    async def test_get_relevant_context(self, mock_query, jd_agent):
        """Test retrieving relevant context from vector DB."""
        # Configure the mock to return our test data
        mock_async = AsyncMock()
        mock_async.return_value = MOCK_VECTOR_RESULTS
        mock_query.side_effect = mock_async

        # Call the method
        result = await jd_agent.get_relevant_context(TEST_JD)

        # Verify the vector DB was queried correctly
        mock_query.assert_called_once()
        assert "doc_type" in mock_query.call_args[1]['where']
        assert mock_query.call_args[1]['where']['doc_type'] == "job"

        # Verify the result contains our mock documents with relevance scores
        assert "Context 1 [Relevance:" in result
        assert "Job Title: Python Backend Developer" in result
        assert "Context 2 [Relevance:" in result
        assert "Job Title: Senior Software Engineer" in result

    @pytest.mark.asyncio
    @patch('recruitx_app.agents.jd_analysis_agent.JDAnalysisAgent.get_relevant_context')
    @patch('recruitx_app.utils.retry_utils.call_gemini_with_backoff')
    async def test_analyze_job_description_with_rag(self, mock_gemini, mock_get_context, jd_agent):
        """Test job description analysis with RAG enhancement."""
        # Mock get_relevant_context to return sample context
        mock_get_context.return_value = "Sample relevant context from similar jobs"

        # Create a mock response object for Gemini
        mock_response = MagicMock()
        mock_candidates = [MagicMock()]
        mock_content = MagicMock()
        mock_parts = [MagicMock()]
        mock_function_call = MagicMock()
        
        # Build the response structure
        mock_response.candidates = mock_candidates
        mock_candidates[0].content = mock_content
        mock_content.parts = mock_parts
        mock_parts[0].function_call = mock_function_call
        mock_function_call.name = "analyze_job_description"
        mock_function_call.args = MOCK_FUNCTION_ARGS
        
        # Configure the mock for Gemini call
        mock_async = AsyncMock()
        mock_async.return_value = mock_response
        mock_gemini.side_effect = mock_async

        # Call the method
        result = await jd_agent.analyze_job_description(TEST_JOB_ID, TEST_JD)

        # Verify get_relevant_context was called
        mock_get_context.assert_called_once_with(TEST_JD)
        
        # Verify the RAG prompt was used (JD_RAG_ANALYSIS_PROMPT)
        assert "retrieved_context" in mock_gemini.call_args[0][1]
        
        # Verify the response was properly processed
        assert isinstance(result, JobAnalysis)
        assert "Python" in result.required_skills
        assert "FastAPI" in result.required_skills
        assert "5+ years" == result.minimum_experience
        assert "high_demand_skills" in result.market_insights["skill_demand"]

    @pytest.mark.asyncio
    @patch('recruitx_app.agents.jd_analysis_agent.JDAnalysisAgent.get_relevant_context')
    @patch('recruitx_app.utils.retry_utils.call_gemini_with_backoff')
    async def test_analyze_job_description_without_rag(self, mock_gemini, mock_get_context, jd_agent):
        """Test job description analysis without RAG enhancement (fallback to standard prompt)."""
        # Mock get_relevant_context to return empty string (no context found)
        mock_get_context.return_value = ""

        # Create a mock response object for Gemini similar to previous test
        mock_response = MagicMock()
        mock_candidates = [MagicMock()]
        mock_content = MagicMock()
        mock_parts = [MagicMock()]
        mock_function_call = MagicMock()
        
        # Build the response structure
        mock_response.candidates = mock_candidates
        mock_candidates[0].content = mock_content
        mock_content.parts = mock_parts
        mock_parts[0].function_call = mock_function_call
        mock_function_call.name = "analyze_job_description"
        mock_function_call.args = MOCK_FUNCTION_ARGS
        
        # Configure the mock for Gemini call
        mock_async = AsyncMock()
        mock_async.return_value = mock_response
        mock_gemini.side_effect = mock_async

        # Call the method
        result = await jd_agent.analyze_job_description(TEST_JOB_ID, TEST_JD)

        # Verify get_relevant_context was called
        mock_get_context.assert_called_once_with(TEST_JD)
        
        # Verify the standard prompt was used (JD_ANALYSIS_PROMPT)
        assert "retrieved_context" not in mock_gemini.call_args[0][1]
        
        # Verify the response was properly processed
        assert isinstance(result, JobAnalysis)
        assert "Python" in result.required_skills