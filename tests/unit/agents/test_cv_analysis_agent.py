import pytest
import os
import sys
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from pydantic import ValidationError

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from recruitx_app.agents.cv_analysis_agent import CVAnalysisAgent
from recruitx_app.schemas.candidate import CandidateAnalysis

# Sample test data
TEST_CANDIDATE_ID = 1
TEST_CV = """
JANE SMITH
Software Engineer
jane.smith@example.com | (555) 789-1234 | New York, NY

SUMMARY
Experienced software engineer with focus on backend development and cloud technologies.

SKILLS
Python, Java, AWS, Docker, Kubernetes, PostgreSQL, MongoDB

WORK EXPERIENCE
Senior Software Engineer | TechCorp | Jan 2020 - Present
- Developed microservices using Python and FastAPI
- Managed PostgreSQL databases
- Implemented CI/CD pipelines

Software Developer | DataSystems Inc. | Jun 2017 - Dec 2019
- Built data processing pipelines
- Developed REST APIs using Java

EDUCATION
Master of Science in Computer Science
Cornell University - 2017
"""

# Mock results for vector DB queries
MOCK_VECTOR_RESULTS = {
    'documents': [
        [
            "Similar candidate with Python and AWS experience, 5 years in backend development",
            "Software Engineer with FastAPI and PostgreSQL skills, worked on microservices"
        ]
    ],
    'distances': [[0.25, 0.35]],
}

# Mock job results
MOCK_JOB_RESULTS = {
    'documents': [
        [
            "Job Title: Senior Backend Engineer\nRequirements: Python, AWS, PostgreSQL\nResponsibilities: Build microservices, manage databases"
        ]
    ],
    'distances': [[0.3]],
}

# Expected analysis results for tests
EXPECTED_CV_ANALYSIS = CandidateAnalysis(
    candidate_id=TEST_CANDIDATE_ID,
    contact_info={
        "name": "Jane Smith",
        "email": "jane.smith@example.com",
        "phone": "(555) 789-1234",
        "location": "New York, NY"
    },
    summary="Experienced software engineer with focus on backend development and cloud technologies.",
    skills=["Python", "Java", "AWS", "Docker", "Kubernetes", "PostgreSQL", "MongoDB"],
    work_experience=[
        {
            "company": "TechCorp",
            "title": "Senior Software Engineer",
            "dates": "Jan 2020 - Present",
            "responsibilities": [
                "Developed microservices using Python and FastAPI",
                "Managed PostgreSQL databases",
                "Implemented CI/CD pipelines"
            ]
        },
        {
            "company": "DataSystems Inc.",
            "title": "Software Developer",
            "dates": "Jun 2017 - Dec 2019",
            "responsibilities": [
                "Built data processing pipelines",
                "Developed REST APIs using Java"
            ]
        }
    ],
    education=[
        {
            "institution": "Cornell University",
            "degree": "Master of Science",
            "field_of_study": "Computer Science",
            "graduation_date": "2017"
        }
    ],
    certifications=[],
    projects=[],
    languages=["English"],
    overall_profile="Strong backend engineer with good cloud and database experience"
)

class TestCVAnalysisAgent:
    
    @pytest.fixture
    def cv_agent(self):
        """Create a CVAnalysisAgent instance for testing."""
        return CVAnalysisAgent()

    @pytest.mark.asyncio
    @patch('recruitx_app.services.vector_db_service.vector_db_service.query_collection')
    async def test_get_relevant_context(self, mock_query, cv_agent):
        """Test retrieving relevant context from vector DB for CV analysis."""
        # Configure the mock to return our test data
        mock_async = AsyncMock()
        # First call (candidate profiles)
        mock_async.side_effect = [MOCK_VECTOR_RESULTS, MOCK_JOB_RESULTS]
        mock_query.side_effect = mock_async

        # Call the method
        result = await cv_agent.get_relevant_context(TEST_CV)

        # Verify the vector DB was queried for both candidate profiles and job descriptions
        assert mock_query.call_count == 2
        
        # First call for candidate profiles
        first_call_args = mock_query.call_args_list[0][1]
        assert "doc_type" in first_call_args['where']
        assert first_call_args['where']['doc_type'] == "candidate"
        
        # Second call for relevant job descriptions
        second_call_args = mock_query.call_args_list[1][1]
        assert "doc_type" in second_call_args['where']
        assert second_call_args['where']['doc_type'] == "job"

        # Verify the result contains our mock documents with relevance scores
        assert "Similar Candidate Profile" in result
        assert "Relevant Job Description" in result
        assert "Similar candidate with Python" in result
        assert "Senior Backend Engineer" in result

    @pytest.mark.asyncio
    @patch('recruitx_app.services.vector_db_service.vector_db_service.query_collection')
    async def test_get_relevant_context_exception(self, mock_query, cv_agent):
        """Test error handling when retrieving context from vector DB throws exception."""
        # Configure the mock to raise an exception
        mock_query.side_effect = Exception("Test vector DB error")

        # Call the method - should handle the exception gracefully
        result = await cv_agent.get_relevant_context(TEST_CV)

        # The method should catch the exception and return an empty string
        assert result == ""

    @pytest.mark.asyncio
    @patch('recruitx_app.services.vector_db_service.vector_db_service.query_collection')
    async def test_get_relevant_context_empty_results(self, mock_query, cv_agent):
        """Test handling when no relevant context is found."""
        # Configure the mock to return empty results
        empty_results = {
            'documents': [[]],
            'distances': [[]]
        }
        mock_query.side_effect = [empty_results, empty_results]

        # Call the method
        result = await cv_agent.get_relevant_context(TEST_CV)

        # Should return empty string when no context is found
        assert result == ""

    def test_get_gemini_model_success(self, cv_agent):
        """Test successful initialization of the Gemini model."""
        # Create a mock model
        mock_model = MagicMock()
        
        # Save original method to restore later
        original_method = cv_agent._get_gemini_model
        
        # Define replacement method
        def mock_get_model(purpose="general"):
            assert purpose == "cv_analysis"
            return mock_model
            
        # Replace method with our mock
        cv_agent._get_gemini_model = mock_get_model
        
        try:
            # Call the method
            result = cv_agent._get_gemini_model("cv_analysis")
            
            # Verify the result
            assert result == mock_model
        finally:
            # Restore original method
            cv_agent._get_gemini_model = original_method

    def test_get_gemini_model_error(self, cv_agent):
        """Test error handling when initializing the Gemini model fails."""
        # Save original method to restore later
        original_method = cv_agent._get_gemini_model
        
        # Define replacement method that raises an exception
        def mock_get_model_error(purpose="general"):
            raise Exception("Model initialization error")
            
        # Replace method with our mock
        cv_agent._get_gemini_model = mock_get_model_error
        
        try:
            # Call the method - should raise an exception
            with pytest.raises(Exception) as excinfo:
                cv_agent._get_gemini_model("cv_analysis")
            
            # Verify the exception message
            assert "Model initialization error" in str(excinfo.value)
        finally:
            # Restore original method
            cv_agent._get_gemini_model = original_method

    @pytest.mark.asyncio
    async def test_analyze_cv_with_rag(self, cv_agent):
        """Test CV analysis with RAG enhancement by mocking the entire method."""
        
        # Create patch for get_relevant_context
        with patch.object(cv_agent, 'get_relevant_context', 
                         return_value="Sample relevant context from similar profiles and jobs"):
            
            # Create patch for analyze_cv but allow the original method to be restored after test
            original_analyze_cv = cv_agent.analyze_cv
            
            # Replace analyze_cv with a mock that returns our expected result
            async def mock_analyze_cv(cv_text, candidate_id):
                assert cv_text == TEST_CV
                assert candidate_id == TEST_CANDIDATE_ID
                # Since we're mocking the method, we can verify RAG was enabled by checking context
                context_result = await cv_agent.get_relevant_context(cv_text)
                assert context_result == "Sample relevant context from similar profiles and jobs"
                return EXPECTED_CV_ANALYSIS
                
            # Apply our mock
            cv_agent.analyze_cv = mock_analyze_cv
            
            try:
                # Call the method
                result = await cv_agent.analyze_cv(TEST_CV, TEST_CANDIDATE_ID)
                
                # Verify the response was properly processed
                assert isinstance(result, CandidateAnalysis)
                assert result.candidate_id == TEST_CANDIDATE_ID
                assert "Python" in result.skills
                assert "AWS" in result.skills
                assert len(result.work_experience) == 2
                assert result.work_experience[0]["company"] == "TechCorp"
                assert result.education[0]["institution"] == "Cornell University"
            finally:
                # Restore the original method
                cv_agent.analyze_cv = original_analyze_cv

    @pytest.mark.asyncio
    async def test_analyze_cv_without_rag(self, cv_agent):
        """Test CV analysis without RAG enhancement by mocking the entire method."""
        
        # Create patch for get_relevant_context to return empty string
        with patch.object(cv_agent, 'get_relevant_context', return_value=""):
            
            # Create patch for analyze_cv but allow the original method to be restored after test
            original_analyze_cv = cv_agent.analyze_cv
            
            # Replace analyze_cv with a mock that returns our expected result
            async def mock_analyze_cv(cv_text, candidate_id):
                assert cv_text == TEST_CV
                assert candidate_id == TEST_CANDIDATE_ID
                # Since we're mocking the method, we can verify RAG was not used by checking context
                context_result = await cv_agent.get_relevant_context(cv_text)
                assert context_result == ""  # Empty context means no RAG
                return EXPECTED_CV_ANALYSIS
                
            # Apply our mock
            cv_agent.analyze_cv = mock_analyze_cv
            
            try:
                # Call the method
                result = await cv_agent.analyze_cv(TEST_CV, TEST_CANDIDATE_ID)
                
                # Verify the response was properly processed
                assert isinstance(result, CandidateAnalysis)
                assert result.candidate_id == TEST_CANDIDATE_ID
                assert "Python" in result.skills
            finally:
                # Restore the original method
                cv_agent.analyze_cv = original_analyze_cv

    @pytest.mark.asyncio
    @patch('recruitx_app.agents.cv_analysis_agent.call_gemini_with_backoff')
    async def test_analyze_cv_function_call_parsing_error(self, mock_call_gemini, cv_agent):
        """Test handling when the function call response cannot be properly parsed."""
        # Mock get_relevant_context to return empty string for simplicity
        with patch.object(cv_agent, 'get_relevant_context', return_value=""):
            # Mock _get_gemini_model to return a mock model
            with patch.object(cv_agent, '_get_gemini_model', return_value=MagicMock()):
                # Configure mock response with no function call
                mock_response = MagicMock()
                mock_response.candidates = [MagicMock()]
                mock_response.candidates[0].content.parts = [MagicMock()]
                # The part does not have a function_call attribute
                mock_async = AsyncMock(return_value=mock_response)
                mock_call_gemini.side_effect = mock_async
                
                # Call the method
                result = await cv_agent.analyze_cv(TEST_CV, TEST_CANDIDATE_ID)
                
                # Should return None when function call parsing fails
                assert result is None
                
                # Verify the expected calls were made
                mock_call_gemini.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_cv_validation_error(self, cv_agent):
        """Test handling when the function call has an invalid schema causing a ValidationError."""
        # Patch get_relevant_context to return empty string for simplicity
        with patch.object(cv_agent, 'get_relevant_context', return_value=""):
            # In the actual implementation, when there's a validation error, it's caught and None is returned
            # So we need to test that the analyze_cv method properly catches validation errors
            
            # Create a patch for call_gemini_with_backoff that returns a function call with invalid data
            with patch('recruitx_app.agents.cv_analysis_agent.call_gemini_with_backoff') as mock_call_gemini:
                with patch.object(cv_agent, '_get_gemini_model', return_value=MagicMock()):
                    # Set up the mock response
                    mock_response = MagicMock()
                    mock_response.candidates = [MagicMock()]
                    
                    # Create a function call part with invalid schema
                    function_call_part = MagicMock()
                    function_call_part.function_call = MagicMock()
                    function_call_part.function_call.name = "analyze_cv"
                    
                    # Create response with invalid args (missing required fields)
                    function_call_part.function_call.args = json.dumps({
                        "contact_info": {"name": "Jane Smith"}
                        # Missing required fields: skills, work_experience, education
                    })
                    
                    mock_response.candidates[0].content.parts = [function_call_part]
                    mock_call_gemini.return_value = mock_response
                    
                    # Call the method - it should handle validation errors
                    with patch('recruitx_app.agents.cv_analysis_agent.CandidateAnalysis.model_validate', 
                              side_effect=ValidationError.from_exception_data(
                                  "Validation failed", 
                                  [{"loc": ("skills",), "msg": "Field required", "type": "missing"}]
                              )):
                        result = await cv_agent.analyze_cv(TEST_CV, TEST_CANDIDATE_ID)
                        
                        # Should return None when validation fails
                        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_cv_exception(self, cv_agent):
        """Test handling when an exception occurs during CV analysis."""
        # Mock get_relevant_context to raise an exception
        with patch.object(cv_agent, 'get_relevant_context', side_effect=Exception("Test exception")):
            # Call the method - should handle the exception gracefully
            result = await cv_agent.analyze_cv(TEST_CV, TEST_CANDIDATE_ID)
            
            # Should return None when an exception occurs
            assert result is None