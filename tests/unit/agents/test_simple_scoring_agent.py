import pytest
import os
import sys
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock, ANY, create_autospec
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from recruitx_app.agents.simple_scoring_agent import OrchestrationAgent
from recruitx_app.schemas.job import JobRequirementFacet
from recruitx_app.utils.retry_utils import call_gemini_with_backoff
from recruitx_app.services.vector_db_service import vector_db_service
from recruitx_app.core.config import settings
from sklearn.metrics.pairwise import cosine_similarity

# Sample test data
TEST_JOB_DESCRIPTION = """
We are looking for a Python developer with 5+ years of experience.
Required skills: Python, Django, Flask, FastAPI
Nice to have: AWS, Docker, Kubernetes
"""

TEST_CANDIDATE_RESUME = """
Experienced Python developer with 7 years of experience.
Proficient in Python, Django, Flask, and FastAPI.
Some experience with AWS and Docker.
"""

# Mock responses
MOCK_SKILL_EXTRACTION_RESPONSE = {
    "job_skills": ["Python", "Django", "Flask", "FastAPI", "AWS", "Docker", "Kubernetes"],
    "candidate_skills": ["Python", "Django", "Flask", "FastAPI", "AWS", "Docker"]
}

MOCK_SCORE_SYNTHESIS_RESPONSE = {
    "overall_score": 85,
    "explanation": "The candidate has most of the required skills for the job."
}

class TestOrchestrationAgent:
    """Test class for OrchestrationAgent (formerly SimpleScoreGenerator)."""
    
    @pytest.fixture
    def orchestration_agent(self):
        """Create an instance of OrchestrationAgent for testing."""
        # Create a patched version of the settings with mocked API keys
        mock_settings = MagicMock()
        mock_settings.get_next_api_key.return_value = "fake-api-key"
        mock_settings.CANDIDATE_SKILL_EXTRACT_MODEL = "gemini-pro"
        mock_settings.AGENT_SCORE_SYNTHESIS_MODEL = "gemini-pro"
        
        with patch('recruitx_app.agents.simple_scoring_agent.settings', mock_settings), \
             patch('recruitx_app.services.vector_db_service.vector_db_service.generate_embeddings', 
                   new_callable=AsyncMock) as mock_generate_embeddings:
            
            # Mock the embeddings generation to return fake embeddings
            mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            
            # Create the agent
            agent = OrchestrationAgent()
            yield agent
    
    def test_get_gemini_model(self, orchestration_agent):
        """Test the _get_gemini_model method for configuring Gemini model."""
        with patch('google.generativeai.GenerativeModel') as mock_generative_model, \
             patch('google.generativeai.configure') as mock_configure:
            
            # Call the method
            model = orchestration_agent._get_gemini_model(purpose="test")
            
            # Verify API key configuration - use ANY to avoid exact string match issues
            mock_configure.assert_called_once_with(api_key=ANY)
            
            # Verify model creation with correct parameters
            mock_generative_model.assert_called_once()
            assert mock_generative_model.call_args[0][0] == orchestration_agent.model_name
            
            # Verify safety settings were passed
            assert "safety_settings" in mock_generative_model.call_args[1]
            assert len(mock_generative_model.call_args[1]["safety_settings"]) == 4
            
            # Verify generation config
            assert "generation_config" in mock_generative_model.call_args[1]
            
    def test_get_gemini_model_error(self, orchestration_agent):
        """Test error handling in _get_gemini_model method."""
        with patch('google.generativeai.GenerativeModel', side_effect=Exception("API Error")), \
             patch('google.generativeai.configure'):
            
            # Call the method and expect exception
            with pytest.raises(Exception, match="API Error"):
                orchestration_agent._get_gemini_model()
    
    @pytest.mark.asyncio
    async def test_extract_skills_successful(self, orchestration_agent):
        """Test successful extraction of skills."""
        # Setup the mock response
        mock_response = MagicMock()
        mock_response.text = json.dumps(MOCK_SKILL_EXTRACTION_RESPONSE)
        
        # Patch call_gemini_with_backoff to return our mock response
        with patch('recruitx_app.agents.simple_scoring_agent.call_gemini_with_backoff', 
                   new_callable=AsyncMock) as mock_call_gemini:
            mock_call_gemini.return_value = mock_response
            
            # Call the method
            result = await orchestration_agent.extract_skills(
                job_description=TEST_JOB_DESCRIPTION,
                candidate_resume=TEST_CANDIDATE_RESUME
            )
            
            # Verify the result
            assert "job_skills" in result
            assert "candidate_skills" in result
            assert result["job_skills"] == MOCK_SKILL_EXTRACTION_RESPONSE["job_skills"]
            assert result["candidate_skills"] == MOCK_SKILL_EXTRACTION_RESPONSE["candidate_skills"]
            
            # Verify the call was made with the right parameters
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_skills_json_decode_error(self, orchestration_agent):
        """Test handling of JSON decode errors in skill extraction."""
        # Setup the mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.text = "Invalid JSON Response"
        
        # Patch call_gemini_with_backoff to return our mock response
        with patch('recruitx_app.agents.simple_scoring_agent.call_gemini_with_backoff', 
                   new_callable=AsyncMock) as mock_call_gemini:
            mock_call_gemini.return_value = mock_response
            
            # Call the method
            result = await orchestration_agent.extract_skills(
                job_description=TEST_JOB_DESCRIPTION,
                candidate_resume=TEST_CANDIDATE_RESUME
            )
            
            # Verify error handling
            assert "error" in result
            assert "Failed to parse skill extraction JSON" in result["error"]
    
    @pytest.mark.asyncio
    async def test_extract_skills_invalid_format(self, orchestration_agent):
        """Test handling of invalid format in skill extraction response."""
        # Setup the mock response with valid JSON but invalid format
        mock_response = MagicMock()
        mock_response.text = json.dumps({"invalid_key": "This is not the expected format"})
        
        # Patch call_gemini_with_backoff to return our mock response
        with patch('recruitx_app.agents.simple_scoring_agent.call_gemini_with_backoff', 
                   new_callable=AsyncMock) as mock_call_gemini:
            mock_call_gemini.return_value = mock_response
            
            # Call the method
            result = await orchestration_agent.extract_skills(
                job_description=TEST_JOB_DESCRIPTION,
                candidate_resume=TEST_CANDIDATE_RESUME
            )
            
            # Verify error handling
            assert "error" in result
            assert "Invalid format from skill extraction" in result["error"]
    
    @pytest.mark.asyncio
    async def test_extract_skills_exception(self, orchestration_agent):
        """Test handling of general exceptions in skill extraction."""
        # Mock the Gemini model to raise an exception
        with patch.object(orchestration_agent, '_get_gemini_model', side_effect=Exception("Test exception")):
            
            # Call the method
            result = await orchestration_agent.extract_skills(
                job_description=TEST_JOB_DESCRIPTION,
                candidate_resume=TEST_CANDIDATE_RESUME
            )
            
            # Verify error handling
            assert "error" in result
            assert "Test exception" in result["error"]
    
    @pytest.mark.asyncio
    async def test_synthesize_score_successful(self, orchestration_agent):
        """Test successful score synthesis."""
        # Create sample facets and evidence
        facets = [
            JobRequirementFacet(
                facet_type="skill",
                detail="Python programming",
                is_required=True
            )
        ]
        
        retrieved_evidence = {
            0: {
                'ids': [['id1']],
                'documents': [['Python experience']],
                'metadatas': [[{'source': 'resume'}]],
                'distances': [[0.1]]
            }
        }
        
        # Setup the mock response
        mock_response = MagicMock()
        mock_response.text = json.dumps(MOCK_SCORE_SYNTHESIS_RESPONSE)
        
        # Patch both call_gemini_with_backoff and cosine_similarity
        with patch('recruitx_app.agents.simple_scoring_agent.call_gemini_with_backoff', 
                   new_callable=AsyncMock) as mock_call_gemini, \
             patch('recruitx_app.agents.simple_scoring_agent.cosine_similarity', 
                  return_value=0.85) as mock_cosine:
            
            mock_call_gemini.return_value = mock_response
            
            # Call the method
            result = await orchestration_agent.synthesize_score(
                job_description=TEST_JOB_DESCRIPTION,
                candidate_resume=TEST_CANDIDATE_RESUME,
                job_facets=facets,
                retrieved_evidence=retrieved_evidence,
                candidate_id=1
            )
            
            # Verify the result
            assert result["overall_score"] == MOCK_SCORE_SYNTHESIS_RESPONSE["overall_score"]
            assert result["explanation"] == MOCK_SCORE_SYNTHESIS_RESPONSE["explanation"]
            
            # Verify the call was made with the right parameters
            mock_call_gemini.assert_called_once()
            mock_cosine.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_synthesize_score_embedding_failure(self, orchestration_agent):
        """Test handling of embedding generation failure in score synthesis."""
        # Create sample facets and evidence
        facets = [
            JobRequirementFacet(
                facet_type="skill",
                detail="Python programming",
                is_required=True
            )
        ]
        
        retrieved_evidence = {
            0: {
                'ids': [['id1']],
                'documents': [['Python experience']],
                'metadatas': [[{'source': 'resume'}]],
                'distances': [[0.1]]
            }
        }
        
        # Setup the mock response
        mock_response = MagicMock()
        mock_response.text = json.dumps(MOCK_SCORE_SYNTHESIS_RESPONSE)
        
        # Set up the mocks with embedding generation failing
        with patch('recruitx_app.agents.simple_scoring_agent.call_gemini_with_backoff', 
                   new_callable=AsyncMock) as mock_call_gemini, \
             patch('recruitx_app.services.vector_db_service.vector_db_service.generate_embeddings', 
                   new_callable=AsyncMock) as mock_generate_embeddings, \
             patch('recruitx_app.agents.simple_scoring_agent.cosine_similarity', 
                  side_effect=Exception("No embeddings to compare")) as mock_cosine:
            
            mock_call_gemini.return_value = mock_response
            mock_generate_embeddings.return_value = []  # Empty result to simulate failure
            
            # Call the method
            result = await orchestration_agent.synthesize_score(
                job_description=TEST_JOB_DESCRIPTION,
                candidate_resume=TEST_CANDIDATE_RESUME,
                job_facets=facets,
                retrieved_evidence=retrieved_evidence,
                candidate_id=1
            )
            
            # Verify that despite embedding failure, score synthesis continued
            assert result["overall_score"] == MOCK_SCORE_SYNTHESIS_RESPONSE["overall_score"]
            assert result["explanation"] == MOCK_SCORE_SYNTHESIS_RESPONSE["explanation"]
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_synthesize_score_json_decode_error(self, orchestration_agent):
        """Test handling of JSON decode errors in score synthesis."""
        # Create sample facets and evidence
        facets = [
            JobRequirementFacet(
                facet_type="skill",
                detail="Python programming",
                is_required=True
            )
        ]
        
        retrieved_evidence = {
            0: {
                'ids': [['id1']],
                'documents': [['Python experience']],
                'metadatas': [[{'source': 'resume'}]],
                'distances': [[0.1]]
            }
        }
        
        # Create mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.text = "Invalid JSON Response"
        
        # Patch both call_gemini_with_backoff and cosine_similarity
        with patch('recruitx_app.agents.simple_scoring_agent.call_gemini_with_backoff', 
                   new_callable=AsyncMock) as mock_call_gemini, \
             patch('recruitx_app.agents.simple_scoring_agent.cosine_similarity', 
                  return_value=0.85) as mock_cosine:
            
            mock_call_gemini.return_value = mock_response
            
            # Call the method
            result = await orchestration_agent.synthesize_score(
                job_description=TEST_JOB_DESCRIPTION,
                candidate_resume=TEST_CANDIDATE_RESUME,
                job_facets=facets,
                retrieved_evidence=retrieved_evidence,
                candidate_id=1
            )
            
            # Verify error handling
            assert "error" in result
            assert "Failed to parse score synthesis JSON" in result["error"]
    
    @pytest.mark.asyncio
    async def test_synthesize_score_invalid_format(self, orchestration_agent):
        """Test handling of invalid format in score synthesis response."""
        # Create sample facets and evidence
        facets = [
            JobRequirementFacet(
                facet_type="skill",
                detail="Python programming",
                is_required=True
            )
        ]
        
        retrieved_evidence = {
            0: {
                'ids': [['id1']],
                'documents': [['Python experience']],
                'metadatas': [[{'source': 'resume'}]],
                'distances': [[0.1]]
            }
        }
        
        # Create mock response with valid JSON but invalid format
        mock_response = MagicMock()
        mock_response.text = json.dumps({"invalid_key": "This is not the expected format"})
        
        # Patch both call_gemini_with_backoff and cosine_similarity
        with patch('recruitx_app.agents.simple_scoring_agent.call_gemini_with_backoff', 
                   new_callable=AsyncMock) as mock_call_gemini, \
             patch('recruitx_app.agents.simple_scoring_agent.cosine_similarity', 
                  return_value=0.85) as mock_cosine:
            
            mock_call_gemini.return_value = mock_response
            
            # Call the method
            result = await orchestration_agent.synthesize_score(
                job_description=TEST_JOB_DESCRIPTION,
                candidate_resume=TEST_CANDIDATE_RESUME,
                job_facets=facets,
                retrieved_evidence=retrieved_evidence,
                candidate_id=1
            )
            
            # Verify error handling
            assert "error" in result
            assert "Invalid format from score synthesis" in result["error"]
    
    @pytest.mark.asyncio
    async def test_synthesize_score_exception(self, orchestration_agent):
        """Test handling of general exceptions in score synthesis."""
        # Create sample facets and evidence
        facets = [
            JobRequirementFacet(
                facet_type="skill",
                detail="Python programming",
                is_required=True
            )
        ]
        
        retrieved_evidence = {
            0: {
                'ids': [['id1']],
                'documents': [['Python experience']],
                'metadatas': [[{'source': 'resume'}]],
                'distances': [[0.1]]
            }
        }
        
        # Mock the Gemini model to raise an exception
        with patch.object(orchestration_agent, '_get_gemini_model', side_effect=Exception("Test exception")):
            
            # Call the method
            result = await orchestration_agent.synthesize_score(
                job_description=TEST_JOB_DESCRIPTION,
                candidate_resume=TEST_CANDIDATE_RESUME,
                job_facets=facets,
                retrieved_evidence=retrieved_evidence,
                candidate_id=1
            )
            
            # Verify error handling
            assert "error" in result
            assert "Test exception" in result["error"]
    
    def test_format_facets_with_evidence(self, orchestration_agent):
        """Test formatting of requirement facets with evidence."""
        # Create sample facets and evidence
        facets = [
            JobRequirementFacet(
                facet_type="skill",
                detail="Python programming",
                is_required=True
            ),
            JobRequirementFacet(
                facet_type="experience",
                detail="5+ years of experience",
                is_required=True
            )
        ]
        
        retrieved_evidence = {
            0: {
                'ids': [['id1']],
                'documents': [['Python experience']],
                'metadatas': [[{'source': 'resume'}]],
                'distances': [[0.1]]
            },
            1: {
                'ids': [['id2']],
                'documents': [['7 years of experience']],
                'metadatas': [[{'source': 'resume'}]],
                'distances': [[0.2]]
            }
        }
        
        # Call the method
        result = orchestration_agent._format_facets_with_evidence(facets, retrieved_evidence)
        
        # Verify formatting
        assert "Python programming" in result
        assert "5+ years of experience" in result
        assert "Python experience" in result
        assert "7 years of experience" in result
    
    def test_format_external_data_section(self, orchestration_agent):
        """Test formatting of external data section."""
        # Create sample external data dictionary as expected by the method
        external_data = {
            "external_data": {
                "salary_benchmark": {
                    "salary_data": {"median": 120000, "min": 100000, "max": 140000}
                },
                "market_insights": {
                    "market_data": {
                        "demand_growth_rate": 0.15,
                        "job_postings_last_period": 5000,
                        "competition_level": "High",
                        "average_time_to_fill": 45
                    }
                },
                "skill_trends": {
                    "skill_trends": [
                        {"skill": "Python", "demand_growth_rate": 0.22}
                    ]
                }
            }
        }
        
        # Call the method with the dictionary
        result = orchestration_agent._format_external_data_section(external_data)
        
        # Verify formatting based on the sample external_data
        assert "EXTERNAL MARKET DATA:" in result
        assert "SALARY BENCHMARK: $120,000 median salary" in result
        assert "MARKET INSIGHTS: 15.0% demand growth, 5,000 recent job postings" in result
        assert "COMPETITION LEVEL: High, avg. 45 days to fill positions" in result
        assert "TOP GROWING SKILLS:" in result
        assert "Python: +22.0%" in result
    
    def test_format_external_data_section_empty(self, orchestration_agent):
        """Test formatting external data section with empty or missing data."""
        # Call the method with None
        result_none = orchestration_agent._format_external_data_section(None)
        assert "EXTERNAL MARKET DATA: None available" in result_none
        
        # Call the method with empty dict
        result_empty = orchestration_agent._format_external_data_section({})
        assert "EXTERNAL MARKET DATA: None available" in result_empty
        
        # Call the method with external_data but no actual data
        result_no_data = orchestration_agent._format_external_data_section({"external_data": {}})
        assert "EXTERNAL MARKET DATA:" in result_no_data