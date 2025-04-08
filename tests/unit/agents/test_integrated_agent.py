import pytest
import os
import sys
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock, Mock, PropertyMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from recruitx_app.agents.integrated_agent import IntegratedAgent, call_gemini_with_backoff

class TestIntegratedAgent:
    """Test class for the IntegratedAgent."""
    
    @pytest.fixture
    def integrated_agent(self):
        """Create an IntegratedAgent instance for testing."""
        with patch('recruitx_app.agents.integrated_agent.JDAnalysisAgent') as mock_jd_agent, \
             patch('recruitx_app.agents.integrated_agent.CodeExecutionAgent') as mock_code_agent, \
             patch('recruitx_app.agents.integrated_agent.ToolUseAgent') as mock_tool_agent, \
             patch('recruitx_app.agents.integrated_agent.MultimodalAgent') as mock_multimodal_agent, \
             patch('recruitx_app.agents.integrated_agent.settings'):
            
            # Mock the agent creation
            agent = IntegratedAgent()
            
            # Replace with mock agents
            agent.jd_analysis_agent = mock_jd_agent.return_value
            agent.code_execution_agent = mock_code_agent.return_value
            agent.tool_use_agent = mock_tool_agent.return_value
            agent.multimodal_agent = mock_multimodal_agent.return_value
            
            yield agent
        
    @pytest.mark.asyncio
    async def test_init(self, integrated_agent):
        """Test initialization of IntegratedAgent."""
        assert integrated_agent is not None
        assert integrated_agent.model_name is not None
        assert integrated_agent.safety_settings is not None
        assert hasattr(integrated_agent, 'jd_analysis_agent')
        assert hasattr(integrated_agent, 'code_execution_agent')
        assert hasattr(integrated_agent, 'tool_use_agent')
        assert hasattr(integrated_agent, 'multimodal_agent')
        
    @pytest.mark.asyncio
    async def test_get_gemini_model_success(self, integrated_agent):
        """Test successful creation of a Gemini model."""
        with patch('recruitx_app.agents.integrated_agent.genai') as mock_genai:
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            
            result = integrated_agent._get_gemini_model()
            
            assert result == mock_model
            mock_genai.GenerativeModel.assert_called_once()
            
    def test_get_gemini_model_error_with_recovery(self, integrated_agent):
        """Test error recovery when creating a Gemini model."""
        with patch('google.generativeai.GenerativeModel') as mock_generative_model, \
             patch('google.generativeai.configure') as mock_configure:
            # Configure the mock
            mock_model = MagicMock()
            
            # Make the first call fail, but the second succeed
            mock_generative_model.side_effect = [Exception("API Error"), mock_model]
            
            # Call the method
            result = integrated_agent._get_gemini_model()
            
            # Verify the result - we expect recovery and returning the mock model
            assert result == mock_model
            assert mock_generative_model.call_count == 2
            mock_configure.assert_called_once()
            
    def test_get_gemini_model_error_without_recovery(self, integrated_agent):
        """Test handling of persistent errors when creating a Gemini model."""
        with patch('google.generativeai.GenerativeModel') as mock_generative_model, \
             patch('google.generativeai.configure') as mock_configure:
            # Configure the mocks 
            test_exception = Exception("API Error")
            second_exception = Exception("Second API Error")
            
            # Make both calls fail
            mock_generative_model.side_effect = [test_exception, second_exception]
            
            # Call the method and expect an exception
            with pytest.raises(Exception) as exc_info:
                integrated_agent._get_gemini_model()
            
            # Verify that the second exception was raised
            assert "Second API Error" in str(exc_info.value)
            assert mock_generative_model.call_count == 2
            mock_configure.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comprehensive_job_candidate_analysis_success(self, integrated_agent):
        """Test successful execution of comprehensive job candidate analysis."""
        # Mock all the agent methods with AsyncMock to properly handle awaitable coroutines
        with patch.object(integrated_agent.jd_analysis_agent, 'get_industry_insights', new_callable=AsyncMock) as mock_get_industry, \
             patch.object(integrated_agent.tool_use_agent, 'analyze_job_candidate_match', new_callable=AsyncMock) as mock_analyze_match, \
             patch.object(integrated_agent.code_execution_agent, 'generate_and_execute_skill_matcher', new_callable=AsyncMock) as mock_skill_matcher, \
             patch.object(integrated_agent.code_execution_agent, 'generate_skill_visualization', new_callable=AsyncMock) as mock_visualization, \
             patch.object(integrated_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.integrated_agent.call_gemini_with_backoff', new_callable=AsyncMock) as mock_call_gemini, \
             patch('recruitx_app.agents.integrated_agent.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            # Configure the mocks
            mock_get_industry.return_value = {"industry_trends": "AI is growing rapidly"}
            
            mock_analyze_match.return_value = {
                "analysis": {"match_score": 85},
                "steps": [
                    {"action": "Called fetch_job_requirements", "result": {"required_skills": ["Python", "FastAPI"]}},
                    {"action": "Called get_candidate_skills", "result": {"technical_skills": ["Python", "Django"]}}
                ]
            }
            
            mock_skill_matcher.return_value = {
                "overall_match": 80,
                "skill_matches": {"Python": {"score": 90}}
            }
            
            mock_visualization.return_value = {
                "visualizations": {
                    "radar_chart": "base64_data",
                    "heatmap": "base64_data"
                }
            }
            
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "overall_assessment": "Strong match",
                "match_score": 85,
                "key_strengths": ["Python experience"]
            })
            
            # Create a candidate object with text property for thinking output
            mock_candidate = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Thinking process details"
            mock_candidate.content = mock_content
            mock_response.candidates = [mock_candidate]
            
            mock_call_gemini.return_value = mock_response
            
            # Call the method
            result = await integrated_agent.comprehensive_job_candidate_analysis(
                job_id=1,
                candidate_id=2,
                include_visualizations=True
            )
            
            # Verify the result
            assert "job_id" in result
            assert result["job_id"] == 1
            assert "candidate_id" in result
            assert result["candidate_id"] == 2
            assert "analysis_components" in result
            assert "integrated_analysis" in result
            assert "visualizations" in result
            
            # Verify the specific components
            assert "job_analysis" in result["analysis_components"]
            assert "match_analysis" in result["analysis_components"]
            assert "skill_match" in result["analysis_components"]
            
            # Verify the visualization data
            assert "radar_chart" in result["visualizations"]
            assert "heatmap" in result["visualizations"]
            
            # Verify all the mocks were called
            mock_get_industry.assert_called_once()
            mock_analyze_match.assert_called_once()
            mock_skill_matcher.assert_called_once()
            mock_visualization.assert_called_once()
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comprehensive_job_candidate_analysis_without_visualizations(self, integrated_agent):
        """Test comprehensive analysis without visualizations."""
        # Mock all the agent methods with AsyncMock
        with patch.object(integrated_agent.jd_analysis_agent, 'get_industry_insights', new_callable=AsyncMock) as mock_get_industry, \
             patch.object(integrated_agent.tool_use_agent, 'analyze_job_candidate_match', new_callable=AsyncMock) as mock_analyze_match, \
             patch.object(integrated_agent.code_execution_agent, 'generate_and_execute_skill_matcher', new_callable=AsyncMock) as mock_skill_matcher, \
             patch.object(integrated_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.integrated_agent.call_gemini_with_backoff', new_callable=AsyncMock) as mock_call_gemini, \
             patch('recruitx_app.agents.integrated_agent.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        
            # Configure the mocks (simplified for this test)
            mock_get_industry.return_value = {"industry_trends": "AI is growing rapidly"}
            
            mock_analyze_match.return_value = {
                "analysis": {"match_score": 75},
                "steps": [
                    {"action": "Called fetch_job_requirements", "result": {"required_skills": ["Python", "FastAPI"]}},
                    {"action": "Called get_candidate_skills", "result": {"technical_skills": ["Python", "Django"]}}
                ]
            }
            
            mock_skill_matcher.return_value = {
                "overall_match": 70,
                "skill_matches": {"Python": {"score": 85}}
            }
            
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "overall_assessment": "Good match",
                "match_score": 75
            })
            mock_call_gemini.return_value = mock_response
            mock_response.candidates = []
            
            # Call the method with visualizations disabled
            result = await integrated_agent.comprehensive_job_candidate_analysis(
                job_id=1,
                candidate_id=2,
                include_visualizations=False
            )
            
            # Verify the result
            assert "job_id" in result
            assert "candidate_id" in result
            assert "analysis_components" in result
            assert "integrated_analysis" in result
            assert "visualizations" in result
            
            # Visualizations should be empty since we disabled them
            assert result["visualizations"] == {}
            
            # Verify all expected mocks were called (except visualization)
            mock_get_industry.assert_called_once()
            mock_analyze_match.assert_called_once()
            mock_skill_matcher.assert_called_once()
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comprehensive_job_candidate_analysis_with_missing_skills(self, integrated_agent):
        """Test handling of analysis when skills data is missing."""
        # Mock all the agent methods with AsyncMock
        with patch.object(integrated_agent.jd_analysis_agent, 'get_industry_insights', new_callable=AsyncMock) as mock_get_industry, \
             patch.object(integrated_agent.tool_use_agent, 'analyze_job_candidate_match', new_callable=AsyncMock) as mock_analyze_match, \
             patch.object(integrated_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.integrated_agent.call_gemini_with_backoff', new_callable=AsyncMock) as mock_call_gemini, \
             patch('recruitx_app.agents.integrated_agent.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        
            # Configure the mocks with incomplete data
            mock_get_industry.return_value = {"industry_trends": "AI is growing rapidly"}
            
            # Return analysis without the expected skills data structure
            mock_analyze_match.return_value = {
                "analysis": {"match_score": 65},
                "steps": [
                    {"action": "Called other_action", "result": {}}
                ]
            }
            
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "overall_assessment": "Limited data available for assessment",
                "match_score": 65
            })
            mock_call_gemini.return_value = mock_response
            mock_response.candidates = []
            
            # Call the method
            result = await integrated_agent.comprehensive_job_candidate_analysis(
                job_id=1,
                candidate_id=2
            )
            
            # Verify the result
            assert "job_id" in result
            assert "candidate_id" in result
            assert "analysis_components" in result
            assert "integrated_analysis" in result
            
            # Skill match should not be in the results since we didn't have the skills data
            assert "skill_match" not in result["analysis_components"]
            
            # Verify mocks were called as expected
            mock_get_industry.assert_called_once()
            mock_analyze_match.assert_called_once()
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comprehensive_job_candidate_analysis_with_empty_match_steps(self, integrated_agent):
        """Test handling of analysis when match steps are empty."""
        # Mock all the agent methods with AsyncMock
        with patch.object(integrated_agent.jd_analysis_agent, 'get_industry_insights', new_callable=AsyncMock) as mock_get_industry, \
             patch.object(integrated_agent.tool_use_agent, 'analyze_job_candidate_match', new_callable=AsyncMock) as mock_analyze_match, \
             patch.object(integrated_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.integrated_agent.call_gemini_with_backoff', new_callable=AsyncMock) as mock_call_gemini, \
             patch('recruitx_app.agents.integrated_agent.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        
            # Configure the mocks with incomplete data
            mock_get_industry.return_value = {"industry_trends": "AI is growing rapidly"}
            
            # Return analysis without any steps
            mock_analyze_match.return_value = {
                "analysis": {"match_score": 60}
                # No steps included
            }
            
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "overall_assessment": "Limited data available for assessment",
                "match_score": 60
            })
            mock_call_gemini.return_value = mock_response
            mock_response.candidates = []
            
            # Call the method
            result = await integrated_agent.comprehensive_job_candidate_analysis(
                job_id=1,
                candidate_id=2
            )
            
            # Verify the result
            assert "job_id" in result
            assert "candidate_id" in result
            assert "analysis_components" in result
            assert "integrated_analysis" in result
            
            # Skill match should not be in the results since we didn't have the steps
            assert "skill_match" not in result["analysis_components"]
            
            # Verify expected mocks were called
            mock_get_industry.assert_called_once()
            mock_analyze_match.assert_called_once()
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comprehensive_job_candidate_analysis_json_parse_error(self, integrated_agent):
        """Test handling of JSON parsing errors in the comprehensive analysis."""
        # Mock all the required methods with AsyncMock
        with patch.object(integrated_agent.jd_analysis_agent, 'get_industry_insights', new_callable=AsyncMock) as mock_get_industry, \
             patch.object(integrated_agent.tool_use_agent, 'analyze_job_candidate_match', new_callable=AsyncMock) as mock_analyze_match, \
             patch.object(integrated_agent.code_execution_agent, 'generate_and_execute_skill_matcher', new_callable=AsyncMock) as mock_skill_matcher, \
             patch.object(integrated_agent.code_execution_agent, 'generate_skill_visualization', new_callable=AsyncMock) as mock_visualization, \
             patch.object(integrated_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.integrated_agent.call_gemini_with_backoff', new_callable=AsyncMock) as mock_call_gemini, \
             patch('recruitx_app.agents.integrated_agent.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        
            # Configure the mocks
            mock_get_industry.return_value = {"industry_trends": "AI is growing rapidly"}
            
            mock_analyze_match.return_value = {
                "analysis": {"match_score": 85},
                "steps": [
                    {"action": "Called fetch_job_requirements", "result": {"required_skills": ["Python", "FastAPI"]}},
                    {"action": "Called get_candidate_skills", "result": {"technical_skills": ["Python", "Django"]}}
                ]
            }
            
            mock_skill_matcher.return_value = {
                "overall_match": 80,
                "skill_matches": {"Python": {"score": 90}}
            }
            
            # Mock visualization to do nothing for this test
            mock_visualization.return_value = {}
            
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            # Return non-JSON text to simulate parsing error
            mock_response = MagicMock()
            mock_response.text = "This is not valid JSON"
            mock_call_gemini.return_value = mock_response
            mock_response.candidates = []
            
            # Call the method
            result = await integrated_agent.comprehensive_job_candidate_analysis(
                job_id=1,
                candidate_id=2
            )
            
            # Verify the result
            assert "job_id" in result
            assert "candidate_id" in result
            assert "analysis_components" in result
            assert "integrated_analysis" in result
            
            # The integrated analysis should contain the text as a fallback
            assert result["integrated_analysis"] == {"text": "This is not valid JSON"}
            
            # Verify mocks were called as expected
            mock_get_industry.assert_called_once()
            mock_analyze_match.assert_called_once()
            mock_skill_matcher.assert_called_once()
            mock_visualization.assert_called_once() # Check if visualization was called
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_resume_with_visuals_success(self, integrated_agent):
        """Test successful analysis of a resume with visual elements."""
        # Mock the required methods
        with patch.object(integrated_agent.multimodal_agent, 'analyze_document_with_images', new_callable=AsyncMock) as mock_analyze, \
             patch.object(integrated_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.integrated_agent.call_gemini_with_backoff', new_callable=AsyncMock) as mock_call_gemini, \
             patch('recruitx_app.agents.integrated_agent.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            # Configure the mocks
            mock_analyze.return_value = {
                "analysis": "Detailed resume analysis...",
                "skills_detected": ["Python", "Data Science"]
            }
            
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            mock_response = MagicMock()
            mock_response.text = "Market insights for this candidate..."
            mock_call_gemini.return_value = mock_response
            
            # Call the method
            result = await integrated_agent.analyze_resume_with_visuals(
                resume_text="Resume content...",
                image_data_list=[b'image1', b'image2']
            )
            
            # Verify the result
            assert "multimodal_analysis" in result
            assert "market_insights" in result
            assert result["multimodal_analysis"]["analysis"] == "Detailed resume analysis..."
            assert result["market_insights"] == "Market insights for this candidate..."
            
            # Verify the mocks were called with expected arguments
            mock_analyze.assert_called_once_with(
                text_content="Resume content...",
                image_data_list=[b'image1', b'image2']
            )
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_resume_with_visuals_error_handling(self, integrated_agent):
        """Test error handling in resume analysis with visuals."""
        # Mock the required methods
        with patch.object(integrated_agent.multimodal_agent, 'analyze_document_with_images', new_callable=AsyncMock) as mock_analyze:
            
            # Make the analysis method raise an exception
            mock_analyze.side_effect = Exception("Test exception")
            
            # Call the method
            result = await integrated_agent.analyze_resume_with_visuals(
                resume_text="Resume content...",
                image_data_list=[b'image1']
            )
            
            # Verify the result contains the error
            assert "error" in result
            assert result["error"] == "Test exception"
            
            # Verify the mock was called
            mock_analyze.assert_called_once() 