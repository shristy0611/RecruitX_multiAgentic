import pytest
import os
import sys
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient
import aiohttp
import json
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from recruitx_app.services.external_tool_service import ExternalToolService, APIResponse

class TestExternalToolService:
    
    @pytest.fixture
    def external_tool_service(self):
        """Create an ExternalToolService instance for testing."""
        return ExternalToolService()
    
    @pytest.mark.asyncio
    async def test_make_api_request_success(self, external_tool_service):
        """Test making a successful API request."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            # Configure the mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"data": "test_data", "success": True})
            mock_response.text = AsyncMock(return_value='{"data": "test_data", "success": true}')
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_request.return_value = mock_response
            
            # Call the method
            result = await external_tool_service._make_api_request(
                url="https://api.example.com/endpoint",
                method="GET",
                headers={"Authorization": "Bearer test_token"},
                params={"param1": "value1"},
                use_cache=True,
                cache_key="test_cache_key"
            )
            
            # Verify the request was made correctly - use named parameters to match implementation
            mock_request.assert_called_once()
            call_args = mock_request.call_args[1]
            assert call_args["method"] == "GET"
            assert call_args["url"] == "https://api.example.com/endpoint"
            assert call_args["headers"] == {"Authorization": "Bearer test_token"}
            assert call_args["params"] == {"param1": "value1"}
            assert call_args["json"] is None
            
            # Verify the result (result contains the entire response, not just data)
            assert result["success"] is True
            assert result["data"] == {"data": "test_data", "success": True}
    
    @pytest.mark.asyncio
    async def test_make_api_request_error(self, external_tool_service):
        """Test handling API request errors."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            # Configure the mock response for an error
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.text = AsyncMock(return_value='{"error": {"message": "Not Found"}}')
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_request.return_value = mock_response
            
            # Call the method
            result = await external_tool_service._make_api_request(
                url="https://api.example.com/endpoint",
                method="GET"
            )
            
            # Verify the result
            assert result["success"] is False
            assert result["data"] is None
            assert "Not Found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_make_api_request_exception(self, external_tool_service):
        """Test handling exceptions during API requests."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            # Configure the mock to raise an exception
            mock_request.side_effect = Exception("Test exception")
            
            # Call the method
            result = await external_tool_service._make_api_request(
                url="https://api.example.com/endpoint",
                method="GET"
            )
            
            # Verify the result
            assert result["success"] is False
            assert result["data"] is None
            assert "Test exception" in result["error"]
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, external_tool_service):
        """Test that caching works correctly."""
        # Clear the cache first
        external_tool_service._cache = {}
        external_tool_service._cache_timestamps = {}
        
        with patch('aiohttp.ClientSession.request') as mock_request:
            # Configure the mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"data": "test_data", "success": True})
            mock_response.text = AsyncMock(return_value='{"data": "test_data", "success": true}')
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_request.return_value = mock_response
            
            # Call the method with caching enabled
            cache_key = "test_cache_key"
            result1 = await external_tool_service._make_api_request(
                url="https://api.example.com/endpoint",
                method="GET",
                use_cache=True,
                cache_key=cache_key
            )
            
            # Call it again with same cache key to test cache hit
            result2 = await external_tool_service._make_api_request(
                url="https://api.example.com/endpoint",
                method="GET",
                use_cache=True,
                cache_key=cache_key
            )
            
            # Verify the request was made only once (on the first call)
            assert mock_request.call_count == 1
            
            # Verify both results are the same
            assert result1 == result2
    
    @pytest.mark.asyncio
    async def test_get_salary_benchmark(self, external_tool_service):
        """Test getting salary benchmark data."""
        # Set API key and URL for testing
        original_api_key = external_tool_service.salary_api_key
        original_api_url = external_tool_service.salary_api_base_url
    
        try:
            # Set values to ensure API request is made (not example.com)
            external_tool_service.salary_api_key = "test_key"
            external_tool_service.salary_api_base_url = "https://api.salarydata.test.com/v1"
    
            # Mock the _make_api_request method
            with patch.object(external_tool_service, '_make_api_request') as mock_request:
                # Configure the mock to return a simulated response
                mock_response: APIResponse = {
                    "success": True,
                    "data": {
                        "job_title": "Software Engineer",
                        "salary_data": {
                            "min": 100000,
                            "max": 150000,
                            "median": 125000,
                            "currency": "USD",
                            "period": "annual"
                        },
                        "market_factors": {
                            "location_factor": 1.5,
                            "experience_factor": 1.2,
                            "skills_factor": 1.03
                        },
                        "reference_date": "2025-04-03"
                    },
                    "error": None
                }
                mock_request.return_value = mock_response
    
                # Call the method
                result = await external_tool_service.get_salary_benchmark(
                    job_title="Software Engineer",
                    location="San Francisco",
                    experience_years=5,
                    skills=["Python", "JavaScript"]
                )
    
                # Verify _make_api_request was called with the right parameters
                mock_request.assert_called_once()
                
                # Extract the actual call parameters
                call_args, call_kwargs = mock_request.call_args
                
                # Verify the URL and parameters
                expected_url = "https://api.salarydata.test.com/v1/salary/benchmark"
                assert expected_url in str(call_args), f"Expected URL {expected_url} not found in {call_args}"
                
                # Verify other parameters
                assert "title" in str(call_kwargs), "Job title parameter not found"
                assert "Software Engineer" in str(call_kwargs), "Software Engineer not found in parameters"
                assert "San Francisco" in str(call_kwargs), "Location not found in parameters"
                assert "Python" in str(call_kwargs), "Skills not found in parameters"
    
                # Verify the result
                assert result == mock_response
    
        finally:
            # Restore original values
            external_tool_service.salary_api_key = original_api_key
            external_tool_service.salary_api_base_url = original_api_url
    
    @pytest.mark.asyncio
    async def test_get_job_market_insights(self, external_tool_service):
        """Test getting job market insights."""
        # Set API key and URL for testing
        original_api_key = external_tool_service.market_api_key
        original_api_url = external_tool_service.market_api_base_url
    
        try:
            # Set values to ensure API request is made (not example.com)
            external_tool_service.market_api_key = "test_key"
            external_tool_service.market_api_base_url = "https://api.marketdata.test.com/v1"
    
            # Mock the _make_api_request method
            with patch.object(external_tool_service, '_make_api_request') as mock_request:
                # Configure the mock to return a simulated response matching actual implementation
                mock_response: APIResponse = {
                    "success": True,
                    "data": {
                        "job_title": "Data Scientist",
                        "market_data": {
                            "demand_growth_rate": 0.08,
                            "demand_trend": [
                                {"date": "2024-10-05", "demand_index": 99.8},
                                {"date": "2024-11-04", "demand_index": 101.9}
                            ],
                            "job_postings_last_period": 15000,
                            "average_time_to_fill": 45
                        },
                        "skill_insights": [
                            {"skill": "Python", "demand_growth": 0.15, "popularity_rank": 5}
                        ]
                    },
                    "error": None
                }
                mock_request.return_value = mock_response
    
                # Call the method
                result = await external_tool_service.get_job_market_insights(
                    job_title="Data Scientist",
                    skills=["Python", "Machine Learning"],
                    location="New York",
                    time_period="3months"
                )
    
                # Verify _make_api_request was called with the right parameters
                mock_request.assert_called_once()
                
                # Extract the actual call parameters
                call_args, call_kwargs = mock_request.call_args
                
                # Verify the URL and parameters
                expected_url = "https://api.marketdata.test.com/v1/job/market-insights"
                assert expected_url in str(call_args), f"Expected URL {expected_url} not found in {call_args}"
                
                # Verify other parameters
                assert "title" in str(call_kwargs), "Job title parameter not found"
                assert "Data Scientist" in str(call_kwargs), "Data Scientist not found in parameters"
                assert "New York" in str(call_kwargs), "Location not found in parameters"
                assert "Python" in str(call_kwargs), "Skills not found in parameters"
    
                # Verify the result
                assert result == mock_response
    
        finally:
            # Restore original values
            external_tool_service.market_api_key = original_api_key
            external_tool_service.market_api_base_url = original_api_url
    
    @pytest.mark.asyncio
    async def test_get_skill_demand_trends(self, external_tool_service):
        """Test getting skill demand trends."""
        # Set API key and URL for testing
        original_api_key = external_tool_service.market_api_key
        original_api_url = external_tool_service.market_api_base_url
    
        try:
            # Set values to ensure API request is made (not example.com)
            external_tool_service.market_api_key = "test_key"
            external_tool_service.market_api_base_url = "https://api.marketdata.test.com/v1"
    
            # Mock the _make_api_request method
            with patch.object(external_tool_service, '_make_api_request') as mock_request:
                # Configure the mock to return a simulated response matching implementation
                mock_response: APIResponse = {
                    "success": True,
                    "data": {
                        "skill_trends": [
                            {
                                "skill": "Python",
                                "demand_growth_rate": 0.1,
                                "demand_trend": [
                                    {"date": "2025-01-09", "demand_index": 99.7}
                                ],
                                "popularity_rank": 5
                            }
                        ],
                        "industry_insights": {
                            "fastest_growing_industries": ["Healthcare Technology", "Fintech"]
                        }
                    },
                    "error": None
                }
                mock_request.return_value = mock_response
    
                # Call the method
                result = await external_tool_service.get_skill_demand_trends(
                    skills=["Python", "React", "AWS"],
                    location="Remote",
                    time_period="12months"
                )
    
                # Verify _make_api_request was called with the right parameters
                mock_request.assert_called_once()
                
                # Extract the actual call parameters
                call_args, call_kwargs = mock_request.call_args
                
                # Verify the URL and parameters
                expected_url = "https://api.marketdata.test.com/v1/skills/trends"
                assert expected_url in str(call_args), f"Expected URL {expected_url} not found in {call_args}"
                
                # Verify other parameters
                assert "skills" in str(call_kwargs), "Skills parameter not found"
                assert "Python" in str(call_kwargs), "Python not found in parameters"
                assert "Remote" in str(call_kwargs), "Location not found in parameters"
    
                # Verify the result
                assert result == mock_response
    
        finally:
            # Restore original values
            external_tool_service.market_api_key = original_api_key
            external_tool_service.market_api_base_url = original_api_url
    
    def test_simulate_salary_data(self, external_tool_service):
        """Test simulating salary data."""
        # Call the method
        result = external_tool_service._simulate_salary_data(
            job_title="Senior Python Developer",
            location="San Francisco",
            experience_years=7,
            skills=["Python", "Django", "AWS"]
        )
        
        # Verify the structure of the result matches implementation
        assert "job_title" in result
        assert result["job_title"] == "Senior Python Developer"
        assert "salary_data" in result  # Changed from salary_range to salary_data structure
        assert "min" in result["salary_data"]
        assert "max" in result["salary_data"]
        assert "median" in result["salary_data"]
        assert "market_factors" in result
        assert "reference_date" in result
    
    def test_simulate_market_data(self, external_tool_service):
        """Test simulating market data."""
        # Call the method
        result = external_tool_service._simulate_market_data(
            job_title="Machine Learning Engineer",
            skills=["Python", "TensorFlow", "PyTorch"],
            location="New York",
            time_period="6months"
        )
        
        # Verify the structure of the result matches implementation
        assert "job_title" in result
        assert result["job_title"] == "Machine Learning Engineer"
        assert "market_data" in result  # Changed from direct fields to nested structure
        assert "demand_growth_rate" in result["market_data"]
        assert "demand_trend" in result["market_data"]
        assert "skill_insights" in result
        assert "regional_insights" in result
    
    def test_simulate_skill_trends(self, external_tool_service):
        """Test simulating skill trend data."""
        # Call the method
        result = external_tool_service._simulate_skill_trends(
            skills=["JavaScript", "TypeScript", "React"],
            location=None,
            time_period="3months"
        )
        
        # Verify the structure of the result matches implementation
        assert "skill_trends" in result  # Changed from skills to skill_trends
        assert len(result["skill_trends"]) == 3
        assert "industry_insights" in result
        assert "reference_period" in result
        assert "metadata" in result 