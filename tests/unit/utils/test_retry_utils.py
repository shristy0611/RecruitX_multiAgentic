import os
import sys
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from recruitx_app.utils.retry_utils import call_gemini_with_backoff, MAX_RETRIES


@pytest.mark.asyncio
class TestRetryUtils:
    """Test class for retry utilities."""
    
    async def test_successful_async_call(self):
        """Test a successful async call with no retries needed."""
        # Create a mock async function that returns a value immediately
        mock_async_func = AsyncMock(return_value="success_result")
        
        # Call the function with backoff
        result = await call_gemini_with_backoff(mock_async_func, "arg1", arg2="value2")
        
        # Verify the result
        assert result == "success_result"
        
        # Verify the mock was called exactly once with the right arguments
        mock_async_func.assert_called_once_with("arg1", arg2="value2")
    
    async def test_successful_sync_call(self):
        """Test a successful synchronous call with no retries needed."""
        # Create a mock synchronous function
        mock_sync_func = MagicMock(return_value="sync_success")
        
        # Call the function with backoff
        result = await call_gemini_with_backoff(mock_sync_func, "sync_arg")
        
        # Verify the result
        assert result == "sync_success"
        
        # Verify the mock was called exactly once with the right arguments
        mock_sync_func.assert_called_once_with("sync_arg")
    
    @patch('asyncio.sleep')
    async def test_rate_limit_with_recovery(self, mock_sleep):
        """Test recovery after rate limit errors."""
        # Mock sleep to avoid actual waiting during tests
        mock_sleep.return_value = None
        
        # Create a mock function that fails with rate limit errors twice, then succeeds
        mock_func = AsyncMock()
        mock_func.side_effect = [
            ResourceExhausted("Rate limit exceeded"),
            ResourceExhausted("Rate limit exceeded"),
            "success_after_retry"
        ]
        
        # Call the function with backoff
        result = await call_gemini_with_backoff(mock_func)
        
        # Verify the result after retries
        assert result == "success_after_retry"
        
        # Verify the mock was called three times
        assert mock_func.call_count == 3
        
        # Verify sleep was called for each retry
        assert mock_sleep.call_count == 2
    
    @patch('asyncio.sleep')
    async def test_server_error_with_recovery(self, mock_sleep):
        """Test recovery after server errors."""
        # Mock sleep to avoid actual waiting during tests
        mock_sleep.return_value = None
        
        # Create a mock function that fails with server errors, then succeeds
        mock_func = AsyncMock()
        mock_func.side_effect = [
            InternalServerError("Server error"),
            ServiceUnavailable("Service unavailable"),
            "success_after_server_errors"
        ]
        
        # Call the function with backoff
        result = await call_gemini_with_backoff(mock_func)
        
        # Verify the result after retries
        assert result == "success_after_server_errors"
        
        # Verify the mock was called three times
        assert mock_func.call_count == 3
        
        # Verify sleep was called for each retry
        assert mock_sleep.call_count == 2
    
    @patch('asyncio.sleep')
    async def test_max_retries_rate_limit(self, mock_sleep):
        """Test reaching maximum retries with rate limit errors."""
        # Mock sleep to avoid actual waiting during tests
        mock_sleep.return_value = None
        
        # Create a mock function that always fails with rate limit errors
        mock_func = AsyncMock()
        mock_func.side_effect = ResourceExhausted("Rate limit exceeded")
        
        # Call the function with backoff and expect it to raise after max retries
        with pytest.raises(ResourceExhausted, match="Rate limit exceeded"):
            await call_gemini_with_backoff(mock_func)
        
        # Verify the mock was called MAX_RETRIES times
        assert mock_func.call_count == MAX_RETRIES
        
        # Verify sleep was called for each retry
        assert mock_sleep.call_count == MAX_RETRIES - 1
    
    @patch('asyncio.sleep')
    async def test_max_retries_server_error(self, mock_sleep):
        """Test reaching maximum retries with server errors."""
        # Mock sleep to avoid actual waiting during tests
        mock_sleep.return_value = None
        
        # Create a mock function that always fails with server errors
        mock_func = AsyncMock()
        mock_func.side_effect = InternalServerError("Server error")
        
        # Call the function with backoff and expect it to raise after max retries
        with pytest.raises(InternalServerError, match="Server error"):
            await call_gemini_with_backoff(mock_func)
        
        # Verify the mock was called MAX_RETRIES times
        assert mock_func.call_count == MAX_RETRIES
        
        # Verify sleep was called for each retry
        assert mock_sleep.call_count == MAX_RETRIES - 1
    
    async def test_unexpected_error_no_retry(self):
        """Test that unexpected errors are not retried."""
        # Create a mock function that raises an unexpected error
        mock_func = AsyncMock()
        mock_func.side_effect = ValueError("Unexpected error")
        
        # Call the function with backoff and expect it to raise immediately
        with pytest.raises(ValueError, match="Unexpected error"):
            await call_gemini_with_backoff(mock_func)
        
        # Verify the mock was called exactly once
        mock_func.assert_called_once()
    
    @patch('asyncio.sleep')
    @patch('asyncio.iscoroutinefunction')
    async def test_coroutine_check(self, mock_iscoroutinefunction, mock_sleep):
        """Test that the function checks if the target is a coroutine function."""
        # Set up iscoroutinefunction to return True
        mock_iscoroutinefunction.return_value = True
        
        # Mock sleep to avoid actual waiting during tests
        mock_sleep.return_value = None
        
        # Create a mock function
        mock_func = AsyncMock(return_value="async_result")
        
        # Call the function with backoff
        result = await call_gemini_with_backoff(mock_func)
        
        # Verify the result
        assert result == "async_result"
        
        # Verify iscoroutinefunction was called
        mock_iscoroutinefunction.assert_called_once_with(mock_func) 