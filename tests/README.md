# RecruitX Test Framework

This directory contains tests for the RecruitX application. The tests are organized into different categories to ensure comprehensive coverage of the codebase.

## Test Structure

```
tests/
├── conftest.py            # Shared test fixtures and configuration
├── unit/                  # Unit tests for individual components
│   ├── agents/            # Tests for AI agents
│   ├── services/          # Tests for service layer
│   └── api/               # Tests for API endpoints
├── integration/           # Tests that verify multiple components work together
└── e2e/                   # End-to-end tests (when applicable)
```

## Running Tests

Run all tests:

```bash
pytest
```

Run a specific test category:

```bash
# Run all unit tests
pytest tests/unit/

# Run only agent tests
pytest tests/unit/agents/

# Run a specific test file
pytest tests/unit/agents/test_jd_analysis_agent.py
```

Run tests with coverage report:

```bash
# Install pytest-cov if not already installed
pip install pytest-cov

# Run tests with coverage
pytest --cov=recruitx_app tests/
```

## Test Categories

### Unit Tests

Unit tests verify that individual components work correctly in isolation. They typically mock dependencies to focus on testing a single component's logic.

### Integration Tests

Integration tests verify that multiple components work correctly together. These tests typically involve fewer mocks and test real interactions between components.

### End-to-End Tests

End-to-end tests verify that complete workflows function correctly from start to finish. These tests typically involve minimal mocking and test the application as a whole.

## Writing New Tests

When writing new tests, follow these guidelines:

1. **Test Structure**: Follow the existing structure and naming conventions.
2. **Test Function Names**: Use descriptive names that explain what is being tested (e.g., `test_analyze_job_description_with_rag`).
3. **AAA Pattern**: Structure tests using the Arrange-Act-Assert pattern:
   - Arrange: Set up test data and dependencies
   - Act: Call the function or method being tested
   - Assert: Verify the expected outcome
4. **Mocking**: Use the `unittest.mock` library to mock dependencies. Prefer `MagicMock` and `AsyncMock` for most scenarios.
5. **Fixtures**: Use pytest fixtures for shared test setup and teardown.
6. **Markers**: Use appropriate pytest markers for categorizing tests.

## Testing Async Functions

Since many RecruitX components use async/await, we use pytest-asyncio to test async functions. Mark async tests with `@pytest.mark.asyncio` and make the test function `async`.

Example:

```python
@pytest.mark.asyncio
async def test_async_function():
    # Test code here
    result = await my_async_function()
    assert result == expected_value
```

## Mocking Guidelines

When mocking the Gemini API or other external dependencies:

1. Use `patch` to replace the external dependency with a mock
2. Configure the mock to return appropriate test data
3. Verify that the mock was called with expected arguments

Example:

```python
@patch('recruitx_app.utils.retry_utils.call_gemini_with_backoff')
async def test_with_mock_gemini(mock_gemini):
    # Configure mock
    mock_async = AsyncMock()
    mock_response = MagicMock()
    # ... configure mock_response ...
    mock_async.return_value = mock_response
    mock_gemini.side_effect = mock_async
    
    # Test code
    result = await my_function_that_calls_gemini()
    
    # Verify
    mock_gemini.assert_called_once()
    assert result == expected_value
``` 