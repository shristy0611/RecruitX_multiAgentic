import pytest
import os
import sys
import asyncio
import json
import base64
from unittest.mock import patch, MagicMock, AsyncMock, Mock, PropertyMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from recruitx_app.agents.multimodal_agent import MultimodalAgent

class TestMultimodalAgent:
    """Test class for the MultimodalAgent."""
    
    @pytest.fixture
    def multimodal_agent(self):
        """Create a MultimodalAgent instance for testing."""
        with patch('recruitx_app.agents.multimodal_agent.settings'):
            return MultimodalAgent()
        
    def test_init(self, multimodal_agent):
        """Test initialization of MultimodalAgent."""
        assert multimodal_agent is not None
        assert multimodal_agent.model_name is not None
        assert multimodal_agent.safety_settings is not None
        assert hasattr(multimodal_agent, '_api_key_index')
        
    def test_get_gemini_model_success(self, multimodal_agent):
        """Test successful creation of a Gemini multimodal model."""
        with patch('google.generativeai.GenerativeModel') as mock_generative_model:
            # Configure the mock
            mock_model = MagicMock()
            mock_generative_model.return_value = mock_model
            
            # Call the method
            result = multimodal_agent._get_gemini_model()
            
            # Verify the result
            assert result == mock_model
            mock_generative_model.assert_called_once()
            
    def test_get_gemini_model_error_with_recovery(self, multimodal_agent):
        """Test error recovery when creating a Gemini model."""
        with patch('google.generativeai.GenerativeModel') as mock_generative_model, \
             patch('google.generativeai.configure') as mock_configure:
            # Configure the mock
            mock_model = MagicMock()
            
            # Make the first call fail, but the second succeed
            mock_generative_model.side_effect = [Exception("API Error"), mock_model]
            
            # Call the method
            result = multimodal_agent._get_gemini_model()
            
            # Verify the result
            assert result == mock_model
            assert mock_generative_model.call_count == 2
            mock_configure.assert_called_once()
            
    def test_get_gemini_model_error_without_recovery(self, multimodal_agent):
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
                multimodal_agent._get_gemini_model()
            
            # Verify that the second exception was raised
            assert "Second API Error" in str(exc_info.value)
            assert mock_generative_model.call_count == 2
            mock_configure.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_document_with_images_success(self, multimodal_agent):
        """Test successful document analysis with images."""
        # Mock the get_model method
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            # Create a mock response
            mock_response = MagicMock()
            mock_response.text = "Analysis of resume: Experienced software engineer with 5 years experience."
            mock_call_gemini.return_value = asyncio.Future()
            mock_call_gemini.return_value.set_result(mock_response)
            
            # Sample data
            text_content = "Sample resume text"
            image_data = b"fake image data"
            
            # Call the method
            result = await multimodal_agent.analyze_document_with_images(
                text_content=text_content,
                image_data_list=[image_data]
            )
            
            # Verify the result
            assert "analysis" in result
            assert result["analysis"] == "Analysis of resume: Experienced software engineer with 5 years experience."
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_document_with_images_no_images(self, multimodal_agent):
        """Test document analysis without images."""
        # Mock the get_model method
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            # Create a mock response
            mock_response = MagicMock()
            mock_response.text = "Analysis of resume text only."
            mock_call_gemini.return_value = asyncio.Future()
            mock_call_gemini.return_value.set_result(mock_response)
            
            # Sample data
            text_content = "Sample resume text"
            
            # Call the method without images
            result = await multimodal_agent.analyze_document_with_images(
                text_content=text_content
            )
            
            # Verify the result
            assert "analysis" in result
            assert result["analysis"] == "Analysis of resume text only."
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_document_with_parts_response(self, multimodal_agent):
        """Test document analysis with a response that has parts instead of text."""
        # Mock the get_model method
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            # Create a mock response with parts
            mock_response = MagicMock()
            mock_response.text = None  # No text attribute
            
            # Create parts with text
            part1 = MagicMock()
            part1.text = "Part 1 analysis"
            part2 = MagicMock()
            part2.text = "Part 2 analysis"
            
            # Set the parts property
            type(mock_response).parts = PropertyMock(return_value=[part1, part2])
            mock_call_gemini.return_value = asyncio.Future()
            mock_call_gemini.return_value.set_result(mock_response)
            
            # Sample data
            text_content = "Sample resume text"
            
            # Call the method
            result = await multimodal_agent.analyze_document_with_images(
                text_content=text_content
            )
            
            # Verify the result
            assert "analysis" in result
            assert result["analysis"] == "Part 1 analysis Part 2 analysis"
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_document_with_empty_response(self, multimodal_agent):
        """Test document analysis with an empty response."""
        # Mock the get_model method
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            # Create a mock response with no text or parts
            mock_response = MagicMock()
            mock_response.text = None  # No text attribute
            type(mock_response).parts = PropertyMock(return_value=[])  # Empty parts
            
            mock_call_gemini.return_value = asyncio.Future()
            mock_call_gemini.return_value.set_result(mock_response)
            
            # Sample data
            text_content = "Sample resume text"
            
            # Call the method
            result = await multimodal_agent.analyze_document_with_images(
                text_content=text_content
            )
            
            # Verify the result
            assert "analysis" in result
            assert result["analysis"] == ""  # Empty analysis
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_document_with_exception(self, multimodal_agent):
        """Test error handling in document analysis."""
        # Mock the get_model method to raise an exception
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model:
            mock_get_model.side_effect = Exception("Test exception")
            
            # Call the method
            result = await multimodal_agent.analyze_document_with_images(
                text_content="Sample resume"
            )
            
            # Verify the error is captured in the result
            assert "error" in result
            assert "Test exception" in result["error"]
            
            # Verify the mock was called
            mock_get_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_document_with_gemini_api_exception(self, multimodal_agent):
        """Test error handling when call_gemini_with_backoff raises an exception."""
        # Mock the get_model method and call_gemini_with_backoff
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            # Make the API call fail
            api_error = Exception("Gemini API error")
            mock_call_gemini.side_effect = api_error
            
            # Sample data
            text_content = "Sample resume text"
            image_data = b"fake image data"
            
            # Call the method
            result = await multimodal_agent.analyze_document_with_images(
                text_content=text_content,
                image_data_list=[image_data]
            )
            
            # Verify the error is captured in the result
            assert "error" in result
            assert "Gemini API error" in result["error"]
            
            # Verify the mocks were called
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_text_from_document_image_success(self, multimodal_agent):
        """Test successful text extraction from document image."""
        # Mock the get_model method
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            # Create a mock response
            mock_response = MagicMock()
            mock_response.text = "Extracted text from image: John Doe, Software Engineer"
            mock_call_gemini.return_value = asyncio.Future()
            mock_call_gemini.return_value.set_result(mock_response)
            
            # Sample image data
            image_data = b"fake image data"
            
            # Call the method
            result = await multimodal_agent.extract_text_from_document_image(image_data)
            
            # Verify the result
            assert result == "Extracted text from image: John Doe, Software Engineer"
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_text_from_document_image_parts_response(self, multimodal_agent):
        """Test text extraction with a response that has parts instead of text."""
        # Mock the get_model method
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            # Create a mock response with parts
            mock_response = MagicMock()
            mock_response.text = None  # No text attribute
            
            # Create parts with text
            part1 = MagicMock()
            part1.text = "John Doe,"
            part2 = MagicMock()
            part2.text = "Software Engineer"
            
            # Set the parts property
            type(mock_response).parts = PropertyMock(return_value=[part1, part2])
            mock_call_gemini.return_value = asyncio.Future()
            mock_call_gemini.return_value.set_result(mock_response)
            
            # Sample image data
            image_data = b"fake image data"
            
            # Call the method
            result = await multimodal_agent.extract_text_from_document_image(image_data)
            
            # Verify the result
            assert result == "John Doe, Software Engineer"
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_text_from_document_image_empty_response(self, multimodal_agent):
        """Test extracting text from document image when response is empty."""
        # Setup
        mock_response = AsyncMock()
        mock_response.text = ""
        
        # Mock the get_model method and call_gemini_with_backoff
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            mock_call_gemini.return_value = mock_response
            
            # Call the method
            result = await multimodal_agent.extract_text_from_document_image(
                image_data=b"fake image data"
            )
            
            # Verify result
            assert result == {"text": ""}
            
            # Verify the mocks were called
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_text_from_document_image_with_exception(self, multimodal_agent):
        """Test error handling in extract_text_from_document_image when an exception occurs."""
        # Mock the get_model method to raise an exception
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model:
            mock_get_model.side_effect = Exception("Test exception")
            
            # Call the method
            result = await multimodal_agent.extract_text_from_document_image(
                image_data=b"fake image data"
            )
            
            # Verify the error is captured in the result
            assert "error" in result
            assert "Test exception" in result["error"]
            
            # Verify the mock was called
            mock_get_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_text_from_document_image_gemini_api_exception(self, multimodal_agent):
        """Test error handling when Gemini API call fails."""
        # Mock the get_model method and call_gemini_with_backoff
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            # Make the API call fail
            api_error = Exception("Gemini API error during extraction")
            mock_call_gemini.side_effect = api_error
            
            # Call the method
            result = await multimodal_agent.extract_text_from_document_image(
                image_data=b"fake image data"
            )
            
            # Verify the error is captured in the result
            assert "error" in result
            assert "Gemini API error during extraction" in result["error"]
            
            # Verify the mocks were called
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_image_description_with_exception(self, multimodal_agent):
        """Test error handling in get_image_description when an exception occurs in model creation."""
        # Mock the get_model method to raise an exception
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model:
            mock_get_model.side_effect = Exception("Test model creation exception")
            
            # Call the method
            result = await multimodal_agent.get_image_description(
                image_data=b"fake image data"
            )
            
            # Verify the error is captured in the result
            assert "error" in result
            assert "Test model creation exception" in result["error"]
            
            # Verify the mock was called
            mock_get_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_image_description_with_api_exception(self, multimodal_agent):
        """Test error handling in get_image_description when the Gemini API call fails."""
        # Mock the get_model method and call_gemini_with_backoff
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            # Make the API call fail
            api_error = Exception("Gemini API error during description")
            mock_call_gemini.side_effect = api_error
            
            # Call the method
            result = await multimodal_agent.get_image_description(
                image_data=b"fake image data"
            )
            
            # Verify the error is captured in the result
            assert "error" in result
            assert "Gemini API error during description" in result["error"]
            
            # Verify the mocks were called
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_image_description_empty_response(self, multimodal_agent):
        """Test handling empty responses in get_image_description."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.text = ""
        
        # Mock the get_model method and call_gemini_with_backoff
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            mock_call_gemini.return_value = mock_response
            
            # Call the method
            result = await multimodal_agent.get_image_description(
                image_data=b"fake image data"
            )
            
            # Verify the result contains empty description
            assert result == {"description": ""}
            
            # Verify the mocks were called
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_image_description_success(self, multimodal_agent):
        """Test successful image description generation."""
        # Mock the get_model method
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            # Create a mock response
            mock_response = MagicMock()
            mock_response.text = "The image shows a professional headshot of a person in business attire against a neutral background."
            mock_call_gemini.return_value = asyncio.Future()
            mock_call_gemini.return_value.set_result(mock_response)
            
            # Sample image data
            image_data = b"fake image data"
            
            # Call the method
            result = await multimodal_agent.get_image_description(image_data)
            
            # Verify the result
            assert "description" in result
            assert "professional headshot" in result["description"]
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_get_image_description_with_parts_response(self, multimodal_agent):
        """Test image description with a response that has parts instead of text."""
        # Mock the get_model method
        with patch.object(multimodal_agent, '_get_gemini_model') as mock_get_model, \
             patch('recruitx_app.agents.multimodal_agent.call_gemini_with_backoff') as mock_call_gemini:
            # Configure the mocks
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            # Create a mock response with parts
            mock_response = MagicMock()
            mock_response.text = None  # No text attribute
            
            # Create parts with text
            part1 = MagicMock()
            part1.text = "The image depicts"
            part2 = MagicMock()
            part2.text = "a data visualization chart"
            
            # Set the parts property
            type(mock_response).parts = PropertyMock(return_value=[part1, part2])
            mock_call_gemini.return_value = asyncio.Future()
            mock_call_gemini.return_value.set_result(mock_response)
            
            # Sample image data
            image_data = b"fake image data"
            
            # Call the method
            result = await multimodal_agent.get_image_description(image_data)
            
            # Verify the result
            assert "description" in result
            assert result["description"] == "The image depicts a data visualization chart"
            mock_get_model.assert_called_once()
            mock_call_gemini.assert_called_once() 