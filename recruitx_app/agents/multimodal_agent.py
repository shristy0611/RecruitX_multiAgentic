import google.generativeai as genai
import json
import base64
import logging
from typing import Dict, Any, Optional, List, Union
import asyncio # Added for sleep

from recruitx_app.core.config import settings
from recruitx_app.utils.retry_utils import call_gemini_with_backoff # Import the retry helper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalAgent:
    """
    A demonstration agent for leveraging Gemini 2.5 Pro's multimodal capabilities.
    This can be used for CV analysis with images, charts, and other visual elements.
    """
    
    def __init__(self):
        self.model_name = settings.GEMINI_PRO_VISION_MODEL
        self._api_key_index = 0
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
    
    def _get_gemini_model(self):
        """Get the Gemini multimodal model, rotating API keys if necessary."""
        try:
            model = genai.GenerativeModel(
                self.model_name,
                safety_settings=self.safety_settings,
                generation_config={
                    "temperature": 0.2,  # Slightly higher for creative analysis
                    "top_p": 0.95,
                    "top_k": 40,
                }
            )
            return model
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            genai.configure(api_key=settings.get_next_api_key())
            logger.info("Rotated to next API key")
            try:
                return genai.GenerativeModel(
                    self.model_name,
                    safety_settings=self.safety_settings,
                    generation_config={
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "top_k": 40,
                    }
                )
            except Exception as e2:
                logger.error(f"Second error initializing Gemini model: {e2}")
                raise e2
    
    async def analyze_document_with_images(
        self, 
        text_content: str, 
        image_data_list: List[bytes] = None
    ) -> Dict[str, Any]:
        """
        Analyze a document (like a resume) with images using Gemini 2.5 Pro's multimodal capabilities.
        
        Args:
            text_content: The textual content of the document
            image_data_list: List of image data bytes (optional)
            
        Returns:
            Dictionary containing the analysis results
        """
        try:
            model = self._get_gemini_model()
            
            # Create the prompt content list - starting with text
            content_parts = [text_content]
            
            # Add images if provided
            if image_data_list:
                for image_data in image_data_list:
                    content_parts.append({
                        "inlineData": {
                            "mimeType": "image/jpeg",  # Adjust if needed for other formats
                            "data": base64.b64encode(image_data).decode("utf-8")
                        }
                    })
            
            # Define the multimodal prompt
            prompt = """
            You are an advanced CV Analysis Agent. Analyze this resume including both the text and any visual 
            elements like charts, photographs, or diagrams. Extract key information about:
            
            1. Candidate's professional experience
            2. Skills and technologies
            3. Education and certifications
            4. Projects and achievements
            5. Any notable visual elements and what they convey
            
            Provide a comprehensive analysis of the candidate's profile.
            """
            
            # Use the retry helper for the API call
            response = await call_gemini_with_backoff(
                model.generate_content,
                [prompt, *content_parts] # Pass content as a list for multimodal
            )
            
            # Process and return the response
            if hasattr(response, 'text') and response.text:
                return {"analysis": response.text}
            elif hasattr(response, 'parts') and response.parts:
                # Handle potential lack of simple text attribute in multimodal response
                text_content = " ".join(part.text for part in response.parts if hasattr(part, 'text'))
                return {"analysis": text_content}
            else:
                logger.warning("No text content found in multimodal analysis response.")
                return {"analysis": ""}
                
        except Exception as e:
            logger.error(f"Error in multimodal analysis: {e}")
            return {"error": str(e)}
    
    async def extract_text_from_document_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Extract text from an image of a document using Gemini 2.5 Pro's OCR capabilities.
        
        Args:
            image_data: The binary image data
            
        Returns:
            Dictionary containing the extracted text or error
        """
        try:
            model = self._get_gemini_model()
            
            # Encode image data
            image_part = {
                "inlineData": {
                    "mimeType": "image/jpeg",  # Adjust if needed
                    "data": base64.b64encode(image_data).decode("utf-8")
                }
            }
            
            # Define the OCR prompt
            prompt = """
            Extract all text from this document image. 
            Maintain formatting where possible, including:
            - Paragraph breaks
            - Bullet points
            - Section headers
            
            Return only the extracted text, nothing else.
            """
            
            # Use the retry helper for the API call
            response = await call_gemini_with_backoff(
                model.generate_content,
                [prompt, image_part] # Pass content as a list
            )
            
            # Process and return the extracted text
            if hasattr(response, 'text') and response.text:
                return {"text": response.text}
            elif hasattr(response, 'parts') and response.parts:
                text_content = " ".join(part.text for part in response.parts if hasattr(part, 'text'))
                return {"text": text_content}
            else:
                logger.warning("No text content found in OCR response.")
                return {"text": ""}
                
        except Exception as e:
            logger.error(f"Error in OCR text extraction: {e}")
            return {"error": str(e)}
    
    async def get_image_description(self, image_data: bytes) -> Dict[str, Any]:
        """
        Get a detailed description of an image using Gemini 2.5 Pro.
        
        Args:
            image_data: The binary image data
            
        Returns:
            Dictionary containing the description or error
        """
        try:
            model = self._get_gemini_model()
            
            # Encode image data
            image_part = {
                "inlineData": {
                    "mimeType": "image/jpeg",  # Adjust if needed
                    "data": base64.b64encode(image_data).decode("utf-8")
                }
            }
            
            # Define the description prompt
            prompt = """
            Describe this image in detail, including:
            - Main subject or focus
            - Visual elements and composition
            - Context or setting
            - Any text visible in the image
            
            Provide a comprehensive description that would help someone understand the image without seeing it.
            """
            
            # Use the retry helper for the API call
            response = await call_gemini_with_backoff(
                model.generate_content,
                [prompt, image_part]  # Pass content as a list
            )
            
            # Process and return the description
            if hasattr(response, 'text') and response.text:
                return {"description": response.text}
            elif hasattr(response, 'parts') and response.parts:
                text_content = " ".join(part.text for part in response.parts if hasattr(part, 'text'))
                return {"description": text_content}
            else:
                logger.warning("No text content found in image description response.")
                return {"description": ""}
                
        except Exception as e:
            logger.error(f"Error in image description generation: {e}")
            return {"error": str(e)} 