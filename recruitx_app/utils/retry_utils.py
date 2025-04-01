import time
import random
import logging
from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 60     # seconds

async def call_gemini_with_backoff(api_call_func, *args, **kwargs):
    """Calls a Gemini API function with exponential backoff for rate limiting and server errors."""
    retries = 0
    backoff_time = INITIAL_BACKOFF
    
    while retries < MAX_RETRIES:
        try:
            # If api_call_func is already a coroutine
            if asyncio.iscoroutinefunction(api_call_func):
                return await api_call_func(*args, **kwargs)
            # If it's a regular function, run it in an executor (or call directly if synchronous)
            else:
                 # Assuming direct call for synchronous functions like model.generate_content
                 # Adjust if using truly async library calls
                 # Note: genai library's generate_content might not be a true async coroutine 
                 # even if called from an async function. We'll call it directly.
                 return api_call_func(*args, **kwargs)
                 
        except ResourceExhausted as e:
            retries += 1
            if retries >= MAX_RETRIES:
                logger.error(f"API rate limit exceeded after {MAX_RETRIES} retries: {e}")
                raise e
            
            # Exponential backoff with jitter
            wait_time = backoff_time + random.uniform(0, 1)
            logger.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds... (Attempt {retries}/{MAX_RETRIES})")
            await asyncio.sleep(wait_time) 
            backoff_time = min(backoff_time * 2, MAX_BACKOFF)
            
        except (InternalServerError, ServiceUnavailable) as e:
            retries += 1
            if retries >= MAX_RETRIES:
                logger.error(f"Server error encountered after {MAX_RETRIES} retries: {e}")
                raise e
                
            # Exponential backoff for server errors
            wait_time = backoff_time + random.uniform(0, 1)
            logger.warning(f"Server error ({type(e).__name__}). Retrying in {wait_time:.2f} seconds... (Attempt {retries}/{MAX_RETRIES})")
            await asyncio.sleep(wait_time)
            backoff_time = min(backoff_time * 2, MAX_BACKOFF)

        except Exception as e:
            logger.error(f"An unexpected error occurred during API call: {e}")
            raise e
            
    # This point should ideally not be reached if MAX_RETRIES is > 0
    raise Exception(f"Failed after {MAX_RETRIES} retries.")

# We need asyncio for await asyncio.sleep
import asyncio 