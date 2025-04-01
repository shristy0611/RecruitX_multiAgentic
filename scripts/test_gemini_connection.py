import google.generativeai as genai
import sys
import os

# Adjust the Python path to include the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    # Attempt to import settings after adjusting the path
    from recruitx_app.core.config import settings
except ImportError as e:
    print(f"Error importing settings: {e}")
    print("Ensure the script is run from the project root or the PYTHONPATH is set correctly.")
    sys.exit(1)


def test_connection():
    """Tests the connection to the Gemini API using the first key."""
    print("--- Starting Gemini Connection Test ---")

    try:
        # Configure the Gemini client
        api_key = settings.GEMINI_API_KEY_1  # Use the first key for this test
        
        if not api_key:
            print("Error: GEMINI_API_KEY_1 is not set in the environment/config.")
            return

        print(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")  # Print partial key for verification
        genai.configure(api_key=api_key)
        
        # First, list available models to see what we can use
        print("\nListing available models...")
        try:
            for model in genai.list_models():
                print(f"- {model.name}")
                if hasattr(model, 'supported_generation_methods'):
                    print(f"  Supported methods: {model.supported_generation_methods}")
                print()
        except Exception as e:
            print(f"Error listing models: {e}")
            
        # Use the configured model from settings
        model_name = settings.GEMINI_PRO_MODEL
        
        print(f"\nAttempting to use model: {model_name}")
        model = genai.GenerativeModel(model_name)

        # Make a recruitment-focused test call
        prompt = """
        Analyze this job description excerpt and extract the key requirements:
        
        Job Title: Senior Python Developer
        
        We are looking for an experienced Python developer with at least 5 years of experience in web development.
        The ideal candidate should have expertise in Django or Flask, RESTful API design, and database management
        (PostgreSQL preferred). Experience with cloud services (AWS/GCP) and containerization (Docker) is a plus.
        Must be comfortable working in an Agile environment.
        """
        
        print(f'Sending recruitment-focused prompt...')
        response = model.generate_content(prompt)

        # Print the response
        print("\nReceived response:")
        if hasattr(response, 'text'):
            print(response.text)
        elif hasattr(response, 'parts'):  # Handle potential different response structures
            print(" ".join(part.text for part in response.parts if hasattr(part, 'text')))
        else:
            print(f"Received unexpected response format: {response}")

        print("\n--- Gemini Connection Test Successful ---")

    except Exception as e:
        print(f"\n--- Gemini Connection Test Failed ---")
        print(f"An error occurred: {e}")
        # You might want to log the full traceback here for debugging
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_connection() 