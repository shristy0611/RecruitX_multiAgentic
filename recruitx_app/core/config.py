from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from typing import List

class Settings(BaseSettings):
    # Load .env file in the parent directory (project root)
    model_config = SettingsConfigDict(env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'), extra='ignore')

    # Gemini API Keys - Load them into a list
    GEMINI_API_KEY_1: str
    GEMINI_API_KEY_2: str
    GEMINI_API_KEY_3: str
    GEMINI_API_KEY_4: str
    GEMINI_API_KEY_5: str
    GEMINI_API_KEY_6: str
    GEMINI_API_KEY_7: str
    GEMINI_API_KEY_8: str
    GEMINI_API_KEY_9: str
    GEMINI_API_KEY_10: str

    # Gemini Models - Using fully qualified names as required by the API
    GEMINI_PRO_MODEL: str = "models/gemini-2.0-flash-lite"  # Updated to use Gemini 2.0 Flash Lite for higher RPM
    GEMINI_PRO_VISION_MODEL: str = "models/gemini-2.0-flash-lite"  # Using the same model for vision as it supports multimodal inputs
    GEMINI_EMBEDDING_MODEL: str

    # Project Specific Settings
    PROJECT_NAME: str = "RecruitX"
    API_V1_STR: str = "/api/v1"

    # Simple round-robin logic for API keys
    _api_key_index: int = 0

    @property
    def gemini_api_keys(self) -> List[str]:
        return [
            self.GEMINI_API_KEY_1,
            self.GEMINI_API_KEY_2,
            self.GEMINI_API_KEY_3,
            self.GEMINI_API_KEY_4,
            self.GEMINI_API_KEY_5,
            self.GEMINI_API_KEY_6,
            self.GEMINI_API_KEY_7,
            self.GEMINI_API_KEY_8,
            self.GEMINI_API_KEY_9,
            self.GEMINI_API_KEY_10,
        ]

    def get_next_api_key(self) -> str:
        key = self.gemini_api_keys[self._api_key_index]
        self._api_key_index = (self._api_key_index + 1) % len(self.gemini_api_keys)
        return key

# Instantiate the settings
settings = Settings()

# You can optionally add validation or logging here to ensure keys are loaded
# For example:
# if not all(settings.gemini_api_keys):
#     print("Warning: Not all Gemini API keys are loaded from .env file!") 