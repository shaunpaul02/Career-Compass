import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

# Model Configuration
MODEL_NAME = "gemini-1.5-pro"
TEMPERATURE = 0.7
MAX_TOKENS = 2048

# Job Search Configuration
DEFAULT_LOCATION = "London, ON"
MAX_SEARCH_RESULTS = 5
COMPATIBILITY_THRESHOLD = 0.6  # 60% minimum match

# Logging
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
