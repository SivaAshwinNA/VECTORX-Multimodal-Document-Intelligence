import os
from dotenv import load_dotenv
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")
