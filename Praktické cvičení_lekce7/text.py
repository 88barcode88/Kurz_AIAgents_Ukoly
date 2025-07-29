# run_demo.py (nebo v main.py)
from dotenv import load_dotenv
import os

load_dotenv()  # načte .env do os.environ
openai_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")
