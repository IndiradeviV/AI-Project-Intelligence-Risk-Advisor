import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if api_key:
    print("SUCCESS: Groq API key loaded.")
else:
    print("ERROR: Groq API key not found. Check your .env file.")
