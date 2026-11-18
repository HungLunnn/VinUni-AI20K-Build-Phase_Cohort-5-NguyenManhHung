import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("FPT_API_KEY")

if api_key:
    print("API Key loaded successfully!")
    print("Key prefix:", api_key[:6] + "******")
else:
    print("API Key not found!")