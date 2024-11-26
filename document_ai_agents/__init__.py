import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

if (Path(__file__).parents[1] / ".env").is_file():
    load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")


genai.configure(api_key=os.environ["GOOGLE_API_KEY"], transport="rest")
