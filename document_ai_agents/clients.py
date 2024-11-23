import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

if (Path(__file__).parents[1] / ".env").is_file():
    load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")


llm_google_client = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-002",
    temperature=0,
    max_retries=5,
)

llm_google_client_8b = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    temperature=0,
    max_retries=5,
)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
