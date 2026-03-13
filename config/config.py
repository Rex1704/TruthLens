import os
from dotenv import load_dotenv

load_dotenv()

def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except Exception:
        return os.getenv(key, "")

GROQ_API_KEY = _get_secret("GROQ_API_KEY")
TAVILY_API_KEY = _get_secret("TAVILY_API_KEY")
TAVILY_MAX_RESULTS = 5

GROQ_MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.3
MAX_TOKENS = 2048

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3