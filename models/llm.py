import os
import sys
from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY, GROQ_MODEL
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    try:
        # Initialize the Groq chat model with the API key
        groq_model = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")
    
def build_system_prompt(response_mode: str) -> str:
    base = (
        "You are an AI-powered fact-checking assistant. "
        "Only answer to the fact checking questions, you are not allowed to interact with user otherwise."
        "Your job is to analyze claims, headlines, or statements and determine if they are "
        "TRUE, FALSE, or MISLEADING based on the context and sources provided to you. "
        "Always base your verdict strictly on evidence. Never guess. "
        "If you are unsure, say so clearly. "
        "At the end, always provide a Credibility Score from 0 (completely false) to 100 (completely true)."
    )

    if response_mode == "concise":
        mode_instruction = (
            "\n\nResponse Format (CONCISE MODE — keep it brief):\n"
            "Verdict: TRUE / FALSE / MISLEADING\n"
            "Reason: One sentence explanation.\n"
            "Credibility Score: X/100"
        )
    else:
        mode_instruction = (
            "\n\nResponse Format (DETAILED MODE):\n"
            "Verdict: TRUE / FALSE / MISLEADING\n\n"
            "Analysis: Detailed breakdown — what is accurate, exaggerated, or false.\n\n"
            "Evidence: What each source says (cite sources clearly).\n\n"
            "Credibility Score: X/100 — with a brief explanation of the score."
        )

    return base + mode_instruction
