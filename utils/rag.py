import numpy as np
from pypdf import PdfReader
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS

def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        return full_text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")
    
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    try:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += chunk_size - overlap
        return chunks
    except Exception as e:
        raise RuntimeError(f"Failed to chunk text: {e}")
    

    