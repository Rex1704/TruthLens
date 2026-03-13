import faiss
import numpy as np
from pypdf import PdfReader
from models.embeddings import embedding_fn
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
    
def build_faiss_index(embeddings) -> faiss.IndexFlatL2:
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to build FAISS index: {e}")

def retrieve_relevant_chunks(
    query_embedding,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    top_k: int = TOP_K_RESULTS,
) -> list[str]:
    try:
        query_vec = np.array([query_embedding]).astype(np.float32)
        distances, indices = index.search(query_vec, top_k)
        results = [chunks[i] for i in indices[0] if i < len(chunks)]
        return results
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve chunks: {e}")

def process_uploaded_pdf(uploaded_file, embedding_model) -> dict:
    try:
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        embeddings = embedding_fn(embedding_model, chunks)
        index = build_faiss_index(np.array(embeddings))
        return {"chunks": chunks, "index": index}
    except Exception as e:
        raise RuntimeError(f"PDF processing pipeline failed: {e}")

def get_rag_context(query: str, rag_store: dict, embedding_model) -> str:
    try:
        query_emb = embedding_fn(embedding_model, [query])[0]
        relevant_chunks = retrieve_relevant_chunks(
            query_emb, rag_store["index"], rag_store["chunks"]
        )
        context = "\n---\n".join(
            [f"[Document {i+1}]:\n{chunk}" for i, chunk in enumerate(relevant_chunks)]
        )
        return context
    except Exception as e:
        return f"RAG retrieval error: {e}"
    