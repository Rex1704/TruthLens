from sentence_transformers import SentenceTransformer
from config.config import EMBEDDING_MODEL

def get_embedding_model():
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
def embedding_fn(model, text: list[str]) -> list:
    try:
        embeddings = model.encode(text, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}")