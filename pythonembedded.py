
from langchain_community.embeddings import HuggingFaceEmbeddings

# embeddings.py

def load_embeddings():
    """
    Load HuggingFace sentence embedding model.
    Only this. Do NOT load PDF, FAISS, or split documents here.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )