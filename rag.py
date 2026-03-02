import os
import time
import logging
import requests
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ── Custom HuggingFace Embeddings (direct API, no silent failures) ─────────────

class HFInferenceEmbeddings(Embeddings):
    """
    Calls the HuggingFace Inference API directly via requests.
    Raises clear exceptions on failure instead of returning empty lists.
    Retries on 503 (model loading) with exponential backoff.
    """

    HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    MAX_RETRIES = 5
    BATCH_SIZE = 32   # stay well within HF free tier limits

    def __init__(self, api_token: str):
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """POST a batch of texts to the HF Inference API and return embeddings."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = requests.post(
                    self.HF_API_URL,
                    headers=self.headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=60,
                )
            except requests.exceptions.RequestException as e:
                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(f"HF API request failed after {self.MAX_RETRIES} attempts: {e}")
                wait = 2 ** attempt
                logger.warning(f"HF API network error (attempt {attempt}), retrying in {wait}s: {e}")
                time.sleep(wait)
                continue

            if response.status_code == 200:
                result = response.json()
                # HF returns either List[List[float]] or List[List[List[float]]] (mean pooling needed)
                if not result:
                    raise RuntimeError("HF API returned an empty response body.")
                # If shape is (n, seq_len, hidden) — take mean over seq_len
                arr = np.array(result)
                if arr.ndim == 3:
                    arr = arr.mean(axis=1)
                if arr.ndim != 2:
                    raise RuntimeError(f"Unexpected embedding shape from HF API: {arr.shape}")
                return arr.tolist()

            elif response.status_code == 503:
                # Model is loading — wait and retry
                wait = min(2 ** attempt, 30)
                logger.warning(f"HF model loading (503), retrying in {wait}s (attempt {attempt}/{self.MAX_RETRIES})...")
                time.sleep(wait)

            elif response.status_code == 401:
                raise RuntimeError("HF API returned 401 Unauthorized. Check your HUGGINGFACEHUB_API_TOKEN.")

            elif response.status_code == 429:
                wait = 10 * attempt
                logger.warning(f"HF API rate limit (429), waiting {wait}s...")
                time.sleep(wait)

            else:
                raise RuntimeError(
                    f"HF API error {response.status_code}: {response.text[:300]}"
                )

        raise RuntimeError(f"HF API failed after {self.MAX_RETRIES} retries.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents in batches."""
        all_embeddings = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            logger.info(f"Embedding batch {i // self.BATCH_SIZE + 1} ({len(batch)} chunks)...")
            embeddings = self._call_api(batch)
            if not embeddings:
                raise RuntimeError(f"Empty embeddings returned for batch starting at index {i}.")
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        result = self._call_api([text])
        if not result:
            raise RuntimeError("Empty embedding returned for query.")
        return result[0]


# ── Vectorstore ───────────────────────────────────────────────────────────────

def create_vectorstore():
    """
    Loads the PDF knowledge base, creates embeddings via HuggingFace
    Inference API, builds and returns a FAISS vectorstore.
    Never called at import time — only during app startup.
    """
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set.")

    # Load PDF
    pdf_path = os.path.join(os.path.dirname(__file__), "Clinic_Base.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    logger.info(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError("No pages loaded from PDF. Check the file is valid.")
    logger.info(f"Loaded {len(documents)} pages from PDF.")

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Text splitting produced no chunks.")
    logger.info(f"Created {len(chunks)} text chunks.")

    # Direct HF Inference API embeddings with retry + clear errors
    embeddings = HFInferenceEmbeddings(api_token=hf_token)

    # Smoke-test the embedding API before processing all chunks
    logger.info("Testing HuggingFace Inference API connection...")
    test_vec = embeddings.embed_query("test connection")
    if not test_vec or len(test_vec) == 0:
        raise RuntimeError("HF embedding smoke test returned empty vector. Check your token and API.")
    logger.info(f"HF API OK — embedding dimension: {len(test_vec)}")

    # Build FAISS vectorstore
    logger.info("Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS vectorstore ready.")
    return vectorstore


def get_llm():
    """
    Returns a ChatGroq instance. Called once at startup.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama3-8b-8192",
        temperature=0.2,
        max_tokens=1024,
    )
    return llm


def ask_question(vectorstore, llm, question: str) -> str:
    """
    Performs similarity search over the vectorstore, builds a context-aware
    prompt, sends it to the Groq LLM, and returns the answer string.
    """
    if not question or not question.strip():
        return "Please provide a valid question."

    # Retrieve top relevant chunks
    docs = vectorstore.similarity_search(question, k=4)

    if not docs:
        return "I could not find relevant information in the knowledge base to answer your question."

    # Build context from retrieved chunks
    context = "\n\n".join(
        f"[Source: Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content.strip()}"
        for doc in docs
    )

    system_prompt = (
        "You are a helpful assistant for GreenValley Clinic. "
        "Answer the user's question using ONLY the context provided below. "
        "If the answer is not in the context, say: "
        "'I don't have that information in the current knowledge base.' "
        "Be concise, accurate, and professional.\n\n"
        f"CONTEXT:\n{context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question.strip()),
    ]

    response = llm.invoke(messages)

    # Extract content safely
    if hasattr(response, "content"):
        return response.content.strip()
    return str(response).strip()