import os
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ----------------------------
# ENV SAFE LOADING
# ----------------------------
GOOGLE_API_KEY = str(os.getenv("GOOGLE_API_KEY", "")).strip()
GROQ_API_KEY = str(os.getenv("GROQ_API_KEY", "")).strip()

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing")


# ----------------------------
# LOAD VECTORSTORE (FAST)
# ----------------------------
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


# ----------------------------
# CREATE VECTORSTORE (RUN ONCE ONLY - OFFLINE)
# ----------------------------
def create_vectorstore():
    pdf_path = os.path.join(os.path.dirname(__file__), "Clinic_Base.pdf")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("📄 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    logger.info(f"📦 Chunks created: {len(chunks)}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    logger.info("⚡ Creating FAISS index...")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # SAVE LOCALLY
    vectorstore.save_local("faiss_index")

    logger.info("✅ FAISS saved locally")

    return vectorstore


# ----------------------------
# LLM
# ----------------------------
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2,
        max_tokens=1024,
    )


# ----------------------------
# CLEAN QUERY (IMPORTANT)
# ----------------------------
def clean_query(text: str) -> str:
    if not text:
        return ""

    text = text.replace("User:", "").replace("Agent:", "").strip()

    if len(text) < 2:
        return ""

    return text


# ----------------------------
# RAG FUNCTION (SAFE)
# ----------------------------
def ask_question(vectorstore, llm, question: str) -> str:
    try:
        question = clean_query(question)

        if not question:
            return "Please ask something valid."

        logger.info(f"🧠 RAG QUERY: {question}")

        docs = vectorstore.similarity_search(question, k=3)

        if not docs:
            return "I don't have that information."

        context = "\n\n".join([d.page_content for d in docs])

        response = llm.invoke([
            SystemMessage(content=f"""
You are a clinic receptionist.

Rules:
- Be short and natural
- Only use context
- If missing say: I don't have that information

Context:
{context}
"""),
            HumanMessage(content=question)
        ])

        return response.content.strip()

    except Exception as e:
        logger.error(f"RAG ERROR: {e}")
        return "System temporarily unavailable."