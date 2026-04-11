import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ----------------------------
# SAFE ENV
# ----------------------------
GOOGLE_API_KEY = str(os.getenv("GOOGLE_API_KEY", "")).strip()
GROQ_API_KEY = str(os.getenv("GROQ_API_KEY", "")).strip()

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY missing")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing")


# ----------------------------
# LOAD FAISS
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
# LLM
# ----------------------------
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=512,
    )


# ----------------------------
# INTENT FILTER (IMPORTANT)
# ----------------------------
def should_use_rag(text: str) -> bool:
    text = (text or "").lower()

    keywords = [
        "doctor", "appointment", "clinic",
        "specialist", "name", "list", "available"
    ]

    return any(k in text for k in keywords)


# ----------------------------
# CLEAN TEXT
# ----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("User:", "").replace("Agent:", "").strip()

    return text if len(text) > 1 else ""


# ----------------------------
# RAG CORE
# ----------------------------
def ask_question(vectorstore, llm, question: str) -> str:
    try:
        question = clean_text(question)

        if not question:
            return "Please say something valid."

        logger.info(f"🧠 QUERY: {question}")

        docs = vectorstore.similarity_search(question, k=3)

        if not docs:
            return "I don't have that information."

        context = "\n\n".join(d.page_content for d in docs)

        response = llm.invoke([
            SystemMessage(content=f"""
You are a clinic receptionist.

Rules:
- Be short (1-2 lines)
- Only use context
- If not found say: I don't have that information

Context:
{context}
"""),
            HumanMessage(content=question)
        ])

        return response.content.strip()

    except Exception as e:
        logger.error(f"RAG ERROR: {e}")
        return "System error. Try again later."