import os
import logging
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
logger = logging.getLogger(__name__)

from groq import Groq
import os


def create_vectorstore():
    """
    Loads PDF, splits into chunks, embeds with Google Generative AI,
    and returns a FAISS vectorstore. Called once at startup only.
    """
    google_api_key = str(os.environ.get("GOOGLE_API_KEY"))
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    

    pdf_path = os.path.join(os.path.dirname(__file__), "Clinic_Base.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    logger.info(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError("PDF loaded but contains no pages.")
    logger.info(f"Loaded {len(documents)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Text splitting produced no chunks.")
    logger.info(f"Created {len(chunks)} chunks.")
    
    embeddings = GoogleGenerativeAIEmbeddings(
      model="models/gemini-embedding-001",
      google_api_key=google_api_key
)

    google_api_key=os.getenv("GOOGLE_API_KEY")


    logger.info("Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS vectorstore ready.")

    return vectorstore


def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing")

    return ChatGroq(
        api_key=groq_api_key.strip(),
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2,
        max_tokens=1024,
    )



def ask_question(vectorstore, llm, question: str) -> str:
    """
    Retrieves relevant chunks from the vectorstore and sends them
    along with the question to the LLM.
    """

    if not question or not question.strip():
        return "Please say something."

    question = question.strip()

    logger.info(f"🧠 RAG QUERY: {question}")

    # 🔍 Step 1: Retrieve docs
    try:
        docs = vectorstore.similarity_search(question, k=4)
    except Exception as e:
        logger.error(f"❌ Vector search failed: {str(e)}")
        return "I'm having trouble accessing the knowledge base."

    logger.info(f"🔍 Retrieved {len(docs)} docs")

    if not docs:
        logger.warning("⚠️ No docs found in RAG")
        return "I don't have that information in the knowledge base."

    # 📄 Step 2: Log chunks (important for debugging)
    for i, doc in enumerate(docs):
        preview = doc.page_content.strip().replace("\n", " ")[:200]
        logger.info(f"📄 Doc {i+1}: {preview}")

    # 🧾 Step 3: Build context
    context = "\n\n".join(
        f"{doc.page_content.strip()}" for doc in docs
    )

    # 🧠 Step 4: Strong prompt (no hallucination)
    system_prompt = f"""
You are a clinic receptionist speaking on a phone call.

Rules:
- Speak naturally and politely
- Keep answers short (1–2 sentences)
- ONLY use the information from the context below
- If not found, say: "I don't have that information in the knowledge base"

CONTEXT:
{context}
"""

    # 💬 Step 5: LLM call
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ])
    except Exception as e:
        logger.error(f"❌ LLM failed: {str(e)}")
        return "I'm having trouble answering right now."

    # 🧾 Step 6: Extract response
    answer = response.content.strip() if hasattr(response, "content") else str(response).strip()

    logger.info(f"🤖 FINAL ANSWER: {answer}")

    # 🏷️ Optional debug tag (remove later if needed)
    return f"[RAG] {answer}"
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )