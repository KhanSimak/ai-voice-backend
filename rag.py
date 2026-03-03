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
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Text splitting produced no chunks.")
    logger.info(f"Created {len(chunks)} chunks.")
    
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_document"
)
    google_api_key=os.getenv("GOOGLE_API_KEY")


    logger.info("Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS vectorstore ready.")

    return vectorstore


def get_llm():
    """Returns a ChatGroq instance using llama-4-scout."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2,
        max_tokens=1024,
    )
    return llm


def ask_question(vectorstore, llm, question: str) -> str:
    """
    Retrieves relevant chunks from the vectorstore and sends them
    along with the question to the LLM. Returns the answer string.
    """
    if not question or not question.strip():
        return "Please provide a valid question."

    docs = vectorstore.similarity_search(question.strip(), k=4)

    if not docs:
        return "I could not find relevant information in the knowledge base to answer your question."

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content.strip()}"
        for doc in docs
    )

    system_prompt = (
        "You are a helpful assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "If the answer is not in the context, say: "
        "'I don't have that information in the knowledge base.' "
        "Be concise, accurate, and professional.\n\n"
        f"CONTEXT:\n{context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question.strip()),
    ]

    response = llm.invoke(messages)

    if hasattr(response, "content"):
        return response.content.strip()
    return str(response).strip()