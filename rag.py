import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage


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

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError("No pages loaded from PDF. Check the file is valid.")

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Text splitting produced no chunks.")

    # HuggingFace Inference API embeddings — no local model download
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Build FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
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