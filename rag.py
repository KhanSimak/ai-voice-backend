from langchain.chains import RetrievalQA

from pythonembedded import load_embeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq



from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# -----------------------------
# Load embeddings
# -----------------------------
load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)
# -----------------------------
# Load FAISS index (prebuilt)
# -----------------------------
vectorstore = FAISS.load_local(
    "faiss_index",  # folder created by create_index.py
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # top 3 doc

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

def ask_rag(question: str):
    result = qa_chain.invoke({"query": question})
    return result["result"]

# -----------------------------
# LLM setup (OpenAI)
# -----------------------------


# -----------------------------
# RAG function
# -----------------------------
def ask_rag(question: str):
    # Step 1: Retrieve documents
    docs = retriever.get_relevant_documents(question)

    # Step 2: Debug - check retrieved docs
    print("===== Retrieved Docs =====")
    if not docs:
        print("No relevant documents found.")
    for i, doc in enumerate(docs):
        print(f"Doc {i+1}:", doc.page_content[:300])  # first 300 chars
        print("------------------------")

    # Step 3: If no docs retrieved
    if not docs:
        return "I do not have that information."

    # Step 4: Combine retrieved docs into context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 5: Strict prompt for LLM
    prompt = f"""
You are a strict assistant.
Answer ONLY using the context below.
If answer is not in the context, say "I do not have that information."

Context:
{context}

Question:
{question}
"""

    # Step 6: Call LLM
    response = llm.invoke(prompt)

    # Step 7: Return answer
    return response.content