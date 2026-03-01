from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load PDF
loader = PyPDFLoader("Clinic_Base.pdf")
documents = loader.load()

# 2. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Create vector store
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")

print("Index created successfully ✅")