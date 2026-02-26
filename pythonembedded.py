from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1️⃣ Load your clinic PDF
loader = PyPDFLoader("GreenValley_Clinic_Knowledge_Base.pdf")
documents = loader.load()

# 2️⃣ Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

chunks = text_splitter.split_documents(documents)

# 3️⃣ Load HuggingFace embedding model (FREE local model)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4️⃣ Create vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="clinic_vector_db"
)

vectorstore.persist()

print("✅ PDF embedded successfully!")