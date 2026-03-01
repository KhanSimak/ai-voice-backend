# create_index.py
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pythonembedded import load_embeddings

# 1️⃣ Load PDF
loader = PyPDFLoader("Clinic_Base.pdf")  # put your PDF path here
documents = loader.load()

# 2️⃣ Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # each chunk ~500 chars
    chunk_overlap=50     # overlap 50 chars between chunks
)
docs = splitter.split_documents(documents)

# 3️⃣ Load embeddings model
embeddings = load_embeddings()

# 4️⃣ Create FAISS index
vectorstore = FAISS.from_documents(docs, embeddings)

# 5️⃣ Save FAISS index to disk
vectorstore.save_local("faiss_index")

print("✅ FAISS index created successfully!")