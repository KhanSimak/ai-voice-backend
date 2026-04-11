from rag import create_vectorstore

vs = create_vectorstore()
vs.save_local("faiss_index")

print("✅ Done")