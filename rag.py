from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from memry import save_message, load_memory
from langchain_openai import ChatOpenAI


# OpenRouter config
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Embeddings
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# Load vector DB
vectorstore = Chroma(
    persist_directory="./clinic_vector_db",
    embedding_function=embedding,
    collection_name="clinic"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Conversation Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# System Prompt (Hallucination Control)
template = """
You are an AI assistant for Green Valley Clinic.

Rules:
- Answer ONLY from the provided clinic context.
- If answer is not in the context, say:
  "I’m sorry, I don’t have that information available."
- Do NOT guess or create information.
- Keep answers professional and short.

Context:
{context}

Question:
{question}

Answer:
"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# LLM
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    base_url=os.getenv("OPENAI_API_BASE")
)

# Conversational RAG Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
)


def ask_clinic_bot(phone_number: str, query: str):

    # Load previous memory from DB
    past_messages = load_memory(phone_number)

    chat_history = []
    for msg in past_messages:
        if msg.role == "user":
            chat_history.append(("human", msg.message))
        else:
            chat_history.append(("ai", msg.message))

    result = qa_chain.invoke({
        "question": query,
        "chat_history": chat_history
    })

    answer = result["answer"]

    # Save current interaction
    save_message(phone_number, "user", query)
    save_message(phone_number, "assistant", answer)

    return answer




