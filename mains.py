import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from rag import create_vectorstore, get_llm, ask_question

# ---------------- LOGGING ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- STARTUP ---------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    app.state.vectorstore = create_vectorstore()
    app.state.llm = get_llm()

    yield

    logger.info("Shutting down...")

# ---------------- APP ---------------- #
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CHAT ENDPOINT ---------------- #
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    print("RAW DATA:", data)

    user_message = ""

    # ✅ CASE 1: Normal function call
    if "query" in data and isinstance(data["query"], str):
        user_message = data["query"]

    # ✅ CASE 2: Sometimes Retell sends stringified JSON
    elif "query" in data and isinstance(data["query"], dict):
        user_message = data["query"].get("query", "")

    # ✅ CASE 3: fallback formats
    if not user_message:
        user_message = (
            data.get("transcript") or
            data.get("message") or
            data.get("text") or
            ""
        )

    # ✅ CLEAN STRING (VERY IMPORTANT)
    if isinstance(user_message, dict):
        user_message = str(user_message)

    user_message = user_message.strip()

    if not user_message:
        return {"message": "Sorry, I didn't catch that."}

    print("USER:", user_message)

    # ✅ RAG CALL
    try:
        answer = ask_question(
            request.app.state.vectorstore,
            request.app.state.llm,
            user_message
        )

        if not answer:
            answer = "Sorry, I couldn't find that information."

    except Exception as e:
        print("RAG ERROR:", e)
        answer = "There was an error processing your request."

    return {"message": answer}

# ---------------- ROOT ---------------- #
@app.get("/")
def home():
    return {"status": "running"}


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)