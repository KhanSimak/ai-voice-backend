import uvicorn
import logging
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

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

# ---------------- EXTRACT MESSAGE ---------------- #
def extract_message(data):
    if not isinstance(data, dict):
        return None

    # Retell format
    if isinstance(data.get("transcript"), str):
        return data["transcript"]

    # fallback
    if isinstance(data.get("message"), str):
        return data["message"]

    if isinstance(data.get("text"), str):
        return data["text"]

    # nested fallback
    try:
        msgs = data.get("artifact", {}).get("messages", [])
        for m in reversed(msgs):
            if m.get("role") == "user":
                return m.get("message")
    except:
        pass

    return None


# ---------------- CHAT FUNCTION (RETELL FUNCTION MODE) ---------------- #
def chat_api(user_message: str):
    if not user_message:
        return {"response": "Please try again."}

    text = user_message.lower()

    # ---- HARD LOGIC ----
    if "doctor" in text and ("name" in text or "list" in text or "all" in text):
        return {
            "response": "We have Dr. Arjun Mehta, Cardiologist. Dr. Priya Sharma, Pediatrician. Dr. Rahul Desai, Orthopedic. And Dr. Sana Khan, Dermatologist."
        }

    if "available" in text or "availability" in text:
        return {
            "response": "Doctors are available today. Which specialist do you need?"
        }

    if "book" in text or "appointment" in text:
        return {
            "response": "Sure. Which doctor would you like to book an appointment with?"
        }

    # ---- RAG ----
    try:
        answer = ask_question(
            app.state.vectorstore,
            app.state.llm,
            user_message
        )

        if answer:
            return {"response": answer}

    except Exception as e:
        print("RAG ERROR:", e)

    return {"response": "I couldn't find that information."}


# ---------------- CHAT ENDPOINT ---------------##
@app.post("/chat")
async def chat(request: Request):
    # ---- SAFE JSON PARSE (VERY IMPORTANT) ----
    try:
        data = await request.json()
    except Exception:
        try:
            raw = await request.body()
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = {}

    # ---- DEBUG ----
    print("RAW DATA:", data)

    # ---- EXTRACT USER MESSAGE (RETELL + VAPI SAFE) ----
    user_message = None

    if isinstance(data, dict):

        # 1. Direct fields
        user_message = (
            data.get("transcript") or
            data.get("message") or
            data.get("text")
        )

        # 2. Retell/VAPI full payload (IMPORTANT FIX)
        if not user_message:
            try:
                msgs = data.get("artifact", {}).get("messages", [])
                for m in reversed(msgs):
                    if m.get("role") == "user":
                        user_message = m.get("message")
                        break
            except Exception:
                pass

    # ---- FAIL SAFE ----
    if not user_message:
        return {"message": "No input received"}

    print("USER:", user_message)

    # ---- ENSURE STRING (FIXES: dict has no attribute lower) ----
    if isinstance(user_message, dict):
        user_message = user_message.get("message") or str(user_message)

    # ---- RAG CALL ----
    try:
        answer = ask_question_for_voice(
            request.app.state.vectorstore,
            request.app.state.llm,
            user_message
        )

        if not answer:
            answer = "Sorry, I couldn't find that information."

    except Exception as e:
        print("RAG ERROR:", e)
        answer = "There was an error processing your request."

    # ---- FINAL RESPONSE (RETELL FORMAT) ----
    return {
        "message": answer
    }

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    uvicorn.run("mains:app", host="0.0.0.0", port=8000)