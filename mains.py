from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()

    print("🔥 RAW DATA:", data)

    # Extract query safely
    query = data.get("query")

    # Fallback if query not present (IMPORTANT FIX)
    if not query:
        # Sometimes Retell sends empty {}
        # Try alternative keys
        query = data.get("message") or data.get("input") or ""

    if not query:
        return {"message": "No input received"}

    # Your actual logic here
    response = handle_query(query)

    return {"message": response}


def handle_query(query: str):
    query = query.lower()

    if "doctor" in query:
        return "We have general physicians, dentists, and cardiologists available. Which one do you need?"

    elif "book" in query:
        return "Sure, please tell me your preferred time and doctor."

    elif "hello" in query:
        return "Hello! How can I help you today?"

    else:
        return f"I understood: {query}. Can you please clarify more?"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)