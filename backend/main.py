import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import os
from contextlib import asynccontextmanager
import asyncio

# --- Import Jarvis Core (RAG Chain) ---
from rag_core import get_jarvis_chain

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)

# --- Global Variables ---
jarvis_chain = None

# --- Lifespan Context Manager ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown lifecycle for the Jarvis backend."""
    global jarvis_chain

    print("\n--- Jarvis Backend Startup ---")

    try:
        jarvis_chain = get_jarvis_chain()
        print("✅ RAG Chain (LLM, Pinecone) loaded successfully.")
    except Exception as e:
        logging.error(f"❌ FATAL ERROR: Could not initialize Jarvis RAG chain: {e}")
        jarvis_chain = None

    yield  # Keeps the app alive while running

    # --- SHUTDOWN ---
    print("\n--- Jarvis Backend Shutdown ---")
    print("✅ All resources cleaned up successfully.")


# --- FastAPI Initialization ---

app = FastAPI(title="Jarvis RAG Backend API (Text Mode)", lifespan=lifespan)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Schema ---

class ChatRequest(BaseModel):
    """Incoming query from the frontend."""
    input: str
    session_id: str = "default_user"


# --- Streaming Generator ---

async def invoke_stream(chain, input_data, config):
    """Streams tokens from the LLM back to the frontend as plain text."""
    # Run synchronous generator in a thread
    stream_generator = await asyncio.to_thread(chain.stream, input_data, config)

    for chunk in stream_generator:
    # Handle both dicts and direct text chunks
        if isinstance(chunk, str):
            yield chunk
        elif isinstance(chunk, dict):
            yield chunk.get("answer", "")
        elif hasattr(chunk, "content"):  # some LLMs return message objects
            yield str(chunk.content)
        else:
            yield str(chunk)


# --- API Endpoints ---

@app.get("/")
def read_root():
    """Health check endpoint."""
    if jarvis_chain:
        return {"status": "Jarvis API Online", "chain_status": "Ready"}
    else:
        raise HTTPException(status_code=503, detail="RAG Chain Initialization Failed.")


@app.post("/stream_chat")
async def stream_chat_endpoint(request: ChatRequest):
    """Main endpoint: streams LLM responses token-by-token."""
    if not jarvis_chain:
        raise HTTPException(status_code=503, detail="Jarvis RAG Chain is not initialized.")

    input_data = {"input": request.input}
    config = {"configurable": {"session_id": request.session_id}}

    logging.info(f"🧠 Query received (session: {request.session_id}): {request.input}")

    return StreamingResponse(
        invoke_stream(jarvis_chain, input_data, config),
        media_type="text/plain"
    )


# --- Server Entry Point ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
