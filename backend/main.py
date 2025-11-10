import os
import uvicorn
import asyncio
import tempfile
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from PyPDF2 import PdfReader

# --- LangChain Imports ---
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- Jarvis Core ---
from rag_core.chat_pipeline import get_jarvis_chain, _combine_documents

# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv()

# --- Global Config ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis_backend")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en-v1.5")

jarvis_chain = None


# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown lifecycle for the Jarvis backend."""
    global jarvis_chain

    print("\n--- Jarvis Backend Startup ---")
    try:
        jarvis_chain = get_jarvis_chain()
        print("✅ RAG Chain (LLM + Pinecone) initialized successfully.")
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: Could not initialize Jarvis RAG chain: {e}")
        jarvis_chain = None

    yield  # Keeps FastAPI running

    # --- Shutdown logic ---
    print("\n--- Jarvis Backend Shutdown ---")
    print("✅ All resources released gracefully.")


# --- Initialize FastAPI ---
app = FastAPI(title="Jarvis RAG Backend API", lifespan=lifespan)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schema for Frontend Input ---
class ChatRequest(BaseModel):
    input: str
    session_id: str = "default_user"


# --- Streaming Function ---
async def invoke_stream(chain, input_data, config):
    """Streams the response from the LLM back to the frontend."""
    loop = asyncio.get_event_loop()
    stream_generator = await loop.run_in_executor(None, lambda: chain.stream(input_data, config))

    for chunk in stream_generator:
        if isinstance(chunk, str):
            yield chunk
        elif isinstance(chunk, dict):
            yield chunk.get("answer", "")
        elif hasattr(chunk, "content"):
            yield str(chunk.content)
        else:
            yield str(chunk)


# --- RAG Query Endpoint ---
@app.post("/stream_chat")
async def stream_chat_endpoint(request: ChatRequest):
    """Main endpoint: Streams LLM output to the frontend."""
    if not jarvis_chain:
        raise HTTPException(status_code=503, detail="Jarvis RAG Chain not initialized.")

    input_data = {"input": request.input}
    config = {"configurable": {"session_id": request.session_id}}

    logger.info(f"🧠 Query received (session: {request.session_id}) -> {request.input}")

    return StreamingResponse(
        invoke_stream(jarvis_chain, input_data, config),
        media_type="text/plain"
    )


# --- Dynamic Document Upload Endpoint ---
@app.post("/upload_doc")
async def upload_doc(file: UploadFile = File(...), session_id: str = Form("default")):
    """
    Accepts PDF or TXT files, extracts text, and adds to Pinecone dynamically.
    """
    try:
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        # Extract text
        if file.filename.lower().endswith(".pdf"):
            reader = PdfReader(temp_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif file.filename.lower().endswith(".txt"):
            with open(temp_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            return {"error": "Unsupported file type. Please upload .pdf or .txt"}

        os.remove(temp_path)

        if not text.strip():
            return {"error": "No readable text found in the uploaded file."}

        # Convert to LangChain Documents
        docs = [Document(page_content=text, metadata={"source": file.filename, "session_id": session_id})]

        # Initialize embeddings + vectorstore
        embeddings = HuggingFaceEmbeddings(model_name=BGE_MODEL_NAME)
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )

        # Add document
        vectorstore.add_documents(docs)
        logger.info(f"✅ Document '{file.filename}' added to Pinecone (session={session_id})")

        return {"status": "success", "filename": file.filename}

    except Exception as e:
        logger.error(f"❌ Failed to upload document: {e}")
        return {"error": str(e)}


# --- Health Check ---
@app.get("/")
def read_root():
    """Health check endpoint."""
    if jarvis_chain:
        return {"status": "Jarvis API Online", "chain_status": "Ready"}
    else:
        raise HTTPException(status_code=503, detail="RAG Chain initialization failed.")


# --- Run Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
