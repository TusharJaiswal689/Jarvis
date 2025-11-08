import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse 
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import threading
import os
from contextlib import asynccontextmanager

# Import the necessary RAG components from your modules
from rag_core import get_jarvis_chain # CLEAN IMPORT
from voice_activation.wake_listener import start_voice_listener, get_transcribed_query, get_handshake_reply 
from voice_activation.tts_generator import generate_tts_audio 

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# --- Global Variables for Lifespan ---
listener_thread = None
is_listening = threading.Event()
MICROPHONE_INDEX = 0 # Confirmed mic index
jarvis_chain = None 

# --- Lifespan Context Manager ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global listener_thread, jarvis_chain
    
    print("\n--- Jarvis Backend Startup ---")
    
    # 1. Initialize RAG Chain (expensive task)
    try:
        jarvis_chain = get_jarvis_chain()
        print("✅ RAG Chain (LLM, Pinecone) loaded successfully.")
    except Exception as e:
        logging.error(f"❌ FATAL ERROR: Could not initialize Jarvis RAG chain: {e}")
        jarvis_chain = None 
    
    # 2. Start the continuous Wake Word listener thread
    print("Starting Jarvis Voice Listener...")
    recorder_config = {'device_index': MICROPHONE_INDEX} 
    listener_thread = start_voice_listener(is_listening, recorder_config)
    
    if listener_thread:
        print(f"✅ Voice Listener thread started on device {MICROPHONE_INDEX}.")
    else:
        print("❌ Failed to start listener thread. Voice input will be disabled.")
        
    yield

    # --- SHUTDOWN: Resource Cleanup ---
    print("\n--- Jarvis Backend Shutdown ---")
    is_listening.clear() 
    if listener_thread and listener_thread.is_alive():
        listener_thread.join(timeout=2) 
        print("✅ Voice Listener thread safely stopped.")


# --- FastAPI Initialization ---

app = FastAPI(title="Jarvis RAG Backend API", lifespan=lifespan)

# --- Middleware & Static Files ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/audio", StaticFiles(directory="voice_activation/tts_output"), name="audio")

# --- Request/Response Schemas ---

class ChatRequest(BaseModel):
    # ✅ FIX: Aligning input key to 'question' for the RAG chain
    question: str 
    session_id: str = "default_user" 

class ChatResponse(BaseModel):
    answer: str

# --- API Endpoints ---

@app.get("/")
def read_root():
    """Simple health check endpoint."""
    if jarvis_chain:
        return {"status": "Jarvis API Online", "chain_status": "Ready", "mic_index": MICROPHONE_INDEX}
    else:
        raise HTTPException(status_code=503, detail="RAG Chain Initialization Failed.")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint for processing text queries through the RAG pipeline."""
    if not jarvis_chain:
         raise HTTPException(status_code=503, detail="Jarvis RAG Chain is not initialized.")

    logging.info(f"Received text query for session '{request.session_id}': {request.question}")
    
    try:
        # The chain returns a string directly, not a dict
        response = jarvis_chain.invoke(
            {"question": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        # Handle string response from the chain
        return ChatResponse(answer=response if isinstance(response, str) else str(response))

    except Exception as e:
        logging.error(f"Error processing text chat request: {e}")
        return ChatResponse(
            answer="I apologize, Boss, but I encountered a system error while processing your request."
        )

# --- Voice Endpoints ---

@app.get("/get_voice_query")
async def get_voice_query():
    """Endpoint for the frontend to retrieve transcribed text."""
    transcribed_text = get_transcribed_query()
    
    if transcribed_text:
        logging.info(f"Transcribed query retrieved: {transcribed_text}")
        return {"status": "ready", "query": transcribed_text}
    else:
        return {"status": "listening", "query": None}


@app.get("/get_handshake_reply")
async def get_handshake_reply_endpoint():
    """Endpoint for the frontend to poll and check if the wake word was detected."""
    handshake_text = get_handshake_reply()
    
    if handshake_text:
        logging.info(f"Handshake signal received: '{handshake_text}'")
        
        audio_file_path = generate_tts_audio(
            text=handshake_text,
            session_id="handshake"
        )
        
        if audio_file_path:
            audio_url = f"/audio/{os.path.basename(audio_file_path)}"
            return {"status": "ready", "audio_url": audio_url}
        else:
            raise HTTPException(status_code=500, detail="Handshake TTS generation failed.")
            
    else:
        return {"status": "listening", "audio_url": None}


@app.post("/speak", response_class=JSONResponse)
async def speak_endpoint(request: ChatRequest):
    """Generates audio from Jarvis's text response using Coqui TTS and returns the URL."""
    if not request.question:
        raise HTTPException(status_code=400, detail="Text required for TTS generation.")

    audio_file_path = generate_tts_audio(
        text=request.question,
        session_id=request.session_id
    )
    
    if audio_file_path:
        audio_url = f"/audio/{os.path.basename(audio_file_path)}"
        return {"status": "success", "audio_url": audio_url}
    else:
        raise HTTPException(status_code=500, detail="TTS generation failed. Check backend logs.")


# --- Server Start ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")