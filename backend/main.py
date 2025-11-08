import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import threading
import os
from contextlib import asynccontextmanager

# --- Project Imports ---
from rag_core.chat_pipeline import get_jarvis_chain # For RAG logic
from voice_activation.wake_listener import start_voice_listener, get_transcribed_query 
from voice_activation.tts_generator import generate_tts_audio # For Jarvis's voice

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# --- Global Variables for Lifespan ---
listener_thread = None
is_listening = threading.Event()
# !!! REPLACE 0 with your actual microphone index if it's different !!!
MICROPHONE_INDEX = 0 
jarvis_chain = None # Global variable to hold the initialized RAG chain

# --- Lifespan Context Manager ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    global listener_thread, jarvis_chain
    
    # --- STARTUP: Resource Initialization ---
    print("\n--- Jarvis Backend Startup ---")
    
    # 1. Initialize RAG Chain (Expensive, must be done once)
    try:
        jarvis_chain = get_jarvis_chain()
        print("✅ RAG Chain (LLM, Pinecone) loaded successfully.")
    except Exception as e:
        logging.error(f"❌ FATAL ERROR: Could not initialize Jarvis RAG chain: {e}")
        jarvis_chain = None # Keep it None if startup fails
    
    # 2. Start the continuous Wake Word listener thread
    print("Starting Jarvis Voice Listener...")
    recorder_config = {'device_index': MICROPHONE_INDEX} 
    listener_thread = start_voice_listener(is_listening, recorder_config)
    
    if listener_thread:
        print(f"✅ Voice Listener thread started on device {MICROPHONE_INDEX}.")
    else:
        print("❌ Failed to start listener thread. Voice input will be disabled.")
        
    # Yield control to the application to handle requests
    yield

    # --- SHUTDOWN: Resource Cleanup ---
    print("\n--- Jarvis Backend Shutdown ---")
    
    # 1. Safely stop the continuous listener thread
    print("Shutting down Jarvis Voice Listener...")
    is_listening.clear() 
    if listener_thread and listener_thread.is_alive():
        listener_thread.join(timeout=2) 
        print("✅ Voice Listener thread safely stopped.")


# --- FastAPI Initialization ---

app = FastAPI(title="Jarvis RAG Backend API", lifespan=lifespan)

# --- Middleware ---

# Essential for the Electron client to talk to the FastAPI server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for local development/EXE packaging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the directory containing the generated audio files
# The frontend will request the saved WAV file from /audio/filename.wav
Path("voice_activation/tts_output").mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory="voice_activation/tts_output"), name="audio")

# --- Request/Response Schemas ---

class ChatRequest(BaseModel):
    """Used for both chat queries and TTS text input."""
    query: str
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

    logging.info(f"Received text query for session '{request.session_id}': {request.query}")
    
    try:
        response_data = jarvis_chain.invoke(
            {"input": request.query},
            config={"configurable": {"session_id": request.session_id}}
        )
        return ChatResponse(answer=response_data['answer'])

    except Exception as e:
        logging.error(f"Error processing text chat request: {e}")
        return ChatResponse(answer="My apologies, Boss. I encountered a system error while processing your text request.")

# --- Voice Endpoints ---

@app.get("/get_voice_query")
async def get_voice_query():
    """
    Endpoint for the frontend to poll and retrieve transcribed text 
    from the background listener queue.
    """
    transcribed_text = get_transcribed_query()
    
    if transcribed_text:
        # Returns the query and clears it from the queue
        logging.info(f"Transcribed query retrieved: {transcribed_text}")
        return {"status": "ready", "query": transcribed_text}
    else:
        return {"status": "listening", "query": None}


@app.post("/speak", response_class=JSONResponse)
async def speak_endpoint(request: ChatRequest):
    """
    Generates audio from Jarvis's text response using Coqui TTS and returns the URL.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Text required for TTS generation.")

    # 1. Generate the audio file
    audio_file_path = generate_tts_audio(
        text=request.query,
        session_id=request.session_id
    )
    
    if audio_file_path:
        # 2. Return the URL for the frontend to play
        audio_url = f"/audio/{os.path.basename(audio_file_path)}"
        return {"status": "success", "audio_url": audio_url}
    else:
        raise HTTPException(status_code=500, detail="TTS generation failed. Check backend logs.")


# --- Server Start ---

if __name__ == "__main__":
    # Command to run the server locally
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")