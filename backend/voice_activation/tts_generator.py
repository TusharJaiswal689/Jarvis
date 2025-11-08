import os
import uuid
import torch
import logging
from TTS.api import TTS
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
# 1. FIX: Define the output directory relative to the backend structure
# It should point to: backend/voice_activation/tts_output
OUTPUT_DIR_NAME = "tts_output"
AUDIO_OUTPUT_DIR = Path(os.path.join(os.path.dirname(__file__), OUTPUT_DIR_NAME))
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

# TTS Model Selection (Using a stable, high-quality VCTK speaker)
TTS_MODEL_NAME = "tts_models/en/vctk/vits"
TTS_SPEAKER_ID = "p236" # A suitable British male voice

# --- Global Model Initialization (FIX for latency) ---
# This code runs ONCE when the FastAPI server starts.
tts_generator = None
DEVICE = "cpu"

try:
    print(f"Loading Coqui TTS model: {TTS_MODEL_NAME}...")
    
    # 1. Check for GPU and select device
    if torch.cuda.is_available():
        DEVICE = "cuda"
    
    # 2. Initialize the model globally
    tts_generator = TTS(model_name=TTS_MODEL_NAME, progress_bar=False).to(DEVICE)
    print(f"✅ Coqui TTS model loaded successfully on {DEVICE}.")

except Exception as e:
    logger.error(f"FATAL ERROR loading Coqui TTS model: {e}")
    print("TTS generation will be disabled.")


# --- Core Function ---

def generate_tts_audio(text: str, session_id: str = "default_user") -> str | None:
    """
    Generates an audio file from text using the globally loaded Coqui TTS model.
    Returns the absolute path to the generated audio file.
    """
    if tts_generator is None:
        logger.error("TTS Generator is not initialized. Cannot produce audio.")
        return None

    try:
        # Generate unique filename (using session ID for context, uuid for uniqueness)
        filename = f"jarvis_{session_id}_{uuid.uuid4()}.wav"
        output_path = AUDIO_OUTPUT_DIR / filename
        
        # 3. Generate audio with specified British speaker
        tts_generator.tts_to_file(
            text=text,
            speaker=TTS_SPEAKER_ID,
            file_path=str(output_path),
            language="en"
        )
        
        # Return the absolute path for the main API to expose
        return str(output_path)
        
    except Exception as e:
        logger.error(f"TTS Generation Error: {e}")
        return None