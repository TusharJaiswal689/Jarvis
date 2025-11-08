from pathlib import Path
import uuid
from TTS.api import TTS

AUDIO_OUTPUT_DIR = Path("./generated_audio")
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_tts_audio(text: str) -> str:
    """
    Generate speech from text using Coqui TTS.
    Returns the path to the generated audio file.
    """
    try:
        # Initialize TTS with British male voice
        tts = TTS(model_name="tts_models/en/vctk/vits",
                  progress_bar=False)
        
        # Generate unique filename
        filename = f"jarvis_{uuid.uuid4()}.wav"
        output_path = AUDIO_OUTPUT_DIR / filename
        
        # Generate audio with British voice (speaker p236)
        tts.tts_to_file(text=text,
                        speaker="p236",
                        file_path=str(output_path),
                        language="en")
        
        return str(output_path)
        
    except Exception as e:
        print(f"TTS Generation Error: {e}")
        return None