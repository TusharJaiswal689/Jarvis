import os
import pvporcupine
import pvrecorder
from vosk import Model, KaldiRecognizer 
import json
import struct
import threading
import queue
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")

# --- Global Configuration & Queues ---
AUDIO_QUEUE = queue.Queue()       # Queue for transcribed user query (from Vosk)
HANDSHAKE_QUEUE = queue.Queue()   # Queue for the handshake signal (Jarvis's verbal reply)

WAKE_WORD_REPLY = "Hey Boss, what's up?"
SAMPLE_RATE = 16000 
TIMEOUT_SECONDS = 5.0 

# --- Paths ---
WAKE_WORD_PATH = os.path.join(os.path.dirname(__file__), "jarvis_wake_word.ppn") 
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../vosk_model") 


class VoiceListener:
    def __init__(self, device_index: int = 0):
        # 1. Porcupine Initialization
        if not os.path.exists(WAKE_WORD_PATH):
            raise FileNotFoundError(f"Missing PPN file: {WAKE_WORD_PATH}. Please create the 'Hey Jarvis' file.")
            
        self.porcupine = pvporcupine.create(
            access_key=PICOVOICE_ACCESS_KEY,
            keyword_paths=[WAKE_WORD_PATH]
        )
        
        # 2. Vosk Initialization
        self.vosk_model = Model(VOSK_MODEL_PATH)
        # Using the global SAMPLE_RATE constant
        self.recognizer = KaldiRecognizer(self.vosk_model, SAMPLE_RATE) 
        
        # 3. PvRecorder Initialization (Fix applied: no sample_rate argument)
        self.recorder = pvrecorder.PvRecorder(
            device_index=device_index,
            frame_length=self.porcupine.frame_length,
            # Sample rate is omitted, as it defaults to 16000 Hz
        )
        
        # State variables
        self.is_recording_command = False
        self.FRAMES_PER_SECOND = SAMPLE_RATE / self.porcupine.frame_length # Using global SAMPLE_RATE


    def process_audio(self, is_listening: threading.Event):
        """
        The core continuous loop running in the background thread.
        """
        print("Microphone Thread: Starting to listen for wake word...")
        
        recording_frames = []
        self.recorder.start()
        
        try:
            while is_listening.is_set():
                pcm_audio = self.recorder.read()
                
                if not self.is_recording_command:
                    # --- PASSIVE STATE: Looking for "Hey Jarvis" ---
                    
                    # Unpack the 16-bit PCM audio frame for Porcupine processing
                    pcm_tuple = struct.unpack_from("h" * self.porcupine.frame_length, pcm_audio)
                    keyword_index = self.porcupine.process(pcm_tuple)
                    
                    if keyword_index >= 0:
                        # WAKE WORD DETECTED!
                        print("\n[--- WAKE WORD DETECTED! ---]")
                        self.is_recording_command = True
                        
                        # Signal the main thread to initiate the verbal handshake (TTS)
                        HANDSHAKE_QUEUE.put(WAKE_WORD_REPLY)
                        
                        # Start recording the user's command
                        recording_frames = [pcm_audio] 
                
                else:
                    # --- ACTIVE STATE: Recording User Command ---
                    recording_frames.append(pcm_audio)
                    
                    # Stop recording after TIMEOUT_SECONDS
                    MAX_FRAMES = int(self.FRAMES_PER_SECOND * TIMEOUT_SECONDS)
                    
                    if len(recording_frames) >= MAX_FRAMES:
                        print(f"\n[--- Command Recording Finished ({TIMEOUT_SECONDS}s). Transcribing... ---]")
                        self.is_recording_command = False
                        
                        # Process the recorded audio using Vosk
                        transcribed_text = self._transcribe_audio_vosk(recording_frames)
                        
                        if transcribed_text:
                            # Put the transcribed command into the global queue for the FastAPI server
                            AUDIO_QUEUE.put(transcribed_text)
                        
                        # Reset and resume listening
                        recording_frames = []
                        print("Microphone Thread: Resuming listening for 'Hey Jarvis'...")

        except Exception as e:
            print(f"Microphone Loop Error: {e}")
        finally:
            if hasattr(self, 'recorder') and self.recorder: self.recorder.stop()
            if hasattr(self, 'porcupine') and self.porcupine: self.porcupine.delete()
            print("Microphone Thread: Cleanly shut down.")

    def _transcribe_audio_vosk(self, audio_frames: list) -> str:
        """Helper function to transcribe the recorded command."""
        audio_data = b"".join(audio_frames)
        
        if self.recognizer.AcceptWaveform(audio_data):
            result = json.loads(self.recognizer.Result())
            return result.get('text', '')
        
        result = json.loads(self.recognizer.FinalResult())
        return result.get('text', '')


# --- Global Access Functions (Called by main.py) ---

def start_voice_listener(is_listening: threading.Event, recorder_config: dict) -> threading.Thread:
    """Initializes VoiceListener and starts the background thread."""
    try:
        listener = VoiceListener(device_index=recorder_config.get('device_index', 0))
        thread = threading.Thread(
            target=listener.process_audio,
            args=(is_listening,),
            daemon=True
        )
        thread.start()
        # is_listening.set() # Set in the main thread (optional, but cleaner here)
        return thread
        
    except Exception as e:
        print(f"❌ Failed to start voice listener: {e}")
        return None

def get_transcribed_query() -> str:
    """Retrieves the last transcribed query from the global audio queue."""
    global AUDIO_QUEUE
    try:
        if not AUDIO_QUEUE.empty():
            return AUDIO_QUEUE.get_nowait()
    except queue.Empty:
        pass
    return ""

def get_handshake_reply() -> str:
    """Retrieves the handshake reply text (Jarvis's 'Hey Boss, what's up?')."""
    global HANDSHAKE_QUEUE
    try:
        if not HANDSHAKE_QUEUE.empty():
            return HANDSHAKE_QUEUE.get_nowait()
    except queue.Empty:
        pass
    return ""