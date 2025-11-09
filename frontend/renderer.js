// frontend/renderer.js

// 1. --- Configuration & Global State ---
const API_URL = 'http://127.0.0.1:8000';
const POLLING_INTERVAL = 300; // Milliseconds to poll the backend (300ms = 3 times per second)
let currentSessionId = `jarvis_session_${Date.now()}`;
let currentAudioUrl = null; // Stores the URL of the generated audio to prevent replay loops
let currentMode = 'voice'; // Tracks if the user last used 'voice' or 'text'
const MAX_CONCURRENT_POLLS = 1; // Limit concurrent HTTP calls to avoid overloading the event loop
let activePolls = 0; // Counter for active polling requests

// --- Element References ---
const hub = document.getElementById('hub');
const statusText = document.getElementById('listeningText');
const statusDiv = document.getElementById('status');
const micDot = document.getElementById('micDot');
const chatBox = document.getElementById('chat-box');
const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send-btn');
const pulse = document.getElementById('pulse');
const ttsAudio = document.getElementById('ttsAudio');

// --- UI STATE MANAGEMENT ---
// State definitions drive the visual updates
const UI_STATE = {
    IDLE: { text: "Listening for 'Hey Jarvis'", ringClass: "state-listening", dot: false, pulsing: false },
    HANDSHAKE: { text: "Hey Boss, what's up?", ringClass: "state-active", dot: true, pulsing: true },
    QUERYING: { text: "Listening...", ringClass: "state-active", dot: true, pulsing: true },
    THINKING: { text: "Processing RAG...", ringClass: "state-active", dot: false, pulsing: true },
    SPEAKING: { text: "Speaking...", ringClass: "state-active", dot: false, pulsing: false }
};

function updateUI(state, customText = null) {
    statusText.innerText = customText || state.text;
    hub.className = 'hub ' + state.ringClass;
    micDot.classList.toggle('on', state.dot);
    pulse.classList.toggle('on', state.pulsing);
    statusDiv.innerText = state.text; // Update statusDiv for clearer feedback
}

// --- API Helpers ---

// CRITICAL: Sends query to RAG pipeline (FastAPI /chat)
async function submitTextQuery(question) {
    const payload = {
        question: question,
        session_id: currentSessionId
    };
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        
        const data = await response.json();
        return data.answer; // Returns Jarvis's text response

    } catch (error) {
        console.error('RAG Query Failed:', error);
        return `My apologies, Boss. The RAG system encountered an internal error. (${error.message})`;
    }
}

// CRITICAL: Generates TTS audio file on the backend and returns the URL
async function getAudioURL(text, sessionId) {
    const payload = {
        question: text, // Reuse ChatRequest structure
        session_id: sessionId
    };
    try {
        const response = await fetch(`${API_URL}/speak`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error(`TTS generation failed: ${response.status}`);
        
        const data = await response.json();
        return `${API_URL}${data.audio_url}`; // Full URL, e.g., http://127.0.0.1:8000/audio/jarvis_...wav

    } catch (error) {
        console.error('TTS Generation Failed:', error);
        // Fallback to text if audio fails
        return null;
    }
}

// CRITICAL: Plays the audio URL and manages the SPEAKING state
function playAudio(url, callback) {
    ttsAudio.src = url;
    currentAudioUrl = url; // Store the current URL
    
    // Ensure we reset state when audio finishes
    ttsAudio.onended = () => {
        updateUI(UI_STATE.IDLE);
        if (callback) callback();
    };
    
    ttsAudio.onerror = (e) => {
        console.error("Audio playback error:", e);
        updateUI(UI_STATE.IDLE, "Audio Playback Error!");
        if (callback) callback();
    };

    updateUI(UI_STATE.SPEAKING);
    ttsAudio.play().catch(e => console.error("Error playing audio:", e));
}

// --- Voice Polling Loop (The state machine) ---

async function checkBackendStatus() {
    // Limit polling concurrency
    if (activePolls >= MAX_CONCURRENT_POLLS || ttsAudio.paused === false) {
        return;
    }
    
    activePolls++;
    
    try {
        // --- 1. Check for Handshake Signal (Wake Word Detected) ---
        const handshakeResponse = await fetch(`${API_URL}/get_handshake_reply`);
        const handshakeData = await handshakeResponse.json();
        
        if (handshakeData.status === 'ready' && handshakeData.audio_url) {
            
            // WAKE WORD HANDSHAKE START: Jarvis says "Hey Boss, what's up?"
            updateUI(UI_STATE.HANDSHAKE);
            const fullAudioUrl = `${API_URL}${handshakeData.audio_url}`;
            
            // Play the handshake audio, then immediately poll for the user query
            playAudio(fullAudioUrl, () => {
                // Once Jarvis finishes talking, transition UI to actively listening for transcription
                updateUI(UI_STATE.QUERYING); 
            });
            return; // Exit poll cycle to wait for playback/transcription
        }


        // --- 2. Check for Transcribed Query (Query Ready) ---
        const queryResponse = await fetch(`${API_URL}/get_voice_query`);
        const queryData = await queryResponse.json();

        if (queryData.status === 'ready' && queryData.query) {
            
            // QUERY RETRIEVED: User's voice command is ready for RAG
            updateUI(UI_STATE.THINKING, `Query: "${queryData.query}"`);
            
            // Call the main handler
            handleChatSubmission(queryData.query, 'voice');
            return; // Exit poll cycle until RAG and TTS are complete
        }

    } catch (error) {
        console.error('Polling error:', error);
        updateUI(UI_STATE.IDLE, "API Error");

    } finally {
        activePolls--;
    }
}

// --- Main Handler for RAG Submission (Voice and Text) ---

async function handleChatSubmission(question, sourceMode) {
    currentMode = sourceMode;
    
    // RAG Processing: Get text response from backend
    const textResponse = await submitTextQuery(question);
    
    if (sourceMode === 'voice') {
        // Voice Mode: Generate and play audio response
        updateUI(UI_STATE.THINKING, "Generating Jarvis's Reply...");

        const audioUrl = await getAudioURL(textResponse, currentSessionId);
        
        if (audioUrl) {
            playAudio(audioUrl, () => {
                // Return to IDLE after speaking
                updateUI(UI_STATE.IDLE);
            });
        } else {
            // Fallback for TTS failure: Display text and wait 3s
            updateUI(UI_STATE.IDLE, textResponse); 
            setTimeout(() => updateUI(UI_STATE.IDLE), 3000);
        }
    } else {
        // Text Mode: Display text response only (Per requirement #2)
        updateUI(UI_STATE.IDLE, "Text Mode Active"); // Reset hub visual
        statusDiv.innerText = textResponse; // Display full response in the status area
    }
}

// --- 4. Initialization & Listeners ---

// FIX: Request Microphone Permission (The critical fix)
async function requestMicrophonePermission() {
    try {
        // This is the call that triggers the OS permission dialog
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        stream.getTracks().forEach(track => track.stop()); 
        console.log("Microphone access granted.");
        return true;
    } catch (error) {
        console.error("Microphone access denied:", error);
        statusDiv.innerText = "🚨 Error: Microphone access denied. Voice commands disabled.";
        return false;
    }
}

function setupTextChatListeners() {
    // Listener for the Send button
    chatSendBtn.addEventListener('click', async () => {
        const question = chatInput.value.trim();
        if (question) {
            updateUI(UI_STATE.THINKING, "Sending Text Query...");
            // Call the main chat API function
            await handleChatSubmission(question, 'text'); 
            chatInput.value = ''; // Clear input field
        }
    });

    // Listener for the Enter key
    chatInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault(); // Prevent newline in input
            chatSendBtn.click();
        }
    });
    
    // Listener to toggle visibility of the chatbox when hub is clicked
    hub.addEventListener('click', () => {
        chatBox.classList.toggle('visible');
        if (chatBox.classList.contains('visible')) {
             chatInput.focus();
             updateUI(UI_STATE.IDLE, "Text Mode Active");
        } else {
             updateUI(UI_STATE.IDLE, "Listening for 'Hey Jarvis'");
        }
    });

    // Initial check for visibility (Chatbox is hidden by default in CSS, but this is a double-check)
    chatBox.classList.add('visible'); // Start in text mode since it's the primary manual input
}


document.addEventListener('DOMContentLoaded', async () => {
    // A. Request permission immediately
    const hasPermission = await requestMicrophonePermission();
    
    // B. Start the core voice loop (Polling runs continuously)
    if (hasPermission) {
        setInterval(checkBackendStatus, POLLING_INTERVAL);
    }
    
    // C. Set up event listeners for the text chatbox
    setupTextChatListeners();
    updateUI(UI_STATE.IDLE); // Set initial UI state
});