// frontend/renderer.js

// 1. --- Configuration & Global State ---
const API_URL = 'http://127.0.0.1:8000';
const POLLING_INTERVAL = 300; // Milliseconds to poll the backend (300ms = 3 times per second)
let currentSessionId = `jarvis_session_${Date.now()}`;
let activePolls = 0; 
const MAX_CONCURRENT_POLLS = 1;
let isThinking = false; 

// --- Element References ---
const hub = document.getElementById('hub');
const statusText = document.getElementById('listeningText');
const statusDiv = document.getElementById('status');
const micDot = document.getElementById('micDot');
const chatBox = document.getElementById('chat-box');
const pulse = document.getElementById('pulse');
const ttsAudio = document.getElementById('ttsAudio');
const chatHistoryContainer = document.getElementById('chat-history-container'); 

// --- UI STATE MANAGEMENT ---
const UI_STATE = {
    IDLE: { text: "Listening for 'Hey Jarvis'", ringClass: "state-listening", dot: false, pulsing: false },
    HANDSHAKE: { text: "Hey Boss, what's up?", ringClass: "state-active", dot: true, pulsing: true },
    QUERYING: { text: "Listening...", ringClass: "state-active", dot: true, pulsing: true },
    THINKING: { text: "Processing RAG...", ringClass: "state-active", dot: false, pulsing: true },
    SPEAKING: { text: "Speaking...", ringClass: "state-active", dot: false, pulsing: false }
};

function setInputLock(isLocked) {
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    const chatClearBtn = document.getElementById('chat-clear-btn');
    
    chatInput.disabled = isLocked;
    chatSendBtn.disabled = isLocked;
    chatClearBtn.disabled = isLocked; 
    
    isThinking = isLocked;
    chatInput.placeholder = isLocked ? "Jarvis is processing your request..." : "Ask a question, Boss (Text Mode)...";
}

function updateUI(state, customText = null) {
    statusText.innerText = customText || state.text;
    hub.className = 'hub ' + state.ringClass;
    micDot.classList.toggle('on', state.dot);
    pulse.classList.toggle('on', state.pulsing);
    statusDiv.innerText = state.text;
    
    const locked = (state === UI_STATE.THINKING || state === UI_STATE.SPEAKING);
    setInputLock(locked);
}

// --- Chat History Display & Clear Functions ---
function displayMessage(sender, text) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender.toLowerCase());
    messageElement.innerHTML = `<strong>${sender}:</strong> ${text}`;
    
    chatHistoryContainer.appendChild(messageElement);
    
    // Scroll to the bottom
    chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
}

function clearChatHistory() {
    if (isThinking) {
        console.warn("Cannot clear chat while processing.");
        return;
    }
    
    if (confirm("Are you sure you want to clear the chat history? This will also reset Jarvis's memory for this session.")) {
        chatHistoryContainer.innerHTML = '';
        currentSessionId = `jarvis_session_${Date.now()}`;
        console.log("Chat history cleared and new session started:", currentSessionId);
        updateUI(UI_STATE.IDLE, "Session Cleared. Ready for new query.");
    }
}

// --- API Helpers ---

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
        return data.answer; 

    } catch (error) {
        console.error('RAG Query Failed:', error);
        return `My apologies, Boss. The RAG system encountered an internal error. (Status: ${error.message})`; 
    }
}

async function getAudioURL(text, sessionId) {
    const payload = {
        question: text, 
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
        return `${API_URL}${data.audio_url}`; 

    } catch (error) {
        console.error('TTS Generation Failed:', error);
        return null;
    }
}

function playAudio(url, callback) {
    ttsAudio.src = url;
    
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
    if (isThinking || ttsAudio.paused === false || activePolls >= MAX_CONCURRENT_POLLS) {
        return;
    }
    
    activePolls++;
    
    try {
        // --- 1. Check for Handshake Signal (Wake Word Detected) ---
        const handshakeResponse = await fetch(`${API_URL}/get_handshake_reply`);
        const handshakeData = await handshakeResponse.json();
        
        if (handshakeData.status === 'ready' && handshakeData.audio_url) {
            
            // WAKE WORD HANDSHAKE START
            updateUI(UI_STATE.HANDSHAKE);
            const fullAudioUrl = `${API_URL}${handshakeData.audio_url}`;
            
            playAudio(fullAudioUrl, () => {
                updateUI(UI_STATE.QUERYING); 
            });
            return; 
        }


        // --- 2. Check for Transcribed Query (Query Ready) ---
        const queryResponse = await fetch(`${API_URL}/get_voice_query`);
        const queryData = await queryResponse.json();

        if (queryData.status === 'ready' && queryData.query) {
            
            // QUERY RETRIEVED
            updateUI(UI_STATE.THINKING, `Query: "${queryData.query}"`);
            
            // Call the main handler (Voice Mode)
            await handleChatSubmission(queryData.query, 'voice');
            return; 
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
    
    // 1. Display user question first (This should be done *before* calling this handler)

    // 2. RAG Processing: Get text response from backend
    const textResponse = await submitTextQuery(question);
    
    // 3. Display Jarvis's response
    if (sourceMode === 'voice') {
        
        const audioUrl = await getAudioURL(textResponse, currentSessionId);
        
        if (audioUrl) {
            playAudio(audioUrl, () => {
                displayMessage('Jarvis', textResponse); 
            });
        } else {
            displayMessage('Jarvis', textResponse); 
            updateUI(UI_STATE.IDLE); // Unlock if TTS fails
        }
    } else {
        // Text Mode: Display text response only
        displayMessage('Jarvis', textResponse); 
        updateUI(UI_STATE.IDLE, "Text Mode Active"); // Unlock immediately
    }
}

// --- Initialization & Listeners ---

async function requestMicrophonePermission() {
    try {
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
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    const chatClearBtn = document.getElementById('chat-clear-btn');
    
    // Listener for the Send button
    chatSendBtn.addEventListener('click', async () => {
        const question = chatInput.value.trim(); 
        
        if (isThinking) {
            console.warn("System busy. Please wait for the current reply.");
            return; 
        }

        if (question) {
            // 1. Lock input
            setInputLock(true); 
            
            // 2. Display user question for immediate feedback
            displayMessage('Boss', question); 

            // 3. Clear input field immediately
            chatInput.value = ''; 

            // 4. Update UI to thinking state while waiting for API
            updateUI(UI_STATE.THINKING, "Sending Text Query...");

            // 5. Process query (Wait for response)
            await handleChatSubmission(question, 'text'); 
        }
    });

    // Listener for the Enter key
    chatInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault(); 
            chatSendBtn.click();
        }
    });
    
    // Listener for Clear Button
    chatClearBtn.addEventListener('click', clearChatHistory);

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

    chatBox.classList.add('visible');
}


document.addEventListener('DOMContentLoaded', async () => {
    const hasPermission = await requestMicrophonePermission();
    
    if (hasPermission) {
        setInterval(checkBackendStatus, POLLING_INTERVAL);
    }
    
    setupTextChatListeners();
    updateUI(UI_STATE.IDLE); 
});