// frontend/renderer.js

// === Configuration ===
const API_URL = "http://127.0.0.1:8000";
let currentSessionId = `jarvis_session_${Date.now()}`;
let isThinking = false;

// === UI Elements ===
const hub = document.getElementById("hub");
const statusText = document.getElementById("listeningText");
const statusDiv = document.getElementById("status");
const chatBox = document.getElementById("chat-box");
const pulse = document.getElementById("pulse");
const chatHistoryContainer = document.getElementById("chat-history-container");

// === UI States ===
const UI_STATE = {
  IDLE: { text: "Text Mode Active — Ready for input.", ringClass: "", pulsing: false },
  THINKING: { text: "Jarvis is thinking...", ringClass: "active", pulsing: true },
};

// === UI Helpers ===
function setInputLock(isLocked) {
  const chatInput = document.getElementById("chat-input");
  const chatSendBtn = document.getElementById("chat-send-btn");
  const chatClearBtn = document.getElementById("chat-clear-btn");

  chatInput.disabled = isLocked;
  chatSendBtn.disabled = isLocked;
  chatClearBtn.disabled = isLocked;
  isThinking = isLocked;

  chatInput.placeholder = isLocked
    ? "Jarvis is processing your request..."
    : "Ask a question, Boss (Text Mode)...";
}

function updateUI(state, customText = null) {
  statusText.innerText = customText || state.text;
  pulse.classList.toggle("on", state.pulsing);
  statusDiv.innerText = customText || state.text;
  setInputLock(state === UI_STATE.THINKING);
}

// === Chat Message Display ===
function displayMessage(sender, text) {
  const msg = document.createElement("div");
  msg.classList.add("chat-message", sender.toLowerCase());
  msg.innerHTML = `<strong>${sender}:</strong> ${text}`;
  chatHistoryContainer.appendChild(msg);
  chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
}

// === Clear Chat ===
function clearChatHistory() {
  if (isThinking) return alert("Cannot clear chat while processing.");
  if (confirm("Clear chat history? This resets Jarvis's memory for this session.")) {
    chatHistoryContainer.innerHTML = "";
    currentSessionId = `jarvis_session_${Date.now()}`;
    updateUI(UI_STATE.IDLE, "Session Cleared. Ready for new query.");
  }
}

// === Streaming Response Handler ===
async function submitTextQuery(question) {
  const payload = { input: question.trim(), session_id: currentSessionId };

  try {
    const response = await fetch(`${API_URL}/stream_chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let fullText = "";

    // Create a single Jarvis message container
    const jarvisMsg = document.createElement("div");
    jarvisMsg.classList.add("chat-message", "jarvis");
    jarvisMsg.innerHTML = `<strong>Jarvis:</strong> `;
    chatHistoryContainer.appendChild(jarvisMsg);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      fullText += chunk;

      // Update streamed content live
      jarvisMsg.innerHTML = `<strong>Jarvis:</strong> ${fullText}`;
      chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
    }

    return fullText.trim();
  } catch (err) {
    console.error("❌ RAG Query Failed:", err);
    return `My apologies, Boss. The RAG system encountered an error (${err.message}).`;
  }
}

// === Main Handler ===
async function handleChatSubmission(question) {
  // Add Boss message
  displayMessage("Boss", question);
  updateUI(UI_STATE.THINKING, "Processing your query...");

  // Send query to backend
  const textResponse = await submitTextQuery(question);

  // Unlock input after completion
  updateUI(UI_STATE.IDLE, "Text Mode Active");
  setInputLock(false);
}

// === Event Listeners ===
function setupTextChatListeners() {
  const chatInput = document.getElementById("chat-input");
  const chatSendBtn = document.getElementById("chat-send-btn");
  const chatClearBtn = document.getElementById("chat-clear-btn");

  // Send button
  chatSendBtn.addEventListener("click", async () => {
    const question = chatInput.value.trim();
    if (!question || isThinking) return;
    chatInput.value = "";
    await handleChatSubmission(question);
  });

  // Press Enter
  chatInput.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      chatSendBtn.click();
    }
  });

  // Clear button
  chatClearBtn.addEventListener("click", clearChatHistory);

  // Hub toggle
  hub.addEventListener("click", () => {
    chatBox.classList.toggle("visible");
    if (chatBox.classList.contains("visible")) {
      chatInput.focus();
      updateUI(UI_STATE.IDLE);
    } else {
      updateUI(UI_STATE.IDLE, "Chat hidden — click Jarvis to reopen.");
    }
  });

  chatBox.classList.add("visible");
}

// === Initialize ===
document.addEventListener("DOMContentLoaded", () => {
  setupTextChatListeners();
  updateUI(UI_STATE.IDLE);
});
