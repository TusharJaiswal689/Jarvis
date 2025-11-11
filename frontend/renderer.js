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
  IDLE: { text: "Text Mode Active — Ready for input.", pulsing: false },
  THINKING: { text: "Jarvis is thinking...", pulsing: true },
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
  const text = customText || state.text;
  statusText.innerText = text;
  statusDiv.innerText = text;
  pulse.classList.toggle("on", state.pulsing);
  setInputLock(state === UI_STATE.THINKING);
}

// === Chat Message Display ===
function displayMessage(sender, text) {
  const msg = document.createElement("div");
  msg.classList.add("chat-message", sender.toLowerCase());

  if (sender === "Jarvis") {
    const parsedHTML = marked.parse(text || ""); // render Markdown
    msg.innerHTML = `<strong>${sender}:</strong> ${parsedHTML}`;
  } else {
    msg.innerHTML = `<strong>${sender}:</strong> ${text}`;
  }

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

    // Stream incoming text
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      fullText += chunk;

      // Render Markdown live
      const parsedHTML = marked.parse(fullText);
      jarvisMsg.innerHTML = `<strong>Jarvis:</strong> ${parsedHTML}`;
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
  displayMessage("Boss", question);
  updateUI(UI_STATE.THINKING, "Processing your query...");

  const textResponse = await submitTextQuery(question);

  updateUI(UI_STATE.IDLE, "Text Mode Active");
  setInputLock(false);
}

// === Event Listeners ===
function setupTextChatListeners() {
  const chatInput = document.getElementById("chat-input");
  const chatSendBtn = document.getElementById("chat-send-btn");
  const chatClearBtn = document.getElementById("chat-clear-btn");

  chatSendBtn.addEventListener("click", async () => {
  const question = chatInput.value.trim();
  if (!question || isThinking) return;

  chatInput.value = "";
  await handleChatSubmission(question);

  // ✅ Instantly refocus on chat input after sending
  setTimeout(() => chatInput.focus(), 100);
  });


  chatInput.addEventListener("keypress", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    chatSendBtn.click();

    // ✅ Keep the input focused even after sending
    setTimeout(() => chatInput.focus(), 100);
    }
  });

  chatClearBtn.addEventListener("click", clearChatHistory);

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

// === File Upload ===
async function setupUploadListener() {
  const uploadInput = document.getElementById("file-upload");
  const uploadLabel = document.getElementById("upload-label");
  if (!uploadInput || !uploadLabel) return;

  // Disable upload icon when Jarvis is processing
  function toggleUploadLock(lock) {
    if (lock) {
      uploadLabel.style.opacity = "0.5";
      uploadLabel.style.pointerEvents = "none";
      uploadInput.disabled = true;
    } else {
      uploadLabel.style.opacity = "1";
      uploadLabel.style.pointerEvents = "auto";
      uploadInput.disabled = false;
    }
  }

  // Add listener for file changes
  uploadInput.addEventListener("click", (event) => {
    // Prevent the file picker from even opening when busy
    if (isThinking) {
      event.preventDefault();
      alert("Hold on, Boss — I'm still working on your last request.");
    }
  });

  uploadInput.addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Lock everything while uploading
    isThinking = true;
    toggleUploadLock(true);
    setInputLock(true);
    updateUI(UI_STATE.THINKING, `Uploading ${file.name}...`);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", currentSessionId);

    try {
      const response = await fetch(`${API_URL}/upload_doc`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (result.status === "success") {
        displayMessage("Jarvis", `Document "${result.filename}" added successfully, Boss. Ready for queries.`);
        updateUI(UI_STATE.IDLE, "Document added successfully!");
      } else {
        displayMessage("Jarvis", `Sorry Boss, I couldn't process that file: ${result.error || "Unknown error"}.`);
        updateUI(UI_STATE.IDLE, "Error in upload.");
      }
    } catch (err) {
      console.error("Upload failed:", err);
      displayMessage("Jarvis", "My apologies, Boss. Upload failed due to a network or server issue.");
      updateUI(UI_STATE.IDLE, "Upload failed.");
    } finally {
      event.target.value = ""; // Reset file input
      isThinking = false;
      toggleUploadLock(false);
      setInputLock(false);
      document.getElementById("chat-input").focus(); // Refocus chat box
    }
  });
}

// === Initialize ===
document.addEventListener("DOMContentLoaded", async () => {
  setupTextChatListeners();
  setupUploadListener();
  updateUI(UI_STATE.IDLE);
});
