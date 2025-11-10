# 🧠 Jarvis AI Agent: RAG Chatbot (Voice & Text)

## Project Overview

The Jarvis AI Agent is a voice-activated, desktop assistant prototype built using **Retrieval-Augmented Generation (RAG)**. Its primary function is to provide witty, factual, and context-aware responses based *exclusively* on a private knowledge base (college notes).

This project demonstrates a fully realized, modern architecture where a self-hosted Large Language Model (LLM) is used locally for low-latency, private retrieval.

### Key Features

* **Witty Persona:** Responses maintain a **sophisticated, humorous tone** and always address the user as 'Boss'.
* **Architecture:** Implements a modern **LCEL (LangChain Expression Language)** pipeline with concurrent threading for stability.
* **Interaction Modes:** Supports both hands-free voice commands and a dedicated, scrollable text chat interface.
* **Deployment Target:** Designed to be bundled into a single **Windows Executable (.exe)**.

***

## 🛠️ Architecture Stack

The project operates as a **Client-Server** model with a Python backend managing all AI resources and an Electron frontend managing the UI.

### 1. Python Backend (Server)

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Server/API** | **FastAPI** | Exposes streaming and synchronous endpoints (`/stream_chat`, `/speak`); manages background thread lifecycle (`lifespan`). |
| **RAG/Memory** | **LangChain** (Modern Runnables) | Builds the conversational chain, handles session memory (`ChatMessageHistory`), and executes context-aware retrieval. |
| **LLM Inference** | **Ollama** + **Llama 3 8B** | Self-hosted LLM for low-latency, private response generation. |
| **Vector Database**| **Pinecone** (Serverless) | Stores vectorized college notes for context retrieval. |
| **Voice I/O** | **Vosk** (STT), **Coqui TTS** (Voice) | Handles Speech-to-Text transcription and high-quality audio response generation. |

### 2. Frontend (Client)

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Desktop App** | **Electron** | Creates the standard desktop application with full window controls and handles application lifecycle. |
| **UI/UX** | **HTML/CSS** | Renders the minimalist, dynamic central **Jarvis circle hub** and the dedicated, scrolling chat history pane. |
| **Logic** | **JavaScript** | Manages the **continuous voice polling cycle** and handles live **Streaming** of LLM responses (token-by-token) via the Fetch API Reader. |

***

## ⚙️ Setup and Installation

### Prerequisites

1.  **Python 3.11** (Must be installed on the system).
2.  **Node.js LTS** (Must be installed globally for Electron).
3.  **Ollama Service:** Must be installed and running in the background.
    * Pull the required model: `ollama pull llama3`
4.  **Picovoice Assets:** A free developer Access Key and the custom **`jarvis_wake_word.ppn`** file (Windows platform).
5.  **Vosk Model:** The English small model must be downloaded and placed in the **`backend/vosk_model`** folder.

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone [repository-link]
    cd jarvis-ai-agent
    ```

2.  **Install Python Dependencies (Backend):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r backend/requirements.txt
    ```

3.  **Install Frontend Dependencies (Node/Electron):**
    ```bash
    npm install
    ```

4.  **Configuration (`backend/.env`):** Populate the `.env` file with your credentials (ensure the index is created in Pinecone):
    ```ini
    PINECONE_API_KEY="YOUR_API_KEY"
    PINECONE_ENVIRONMENT="us-east-1"
    PICOVOICE_ACCESS_KEY="YOUR_PICOVOICE_KEY"
    OLLAMA_MODEL="llama3" 
    ```

5.  **Data Ingestion:** Run the ingestion script once to populate Pinecone (Ensure PDF files are in `backend/knowledge_base/`):
    ```bash
    python backend/rag_core/ingestion.py
    ```

***

## ▶️ Running the Application

### 1. Start the Backend Server (Terminal 1)

Open the terminal, activate the venv, navigate to the `backend` folder, and start the FastAPI server:

```bash
cd backend
.\venv\Scripts\activate
uvicorn main:app --reload
