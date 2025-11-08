import requests
import json
import time

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{API_URL}/chat"
SESSION_ID = f"test_session_{int(time.time())}" # Unique session ID for memory test

# --- Test Queries ---
# Query 1: Tests RAG retrieval (must use your college notes)
QUERY_1 = "What are the main topics covered in the ML course and who is the professor?"

# Query 2: Tests Conversational Memory (references Q1)
QUERY_2 = "Can you give me the due date for the first assignment in that course?" 


def post_query(query: str, session_id: str):
    """Sends a POST request to the /chat endpoint."""
    payload = {
        "query": query,
        "session_id": session_id
    }
    
    print(f"\n[QUERY] Boss: {query}")
    
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json=payload, # Send payload as JSON body
            timeout=120 # Allow 30 seconds for Ollama/LLM response
        )
        response.raise_for_status() # Raises exception for 4xx or 5xx errors
        
        # Parse the JSON response
        data = response.json()
        
        # Print the response, confirming the Jarvis tone
        print(f"[JARVIS] {data.get('answer', 'Error: No answer received.')}")
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to connect to API or request timed out: {e}")
        
    except json.JSONDecodeError:
        print(f"[ERROR] Received non-JSON response: {response.text[:100]}...")


if __name__ == "__main__":
    print(f"--- Running Jarvis Backend Test (Session: {SESSION_ID}) ---")
    
    # 1. Test RAG Retrieval (Q1)
    post_query(QUERY_1, SESSION_ID)
    
    # 2. Test Conversational Memory (Q2)
    # The LLM should use the history of Q1 to understand "that course."
    post_query(QUERY_2, SESSION_ID)
    
    print("\n--- Test Finished ---")