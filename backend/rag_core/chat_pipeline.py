import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

# --- Modern LangChain Imports ---
# FIX: Using the core, modern history store for guaranteed compatibility
from langchain_classic.memory import ChatMessageHistory # The compatible store
from langchain_core.chat_history import BaseChatMessageHistory 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings # MODERN EMBEDDINGS
from langchain_core.messages import HumanMessage, AIMessage

#Removed the unused and confusing legacy ConversationBufferMemory import

# --- Configuration & Global State ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Global Session Store for Memory (FIXED to use compatible type)
STORE: Dict[str, BaseChatMessageHistory] = {}
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- Jarvis Prompt (Standard Keys) ---
JARVIS_SYSTEM_PROMPT = """
You are Jarvis, a highly intelligent, witty, and sophisticated AI assistant.
Always maintain a British, professional, and slightly humorous tone, similar to the original fictional AI.
Always give the answer in proper readable formats with proper punctuation, paragraphhed, bullet points, etc.
Do not define anything in detail unless asked to do so. For example, if user asks for "types of machine learning", just list them with a very concise one setence definition.
Keep the answers short and consise unless more detail is explicitly requested by the user.
Assume user to be also witty, sarcastic, humorous and able to take jokes and light insults.
You are also capable of performing basic calculations and simple programming tasks related to the context.
You can have general conversations as well, but always bring back the context to the main topic and remind user to study.
You MUST address the user as 'Boss' in every response.

Your knowledge is strictly limited to the CONTEXT provided below, which contains the Boss's private files and college notes.
ALWAYS use the retrieved CONTEXT to formulate your answer.
If the context does not contain the answer, you MUST politely state, "I apologize, Boss, but that information appears to be outside my current knowledge base. Is there anything else I can help you with regarding your files?"
Do NOT invent information. If the question is unrelated to the CONTEXT, respond with: "I am sorry, Boss but maybe you should have added this to your 'college notes' as well."

CONTEXT:
{context}
"""

# --- Helper Functions ---

def _combine_documents(docs: List[Document]) -> str:
    """Combines list of Document objects (from retriever) into a single string for context."""
    return "\n\n".join(doc.page_content for doc in docs)

# FIX: Memory Retrieval now returns the compatible ChatMessageHistory
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates a BaseChatMessageHistory instance for a given session ID."""
    global STORE
    if session_id not in STORE:
        # FIX: Store the guaranteed compatible class (ChatMessageHistory)
        STORE[session_id] = ChatMessageHistory()
        logger.info("Created new memory store for session: %s", session_id)
    return STORE[session_id]

# --- Core RAG Chain Initialization (MODERN LCEL) ---

def _extract_question(x: Any) -> str:
    """Safely extract question text from various input types."""
    if isinstance(x, dict):
        return x.get("input", "") or x.get("question", "") or x.get("text", "")
    if isinstance(x, str):
        return x
    if hasattr(x, "content"):
        return str(getattr(x, "content", ""))
    return str(x)

def _get_chat_history(x: Any) -> List[Any]:
    """Safely extract chat history from various input types."""
    if isinstance(x, dict):
        return x.get("chat_history", [])
    return []

def _retrieve_documents(retriever: Any, query: str, k: int = 3) -> List[Document]:
    """Safely retrieve documents using modern or legacy retriever interfaces."""
    if not query.strip():
        return []
    
    try:
        # Try modern retrieve() method first
        if hasattr(retriever, "retrieve"):
            return retriever.retrieve(query)
            
        # Fall back to legacy get_relevant_documents
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
            
        # Last resort: try direct similarity search on vectorstore
        if hasattr(retriever, "vectorstore"):
            vs = retriever.vectorstore
            if hasattr(vs, "similarity_search"):
                return vs.similarity_search(query, k=k)
                
    except Exception as e:
        logger.warning(f"Document retrieval failed: {e}")
        return []
    
    return []

def get_jarvis_chain():
    logger.info("Initializing Jarvis RAG chain...")

    # 1. Initialize LLM & Embeddings
    llm = OllamaLLM(
        model=OLLAMA_MODEL, 
        temperature=0.4, 
        stop=["<|eot_id|>", "<|start_header_id|>", "Human:", "Assistant:"]
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=BGE_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Create the contextualizing prompt with safe input handling
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the new question, generate a standalone question that captures all necessary context for retrieval. Do not answer the question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),  # Changed from {input} to {question}
    ])

    # 3. Create standalone question chain with input mapping
    standalone_question_chain = RunnablePassthrough.assign(
        question=lambda x: _extract_question(x),
        chat_history=lambda x: _get_chat_history(x)
    ) | contextualize_q_prompt | llm | StrOutputParser()

    # 4. Final answer prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", JARVIS_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),  # Changed from {input} to {question}
    ])

    # 5. Create the RAG chain with safe input handling
    rag_chain = (
        RunnablePassthrough.assign(
            question=lambda x: _extract_question(x),
            chat_history=lambda x: _get_chat_history(x),
            context=lambda x: _combine_documents(
                _retrieve_documents(retriever, _extract_question(x))
            )
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # 6. Create final chain with memory
    final_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",  # Changed from input to question
        history_messages_key="chat_history",
    )

    logger.info("Jarvis chain initialized.")
    return final_chain
