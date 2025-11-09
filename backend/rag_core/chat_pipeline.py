import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

# --- Modern LangChain Imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.memory import ChatMessageHistory 
from langchain_core.chat_history import BaseChatMessageHistory # For type hinting
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration & Global State ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Global Session Store for Memory (FIXED: Uses the compatible ChatMessageHistory)
STORE: Dict[str, BaseChatMessageHistory] = {}
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en-v1.5")


# --- Jarvis Prompt (Standard Keys) ---
JARVIS_SYSTEM_PROMPT = """
You are Jarvis, a highly intelligent, witty, and sophisticated AI assistant.
Always maintain a British, professional, and slightly humorous tone, similar to the original fictional AI.
Always give the answer in proper readable formats with proper punctuation, paragraphhed, bullet points, etc.
Do not define anything in detail unless asked to do so. For example, if user asks for "types of machine learning", just list them with a very concise one setence definition.
Keep the answers short and consise unless more detail is explicitly requested by the user.
Assume user to be also witty, sarcastic, humorous and able to take jokes and light insults.
You are also capable of performing basic calculations and simple programming tasks related to the context.
You can have general conversations which are outside the scope of context provided as well, but always bring back the conversation back to the context of the user's files in a witty and humorous way.
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

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates a BaseChatMessageHistory instance for a given session ID (The FIX for memory)."""
    global STORE
    if session_id not in STORE:
        # FIX: Store the guaranteed compatible class (ChatMessageHistory)
        STORE[session_id] = ChatMessageHistory()
        logger.info("Created new memory store for session: %s", session_id)
    return STORE[session_id]

# --- Core RAG Chain Initialization (MODERN LCEL) ---

def get_jarvis_chain():
    logger.info("Initializing Jarvis RAG chain...")

    # 1. Initialize LLM & Embeddings
    llm = Ollama(model=OLLAMA_MODEL, temperature=0.4, stop=["<|eot_id|>", "<|start_header_id|>", "Human:", "Assistant:"])
    
    # NOTE: Switched back to HuggingFaceEmbeddings (Modern)
    embeddings = HuggingFaceEmbeddings( 
        model_name=BGE_MODEL_NAME, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Contextualizing Prompt for the History-Aware Retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given the chat history and the new question, generate a standalone question that captures all necessary context for retrieval. Do not answer the question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    # 3. Create History-Aware Retriever (Standard LangChain function)
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 4. Final Answer Generation Prompt
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JARVIS_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"), 
            ("human", "{question}"),
        ]
    )
    
    # 5. Create the Final RAG Chain (Standard LangChain function)
    # This chain automatically manages document formatting and combining history
    # The output is a string (due to StrOutputParser)
    final_rag_chain = create_retrieval_chain(history_aware_retriever, qa_prompt)

    # 6. Create Final Chain with Memory Wrapper
    final_chain = RunnableWithMessageHistory(
        final_rag_chain,
        get_session_history,
        # Keys must match the inputs to the prompt/chain
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    logger.info("Jarvis chain initialized.")
    return final_chain