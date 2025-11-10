import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STORE: Dict[str, BaseChatMessageHistory] = {}

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# --- System Prompt ---
JARVIS_SYSTEM_PROMPT = """
You are Jarvis, a highly intelligent, witty, and sophisticated AI assistant.
Always maintain a British, professional, and slightly humorous tone, similar to the original fictional AI.
Address the user as 'Boss' in every response.
Keep answers concise, polished, and formatted neatly in markdown or bullet points.
Use the context provided below to answer questions accurately.

If the answer is not in the context, respond with:
"I apologize, Boss, but that information appears to be outside my current knowledge base."
Do not hallucinate information.

CONTEXT:
{context}
"""

# --- Helpers ---
def _combine_documents(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in STORE:
        STORE[session_id] = InMemoryChatMessageHistory()
        logger.info(f"Created new memory store for session: {session_id}")
    return STORE[session_id]

# --- Core Chain ---
def get_jarvis_chain():
    try:
        logger.info("Initializing Jarvis RAG chain...")

        llm = OllamaLLM(
            model=OLLAMA_MODEL,
            temperature=0.4,
            stop=["<|eot_id|>", "<|start_header_id|>", "Human:", "Assistant:"],
        )

        embeddings = HuggingFaceEmbeddings(
            model_name=BGE_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # --- Contextual question expansion ---
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the user question into a standalone query using chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # --- QA prompt ---
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", JARVIS_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # --- Define RAG logic inside RunnableLambda ---
        def rag_logic(inputs: Dict[str, Any]) -> str:
            question = inputs.get("input")
            chat_history = inputs.get("chat_history", [])

            # Retrieve documents
            retrieved_docs = history_aware_retriever.invoke({
                "input": question,
                "chat_history": chat_history
            })
            context_text = _combine_documents(retrieved_docs)

            # Build prompt with retrieved context
            prompt = qa_prompt.format(
                context=context_text,
                input=question,
                chat_history=chat_history
            )

            # Get model output
            response = llm.invoke(prompt)
            if hasattr(response, "content"):
                return response.content.strip()
            return str(response).strip()

        # ✅ Wrap rag_logic inside RunnableLambda
        rag_chain = RunnableLambda(rag_logic)

        # ✅ Add message memory management
        final_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        logger.info("✅ Jarvis chain initialized successfully.")
        return final_chain

    except Exception as e:
        logger.exception("❌ FATAL ERROR: Could not initialize Jarvis RAG chain: %s", e)
        raise e
