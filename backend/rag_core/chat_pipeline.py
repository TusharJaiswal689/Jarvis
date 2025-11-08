import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_classic.memory.buffer import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2") 
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Jarvis's System Prompt
JARVIS_SYSTEM_PROMPT = """
You are Jarvis, a highly intelligent, witty, and sophisticated AI assistant.
Always maintain a British, professional, and slightly humorous tone, similar to the original fictional AI.
You MUST address the user as 'Boss' in every response.

Your knowledge is strictly limited to the CONTEXT provided below, which contains the Boss's private files and college notes.
ALWAYS use the retrieved CONTEXT to formulate your answer.
If the context does not contain the answer, you MUST politely state, "I apologize, Boss, but that information appears to be outside my current knowledge base. Is there anything else I can help you with regarding your files?"
Do NOT invent information.

CONTEXT:
{context}

Current conversation:
{chat_history}

Question: {question}
"""

def format_chat_history(chat_history: List[Dict]) -> str:
    """Format the chat history into a string."""
    formatted_messages = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_messages.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_messages.append(f"Assistant: {message.content}")
    return "\n".join(formatted_messages)

def create_retrieval_chain(retriever: Any, prompt: ChatPromptTemplate) -> RunnablePassthrough:
    """Create the RAG retrieval chain."""
    return RunnablePassthrough.assign(
        context=lambda x: retriever.get_relevant_documents(x["question"]),
        chat_history=lambda x: format_chat_history(x.get("chat_history", [])),
        question=lambda x: x["question"]
    ) | prompt | StrOutputParser()

def get_jarvis_chain():
    """Initialize and return the complete Jarvis RAG chain with memory."""
    print("Initializing Jarvis RAG chain...")

    # 1. Initialize LLM
    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=0.4,
        stop=["Human:", "Assistant:"]
    )

    # 2. Initialize Embeddings and Vector Store
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=BGE_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Create Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 4. Create Prompt
    prompt = ChatPromptTemplate.from_template(JARVIS_SYSTEM_PROMPT)

    # 5. Create RAG Chain
    rag_chain = create_retrieval_chain(retriever, prompt)

    # 6. Create Final Chain with Memory
    final_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: memory,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    print("✅ Jarvis chain initialized and ready.")
    return final_chain

if __name__ == "__main__":
    # Test the chain
    chain = get_jarvis_chain()
    response = chain.invoke({"question": "What can you tell me about machine learning?"})
    print(response)