import os
from dotenv import load_dotenv

# 1. LangChain Document Loaders & Splitters
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 2. Embedding Model (Your Choice: BGE-Small)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# 3. Vector Store (Your Choice: Pinecone)
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # e.g., "us-east-1"

# Model and Splitter settings
KNOWLEDGE_DIR = "./knowledge_base"  # Relative path to your documents
CHUNK_SIZE = 1000                    # Optimal size for RAG chunk
CHUNK_OVERLAP = 200                  # Overlap helps maintain context
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE-Small has 384 output dimensions
BGE_DIMENSION = 384 

# --- Functions ---

def load_documents(directory: str) -> list[Document]:
    """Loads all PDF and TXT documents from the specified directory."""
    print("-> Loading documents...")
    
    # 1. Load PDF files using PyPDFLoader
    pdf_loader = DirectoryLoader(
        directory,
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader,
        recursive=True
    )
    documents = pdf_loader.load()
    
    # 2. Load TXT files (uncomment this if you have .txt notes)
    # txt_loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader, recursive=True)
    # documents.extend(txt_loader.load())
    
    print(f"   Loaded {len(documents)} source pages/documents.")
    if not documents:
        print("   WARNING: Document list is empty. Check the files in the knowledge_base folder.")
    return documents

def split_documents(documents: list[Document]) -> list[Document]:
    """Splits documents into smaller, semantically meaningful chunks."""
    print("-> Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} text chunks.")
    return chunks

def initialize_embeddings():
    """Initializes the BGE embedding model on CPU."""
    print(f"-> Initializing {BGE_MODEL_NAME} embedding model on CPU...")
    
    # Force running on CPU to save VRAM and use ample RAM
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True} 
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=BGE_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def ingest_to_pinecone(chunks: list[Document], embeddings: HuggingFaceBgeEmbeddings):
    """Initializes Pinecone and uploads documents/vectors."""
    print("-> Initializing Pinecone client...")
    
    # 1. Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # 2. Check and Create Index (if it doesn't exist)
    existing_indexes = [index.name for index in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"   Index '{PINECONE_INDEX_NAME}' not found. Creating new index...")
        
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=BGE_DIMENSION, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=PINECONE_ENVIRONMENT
            )
        )
        print("   Index created successfully.")
    
    # Rest of the function remains the same...
    
    # 3. Create Vector Store and Embed/Upload Data
    print(f"-> Starting ingestion of {len(chunks)} chunks into Pinecone. This may take a few minutes...")
    
    # This single call handles embedding the chunks and uploading them to Pinecone
    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    print("\n✅ Ingestion complete! Pinecone index is ready for RAG.")

# --- Main Execution ---

if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_DIR) or not os.listdir(KNOWLEDGE_DIR):
        print(f"ERROR: Knowledge directory '{KNOWLEDGE_DIR}' is empty or not found.")
        print("Please ensure you create the folder and add your college notes (PDFs) inside.")
    else:
        try:
            # 1. Load and Split
            raw_documents = load_documents(KNOWLEDGE_DIR)
            if raw_documents:
                text_chunks = split_documents(raw_documents)
                
                # 2. Initialize Embeddings
                bge_embeddings = initialize_embeddings()
                
                # 3. Ingest to Pinecone
                ingest_to_pinecone(text_chunks, bge_embeddings)

        except Exception as e:
            print(f"\nFATAL ERROR during ingestion. Check keys and logs.")
            print(f"Error details: {e}")
            
# Run this script using: python rag_core/ingestion.py