import os
from dotenv import load_dotenv

# 1. LangChain Document Loaders & Splitters
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 2. Embedding Model (MODERN FIX)
from langchain_huggingface import HuggingFaceEmbeddings # ✅ MODERN IMPORT

# 3. Vector Store (Pinecone)
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# --- Configuration ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

KNOWLEDGE_DIR = "./knowledge_base"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_DIMENSION = 384 

# --- Functions ---

def load_documents(directory: str) -> list[Document]:
    """Loads all PDF and TXT documents from the specified directory."""
    print("-> Loading documents...")
    
    pdf_loader = DirectoryLoader(
        directory, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True
    )
    documents = pdf_loader.load()
    
    print(f"   Loaded {len(documents)} source pages/documents.")
    if not documents:
        print("   WARNING: Document list is empty. Check the files in the knowledge_base folder.")
    return documents

def split_documents(documents: list[Document]) -> list[Document]:
    """Splits documents into smaller, semantically meaningful chunks."""
    print("-> Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} text chunks.")
    return chunks

def initialize_embeddings():
    """Initializes the BGE embedding model on CPU."""
    print(f"-> Initializing {BGE_MODEL_NAME} embedding model on CPU...")
    
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True} 
    
    # ✅ FIX: Use the updated class from langchain-huggingface
    embeddings = HuggingFaceEmbeddings( 
        model_name=BGE_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def ingest_to_pinecone(chunks: list[Document], embeddings: HuggingFaceEmbeddings):
    """Initializes Pinecone and uploads documents/vectors."""
    print("-> Initializing Pinecone client...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    existing_indexes = [index.name for index in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"   Index '{PINECONE_INDEX_NAME}' not found. Creating new index...")
        
        pc.create_index(
            name=PINECONE_INDEX_NAME, dimension=BGE_DIMENSION, metric='cosine',
            spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
        )
        print("   Index created successfully.")
    
    print(f"-> Starting ingestion of {len(chunks)} chunks into Pinecone. This may take a few minutes...")
    
    PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=PINECONE_INDEX_NAME,
    )
    print("\n✅ Ingestion complete! Pinecone index is ready for RAG.")

# --- Main Execution ---

if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_DIR) or not os.listdir(KNOWLEDGE_DIR):
        print(f"ERROR: Knowledge directory '{KNOWLEDGE_DIR}' is empty or not found.")
        print("Please ensure you create the folder and add your college notes (PDFs) inside.")
    else:
        try:
            raw_documents = load_documents(KNOWLEDGE_DIR)
            if raw_documents:
                text_chunks = split_documents(raw_documents)
                bge_embeddings = initialize_embeddings() 
                ingest_to_pinecone(text_chunks, bge_embeddings)
        except Exception as e:
            print(f"\nFATAL ERROR during ingestion. Check keys and logs.")
            print(f"Error details: {e}")