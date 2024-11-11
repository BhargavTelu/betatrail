from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any
from langchain_core.documents import Document
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    raise ImportError(
        "Could not import faiss. Please install it with `pip install faiss-cpu` "
        "or `pip install faiss-gpu` depending on your system."
    )

# Initialize text splitter and embedding function at module level
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding_function = OpenAIEmbeddings()

# Initialize vectorstore
PERSIST_DIRECTORY = "./faiss_db"

def initialize_vectorstore() -> FAISS:
    """Initialize or load the FAISS vectorstore."""
    try:
        if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
            return FAISS.load_local(PERSIST_DIRECTORY, embedding_function, allow_dangerous_deserialization=True )
        
        # Create a dummy document to initialize the vectorstore
        dummy_texts = ["initialization document"]
        dummy_embeddings = embedding_function.embed_documents(dummy_texts)
        
        # Initialize with dummy document
        vectorstore = FAISS.from_texts(dummy_texts, embedding_function)
        
        # Save the initialized vectorstore
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        vectorstore.save_local(PERSIST_DIRECTORY)
        
        return vectorstore
    except Exception as e:
        logger.error(f"Error initializing vectorstore: {e}")
        raise

vectorstore = initialize_vectorstore()

def save_vectorstore() -> bool:
    """Save the current state of the vectorstore."""
    try:
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        vectorstore.save_local(PERSIST_DIRECTORY)
        return True
    except Exception as e:
        logger.error(f"Error saving vectorstore: {e}")
        return False

def load_and_split_document(file_path: str) -> List[Document]:
    """Load and split a document into chunks."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_faiss(file_path: str, file_id: int) -> bool:
    """Index a document to the vectorstore. Maintains original function name for compatibility."""
    try:
        splits = load_and_split_document(file_path)
        
        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
        
        # Create new vectorstore with the documents
        global vectorstore
        new_vectorstore = FAISS.from_documents(splits, embedding_function)
        
        # Merge with existing vectorstore
        vectorstore.merge_from(new_vectorstore)
        
        # Save the updated vectorstore
        return save_vectorstore()
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        return False

def delete_doc_from_faiss(file_id: int) -> bool:
    """Delete a document from the vectorstore. Maintains original function name for compatibility."""
    try:
        global vectorstore
        # Get all documents
        all_docs = list(vectorstore.docstore._dict.values())
        
        # Filter out documents with matching file_id
        remaining_docs = [doc for doc in all_docs 
                         if doc.metadata.get('file_id') != file_id]
        
        # If no documents were filtered out, the file_id doesn't exist
        if len(remaining_docs) == len(all_docs):
            logger.warning(f"No documents found with file_id {file_id}")
            return False
        
        # Create new vectorstore with remaining documents
        vectorstore = FAISS.from_documents(remaining_docs, embedding_function)
        
        # Save the updated vectorstore
        return save_vectorstore()
    except Exception as e:
        logger.error(f"Error deleting document with file_id {file_id}: {e}")
        return False

def search_documents(query: str, k: int = 4) -> List[Document]:
    """Search for similar documents."""
    try:
        return vectorstore.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []