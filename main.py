import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import pinecone
from groq import Groq
from dotenv import load_dotenv

from app import init_session_state, create_ui, create_sidebar, create_chat_interface
from llm import generate_response

def initialize_components():
    """Initialize and cache all required components"""
    try:
        client = Groq(api_key=GROQ_API_KEY)

        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=768,  
                metric="cosine", 
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")  
            )
        
        # Connect to the index
        index = pc.Index(PINECONE_INDEX_NAME)
        return client, tokenizer, model, index
    
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()  

def main():
    load_dotenv()

    global APP_TITLE, PINECONE_INDEX_NAME, EMBEDDING_MODEL_NAME
    global CHUNK_SIZE, CHUNK_OVERLAP, GROQ_API_KEY, PINECONE_API_KEY
    global FILE_SIZE_LIMIT_MB, LLM_MODEL
    
    # Configuration from environment variables
    APP_TITLE = os.getenv("APP_TITLE", "PDF Assistant")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "assessment")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    FILE_SIZE_LIMIT_MB = int(os.getenv("FILE_SIZE_LIMIT_MB", 1))
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3-70b-8192")
    
    # Validate essential environment variables
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    
    # Initialize session state
    init_session_state()
    
    # Create UI
    create_ui(APP_TITLE, FILE_SIZE_LIMIT_MB)
    
    # Initialize components with Streamlit caching
    groq_client, tokenizer, embedding_model, pinecone_index = st.cache_resource(initialize_components)()
    
    # Create sidebar with file upload functionality
    uploaded_file = create_sidebar(
        tokenizer, 
        embedding_model, 
        pinecone_index, 
        CHUNK_SIZE, 
        CHUNK_OVERLAP, 
        FILE_SIZE_LIMIT_MB
    )
    
    # Create chat interface
    create_chat_interface(
        groq_client, 
        tokenizer, 
        embedding_model, 
        pinecone_index, 
        LLM_MODEL,
        generate_response
    )

if __name__ == "__main__":
    main()