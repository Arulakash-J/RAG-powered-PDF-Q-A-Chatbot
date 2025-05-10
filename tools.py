import torch
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64
import streamlit as st
from typing import List, Dict

def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings to create sentence embeddings
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_embeddings(text, tokenizer, model):
    """
    Generate embeddings for the given text using the provided tokenizer and model
    """
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask']).squeeze().tolist()

def extract_text_from_pdf(uploaded_file):
    """
    Extract text content from a PDF file
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None

def process_pdf_document(pdf_file, tokenizer, model, pinecone_index, chunk_size, chunk_overlap):
    """
    Process PDF document and store chunks in Pinecone
    """
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_file)
    if not text:
        return False, 0
    
    document_name = pdf_file.name
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    
    # Generate embeddings and store in Pinecone
    vectors = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{document_name}_chunk_{i}"
        vectors.append({
            "id": chunk_id,
            "values": generate_embeddings(chunk, tokenizer, model),
            "metadata": {
                "text": chunk,
                "source": document_name,
                "chunk_id": i
            }
        })
    
    # Delete existing vectors for this document if any
    existing_ids = [f"{document_name}_chunk_{i}" for i in range(len(chunks))]
    try:
        pinecone_index.delete(ids=existing_ids)
    except:
        pass  # Ignore if no existing vectors
    
    # Insert new vectors
    pinecone_index.upsert(vectors=vectors)
    
    return True, len(chunks)

def retrieve_relevant_chunks(query, pinecone_index, tokenizer, model, top_k=5):
    """
    Retrieve relevant chunks based on query
    """
    query_embedding = generate_embeddings(query, tokenizer, model)
    
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    if not results.matches:
        return []
    
    chunks = []
    for match in results.matches:
        chunks.append({
            "text": match.metadata['text'],
            "score": match.score,
            "source": match.metadata['source'],
            "chunk_id": match.metadata['chunk_id']
        })
    
    return chunks

def highlight_matching_chunks(response, chunks):
    """
    Extract and highlight chunks that match the response
    """
    highlighted_chunks = []
    
    for chunk in chunks:
        chunk_text = chunk["text"]
        chunk_source = chunk["source"]
        chunk_id = chunk["chunk_id"]
        
        words = set(chunk_text.lower().split())
        significant_words = [w for w in words if len(w) > 5]  # Only consider words longer than 5 chars
        
        # Calculate overlap between response and significant words
        if significant_words:
            matches = sum(1 for word in significant_words if word in response.lower())
            match_ratio = matches / len(significant_words)
            
            if match_ratio > 0.2:  # If more than 20% of significant words appear in response
                highlighted_chunks.append({
                    "text": chunk_text,
                    "source": chunk_source,
                    "chunk_id": chunk_id,
                    "match_score": match_ratio
                })
    
    # Sort by match score
    highlighted_chunks.sort(key=lambda x: x["match_score"], reverse=True)
    return highlighted_chunks[:3]

def get_pdf_download_link(file_name):
    """
    Generate a link allowing the user to download the uploaded PDF file
    """
    with open(file_name, "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Download PDF</a>'
    return href