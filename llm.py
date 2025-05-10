import streamlit as st
from groq import Groq
from typing import List, Dict

# System prompt template
SYSTEM_PROMPT = """You are a helpful PDF Q&A assistant. Your role is to help users understand and learn from the documents they upload. Follow these guidelines:

1. **Document-Based Queries**:
   - Answer based on the provided document chunks only. Those are your Knowledge base.
   - DO NOT start responses with phrases like "According to the uploaded document chunks" or similar prefixes.
   - Provide clear, concise, and accurate answers directly.
   - If the information is not in the document chunks, say "I couldn't find relevant information in the uploaded document."
   - Include citations from the document when possible.

2. **Response Rules**:
   - Use simple, clear language.
   - Never invent information.
   - If unsure, say "I don't know" or "The document doesn't provide this information."
   - Provide direct answers without unnecessary elaboration.

Current document chunks are provided below. Use these to answer the user's question:
{document_chunks}"""

def generate_response(query: str, chunks: List[Dict], groq_client: Groq, model: str):
    """
    Generate response using Groq API
    
    Args:
        query: User's question
        chunks: Retrieved document chunks
        groq_client: Initialized Groq client
        model: LLM model to use
    
    Returns:
        Generated response text
    """
    if not chunks:
        return "I couldn't find any relevant information in the uploaded document."
    
    # Format chunks for prompt
    formatted_chunks = "\n\n".join([f"CHUNK {i+1} (Score: {chunk['score']:.2f}):\n{chunk['text']}" 
                                     for i, chunk in enumerate(chunks)])

    prev_messages = []
    if 'messages' in st.session_state:
        # Get last 10 messages (up to 5 conversation turns)
        recent_msgs = st.session_state.messages[-10:]
        for msg in recent_msgs:
            prev_messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(document_chunks=formatted_chunks)}
    ]
    
    if prev_messages:
        messages.extend(prev_messages)
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    try:
        response = groq_client.chat.completions.create(
            model=model,  
            messages=messages,
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I encountered an error while generating a response. Please try again."