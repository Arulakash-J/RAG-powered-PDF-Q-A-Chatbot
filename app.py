import streamlit as st
import os
from tools import process_pdf_document, retrieve_relevant_chunks, highlight_matching_chunks, get_pdf_download_link

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    if 'document_name' not in st.session_state:
        st.session_state.document_name = None
    if 'highlighted_chunks' not in st.session_state:
        st.session_state.highlighted_chunks = []

def create_ui(app_title, file_size_limit_mb):
    """Create the main UI components"""
    st.set_page_config(page_title=app_title, page_icon="📄", layout="wide")
    st.title(f"{app_title} 📄")
    st.markdown("Upload a PDF and ask questions about its content!")

def create_sidebar(tokenizer, embedding_model, pinecone_index, chunk_size, chunk_overlap, file_size_limit_mb):
    """Create the sidebar upload functionality"""
    with st.sidebar:
        st.header("📁 Document Upload")

        uploaded_file = st.file_uploader(
            "Upload PDF", 
            type=["pdf"],
            help=f"Maximum file size: {file_size_limit_mb}MB"
        )
        
        # Display the file size limit explicitly
        st.caption(f"Limit {file_size_limit_mb}MB per file • PDF")

        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024) 
            if file_size_mb > file_size_limit_mb:
                st.error(f"File size ({file_size_mb:.1f} MB) exceeds the {file_size_limit_mb} MB limit. Please upload a smaller file.")
                st.session_state.document_loaded = False
                st.session_state.document_name = None
            elif st.session_state.document_name != uploaded_file.name:
                st.session_state.document_loaded = False
                st.session_state.document_name = uploaded_file.name
            
            if not st.session_state.document_loaded:
                with st.spinner("Processing document..."):
                    success, chunk_count = process_pdf_document(
                        uploaded_file, tokenizer, embedding_model, pinecone_index, chunk_size, chunk_overlap
                    )
                    if success:
                        st.session_state.document_loaded = True
                        st.success(f"✅ Document '{uploaded_file.name}' processed into {chunk_count} chunks!")

                        with open(uploaded_file.name, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        st.markdown(get_pdf_download_link(uploaded_file.name), unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed to process document")
            else:
                st.success(f"✅ Document '{uploaded_file.name}' is loaded and ready for questions!")

                if os.path.exists(uploaded_file.name):
                    st.markdown(get_pdf_download_link(uploaded_file.name), unsafe_allow_html=True)
        
        return uploaded_file

def create_chat_interface(groq_client, tokenizer, embedding_model, pinecone_index, llm_model, generate_response_func):
    """Create the chat interface for Q&A"""
    st.header("💬 Ask Questions")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "highlighted_chunks" in message:
                with st.expander("View Source Chunks"):
                    for i, chunk in enumerate(message["highlighted_chunks"]):
                        st.markdown(f"**Source Chunk {i+1}** (from {chunk['source']}):")
                        st.text(chunk["text"])
    
    # User input
    if user_query := st.chat_input("Ask a question about the document..."):
        if not st.session_state.document_loaded:
            with st.chat_message("assistant"):
                st.markdown("⚠️ Please upload a document first before asking questions.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("user"):
                st.markdown(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chunks = retrieve_relevant_chunks(
                        user_query, pinecone_index, tokenizer, embedding_model
                    )

                    response = generate_response_func(user_query, chunks, groq_client, llm_model)

                    st.markdown(response)

                    highlighted_chunks = highlight_matching_chunks(response, chunks)
                    
                    if highlighted_chunks:
                        with st.expander("View Source Chunks"):
                            for i, chunk in enumerate(highlighted_chunks):
                                st.markdown(f"**Source Chunk {i+1}** (from {chunk['source']}):")
                                st.text(chunk["text"])
            
            # Add assistant message with highlighted chunks to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "highlighted_chunks": highlighted_chunks
            })