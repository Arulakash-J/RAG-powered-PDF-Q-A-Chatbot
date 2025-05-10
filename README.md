# PDF Assistant - RAG-powered PDF Q&A Chatbot

PDF Assistant is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content. The application leverages state-of-the-art language models to provide accurate answers based on the document content.

## Features

- **PDF Upload**: Upload PDF documents to be processed and queried
- **Conversational Interface**: Ask questions and receive answers in a chat-like interface
- **Source Highlighting**: View the source chunks of text that were used to generate the answer
- **PDF Download**: Download the uploaded PDF for reference

## Technologies Used

- **Frontend**: Streamlit
- **Vector Store**: Pinecone
- **Text Embedding**: Sentence Transformers (all-mpnet-base-v2)
- **Language Model**: Groq (llama3-70b-8192)
- **PDF Processing**: PyPDF2, LangChain

## Live Demo

[Access the live application here](arulakashragpdfsystem.streamlit.app) <!-- Replace with your deployed URL -->

## Installation & Setup

### Prerequisites

- Python 3.8+
- Pinecone API Key
- Groq API Key

### Local Setup

1. **Clone the repository**

   ```
   git clone https://github.com/Arulakash-J/RAG-powered-PDF-Q-A-Chatbot.git

   ```

2. **Create virtual environment**

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root directory (copy from `.env.example`):

   ```
   # API Keys
   GROQ_API_KEY=your_groq_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here

   # Application Configuration
   APP_TITLE=PDF Assistant
   PINECONE_INDEX_NAME=assestment
   EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=100
   FILE_SIZE_LIMIT_MB=1
   LLM_MODEL=llama3-70b-8192
   ```


5. **Run the application**

   ```
   streamlit run app.py
   ```
