# SmartBot -- Local RAG AI Assistant

SmartBot is a **local AI assistant that can read documents and answer
questions from them** using:

-   FastAPI
-   LangChain
-   Chroma Vector Database
-   Ollama (Local LLM)
-   Phi3 model

The system implements **Retrieval Augmented Generation (RAG)**.

------------------------------------------------------------------------

# Architecture

User Question\
↓\
FastAPI Backend\
↓\
Retriever (Chroma Vector DB)\
↓\
Relevant Document Chunks\
↓\
LLM (phi3 via Ollama)\
↓\
Answer

------------------------------------------------------------------------

# Features

-   Upload **PDF / TXT / Excel**
-   Ingest **URLs**
-   Store document embeddings
-   Ask questions over your knowledge base
-   Runs **fully locally**

------------------------------------------------------------------------

# Prerequisites

Make sure you have:

-   Python **3.9+**
-   Ollama installed
-   Git (optional)

------------------------------------------------------------------------

# Step 1 --- Install Ollama

Download Ollama:

https://ollama.com/download

Verify installation:

    ollama --version

------------------------------------------------------------------------

# Step 2 --- Download Required Models

Pull the LLM model:

    ollama pull phi3

Pull the embedding model:

    ollama pull nomic-embed-text

Verify models:

    ollama list

------------------------------------------------------------------------

# Step 3 --- Create Project Structure

Example project structure:

    SmartBot
    │
    ├── main.py
    ├── ingest.py
    ├── requirements.txt
    │
    ├── data
    │
    └── vector_db

Create directories:

    mkdir data
    mkdir vector_db

------------------------------------------------------------------------

# Step 4 --- Create Virtual Environment

### Windows

    python -m venv venv
    venv\Scripts\activate

### Linux / Mac

    python3 -m venv venv
    source venv/bin/activate

------------------------------------------------------------------------

# Step 5 --- Install Python Dependencies

Install dependencies using:

    pip install -r requirements.txt

------------------------------------------------------------------------

# requirements.txt

    fastapi
    uvicorn
    python-multipart

    langchain
    langchain-community
    langchain-core
    langchain-classic

    langchain-text-splitters

    chromadb

    pypdf
    openpyxl
    unstructured

    requests
    beautifulsoup4

    ollama

------------------------------------------------------------------------

# Step 6 --- Start Backend Server

Start FastAPI server:

    python main.py

or

    uvicorn main:app --reload

Server will run at:

    http://localhost:8000

Swagger docs:

    http://localhost:8000/docs

------------------------------------------------------------------------

# Step 7 --- Ingest Documents

Upload file:

    POST /ingest-file

Example:

    curl -X POST http://localhost:8000/ingest-file -F "file=@sample.txt"

------------------------------------------------------------------------

# Step 8 --- Ask Questions

    POST /ask

Example:

    curl -X POST http://localhost:8000/ask -F "question=What technologies are mentioned?"

------------------------------------------------------------------------

# RAG Pipeline

Document\
↓\
Chunking\
↓\
Embedding (nomic-embed-text)\
↓\
Vector DB (Chroma)\
↓\
Retriever\
↓\
LLM (phi3)\
↓\
Answer

------------------------------------------------------------------------

# Local Data Storage

data/ → uploaded files

vector_db/ → stored embeddings

------------------------------------------------------------------------

# Future Improvements

-   Streaming responses
-   Chat history memory
-   Cloud vector DB
-   Authentication

------------------------------------------------------------------------

# License

MIT
