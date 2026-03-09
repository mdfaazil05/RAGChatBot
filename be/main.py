import os
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredExcelLoader,
    TextLoader,
    WebBaseLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

app = FastAPI()

# -----------------------------
# CORS (for React frontend)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "./data"
VECTOR_DB_DIR = "./vector_db"

LLM_MODEL = "phi3"
EMBED_MODEL = "nomic-embed-text"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# -----------------------------
# Initialize Embeddings
# -----------------------------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# -----------------------------
# Initialize Vector DB
# -----------------------------
vector_db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)

# -----------------------------
# Initialize LLM
# -----------------------------
llm = Ollama(model=LLM_MODEL)

# -----------------------------
# Prompt Template
# -----------------------------
template = """
You are a highly intelligent Project Assistant helping developers.

Use ONLY the following context to answer the question.
If the answer is not found, say:
"I don't have this specific information in the project files."

Context:
{context}

Question:
{question}

Helpful Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# -----------------------------
# Build RAG Chain
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

# -----------------------------
# File Ingestion
# -----------------------------
@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)):

    file_path = os.path.join(DATA_DIR, file.filename)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)

        elif file.filename.endswith((".xlsx", ".xls")):
            loader = UnstructuredExcelLoader(file_path)

        else:
            loader = TextLoader(file_path)

        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        chunks = text_splitter.split_documents(docs)

        vector_db.add_documents(chunks)
        vector_db.persist()

        return {
            "status": "success",
            "chunks_indexed": len(chunks),
            "file": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# URL Ingestion
# -----------------------------
@app.post("/ingest-link")
async def ingest_link(url: str = Form(...)):

    try:
        loader = WebBaseLoader(url)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100
        )

        chunks = text_splitter.split_documents(docs)

        vector_db.add_documents(chunks)
        vector_db.persist()

        return {
            "status": "success",
            "url": url,
            "chunks_indexed": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Ask Question (RAG)
# -----------------------------
@app.post("/ask")
async def ask_question(question: str = Form(...)):

    try:
        result = qa_chain.invoke({"query": question})

        sources = list(set([
            doc.metadata.get("source", "Unknown")
            for doc in result["source_documents"]
        ]))

        return {
            "answer": result["result"],
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health():
    return {"status": "SmartBot backend running"}


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )