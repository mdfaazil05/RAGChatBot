from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    WebBaseLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


def process_docs(file_path):

    print("Loading document...")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif "http" in file_path:
        loader = WebBaseLoader(file_path)

    elif file_path.endswith((".xlsx", ".xls")):
        loader = UnstructuredExcelLoader(file_path)

    else:
        loader = TextLoader(file_path)

    docs = loader.load()

    print(f"Loaded {len(docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="../vector_db"
    )

    vector_db.persist()

    print("Documents indexed successfully!")


if __name__ == "__main__":
    process_docs("../data/mdDes.txt")