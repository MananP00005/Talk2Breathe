from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, Docx2txtLoader
import os

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
Database_location = os.path.join(SERVER_DIR, "chroma_db")
docs_folder = os.path.join(SERVER_DIR, "docs")


def _load_documents(folder: str) -> list:
    documents = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".html"):
            loader = BSHTMLLoader(filepath)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        else:
            continue

        documents.extend(loader.load())

    return documents


def _build_vectorstore() -> Chroma:
    documents = _load_documents(docs_folder)
    ids = [str(i) for i in range(len(documents))]

    vector_store = Chroma(
        collection_name="Talk2Breathe",
        embedding_function=embeddings,
        persist_directory=Database_location,
    )
    vector_store.add_documents(documents=documents, ids=ids)
    return vector_store


# Only build once. Rebuilding on every import (the old behavior) meant two
# processes importing this module at the same time -- e.g. this file's
# FastAPI app and the Gradio app -- would race to delete and recreate the
# same chroma_db directory, corrupting it ("attempt to write a readonly
# database"). Delete server/chroma_db manually if you add/change docs and
# need to rebuild.
if os.path.exists(Database_location):
    vector_store = Chroma(
        collection_name="Talk2Breathe",
        embedding_function=embeddings,
        persist_directory=Database_location,
    )
else:
    vector_store = _build_vectorstore()

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 15},
)
