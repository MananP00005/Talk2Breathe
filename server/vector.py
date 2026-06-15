from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, Docx2txtLoader
import os
import shutil

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
Database_location = "./chroma_db"

if os.path.exists(Database_location):
    shutil.rmtree(Database_location)

documents = []
ids = []

docs_folder = "./server/docs"
doc_id = 0

for filename in os.listdir(docs_folder):
    filepath = os.path.join(docs_folder, filename)

    if filename.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filename.endswith(".html"):
        loader = BSHTMLLoader(filepath)
    elif filename.endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    else:
        continue

    loaded = loader.load()
    for page in loaded:
        ids.append(str(doc_id))
        documents.append(page)
        doc_id += 1

vector_store = Chroma(
    collection_name="Talk2Breathe",
    embedding_function=embeddings,
    persist_directory=Database_location
)

vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 15}
)