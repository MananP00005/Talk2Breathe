from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import shutil

DataFrame = pd.read_csv('Smoking_CSV.csv')
DataFrame = DataFrame.dropna(subset=['text', 'location'])
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
Database_location = "./chroma_db"

if os.path.exists(Database_location):
    shutil.rmtree(Database_location)

documents = []
ids = []
for i, row in DataFrame.iterrows():
    doc = Document(
        page_content=row['text'],
        metadata={"source": row['location']},
        id=str(i)
    )
    ids.append(str(i))
    documents.append(doc)

vector_store = Chroma(
    collection_name="Smoking_CSV",
    embedding_function=embeddings,
    persist_directory=Database_location
)

vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 15}
)