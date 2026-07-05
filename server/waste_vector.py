import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

DOCS_FOLDER = os.path.join(SERVER_DIR, "docs", "waste")
DATABASE_LOCATION = os.path.join(SERVER_DIR, "chroma_db_waste")

# Fallback seed content so the feature works before real disposal-guideline
# PDFs are added to DOCS_FOLDER. Drop PDFs/DOCX/HTML in there and delete
# ./server/chroma_db_waste to rebuild with the real knowledge base.
SEED_TEXT = """
General waste sorting categories for kids:
- Recycling: clean plastic bottles, aluminum cans, glass jars, paper, cardboard. Rinse food residue first.
- Compost: fruit and vegetable scraps, coffee grounds, eggshells, plain paper towels.
- General waste (landfill): chip bags, wrappers, styrofoam, greasy pizza boxes, tissues.
- Hazardous waste: batteries, paint, chemicals, light bulbs - never put these in regular trash, take to a hazardous waste drop-off.
- E-waste: phones, chargers, electronics - take to an e-waste recycling center, never throw in regular trash.
- Sharps/medical waste: needles, medical supplies - use a designated sharps container, never regular trash.
"""


def _load_documents(folder: str) -> list:
    docs = []
    if not os.path.isdir(folder):
        return docs

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

        docs.extend(loader.load())

    return docs


def _build_vectorstore() -> Chroma:
    documents = _load_documents(DOCS_FOLDER)

    if not documents:
        documents = [Document(page_content=SEED_TEXT)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    return Chroma.from_documents(
        collection_name="Talk2Breathe-Waste",
        documents=chunks,
        embedding=embeddings,
        persist_directory=DATABASE_LOCATION,
    )


if os.path.exists(DATABASE_LOCATION):
    vector_store = Chroma(
        collection_name="Talk2Breathe-Waste",
        embedding_function=embeddings,
        persist_directory=DATABASE_LOCATION,
    )
else:
    vector_store = _build_vectorstore()

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 15},
)
