from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from app.core.config import settings
import os

embedding = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

def index_chunks(chunks: list, persist_path: str = "faiss_index"):
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    vectordb = FAISS.from_texts(texts, embedding, metadatas=metadatas)
    vectordb.save_local(persist_path)

def search_similar(query: str, k: int = 5, persist_path: str = "faiss_index"):
    vectordb = FAISS.load_local(persist_path, embedding)
    return vectordb.similarity_search(query, k=k)
