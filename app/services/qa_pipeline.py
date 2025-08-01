# app/services/qa_pipeline.py
import aiohttp
import tempfile
from typing import List
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "<your-openai-key>"  # or use a config file

async def run_qa_pipeline(pdf_url: str, questions: List[str]) -> List[str]:
    # 1. Download PDF asynchronously
    async with aiohttp.ClientSession() as session:
        async with session.get(pdf_url) as resp:
            if resp.status != 200:
                raise ValueError("Failed to fetch PDF from URL")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(await resp.read())
                local_pdf_path = tmp_file.name

    # 2. Load and chunk PDF
    loader = PyMuPDFLoader(local_pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # 3. Embed & index
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. QA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

    # 5. Get answers
    answers = [qa.run(q) for q in questions]
    return answers
