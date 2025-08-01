from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
from PyPDF2 import PdfReader
import json
import os
import tiktoken
import numpy as np

# --- Initialize ---
app = FastAPI()
client = OpenAI(api_key="your-openai-api-key")  # Replace with your actual key
MODEL = "gpt-4-1106-preview"
ENCODING = tiktoken.encoding_for_model(MODEL)

# --- PDF Text Extraction ---
def read_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

# --- Token-Based Text Chunking ---
def chunk_text(text: str, max_tokens: int = 500) -> list:
    words = text.split()
    chunks, current, token_count = [], [], 0

    for word in words:
        word_tokens = len(ENCODING.encode(word + " "))
        if token_count + word_tokens > max_tokens:
            chunks.append(" ".join(current))
            current, token_count = [], 0
        current.append(word)
        token_count += word_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks

# --- Embedding Generation ---
def get_embedding(text: str) -> list:
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# --- Embedding Search ---
def semantic_search(chunks: list, query: str, top_k: int = 3) -> list:
    query_emb = get_embedding(query)
    chunk_embs = [get_embedding(chunk) for chunk in chunks]
    scores = [np.dot(query_emb, emb) for emb in chunk_embs]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# --- Endpoint ---
@app.post("/hackrx/run")
async def answer_from_pdf(
    document: UploadFile = File(...),
    questions: str = Form(...)
):
    try:
        question_list = json.loads(questions)
        text = read_pdf(document)
        chunks = chunk_text(text)

        answers = []
        for question in question_list:
            relevant_chunks = semantic_search(chunks, question)
            context = "\n".join(relevant_chunks)

            prompt = f"""Answer the question strictly based on the context below.
If the answer is not in the context, say 'Not mentioned in the document.'

Context:
{context}

Question: {question}
Answer:"""

            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            answers.append(response.choices[0].message.content.strip())

        return JSONResponse(content={"answers": answers})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
