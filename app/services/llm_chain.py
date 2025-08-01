from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from app.services.vector_store import search_similar
from app.core.config import settings

def get_answer(query: str, domain: str = "general") -> str:
    llm = ChatOpenAI(temperature=0.1, model="gpt-4", openai_api_key=settings.OPENAI_API_KEY)
    relevant_docs = search_similar(query)
    docs = [d.page_content for d in relevant_docs]
    prompt = f"You are an expert in {domain}. Answer the following question based on the context:\n\nContext:\n{docs}\n\nQuestion: {query}"
    return llm.predict(prompt)
