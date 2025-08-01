# app/api/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
from app.services.qa_pipeline import run_qa_pipeline

router = APIRouter()

class QARequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

@router.post("/run", response_model=QAResponse)
async def qa_endpoint(payload: QARequest):
    try:
        answers = await run_qa_pipeline(payload.documents, payload.questions)
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
