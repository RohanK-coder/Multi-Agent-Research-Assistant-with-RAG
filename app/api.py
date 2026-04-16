from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.graph import run_research
from app.models import ResearchRequest, ResearchResponse

app = FastAPI(title="Gemini Multi-Agent Research Assistant")


@app.get("/")
def root():
    return {"message": "Gemini Multi-Agent Research Assistant is running."}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    try:
        return run_research(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
