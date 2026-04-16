from __future__ import annotations

from operator import add
from typing import Annotated, Literal, Optional, TypedDict
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    source_id: str
    source_type: Literal["web", "pdf"]
    title: str
    excerpt: str
    url: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    published_at: Optional[str] = None
    relevance_score: float = 0.0


class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research question")
    use_web: bool = True
    use_pdf: bool = True


class ResearchResponse(BaseModel):
    query: str
    answer: str
    citations: list[Evidence]
    notes: list[str] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    query: str
    use_web: bool
    use_pdf: bool
    plan: Annotated[list[str], add]
    notes: Annotated[list[str], add]
    web_evidence: list[dict]
    pdf_evidence: list[dict]
    evidence: list[dict]
    draft_answer: str
    answer: str