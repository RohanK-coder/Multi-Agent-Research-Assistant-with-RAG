from __future__ import annotations

from typing import Any

from duckduckgo_search import DDGS

from app.config import settings
from app.models import Evidence


def _tavily_search(query: str, max_results: int) -> list[dict[str, Any]]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query=query, max_results=max_results)
    return response.get("results", [])


def _duckduckgo_search(query: str, max_results: int) -> list[dict[str, Any]]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    normalized = []
    for r in results:
        normalized.append(
            {
                "title": r.get("title") or "Untitled",
                "url": r.get("href") or r.get("url"),
                "content": r.get("body") or r.get("snippet") or "",
                "published_at": None,
            }
        )
    return normalized


def search_web(query: str, max_results: int = 5) -> list[Evidence]:
    if settings.tavily_api_key:
        raw_results = _tavily_search(query, max_results=max_results)
    else:
        raw_results = _duckduckgo_search(query, max_results=max_results)

    evidence: list[Evidence] = []
    for idx, item in enumerate(raw_results, start=1):
        evidence.append(
            Evidence(
                source_id=f"web_{idx}",
                source_type="web",
                title=item.get("title") or "Untitled",
                excerpt=(item.get("content") or "")[:1000],
                url=item.get("url"),
                published_at=item.get("published_at"),
                relevance_score=1.0 - (idx * 0.01),
            )
        )
    return evidence
