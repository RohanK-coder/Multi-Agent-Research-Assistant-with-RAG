from __future__ import annotations

from langgraph.graph import START, END, StateGraph

from app.llm import get_llm
from app.models import Evidence, GraphState, ResearchRequest, ResearchResponse
from app.prompts import PLANNER_PROMPT, SYNTHESIS_PROMPT
from app.rag import retrieve_pdf_evidence
from app.tools.web_search import search_web


def plan_node(state: GraphState) -> GraphState:
    llm = get_llm(temperature=0.1)
    prompt = f"{PLANNER_PROMPT}\n\nUser query: {state['query']}"
    plan = llm.invoke(prompt).content
    bullets = [line.strip("- ").strip() for line in str(plan).splitlines() if line.strip()]
    return {
        "plan": bullets,
        "notes": ["Planning complete"],
    }


def web_node(state: GraphState) -> GraphState:
    if not state.get("use_web", True):
        return {
            "web_evidence": [],
            "notes": ["Web search skipped"],
        }

    evidence = search_web(state["query"], max_results=5)
    return {
        "web_evidence": [e.model_dump() for e in evidence],
        "notes": [f"Web search returned {len(evidence)} sources"],
    }


def pdf_node(state: GraphState) -> GraphState:
    if not state.get("use_pdf", True):
        return {
            "pdf_evidence": [],
            "notes": ["PDF retrieval skipped"],
        }

    evidence = retrieve_pdf_evidence(state["query"])
    return {
        "pdf_evidence": [e.model_dump() for e in evidence],
        "notes": [f"PDF retrieval returned {len(evidence)} chunks"],
    }


def merge_node(state: GraphState) -> GraphState:
    combined = (state.get("web_evidence") or []) + (state.get("pdf_evidence") or [])
    return {
        "evidence": combined,
        "notes": [f"Merged {len(combined)} evidence items"],
    }


def synthesize_node(state: GraphState) -> GraphState:
    llm = get_llm(temperature=0.2)
    evidence_items = [Evidence.model_validate(item) for item in state.get("evidence", [])]

    if not evidence_items:
        return {
            "draft_answer": "I could not find any evidence to answer the question.",
            "answer": "I could not find any evidence to answer the question.",
            "notes": ["No evidence found for synthesis"],
        }

    lines = []
    for idx, ev in enumerate(evidence_items, start=1):
        page_txt = f", page {ev.page}" if ev.page else ""
        url_txt = f", URL: {ev.url}" if ev.url else ""
        lines.append(f"[{idx}] {ev.title}{page_txt}{url_txt}\nExcerpt: {ev.excerpt}")

    evidence_blob = "\n\n".join(lines)
    prompt = SYNTHESIS_PROMPT.format(query=state["query"], evidence_blob=evidence_blob)
    answer = llm.invoke(prompt).content

    return {
        "draft_answer": str(answer),
        "answer": str(answer),
        "notes": ["Synthesis complete"],
    }


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("plan", plan_node)
    graph.add_node("web", web_node)
    graph.add_node("pdf", pdf_node)
    graph.add_node("merge", merge_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "web")
    graph.add_edge("plan", "pdf")
    graph.add_edge("web", "merge")
    graph.add_edge("pdf", "merge")
    graph.add_edge("merge", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_research(request: ResearchRequest) -> ResearchResponse:
    app = build_graph()
    state: GraphState = {
        "query": request.query,
        "use_web": request.use_web,
        "use_pdf": request.use_pdf,
        "notes": [],
        "plan": [],
    }
    result = app.invoke(state)

    citations = [Evidence.model_validate(e) for e in result.get("evidence", [])]
    return ResearchResponse(
        query=request.query,
        answer=result.get("answer", "No answer generated."),
        citations=citations,
        notes=result.get("notes", []),
    )