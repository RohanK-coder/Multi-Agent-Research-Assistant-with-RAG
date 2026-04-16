from __future__ import annotations

import argparse
import json

from app.graph import run_research
from app.models import ResearchRequest
from app.rag import ingest_pdfs


def cmd_ingest(args):
    result = ingest_pdfs(args.data_dir)
    print(json.dumps(result, indent=2))


def cmd_ask(args):
    req = ResearchRequest(query=args.query, use_web=not args.no_web, use_pdf=not args.no_pdf)
    result = run_research(req)
    print("\n=== ANSWER ===\n")
    print(result.answer)
    print("\n=== CITATIONS ===\n")
    for idx, item in enumerate(result.citations, start=1):
        where = f" (page {item.page})" if item.page else ""
        url = f" - {item.url}" if item.url else ""
        print(f"[{idx}] {item.title}{where}{url}")
    if result.notes:
        print("\n=== NOTES ===\n")
        for note in result.notes:
            print(f"- {note}")


def build_parser():
    parser = argparse.ArgumentParser(description="Gemini Multi-Agent Research Assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest PDFs into Chroma")
    ingest.add_argument("--data-dir", default="./data", help="Directory containing PDFs")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="Ask a research question")
    ask.add_argument("query", help="Research question")
    ask.add_argument("--no-web", action="store_true", help="Disable web search")
    ask.add_argument("--no-pdf", action="store_true", help="Disable PDF retrieval")
    ask.set_defaults(func=cmd_ask)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
