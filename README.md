# Multi-Agent Research Assistant with RAG (Gemini Edition)

A student-budget friendly research assistant built with **LangGraph**, **LangChain**, **Gemini**, local **Chroma** vector storage, optional web search, PDF retrieval, and citation-aware synthesis.

## Features

- Supervisor-style LangGraph workflow
- Specialized steps for planning, web search, PDF retrieval, synthesis, and final answering
- Gemini chat model + Gemini embeddings
- Local Chroma vector store for PDFs
- CLI and FastAPI server
- Answers include citations tied to evidence objects
- DuckDuckGo fallback search, with optional Tavily support

## Project structure

```text
agentic_research_rag_gemini/
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── graph.py
│   ├── llm.py
│   ├── main.py
│   ├── models.py
│   ├── rag.py
│   ├── prompts.py
│   └── tools/
│       ├── __init__.py
│       └── web_search.py
├── data/
├── .env.example
├── README.md
└── requirements.txt
```

## 1) Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

## 3) Configure environment variables

Copy the example env file:

### macOS / Linux

```bash
cp .env.example .env
```

### Windows PowerShell

```powershell
copy .env.example .env
```

Then edit `.env` and paste your Gemini API key.

Minimum required settings:

```env
GEMINI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemini-3.1-flash-lite
GOOGLE_EMBEDDING_MODEL=models/text-embedding-004
CHROMA_DIR=./.chroma
RESEARCH_TOP_K=5
```

## 4) Add PDFs

Put your PDFs inside the `data/` folder.

Example:

```text
data/
├── paper1.pdf
├── survey.pdf
└── notes.pdf
```

## 5) Ingest PDFs into the vector store

```bash
python -m app.main ingest --data-dir ./data
```

This reads PDFs, chunks them, embeds them with Gemini embeddings, and stores them in the local Chroma database.

## 6) Ask questions from the CLI

```bash
python -m app.main ask "What do my PDFs say about agentic RAG systems?"
```

Disable web search:

```bash
python -m app.main ask "Summarize the PDFs only" --no-web
```

Disable PDF retrieval:

```bash
python -m app.main ask "What are current design patterns for multi-agent research systems?" --no-pdf
```

## 7) Run the FastAPI server

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## 8) Test the API

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the most important design choices in agentic RAG systems?",
    "use_web": true,
    "use_pdf": true
  }'
```

## CLI commands

### Ingest PDFs

```bash
python -m app.main ingest --data-dir ./data
```

### Ask a question

```bash
python -m app.main ask "Compare what the web and my PDFs say about rerankers in RAG"
```

## Troubleshooting

### `GEMINI_API_KEY` not set

Make sure `.env` exists and contains a valid Gemini API key.

### No PDF results

- Put at least one `.pdf` file in `data/`
- Re-run:

```bash
python -m app.main ingest --data-dir ./data
```

### Dependency install issues

Use a fresh virtual environment and update pip first:

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### Web search quality is weak

Set `TAVILY_API_KEY` in `.env` for stronger search results.

## Notes

- This starter is designed to be cheap to run.
- The default model is `gemini-3.1-flash-lite`.
- For better reasoning quality, you can switch to `gemini-2.5-flash` or another Gemini model in `.env`.
- The app stores PDF embeddings locally in `CHROMA_DIR`.
# Multi-Agent-Research-Assistant-with-RAG
