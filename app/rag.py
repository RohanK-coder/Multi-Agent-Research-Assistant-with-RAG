from __future__ import annotations

import os
import uuid

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.llm import get_embeddings
from app.models import Evidence

COLLECTION_NAME = "research_pdfs"


def _vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=settings.chroma_dir,
        embedding_function=get_embeddings(),
    )


def _clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _to_float_list(vec) -> list[float]:
    return [float(x) for x in list(vec)]


def ingest_pdfs(data_dir: str) -> dict:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    loader = PyPDFDirectoryLoader(data_dir, silent_errors=True)
    docs = loader.load()
    if not docs:
        return {"indexed": 0, "chunks": 0}

    docs = [d for d in docs if _clean_text(d.page_content)]
    if not docs:
        return {"indexed": 0, "chunks": 0}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.pdf_chunk_size,
        chunk_overlap=settings.pdf_chunk_overlap,
    )
    raw_chunks = splitter.split_documents(docs)

    chunks = []
    for idx, doc in enumerate(raw_chunks):
        cleaned = _clean_text(doc.page_content)
        if not cleaned:
            continue
        doc.page_content = cleaned
        doc.metadata = dict(doc.metadata or {})
        doc.metadata["chunk_id"] = doc.metadata.get("chunk_id") or f"chunk-{idx}"
        chunks.append(doc)

    if not chunks:
        return {"indexed": 0, "chunks": 0}

    vs = _vectorstore()
    emb = get_embeddings()

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    embeddings: list[list[float]] = []

    for doc in chunks:
        text = doc.page_content
        metadata = dict(doc.metadata or {})

        try:
            vec = emb.embed_query(text)
            vec = _to_float_list(vec)
        except Exception as e:
            print(f"Skipping chunk due to embedding error: {e}")
            continue

        if not text or not vec:
            continue

        ids.append(str(uuid.uuid4()))
        texts.append(text)
        metadatas.append(metadata)
        embeddings.append(vec)

    if not ids:
        return {"indexed": 0, "chunks": 0}

    vs._collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    pdf_paths = {d.metadata.get("source", "unknown") for d in docs}
    return {"indexed": len(pdf_paths), "chunks": len(ids)}


def retrieve_pdf_evidence(query: str, top_k: int | None = None) -> list[Evidence]:
    k = top_k or settings.top_k
    vs = _vectorstore()
    docs = vs.similarity_search_with_relevance_scores(query, k=k)

    evidence: list[Evidence] = []
    for idx, item in enumerate(docs, start=1):
        doc, score = item
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page")
        chunk_id = doc.metadata.get("chunk_id")
        title = os.path.basename(source)

        evidence.append(
            Evidence(
                source_id=f"pdf_{idx}",
                source_type="pdf",
                title=title,
                excerpt=doc.page_content[:1200],
                url=None,
                page=(page + 1) if isinstance(page, int) else None,
                chunk_id=str(chunk_id) if chunk_id is not None else None,
                relevance_score=float(score or 0.0),
            )
        )

    return evidence