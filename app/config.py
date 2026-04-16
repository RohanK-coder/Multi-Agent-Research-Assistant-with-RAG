from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    google_model: str = os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite")
    google_embedding_model: str = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./.chroma")
    top_k: int = int(os.getenv("RESEARCH_TOP_K", "5"))
    pdf_chunk_size: int = int(os.getenv("PDF_CHUNK_SIZE", "1200"))
    pdf_chunk_overlap: int = int(os.getenv("PDF_CHUNK_OVERLAP", "150"))
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")


settings = Settings()


def validate_api_key() -> None:
    if not settings.google_api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY (and optionally GOOGLE_API_KEY) in your .env file."
        )
