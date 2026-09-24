import os
import tempfile
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore

EVIDENCE_METRICS = ("coverage", "idf_coverage")


def _default_storage_dir() -> Path:
    # Vercel Functions: only temp dir is writable. Locally: backend/storage.
    if os.getenv("VERCEL") == "1" or os.getenv("VERCEL_ENV"):
        return Path(tempfile.gettempdir()) / "rag_storage"
    return Path(__file__).resolve().parents[1] / "storage"


class Settings(BaseSettings):
    app_name: str = "Multimodal RAG Chatbot"
    backend_cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,https://soll102.github.io"
    # Optional API key: khi set, các endpoint mutate (upload/delete/reindex/chat)
    # yêu cầu header X-API-Key. Bỏ trống = local-only, giữ tương thích test cũ.
    api_key: str = Field(default="", validation_alias=AliasChoices("API_KEY", "BACKEND_API_KEY"))

    openrouter_api_key: str = Field(default="", validation_alias=AliasChoices("OPENROUTER_API_KEY", "GEMINI_API_KEY"))
    openrouter_model: str = Field(
        default="google/gemini-2.5-flash-lite", validation_alias=AliasChoices("OPENROUTER_MODEL", "GEMINI_MODEL")
    )
    openrouter_fallback_model: str = ""
    enable_gemini_vision_fallback: bool = False
    vision_min_text_chars: int = Field(default=80, ge=0, le=1000)
    # Vision fallback costs one sequential network round trip per low-text page,
    # which is the slowest stage of ingest by far. This caps it per document;
    # 0 means "no cap" (only sensible for small documents). Pages past the cap
    # keep whatever text was extracted from them.
    vision_max_pages: int = Field(default=20, ge=0, le=2000)

    enable_tool_planning: bool = False
    chunk_size: int = Field(default=1100, ge=300, le=3000)
    chunk_overlap: int = Field(default=180, ge=0, le=800)
    top_k: int = Field(default=6, ge=1, le=20)

    # Hard cap on how many chunks may enter the LLM context in one turn.
    # ``search_many`` grows the per-turn budget past ``top_k`` when the user
    # selected more documents than ``top_k``, so every selected document gets
    # at least one chunk. Beyond this cap the extra documents are reported
    # back to the caller instead of being dropped silently.
    max_context_chunks: int = Field(default=24, ge=1, le=100)

    # Upload guard: reject a single PDF larger than this before writing it.
    max_upload_mb: int = Field(default=50, ge=1, le=2000)
    # Minimum share of the question's content words that must appear in the
    # best matching chunk before retrieval treats it as evidence. 0 disables
    # the gate (any single shared word counts as evidence). Calibrated with
    # tests/eval_retrieval.py — see the README for the measured trade-off.
    min_query_coverage: float = Field(default=0.25, ge=0.0, le=1.0)
    # "coverage" keeps answerable recall at 1.00 on the Vietnamese benchmark;
    # "idf_coverage" refuses more aggressively but costs recall. See README.
    evidence_metric: str = "coverage"

    # How many PDFs to extract/chunk in parallel during a batch upload.
    ingest_workers: int = Field(default=4, ge=1, le=16)
    # Previous chat turns handed to the LLM so follow-up questions keep context.
    history_turns: int = Field(default=4, ge=0, le=20)

    # Second LLM pass that checks the answer against the retrieved context and
    # rewrites/refuses when it is unsupported. Costs one extra round trip.
    enable_answer_verification: bool = True

    base_dir: Path = Path(__file__).resolve().parents[1]
    storage_dir: Path = Field(default_factory=_default_storage_dir)
    uploads_dir: Path | None = None
    index_dir: Path | None = None
    chat_db_path: Path | None = None

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @model_validator(mode="after")
    def _derive_storage_paths(self):
        # Derive sub-paths from storage_dir so VERCEL=/tmp works and
        # explicit env overrides still win.
        if self.uploads_dir is None:
            self.uploads_dir = self.storage_dir / "uploads"
        if self.index_dir is None:
            self.index_dir = self.storage_dir / "index"
        if self.chat_db_path is None:
            self.chat_db_path = self.storage_dir / "chat_history.sqlite3"
        if self.evidence_metric not in EVIDENCE_METRICS:
            raise ValueError(
                f"evidence_metric phải là một trong {EVIDENCE_METRICS}, nhận được {self.evidence_metric!r}"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    assert settings.uploads_dir is not None and settings.index_dir is not None
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    return settings
