import os
import tempfile
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict # type: ignore


def _default_storage_dir() -> Path:
    # Vercel Functions: only temp dir is writable. Locally: backend/storage.
    if os.getenv("VERCEL") == "1" or os.getenv("VERCEL_ENV"):
        return Path(tempfile.gettempdir()) / "rag_storage"
    return Path(__file__).resolve().parents[1] / "storage"


class Settings(BaseSettings):
    app_name: str = "Multimodal RAG Chatbot"
    backend_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,https://soll102.github.io"

    openrouter_api_key: str = Field(default="", validation_alias=AliasChoices("OPENROUTER_API_KEY", "GEMINI_API_KEY"))
    openrouter_model: str = Field(
        default="google/gemini-2.5-flash-lite", validation_alias=AliasChoices("OPENROUTER_MODEL", "GEMINI_MODEL")
    )
    openrouter_fallback_model: str = ""
    enable_gemini_vision_fallback: bool = False
    vision_min_text_chars: int = Field(default=80, ge=0, le=1000)

    embedding_model: str = ""
    rerank_model: str = ""
    rerank_candidates: int = Field(default=12, ge=4, le=60)
    enable_tool_planning: bool = False
    chunk_size: int = Field(default=1100, ge=300, le=3000)
    chunk_overlap: int = Field(default=180, ge=0, le=800)
    top_k: int = Field(default=6, ge=1, le=20)
    enable_answer_verification: bool = False

    base_dir: Path = Path(__file__).resolve().parents[1]
    storage_dir: Path = Field(default_factory=_default_storage_dir)
    uploads_dir: Path | None = None
    chroma_dir: Path | None = None
    chat_db_path: Path | None = None

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def _derive_storage_paths(self):
        # Derive sub-paths from storage_dir so VERCEL=/tmp works and
        # explicit env overrides still win.
        if self.uploads_dir is None:
            self.uploads_dir = self.storage_dir / "uploads"
        if self.chroma_dir is None:
            self.chroma_dir = self.storage_dir / "chroma"
        if self.chat_db_path is None:
            self.chat_db_path = self.storage_dir / "chat_history.sqlite3"
        return self


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    assert settings.uploads_dir is not None and settings.chroma_dir is not None
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return settings
