from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict # type: ignore


class Settings(BaseSettings):
    app_name: str = "Multimodal RAG Chatbot"
    backend_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    enable_gemini_vision_fallback: bool = False
    vision_min_text_chars: int = Field(default=80, ge=0, le=1000)

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = Field(default=1100, ge=300, le=3000)
    chunk_overlap: int = Field(default=180, ge=0, le=800)
    top_k: int = Field(default=6, ge=1, le=20)
    enable_answer_verification: bool = True

    base_dir: Path = Path(__file__).resolve().parents[1]
    storage_dir: Path = base_dir / "storage"
    uploads_dir: Path = storage_dir / "uploads"
    chroma_dir: Path = storage_dir / "chroma"
    chat_db_path: Path = storage_dir / "chat_history.sqlite3"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return settings
