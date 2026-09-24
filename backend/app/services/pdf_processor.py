from dataclasses import dataclass
from pathlib import Path
import re

import fitz


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    text: str
    page: int


def extract_pdf_pages(pdf_path: Path) -> tuple[list[PageText], int]:
    document = fitz.open(pdf_path)
    pages: list[PageText] = []
    try:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)
            text = normalize_text(text)
            pages.append(PageText(page=index, text=text))
        return pages, document.page_count
    finally:
        document.close()


def render_page_png(pdf_path: Path, page_number: int, zoom: float = 2.0) -> bytes:
    if page_number < 1:
        raise ValueError(f"Số trang phải >= 1, nhận được {page_number}")
    document = fitz.open(pdf_path)
    try:
        if page_number > document.page_count:
            raise ValueError(f"Trang {page_number} vượt quá {document.page_count} trang của PDF")
        page = document.load_page(page_number - 1)
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return pixmap.tobytes("png")
    finally:
        document.close()


_JUNK_TITLES = {"", "title", "untitled", "unknown", "document", "microsoft word", "pdf"}


def display_name_from_pdf(pdf_path: Path) -> str:
    """Best-effort human name for a PDF that has no recorded upload filename.

    Used by the index reconciler for PDFs that were uploaded before the
    filename was stored. Falls back to the file stem when the embedded title is
    missing or obviously a placeholder (a lot of PDFs ship with `/Title (Title)`).
    """
    fallback = f"{pdf_path.stem}.pdf"
    try:
        document = fitz.open(pdf_path)
    except Exception:  # pragma: no cover - unreadable/corrupt PDF
        return fallback
    try:
        title = (document.metadata or {}).get("title") or ""
    finally:
        document.close()

    cleaned = normalize_text(str(title))
    if len(cleaned) < 3 or cleaned.strip().lower() in _JUNK_TITLES:
        return fallback
    return cleaned[:160]


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list[PageText], chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    # Guard overlap >= size: trước đây start = max(end-overlap, start+1) tạo
    # ~100k chunk cho 1 paragraph dài và treo ingest (có thể bị trigger bằng
    # CHUNK_OVERLAP>=CHUNK_SIZE trong .env).
    chunk_size = max(100, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    chunks: list[TextChunk] = []
    for page in pages:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page.text) if part.strip()]
        buffer = ""

        for paragraph in paragraphs:
            if len(paragraph) > chunk_size:
                if buffer:
                    chunks.append(TextChunk(text=buffer.strip(), page=page.page))
                    buffer = ""
                chunks.extend(split_long_text(paragraph, chunk_size, chunk_overlap, page.page))
                continue

            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                chunks.append(TextChunk(text=buffer.strip(), page=page.page))
                overlap = buffer[-chunk_overlap:].strip() if chunk_overlap else ""
                buffer = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph

        if buffer:
            chunks.append(TextChunk(text=buffer.strip(), page=page.page))

    kept: list[TextChunk] = []
    for chunk in chunks:
        text = chunk.text.strip()
        if len(text) >= 30:
            kept.append(chunk)
        elif any(char.isdigit() for char in text) and len(text) >= 2:
            # Giữ chunk ngắn nhưng chứa mã/số (vd. "MUA-07", "Ngày nghỉ: 12"):
            # trước đây bị loại trước index nên escape/recall miss.
            kept.append(chunk)
    return kept


def split_long_text(text: str, chunk_size: int, chunk_overlap: int, page: int) -> list[TextChunk]:
    chunk_size = max(100, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    chunks: list[TextChunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(TextChunk(text=chunk_text, page=page))
        if end == len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks
