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
    document = fitz.open(pdf_path)
    try:
        page = document.load_page(page_number - 1)
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return pixmap.tobytes("png")
    finally:
        document.close()


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list[PageText], chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
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

    return [chunk for chunk in chunks if len(chunk.text) >= 30]


def split_long_text(text: str, chunk_size: int, chunk_overlap: int, page: int) -> list[TextChunk]:
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
