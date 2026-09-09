"""Lightweight lexical vector store for serverless deploys (Vercel Free).

Replaces ChromaDB + SentenceTransformers (which blow past Vercel's 500MB
function limit via torch/onnxruntime) with a stdlib+json + lexical ranking
store. Same public API as the old VectorStore so main.py / rag_tools.py
keep working.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.schemas import DocumentSummary, SourceChunk
from app.services.pdf_processor import TextChunk


class VectorStore:
    """JSON-persisted chunk store with TF-style lexical ranking."""

    def __init__(
        self,
        persist_dir: Path,
        embedding_model_name: str = "",
        rerank_model_name: str = "",
        rerank_candidates: int = 24,
    ) -> None:
        # Args kept for backward compat with config.py / main.py.
        # embedding_model_name / rerank_model_name are intentionally ignored:
        # no torch / transformers on Vercel.
        _ = (embedding_model_name, rerank_model_name, rerank_candidates)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.persist_dir / "lexical_store.json"
        self._chunks: list[dict[str, Any]] = []
        self._documents: dict[str, dict[str, Any]] = {}
        self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
            self._chunks = payload.get("chunks", [])
            self._documents = payload.get("documents", {})
        except (ValueError, OSError):
            self._chunks = []
            self._documents = {}

    def _save(self) -> None:
        try:
            self.store_path.write_text(
                json.dumps(
                    {"chunks": self._chunks, "documents": self._documents},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            # Vercel read-only FS outside /tmp — search still works in-memory.
            pass

    # -- writes ----------------------------------------------------------
    def add_document(self, document_id: str, filename: str, pages: int, chunks: list[TextChunk]) -> DocumentSummary:
        # Remove any stale chunks for same id (re-upload safety).
        self._chunks = [c for c in self._chunks if c["document_id"] != document_id]
        for index, chunk in enumerate(chunks):
            self._chunks.append(
                {
                    "id": f"{document_id}:{index}",
                    "document_id": document_id,
                    "filename": filename,
                    "page": chunk.page,
                    "pages": pages,
                    "chunk_index": index,
                    "text": chunk.text,
                }
            )
        self._documents[document_id] = {"filename": filename, "pages": pages}
        self._save()
        return DocumentSummary(id=document_id, filename=filename, pages=pages, chunks=len(chunks))

    def delete_document(self, document_id: str) -> bool:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c["document_id"] != document_id]
        existed = len(self._chunks) != before or document_id in self._documents
        self._documents.pop(document_id, None)
        if existed:
            self._save()
        return existed

    def new_document_id(self) -> str:
        return uuid4().hex

    # -- reads -----------------------------------------------------------
    def list_documents(self) -> list[DocumentSummary]:
        counts: dict[str, int] = {}
        for chunk in self._chunks:
            counts[chunk["document_id"]] = counts.get(chunk["document_id"], 0) + 1
        summaries: list[DocumentSummary] = []
        for document_id, meta in self._documents.items():
            summaries.append(
                DocumentSummary(
                    id=document_id,
                    filename=str(meta.get("filename", document_id)),
                    pages=int(meta.get("pages", 0)),
                    chunks=counts.get(document_id, 0),
                )
            )
        # Orphan chunks without doc meta (e.g. partial writes).
        for document_id in counts:
            if document_id not in self._documents:
                summaries.append(
                    DocumentSummary(id=document_id, filename=document_id, pages=0, chunks=counts[document_id])
                )
        return summaries

    def search(self, query: str, top_k: int, document_id: str | None = None) -> list[SourceChunk]:
        candidates: list[SourceChunk] = []
        for chunk in self._chunks:
            if document_id and chunk["document_id"] != document_id:
                continue
            score = lexical_score(query, str(chunk["text"]))
            # Small boost so identifier-heavy queries (table ids, codes)
            # surface exact matches first, mirroring old behaviour.
            candidates.append(
                SourceChunk(
                    document_id=str(chunk["document_id"]),
                    filename=str(chunk["filename"]),
                    page=int(chunk["page"]),
                    preview_page=int(chunk["page"]),
                    text=str(chunk["text"]),
                    score=score,
                )
            )
        # Drop zero-overlap chunks unless nothing matched at all.
        scored = [c for c in candidates if (c.score or 0) > 0]
        ranked = sorted(scored or candidates, key=lambda s: s.score or 0.0, reverse=True)

        identifier_terms = important_identifier_terms(query)
        if identifier_terms:
            exact_sources = sources_matching_identifiers(identifier_terms, ranked)
            if exact_sources:
                ranked = exact_sources[:top_k]
            else:
                ranked = ranked[:top_k]
        else:
            ranked = ranked[:top_k]

        top_sources = ranked
        for source in top_sources:
            nearby_sources = self.sources_near_page(source.document_id, source.page, lookback_pages=12)
            source.preview_page = context_start_page(query, source, [*nearby_sources, *ranked])
        result = dedupe_sources(top_sources)
        # Không có bằng chứng thật (score 0 và không chứa identifier) thì trả
        # rỗng để caller báo "không tìm thấy" thay vì gợi ý bừa.
        if result and max((s.score or 0.0) for s in result) <= 0:
            identifier_terms = important_identifier_terms(query)
            if not identifier_terms or not any(
                has_any_identifier(s.text, identifier_terms) for s in result
            ):
                return []
        return result

    def sources_near_page(self, document_id: str, page: int, lookback_pages: int) -> list[SourceChunk]:
        window_start = max(1, page - lookback_pages)
        sources: list[SourceChunk] = []
        for chunk in self._chunks:
            if chunk["document_id"] != document_id:
                continue
            chunk_page = int(chunk["page"])
            if window_start <= chunk_page <= page:
                sources.append(
                    SourceChunk(
                        document_id=str(chunk["document_id"]),
                        filename=str(chunk["filename"]),
                        page=chunk_page,
                        preview_page=chunk_page,
                        text=str(chunk["text"]),
                        score=None,
                    )
                )
        return sources


ALIASES = {
    "relu": ["relu", "rectified", "linear", "unit"],
    "overfitting": ["overfitting", "overfit", "regularization", "dropout", "validation"],
    "dropout": ["dropout"],
    "regularization": ["regularization", "regularizer", "l1", "l2"],
    "gradient": ["gradient", "derivative", "backpropagation"],
    "activation": ["activation", "relu", "sigmoid", "tanh"],
    "model": ["model", "models", "algorithm", "training", "train", "trained", "fine", "tune"],
    "huan": ["train", "training", "trained"],
    "luyen": ["train", "training", "trained"],
    "huấn": ["train", "training", "trained"],
    "luyện": ["train", "training", "trained"],
}


PROCEDURE_TERMS = {
    "bước",
    "buoc",
    "cách",
    "cach",
    "quy",
    "trình",
    "trinh",
    "tạo",
    "tao",
    "làm",
    "lam",
    "build",
    "create",
    "make",
    "train",
    "training",
    "pipeline",
    "process",
    "step",
    "steps",
}


STOPWORDS = {
    "la",
    "là",
    "gi",
    "gì",
    "cua",
    "của",
    "cho",
    "toi",
    "tôi",
    "hay",
    "hãy",
    "nhu",
    "như",
    "the",
    "and",
    "or",
    "with",
    "what",
    "how",
    "why",
    "is",
    "are",
    "a",
    "an",
    "to",
    "in",
    "of",
}


def confident_sources(ranked: list[SourceChunk], query: str) -> list[SourceChunk]:
    """Gợi ý tối đa 1 đoạn; đoạn thứ 2 chỉ khi cực kì chắc chắn.

    - Không có bằng chứng (score 0 và không chứa identifier) -> rỗng để
      caller trả lời "không tìm thấy" thay vì gợi ý bừa.
    - Đoạn 2 chỉ được thêm khi ngang ngửa đoạn 1 (score >= 85% và
      tuyệt đối >= 0.4), hoặc cả 2 chứa identifier chính xác từ câu hỏi
      (mã bảng, số hiệu, công thức có số).
    """
    if not ranked:
        return []
    identifier_terms = important_identifier_terms(query)

    first = ranked[0]
    first_score = first.score or 0.0
    first_has_id = bool(identifier_terms) and has_any_identifier(first.text, identifier_terms)
    if first_score <= 0 and not first_has_id:
        return []

    sources = [first]
    if len(ranked) >= 2:
        second = ranked[1]
        second_score = second.score or 0.0
        second_has_id = bool(identifier_terms) and has_any_identifier(second.text, identifier_terms)
        near_tie = first_score > 0 and second_score >= 0.85 * first_score and second_score >= 0.4
        both_identifiers = first_has_id and second_has_id and second_score > 0
        if near_tie or both_identifiers:
            sources.append(second)
    return sources


def has_any_identifier(text: str, identifier_terms: set[str]) -> bool:
    return any(contains_identifier(text, term) for term in identifier_terms)


def dedupe_sources(sources: list[SourceChunk]) -> list[SourceChunk]:
    unique_sources: list[SourceChunk] = []
    seen_locations: set[tuple[str, int]] = set()

    for source in sources:
        location = (source.document_id, source.preview_page or source.page)
        if location in seen_locations:
            continue
        if any(text_similarity(source.text, kept.text) >= 0.72 for kept in unique_sources):
            continue
        unique_sources.append(source)
        seen_locations.add(location)

    return unique_sources


def text_similarity(left: str, right: str) -> float:
    left_terms = set(tokenize(left))
    right_terms = set(tokenize(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def important_identifier_terms(query: str) -> set[str]:
    return {
        token
        for token in tokenize(query)
        if any(char.isdigit() for char in token) and len(token) >= 2
    }


def sources_matching_identifiers(identifier_terms: set[str], ranked_sources: list[SourceChunk]) -> list[SourceChunk]:
    selected: list[SourceChunk] = []
    seen_keys: set[tuple[str, int, str]] = set()

    for term in sorted(identifier_terms, key=len, reverse=True):
        matches = [source for source in ranked_sources if contains_identifier(source.text, term)]
        for source in matches[:1]:
            key = (source.document_id, source.page, source.text)
            if key not in seen_keys:
                selected.append(source)
                seen_keys.add(key)

    return sorted(selected, key=lambda source: source.score or 0, reverse=True)


def contains_identifier(text: str, identifier: str) -> bool:
    if len(identifier) <= 2 and identifier.isdigit():
        # Short numeric IDs are common in table columns, so require a row-like ID position.
        pattern = rf"(?m)(^|\n|\|)\s*{re.escape(identifier)}\s+"
        return re.search(pattern, text) is not None

    return identifier in tokenize(text)


def context_start_page(query: str, source: SourceChunk, candidates: list[SourceChunk]) -> int:
    if not is_procedure_query(query):
        return source.page

    source_score = source.score or 0.0
    if source_score <= 0:
        return source.page

    window_start = max(1, source.page - 12)
    related = [
        candidate
        for candidate in candidates
        if candidate.document_id == source.document_id
        and window_start <= candidate.page <= source.page
        and lexical_score(query, candidate.text) > 0
    ]
    if not related:
        return source.page

    return min(candidate.page for candidate in related)


def is_procedure_query(query: str) -> bool:
    tokens = set(tokenize(query))
    return bool(tokens & PROCEDURE_TERMS)


def lexical_score(query: str, text: str) -> float:
    query_terms = expand_terms(tokenize(query))
    if not query_terms:
        return 0.0

    text_terms = set(tokenize(text))
    matches = sum(1 for term in query_terms if term in text_terms)
    return matches / max(len(query_terms), 1)


def expand_terms(tokens: list[str]) -> set[str]:
    terms = {token for token in tokens if token not in STOPWORDS and len(token) > 1}
    for token in list(terms):
        terms.update(ALIASES.get(token, []))
    return terms


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)
