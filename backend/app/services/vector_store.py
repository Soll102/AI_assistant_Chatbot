from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import chromadb # type: ignore
from chromadb.config import Settings as ChromaSettings # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore

from app.schemas import DocumentSummary, SourceChunk
from app.services.pdf_processor import TextChunk


class VectorStore:
    def __init__(self, persist_dir: Path, embedding_model_name: str) -> None:
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name="pdf_chunks")
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def add_document(self, document_id: str, filename: str, pages: int, chunks: list[TextChunk]) -> DocumentSummary:
        ids = [f"{document_id}:{index}" for index in range(len(chunks))]
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embed(texts)
        metadatas: list[dict[str, Any]] = [
            {
                "document_id": document_id,
                "filename": filename,
                "page": chunk.page,
                "pages": pages,
                "chunk_index": index,
            }
            for index, chunk in enumerate(chunks)
        ]

        if ids:
            self.collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

        return DocumentSummary(id=document_id, filename=filename, pages=pages, chunks=len(chunks))

    def search(self, query: str, top_k: int, document_id: str | None = None) -> list[SourceChunk]:
        where = {"document_id": document_id} if document_id else None
        candidate_count = max(top_k * 20, 80)
        results = self.collection.query(
            query_embeddings=self._embed([query]),
            n_results=candidate_count,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        sources: list[SourceChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            combined_score = rerank_score(query=query, text=str(text), distance=float(distance))
            sources.append(
                SourceChunk(
                    document_id=str(metadata["document_id"]),
                    filename=str(metadata["filename"]),
                    page=int(metadata["page"]),
                    preview_page=int(metadata["page"]),
                    text=str(text),
                    score=combined_score,
                )
            )
        ranked_sources = sorted(sources, key=lambda source: source.score or 0, reverse=True)
        top_sources = relevant_sources(query, ranked_sources, top_k)
        for source in top_sources:
            nearby_sources = self.sources_near_page(source.document_id, source.page, lookback_pages=12)
            source.preview_page = context_start_page(query, source, [*nearby_sources, *ranked_sources])
        return dedupe_sources(top_sources)

    def list_documents(self) -> list[DocumentSummary]:
        results = self.collection.get(include=["metadatas"])
        grouped: dict[str, dict[str, Any]] = {}
        for metadata in results.get("metadatas", []):
            document_id = str(metadata["document_id"])
            item = grouped.setdefault(
                document_id,
                {
                    "id": document_id,
                    "filename": str(metadata["filename"]),
                    "pages": int(metadata.get("pages", 0)),
                    "chunks": 0,
                },
            )
            item["chunks"] += 1
        return [DocumentSummary(**item) for item in grouped.values()]

    def delete_document(self, document_id: str) -> bool:
        existing = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
        ids = existing.get("ids", [])
        if not ids:
            return False
        self.collection.delete(where={"document_id": document_id})
        return True

    def new_document_id(self) -> str:
        return uuid4().hex

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedding_model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def sources_near_page(self, document_id: str, page: int, lookback_pages: int) -> list[SourceChunk]:
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        sources: list[SourceChunk] = []
        window_start = max(1, page - lookback_pages)
        for text, metadata in zip(results.get("documents", []), results.get("metadatas", [])):
            chunk_page = int(metadata["page"])
            if window_start <= chunk_page <= page:
                sources.append(
                    SourceChunk(
                        document_id=str(metadata["document_id"]),
                        filename=str(metadata["filename"]),
                        page=chunk_page,
                        preview_page=chunk_page,
                        text=str(text),
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


def rerank_score(query: str, text: str, distance: float) -> float:
    vector_score = 1.0 / (1.0 + max(distance, 0.0))
    keyword_score = lexical_score(query, text)
    exact_bonus = exact_match_bonus(query, text)
    return (0.62 * vector_score) + (0.33 * keyword_score) + exact_bonus


def relevant_sources(query: str, ranked_sources: list[SourceChunk], top_k: int) -> list[SourceChunk]:
    if not ranked_sources:
        return []

    identifier_terms = important_identifier_terms(query)
    if identifier_terms:
        exact_sources = sources_matching_identifiers(identifier_terms, ranked_sources)
        if exact_sources:
            return exact_sources[:top_k]

    best_score = ranked_sources[0].score or 0.0
    best_keyword_score = lexical_score(query, ranked_sources[0].text)
    selected = [ranked_sources[0]]

    for source in ranked_sources[1:]:
        source_score = source.score or 0.0
        keyword_score = lexical_score(query, source.text)
        close_enough = best_score > 0 and source_score >= best_score * 0.88
        has_keyword_signal = keyword_score >= max(0.16, best_keyword_score * 0.45)

        if close_enough and has_keyword_signal:
            selected.append(source)
        if len(selected) >= top_k:
            break

    return selected


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
    minimum_score = source_score * 0.72
    related = []
    for candidate in candidates:
        if candidate.document_id != source.document_id or not window_start <= candidate.page <= source.page:
            continue

        keyword_score = lexical_score(query, candidate.text)
        if candidate.score is None and keyword_score >= 0.30:
            related.append(candidate)
            continue

        candidate_score = candidate.score if candidate.score is not None else source_score * keyword_score
        if candidate_score >= minimum_score and keyword_score > 0:
            related.append(candidate)
    if not related:
        return source.page

    return min(candidate.page for candidate in related)


def is_procedure_query(query: str) -> bool:
    tokens = set(tokenize(query))
    return bool(tokens & PROCEDURE_TERMS)


def add_procedure_candidates(
    query: str,
    ranked_sources: list[SourceChunk],
    where: dict[str, str] | None,
    query_collection,
    embed,
) -> list[SourceChunk]:
    expanded_query = f"{query} prepare data select model train fine tune evaluate pipeline steps"
    results = query_collection.query(
        query_embeddings=embed([expanded_query]),
        n_results=80,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    existing_keys = {(source.document_id, source.page, source.text) for source in ranked_sources}
    expanded_sources: list[SourceChunk] = []
    for text, metadata, distance in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        key = (str(metadata["document_id"]), int(metadata["page"]), str(text))
        if key in existing_keys:
            continue
        score = rerank_score(query=expanded_query, text=str(text), distance=float(distance))
        expanded_sources.append(
            SourceChunk(
                document_id=str(metadata["document_id"]),
                filename=str(metadata["filename"]),
                page=int(metadata["page"]),
                preview_page=int(metadata["page"]),
                text=str(text),
                score=score,
            )
        )

    return sorted([*ranked_sources, *expanded_sources], key=lambda source: source.score or 0, reverse=True)


def lexical_score(query: str, text: str) -> float:
    query_terms = expand_terms(tokenize(query))
    if not query_terms:
        return 0.0

    text_terms = set(tokenize(text))
    matches = sum(1 for term in query_terms if term in text_terms)
    return matches / max(len(query_terms), 1)


def exact_match_bonus(query: str, text: str) -> float:
    query_tokens = [token for token in tokenize(query) if token not in STOPWORDS]
    text_lower = text.lower()
    bonus = 0.0
    for token in query_tokens:
        if len(token) >= 4 and token in text_lower:
            bonus += 0.025
    return min(bonus, 0.12)


def expand_terms(tokens: list[str]) -> set[str]:
    terms = {token for token in tokens if token not in STOPWORDS and len(token) > 1}
    for token in list(terms):
        terms.update(ALIASES.get(token, []))
    return terms


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)
