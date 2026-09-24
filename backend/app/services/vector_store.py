"""Lightweight lexical store with BM25 ranking (stdlib only).

Replaces ChromaDB + SentenceTransformers — torch/onnxruntime blow past
Vercel's 500MB function limit — with a JSON store plus a real BM25
ranker. Same public API as the previous lexical version, so main.py and
rag_tools.py keep working unchanged.

Why BM25 instead of the old overlap ratio:
    The old ``lexical_score`` was ``matched_terms / query_terms``, which
    gives a one-line junk chunk containing a single rare keyword the same
    1.0 score as the chunk that actually answers the question, and it
    penalised Vietnamese->English alias expansion. BM25 adds IDF (rare
    terms matter more) and length normalisation (long chunks don't win by
    accident).

Performance notes (the whole point of this module):
    - Per-chunk token counts and lengths are computed once at ingest, so
      a query never re-tokenizes the corpus.
    - ``_df`` / total length / average length are maintained incrementally
      on add and delete.
    - Chunks are bucketed by document, so a scoped or fan-out search only
      scans the documents it actually needs instead of the whole corpus.
    - Scores are normalised once per result list so the top hit is always
      1.0, which keeps the absolute thresholds in ``confident_sources``
      meaningful. Multi-document merges normalise only *after* combining
      raw scores, so a weakly-matching document cannot tie with a strong one.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger("app.store")

from app.schemas import DocumentSummary, SourceChunk
from app.services.pdf_processor import TextChunk

BM25_K1 = 1.5
BM25_B = 0.75

# Keys written to disk. Index structures (``_tf``, ``_len``) are rebuilt
# from ``text`` on load so the JSON stays small.
PERSISTED_KEYS = ("id", "document_id", "filename", "page", "pages", "chunk_index", "text")

DEFAULT_MAX_CONTEXT_CHUNKS = 24

# Minimum share of the question's content words that must appear in the best
# matching chunk before the store is willing to call it evidence. A question
# that only shares one common word with the corpus ("... có bán cà phê ..."
# matching "Công ty") would otherwise always return *something* and the
# refusal gate would never fire. Calibrated by tests/eval_retrieval.py:
# at 0.25 refusal recall rises 0.00 -> 0.33 with no loss of answerable recall.
DEFAULT_MIN_QUERY_COVERAGE = 0.25

# "coverage" (unweighted) or "idf_coverage" (IDF-weighted). See
# ``_weighted_coverage`` for the measured difference.
EVIDENCE_METRICS = ("coverage", "idf_coverage")
DEFAULT_EVIDENCE_METRIC = "coverage"

# How many of the top-ranked chunks the identifier escape inspects.
#
# Checking only rank 0 was too fragile. In a live query about "mẫu MUA-07", a
# chunk describing a *different* step outranked the chunk that actually held
# the code, so the escape never fired and the document containing the exact
# identifier the user asked about was gated out as irrelevant. A named code is
# such a strong signal that it is worth a small window rather than one slot.
IDENTIFIER_ESCAPE_SCAN = 5


class VectorStore:
    """JSON-persisted chunk store ranked with BM25."""

    def __init__(
        self,
        persist_dir: Path,
        embedding_model_name: str = "",
        rerank_model_name: str = "",
        rerank_candidates: int = 24,
        max_context_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS,
        min_query_coverage: float = DEFAULT_MIN_QUERY_COVERAGE,
        evidence_metric: str = DEFAULT_EVIDENCE_METRIC,
    ) -> None:
        # First three args kept for backward compat with config.py / main.py.
        # embedding_model_name / rerank_model_name are intentionally ignored:
        # no torch / transformers here.
        _ = (embedding_model_name, rerank_model_name, rerank_candidates)
        self.max_context_chunks = max(1, int(max_context_chunks))
        self.min_query_coverage = max(0.0, min(1.0, float(min_query_coverage)))
        self.evidence_metric = (
            evidence_metric if evidence_metric in EVIDENCE_METRICS else DEFAULT_EVIDENCE_METRIC
        )
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.persist_dir / "lexical_store.json"
        self._chunks: list[dict[str, Any]] = []
        self._documents: dict[str, dict[str, Any]] = {}
        self._by_document: dict[str, list[dict[str, Any]]] = {}
        self._df: Counter = Counter()
        self._total_len = 0
        self._avgdl = 0.0
        self._lock = threading.RLock()
        self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            # File corrupt (vd. crash giữa lần ghi cũ): backup lại để
            # không mất trắng silent, rồi khởi động với store rỗng.
            try:
                backup = self.store_path.with_suffix(".json.corrupt")
                backup.write_bytes(self.store_path.read_bytes())
            except OSError:
                backup = self.store_path
            logger.warning("Index bị corrupt, đã backup tại %s: %s", backup, exc)
            return

        self._documents = payload.get("documents", {}) or {}
        raw_chunks = payload.get("chunks", []) or []
        self._chunks = []
        for raw in raw_chunks:
            if not isinstance(raw, dict):
                continue
            record: dict[str, Any] = {key: raw.get(key) for key in PERSISTED_KEYS}
            record["document_id"] = str(record.get("document_id") or "")
            record["filename"] = str(record.get("filename") or "")
            record["text"] = str(record.get("text") or "")
            record["page"] = _as_int(record.get("page"), 1)
            record["pages"] = _as_int(record.get("pages"), 0)
            record["chunk_index"] = _as_int(record.get("chunk_index"), 0)
            record["id"] = str(record.get("id") or f"{record['document_id']}:{record['chunk_index']}")
            self._chunks.append(record)
        self._rebuild_index()

    def _save(self) -> None:
        payload = {
            "chunks": [{key: record.get(key) for key in PERSISTED_KEYS} for record in self._chunks],
            "documents": self._documents,
        }
        try:
            text = json.dumps(payload, ensure_ascii=False)
            # Atomic write: crash giữa chừng không để lại JSON cụt.
            tmp_fd, tmp_name = tempfile.mkstemp(
                dir=str(self.persist_dir), prefix="lexical_store.", suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, self.store_path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError as exc:
            # Vercel read-only FS outside /tmp — search still works in-memory.
            logger.warning("Không ghi được index xuống đĩa: %s", exc)

    # -- index maintenance ----------------------------------------------
    def _rebuild_index(self) -> None:
        self._df = Counter()
        self._by_document = {}
        self._total_len = 0
        for record in self._chunks:
            tokens = content_tokens(record["text"])
            record["_tf"] = Counter(tokens)
            record["_len"] = len(tokens)
            self._total_len += len(tokens)
            for term in record["_tf"]:
                self._df[term] += 1
            self._by_document.setdefault(record["document_id"], []).append(record)
        self._refresh_avgdl()

    def _index_record(self, record: dict[str, Any]) -> None:
        tokens = content_tokens(record["text"])
        record["_tf"] = Counter(tokens)
        record["_len"] = len(tokens)
        self._total_len += len(tokens)
        for term in record["_tf"]:
            self._df[term] += 1
        self._by_document.setdefault(record["document_id"], []).append(record)

    def _deindex_record(self, record: dict[str, Any]) -> None:
        for term in record.get("_tf") or {}:
            remaining = self._df.get(term, 0) - 1
            if remaining > 0:
                self._df[term] = remaining
            else:
                self._df.pop(term, None)
        self._total_len -= record.get("_len") or 0

    def _refresh_avgdl(self) -> None:
        count = len(self._chunks)
        self._avgdl = (self._total_len / count) if count else 0.0

    # -- writes ----------------------------------------------------------
    def add_document(
        self,
        document_id: str,
        filename: str,
        pages: int,
        chunks: list[TextChunk],
        save: bool = True,
    ) -> DocumentSummary:
        """Index one document.

        Pass ``save=False`` when adding several documents in a row and call
        :meth:`flush` once at the end. ``_save`` rewrites the whole JSON file,
        so saving per document makes a 20-file batch do 20 full rewrites of a
        file that grows to tens of megabytes. Measured on a 718-page book:
        74.8s -> see tests/eval_retrieval.py --scale-bench.
        """
        with self._lock:
            self._remove_document_locked(document_id)

            for index, chunk in enumerate(chunks):
                record: dict[str, Any] = {
                    "id": f"{document_id}:{index}",
                    "document_id": document_id,
                    "filename": filename,
                    "page": chunk.page,
                    "pages": pages,
                    "chunk_index": index,
                    "text": chunk.text,
                }
                self._chunks.append(record)
                self._index_record(record)

            self._documents[document_id] = {"filename": filename, "pages": pages}
            self._refresh_avgdl()
            if save:
                self._save()
            return DocumentSummary(id=document_id, filename=filename, pages=pages, chunks=len(chunks))

    def flush(self) -> None:
        """Persist once after a run of ``add_document(..., save=False)`` calls."""
        with self._lock:
            self._save()

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            existed = self._remove_document_locked(document_id)
            if existed:
                self._save()
            return existed

    def _remove_document_locked(self, document_id: str) -> bool:
        bucket = self._by_document.pop(document_id, [])
        had_meta = self._documents.pop(document_id, None) is not None
        for record in bucket:
            self._deindex_record(record)
        if bucket:
            removed = {id(record) for record in bucket}
            self._chunks = [record for record in self._chunks if id(record) not in removed]
            self._refresh_avgdl()
        return bool(bucket) or had_meta

    def new_document_id(self) -> str:
        return uuid4().hex

    # -- reads -----------------------------------------------------------
    def list_documents(self) -> list[DocumentSummary]:
        # Đọc dưới lock: ghi (add/delete) chạy trên thread khác (batch
        # ingest song song) nên duyệt dict trần dễ gặp
        # "RuntimeError: dict changed size during iteration".
        with self._lock:
            documents_snapshot = dict(self._documents)
            by_doc_counts = {doc_id: len(records) for doc_id, records in self._by_document.items()}
        summaries: list[DocumentSummary] = []
        for document_id, meta in documents_snapshot.items():
            summaries.append(
                DocumentSummary(
                    id=document_id,
                    filename=str(meta.get("filename", document_id)),
                    pages=int(meta.get("pages", 0)),
                    chunks=by_doc_counts.get(document_id, 0),
                )
            )
        # Orphan chunks without doc meta (e.g. partial writes).
        for document_id, count in by_doc_counts.items():
            if document_id not in documents_snapshot:
                summaries.append(
                    DocumentSummary(
                        id=document_id,
                        filename=document_id,
                        pages=0,
                        chunks=count,
                    )
                )
        return summaries

    def search(self, query: str, top_k: int, document_id: str | None = None) -> list[SourceChunk]:
        """Single-scope search. Scores are normalised so the top hit is 1.0."""
        top_k = max(1, int(top_k or 1))
        # Giữ lock suốt lần đọc để snapshot _df/_avgdl/_chunks nhất quán
        # với batch ingest đang chạy song song.
        with self._lock:
            return _normalize_scores(self._search_raw(query, top_k, document_id))

    def _search_raw(
        self,
        query: str,
        top_k: int,
        document_id: str | None = None,
        skip_gate: bool = False,
    ) -> list[SourceChunk]:
        """Search one scope and return RAW BM25 scores.

        Raw scores are required for multi-document merging: normalising per
        document would make each document's best chunk score exactly 1.0, so
        every document would look equally relevant and the merge order would
        fall back to the caller's document order instead of actual relevance.
        Normalisation happens once, after the merge.

        ``skip_gate`` disables the evidence gate for this single search. It is
        set by :meth:`search_many_report` during a fan-out: there the user has
        explicitly scoped the query to specific documents, and a compare/
        summarise question only matches *part* of its text in any one document.
        Gating each document against the whole query would silently drop a
        document that is clearly relevant to the question (e.g. "so sánh số
        ngày phép và mẫu MUA-07" drops the leave-policy doc because its coverage
        of the full query is below threshold). The gate stays on for unscoped
        global search, where "nothing matches anywhere" is the refuse signal.
        """
        top_k = max(1, int(top_k or 1))
        if document_id:
            pool = self._by_document.get(document_id, [])
            if not pool:
                return []
        else:
            pool = self._chunks

        query_terms = expand_terms(tokenize(query))
        # Alias cụm từ nhiều từ ("quy trình", "tóm tắt"...): expand_terms
        # chỉ lookup từng token đơn nên không bao giờ hit. Bổ sung ở đây
        # bằng substring trên query gốc.
        query_terms = query_terms | expand_phrase_aliases(query)
        ranked = self._rank(query_terms, pool)
        identifier_terms = important_identifier_terms(query)
        if not ranked:
            # Chunk chứa mã đích danh nhưng BM25=0 (vd. chỉ có "MUA-07"
            # mà query dài) thì vẫn phải trả về — mã là tín hiệu mạnh hơn điểm.
            if identifier_terms and pool:
                fallback = [
                    _to_source(record, 0.5)
                    for record in pool
                    if has_any_identifier(record.get("text") or "", identifier_terms)
                ][:top_k]
                if fallback:
                    return dedupe_sources(fallback)
            return []

        # Evidence gate: the best chunk must cover a meaningful share of the
        # question's *information*, weighted by how distinctive each word is.
        # Plain coverage counts "công" (in almost every chunk) the same as
        # "visa" (in none), so a question sharing only common words used to
        # pass. IDF weighting makes the gate ask "did you match the words
        # that actually carry the question?" instead.
        if self.min_query_coverage > 0.0 and query_terms and not skip_gate:
            best_record = ranked[0][0]
            # Scan a small window, not just the top chunk -- see
            # IDENTIFIER_ESCAPE_SCAN for the live query that exposed this.
            identifier_escape = bool(identifier_terms) and any(
                contains_identifier(record["text"], term)
                for record, _score in ranked[:IDENTIFIER_ESCAPE_SCAN]
                for term in identifier_terms
            )
            if not identifier_escape:
                coverage = self._evidence_coverage(query_terms, best_record)
                if coverage < self.min_query_coverage:
                    return []

        sources = [_to_source(record, score) for record, score in ranked]

        if identifier_terms:
            exact_sources = sources_matching_identifiers(identifier_terms, sources)
            if exact_sources:
                sources = exact_sources

        top_sources = sources[:top_k]
        for source in top_sources:
            nearby_sources = self.sources_near_page(source.document_id, source.page, lookback_pages=12)
            source.preview_page = context_start_page(query, source, [*nearby_sources, *top_sources])
        return dedupe_sources(top_sources)

    def search_many(
        self,
        query: str,
        top_k: int,
        document_ids: list[str] | None = None,
        per_doc_k: int | None = None,
        max_chunks: int | None = None,
    ) -> list[SourceChunk]:
        """Multi-document fan-out retrieval. See :meth:`search_many_report`."""
        sources, _ = self.search_many_report(
            query,
            top_k=top_k,
            document_ids=document_ids,
            per_doc_k=per_doc_k,
            max_chunks=max_chunks,
        )
        return sources

    def search_many_report(
        self,
        query: str,
        top_k: int,
        document_ids: list[str] | None = None,
        per_doc_k: int | None = None,
        max_chunks: int | None = None,
    ) -> tuple[list[SourceChunk], list[str]]:
        """Fan-out retrieval that guarantees coverage, and reports what it dropped.

        Workflow:
          1. Fan-out: run :meth:`search` once per requested document.
          2. Coverage pass: take each document's single best chunk and order
             those by score, so every requested document is represented.
          3. Fill pass: append every remaining chunk from all documents,
             ordered by score, until the budget is spent.
          4. Truncate to the context budget and dedupe.

        Why two passes instead of plain round-robin:
            Round-robin interleaving guarantees coverage but destroys global
            score order — the first returned source is always the best chunk
            of the *first requested document*, even when a later document
            matches far better. That made top-1 precision collapse as soon as
            more than one document matched. The coverage pass keeps the
            guarantee; the fill pass restores relevance ordering.

        Budget rule — the fix for the old silent-drop bug:
            budget = max(top_k, min(n_docs, max_context_chunks))

            The old code used ``per_doc_k = max(2, ceil(top_k / n_docs))``
            and truncated to ``top_k``. With 20 documents and top_k=6 that
            kept only the first 6 documents and silently discarded the
            other 14. Now the budget grows to cover every document (one
            chunk each) up to ``max_context_chunks``, and anything still
            dropped is returned in the second element so callers can warn
            the user instead of quietly answering from partial evidence.

        Args:
            query: user/tool query string.
            top_k: minimum number of sources to return.
            document_ids: None/[] means "all documents" (global search).
            per_doc_k: hits fetched per document before merging. Defaults
                to ``ceil(budget / n_docs)``.
            max_chunks: hard cap on the context budget. Defaults to
                ``self.max_context_chunks``.

        Returns:
            ``(sources, skipped_document_ids)``.
        """
        top_k = max(1, int(top_k or 1))
        with self._lock:
            return self._search_many_report_locked(query, top_k, document_ids, per_doc_k, max_chunks)

    def _search_many_report_locked(
        self,
        query: str,
        top_k: int,
        document_ids: list[str] | None,
        per_doc_k: int | None,
        max_chunks: int | None,
    ) -> tuple[list[SourceChunk], list[str]]:
        if not document_ids:
            return _normalize_scores(self._search_raw(query, top_k, None)), []

        ordered_ids = _dedupe_preserving_order(document_ids)
        if not ordered_ids:
            return _normalize_scores(self._search_raw(query, top_k, None)), []

        n_docs = len(ordered_ids)
        ceiling = self.max_context_chunks if max_chunks is None else max(1, int(max_chunks))
        budget = max(int(top_k), min(n_docs, ceiling))

        if per_doc_k is None:
            per_doc_k = max(1, math.ceil(budget / n_docs))
        per_doc_k = max(1, int(per_doc_k))

        # skip_gate=True: câu compare/summarize chỉ khớp 1 phần query ở mỗi
        # doc. Gating từng doc bằng full query sẽ drop doc liên quan thầm lặng
        # (vd. "so sánh ngày phép và MUA-07" drop doc nghỉ phép).
        per_doc_hits = [
            self._search_raw(query, top_k=per_doc_k, document_id=doc_id, skip_gate=True)
            for doc_id in ordered_ids
        ]

        # Pass 1 — one best chunk per document, ordered by raw relevance.
        coverage = [hits[0] for hits in per_doc_hits if hits]
        coverage.sort(key=lambda source: source.score or 0.0, reverse=True)

        # Pass 2 — everything else, ordered by raw relevance.
        leftovers = [hit for hits in per_doc_hits for hit in hits[1:]]
        leftovers.sort(key=lambda source: source.score or 0.0, reverse=True)

        # Which documents the budget can reach, measured BEFORE dedupe. Dedupe
        # removes near-duplicate *chunks* (two documents saying the same thing at
        # 0.72+ token overlap), which is a redundancy decision, not a "we ran out
        # of room" one. Reporting a deduplicated document as ``skipped`` would
        # tell the user their document was never read when in fact it was read
        # and matched. The budget itself is still filled after dedupe, so the
        # freed slots go to distinct chunks rather than being wasted.
        budgeted = [*coverage, *leftovers][:budget]
        covered = {source.document_id for source in budgeted}
        skipped = [doc_id for doc_id in ordered_ids if doc_id not in covered]

        merged = _normalize_scores(dedupe_sources([*coverage, *leftovers])[:budget])
        return merged, skipped

    def coverage_by_document(self, sources: list[SourceChunk]) -> dict[str, int]:
        """Count how many returned sources came from each document id."""
        coverage: dict[str, int] = {}
        for source in sources:
            coverage[source.document_id] = coverage.get(source.document_id, 0) + 1
        return coverage

    def sources_near_page(self, document_id: str, page: int, lookback_pages: int) -> list[SourceChunk]:
        window_start = max(1, page - lookback_pages)
        with self._lock:
            records = list(self._by_document.get(document_id, []))
        sources: list[SourceChunk] = []
        for record in records:
            chunk_page = int(record["page"])
            if window_start <= chunk_page <= page:
                sources.append(_to_source(record, None, preview_page=chunk_page))
        return sources

    # -- ranking ---------------------------------------------------------
    def _rank(self, query_terms: set[str], pool: list[dict[str, Any]]) -> list[tuple[dict[str, Any], float]]:
        """BM25-rank ``pool``. Returns RAW scores, best first.

        Normalisation is deliberately left to the caller: single-scope
        searches normalise immediately, multi-document merges normalise only
        after combining every document's raw scores.
        """
        if not query_terms or not pool:
            return []

        scored: list[tuple[dict[str, Any], float]] = []
        for record in pool:
            raw = self._bm25(query_terms, record)
            if raw > 0.0:
                scored.append((record, raw))
        if not scored:
            return []

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _bm25(self, query_terms: set[str], record: dict[str, Any]) -> float:
        tf_map = record.get("_tf")
        if not tf_map:
            return 0.0

        length = record.get("_len") or 0
        avgdl = self._avgdl or 1.0
        total_chunks = len(self._chunks)
        score = 0.0

        for term in query_terms:
            freq = tf_map.get(term)
            if not freq:
                continue
            doc_freq = self._df.get(term, 0)
            idf = math.log(1.0 + (total_chunks - doc_freq + 0.5) / (doc_freq + 0.5))
            denominator = freq + BM25_K1 * (1.0 - BM25_B + BM25_B * (length / avgdl))
            score += idf * (freq * (BM25_K1 + 1.0)) / denominator

        return score

    def _evidence_coverage(self, query_terms: set[str], record: dict[str, Any]) -> float:
        if self.evidence_metric == "idf_coverage":
            return _weighted_coverage(query_terms, record, self._df, len(self._chunks))
        return _plain_coverage(query_terms, record)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _normalize_scores(sources: list[SourceChunk]) -> list[SourceChunk]:
    """Scale BM25 scores so the best source in the list scores exactly 1.0.

    ``confident_sources`` compares scores against absolute thresholds (0.4)
    and against each other (85% of the top score), so every list handed to
    it must be normalised — and it must be normalised across the whole list,
    never per document.
    """
    if not sources:
        return sources
    best = max((source.score or 0.0) for source in sources)
    if best <= 0.0:
        return sources
    for source in sources:
        source.score = (source.score or 0.0) / best
    return sources


def _weighted_coverage(
    query_terms: set[str],
    record: dict[str, Any],
    doc_freq: Counter,
    total_chunks: int,
) -> float:
    """IDF-weighted share of the question covered by one chunk.

    Counts "did this chunk match the words that carry the question?" rather
    than "how many words matched". Measured trade-off (see README): it raises
    refusal recall from 0.33 to 0.67 on the Vietnamese benchmark, but costs
    answerable recall (1.00 -> 0.91) because a paraphrase that shares few
    words with the source looks identical to a question about a topic the
    corpus does not cover. Lexical matching cannot separate those two.
    """
    if not query_terms:
        return 0.0
    tf_map = record.get("_tf") or {}
    matched_weight = 0.0
    total_weight = 0.0
    for term in query_terms:
        freq = doc_freq.get(term, 0)
        idf = math.log(1.0 + (total_chunks - freq + 0.5) / (freq + 0.5))
        total_weight += idf
        if term in tf_map:
            matched_weight += idf
    if total_weight <= 0.0:
        return 0.0
    return matched_weight / total_weight


def _plain_coverage(query_terms: set[str], record: dict[str, Any]) -> float:
    """Unweighted share of the question's content words present in one chunk."""
    if not query_terms:
        return 0.0
    tf_map = record.get("_tf") or {}
    if not tf_map:
        return 0.0
    matched = sum(1 for term in query_terms if term in tf_map)
    return matched / len(query_terms)


def _to_source(
    record: dict[str, Any], score: float | None, preview_page: int | None = None
) -> SourceChunk:
    page = int(record["page"])
    return SourceChunk(
        document_id=str(record["document_id"]),
        filename=str(record["filename"]),
        page=page,
        preview_page=page if preview_page is None else preview_page,
        text=str(record["text"]),
        score=score,
    )


ALIASES = {
    "relu": ["relu", "rectified", "linear", "unit"],
    "overfitting": ["overfitting", "overfit", "regularization", "dropout", "validation"],
    "dropout": ["dropout"],
    "regularization": ["regularization", "regularizer", "l1", "l2"],
    "gradient": ["gradient", "derivative", "backpropagation"],
    "activation": ["activation", "relu", "sigmoid", "tanh"],
    "model": ["model", "models", "algorithm", "training", "train", "trained", "fine", "tune"],
    # Vietnamese -> English training vocabulary. Both accented and folded
    # forms are listed because query terms are indexed in both forms.
    "huan": ["train", "training", "trained"],
    "luyen": ["train", "training", "trained"],
    "huấn": ["train", "training", "trained"],
    "luyện": ["train", "training", "trained"],
    "tài liệu": ["document", "documents", "file"],
    "tai lieu": ["document", "documents", "file"],
    "quy trình": ["process", "pipeline", "procedure", "steps"],
    "quy trinh": ["process", "pipeline", "procedure", "steps"],
    "bước": ["step", "steps", "stage"],
    "buoc": ["step", "steps", "stage"],
    "cấu hình": ["config", "configuration", "settings"],
    "cau hinh": ["config", "configuration", "settings"],
    "độ chính xác": ["accuracy", "precision", "score"],
    "do chinh xac": ["accuracy", "precision", "score"],
    "tóm tắt": ["summary", "summarize", "overview"],
    "tom tat": ["summary", "summarize", "overview"],
}


PROCEDURE_TERMS = {
    "bước",
    "buoc",
    "quy",
    "trình",
    "trinh",
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
# Động từ chung ("cách", "làm", "tạo") cố ý loại khỏi set trên: chúng xuất
# hiện trong hầu hết câu hỏi tiếng Việt nên trước đây mọi query đều bị coi
# là procedure và preview_page bị rewind 12 trang bừa.


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

    Scores are BM25 values normalised so the top hit is 1.0, so the
    thresholds below stay meaningful:
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
    # Identity = document + true chunk page + exact text.
    #
    # Do NOT use ``preview_page`` here. For procedure questions
    # :func:`context_start_page` rewinds ``preview_page`` to the start of the
    # section, so two genuinely different chunks -- e.g. step 3 on page 10 and
    # step 7 on page 14 of the same multi-step quy trình -- can both be rewound
    # to ``preview_page=10``. Keying on the display hint would collapse them
    # into one entry and silently drop real evidence. The true chunk ``page``
    # stays distinct, and a long page split into several chunks stays distinct
    # because their text differs (the text hash). Exact-duplicate chunks (same
    # page, same text) still collapse; near-duplicates are caught by the
    # similarity pass below.
    seen_locations: set[tuple[str, int, int]] = set()

    for source in sources:
        location = (source.document_id, source.page, hash(source.text))
        if location in seen_locations:
            continue
        if any(text_similarity(source.text, kept.text) >= 0.72 for kept in unique_sources):
            continue
        unique_sources.append(source)
        seen_locations.add(location)

    return unique_sources


def text_similarity(left: str, right: str) -> float:
    # Dùng content_tokens (bỏ stopword/header-footer boilerplate) thay vì
    # tokenize trần: trước đây 2 bước quy trình khác nhau nhưng chung
    # boilerplate vẫn đạt Jaccard >= 0.72 và bị drop thầm lặng.
    left_terms = set(content_tokens(left))
    right_terms = set(content_tokens(right))
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
        # A bare 2-digit number is usually a table column, so require a
        # row-like position.
        #
        # But those same digits are often the numeric half of a code like
        # "MUA-07", "HR-01" or "IT-09". There the fragment is unambiguous, and
        # insisting on row position made the identifier escape fail silently:
        # a document containing the exact code the user asked about was gated
        # out as "not relevant". Found by asking a live server about MUA-07 and
        # watching the document that contains it disappear from the sources.
        code_pattern = rf"\b[A-Za-z][A-Za-z0-9]*-{re.escape(identifier)}\b"
        if re.search(code_pattern, text):
            return True
        # Số đứng giữa dòng kẹp dấu câu: "mẫu 07:", "(07)", "07.", "[07]",
        # "07," — trước đây chỉ match ở đầu dòng nên escape không fire.
        # Cố ý KHÔNG match "có 07 tháng" (space cả 2 bên): số trần trong câu
        # vẫn không phải identifier.
        punct_after = rf"(?<![0-9A-Za-z]){re.escape(identifier)}(?=[\)\]\}}:;,.!?\"'’”])"
        if re.search(punct_after, text):
            return True
        punct_before = rf"(?<=[\(\[\{{:;(\"']){re.escape(identifier)}(?![0-9])"
        if re.search(punct_before, text):
            return True
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
    """Cheap overlap ratio. Only used to widen the context window for
    procedure questions, where ordering matters more than precision."""
    query_terms = expand_terms(tokenize(query)) | expand_phrase_aliases(query)
    if not query_terms:
        return 0.0

    text_terms = set(tokenize(text))
    matches = sum(1 for term in query_terms if term in text_terms)
    return matches / max(len(query_terms), 1)


def fold_vietnamese(text: str) -> str:
    """Strip Vietnamese diacritics so unaccented input still matches.

    ``đ``/``Đ`` are distinct letters that NFD does not decompose, so they
    are mapped explicitly before stripping combining marks.
    """
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _plural_fold(token: str) -> str:
    """Very light English plural folding ("classifiers" -> "classifier").

    Safe to apply unconditionally: Vietnamese orthography has no final "s",
    so this can never corrupt a Vietnamese token.
    """
    if len(token) >= 5 and token.endswith("s") and not token.endswith(("ss", "us", "is", "os")):
        return token[:-1]
    return token


def _token_variants(token: str) -> list[str]:
    """Every index form of one raw token: original, diacritic-folded, and
    their plural-folded variants."""
    variants: list[str] = []
    for candidate in (token, fold_vietnamese(token)):
        for form in (candidate, _plural_fold(candidate)):
            if len(form) > 1 and form not in STOPWORDS and form not in variants:
                variants.append(form)
    return variants


def content_tokens(text: str) -> list[str]:
    """Index tokens for one chunk: content words plus folded variants."""
    tokens: list[str] = []
    for token in tokenize(text):
        if token in STOPWORDS or len(token) <= 1:
            continue
        tokens.extend(_token_variants(token))
    return tokens


def expand_terms(tokens: list[str]) -> set[str]:
    terms: set[str] = set()
    for token in tokens:
        if token in STOPWORDS or len(token) <= 1:
            continue
        terms.update(_token_variants(token))
    for token in list(terms):
        terms.update(ALIASES.get(token, []))
    return terms


def expand_phrase_aliases(query: str) -> set[str]:
    """Mở rộng alias có dấu cách ("quy trình", "tóm tắt", ...).

    ``expand_terms`` chỉ lookup từng token đơn nên các key nhiều từ không
    bao giờ hit. Ở đây quét substring trên query gốc (cả dạng có dấu và
    fold không dấu) để bù lại.
    """
    lowered = query.lower()
    folded = fold_vietnamese(lowered)
    expanded: set[str] = set()
    for key, values in ALIASES.items():
        if " " not in key:
            continue
        if key.lower() in lowered or fold_vietnamese(key).lower() in folded:
            expanded.update(values)
    return expanded


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)
