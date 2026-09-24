"""Retrieval quality benchmark — the honest accuracy check.

Runs fully offline (no API key, no network): it measures the retrieval layer,
which is where accuracy is actually won or lost in a lexical RAG system. If
retrieval returns the wrong chunk, no amount of prompt engineering saves the
answer.

Usage (from ``backend/``)::

    python -m tests.eval_retrieval
    python -m tests.eval_retrieval --json eval_report.json
    python -m tests.eval_retrieval --sweep-coverage
    python -m tests.eval_retrieval --docs path/to/pdfs

Metrics
-------
top1_accuracy   share of answerable cases whose FIRST source is the right doc
recall@k        share whose returned pool contains the right doc at all
mrr             mean reciprocal rank of the first correct source
compare_coverage share of cross-document cases covering EVERY expected doc
citation_rate   share of returned sources that are valid citations
refusal_recall  share of unanswerable cases that correctly returned nothing
false_refusal   share of answerable cases that wrongly returned nothing
latency p50/p95 wall-clock per query
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.pdf_processor import PageText, chunk_pages, extract_pdf_pages
from app.services.vector_store import VectorStore, content_tokens

from tests.eval_corpus import EVAL_CASES, EVAL_DOCUMENTS, EvalCase, EvalDocument

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 180
TOP_K = 6
PARAGRAPHS_PER_PAGE = 2
MAX_CONTEXT_CHUNKS = 24


@dataclass
class CaseResult:
    case: EvalCase
    sources: list
    latency_ms: float

    @property
    def returned_nothing(self) -> bool:
        return not self.sources

    @property
    def top_document(self) -> str | None:
        return self.sources[0].document_id if self.sources else None

    @property
    def documents(self) -> set[str]:
        return {source.document_id for source in self.sources}

    def rank_of(self, document_id: str) -> int | None:
        for index, source in enumerate(self.sources, start=1):
            if source.document_id == document_id:
                return index
        return None

    @property
    def pool_text(self) -> str:
        return " ".join(source.text.lower() for source in self.sources)

    @property
    def citations_valid(self) -> bool:
        for source in self.sources:
            if not source.document_id or not source.filename.endswith(".pdf"):
                return False
            if source.page < 1 or len(source.text.strip()) < 30:
                return False
            if source.score is None or source.score <= 0:
                return False
        return True


@dataclass
class Report:
    total_cases: int = 0
    top1_accuracy: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    compare_coverage: float = 0.0
    citation_rate: float = 0.0
    refusal_recall: float = 0.0
    false_refusal_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    by_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    min_query_coverage: float = 0.0
    evidence_metric: str = "coverage"
    chunk_count: int = 0


def build_store(
    documents: list[EvalDocument],
    min_query_coverage: float,
    workdir: Path,
    evidence_metric: str = "coverage",
) -> VectorStore:
    """Ingest the corpus through the REAL pipeline (chunk_pages, not a stub)."""
    store = VectorStore(
        workdir,
        max_context_chunks=MAX_CONTEXT_CHUNKS,
        min_query_coverage=min_query_coverage,
        evidence_metric=evidence_metric,
    )
    for document in documents:
        pages: list[PageText] = []
        for start in range(0, len(document.paragraphs), PARAGRAPHS_PER_PAGE):
            group = document.paragraphs[start : start + PARAGRAPHS_PER_PAGE]
            pages.append(PageText(page=len(pages) + 1, text="\n\n".join(group)))
        chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        assert chunks, f"no chunks produced for {document.document_id}"
        store.add_document(document.document_id, document.filename, len(pages), chunks)
    return store


def load_external_documents(folder: Path) -> list[EvalDocument]:
    """Load a folder of .pdf/.txt files as evaluation documents."""
    documents: list[EvalDocument] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() == ".pdf":
            pages, _ = extract_pdf_pages(path)
            paragraphs = [page.text for page in pages if page.text.strip()]
        elif path.suffix.lower() in {".txt", ".md"}:
            raw = path.read_text(encoding="utf-8")
            paragraphs = [part.strip() for part in raw.split("\n\n") if part.strip()]
        else:
            continue
        documents.append(
            EvalDocument(document_id=path.stem, filename=f"{path.stem}.pdf", paragraphs=paragraphs)
        )
    return documents


def run_case(store: VectorStore, case: EvalCase, top_k: int) -> CaseResult:
    scope = list(case.scope) if case.scope else None
    started = time.perf_counter()
    if scope:
        sources = store.search_many(case.question, top_k=top_k, document_ids=scope)
    else:
        sources = store.search(case.question, top_k=top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return CaseResult(case=case, sources=sources, latency_ms=elapsed_ms)


def evaluate(
    store: VectorStore,
    cases: list[EvalCase],
    top_k: int = TOP_K,
    min_query_coverage: float = 0.0,
    evidence_metric: str = "coverage",
) -> Report:
    results = [run_case(store, case, top_k) for case in cases]
    report = Report(total_cases=len(results), min_query_coverage=min_query_coverage)
    report.evidence_metric = evidence_metric
    report.chunk_count = len(store._chunks)

    answerable = [r for r in results if not r.case.should_refuse]
    refusable = [r for r in results if r.case.should_refuse]
    single = [r for r in answerable if r.case.expected_document]
    compare = [r for r in answerable if r.case.expected_documents]

    if single:
        report.top1_accuracy = sum(
            1 for r in single if r.top_document == r.case.expected_document
        ) / len(single)
        report.recall_at_k = sum(
            1 for r in single if r.case.expected_document in r.documents
        ) / len(single)
        ranks = [
            rank
            for r in single
            if (rank := r.rank_of(r.case.expected_document or "")) is not None
        ]
        report.mrr = sum(1.0 / rank for rank in ranks) / len(single)

    if compare:
        report.compare_coverage = sum(
            1 for r in compare if set(r.case.expected_documents) <= r.documents
        ) / len(compare)

    all_sources = [source for r in results for source in r.sources]
    if all_sources:
        report.citation_rate = sum(
            1
            for source in all_sources
            if source.document_id
            and source.filename.endswith(".pdf")
            and source.page >= 1
            and len(source.text.strip()) >= 30
            and (source.score or 0) > 0
        ) / len(all_sources)

    if refusable:
        report.refusal_recall = sum(1 for r in refusable if r.returned_nothing) / len(refusable)
    if answerable:
        report.false_refusal_rate = sum(1 for r in answerable if r.returned_nothing) / len(answerable)

    latencies = [r.latency_ms for r in results]
    if latencies:
        report.latency_p50_ms = statistics.median(latencies)
        ordered = sorted(latencies)
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        report.latency_p95_ms = ordered[index]

    kinds = sorted({r.case.kind for r in results})
    for kind in kinds:
        bucket = [r for r in results if r.case.kind == kind]
        answerable = [r for r in bucket if not r.case.should_refuse]
        refusable = [r for r in bucket if r.case.should_refuse]

        with_expectation = [
            r for r in answerable if r.case.expected_document or r.case.expected_documents
        ]
        correct = sum(
            1
            for r in with_expectation
            if (r.case.expected_document and r.top_document == r.case.expected_document)
            or (r.case.expected_documents and set(r.case.expected_documents) <= r.documents)
        )
        keyword_cases = [r for r in answerable if r.case.keywords]
        keyword_hits = sum(
            1
            for r in keyword_cases
            if all(kw.lower() in r.pool_text for kw in r.case.keywords)
        )

        report.by_kind[kind] = {
            "cases": len(bucket),
            # None means "not applicable for this bucket" so the table can
            # print a dash instead of a misleading 0.000.
            "top1_accuracy": (correct / len(with_expectation)) if with_expectation else None,
            "keyword_recall": (keyword_hits / len(keyword_cases)) if keyword_cases else None,
            "refusal_rate": (
                sum(1 for r in refusable if r.returned_nothing) / len(refusable)
                if refusable
                else None
            ),
            "returned_nothing": sum(1 for r in bucket if r.returned_nothing),
        }

    for result in results:
        case = result.case
        if case.should_refuse:
            if not result.returned_nothing:
                report.failures.append(
                    f"[refusal] '{case.question}' -> phải từ chối nhưng trả về "
                    f"{sorted(result.documents)}"
                )
            continue
        if result.returned_nothing:
            report.failures.append(f"[miss] '{case.question}' -> không tìm thấy gì")
            continue
        if case.expected_document and result.top_document != case.expected_document:
            report.failures.append(
                f"[top1] '{case.question}' -> mong {case.expected_document}, "
                f"nhận {result.top_document}"
            )
        if case.expected_documents and not set(case.expected_documents) <= result.documents:
            missing = sorted(set(case.expected_documents) - result.documents)
            report.failures.append(
                f"[coverage] '{case.question}' -> thiếu {missing}"
            )
        if case.keywords and not all(kw.lower() in result.pool_text for kw in case.keywords):
            report.failures.append(
                f"[keyword] '{case.question}' -> thiếu {list(case.keywords)}"
            )

    return report


def _find_bench_pdf() -> Path | None:
    """Largest PDF sitting in the repository root, for the scale benchmark."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = sorted(
        repo_root.glob("*.pdf"),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_scale_bench(pdf_path: Path, copies: list[int], workdir: Path) -> int:
    """Measure ingest cost, memory and query latency as the corpus grows.

    Answers the two claims that matter for "chạy được trên máy yếu": how much
    RAM the in-memory index costs, and how query latency scales when many
    files are loaded at once. Uses a real PDF, not a synthetic one.
    """
    import tracemalloc

    queries = [
        "overfitting dropout regularization",
        "gradient descent learning rate",
        "convolutional neural network pooling layer",
        "attention mechanism transformer",
        "batch normalization layer",
        "quy trình phê duyệt thời hạn",
    ]

    print("=" * 78)
    print("SCALE BENCHMARK — real PDF, growing document count")
    print("=" * 78)
    print(f"file: {pdf_path.name}  ({pdf_path.stat().st_size / (1024 * 1024):.1f} MB)")

    pages, page_count = extract_pdf_pages(pdf_path)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"pages: {page_count}   chunks per copy: {len(chunks)}")
    print()

    print(f"{'copies':>7}{'chunks':>9}{'save_each':>11}{'save_once':>11}"
          f"{'peak_MB':>9}{'p50_ms':>9}{'p95_ms':>9}{'max_ms':>9}")
    print("-" * 74)

    for copy_count in copies:
        # Timing passes run WITHOUT tracemalloc: tracing every allocation
        # distorts wall-clock by a large factor and would make the two write
        # strategies incomparable.
        store_a = VectorStore(
            workdir / f"scale_{copy_count}_each",
            max_context_chunks=MAX_CONTEXT_CHUNKS,
            min_query_coverage=0.0,
        )
        started = time.perf_counter()
        for index in range(copy_count):
            store_a.add_document(f"each-{index}", f"each-{index}.pdf", page_count, chunks)
        save_each_seconds = time.perf_counter() - started
        del store_a

        store = VectorStore(
            workdir / f"scale_{copy_count}_batch",
            max_context_chunks=MAX_CONTEXT_CHUNKS,
            min_query_coverage=0.0,
        )
        started = time.perf_counter()
        for index in range(copy_count):
            store.add_document(
                f"batch-{index}", f"batch-{index}.pdf", page_count, chunks, save=False
            )
        store.flush()
        save_once_seconds = time.perf_counter() - started

        timings: list[float] = []
        for _ in range(3):
            for query in queries:
                begin = time.perf_counter()
                store.search_many(query, top_k=TOP_K, document_ids=None)
                timings.append((time.perf_counter() - begin) * 1000)

        ordered = sorted(timings)
        p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))

        # Memory pass — separate store, tracing on, timing ignored.
        tracemalloc.start()
        store_m = VectorStore(
            workdir / f"scale_{copy_count}_mem",
            max_context_chunks=MAX_CONTEXT_CHUNKS,
            min_query_coverage=0.0,
        )
        for index in range(copy_count):
            store_m.add_document(
                f"mem-{index}", f"mem-{index}.pdf", page_count, chunks, save=False
            )
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del store_m

        print(
            f"{copy_count:>7}{len(store._chunks):>9}{save_each_seconds:>11.2f}"
            f"{save_once_seconds:>11.2f}{peak_bytes / (1024 * 1024):>9.1f}"
            f"{statistics.median(ordered):>9.1f}"
            f"{ordered[p95_index]:>9.1f}"
            f"{ordered[-1]:>9.1f}"
        )

    print()
    print("save_each = ghi JSON sau mỗi tài liệu (hành vi cũ)")
    print("save_once = ghi JSON một lần cho cả batch (hành vi mới của /batch)")
    print("peak_MB   = bộ nhớ Python cho index trong RAM")
    print("latency   = search_many toàn cục (quét mọi chunk), trường hợp xấu nhất")
    return 0


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def escape_is_load_bearing(store: VectorStore, cases: list[EvalCase]) -> tuple[int, int]:
    """How many identifier cases the identifier escape actually decides.

    Returns ``(load_bearing, total_identifier_cases)``. A case is load-bearing
    only if the gate would block it without the escape — i.e. the best chunk's
    coverage sits below the threshold. If this returns 0, the ``identifier``
    row in the report is measuring coverage, not the escape, and a clean 1.000
    says nothing about whether the escape works.
    """
    from app.services.vector_store import (
        IDENTIFIER_ESCAPE_SCAN,
        contains_identifier,
        expand_terms,
        important_identifier_terms,
        tokenize,
    )

    load_bearing = 0
    total = 0
    for case in cases:
        if case.kind != "identifier":
            continue
        total += 1
        query_terms = expand_terms(tokenize(case.question))
        ranked = store._rank(query_terms, store._chunks)
        if not ranked:
            continue
        best = ranked[0][0]
        if store._evidence_coverage(query_terms, best) >= store.min_query_coverage:
            continue  # gate passes on its own; the escape is irrelevant
        identifier_terms = important_identifier_terms(case.question)
        if any(
            contains_identifier(record["text"], term)
            for record, _score in ranked[:IDENTIFIER_ESCAPE_SCAN]
            for term in identifier_terms
        ):
            load_bearing += 1
    return load_bearing, total


def print_report(report: Report) -> None:
    print("=" * 74)
    print("RETRIEVAL QUALITY REPORT")
    print("=" * 74)
    print(f"cases={report.total_cases}  chunks={report.chunk_count}  "
          f"metric={report.evidence_metric}  min_query_coverage={report.min_query_coverage}")
    print()
    print(f"  top1_accuracy      {report.top1_accuracy:6.3f}")
    print(f"  recall@k           {report.recall_at_k:6.3f}")
    print(f"  mrr                {report.mrr:6.3f}")
    print(f"  compare_coverage   {report.compare_coverage:6.3f}")
    print(f"  citation_rate      {report.citation_rate:6.3f}")
    print(f"  refusal_recall     {report.refusal_recall:6.3f}")
    print(f"  false_refusal_rate {report.false_refusal_rate:6.3f}")
    print(f"  latency p50        {report.latency_p50_ms:6.1f} ms")
    print(f"  latency p95        {report.latency_p95_ms:6.1f} ms")
    print()
    print(f"{'kind':<14}{'cases':>6}{'top1':>8}{'keyword':>9}{'refusal':>9}{'no-src':>8}")
    print("-" * 54)
    for kind, stats in report.by_kind.items():
        print(
            f"{kind:<14}{int(stats['cases']):>6}"
            f"{_fmt(stats['top1_accuracy']):>8}"
            f"{_fmt(stats['keyword_recall']):>9}"
            f"{_fmt(stats['refusal_rate']):>9}"
            f"{int(stats['returned_nothing']):>8}"
        )
    if report.failures:
        print()
        print(f"FAILURES ({len(report.failures)})")
        print("-" * 74)
        for failure in report.failures:
            print(f"  {failure}")


def sweep_coverage(workdir: Path, values: list[float]) -> None:
    print("=" * 78)
    print("COVERAGE GATE SWEEP — choosing DEFAULT_MIN_QUERY_COVERAGE")
    print("=" * 78)
    print("Đọc bảng theo hướng: refusal_recall cao = ít bịa; "
          "false_refusal cao = hay từ chối oan.")
    for metric in ("coverage", "idf_coverage"):
        print()
        print(f"metric = {metric}")
        print(f"{'min_cov':>8}{'top1':>8}{'recall':>8}{'refusal':>9}{'false_ref':>11}{'p50ms':>8}")
        print("-" * 52)
        for value in values:
            store = build_store(
                EVAL_DOCUMENTS, value, workdir / f"{metric}_{value}", evidence_metric=metric
            )
            report = evaluate(
                store, EVAL_CASES, TOP_K, min_query_coverage=value, evidence_metric=metric
            )
            print(
                f"{value:>8.2f}{report.top1_accuracy:>8.3f}{report.recall_at_k:>8.3f}"
                f"{report.refusal_recall:>9.3f}{report.false_refusal_rate:>11.3f}"
                f"{report.latency_p50_ms:>8.1f}"
            )


def report_to_dict(report: Report) -> dict:
    return {
        "total_cases": report.total_cases,
        "chunks": report.chunk_count,
        "evidence_metric": report.evidence_metric,
        "min_query_coverage": report.min_query_coverage,
        "top1_accuracy": round(report.top1_accuracy, 4),
        "recall_at_k": round(report.recall_at_k, 4),
        "mrr": round(report.mrr, 4),
        "compare_coverage": round(report.compare_coverage, 4),
        "citation_rate": round(report.citation_rate, 4),
        "refusal_recall": round(report.refusal_recall, 4),
        "false_refusal_rate": round(report.false_refusal_rate, 4),
        "latency_p50_ms": round(report.latency_p50_ms, 2),
        "latency_p95_ms": round(report.latency_p95_ms, 2),
        "by_kind": report.by_kind,
        "failures": report.failures,
    }


def analyze_gate(store: VectorStore) -> int:
    """Show whether corpus-level term presence can separate refusal from answerable.

    This is a recorded negative result, kept runnable so nobody re-tries it.

    The idea sounds right: an off-topic question ("Công ty có bán cà phê rang
    xay không?") names things the corpus has never heard of, so measuring the
    share of the question's content words that exist anywhere in the corpus
    should flag it. It does not work, because a legitimate paraphrase ("Tôi bị
    cảm nhẹ thì cần làm thủ tục gì để được nghỉ?") also introduces vocabulary the
    corpus does not literally contain. The two distributions overlap, so any
    threshold either refuses valid questions or lets off-topic ones through.

    Caveat worth stating: this corpus is small (24 chunks, ~580 terms), so
    "thủ tục" is absent as an artifact of size rather than because the topic is
    out of domain. A presence gate might separate cleanly on a corpus of
    thousands of documents — but that is an assumption, not a measurement, and
    this benchmark cannot confirm it.
    """
    total = len(store._chunks)
    df = store._df

    header = f"{'kind':<12} {'presence':>9} {'unknown':>8} {'max_idf':>8}  question"
    print(header)
    print("-" * len(header))

    by_kind: dict[str, list[float]] = {}
    for case in EVAL_CASES:
        terms = content_tokens(case.question)
        if not terms:
            continue
        known = [term for term in terms if df.get(term, 0) > 0]
        presence = len(known) / len(terms)
        max_idf = 0.0
        for term in known:
            idf = math.log(1 + (total - df[term] + 0.5) / (df[term] + 0.5))
            max_idf = max(max_idf, idf)
        by_kind.setdefault(case.kind, []).append(presence)
        print(
            f"{case.kind:<12} {presence:>9.2f} {len(terms) - len(known):>8} {max_idf:>8.2f}  {case.question[:48]}"
        )

    print()
    print("Presence theo dạng câu hỏi (càng thấp càng ít từ có trong corpus):")
    for kind in ("refusal", "paraphrase", "factual", "identifier", "unaccented", "compare"):
        values = by_kind.get(kind)
        if not values:
            continue
        print(f"  {kind:<12} n={len(values):<3} min={min(values):.2f}  max={max(values):.2f}")

    refusal = by_kind.get("refusal", [])
    answerable = [value for kind, values in by_kind.items() if kind != "refusal" for value in values]
    if refusal and answerable:
        print()
        print(f"  Câu ngoài phạm vi thấp nhất: {min(refusal):.2f}")
        print(f"  Câu trả lời được thấp nhất:  {min(answerable):.2f}")
        overlap = sum(1 for value in answerable if value <= max(refusal))
        print(
            f"  => {overlap}/{len(answerable)} câu trả lời được nằm TRONG dải presence của câu ngoài phạm vi."
        )
        print("  => Không có ngưỡng nào tách sạch hai nhóm. Kết quả phủ định, đã ghi lại.")
    return 0


# Vietnamese phrasings that mean "the documents do not answer this". The
# definition lives in the app (``llm_client``) so the benchmark cannot drift
# from the behaviour it is measuring.
from app.services.llm_client import looks_like_refusal as looks_like_no_evidence


def looks_like_refusal(answer: str, verification: str) -> bool:
    """Answer-level refusal test, using the same definition the app uses.

    ``no_evidence`` is the label the verifier now returns when the answer itself
    says the documents do not cover the question — a weak model will otherwise
    label its own refusal ``is_supported: true``.
    """
    if verification == "gate" or verification.startswith(("unsupported", "no_evidence")):
        return True
    return looks_like_no_evidence(answer)


def run_end_to_end(
    store: VectorStore,
    cases: list[EvalCase],
    top_k: int,
) -> int:
    """Measure refusal at the ANSWER level, not the retrieval level.

    The offline report above measures the evidence gate: whether retrieval
    returned nothing. That is not the property users care about. "Does the
    assistant make things up?" is decided by two layers working together —
    the gate, and the verification pass that checks the drafted answer against
    the retrieved chunks. A question can retrieve four irrelevant chunks and
    still end in a correct refusal, because the verifier reads them and says
    they do not answer the question.

    So this mode runs the real pipeline end to end (retrieval -> draft ->
    verify) against the configured model and reports whether the FINAL answer
    refused. It needs an API key and network access, which is why it is
    opt-in and separate from the offline numbers.
    """
    from app.config import get_settings
    from app.services.llm_client import LLMClient
    from app.services.rag_tools import ToolPlan

    settings = get_settings()
    if not settings.openrouter_api_key:
        print("Cần OPENROUTER_API_KEY trong backend/.env để chạy --end-to-end.", file=sys.stderr)
        return 2

    llm = LLMClient(settings.openrouter_api_key, settings.openrouter_model, settings.openrouter_fallback_model)
    plan = ToolPlan(name="search_pdf", query="", reason="end-to-end refusal measurement")

    print(f"model: {settings.openrouter_model}")
    print("Chạy end-to-end (retrieval -> draft -> verify). Mỗi câu tốn 1-2 lời gọi API.")
    print()

    header = f"{'kind':<12} {'gate':>5} {'refused':>8}  answer / reason"
    print(header)
    print("-" * len(header))

    totals: dict[str, list[bool]] = {}
    for case in cases:
        sources = store.search(case.question, top_k=top_k)
        gate_blocked = not sources

        if gate_blocked:
            answer = "Chưa tìm thấy nội dung liên quan trong tài liệu."
            verification = "gate"
        else:
            answer = llm.finalize_with_sources(case.question, sources, plan, None)
            if settings.enable_answer_verification:
                answer, verification = llm.verify_answer(case.question, answer, sources)
            else:
                verification = "disabled"

        refused = looks_like_refusal(answer, verification)
        totals.setdefault(case.kind, []).append(refused)
        preview = " ".join(answer.split())[:52]
        print(
            f"{case.kind:<12} {'block' if gate_blocked else 'pass':>5} "
            f"{'YES' if refused else 'no':>8}  [{verification[:12]}] {preview}"
        )

    print()
    print("=" * 74)
    print("END-TO-END REFUSAL")
    print("=" * 74)
    refusal_cases = totals.get("refusal", [])
    answerable = [flag for kind, flags in totals.items() if kind != "refusal" for flag in flags]
    if refusal_cases:
        rate = sum(refusal_cases) / len(refusal_cases)
        print(f"  refusal_correct      {rate:.3f}  ({sum(refusal_cases)}/{len(refusal_cases)} câu ngoài phạm vi được từ chối)")
    if answerable:
        wrong = sum(answerable) / len(answerable)
        print(f"  false_refusal        {wrong:.3f}  ({sum(answerable)}/{len(answerable)} câu trả lời được bị từ chối nhầm)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval quality benchmark")
    parser.add_argument("--docs", type=Path, default=None,
                        help="folder of .pdf/.txt to use instead of the built-in corpus")
    parser.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--min-coverage", type=float, default=0.25,
                        help="min_query_coverage to evaluate (default 0.25)")
    parser.add_argument("--evidence-metric", choices=["coverage", "idf_coverage"],
                        default="coverage", help="evidence gate metric (default coverage)")
    parser.add_argument("--sweep-coverage", action="store_true",
                        help="sweep min_query_coverage values and print a comparison table")
    parser.add_argument("--scale-bench", action="store_true",
                        help="benchmark ingest/memory/latency on a real PDF")
    parser.add_argument("--pdf", type=Path, default=None,
                        help="PDF for --scale-bench (default: largest PDF in repo root)")
    parser.add_argument("--copies", type=str, default="1,5,10,20",
                        help="comma-separated document counts for --scale-bench")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when any metric is below the thresholds")
    parser.add_argument("--end-to-end", action="store_true",
                        help="also measure refusal at the ANSWER level (needs API key + network)")
    parser.add_argument("--analyze-gate", action="store_true",
                        help="show why corpus-level term presence cannot gate refusal")
    parser.add_argument("--kinds", type=str, default="",
                        help="comma-separated case kinds to run, e.g. 'refusal' (default: all)")
    args = parser.parse_args()

    if args.scale_bench:
        pdf_path = args.pdf or _find_bench_pdf()
        if pdf_path is None:
            print("Không tìm thấy PDF nào để benchmark.", file=sys.stderr)
            return 2
        copies = [int(part) for part in args.copies.split(",") if part.strip()]
        with tempfile.TemporaryDirectory() as tmp:
            return run_scale_bench(pdf_path, copies, Path(tmp))

    documents = load_external_documents(args.docs) if args.docs else EVAL_DOCUMENTS
    if not documents:
        print(f"Không load được tài liệu nào từ {args.docs}", file=sys.stderr)
        return 2

    cases = EVAL_CASES
    if args.kinds:
        wanted = {part.strip() for part in args.kinds.split(",") if part.strip()}
        cases = [case for case in EVAL_CASES if case.kind in wanted]
        if not cases:
            print(f"Không có case nào thuộc {sorted(wanted)}", file=sys.stderr)
            return 2

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        if args.sweep_coverage:
            sweep_coverage(workdir, [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
            return 0

        store = build_store(
            documents, args.min_coverage, workdir / "index", evidence_metric=args.evidence_metric
        )

        if args.analyze_gate:
            return analyze_gate(store)

        report = evaluate(
            store,
            cases,
            args.top_k,
            min_query_coverage=args.min_coverage,
            evidence_metric=args.evidence_metric,
        )
        print_report(report)

        load_bearing, total_identifier = escape_is_load_bearing(store, cases)
        if total_identifier and not load_bearing:
            print()
            print(f"NOTE: the {total_identifier} identifier cases do NOT exercise the identifier")
            print("      escape — the gate passes them on coverage alone, so their top1 score")
            print("      says nothing about the escape. Covered by a unit test instead:")
            print("      test_identifier_escape_scans_past_the_top_chunk")

        if args.json:
            args.json.write_text(
                json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\nJSON report -> {args.json}")

        if args.end_to_end:
            print()
            return run_end_to_end(store, cases, args.top_k)

        if args.strict:
            checks = {
                "top1_accuracy >= 0.70": report.top1_accuracy >= 0.70,
                "recall@k >= 0.85": report.recall_at_k >= 0.85,
                "citation_rate == 1.0": report.citation_rate >= 1.0,
                "refusal_recall >= 0.80": report.refusal_recall >= 0.80,
                "false_refusal_rate <= 0.20": report.false_refusal_rate <= 0.20,
            }
            failed = [name for name, ok in checks.items() if not ok]
            if failed:
                print("\nSTRICT GATE FAILED:")
                for name in failed:
                    print(f"  - {name}")
                return 1
            print("\nSTRICT GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
