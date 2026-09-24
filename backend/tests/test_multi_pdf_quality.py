"""Multi-PDF workflow — output quality test suite.

Covers the NEW multi-PDF RAG workflow end to end (offline, no API key):

  Phase 1 — Ingestion quality  (extract -> chunk -> store, per-doc metadata)
  Phase 2 — Retrieval quality  (isolation, recall, fairness, refusal)
  Phase 3 — Tool routing       (search / summarize / compare / list intents)
  Phase 4 — Generation context (prompt carries filename+page for every chunk)
  Phase 5 — API integration    (batch upload + multi-doc chat via TestClient)
  Phase 6 — Aggregate report   (precision / recall / F1 over a labeled set)

Run:
    cd backend
    pytest tests/test_multi_pdf_quality.py -v

Design under test (see VectorStore.search_many + RagToolRunner.run):
  fan-out per document_id -> round-robin merge -> dedupe -> top_k.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.config import get_settings
from app.schemas import ChatRequest
from app.services.chat_history import ChatHistoryStore
from app.services.llm_client import LLMClient, build_prompt, build_verification_prompt
from app.services.pdf_processor import chunk_pages, extract_pdf_pages
from app.services.rag_tools import (
    RagToolRunner,
    ToolPlan,
    fallback_tool_plan,
    quick_tool_plan,
    resolve_document_ids,
)
from app.services.vector_store import VectorStore, confident_sources

# ---------------------------------------------------------------------------
# Synthetic corpus: 3 PDFs with disjoint rare tokens so lexical retrieval is
# deterministic without embeddings or network.
# ---------------------------------------------------------------------------

DOC_A_PAGES = [
    "Meowtron cat image classifier. Overfitting observed on the feline whiskers "
    "dataset. We applied dropout with rate 0.5 and early stopping. Validation "
    "accuracy reached 87 percent on the held-out cat portraits split.",
    "Meowtron training notes part two. Dropout layers placed after dense blocks. "
    "Data augmentation with horizontal flips reduced overfitting further. Best "
    "checkpoint saved at epoch 42 with validation loss 0.31.",
]

DOC_B_PAGES = [
    "Barkformer dog breed classifier. Regularization with L2 penalty lambda 0.01 "
    "on the kennel dataset. Test accuracy reached 91 percent across 120 dog breeds. "
    "Label smoothing 0.1 helped calibration of breed probabilities.",
    "Barkformer training notes part two. Weight decay 0.01 and cosine schedule. "
    "Hard negative mining on similar terrier breeds improved recall to 0.89.",
]

DOC_C_PAGES = [
    "Photron Hanoi pho recipe. Simmer beef bones for 6 hours with star anise, "
    "cinnamon, cardamom and charred ginger. Season with fish sauce and rock sugar. "
    "Serve with flat rice noodles and fresh herbs.",
]

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 180
TOP_K = 6

# Labeled eval set: (question, expected_document_filename, must-contain keywords)
LABELED_QUERIES = [
    ("What dropout rate did the Meowtron cat classifier use?", "cats.pdf", ["dropout", "0.5"]),
    ("What accuracy did the Barkformer dog classifier reach?", "dogs.pdf", ["91", "accuracy"]),
    ("How long should the Photron pho broth simmer?", "cooking.pdf", ["6 hours", "simmer"]),
    ("Which L2 lambda was used for Barkformer regularization?", "dogs.pdf", ["lambda", "0.01"]),
    ("What validation accuracy did Meowtron achieve?", "cats.pdf", ["87", "validation"]),
]

QUALITY_THRESHOLDS = {
    "min_precision": 0.8,   # fraction of labeled queries whose top-1 doc is correct
    "min_recall": 0.8,      # fraction whose top-k pool contains a keyword hit
    "min_citation_valid": 1.0,  # every returned source must be a valid citation
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pdf_bytes(pages_text: list[str]) -> bytes:
    """Build a minimal in-memory PDF with one wrapped text box per page.

    NOTE: fitz insert_text() does NOT wrap — long strings are clipped at the
    page edge. insert_textbox() with a full-page rect preserves all text so
    extraction/chunking tests see the complete content.
    """
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 550, 750), text)
    buffer = io.BytesIO()
    doc.save(buffer)
    doc.close()
    return buffer.getvalue()


def write_pdf(path: Path, pages_text: list[str]) -> Path:
    path.write_bytes(make_pdf_bytes(pages_text))
    return path


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    write_pdf(tmp_path / "cats.pdf", DOC_A_PAGES)
    write_pdf(tmp_path / "dogs.pdf", DOC_B_PAGES)
    write_pdf(tmp_path / "cooking.pdf", DOC_C_PAGES)
    return tmp_path


@pytest.fixture()
def store(tmp_path: Path, corpus_dir: Path) -> VectorStore:
    """VectorStore ingested with the 3-doc synthetic corpus."""
    vs = VectorStore(tmp_path / "chroma")
    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        pages, page_count = extract_pdf_pages(pdf_path)
        chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        assert chunks, f"no chunks extracted from {pdf_path.name}"
        vs.add_document(
            document_id=pdf_path.stem,  # stable ids: cats / dogs / cooking
            filename=pdf_path.name,
            pages=page_count,
            chunks=chunks,
        )
    return vs


@pytest.fixture()
def api_client(tmp_path: Path, corpus_dir: Path):
    """TestClient with isolated store/history/uploads + offline LLM (no key)."""
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    test_store = VectorStore(tmp_path / "chroma_api")
    test_history = ChatHistoryStore(tmp_path / "chat.sqlite3")
    offline_llm = LLMClient(api_key="", model="test-model")

    settings = get_settings()
    real_uploads = settings.uploads_dir
    settings.uploads_dir = uploads
    try:
        main_module.app.dependency_overrides[main_module.vector_store] = lambda: test_store
        main_module.app.dependency_overrides[main_module.chat_history] = lambda: test_history
        main_module.app.dependency_overrides[main_module.llm_client] = lambda: offline_llm
        with TestClient(main_module.app) as client:
            yield client, test_store
    finally:
        main_module.app.dependency_overrides.clear()
        settings.uploads_dir = real_uploads


# ===========================================================================
# Phase 1 — Ingestion quality
# ===========================================================================

class TestIngestionQuality:
    def test_extract_preserves_pages(self, corpus_dir: Path):
        pages, count = extract_pdf_pages(corpus_dir / "cats.pdf")
        assert count == len(DOC_A_PAGES)
        assert all(p.text for p in pages), "every page must yield non-empty text"
        assert "Meowtron" in pages[0].text
        assert pages[0].page == 1 and pages[1].page == 2

    def test_chunk_respects_size_and_page_metadata(self, corpus_dir: Path):
        pages, _ = extract_pdf_pages(corpus_dir / "dogs.pdf")
        chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        assert chunks, "expected at least one chunk"
        for chunk in chunks:
            assert len(chunk.text) <= CHUNK_SIZE + 1, f"oversized chunk: {len(chunk.text)}"
            assert len(chunk.text) >= 30, "tiny/noise chunks must be filtered"
            assert chunk.page >= 1

    def test_store_registers_all_documents(self, store: VectorStore):
        docs = {d.filename: d for d in store.list_documents()}
        assert set(docs) == {"cats.pdf", "dogs.pdf", "cooking.pdf"}
        for doc in docs.values():
            assert doc.pages == 2 if doc.filename != "cooking.pdf" else doc.pages == 1
            assert doc.chunks >= 1

    def test_chunks_carry_document_id(self, store: VectorStore):
        hits = store.search("Meowtron dropout", top_k=10)
        assert hits
        assert {h.document_id for h in hits} <= {"cats", "dogs", "cooking"}
        cat_hits = [h for h in hits if h.document_id == "cats"]
        assert cat_hits, "expected cat-doc chunks retrievable by doc token"


# ===========================================================================
# Phase 2 — Retrieval quality (the core multi-PDF guarantees)
# ===========================================================================

class TestRetrievalQuality:
    def test_single_doc_isolation(self, store: VectorStore):
        """A query scoped to one doc must not leak chunks from other docs."""
        hits = store.search("Meowtron dropout rate", top_k=4, document_id="cats")
        assert hits, "scoped search should hit"
        assert all(h.document_id == "cats" for h in hits)
        assert any("dropout" in h.text for h in hits)

    def test_legacy_single_doc_filter_still_works(self, store: VectorStore):
        runner = RagToolRunner(store)
        result = runner.run(ToolPlan(name="search_pdf", query="pho simmer"), top_k=4, document_id="cooking")
        assert result.sources
        assert all(s.document_id == "cooking" for s in result.sources)

    def test_search_many_covers_all_requested_docs(self, store: VectorStore):
        """Round-robin merge: a broad query must surface every requested doc."""
        hits = store.search_many(
            "Meowtron Barkformer classifier accuracy",
            top_k=6,
            document_ids=["cats", "dogs"],
        )
        returned = {h.document_id for h in hits}
        assert {"cats", "dogs"} <= returned, f"missing doc coverage: {returned}"

    def test_search_many_fair_quota_no_starvation(self, store: VectorStore):
        """Even when one doc matches more strongly, the other keeps a slot."""
        hits = store.search_many(
            "classifier accuracy", top_k=4, document_ids=["cats", "dogs"], per_doc_k=2
        )
        coverage = store.coverage_by_document(hits)
        assert coverage.get("cats", 0) >= 1
        assert coverage.get("dogs", 0) >= 1
        assert len(hits) <= 4

    def test_search_many_dedupes_and_truncates(self, store: VectorStore):
        hits = store.search_many("classifier", top_k=3, document_ids=["cats", "cats", "dogs"])
        assert len(hits) <= 3
        keys = [(h.document_id, h.preview_page or h.page, h.text) for h in hits]
        assert len(set(keys)) == len(keys), "duplicate sources leaked through"

    def test_unknown_document_id_returns_empty(self, store: VectorStore):
        assert store.search("anything here", top_k=4, document_id="no-such-doc") == []
        assert store.search_many("anything", top_k=4, document_ids=["no-such-doc"]) == []

    def test_nonsense_query_refuses_instead_of_hallucinating(self, store: VectorStore):
        hits = store.search("xqzt blender quantum zebraaaa", top_k=4)
        assert hits == [], "zero-overlap query must return [] so API says 'not found'"

    def test_citation_fields_valid(self, store: VectorStore):
        hits = store.search_many("Meowtron Barkformer", top_k=6, document_ids=["cats", "dogs"])
        assert hits
        for hit in hits:
            assert hit.document_id in {"cats", "dogs"}
            assert hit.filename.endswith(".pdf")
            assert hit.page >= 1
            assert (hit.preview_page or 0) >= 1
            assert len(hit.text.strip()) >= 30
            assert hit.score is not None and hit.score > 0

    def test_confident_sources_keeps_only_supported(self, store: VectorStore):
        pool = store.search("Meowtron dropout", top_k=6)
        shown = confident_sources(pool, "Meowtron dropout")
        assert 1 <= len(shown) <= 2
        assert shown[0].score and shown[0].score > 0

    def test_latency_budget(self, store: VectorStore):
        start = time.perf_counter()
        for _ in range(5):
            store.search_many("classifier accuracy regularization", top_k=6,
                              document_ids=["cats", "dogs", "cooking"])
        elapsed_ms = (time.perf_counter() - start) / 5 * 1000
        assert elapsed_ms < 1000, f"retrieval too slow: {elapsed_ms:.1f}ms"


# ===========================================================================
# Phase 3 — Tool routing (intent -> plan, incl. new compare_pdfs)
# ===========================================================================

class TestToolRouting:
    def test_compare_intent_detected(self):
        for question in [
            "So sánh Meowtron và Barkformer",
            "Compare the cat and dog classifiers",
            "Điểm khác nhau giữa hai tài liệu là gì?",
            "Tổng hợp kết quả across documents",
        ]:
            plan = quick_tool_plan(question)
            assert plan is not None and plan.name == "compare_pdfs", question

    def test_summary_and_list_intents_unchanged(self):
        assert quick_tool_plan("Tóm tắt tài liệu này").name == "summarize_pdf"
        assert quick_tool_plan("Có những PDF nào?").name == "list_pdfs"
        assert quick_tool_plan("What dropout was used?") is None  # -> search_pdf fallback

    def test_fallback_defaults_to_search(self):
        assert fallback_tool_plan("What dropout was used?").name == "search_pdf"

    def test_resolve_document_ids_merges_legacy_and_new(self):
        assert resolve_document_ids(None, None) is None
        assert resolve_document_ids(None, []) is None
        assert resolve_document_ids("a", None) == ["a"]
        assert resolve_document_ids(None, ["a", "b"]) == ["a", "b"]
        assert resolve_document_ids("b", ["a", "b"]) == ["a", "b"]  # dedupe, order kept
        assert resolve_document_ids("c", ["a"]) == ["a", "c"]

    def test_compare_runner_returns_multi_doc_sources(self, store: VectorStore):
        runner = RagToolRunner(store)
        result = runner.run(
            ToolPlan(name="compare_pdfs", query="Compare Meowtron Barkformer classifier accuracy"),
            top_k=6,
            document_ids=["cats", "dogs"],
        )
        assert result.name == "compare_pdfs"
        assert {s.document_id for s in result.sources} == {"cats", "dogs"}

    def test_runner_backward_compat_positional_document_id(self, store: VectorStore):
        """Old call style run(plan, top_k, document_id) must keep working."""
        runner = RagToolRunner(store)
        result = runner.run(ToolPlan(name="search_pdf", query="Meowtron"), 4, "cats")
        assert all(s.document_id == "cats" for s in result.sources)


# ===========================================================================
# Phase 4 — Generation context quality (what the LLM actually sees)
# ===========================================================================

class TestGenerationContext:
    def test_prompt_contains_filename_and_page_per_chunk(self, store: VectorStore):
        sources = store.search_many("Meowtron Barkformer", top_k=4,
                                    document_ids=["cats", "dogs"])
        prompt = build_prompt("Compare the two classifiers", sources)
        for source in sources:
            assert source.filename in prompt, f"{source.filename} missing from prompt"
            assert f"p.{source.page}" in prompt
        assert "CÂU HỎI" in prompt and "NGỮ CẢNH" in prompt

    def test_compare_prompt_mentions_both_documents(self, store: VectorStore):
        runner = RagToolRunner(store)
        result = runner.run(
            ToolPlan(name="compare_pdfs", query="Compare Meowtron Barkformer"),
            top_k=6, document_ids=["cats", "dogs"],
        )
        prompt = build_prompt("Compare Meowtron and Barkformer", result.sources)
        assert "cats.pdf" in prompt and "dogs.pdf" in prompt

    def test_verification_prompt_is_multi_doc_aware(self, store: VectorStore):
        sources = store.search_many("Meowtron Barkformer", top_k=2,
                                    document_ids=["cats", "dogs"])
        prompt = build_verification_prompt("Compare?", "Both are classifiers.", sources)
        assert "is_supported" in prompt and "CONTEXT" in prompt

    def test_offline_llm_never_calls_network(self, store: VectorStore):
        llm = LLMClient(api_key="", model="test-model")
        sources = store.search("Meowtron dropout", top_k=2)
        answer = llm.finalize_with_sources("What dropout?", sources, None, None)
        assert "OPENROUTER_API_KEY" in answer  # graceful offline message, no exception

    def test_dedupe_compare_sources_keeps_one_per_doc(self, store: VectorStore):
        sources = store.search_many("classifier", top_k=6,
                                    document_ids=["cats", "dogs", "cooking"])
        shown = main_module.dedupe_compare_sources(sources, "compare classifiers")
        docs = {s.document_id for s in shown}
        assert len(docs) >= 2, f"compare view must span docs, got {docs}"
        assert len(shown) <= 4


# ===========================================================================
# Phase 5 — API integration (batch upload + multi-doc chat)
# ===========================================================================

class TestMultiPdfApi:
    def test_batch_upload_indexes_all_pdfs(self, api_client):
        client, test_store = api_client
        files = [
            ("files", ("cats.pdf", make_pdf_bytes(DOC_A_PAGES), "application/pdf")),
            ("files", ("dogs.pdf", make_pdf_bytes(DOC_B_PAGES), "application/pdf")),
        ]
        response = client.post("/api/documents/batch", files=files)
        assert response.status_code == 200, response.text
        summaries = response.json()
        assert len(summaries) == 2
        assert {s["filename"] for s in summaries} == {"cats.pdf", "dogs.pdf"}
        assert all(s["chunks"] >= 1 and s["pages"] == 2 for s in summaries)

    def test_batch_upload_rejects_non_pdf(self, api_client):
        client, _ = api_client
        response = client.post(
            "/api/documents/batch",
            files=[("files", ("note.txt", b"hello", "text/plain"))],
        )
        assert response.status_code == 422

    def test_chat_scoped_to_single_document_id(self, api_client):
        client, test_store = api_client
        self._upload(client, "cats.pdf", DOC_A_PAGES)
        dogs_id = self._upload(client, "dogs.pdf", DOC_B_PAGES)

        response = client.post("/api/chat", json={
            "question": "What accuracy did the classifier reach?",
            "document_ids": [dogs_id],
        })
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["sources"], "expected hits scoped to dogs doc"
        assert all(s["document_id"] == dogs_id for s in payload["sources"])
        assert payload["documents_used"] == [dogs_id]

    def test_chat_across_multiple_document_ids(self, api_client):
        client, _ = api_client
        cats_id = self._upload(client, "cats.pdf", DOC_A_PAGES)
        dogs_id = self._upload(client, "dogs.pdf", DOC_B_PAGES)

        response = client.post("/api/chat", json={
            "question": "Compare Meowtron and Barkformer classifiers",
            "document_ids": [cats_id, dogs_id],
        })
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["tool_name"] == "compare_pdfs"
        assert set(payload["documents_used"]) == {cats_id, dogs_id}

    def test_chat_legacy_document_id_still_works(self, api_client):
        client, _ = api_client
        cats_id = self._upload(client, "cats.pdf", DOC_A_PAGES)
        response = client.post("/api/chat", json={
            "question": "What dropout rate was used?",
            "document_id": cats_id,  # old single-doc field, no document_ids
        })
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["sources"]
        assert all(s["document_id"] == cats_id for s in payload["sources"])

    def test_chat_unknown_question_returns_not_found(self, api_client):
        client, _ = api_client
        self._upload(client, "cats.pdf", DOC_A_PAGES)
        response = client.post("/api/chat", json={"question": "xqzt zebraaaa quantum"})
        assert response.status_code == 200
        assert response.json()["sources"] == []

    def test_chat_request_schema_accepts_both_scopes(self):
        legacy = ChatRequest(question="hi", document_id="abc")
        assert legacy.document_ids is None
        multi = ChatRequest(question="hi", document_ids=["a", "b"])
        assert multi.document_ids == ["a", "b"]

    @staticmethod
    def _upload(client: TestClient, filename: str, pages_text: list[str]) -> str:
        response = client.post(
            "/api/documents",
            files={"file": (filename, make_pdf_bytes(pages_text), "application/pdf")},
        )
        assert response.status_code == 200, response.text
        return response.json()["id"]


# ===========================================================================
# Phase 6 — Aggregate output-quality report
# ===========================================================================

class TestQualityReport:
    def test_labeled_eval_meets_thresholds(self, store: VectorStore, capsys):
        """End-to-end quality gate: precision/recall/F1 over LABELED_QUERIES."""
        top1_correct = 0
        keyword_hits = 0
        citation_valid = 0
        citation_total = 0

        for question, expected_file, keywords in LABELED_QUERIES:
            hits = store.search_many(
                question, top_k=TOP_K, document_ids=["cats", "dogs", "cooking"]
            )
            assert hits, f"no retrieval for labeled query: {question!r}"
            if hits[0].filename == expected_file:
                top1_correct += 1
            pool_text = " ".join(h.text.lower() for h in hits)
            if any(kw.lower() in pool_text for kw in keywords):
                keyword_hits += 1
            for hit in hits:
                citation_total += 1
                if (hit.document_id and hit.filename.endswith(".pdf")
                        and hit.page >= 1 and len(hit.text.strip()) >= 30
                        and (hit.score or 0) > 0):
                    citation_valid += 1

        n = len(LABELED_QUERIES)
        precision = top1_correct / n
        recall = keyword_hits / n
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        citation_rate = citation_valid / citation_total if citation_total else 0.0

        with capsys.disabled():
            print(
                f"\n[multi-pdf quality] n={n} "
                f"precision(top1)={precision:.2f} recall(keyword)={recall:.2f} "
                f"F1={f1:.2f} citation_valid={citation_rate:.2f} "
                f"({citation_valid}/{citation_total})"
            )

        assert precision >= QUALITY_THRESHOLDS["min_precision"], (
            f"top-1 doc precision {precision:.2f} < {QUALITY_THRESHOLDS['min_precision']}"
        )
        assert recall >= QUALITY_THRESHOLDS["min_recall"], (
            f"keyword recall {recall:.2f} < {QUALITY_THRESHOLDS['min_recall']}"
        )
        assert citation_rate >= QUALITY_THRESHOLDS["min_citation_valid"], (
            f"citation validity {citation_rate:.2f} < 1.0"
        )

    def test_real_book_pdf_smoke(self):
        """Smoke test on the real 718-page ML book if present in repo root."""
        repo_root = Path(__file__).resolve().parents[2]
        candidates = list(repo_root.glob("*.pdf"))
        if not candidates:
            pytest.skip("no real PDF in repo root")
        pages, count = extract_pdf_pages(candidates[0])
        assert count > 100
        chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        assert len(chunks) > 50, "real book should yield a large chunk pool"
        vs = VectorStore(Path(str(repo_root / "backend" / "storage" / "pytest_smoke")))
        vs._chunks = []
        vs._documents = {}
        vs.add_document("smoke", candidates[0].name, count, chunks[:200])
        hits = vs.search("overfitting dropout regularization", top_k=4)
        assert hits, "real book must answer a core ML query"
