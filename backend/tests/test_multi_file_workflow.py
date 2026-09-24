"""Multi-file workflow tests — the behaviour that makes this usable for real.

Covers what the original suite did not:
  - fan-out over MORE documents than ``top_k`` (the silent-drop bug)
  - reporting documents that could not fit the context budget
  - relevance ordering across documents (not caller order)
  - the evidence gate that makes refusal possible
  - Vietnamese diacritic folding and English plural folding
  - upload size limits and partial batch failures
  - conversation history reaching the LLM, and the follow-up rescue path
  - batched persistence (``save=False`` + ``flush``)

Run:
    cd backend
    pytest tests/test_multi_file_workflow.py -v
"""
from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.config import get_settings
from app.schemas import SourceChunk
from app.services.chat_history import ChatHistoryStore
from app.services.llm_client import (
    EMPTY_ANSWER_MESSAGE,
    LLMClient,
    looks_like_refusal,
    strip_safety_preamble,
)
from app.services.pdf_processor import PageText, TextChunk, chunk_pages, extract_pdf_pages
from app.services.rag_tools import (
    RagToolRunner,
    ToolPlan,
    blend_query_with_history,
    previous_user_question,
    rescue_queries,
    resolve_document_ids,
)
from app.services.vector_store import VectorStore, contains_identifier, dedupe_sources

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 180


def make_pdf_bytes(pages_text: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 550, 750), text)
    buffer = io.BytesIO()
    doc.save(buffer)
    doc.close()
    return buffer.getvalue()


def ingest_text(store: VectorStore, document_id: str, text: str, page: int = 1) -> None:
    """Ingest a plain string through the real chunker (no PDF needed)."""
    pages = [PageText(page=page, text=text)]
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    assert chunks, f"no chunks from {document_id}"
    store.add_document(document_id, f"{document_id}.pdf", page, chunks)


class FakeLLM:
    """Records the arguments the chat endpoint hands to the model layer."""

    def __init__(self, answer: str = "Câu trả lời thử nghiệm.") -> None:
        self.answer = answer
        self.finalize_calls: list[dict] = []
        self.verify_calls = 0

    def start_chat(self, question: str, has_document: bool):  # pragma: no cover
        raise AssertionError("tool planning must stay disabled in these tests")

    def list_documents_answer(self, documents) -> str:
        return "danh sách"

    def finalize_with_sources(self, question, sources, tool_plan, tool_call_id, history=None) -> str:
        self.finalize_calls.append(
            {
                "question": question,
                "source_count": len(sources),
                "tool": tool_plan.name if tool_plan else None,
                "history": list(history or []),
            }
        )
        return self.answer

    def verify_answer(self, question, answer, sources):
        self.verify_calls += 1
        return answer, "supported"

    def supports_vision(self) -> bool:
        return False


@pytest.fixture()
def api_client(tmp_path: Path):
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    store = VectorStore(tmp_path / "index")
    history = ChatHistoryStore(tmp_path / "chat.sqlite3")
    fake_llm = FakeLLM()

    settings = get_settings()
    real_uploads = settings.uploads_dir
    real_max_mb = settings.max_upload_mb
    real_verification = settings.enable_answer_verification
    settings.uploads_dir = uploads
    settings.enable_answer_verification = True
    try:
        main_module.app.dependency_overrides[main_module.vector_store] = lambda: store
        main_module.app.dependency_overrides[main_module.chat_history] = lambda: history
        main_module.app.dependency_overrides[main_module.llm_client] = lambda: fake_llm
        with TestClient(main_module.app) as client:
            yield client, store, fake_llm, settings
    finally:
        main_module.app.dependency_overrides.clear()
        settings.uploads_dir = real_uploads
        settings.max_upload_mb = real_max_mb
        settings.enable_answer_verification = real_verification


# ===========================================================================
# Fan-out coverage — the silent-drop bug
# ===========================================================================

class TestFanOutCoverage:
    # Distinct topics that SHARE the phrase "quy định thời hạn phê duyệt ...
    # ngày làm việc". They are similar enough that retrieval has to
    # discriminate, but not so similar that dedupe_sources (Jaccard >= 0.72)
    # collapses them into one.
    FANOUT_TEXTS = [
        "Quy trình tuyển dụng quy định thời hạn phê duyệt phiếu đề nghị là năm ngày làm việc.",
        "Chính sách nghỉ phép quy định thời hạn phê duyệt đơn nghỉ là ba ngày làm việc.",
        "Quy trình hoàn tiền quy định thời hạn phê duyệt phiếu chi là hai ngày làm việc.",
        "Chính sách bảo mật quy định thời hạn phê duyệt quyền truy cập là bảy ngày làm việc.",
        "Quy trình xử lý sự cố quy định thời hạn phê duyệt báo cáo là một ngày làm việc.",
        "Hướng dẫn phần mềm quy định thời hạn phê duyệt yêu cầu cấp quyền là bốn ngày làm việc.",
        "Quy trình mua sắm quy định thời hạn phê duyệt đề xuất là sáu ngày làm việc.",
        "Chính sách đào tạo quy định thời hạn phê duyệt khóa học là tám ngày làm việc.",
        "Quy trình đánh giá quy định thời hạn phê duyệt kết quả là chín ngày làm việc.",
        "Chính sách lương thưởng quy định thời hạn phê duyệt điều chỉnh là mười ngày làm việc.",
    ]

    def _ingest_fanout_docs(self, store: VectorStore, count: int) -> list[str]:
        for index in range(count):
            ingest_text(store, f"doc{index}", self.FANOUT_TEXTS[index % len(self.FANOUT_TEXTS)])
        return [f"doc{index}" for index in range(count)]

    def test_covers_more_documents_than_top_k(self, tmp_path: Path):
        """10 selected documents, top_k=6 -> every document must contribute."""
        store = VectorStore(tmp_path / "idx")
        ids = self._ingest_fanout_docs(store, 10)

        sources, skipped = store.search_many_report(
            "thời hạn phê duyệt quy định", top_k=6, document_ids=ids
        )

        assert not skipped, f"documents dropped silently: {skipped}"
        assert {source.document_id for source in sources} == set(ids)

    def test_reports_documents_beyond_the_budget(self, tmp_path: Path):
        """More documents than max_context_chunks -> the extras are REPORTED."""
        store = VectorStore(tmp_path / "idx", max_context_chunks=5)
        ids = self._ingest_fanout_docs(store, 8)

        sources, skipped = store.search_many_report(
            "thời hạn phê duyệt quy định", top_k=3, document_ids=ids
        )

        assert len(sources) <= 5
        assert len(skipped) == 3
        assert set(skipped) | {source.document_id for source in sources} == set(ids)

    def test_relevance_order_beats_caller_order(self, tmp_path: Path):
        """The best-matching document must rank first even if listed last."""
        store = VectorStore(tmp_path / "idx")
        ingest_text(store, "weak", "Tài liệu nói chung về công ty và các phòng ban nội bộ.")
        ingest_text(
            store,
            "strong",
            "Mật khẩu hệ thống phải có tối thiểu mười hai ký tự và đổi định kỳ chín mươi ngày.",
        )
        ingest_text(store, "medium", "Chính sách nghỉ phép quy định mười hai ngày phép năm.")

        sources, _ = store.search_many_report(
            "mật khẩu hệ thống tối thiểu bao nhiêu ký tự",
            top_k=3,
            document_ids=["weak", "medium", "strong"],
        )

        assert sources[0].document_id == "strong"

    def test_single_doc_and_all_docs_paths_still_work(self, tmp_path: Path):
        store = VectorStore(tmp_path / "idx")
        ingest_text(store, "only", "Quy trình hoàn tiền xử lý trong năm ngày làm việc.")

        assert store.search_many("hoàn tiền", top_k=3, document_ids=["only"])
        assert store.search_many("hoàn tiền", top_k=3, document_ids=None)

    def test_near_duplicate_documents_are_not_reported_as_skipped(self, tmp_path: Path):
        """Dedupe is a redundancy call, not a budget call.

        Four documents that differ by a single token are ~0.82 similar, above
        the 0.72 dedupe threshold, so only one chunk survives. The other three
        were still retrieved and still matched — reporting them as "skipped"
        would tell the user their documents were never read.
        """
        store = VectorStore(tmp_path / "idx")
        ids = [f"doc{index}" for index in range(4)]
        for index, document_id in enumerate(ids):
            ingest_text(
                store,
                document_id,
                f"Quy trình phê duyệt phiếu số {index} trong ba ngày làm việc.",
            )

        sources, skipped = store.search_many_report(
            "quy trình phê duyệt phiếu", top_k=6, document_ids=ids
        )

        assert skipped == [], "a document dropped by dedupe was still read and matched"
        assert len(sources) == 1, "the near-duplicates collapse to one source"

    def test_dedupe_still_fills_the_budget_with_distinct_chunks(self, tmp_path: Path):
        """Freed slots go to distinct chunks instead of being wasted."""
        store = VectorStore(tmp_path / "idx", max_context_chunks=6)
        ingest_text(store, "a1", "Quy trình phê duyệt phiếu số một trong ba ngày làm việc.")
        ingest_text(store, "a2", "Quy trình phê duyệt phiếu số hai trong ba ngày làm việc.")
        ingest_text(store, "b1", "Mật khẩu hệ thống phải dài tối thiểu mười hai ký tự.")
        ingest_text(store, "b2", "Mật khẩu hệ thống phải đổi định kỳ chín mươi ngày.")

        sources, skipped = store.search_many_report(
            "quy trình phê duyệt phiếu và mật khẩu hệ thống",
            top_k=6,
            document_ids=["a1", "a2", "b1", "b2"],
        )

        # a1/a2 collapse into one chunk; b1/b2 stay distinct. Truncating before
        # dedupe instead of after would return 2 sources here even though the
        # caller has budget for 3 distinct chunks.
        assert len(sources) == 3
        assert {source.document_id for source in sources} & {"b1", "b2"} == {"b1", "b2"}
        assert skipped == []
        assert store.search_many("hoàn tiền", top_k=3, document_ids=["missing"]) == []

    def test_runner_surfaces_skipped_documents(self, tmp_path: Path):
        """Budget smaller than the selection -> runner reports the overflow."""
        store = VectorStore(tmp_path / "idx", max_context_chunks=2)
        ids = self._ingest_fanout_docs(store, 5)
        runner = RagToolRunner(store)

        result = runner.run(
            ToolPlan(name="compare_pdfs", query="thời hạn phê duyệt quy định"),
            top_k=2,
            document_ids=ids,
        )

        assert len(result.documents_skipped) == 3
        assert result.sources


# ===========================================================================
# Evidence gate — what makes refusal possible
# ===========================================================================

class TestEvidenceGate:
    def test_off_topic_question_is_refused(self, tmp_path: Path):
        store = VectorStore(tmp_path / "idx", min_query_coverage=0.25)
        ingest_text(store, "policy", "Chính sách nghỉ phép quy định mười hai ngày phép năm.")

        assert store.search("Công ty có bán cà phê rang xay không?", top_k=5) == []
        assert store.search("Giá vé máy bay đi Tokyo tháng sau?", top_k=5) == []

    def test_relevant_question_still_answers(self, tmp_path: Path):
        store = VectorStore(tmp_path / "idx", min_query_coverage=0.25)
        ingest_text(store, "policy", "Chính sách nghỉ phép quy định mười hai ngày phép năm.")

        hits = store.search("Nhân viên được hưởng bao nhiêu ngày phép năm?", top_k=5)
        assert hits
        assert hits[0].document_id == "policy"

    def test_gate_can_be_disabled(self, tmp_path: Path):
        """Without the gate, sharing only the words "công ty" is enough."""
        text = "Công ty quy định mười hai ngày phép năm cho nhân viên chính thức."

        gated = VectorStore(tmp_path / "gated", min_query_coverage=0.25)
        ingest_text(gated, "policy", text)
        assert gated.search("Công ty có bán cà phê rang xay không?", top_k=5) == []

        ungated = VectorStore(tmp_path / "ungated", min_query_coverage=0.0)
        ingest_text(ungated, "policy", text)
        assert ungated.search("Công ty có bán cà phê rang xay không?", top_k=5)

    def test_identifier_query_bypasses_the_gate(self, tmp_path: Path):
        store = VectorStore(tmp_path / "idx", min_query_coverage=0.9)
        ingest_text(store, "form", "Phiếu đề nghị tuyển dụng theo mẫu HR-01 do trưởng phòng gửi.")

        hits = store.search("HR-01", top_k=5)
        assert hits
        assert hits[0].document_id == "form"

    def test_hyphenated_identifier_is_recognised(self):
        """A code like MUA-07 must match, not just a bare table column.

        The old code sent every 2-digit token down the table-column path,
        which required the digits to sit at a row start. The numeric half of a
        hyphenated code never does, so ``contains_identifier("...MUA-07", "07")``
        returned False and the escape silently never fired.
        """
        assert contains_identifier("lập phiếu theo mẫu MUA-07.", "07")
        assert contains_identifier("Phiếu IT-09 ghi thời điểm sự cố.", "09")
        assert contains_identifier("Bảng KPI-02 đánh giá nhân viên.", "02")
        # A genuine table cell still counts.
        assert contains_identifier("| 07 | Nguyễn Văn A |", "07")
        # And a bare number mid-sentence is still not an identifier.
        assert not contains_identifier("Năm 2024 có 07 tháng dữ liệu.", "07")

    def test_identifier_escape_scans_past_the_top_chunk(self, tmp_path: Path):
        """The identifier-bearing chunk is not always rank 0.

        Reproduces a live failure: a question naming "mẫu MUA-07" had a
        different step outrank the chunk holding the code, so checking only
        rank 0 let the gate drop the very document that contained the
        identifier. Coverage of the top chunk is deliberately below threshold.
        """
        store = VectorStore(tmp_path / "idx", min_query_coverage=0.9)
        store.add_document(
            "proc",
            "proc.pdf",
            2,
            [
                TextChunk(text="Bước 2: phòng mua hàng báo giá ba nhà cung cấp.", page=1),
                TextChunk(text="Bước 1: lập phiếu theo mẫu MUA-07.", page=2),
            ],
        )

        hits = store.search("Mẫu MUA-07 dùng ở bước nào?", top_k=5)

        assert hits, "the document holding the exact identifier must not be gated out"
        assert any("MUA-07" in hit.text for hit in hits)


# ===========================================================================
# Token normalisation — Vietnamese diacritics and English plurals
# ===========================================================================

class TestTokenNormalisation:
    def test_unaccented_query_matches_accented_text(self, tmp_path: Path):
        store = VectorStore(tmp_path / "idx")
        ingest_text(store, "bhxh", "Nhân viên được hưởng mười hai ngày phép năm theo quy định.")

        hits = store.search("nhan vien duoc huong bao nhieu ngay phep nam", top_k=3)
        assert hits
        assert hits[0].document_id == "bhxh"

    def test_plural_query_matches_singular_text(self, tmp_path: Path):
        store = VectorStore(tmp_path / "idx")
        ingest_text(store, "ml", "A dropout layer with rate 0.5 regularized the classifier.")

        hits = store.search("dropout layers classifiers", top_k=3)
        assert hits
        assert hits[0].document_id == "ml"


# ===========================================================================
# Persistence — batched writes
# ===========================================================================

class TestPersistence:
    def test_save_false_then_flush_persists_everything(self, tmp_path: Path):
        index_dir = tmp_path / "idx"
        store = VectorStore(index_dir)
        for index in range(4):
            store.add_document(
                f"doc{index}",
                f"doc{index}.pdf",
                1,
                chunk_pages([PageText(page=1, text=f"Nội dung số {index} về quy trình phê duyệt.")],
                            CHUNK_SIZE, CHUNK_OVERLAP),
                save=False,
            )
        store.flush()

        reloaded = VectorStore(index_dir)
        assert len(reloaded.list_documents()) == 4
        assert reloaded.search("quy trình phê duyệt", top_k=5)

    def test_delete_document_clears_index_state(self, tmp_path: Path):
        store = VectorStore(tmp_path / "idx")
        ingest_text(store, "doc1", "Chính sách bảo mật quy định mã hóa dữ liệu mật.")
        ingest_text(store, "doc2", "Quy trình hoàn tiền xử lý trong năm ngày làm việc.")

        assert store.delete_document("doc1") is True
        assert store.search("mã hóa dữ liệu mật", top_k=3) == []
        assert store.search("hoàn tiền", top_k=3)
        assert store.delete_document("doc1") is False


# ===========================================================================
# API — multi-file upload
# ===========================================================================

class TestMultiFileUploadApi:
    def test_batch_upload_indexes_every_file(self, api_client):
        client, store, _, _ = api_client
        files = [
            ("files", (f"doc{index}.pdf", make_pdf_bytes([f"Nội dung tài liệu số {index} về quy trình."]), "application/pdf"))
            for index in range(5)
        ]

        response = client.post("/api/documents/batch", files=files)

        assert response.status_code == 200, response.text
        assert len(response.json()) == 5
        assert len(store.list_documents()) == 5

    def test_partial_failure_keeps_the_good_files(self, api_client):
        client, store, _, _ = api_client
        files = [
            ("files", ("good.pdf", make_pdf_bytes(["Nội dung hợp lệ về quy trình phê duyệt."]), "application/pdf")),
            ("files", ("bad.txt", b"not a pdf", "text/plain")),
        ]

        response = client.post("/api/documents/batch", files=files)

        assert response.status_code == 200, response.text
        assert [item["filename"] for item in response.json()] == ["good.pdf"]
        assert len(store.list_documents()) == 1

    def test_all_files_invalid_returns_422(self, api_client):
        client, _, _, _ = api_client
        response = client.post(
            "/api/documents/batch",
            files=[("files", ("bad.txt", b"nope", "text/plain"))],
        )
        assert response.status_code == 422

    def test_oversized_upload_is_rejected(self, api_client):
        client, store, _, settings = api_client
        settings.max_upload_mb = 1

        # The size guard aborts the stream before any PDF parsing happens, so
        # the payload only has to *look* like a PDF by filename. Padding a real
        # PDF is not worth it: fitz clips textbox content, so generating a
        # >1MB valid PDF here would need hundreds of pages and slow the suite.
        big = b"%PDF-1.7\n" + b"x" * (2 * 1024 * 1024) + b"\n%%EOF\n"
        assert len(big) > 1024 * 1024, "fixture must exceed the 1MB limit"

        response = client.post(
            "/api/documents",
            files={"file": ("big.pdf", big, "application/pdf")},
        )

        assert response.status_code == 413
        assert "giới hạn" in response.json()["detail"]
        assert store.list_documents() == []

    def test_upload_rejects_non_pdf(self, api_client):
        client, _, _, _ = api_client
        response = client.post("/api/documents", files={"file": ("note.txt", b"hello", "text/plain")})
        assert response.status_code == 400


# ===========================================================================
# Chat — scope, history, follow-up rescue
# ===========================================================================

class TestChatScopeAndHistory:
    def _upload(self, client: TestClient, filename: str, text: str) -> str:
        response = client.post(
            "/api/documents",
            files={"file": (filename, make_pdf_bytes([text]), "application/pdf")},
        )
        assert response.status_code == 200, response.text
        return response.json()["id"]

    def test_chat_scoped_to_selected_documents(self, api_client):
        client, _, _, _ = api_client
        cats = self._upload(client, "cats.pdf", "Meowtron classifier used dropout rate 0.5.")
        self._upload(client, "dogs.pdf", "Barkformer classifier used L2 regularization.")

        response = client.post("/api/chat", json={
            "question": "What dropout rate was used?",
            "document_ids": [cats],
        })

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["sources"]
        assert {source["document_id"] for source in payload["sources"]} == {cats}
        assert payload["documents_skipped"] == []

    def test_chat_reports_skipped_documents(self, api_client):
        client, _, _, _ = api_client
        ids = [
            self._upload(client, f"doc{index}.pdf", f"Quy trình phê duyệt phiếu số {index} trong ba ngày.")
            for index in range(4)
        ]

        response = client.post("/api/chat", json={
            "question": "So sánh quy trình phê duyệt phiếu giữa các tài liệu",
            "document_ids": ids,
        })

        assert response.status_code == 200, response.text
        assert response.json()["tool_name"] == "compare_pdfs"
        # top_k=6 >= 4 docs, so nothing may be dropped.
        assert response.json()["documents_skipped"] == []

    def test_history_reaches_the_llm(self, api_client):
        client, _, fake_llm, _ = api_client
        cats = self._upload(client, "cats.pdf", "Meowtron classifier used dropout rate 0.5.")

        first = client.post("/api/chat", json={
            "question": "What dropout rate did Meowtron use?",
            "document_ids": [cats],
        }).json()

        client.post("/api/chat", json={
            "question": "What regularization did the other model use?",
            "document_ids": [cats],
            "session_id": first["session_id"],
        })

        assert len(fake_llm.finalize_calls) == 2
        second_history = fake_llm.finalize_calls[1]["history"]
        assert second_history, "second turn must carry prior conversation"
        assert second_history[0].role == "user"
        assert "dropout" in second_history[0].content.lower()

    def test_follow_up_question_is_rescued_with_history(self, api_client):
        client, _, fake_llm, _ = api_client
        cats = self._upload(client, "cats.pdf", "Meowtron classifier used dropout rate 0.5.")

        first = client.post("/api/chat", json={
            "question": "What dropout rate did Meowtron use?",
            "document_ids": [cats],
        }).json()
        assert first["sources"]

        follow_up = client.post("/api/chat", json={
            "question": "Còn cái kia thì sao?",
            "document_ids": [cats],
            "session_id": first["session_id"],
        }).json()

        assert follow_up["sources"], "follow-up must be rescued, not refused"
        assert follow_up["query_rewritten"] is True

    def test_verification_runs_when_enabled(self, api_client):
        client, _, fake_llm, _ = api_client
        cats = self._upload(client, "cats.pdf", "Meowtron classifier used dropout rate 0.5.")

        payload = client.post("/api/chat", json={
            "question": "What dropout rate did Meowtron use?",
            "document_ids": [cats],
        }).json()

        assert fake_llm.verify_calls == 1
        assert payload["verification"] == "supported"

    def test_unknown_question_returns_not_found(self, api_client):
        client, _, _, _ = api_client
        self._upload(client, "cats.pdf", "Meowtron classifier used dropout rate 0.5.")

        payload = client.post("/api/chat", json={"question": "xqzt zebraaaa quantum"}).json()

        assert payload["sources"] == []
        assert "tìm thấy" in payload["answer"].lower()


class TestBlendQueryWithHistory:
    def test_blends_previous_user_turn(self):
        class Message:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        history = [
            Message("user", "Chính sách nghỉ phép là gì?"),
            Message("assistant", "Mười hai ngày phép năm."),
        ]
        blended = blend_query_with_history("Còn cái kia thì sao?", history)

        assert blended.startswith("Chính sách nghỉ phép là gì?")
        assert "Còn cái kia" in blended

    def test_returns_question_when_history_is_empty(self):
        assert blend_query_with_history("Hỏi gì đó", []) == "Hỏi gì đó"

    def test_previous_user_question_skips_assistant_turns(self):
        class Message:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        history = [
            Message("user", "Câu một"),
            Message("assistant", "Trả lời một"),
            Message("user", "Câu hai"),
            Message("assistant", "Trả lời hai"),
        ]
        assert previous_user_question(history) == "Câu hai"
        assert previous_user_question([]) is None

    def test_rescue_queries_offers_blend_then_previous_question(self):
        """The ladder matters: blending lengthens the query and can trip the gate."""
        class Message:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        history = [Message("user", "Chính sách nghỉ phép là gì?")]

        candidates = rescue_queries("Còn cái kia thì sao?", history)

        assert candidates == [
            "Chính sách nghỉ phép là gì? Còn cái kia thì sao?",
            "Chính sách nghỉ phép là gì?",
        ]

    def test_rescue_queries_is_empty_without_history(self):
        assert rescue_queries("Hỏi gì đó", []) == []
        # A question identical to the previous turn needs no rescue.
        class Message:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        same = [Message("user", "Hỏi gì đó")]
        assert rescue_queries("Hỏi gì đó", same) == []

    def test_resolve_document_ids_unchanged(self):
        assert resolve_document_ids(None, None) is None
        assert resolve_document_ids(None, []) is None
        assert resolve_document_ids("a", ["b"]) == ["b", "a"]


# ===========================================================================
# Index reconciliation — recovering PDFs that predate the current index
# ===========================================================================

class TestIndexReconciliation:
    def _drop_pdf(self, settings, document_id: str, text: str) -> Path:
        """Simulate a PDF already on disk but absent from the index."""
        path = settings.uploads_dir / f"{document_id}.pdf"
        path.write_bytes(make_pdf_bytes([text]))
        return path

    def test_status_reports_unindexed_pdfs(self, api_client):
        client, _, _, settings = api_client
        self._drop_pdf(settings, "legacy1", "Quy trình hoàn tiền trong năm ngày làm việc.")
        self._drop_pdf(settings, "legacy2", "Chính sách bảo mật quy định mã hóa dữ liệu.")

        payload = client.get("/api/documents/status").json()

        assert payload["indexed"] == 0
        assert payload["unindexed_count"] == 2
        assert sorted(payload["unindexed"]) == ["legacy1", "legacy2"]

    def test_reindex_indexes_orphan_pdfs(self, api_client):
        client, store, _, settings = api_client
        self._drop_pdf(settings, "legacy1", "Quy trình hoàn tiền xử lý trong năm ngày làm việc.")
        self._drop_pdf(settings, "legacy2", "Chính sách bảo mật quy định mã hóa dữ liệu mật.")

        payload = client.post("/api/documents/reindex").json()

        assert payload["errors"] == []
        assert {item["id"] for item in payload["indexed"]} == {"legacy1", "legacy2"}
        assert len(store.list_documents()) == 2
        assert store.search("hoàn tiền", top_k=3)
        assert client.get("/api/documents/status").json()["unindexed_count"] == 0

    def test_reindex_is_idempotent(self, api_client):
        client, store, _, settings = api_client
        self._drop_pdf(settings, "legacy1", "Quy trình hoàn tiền xử lý trong năm ngày làm việc.")

        first = client.post("/api/documents/reindex").json()
        second = client.post("/api/documents/reindex").json()

        assert len(first["indexed"]) == 1
        assert second["indexed"] == []
        assert len(store.list_documents()) == 1

    def test_reindex_survives_a_corrupt_pdf(self, api_client):
        client, store, _, settings = api_client
        broken = settings.uploads_dir / "broken.pdf"
        broken.write_bytes(b"not really a pdf")
        self._drop_pdf(settings, "good", "Quy trình hoàn tiền xử lý trong năm ngày làm việc.")

        payload = client.post("/api/documents/reindex").json()

        assert [item["id"] for item in payload["indexed"]] == ["good"]
        assert len(payload["errors"]) == 1
        assert "broken" in payload["errors"][0]
        assert len(store.list_documents()) == 1
        # A failed re-index must NOT delete a stored document: this PDF is the
        # user's only copy, unlike a freshly uploaded file that cannot be read.
        assert broken.exists(), "reindex must never destroy the stored PDF"

    def test_status_reports_zero_when_everything_is_indexed(self, api_client):
        client, _, _, _ = api_client
        client.post(
            "/api/documents",
            files={"file": ("a.pdf", make_pdf_bytes(["Nội dung về quy trình phê duyệt."]), "application/pdf")},
        )

        assert client.get("/api/documents/status").json()["unindexed_count"] == 0


# ===========================================================================
# Vision fallback — the stage that made ingest look like a hang
# ===========================================================================

class TestVisionFallbackBudget:
    """One network round trip per low-text page, so the count must be bounded.

    Measured on this project's own library: vision costs ~15s per page, and a
    682-page textbook had 78 figure pages. Uncapped, a re-index of the library
    made 112 sequential calls and ran for over 13 minutes with no output.
    """

    class CountingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def extract_page_from_image(self, image_bytes: bytes, page_number: int) -> str:
            self.calls += 1
            return f"nội dung đọc từ ảnh trang {page_number}"

        def supports_vision(self) -> bool:
            return True

    def _blank_pdf(self, tmp_path: Path, pages: int) -> Path:
        path = tmp_path / "scan.pdf"
        path.write_bytes(make_pdf_bytes([""] * pages))
        return path

    def test_vision_calls_are_capped_per_document(self, tmp_path: Path):
        from app.main import enrich_low_text_pages_with_vision

        pdf_path = self._blank_pdf(tmp_path, 5)
        pages = [PageText(page=number, text="") for number in range(1, 6)]
        llm = self.CountingLLM()

        enriched = enrich_low_text_pages_with_vision(
            pdf_path=pdf_path,
            pages=pages,
            min_text_chars=80,
            llm=llm,
            max_pages=2,
        )

        assert llm.calls == 2, "the cap must bound network calls, not just log"
        assert len(enriched) == 5, "every page must still be returned"
        assert "nội dung đọc từ ảnh" in enriched[0].text
        assert enriched[4].text == "", "pages past the cap keep their extracted text"

    def test_zero_means_no_cap(self, tmp_path: Path):
        from app.main import enrich_low_text_pages_with_vision

        pdf_path = self._blank_pdf(tmp_path, 3)
        pages = [PageText(page=number, text="") for number in range(1, 4)]
        llm = self.CountingLLM()

        enrich_low_text_pages_with_vision(
            pdf_path=pdf_path,
            pages=pages,
            min_text_chars=80,
            llm=llm,
            max_pages=0,
        )

        assert llm.calls == 3

    def test_pages_with_text_never_cost_a_call(self, tmp_path: Path):
        from app.main import enrich_low_text_pages_with_vision

        pdf_path = self._blank_pdf(tmp_path, 2)
        pages = [
            PageText(page=1, text="Trang này đã có đủ chữ để không cần đọc ảnh."),
            PageText(page=2, text=""),
        ]
        llm = self.CountingLLM()

        enrich_low_text_pages_with_vision(
            pdf_path=pdf_path,
            pages=pages,
            min_text_chars=10,
            llm=llm,
            max_pages=20,
        )

        assert llm.calls == 1

    def test_reindex_does_not_spend_vision_calls(self, api_client):
        """Repairing an index must be offline and fast, whatever the settings say."""
        client, store, _, settings = api_client
        settings.enable_gemini_vision_fallback = True
        settings.vision_max_pages = 0
        try:
            path = settings.uploads_dir / "scan.pdf"
            path.write_bytes(make_pdf_bytes([""] * 2))

            from app import main as main_module

            calls = []

            class SpyLLM(TestVisionFallbackBudget.CountingLLM):
                def extract_page_from_image(self, image_bytes, page_number):
                    calls.append(page_number)
                    return super().extract_page_from_image(image_bytes, page_number)

            original = main_module.app.dependency_overrides[main_module.llm_client]
            main_module.app.dependency_overrides[main_module.llm_client] = lambda: SpyLLM()
            try:
                payload = client.post("/api/documents/reindex").json()
            finally:
                main_module.app.dependency_overrides[main_module.llm_client] = original

            assert calls == [], "reindex must not call the vision model"
            # A page with no text at all yields no chunks, so this is reported
            # as an error rather than silently producing an empty document.
            assert payload["indexed"] == []
            assert len(payload["errors"]) == 1
            assert path.exists(), "reindex must never destroy the stored PDF"
        finally:
            settings.enable_gemini_vision_fallback = False
            settings.vision_max_pages = 20


# ===========================================================================
# Answer-level refusal labelling
# ===========================================================================

class TestRefusalLabelling:
    """The `verification` field is user-facing, so it must not contradict the answer.

    A weak verifier model labels its own refusal ``is_supported: true``: the
    answer text says "tài liệu không cung cấp..." while the label claims the
    answer was supported. Measured with liquid/lfm-2.5-2.6b:free, that happened
    on 3 of 6 off-topic questions.
    """

    @staticmethod
    def _source() -> SourceChunk:
        return SourceChunk(
            document_id="doc",
            filename="doc.pdf",
            page=1,
            text="Chính sách nghỉ phép quy định mười hai ngày phép năm.",
        )

    @staticmethod
    def _llm_returning(payload_json: str) -> LLMClient:
        llm = LLMClient("fake-key", "fake-model")
        llm._generate_text = lambda prompt, image_bytes=None: payload_json  # type: ignore[method-assign]
        return llm

    def test_detects_the_standard_refusal(self):
        assert looks_like_refusal("Tài liệu không cung cấp đủ thông tin để trả lời câu hỏi.")
        assert looks_like_refusal("Tài liệu không cung cấp thông tin về giá vé máy bay.")
        assert looks_like_refusal("Chưa tìm thấy nội dung liên quan trong tài liệu.")

    def test_does_not_flag_a_real_answer(self):
        assert not looks_like_refusal("Nhân viên chính thức được hưởng mười hai ngày phép năm.")
        assert not looks_like_refusal("Mật khẩu hệ thống phải có tối thiểu mười hai ký tự.")

    def test_a_noted_gap_inside_a_real_answer_is_not_a_refusal(self):
        """Only the opening counts — a comparison may note gaps and still answer.

        Matching the phrase anywhere flagged all three comparison answers as
        refusals, which made false_refusal look like 23% instead of 0%.
        """
        answer = (
            "Dựa trên các đoạn văn bản được cung cấp: quy trình hoàn tiền xử lý trong năm ngày "
            "làm việc, còn chính sách bảo mật không cung cấp thông tin về thời hạn này."
        )
        assert not looks_like_refusal(answer)

    def test_verifier_label_is_overridden_when_the_answer_refuses(self):
        """A model claiming 'supported' on its own refusal must not win."""
        llm = self._llm_returning(
            '{"is_supported": true, "fixed_answer": "", "reason": "câu trả lời đúng"}'
        )

        answer, verification = llm.verify_answer(
            "Công ty có bán cà phê rang xay không?",
            "Tài liệu không cung cấp thông tin về cà phê rang xay.",
            [self._source()],
        )

        assert verification.startswith("no_evidence"), verification
        assert "không cung cấp" in answer

    def test_verifier_label_stays_supported_for_a_real_answer(self):
        llm = self._llm_returning(
            '{"is_supported": true, "fixed_answer": "", "reason": "khớp tài liệu"}'
        )

        _, verification = llm.verify_answer(
            "Bao nhiêu ngày phép năm?",
            "Nhân viên chính thức được hưởng mười hai ngày phép năm.",
            [self._source()],
        )

        assert verification.startswith("supported"), verification


# ===========================================================================
# A preamble-only model response must never reach the user as an answer
# ===========================================================================

class TestSafetyPreambleNeverLeaks:
    """Regression: a live run returned the literal string "User Safety: safe".

    The model occasionally prefixes answers with a safety classification line,
    and ``strip_safety_preamble`` removed it. But when that line was the ENTIRE
    response, the stripper's ``or text.strip()`` fallback put it straight back,
    so the user's "answer" was the model's internal bookkeeping. Found by
    calling the real API, not by a unit test -- hence these.
    """

    def test_preamble_only_response_strips_to_empty(self):
        assert strip_safety_preamble("User Safety: safe") == ""
        assert strip_safety_preamble("Safety: safe\n\n") == ""

    def test_blank_input_is_unchanged(self):
        # Old behaviour, kept deliberately: empty in, empty out.
        assert strip_safety_preamble("") == ""
        assert strip_safety_preamble("   \n  ") == ""

    def test_real_answer_survives_the_strip(self):
        assert strip_safety_preamble("User Safety: safe\n\nCông ty có 12 ngày phép.") == (
            "Công ty có 12 ngày phép."
        )

    def test_verify_answer_replaces_a_preamble_only_draft(self):
        """The draft is preamble-only, so there is nothing to verify."""
        llm = TestRefusalLabelling._llm_returning('{"is_supported": true}')

        answer, verification = llm.verify_answer(
            "Bao nhiêu ngày phép năm?",
            "User Safety: safe",
            [TestRefusalLabelling._source()],
        )

        assert answer == EMPTY_ANSWER_MESSAGE
        assert answer != "User Safety: safe"
        assert verification == "skipped", verification

    def test_verify_answer_strips_a_preamble_from_a_good_draft(self):
        llm = TestRefusalLabelling._llm_returning(
            '{"is_supported": true, "fixed_answer": "", "reason": "khớp tài liệu"}'
        )

        answer, _ = llm.verify_answer(
            "Bao nhiêu ngày phép năm?",
            "User Safety: safe\n\nNhân viên được mười hai ngày phép năm.",
            [TestRefusalLabelling._source()],
        )

        assert answer == "Nhân viên được mười hai ngày phép năm."

    def test_verify_answer_ignores_a_preamble_only_rewrite(self):
        """A rewrite that is only a preamble is no rewrite at all."""
        llm = TestRefusalLabelling._llm_returning(
            '{"is_supported": false, "fixed_answer": "User Safety: safe", "reason": "thiếu căn cứ"}'
        )

        answer, verification = llm.verify_answer(
            "Bao nhiêu ngày phép năm?",
            "Nhân viên được mười hai ngày phép năm.",
            [TestRefusalLabelling._source()],
        )

        assert answer == "Tài liệu không cung cấp đủ thông tin để trả lời chắc chắn."
        assert verification == "no_evidence", verification


# ===========================================================================
# Real PDF ingestion still works end to end
# ===========================================================================

class TestRealPdfPath:
    def test_extract_chunk_index_search(self, tmp_path: Path):
        repo_root = Path(__file__).resolve().parents[2]
        candidates = sorted(repo_root.glob("*.pdf"), key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            pytest.skip("no real PDF in repo root")

        pages, page_count = extract_pdf_pages(candidates[0])
        assert page_count > 100
        chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)

        store = VectorStore(tmp_path / "idx")
        store.add_document("book", candidates[0].name, page_count, chunks[:400])

        hits = store.search("overfitting dropout regularization", top_k=4)
        assert hits
        assert hits[0].document_id == "book"
        assert hits[0].page >= 1
        assert hits[0].score > 0


# ===========================================================================
# dedupe_sources must not drop distinct evidence because of preview_page
# ===========================================================================

class TestDedupeKeepsDistinctProcedureChunks:
    """Regression for a silent evidence-loss bug.

    For procedure questions ``context_start_page`` rewinds ``preview_page`` to
    the start of the section. Two genuinely different chunks -- e.g. step 3 on
    page 10 and step 7 on page 14 of the same multi-step quy trình -- can both
    be rewound to ``preview_page=10``. Keying the dedupe set on ``preview_page``
    collapsed them into one entry and dropped the second chunk's evidence.
    """

    def _chunk(self, document_id: str, page: int, preview_page: int, text: str) -> SourceChunk:
        return SourceChunk(
            document_id=document_id,
            filename="quy-trinh.pdf",
            page=page,
            preview_page=preview_page,
            text=text,
            score=1.0,
        )

    def test_distinct_pages_with_same_preview_page_both_kept(self):
        sources = [
            self._chunk("doc1", page=10, preview_page=10, text="Bước 3: xác nhận đơn hàng với khách."),
            self._chunk("doc1", page=14, preview_page=10, text="Bước 7: gửi hàng và cập nhật trạng thái."),
        ]
        kept = dedupe_sources(sources)
        assert len(kept) == 2, [s.page for s in kept]
        assert {s.page for s in kept} == {10, 14}

    def test_long_page_split_into_several_chunks_kept(self):
        # One page split into several chunks must stay distinct (text differs).
        sources = [
            self._chunk("doc1", page=5, preview_page=5, text="Phần đầu của đoạn văn rất dài trên trang 5."),
            self._chunk("doc1", page=5, preview_page=5, text="Phần cuối của đoạn văn rất dài trên trang 5, khác hẳn ý."),
        ]
        kept = dedupe_sources(sources)
        assert len(kept) == 2, [s.text[:20] for s in kept]

    def test_exact_duplicate_still_collapses(self):
        sources = [
            self._chunk("doc1", page=10, preview_page=10, text="Nội dung trùng lặp exactly."),
            self._chunk("doc1", page=10, preview_page=14, text="Nội dung trùng lặp exactly."),
        ]
        kept = dedupe_sources(sources)
        assert len(kept) == 1

