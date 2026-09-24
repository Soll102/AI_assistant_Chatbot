import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import Response as FastAPIResponse

from app.config import Settings, get_settings
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSession,
    CreateSessionRequest,
    DocumentSummary,
    ReindexResult,
)
from app.services.chat_history import ChatHistoryStore
from app.services.llm_client import ChatState, LLMClient, is_api_error
from app.services.pdf_processor import (
    PageText,
    chunk_pages,
    display_name_from_pdf,
    extract_pdf_pages,
    render_page_png,
)
from app.services.rag_tools import (
    RagToolRunner,
    ToolPlan,
    fallback_tool_plan,
    match_document_by_name,
    match_documents_by_name,
    quick_tool_plan,
    rescue_queries,
    resolve_document_ids,
)
from app.services.vector_store import VectorStore, confident_sources

MAX_BATCH_FILES = 20
UPLOAD_CHUNK_BYTES = 1024 * 1024
# document_id là uuid4 hex do server tự sinh — mọi giá trị khác (kể cả
# "../secret") đều bị từ chối để chặn path traversal ở /file và /delete.
DOCUMENT_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")

logger = logging.getLogger("app.ingest")


def _require_api_key(request: Request, config: Settings = Depends(get_settings)) -> None:
    """Optional guard: chỉ bật khi backend có API_KEY (deploy public).

    Bỏ trống = local-only, giữ tương thích toàn bộ test cũ.
    """
    if not config.api_key:
        return
    provided = request.headers.get("x-api-key", "")
    if provided != config.api_key:
        raise HTTPException(status_code=401, detail="Thiếu hoặc sai API key.")


def _resolve_pdf_path(config: Settings, document_id: str) -> Path:
    if not DOCUMENT_ID_RE.match(document_id or ""):
        raise HTTPException(status_code=404, detail="Không tìm thấy PDF.")
    return config.uploads_dir / f"{document_id}.pdf"


def _safe_filename(document_id: str) -> str:
    # Content-Disposition dùng id đã validate nên không còn header injection
    # qua '"', '\r', '\n' như trước.
    return f"{document_id}.pdf"


def _init_state(target) -> None:
    settings = get_settings()
    if not hasattr(target, "vector_store"):
        target.vector_store = VectorStore(
            settings.index_dir,
            max_context_chunks=settings.max_context_chunks,
            min_query_coverage=settings.min_query_coverage,
            evidence_metric=settings.evidence_metric,
        )
    if not hasattr(target, "llm"):
        target.llm = LLMClient(
            settings.openrouter_api_key, settings.openrouter_model, settings.openrouter_fallback_model
        )
    if not hasattr(target, "chat_history"):
        target.chat_history = ChatHistoryStore(settings.chat_db_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_state(app.state)
    yield


app = FastAPI(title="Multimodal RAG Chatbot", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # Không dùng cookie/auth nên tắt credentials để thu hẹp bề mặt CORS.
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


def vector_store() -> VectorStore:
    _init_state(app.state)
    return app.state.vector_store


def llm_client() -> LLMClient:
    _init_state(app.state)
    return app.state.llm


def chat_history() -> ChatHistoryStore:
    _init_state(app.state)
    return app.state.chat_history


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    config = get_settings()
    return {
        "status": "ok",
        "max_upload_mb": config.max_upload_mb,
        "max_context_chunks": config.max_context_chunks,
        "answer_verification": config.enable_answer_verification,
    }


@app.get("/api/documents", response_model=list[DocumentSummary])
def list_documents(store: VectorStore = Depends(vector_store)) -> list[DocumentSummary]:
    return store.list_documents()


@app.get("/api/chat/sessions", response_model=list[ChatSession])
def list_chat_sessions(history: ChatHistoryStore = Depends(chat_history)) -> list[ChatSession]:
    return history.list_sessions()


@app.post("/api/chat/sessions", response_model=ChatSession)
def create_chat_session(
    request: CreateSessionRequest,
    history: ChatHistoryStore = Depends(chat_history),
) -> ChatSession:
    return history.create_session(title=request.title or "Chat mới", document_id=request.document_id)


@app.get("/api/chat/sessions/{session_id}/messages", response_model=list[ChatMessage])
def list_chat_messages(
    session_id: str,
    history: ChatHistoryStore = Depends(chat_history),
) -> list[ChatMessage]:
    if not history.get_session(session_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy lịch sử chat.")
    return history.list_messages(session_id)


@app.delete("/api/chat/sessions/{session_id}", status_code=204)
def delete_chat_session(
    session_id: str,
    history: ChatHistoryStore = Depends(chat_history),
) -> None:
    if not history.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy lịch sử chat.")


@app.post("/api/documents", response_model=DocumentSummary)
def upload_document(
    file: UploadFile = File(...),
    config: Settings = Depends(get_settings),
    store: VectorStore = Depends(vector_store),
    _auth: None = Depends(_require_api_key),
) -> DocumentSummary:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF.")
    # Validate magic bytes tối thiểu: tên .pdf nhưng ruột text phải 422 sớm
    # thay vì index rác. Đọc peek 5 byte đầu qua file.file mà không tiêu thụ.
    try:
        pos = file.file.tell()
    except Exception:
        pos = None
    try:
        head = file.file.read(5)
    except Exception:
        head = b""
    finally:
        try:
            if pos is not None:
                file.file.seek(pos)
        except Exception:
            pass
    if head and not head.startswith(b"%PDF"):
        raise HTTPException(status_code=422, detail="File không phải PDF hợp lệ (thiếu magic %PDF).")

    document_id = store.new_document_id()
    safe_name = Path(file.filename).name
    pdf_path = config.uploads_dir / f"{document_id}.pdf"
    save_upload(file, pdf_path, config.max_upload_bytes)
    try:
        return ingest_stored_pdf(
            document_id=document_id,
            safe_name=safe_name,
            pdf_path=pdf_path,
            config=config,
            store=store,
        )
    except Exception:
        # Fresh upload lỗi (422/500) phải dọn file mồ côi, nếu không uploads/
        # phình với PDF unindexed mà /status cứ báo thiếu mãi.
        _safe_unlink(pdf_path)
        raise
    finally:
        try:
            file.file.close()
        except Exception:
            pass


@app.get("/api/documents/status")
def documents_status(
    config: Settings = Depends(get_settings),
    store: VectorStore = Depends(vector_store),
) -> dict[str, Any]:
    """Indexed vs on-disk documents, so the UI can offer to repair the index."""
    unindexed = _unindexed_pdfs(config, store)
    return {
        "indexed": len(store.list_documents()),
        "unindexed": [path.stem for path in unindexed],
        "unindexed_count": len(unindexed),
    }


@app.post("/api/documents/reindex", response_model=ReindexResult)
def reindex_documents(
    config: Settings = Depends(get_settings),
    store: VectorStore = Depends(vector_store),
    _auth: None = Depends(_require_api_key),
) -> ReindexResult:
    """Index every stored PDF that is missing from the index.

    Idempotent. Used to recover a library uploaded before the index format
    changed, and to repair a half-finished ingest after a crash.
    """
    indexed, errors = reconcile_index(config, store)
    return ReindexResult(indexed=indexed, errors=errors)


@app.post("/api/documents/batch", response_model=list[DocumentSummary])
def upload_documents_batch(
    files: list[UploadFile] = File(...),
    config: Settings = Depends(get_settings),
    store: VectorStore = Depends(vector_store),
    response: FastAPIResponse = None,
    _auth: None = Depends(_require_api_key),
) -> list[DocumentSummary]:
    """Multi-PDF ingest: upload nhiều PDF trong 1 request.

    Mỗi file được index độc lập (doc_id riêng, chunks mang page metadata +
    document_id). File lỗi được bỏ qua kèm lý do thay vì fail cả batch khi
    còn ít nhất 1 file thành công.

    Phần extract + chunk (CPU-bound, chậm) chạy song song
    ``settings.ingest_workers`` luồng; phần ghi vào store được VectorStore
    tự khoá nên an toàn.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Chưa có file nào được gửi.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"Mỗi batch tối đa {MAX_BATCH_FILES} file PDF.")

    jobs: list[tuple[str, str, Path]] = []
    errors: list[str] = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            errors.append(f"{file.filename or '?'}: chỉ hỗ trợ file PDF.")
            continue
        document_id = store.new_document_id()
        safe_name = Path(file.filename).name
        pdf_path = config.uploads_dir / f"{document_id}.pdf"
        try:
            save_upload(file, pdf_path, config.max_upload_bytes)
        except HTTPException as exc:
            errors.append(f"{safe_name}: {exc.detail}")
            continue
        jobs.append((document_id, safe_name, pdf_path))

    if not jobs:
        raise HTTPException(status_code=422, detail="; ".join(errors) or "Không index được file nào.")

    results = _run_ingest_jobs(jobs, config, store)
    # Một lần ghi đĩa cho cả batch thay vì mỗi file một lần (mỗi lần ghi là
    # một lần serialize toàn bộ store).
    store.flush()

    summaries: list[DocumentSummary] = []
    for summary, error in results:
        if summary is not None:
            summaries.append(summary)
        elif error:
            errors.append(error)

    if not summaries:
        raise HTTPException(status_code=422, detail="; ".join(errors) or "Không index được file nào.")
    if errors:
        # Trước đây errors bị nuốt khi có >=1 file thành công nên UI chỉ đoán
        # "failed = files - summaries". Log + header để UI hiện đúng file nào.
        logger.warning("Batch partial failure: %s", "; ".join(errors))
        if response is not None:
            try:
                response.headers["X-Batch-Errors"] = "; ".join(errors)[:2000]
                response.headers["X-Batch-Failed"] = str(len(errors))
            except Exception:
                pass
    return summaries


def _unindexed_pdfs(config: Settings, store: VectorStore) -> list[Path]:
    """PDFs sitting in ``uploads/`` that the index does not know about.

    Every stored PDF is named ``<document_id>.pdf``, so the id is recoverable
    from the filename and the check is exact. This is how the app recovers a
    library that was uploaded before the index format changed (the old
    ChromaDB store was dropped in favour of the in-process BM25 index).
    """
    indexed = {document.id for document in store.list_documents()}
    return [
        path
        for path in sorted(config.uploads_dir.glob("*.pdf"))
        if path.stem not in indexed
    ]


def reconcile_index(
    config: Settings,
    store: VectorStore,
) -> tuple[list[DocumentSummary], list[str]]:
    """Index every stored PDF that is missing from the index.

    Returns ``(indexed, errors)``. Safe to call repeatedly: already-indexed
    documents are skipped, so this doubles as a repair step after a crash
    between writing the PDF and persisting the index.
    """
    jobs = [
        (path.stem, display_name_from_pdf(path), path)
        for path in _unindexed_pdfs(config, store)
    ]
    if not jobs:
        return [], []

    # delete_on_failure=False: these PDFs are the user's stored documents, not a
    # fresh upload. A failed re-index must leave the file exactly where it was.
    # use_vision=False: repairing an index must not depend on the network or
    # spend hundreds of vision calls the user never asked for.
    results = _run_ingest_jobs(jobs, config, store, delete_on_failure=False, use_vision=False)
    store.flush()

    summaries: list[DocumentSummary] = []
    errors: list[str] = []
    for summary, error in results:
        if summary is not None:
            summaries.append(summary)
        elif error:
            errors.append(error)
    return summaries, errors


def _safe_unlink(path: Path) -> None:
    """Delete a file without letting a locked file mask the real failure.

    On Windows an unreadable PDF is often still held open by a viewer, so
    ``unlink`` raises ``PermissionError``. Cleanup is best-effort: the ingest
    error we are already reporting is the useful information.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _run_ingest_jobs(
    jobs: list[tuple[str, str, Path]],
    config: Settings,
    store: VectorStore,
    delete_on_failure: bool = True,
    progress: Callable[[int, DocumentSummary | None, str | None], None] | None = None,
    use_vision: bool = True,
) -> list[tuple[DocumentSummary | None, str | None]]:
    """Extract + chunk + index every job, keeping the caller's file order.

    ``delete_on_failure`` must be ``True`` only when the PDF was just received
    from the client (a file we cannot read is worthless). When re-indexing PDFs
    that are already stored, pass ``False``: a failed re-index must never
    destroy the user's only copy of a document.

    ``progress`` is called as each job finishes, with its index in ``jobs``.
    Indexing a large PDF takes minutes, so callers that report to a human need
    this to avoid looking hung.
    """
    workers = max(1, min(config.ingest_workers, len(jobs)))
    results: list[tuple[DocumentSummary | None, str | None]] = [(None, None)] * len(jobs)

    if workers == 1:
        for index, job in enumerate(jobs):
            results[index] = _ingest_job(job, config, store, delete_on_failure, use_vision)
            if progress is not None:
                progress(index, *results[index])
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_ingest_job, job, config, store, delete_on_failure, use_vision): index
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                _document_id, safe_name, pdf_path = jobs[index]
                if delete_on_failure:
                    _safe_unlink(pdf_path)
                results[index] = (None, f"{safe_name}: {exc.__class__.__name__}: {exc}")
            if progress is not None:
                progress(index, *results[index])
    return results


def _ingest_job(
    job: tuple[str, str, Path],
    config: Settings,
    store: VectorStore,
    delete_on_failure: bool = True,
    use_vision: bool = True,
) -> tuple[DocumentSummary | None, str | None]:
    document_id, safe_name, pdf_path = job
    try:
        summary = ingest_stored_pdf(
            document_id=document_id,
            safe_name=safe_name,
            pdf_path=pdf_path,
            config=config,
            store=store,
            save=False,
            use_vision=use_vision,
        )
        return summary, None
    except HTTPException as exc:
        if delete_on_failure:
            _safe_unlink(pdf_path)
        return None, f"{safe_name}: {exc.detail}"
    except Exception as exc:  # pragma: no cover - defensive
        if delete_on_failure:
            _safe_unlink(pdf_path)
        return None, f"{safe_name}: {exc.__class__.__name__}: {exc}"


def save_upload(file: UploadFile, dest: Path, max_bytes: int) -> int:
    """Stream an upload to disk, aborting as soon as it exceeds ``max_bytes``.

    Reads in 1MB blocks so a huge (or hostile) file never lands in memory
    whole, and deletes the partial file when the limit trips.
    """
    written = 0
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as output:
            while True:
                block = file.file.read(UPLOAD_CHUNK_BYTES)
                if not block:
                    break
                written += len(block)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File vượt quá giới hạn {max_bytes // (1024 * 1024)}MB. "
                            "Tăng MAX_UPLOAD_MB trong backend/.env nếu cần."
                        ),
                    )
                output.write(block)
    except HTTPException:
        _safe_unlink(dest)
        raise
    except OSError as exc:
        _safe_unlink(dest)
        raise HTTPException(status_code=500, detail=f"Không ghi được file lên đĩa: {exc}") from exc
    return written


def ingest_stored_pdf(
    document_id: str,
    safe_name: str,
    pdf_path: Path,
    config: Settings,
    store: VectorStore,
    save: bool = True,
    use_vision: bool = True,
) -> DocumentSummary:
    """Shared single-PDF pipeline used by both /api/documents and /batch.

    ``use_vision=False`` skips the vision fallback even when it is enabled.
    Re-indexing uses this: repairing an index should be a fast, offline
    operation, not a few hundred calls to a vision model.
    """
    pages, page_count = extract_pdf_pages(pdf_path)
    if use_vision and config.enable_gemini_vision_fallback:
        # Vision là enhancement optional: render/network fail không được làm
        # chết cả ingest (trước đây raise -> _ingest_job xoá luôn file upload).
        try:
            pages = enrich_low_text_pages_with_vision(
                pdf_path=pdf_path,
                pages=pages,
                min_text_chars=config.vision_min_text_chars,
                llm=llm_client(),
                max_pages=config.vision_max_pages,
            )
        except Exception as exc:
            logger.warning("Vision fallback thất bại, dùng text gốc: %s", exc)

    chunks = chunk_pages(pages, config.chunk_size, config.chunk_overlap)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                "Không extract được text từ PDF. Nếu đây là PDF scan, hãy cấu hình "
                "OPENROUTER_API_KEY và bật ENABLE_GEMINI_VISION_FALLBACK=true."
            ),
        )

    return store.add_document(document_id, safe_name, page_count, chunks, save=save)


@app.get("/api/documents/{document_id}/file")
def get_document_file(document_id: str, config: Settings = Depends(get_settings)) -> FileResponse:
    pdf_path = _resolve_pdf_path(config, document_id)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy PDF.")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{_safe_filename(document_id)}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    config: Settings = Depends(get_settings),
    store: VectorStore = Depends(vector_store),
    _auth: None = Depends(_require_api_key),
) -> None:
    pdf_path = _resolve_pdf_path(config, document_id)
    deleted_vectors = store.delete_document(document_id)
    deleted_file = False
    if pdf_path.exists():
        # Use _safe_unlink, not a bare unlink: on Windows a PDF still held open
        # by a viewer raises PermissionError, which would otherwise turn a
        # successful vector delete into a 500. The only copy of the document
        # must not be destroyed by an unrelated lock, so a locked file is left
        # on disk for the user to retry later.
        _safe_unlink(pdf_path)
        deleted_file = not pdf_path.exists()
    if not deleted_vectors and not deleted_file:
        raise HTTPException(status_code=404, detail="Không tìm thấy PDF.")
    if deleted_vectors and not deleted_file and pdf_path.exists():
        # Vector đã xoá nhưng file bị khoá (Windows): báo 409 để user thử lại,
        # thay vì 204 rồi lần reconcile sau resurrect doc gây khó hiểu.
        raise HTTPException(
            status_code=409,
            detail="Đã xoá index nhưng file PDF đang bị khoá (có thể đang mở). Đóng file và xoá lại.",
        )


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    config: Settings = Depends(get_settings),
    store: VectorStore = Depends(vector_store),
    llm: LLMClient = Depends(llm_client),
    history: ChatHistoryStore = Depends(chat_history),
    _auth: None = Depends(_require_api_key),
) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    try:
        session = history.get_or_create_session(
            session_id=request.session_id,
            title=question,
            document_id=request.document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # Snapshot prior turns BEFORE recording this question, so the LLM sees the
    # conversation that led here but not the current question twice.
    prior_messages = history.list_messages(session.id)
    try:
        history.add_message(session.id, "user", question)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    history.update_title_from_question(session.id, question)

    # Multi-PDF scope resolution (backward compatible):
    #  - document_ids=[...] (new) wins; legacy document_id merged in.
    #  - None/[] means "all documents".
    #  - Name-match is only a fallback for legacy single-doc callers that
    #    did not specify any scope, và chỉ khi câu hỏi có cue phạm vi
    #    ("trong file X", "theo X"...) để tránh bóp ALL->1 doc oan.
    resolved_ids = resolve_document_ids(
        document_id=request.document_id,
        document_ids=request.document_ids,
    )
    all_documents = store.list_documents()
    if resolved_ids is None:
        matched_docs = match_documents_by_name(question, all_documents)
        if matched_docs and _has_scope_cue(question):
            if len(matched_docs) == 1:
                resolved_ids = [matched_docs[0].id]
            else:
                # Compare nhắc 2+ tên file: giữ hết để fan-out, không giữ 1.
                resolved_ids = [doc.id for doc in matched_docs]

    quick = quick_tool_plan(question)
    if quick is not None:
        state = ChatState(tool=quick)
    elif not config.enable_tool_planning:
        state = ChatState(tool=fallback_tool_plan(question))
    else:
        state = llm.start_chat(question, has_document=True)
    if state.answer is not None:
        # start_chat trả text thẳng chưa qua retrieval (hallucination risk).
        # Grounding lại: tìm sources, không có thì từ chối thay vì trả bừa.
        grounding = RagToolRunner(store).run(
            fallback_tool_plan(question), top_k=config.top_k, document_ids=resolved_ids
        )
        if not grounding.sources:
            answer = "Chưa tìm thấy nội dung liên quan trong tài liệu."
            history.add_message(session.id, "assistant", answer)
            return ChatResponse(
                session_id=session.id,
                answer=answer,
                sources=[],
                tool_name="search_pdf",
                verification="skipped",
                documents_used=[],
            )
        state = ChatState(tool=fallback_tool_plan(question))

    tool_plan = state.tool
    if tool_plan.name == "list_pdfs":
        answer = llm.list_documents_answer(store.list_documents())
        history.add_message(session.id, "assistant", answer)
        return ChatResponse(
            session_id=session.id,
            answer=answer,
            sources=[],
            tool_name=tool_plan.name,
            verification="skipped",
            documents_used=[],
        )

    runner = RagToolRunner(store)
    tool_result = runner.run(tool_plan, top_k=config.top_k, document_ids=resolved_ids)

    # Follow-up rescue: "còn cái kia thì sao?" has no lexical overlap with any
    # chunk, so retrieval comes back empty and the assistant would refuse.
    # Re-running with the previous user turn blended in recovers the topic
    # without paying for an LLM query-rewrite call. Two candidates are tried,
    # cheapest first: the blend, then the previous question alone (blending
    # lengthens the query and can trip the evidence gate).
    query_rewritten = False
    if not tool_result.sources and prior_messages:
        for candidate in rescue_queries(question, prior_messages):
            retry = runner.run(
                ToolPlan(name=tool_plan.name, query=candidate, reason="follow-up widened with history"),
                top_k=config.top_k,
                document_ids=resolved_ids,
            )
            if retry.sources:
                tool_result = retry
                query_rewritten = True
                break

    sources = tool_result.sources
    if not sources:
        answer = "Chưa tìm thấy nội dung liên quan trong tài liệu."
        history.add_message(session.id, "assistant", answer)
        # Single-doc scope trỏ id lạ: trước đây skipped=[] nên im lặng như
        # "không có gì". Báo rõ id đó để UI phân biệt với "đã tìm mà rỗng".
        skipped_ids = list(tool_result.documents_skipped)
        if resolved_ids is not None and len(resolved_ids) == 1:
            known_ids = {doc.id for doc in all_documents}
            if resolved_ids[0] not in known_ids and resolved_ids[0] not in skipped_ids:
                skipped_ids = [resolved_ids[0]]
        return ChatResponse(
            session_id=session.id,
            answer=answer,
            sources=[],
            tool_name=tool_result.name,
            verification="skipped",
            documents_used=[],
            documents_skipped=skipped_ids,
        )

    sources = enrich_formula_sources(question, sources, config, llm)
    answer = llm.finalize_with_sources(
        question,
        sources,
        tool_plan,
        state.tool_call_id,
        trim_history(prior_messages, config.history_turns),
    )
    verification = "disabled"
    if config.enable_answer_verification:
        answer, verification = llm.verify_answer(question, answer, sources)
    history.add_message(session.id, "assistant", answer)
    # Hiển thị tối đa 1-2 gợi ý chắc chắn nhất; LLM vẫn dùng full pool ở trên.
    # Với compare_pdfs (đa tài liệu), giữ lại ít nhất 1 nguồn mỗi tài liệu để
    # UI thể hiện được độ phủ cross-doc.
    if tool_result.name == "compare_pdfs":
        shown_sources = dedupe_compare_sources(sources, question)
    else:
        shown_sources = confident_sources(sources, question)
    return ChatResponse(
        session_id=session.id,
        answer=answer,
        sources=shown_sources,
        tool_name=tool_result.name,
        verification=verification,
        documents_used=sorted({s.document_id for s in shown_sources}),
        documents_skipped=list(tool_result.documents_skipped),
        query_rewritten=query_rewritten,
    )


_SCOPE_CUE_RE = re.compile(
    r"(trong\s+(file|tài liệu|tai lieu|pdf)|theo\s+(file|tài liệu|tai lieu)|"
    r"ở\s+(file|tài liệu|tai lieu)|từ\s+(file|tài liệu|tai lieu)|"
    r"tài liệu\s+\w+\s+nói|file\s+\w+)",
    re.IGNORECASE,
)


def _has_scope_cue(question: str) -> bool:
    return _SCOPE_CUE_RE.search(question or "") is not None


def trim_history(messages: list[ChatMessage], turns: int) -> list[ChatMessage]:
    """Keep the last ``turns`` user/assistant pairs for LLM context."""
    if not turns or not messages:
        return []
    trimmed = list(messages)[-(turns * 2) :]
    # Cắt mù có thể bắt đầu bằng assistant (một số provider từ chối
    # conversation mở đầu bằng assistant). Bỏ leading assistant turns.
    while trimmed and getattr(trimmed[0], "role", "") != "user":
        trimmed.pop(0)
    return trimmed


def enrich_low_text_pages_with_vision(
    pdf_path: Path,
    pages: list[PageText],
    min_text_chars: int,
    llm: LLMClient,
    max_pages: int = 0,
) -> list[PageText]:
    """Fill in pages that have almost no extractable text by reading them as images.

    Each page costs one **sequential network round trip** to a vision model, so
    this is by far the slowest stage of ingest: a 682-page textbook with 78
    figure pages took over ten minutes and looked like a hang. ``max_pages``
    bounds that cost — set it to ``0`` for no limit (the old behaviour, only
    sensible for a single small document).
    """
    low_text = [page for page in pages if len(page.text.strip()) < min_text_chars]
    budget = len(low_text) if max_pages <= 0 else min(max_pages, len(low_text))
    if budget < len(low_text):
        logger.warning(
            "PDF có %d trang ít chữ nhưng chỉ đọc ảnh %d trang đầu (VISION_MAX_PAGES=%d); "
            "%d trang còn lại giữ nguyên text trích được. Tăng VISION_MAX_PAGES nếu cần.",
            len(low_text),
            budget,
            max_pages,
            len(low_text) - budget,
        )

    used = 0
    enriched: list[PageText] = []
    for page in pages:
        if len(page.text.strip()) >= min_text_chars or used >= budget:
            enriched.append(page)
            continue

        used += 1
        started = time.perf_counter()
        image_bytes = render_page_png(pdf_path, page.page)
        vision_text = llm.extract_page_from_image(image_bytes, page.page).strip()
        combined = "\n\n".join(part for part in [page.text.strip(), vision_text] if part)
        enriched.append(PageText(page=page.page, text=combined))
        logger.info(
            "vision page %d/%d of %s in %.1fs",
            used,
            budget,
            pdf_path.name,
            time.perf_counter() - started,
        )

    return enriched


def enrich_formula_sources(
    question: str,
    sources: list,
    config: Settings,
    llm: LLMClient,
) -> list:
    if not should_read_formula_from_page_image(question) or not llm.supports_vision():
        return sources
    if not sources:
        return sources

    try:
        enriched_sources = list(sources)
        source = enriched_sources[0]
        try:
            pdf_path = _resolve_pdf_path(config, str(getattr(source, "document_id", "")))
        except HTTPException:
            return sources
        if not pdf_path.exists():
            return sources

        page_number = int(source.preview_page or source.page or 1)
        if page_number < 1:
            return sources
        image_bytes = render_page_png(pdf_path, page_number)
        visual_text = llm.extract_page_from_image(image_bytes, page_number).strip()
        if visual_text and not is_api_error(visual_text):
            source.text = (
                f"{source.text}\n\n"
                f"[Nội dung đọc thêm từ ảnh trang {page_number}, dùng cho công thức/hình ảnh]\n"
                f"{visual_text}"
            )
    except Exception as exc:
        # Enrich là optional: lỗi render/vision không được 500 sau khi đã
        # retrieval thành công. Trước đây crash tại đây sau khi đã tốn LLM.
        logger.warning("Enrich formula thất bại, dùng text gốc: %s", exc)
        return sources

    return enriched_sources


def should_read_formula_from_page_image(question: str) -> bool:
    lowered = question.lower()
    formula_terms = [
        "công thức",
        "cong thuc",
        "formula",
        "equation",
        "phương trình",
        "phuong trinh",
        "ký hiệu",
        "ki hieu",
        "latex",
    ]
    return any(term in lowered for term in formula_terms)


def dedupe_compare_sources(sources: list, question: str) -> list:
    """Keep at least one top source per document for compare_pdfs answers.

    Falls back to the :func:`confident_sources` pool when a document has no
    confident hit, so cross-document coverage is preserved for the quality
    tests and the UI. Capped at 4 entries so the UI stays readable.
    """
    if not sources:
        return []

    # Group by document, keep best-ranked chunk per doc (sources are already
    # ranked by search_many round-robin relevance).
    best_per_doc: dict[str, object] = {}
    for source in sources:
        key = getattr(source, "document_id", "")
        if key not in best_per_doc:
            best_per_doc[key] = source

    # Order docs by their best score so the most relevant doc comes first,
    # but every contributing doc stays visible.
    ordered = sorted(
        best_per_doc.values(),
        key=lambda s: (getattr(s, "score", 0) or 0.0),
        reverse=True,
    )
    shortlist = list(ordered[:4])

    for source in confident_sources(sources, question):
        # Compare on the true chunk page, not preview_page: context_start_page()
        # rewinds preview_page for procedure questions, so two distinct chunks
        # (different pages) can share a preview_page and would be wrongly
        # collapsed as duplicates of each other.
        duplicate = any(
            getattr(existing, "document_id", None) == getattr(source, "document_id", None)
            and getattr(existing, "page", None) == getattr(source, "page", None)
            for existing in shortlist
        )
        if not duplicate and len(shortlist) < 4:
            shortlist.append(source)

    return shortlist
