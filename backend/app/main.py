from pathlib import Path
from shutil import copyfileobj

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import Settings, get_settings
from app.schemas import ChatMessage, ChatRequest, ChatResponse, ChatSession, CreateSessionRequest, DocumentSummary
from app.services.chat_history import ChatHistoryStore
from app.services.gemini_client import GeminiClient
from app.services.pdf_processor import PageText, chunk_pages, extract_pdf_pages, render_page_png
from app.services.vector_store import VectorStore

app = FastAPI(title="Multimodal RAG Chatbot")


@app.on_event("startup")
def startup() -> None:
    settings = get_settings()
    app.state.vector_store = VectorStore(settings.chroma_dir, settings.embedding_model)
    app.state.gemini = GeminiClient(settings.gemini_api_key, settings.gemini_model)
    app.state.chat_history = ChatHistoryStore(settings.chat_db_path)


@app.middleware("http")
async def add_cors_headers(request, call_next):
    return await call_next(request)


settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def vector_store() -> VectorStore:
    return app.state.vector_store


def gemini_client() -> GeminiClient:
    return app.state.gemini


def chat_history() -> ChatHistoryStore:
    return app.state.chat_history


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
) -> DocumentSummary:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF.")

    document_id = store.new_document_id()
    safe_name = Path(file.filename).name
    pdf_path = config.uploads_dir / f"{document_id}.pdf"

    with pdf_path.open("wb") as output:
        copyfileobj(file.file, output)

    pages, page_count = extract_pdf_pages(pdf_path)
    if config.enable_gemini_vision_fallback:
        pages = enrich_low_text_pages_with_vision(
            pdf_path=pdf_path,
            pages=pages,
            min_text_chars=config.vision_min_text_chars,
            gemini=gemini_client(),
        )

    chunks = chunk_pages(pages, config.chunk_size, config.chunk_overlap)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                "Không extract được text từ PDF. Nếu đây là PDF scan, hãy cấu hình "
                "GEMINI_API_KEY và bật ENABLE_GEMINI_VISION_FALLBACK=true."
            ),
        )

    return store.add_document(document_id, safe_name, page_count, chunks)


@app.get("/api/documents/{document_id}/file")
def get_document_file(document_id: str, config: Settings = Depends(get_settings)) -> FileResponse:
    pdf_path = config.uploads_dir / f"{document_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy PDF.")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{document_id}.pdf"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    config: Settings = Depends(get_settings),
    store: VectorStore = Depends(vector_store),
    gemini: GeminiClient = Depends(gemini_client),
    history: ChatHistoryStore = Depends(chat_history),
) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    session = history.get_or_create_session(
        session_id=request.session_id,
        title=question,
        document_id=request.document_id,
    )
    history.add_message(session.id, "user", question)
    history.update_title_from_question(session.id, question)

    sources = store.search(question, top_k=config.top_k, document_id=request.document_id)
    if not sources:
        answer = "Chưa tìm thấy nội dung liên quan trong tài liệu."
        history.add_message(session.id, "assistant", answer)
        return ChatResponse(session_id=session.id, answer=answer, sources=[])

    answer = gemini.answer(question, sources)
    history.add_message(session.id, "assistant", answer)
    return ChatResponse(session_id=session.id, answer=answer, sources=sources)


def enrich_low_text_pages_with_vision(
    pdf_path: Path,
    pages: list[PageText],
    min_text_chars: int,
    gemini: GeminiClient,
) -> list[PageText]:
    enriched: list[PageText] = []
    for page in pages:
        if len(page.text.strip()) >= min_text_chars:
            enriched.append(page)
            continue

        image_bytes = render_page_png(pdf_path, page.page)
        vision_text = gemini.extract_page_from_image(image_bytes, page.page).strip()
        combined = "\n\n".join(part for part in [page.text.strip(), vision_text] if part)
        enriched.append(PageText(page=page.page, text=combined))

    return enriched
