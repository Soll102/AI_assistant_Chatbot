from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    id: str
    filename: str
    pages: int
    chunks: int


class ReindexResult(BaseModel):
    """Outcome of reconciling the index with the PDFs on disk."""

    indexed: list[DocumentSummary] = []
    errors: list[str] = []


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    page: int
    preview_page: int | None = None
    text: str
    score: float | None = None


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=500)
    document_id: str | None = None
    # Multi-PDF workflow: explicit set of documents to search.
    #   None (and legacy document_id=None) -> search ALL documents.
    #   [] (empty list) -> also treated as ALL for convenience.
    #   [id1, id2, ...] -> fan-out retrieval across those docs only.
    document_ids: list[str] | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk]
    tool_name: str | None = None
    verification: str | None = None
    # Multi-PDF observability: which document ids actually contributed
    # sources to this answer (subset of requested ids, or all-doc search).
    documents_used: list[str] = []
    # Requested documents that did NOT make it into the context because the
    # per-turn context budget was exhausted. Non-empty means the answer may
    # be incomplete — the UI must surface this instead of staying silent.
    #
    # Note: a document whose retrieved chunk was dropped by near-duplicate
    # dedupe is NOT reported here. It was read and it matched; it simply
    # repeated something another document already said, so it contributes
    # nothing distinct to the answer.
    documents_skipped: list[str] = []
    # Set when the retrieval query had to be widened with the previous user
    # turn (follow-up question with no lexical overlap on its own).
    query_rewritten: bool = False


class ChatSession(BaseModel):
    id: str
    title: str
    document_id: str | None = None
    created_at: str
    updated_at: str


class ChatMessage(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    created_at: str


class CreateSessionRequest(BaseModel):
    title: str | None = None
    document_id: str | None = None
