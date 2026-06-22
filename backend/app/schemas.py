from pydantic import BaseModel


class DocumentSummary(BaseModel):
    id: str
    filename: str
    pages: int
    chunks: int


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    page: int
    preview_page: int | None = None
    text: str
    score: float | None = None


class ChatRequest(BaseModel):
    question: str
    document_id: str | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk]


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
