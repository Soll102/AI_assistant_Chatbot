from dataclasses import dataclass, field
import re

from app.schemas import DocumentSummary, SourceChunk
from app.services.vector_store import VectorStore, confident_sources


@dataclass(frozen=True)
class ToolPlan:
    name: str
    query: str
    reason: str = ""


@dataclass(frozen=True)
class ToolResult:
    name: str
    sources: list[SourceChunk]
    documents: list[DocumentSummary] = field(default_factory=list)


LIST_PDF_TERMS = [
    "có những pdf",
    "những pdf nào",
    "pdf nào",
    "danh sách pdf",
    "danh sach pdf",
    "liệt kê pdf",
    "liet ke pdf",
    "có tài liệu nào",
    "co tai lieu nao",
    "tài liệu nào",
    "tai lieu nao",
    "bao nhiêu pdf",
    "bao nhieu pdf",
    "which pdf",
    "list pdf",
    "list of pdf",
]


class RagToolRunner:
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def run(self, plan: ToolPlan, top_k: int, document_id: str | None) -> ToolResult:
        if plan.name == "list_pdfs":
            return ToolResult(name=plan.name, sources=[], documents=self.store.list_documents())

        if plan.name == "summarize_pdf":
            # Tóm tắt cần context rộng (tới top_k chunks cho LLM); việc siết
            # hiển thị còn 1-2 gợi ý do main.py đảm nhiệm.
            return ToolResult(
                name=plan.name,
                sources=self.store.search(build_summary_query(plan.query), top_k=top_k, document_id=document_id),
            )

        # Hỏi đáp chi tiết: siết còn tối đa 1-2 đoạn chắc chắn nhất.
        pool = self.store.search(plan.query, top_k=top_k, document_id=document_id)
        return ToolResult(name="search_pdf", sources=confident_sources(pool, plan.query))


def build_summary_query(query: str) -> str:
    return (
        f"{query} summary overview main ideas key points conclusion "
        "tóm tắt ý chính nội dung chính kết luận"
    )


def quick_tool_plan(question: str) -> ToolPlan | None:
    lowered = question.lower()
    if any(term in lowered for term in LIST_PDF_TERMS):
        return ToolPlan(name="list_pdfs", query=question, reason="list-intent keywords")
    summary_terms = ["tóm tắt", "tom tat", "summary", "summarize", "overview", "ý chính", "y chinh"]
    if any(term in lowered for term in summary_terms):
        return ToolPlan(name="summarize_pdf", query=question, reason="summary intent")
    return None


def fallback_tool_plan(question: str) -> ToolPlan:
    plan = quick_tool_plan(question)
    if plan is not None:
        return plan
    return ToolPlan(name="search_pdf", query=question, reason="default RAG search")


def normalize_name(text: str, strip_extension: bool = False) -> str:
    if strip_extension and "." in text:
        text = text.rsplit(".", 1)[0]
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def match_document_by_name(question: str, documents: list[DocumentSummary]) -> DocumentSummary | None:
    normalized_question = normalize_name(question)
    best: DocumentSummary | None = None
    best_length = 0
    for document in documents:
        normalized_name = normalize_name(document.filename, strip_extension=True)
        if len(normalized_name) >= 4 and normalized_name in normalized_question:
            if len(normalized_name) > best_length:
                best = document
                best_length = len(normalized_name)
    return best
