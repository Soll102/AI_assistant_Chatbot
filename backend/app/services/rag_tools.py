from dataclasses import dataclass

from app.schemas import SourceChunk
from app.services.vector_store import VectorStore


@dataclass(frozen=True)
class ToolPlan:
    name: str
    query: str
    reason: str = ""


@dataclass(frozen=True)
class ToolResult:
    name: str
    sources: list[SourceChunk]


class RagToolRunner:
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def run(self, plan: ToolPlan, top_k: int, document_id: str | None) -> ToolResult:
        if plan.name == "summarize_pdf":
            return ToolResult(
                name=plan.name,
                sources=self.store.search(build_summary_query(plan.query), top_k=top_k, document_id=document_id),
            )

        return ToolResult(
            name="search_pdf",
            sources=self.store.search(plan.query, top_k=top_k, document_id=document_id),
        )


def build_summary_query(query: str) -> str:
    return (
        f"{query} summary overview main ideas key points conclusion "
        "tóm tắt ý chính nội dung chính kết luận"
    )


def fallback_tool_plan(question: str) -> ToolPlan:
    lowered = question.lower()
    summary_terms = ["tóm tắt", "tom tat", "summary", "summarize", "overview", "ý chính", "y chinh"]
    if any(term in lowered for term in summary_terms):
        return ToolPlan(name="summarize_pdf", query=question, reason="summary intent")
    return ToolPlan(name="search_pdf", query=question, reason="default RAG search")
