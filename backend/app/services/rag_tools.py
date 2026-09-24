from dataclasses import dataclass, field
import re

from app.schemas import DocumentSummary, SourceChunk
from app.services.vector_store import VectorStore, content_tokens


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
    # Requested documents that could not be fitted into the context budget.
    documents_skipped: list[str] = field(default_factory=list)


COMPARE_TERMS = [
    "so sánh",
    "so sanh",
    "đối chiếu",
    "doi chieu",
    "khác nhau",
    "khac nhau",
    "giống nhau",
    "giong nhau",
    "compare",
    "comparison",
    "difference between",
    "differences between",
    "contrast",
    "versus",
    " vs ",
    "across documents",
    "across pdfs",
    "giữa các tài liệu",
    "giua cac tai lieu",
    "giữa các pdf",
    # "tổng hợp"/"tong hop" cố ý KHÔNG ở đây: đó là summary 1-doc, trước
    # đây bị ép thành compare và fan-out toàn bộ docs.
]

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
    # Bare "tài liệu nào"/"tai lieu nao" cố ý loại: câu "tài liệu nào nói
    # về X?" là search, không phải list. Chỉ match các cụm list rõ ràng.
    "bao nhiêu pdf",
    "bao nhieu pdf",
    "which pdf",
    "list pdf",
    "list of pdf",
]


class RagToolRunner:
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def run(
        self,
        plan: ToolPlan,
        top_k: int,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> ToolResult:
        """Run a tool plan against one, several, or all documents.

        Backward compatible: old callers pass only ``document_id``.
        New multi-PDF callers pass ``document_ids`` (list). Resolution:
          - document_ids non-empty -> fan-out across those ids.
          - else document_id set -> single-doc search.
          - else -> all documents.
        """
        resolved = resolve_document_ids(document_id=document_id, document_ids=document_ids)

        if plan.name == "list_pdfs":
            return ToolResult(name=plan.name, sources=[], documents=self.store.list_documents())

        # summarize / compare need cross-document coverage, so an unspecified
        # scope means "fan out over every known document", not "global top-k".
        coverage_scope = resolved if resolved is not None else self._all_document_ids()

        if plan.name == "summarize_pdf":
            # Tóm tắt cần context rộng (tới top_k chunks cho LLM); việc siết
            # hiển thị còn 1-2 gợi ý do main.py đảm nhiệm.
            sources, skipped = self._fan_out(build_summary_query(plan.query), top_k, coverage_scope)
            return ToolResult(name=plan.name, sources=sources, documents_skipped=skipped)

        if plan.name == "compare_pdfs":
            # So sánh / tổng hợp đa tài liệu: bắt buộc quota công bằng để
            # mỗi tài liệu đều có mặt trong context gửi cho LLM.
            sources, skipped = self._fan_out(plan.query, top_k, coverage_scope)
            return ToolResult(name="compare_pdfs", sources=sources, documents_skipped=skipped)

        # Hỏi đáp chi tiết (search_pdf). Trả về cả pool (không siết còn 1-2
        # đoạn ở đây) để LLM có đủ bằng chứng cho câu hỏi kiểu "liệt kê các
        # bước"; main.py vẫn chỉ hiển thị 1-2 nguồn chắc chắn nhất.
        if resolved is None:
            # Không chỉ định tài liệu: tìm toàn cục theo điểm, KHÔNG fan-out.
            # Fan-out ở đây sẽ nhồi 1 đoạn yếu từ mọi tài liệu vào context.
            pool = self.store.search(plan.query, top_k=top_k, document_id=None)
            return ToolResult(name="search_pdf", sources=pool)
        if len(resolved) == 1:
            pool = self.store.search(plan.query, top_k=top_k, document_id=resolved[0])
            return ToolResult(name="search_pdf", sources=pool)

        pool, skipped = self._fan_out(plan.query, top_k, resolved)
        return ToolResult(name="search_pdf", sources=pool, documents_skipped=skipped)

    def _all_document_ids(self) -> list[str]:
        return [document.id for document in self.store.list_documents()]

    def _fan_out(
        self, query: str, top_k: int, document_ids: list[str] | None
    ) -> tuple[list[SourceChunk], list[str]]:
        if not document_ids:
            return self.store.search(query, top_k=top_k, document_id=None), []
        return self.store.search_many_report(query, top_k=top_k, document_ids=document_ids)


def build_summary_query(query: str) -> str:
    # Trước đây append ~10 từ generic làm query dài ra -> coverage
    # (matched/len(query)) tụt và cổng evidence tự trip chính câu summarize.
    # Chỉ bổ sung tối thiểu khi query chưa có từ tóm tắt.
    lowered = query.lower()
    if any(term in lowered for term in ("tóm tắt", "tom tat", "summary", "summarize", "overview")):
        return query
    return f"{query} tóm tắt summary"


def quick_tool_plan(question: str) -> ToolPlan | None:
    lowered = question.lower()
    if any(term in lowered for term in LIST_PDF_TERMS):
        return ToolPlan(name="list_pdfs", query=question, reason="list-intent keywords")
    if any(term in lowered for term in COMPARE_TERMS):
        return ToolPlan(name="compare_pdfs", query=question, reason="compare/synthesize intent")
    summary_terms = [
        "tóm tắt",
        "tom tat",
        "tổng hợp",
        "tong hop",
        "summary",
        "summarize",
        "overview",
        "ý chính",
        "y chinh",
    ]
    if any(term in lowered for term in summary_terms):
        return ToolPlan(name="summarize_pdf", query=question, reason="summary intent")
    return None


def fallback_tool_plan(question: str) -> ToolPlan:
    plan = quick_tool_plan(question)
    if plan is not None:
        return plan
    return ToolPlan(name="search_pdf", query=question, reason="default RAG search")


def resolve_document_ids(
    document_id: str | None,
    document_ids: list[str] | None,
) -> list[str] | None:
    """Merge legacy single ``document_id`` + new ``document_ids``.

    Returns None (= search ALL documents) when neither is provided or when
    an explicitly empty list is provided.
    """
    merged: list[str] = []
    seen: set[str] = set()
    for candidate in (document_ids or []):
        if candidate and candidate not in seen:
            seen.add(candidate)
            merged.append(candidate)
    if document_id and document_id not in seen:
        merged.append(document_id)
    return merged or None


def blend_query_with_history(question: str, history: list) -> str:
    """Widen a follow-up question with the previous user turn.

    "Còn cái kia thì sao?" has no lexical overlap with anything, so retrieval
    returns nothing and the assistant would refuse. Blending the previous
    user message back in recovers the topic without spending an LLM call on
    query rewriting.

    Args:
        question: the current user question.
        history: prior :class:`ChatMessage` objects for this session, oldest
            first. The most recent user message is used.

    Returns:
        The blended query, or ``question`` unchanged when there is no usable
        previous user turn.
    """
    previous_user = [
        str(getattr(message, "content", "") or "").strip()
        for message in history
        if getattr(message, "role", "") == "user"
    ]
    previous_user = [content for content in previous_user if content]
    if not previous_user:
        return question

    blended = f"{previous_user[-1]} {question}".strip()
    return blended[:500] or question


def previous_user_question(history: list) -> str | None:
    """Return the most recent user turn in ``history``, or ``None``."""
    for message in reversed(list(history)):
        if getattr(message, "role", "") == "user":
            content = str(getattr(message, "content", "") or "").strip()
            if content:
                return content
    return None


def rescue_queries(question: str, history: list) -> list[str]:
    """Ordered candidate queries to retry a follow-up that retrieved nothing.

    Two candidates, cheapest first:

    1. The blended query (previous turn + follow-up). Keeps any new topic words
       the follow-up introduces, e.g. "Còn chính sách nghỉ phép thì sao?".
    2. The previous user turn alone. Needed because blending *lengthens* the
       query, and the evidence gate measures the share of query terms the best
       chunk covers — so a purely conversational follow-up like
       "Còn cái kia thì sao?" adds five empty words that can push coverage
       below the threshold and suppress an otherwise valid hit.

    A candidate is dropped when it adds no content word the original question
    did not already have, because re-running the identical search cannot change
    the outcome. Returns an empty list when there is nothing worth retrying.
    """
    previous = previous_user_question(history)
    if not previous:
        return []

    original_terms = content_tokens(question)
    candidates: list[str] = []
    for candidate in (blend_query_with_history(question, history), previous):
        candidate = candidate.strip()
        if not candidate or candidate == question.strip() or candidate in candidates:
            continue
        if not set(content_tokens(candidate)) - set(original_terms):
            continue
        candidates.append(candidate)
    return candidates


def normalize_name(text: str, strip_extension: bool = False) -> str:
    from app.services.vector_store import fold_vietnamese

    if strip_extension and "." in text:
        text = text.rsplit(".", 1)[0]
    # Fold dấu trước khi strip: trước đây "tài" -> "ti" còn "tai" -> "tai"
    # nên file có dấu không bao giờ match query không dấu.
    return re.sub(r"[^0-9a-z]+", "", fold_vietnamese(text).lower())


def match_document_by_name(question: str, documents: list[DocumentSummary]) -> DocumentSummary | None:
    matches = match_documents_by_name(question, documents)
    return matches[0] if matches else None


def match_documents_by_name(question: str, documents: list[DocumentSummary]) -> list[DocumentSummary]:
    """Mọi tài liệu được nhắc tên trong câu hỏi, xếp theo độ dài tên giảm dần.

    Dùng cho compare 2+ file: hàm cũ chỉ giữ longest-match duy nhất nên
    "so sánh A và B" bị fan-out 1 doc.
    """
    normalized_question = normalize_name(question)
    scored: list[tuple[int, DocumentSummary]] = []
    for document in documents:
        normalized_name = normalize_name(document.filename, strip_extension=True)
        if len(normalized_name) >= 4 and normalized_name in normalized_question:
            scored.append((len(normalized_name), document))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in scored]
