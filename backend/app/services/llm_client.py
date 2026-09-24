from __future__ import annotations

from collections.abc import Iterable
import base64
import json
import re
from dataclasses import dataclass

import httpx

from app.schemas import SourceChunk
from app.services.rag_tools import ToolPlan, fallback_tool_plan, quick_tool_plan

TOOL_NAMES = {"search_pdf", "summarize_pdf", "list_pdfs", "compare_pdfs"}

PDF_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_pdf",
            "description": "Tìm kiếm thông tin cụ thể trong tài liệu PDF để trả lời câu hỏi của người dùng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Câu truy vấn retrieval ngắn gọn, tối ưu cho việc tìm kiếm trong PDF.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_pdf",
            "description": "Thu thập nội dung để tóm tắt, nêu ý chính hoặc kết luận của tài liệu PDF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Mô tả phạm vi cần tóm tắt nếu người dùng nêu rõ, nếu không nhắc lại yêu cầu.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_pdfs",
            "description": "So sánh, đối chiếu hoặc tổng hợp thông tin ACROSS nhiều tài liệu PDF (multi-document synthesis).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Câu truy vấn so sánh/tổng hợp, giữ nguyên ý người dùng.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_pdfs",
            "description": "Liệt kê tên các tài liệu PDF đã được upload vào hệ thống.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

SYSTEM_PROMPT = """
Bạn là trợ lý AI đọc PDF trong một hệ thống RAG đa tài liệu.
Bạn chỉ được trả lời dựa trên nội dung tool trả về từ các tài liệu.
Mỗi đoạn context có kèm tên file và số trang — khi tổng hợp từ nhiều
tài liệu, hãy nêu rõ thông tin nào đến từ tài liệu nào.
Trả lời bằng tiếng Việt, ngắn gọn, trực tiếp, không lan man.
Không tự ghi nguồn, số trang, tên file, hoặc citation trong câu trả lời.
Cấm thêm bất kỳ tiền tố phân loại nào (như "User Safety:", "Safety:",
"Content Safety:"). Bắt đầu thẳng vào câu trả lời.
Nếu dữ liệu từ tool không đủ, nói ngắn gọn rằng tài liệu không cung cấp đủ thông tin.
Với câu hỏi so sánh nhiều tài liệu mà context chỉ có 1 tài liệu, hãy nói rõ
giới hạn đó thay vì suy đoán phần còn lại.
""".strip()


# Một số model free có thói quen prepend verdict an toàn
# (vd. "User Safety: safe") trước câu trả lời. Cắt bỏ các dòng đó ở đầu.
SAFETY_PREAMBLE_RE = re.compile(
    r"^\s*(user\s*safety|content\s*safety|safety\s*assessment|safety|an\s*toàn)\s*:.*$",
    re.IGNORECASE,
)


def strip_safety_preamble(text: str) -> str:
    lines = text.splitlines()
    index = 0
    while index < len(lines) and SAFETY_PREAMBLE_RE.match(lines[index]):
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    cleaned = "\n".join(lines[index:]).strip()
    if cleaned:
        return cleaned
    # Nothing survived the strip. When the model returned *only* a safety
    # preamble (observed in production: a bare "User Safety: safe"), returning
    # the original would hand the user that line as if it answered their
    # question. Return "" so the caller substitutes a real message instead.
    # Blank input keeps the old behaviour of coming back unchanged.
    return "" if text.strip() else text.strip()


# Shown when the model produced no usable prose -- an empty response, or one
# that consisted entirely of a safety preamble. Both mean "no answer", and the
# user must be told that rather than shown the model's internal bookkeeping.
EMPTY_ANSWER_MESSAGE = (
    "Không nhận được nội dung trả lời từ mô hình. Hãy thử hỏi lại câu hỏi này."
)


# Verdict ngắn của model ("User Safety: safe") — khác với heading thật
# ("Safety: wear helmet when..."). Ingest chỉ strip verdict để không xoá
# nhầm nội dung tài liệu.
SAFETY_VERDICT_RE = re.compile(
    r"^\s*(user\s*safety|content\s*safety|safety\s*assessment|safety|an\s*toàn)\s*:\s*"
    r"(safe|unsafe|ok|okay|pass|fail|clean|clear|fine|yes|no|không\s*vấn\s*đề|an\s*toàn)?\s*[.!]?\s*$",
    re.IGNORECASE,
)


def strip_safety_lines(text: str) -> str:
    """Remove safety-classification lines from anywhere in extracted text.

    This is for the *ingest* path, where the text is about to become indexed
    document content. ``strip_safety_preamble`` only handles the leading case
    (a chat answer that is prefixed), but vision extraction appends the line at
    the END -- found in a real index as
    ``"...FUNDAMENTALS OF SUPERVISED LEARNING\\n\\nUser Safety: safe"``.

    A classification verdict is never document content, so it is dropped
    wherever it appears. Leaving it in makes the line retrievable, and the
    assistant then quotes the model's own bookkeeping back to the user as if it
    came from their document.
    """
    kept = [line for line in text.splitlines() if not SAFETY_VERDICT_RE.match(line)]
    return "\n".join(kept).strip()


# Vietnamese phrasings that mean "the documents do not answer this".
NO_EVIDENCE_MARKERS = (
    "không cung cấp",
    "không có thông tin",
    "không đề cập",
    "không liên quan",
    "chưa tìm thấy",
    "không tìm thấy",
    "không đủ thông tin",
    "không thể trả lời",
    "không có trong tài liệu",
    "ngoài phạm vi",
)


def looks_like_refusal(answer: str, prefix_chars: int = 120) -> bool:
    """True when an answer says the documents do not cover the question.

    Only the opening of the answer counts. A long comparative answer can
    legitimately say "chính sách bảo mật không cung cấp thông tin về thời hạn..."
    while still answering the question — that is a noted gap inside a real
    answer, not a refusal. Matching the phrase anywhere made 3 of 3 comparison
    answers look like refusals, which is why this is a prefix check.

    The verification pass asks a model for a JSON verdict, and a weak model will
    happily label its own refusal ``is_supported: true`` — the answer text says
    "tài liệu không cung cấp..." while the label claims support. The label is
    shown to the user, so it must not contradict the answer it describes. This
    check is deterministic and overrides the model on that specific point.
    """
    opening = " ".join(answer.split())[:prefix_chars].lower()
    return any(marker in opening for marker in NO_EVIDENCE_MARKERS)


@dataclass(frozen=True)
class ChatState:
    answer: str | None = None
    tool: ToolPlan | None = None
    tool_call_id: str | None = None

    @property
    def needs_tool(self) -> bool:
        return self.tool is not None


class LLMClient:
    def __init__(self, api_key: str, model: str, fallback_model: str = "") -> None:
        self.api_key = api_key
        self.model = model
        self.fallback_model = fallback_model
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self._tool_unsupported_models: set[str] = set()
        self._vision_unsupported_models: set[str] = set()
        self._failed_models: set[str] = set()

    def supports_vision(self) -> bool:
        for model in (self.model, self.fallback_model):
            if model and model not in self._failed_models and model not in self._vision_unsupported_models:
                return True
        return False

    def start_chat(self, question: str, has_document: bool) -> ChatState:
        if not self.api_key or not has_document:
            return ChatState(tool=fallback_tool_plan(question))

        quick = quick_tool_plan(question)
        if quick is not None:
            return ChatState(tool=quick)

        message, error = self._post_chat([system_message(SYSTEM_PROMPT), user_message(question)], tools=PDF_TOOLS)
        if error:
            return ChatState(tool=fallback_tool_plan(question))

        tool_call_id, tool_name, tool_query = parse_tool_call(message or {})
        if tool_name:
            return ChatState(
                tool=ToolPlan(name=tool_name, query=tool_query or question, reason="native tool call"),
                tool_call_id=tool_call_id,
            )

        # Model gọi tool lạ (không thuộc TOOL_NAMES): parse_tool_call bỏ qua
        # nên trước đây content của nó bị dùng làm answer trực tiếp
        # (hallucination, không retrieval). Ép về retrieval thay vì tin model.
        raw_calls = (message or {}).get("tool_calls") or []
        if raw_calls:
            return ChatState(tool=fallback_tool_plan(question))

        text = str((message or {}).get("content") or "").strip()
        cleaned = strip_safety_preamble(text)
        if cleaned:
            # start_chat chỉ dùng khi enable_tool_planning=True. Text trả
            # thẳng ở đây chưa qua retrieval nên caller (main.py) phải
            # grounding lại trước khi trả cho user — xem main.chat.
            return ChatState(answer=cleaned)

        # Either the model said nothing, or it returned only a safety preamble.
        # Neither is an answer, so search the documents instead of handing the
        # user a blank response.
        return ChatState(tool=fallback_tool_plan(question))

    def finalize_with_sources(
        self,
        question: str,
        sources: list[SourceChunk],
        tool_plan: ToolPlan | None,
        tool_call_id: str | None,
        history: list | None = None,
    ) -> str:
        """Generate the grounded answer.

        ``history`` carries previous turns so follow-up questions ("còn cái
        kia thì sao?") can be interpreted. It is optional to keep the old
        positional call signature working.
        """
        if not self.api_key:
            return (
                "Backend chưa có OPENROUTER_API_KEY. Hãy thêm key vào backend/.env rồi khởi động lại server. "
                "Mình vẫn đã tìm được các nguồn liên quan bên dưới để bạn kiểm tra."
            )
        if tool_plan is None or tool_call_id is None:
            return self._answer_fallback(question, sources, history)

        call_id = tool_call_id or "call_search_1"
        plan = tool_plan
        documents_payload = [
            {"filename": source.filename, "page": source.page, "text": source.text}
            for source in sources
        ]
        tool_result_payload = json.dumps(
            {"document_count": len(sources), "documents": documents_payload},
            ensure_ascii=False,
        )
        # Cắt JSON giữa chuỗi tạo tool message invalid. Thay vào đó bớt
        # dần số document cho tới khi vừa budget, giữ JSON luôn hợp lệ.
        while len(tool_result_payload) > 60000 and len(documents_payload) > 1:
            documents_payload = documents_payload[:-1]
            tool_result_payload = json.dumps(
                {"document_count": len(documents_payload), "documents": documents_payload},
                ensure_ascii=False,
            )
        if len(tool_result_payload) > 60000:
            # Vẫn quá lớn với 1 doc duy nhất: cắt text của doc đó rồi dump lại.
            single = dict(documents_payload[0])
            single["text"] = str(single.get("text") or "")[:55000]
            tool_result_payload = json.dumps(
                {"document_count": 1, "documents": [single]}, ensure_ascii=False
            )

        tool_name = plan.name if tool_name_supported(plan.name) else "search_pdf"
        messages = [system_message(SYSTEM_PROMPT)]
        messages.extend(history_to_messages(history))
        messages.extend(
            [
                user_message(question),
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps({"query": plan.query}, ensure_ascii=False),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "content": tool_result_payload},
            ]
        )
        message, error = self._post_chat(messages)
        if error:
            return self._answer_fallback(question, sources, history)

        text = str((message or {}).get("content") or "").strip()
        cleaned = strip_safety_preamble(text)
        return cleaned or EMPTY_ANSWER_MESSAGE

    def list_documents_answer(self, documents: list) -> str:
        if not documents:
            return "Chưa có PDF nào được upload vào hệ thống."
        lines = [f"{index}. {document.filename} ({document.pages} trang, {document.chunks} đoạn)" for index, document in enumerate(documents, start=1)]
        return "Các PDF đã upload:\n" + "\n".join(lines)

    def _answer_fallback(self, question: str, sources: list[SourceChunk], history: list | None = None) -> str:
        # This path skips the tool-calling round trip, so it has to guard the
        # empty case itself -- otherwise a preamble-only response reaches the
        # user as a blank answer.
        return strip_safety_preamble(self._generate_text(build_prompt(question, sources, history))) or EMPTY_ANSWER_MESSAGE

    def verify_answer(self, question: str, answer: str, sources: list[SourceChunk]) -> tuple[str, str]:
        # Normalise first: every branch below returns ``answer`` or a rewrite of
        # it, so a preamble-only draft must be caught before any of them can
        # hand it back. There is nothing to verify in that case, hence "skipped".
        answer = strip_safety_preamble(answer)
        if not answer:
            return EMPTY_ANSWER_MESSAGE, "skipped"
        if not self.api_key or not sources or is_api_error(answer):
            return answer, "skipped"

        raw_text = self._generate_text(build_verification_prompt(question, answer, sources))
        try:
            payload = parse_json_object(raw_text)
        except ValueError:
            return answer, "unverified"

        is_supported = bool(payload.get("is_supported"))
        fixed_answer = strip_safety_preamble(str(payload.get("fixed_answer") or ""))
        reason = str(payload.get("reason") or "").strip()
        if is_supported:
            # A model can call its own refusal "supported". The label is
            # user-facing, so it must match what the answer actually says.
            if looks_like_refusal(answer):
                return answer, f"no_evidence: {reason}" if reason else "no_evidence"
            return answer, f"supported: {reason}" if reason else "supported"
        if fixed_answer:
            if looks_like_refusal(fixed_answer):
                return fixed_answer, f"no_evidence: {reason}" if reason else "no_evidence"
            return fixed_answer, f"revised: {reason}" if reason else "revised"
        return "Tài liệu không cung cấp đủ thông tin để trả lời chắc chắn.", "no_evidence"

    def extract_page_from_image(self, image_bytes: bytes, page_number: int) -> str:
        if not self.api_key or not self.supports_vision():
            return ""

        prompt = f"""
Bạn đang đọc ảnh render từ trang {page_number} của một PDF.
Hãy trích xuất nội dung quan trọng để dùng cho RAG:
- Giữ văn bản chính.
- Viết công thức toán ở dạng LaTeX nếu thấy được.
- Mô tả hình ảnh, biểu đồ, bảng hoặc sơ đồ bằng tiếng Việt.
- Nếu không đọc được, nói ngắn gọn là không đọc được.
Không thêm suy đoán ngoài nội dung trong ảnh.
""".strip()
        text = self._generate_text(prompt, image_bytes=image_bytes)
        if is_api_error(text):
            return ""
        # Strip safety lines HERE, before the text is indexed. This output is
        # persisted as page content, so a stray "User Safety: safe" line does
        # not merely appear once -- it becomes a retrievable chunk that later
        # gets echoed back to users as if it were document text. That is
        # exactly how a live query came back with the answer
        # "User Safety: safe". Note this removes the line anywhere in the
        # output: vision models append it, they do not only prepend it.
        return strip_safety_lines(text)

    def _generate_text(self, prompt: str, image_bytes: bytes | None = None) -> str:
        if image_bytes is None:
            content = prompt
        else:
            image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]

        message, error = self._post_chat([user_message(content)])
        if error:
            return error
        text = str((message or {}).get("content") or "").strip()
        return text or "Không nhận được nội dung trả lời từ OpenRouter."

    def _candidate_models(self, has_image: bool = False) -> list[str]:
        models: list[str] = []
        for name in (self.model, self.fallback_model):
            if not name or name in self._failed_models:
                continue
            if has_image and name in self._vision_unsupported_models:
                continue
            if name not in models:
                models.append(name)
        return models

    def _post_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> tuple[dict | None, str | None]:
        has_image = contains_image(messages)
        candidates = self._candidate_models(has_image=has_image)
        if tools:
            candidates = [model for model in candidates if model not in self._tool_unsupported_models]
        if not candidates:
            if has_image:
                return None, (
                    "Model hiện tại không hỗ trợ đọc ảnh (vision). "
                    "Đổi OPENROUTER_MODEL trong backend/.env sang model có hỗ trợ image input."
                )
            return None, (
                "Không có model khả dụng trên OpenRouter. Kiểm tra OPENROUTER_MODEL và "
                "OPENROUTER_FALLBACK_MODEL trong backend/.env."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Multimodal PDF RAG Chatbot",
        }
        last_error: str | None = None
        for model in candidates:
            payload: dict = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
            }
            if tools:
                payload["tools"] = tools
            try:
                with httpx.Client(timeout=120) as client:
                    response = client.post(self.endpoint, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                # Lỗi mạng/timeout phải thử fallback model, không return ngay.
                last_error = (
                    f"Không gọi được OpenRouter API: {exc.__class__.__name__}. Hãy kiểm tra mạng hoặc thử lại sau."
                )
                continue

            if response.status_code >= 400:
                error = build_api_error_message(response)
                if response.status_code in {401, 403}:
                    return None, error
                if has_image and is_vision_unsupported_error(error):
                    self._vision_unsupported_models.add(model)
                elif tools and is_tool_unsupported_error(error):
                    self._tool_unsupported_models.add(model)
                elif response.status_code == 404:
                    self._failed_models.add(model)
                last_error = error
                continue

            choices = response.json().get("choices") or []
            if not choices:
                last_error = "Không nhận được nội dung trả lời từ OpenRouter."
                continue

            message = choices[0].get("message") or {}
            text = str(message.get("content") or "")
            if has_image and is_vision_unsupported_error(text):
                self._vision_unsupported_models.add(model)
                last_error = f"OpenRouter model {model} không hỗ trợ đọc ảnh."
                continue
            return message, None

        return None, last_error or "Không gọi được OpenRouter API."


def is_tool_unsupported_error(error: str) -> bool:
    lowered = error.lower()
    return "support tool use" in lowered or "tools" in lowered and "not supported" in lowered


def is_vision_unsupported_error(error: str) -> bool:
    lowered = error.lower()
    return (
        "does not support image input" in lowered
        or "image input is not supported" in lowered
        or "cannot read" in lowered and "image" in lowered
    )


def contains_image(messages: list[dict]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            if any(isinstance(part, dict) and part.get("type") == "image_url" for part in content):
                return True
    return False


def tool_name_supported(name: str) -> bool:
    return name in TOOL_NAMES


def parse_tool_call(message: dict) -> tuple[str | None, str | None, str | None]:
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        if name not in TOOL_NAMES:
            continue

        query: str | None = None
        try:
            args = json.loads(function.get("arguments") or "{}")
            query = str(args.get("query") or "").strip() or None
        except ValueError:
            query = None

        return str(call.get("id") or "tool_call_1"), name, query

    return None, None, None


def system_message(text: str) -> dict:
    return {"role": "system", "content": text}


def user_message(content) -> dict:
    return {"role": "user", "content": content}


def history_to_messages(history: list | None, max_messages: int = 20) -> list[dict]:
    """Convert stored chat messages into OpenAI-compatible history turns.

    Accepts both :class:`ChatMessage` objects and plain dicts. Anything that
    is not a non-empty user/assistant turn is dropped, leading assistant
    turns are trimmed (some providers reject a conversation that starts with
    an assistant message), and the tail is capped.
    """
    converted: list[dict] = []
    for message in history or []:
        if isinstance(message, dict):
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
        else:
            role = str(getattr(message, "role", "") or "")
            content = str(getattr(message, "content", "") or "")
        content = content.strip()
        if role not in {"user", "assistant"} or not content:
            continue
        converted.append({"role": role, "content": content})

    while converted and converted[0]["role"] != "user":
        converted.pop(0)
    return converted[-max_messages:]


def build_api_error_message(response: httpx.Response) -> str:
    status = response.status_code
    fallback = response.reason_phrase or "Unknown error"
    try:
        payload = response.json()
        raw_error = payload.get("error", {})
        # OpenRouter đôi khi trả {"error": "chuỗi"} thay vì {"error": {...}}.
        # .get("message") trên str gây AttributeError -> 500 thay vì message.
        if isinstance(raw_error, dict):
            detail = raw_error.get("message") or fallback
        elif isinstance(raw_error, str) and raw_error.strip():
            detail = raw_error
        else:
            detail = payload.get("message") or fallback
    except ValueError:
        detail = fallback

    detail = detail.replace("\n", " ")[:260]
    if status in {401, 403}:
        return f"OpenRouter API từ chối xác thực ({status}). Kiểm tra lại OPENROUTER_API_KEY trong backend/.env."
    if status == 429:
        return (
            "OpenRouter API đang bị giới hạn quota/rate limit (429). "
            "Hãy thử lại sau hoặc đổi OPENROUTER_MODEL trong backend/.env. "
            f"Chi tiết: {detail}"
        )
    return f"OpenRouter API lỗi {status}: {detail}"


def build_prompt(question: str, sources: Iterable[SourceChunk], history: list | None = None) -> str:
    context = "\n\n".join(
        f"[Đoạn {index} | {source.filename} p.{source.page}]\n{source.text}"
        for index, source in enumerate(sources, start=1)
    )
    previous_turns = history_to_messages(history)
    history_block = ""
    if previous_turns:
        rendered = "\n".join(
            f"{'Người dùng' if turn['role'] == 'user' else 'Trợ lý'}: {turn['content']}"
            for turn in previous_turns
        )
        history_block = f"\n\nHỘI THOẠI TRƯỚC ĐÓ (chỉ để hiểu ngữ cảnh, không phải nguồn dữ liệu):\n{rendered}"
    return f"""
Bạn là trợ lý AI đọc PDF đa tài liệu. Chỉ trả lời dựa trên phần NGỮ CẢNH.
Mỗi đoạn ghi rõ tài liệu và số trang — khi câu hỏi liên quan nhiều tài liệu,
hãy tổng hợp và chỉ rõ điểm giống/khác nhau theo từng tài liệu.
Tập trung trả lời đúng câu hỏi của người dùng.
Trả lời bằng tiếng Việt, ngắn gọn, trực tiếp, không lan man.
Không tự ghi nguồn, số trang, tên file, hoặc citation trong câu trả lời.
Nếu ngữ cảnh không đủ dữ liệu, hãy nói ngắn gọn rằng tài liệu không cung cấp đủ thông tin.
{history_block}

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
""".strip()


def build_verification_prompt(question: str, answer: str, sources: Iterable[SourceChunk]) -> str:
    context = "\n\n".join(
        f"[Đoạn {index} | {source.filename} p.{source.page}]\n{source.text}"
        for index, source in enumerate(sources, start=1)
    )
    return f"""
Bạn là bộ kiểm chứng câu trả lời cho PDF RAG.
Kiểm tra ANSWER có được hỗ trợ bởi CONTEXT hay không.
Trả về JSON thuần, không markdown.

Quy tắc:
- is_supported=true nếu câu trả lời bám sát context.
- is_supported=false nếu câu trả lời bịa, suy đoán, hoặc context không đủ.
- Nếu false, fixed_answer phải là câu trả lời ngắn gọn chỉ dựa trên context.
- Nếu context không đủ, fixed_answer nói rằng tài liệu không cung cấp đủ thông tin.

JSON schema:
{{"is_supported":true|false,"reason":"lý do ngắn","fixed_answer":"câu trả lời đã sửa nếu cần"}}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
""".strip()


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(cleaned[start : end + 1])


def is_api_error(answer: str) -> bool:
    lowered = answer.lower()
    return (
        answer.startswith("OpenRouter API lỗi")
        or answer.startswith("OpenRouter API từ chối xác thực")
        or answer.startswith("OpenRouter API đang bị giới hạn")
        or answer.startswith("Không gọi được OpenRouter API")
        or answer.startswith("Không nhận được nội dung trả lời từ OpenRouter")
        or answer.startswith("Không nhận được nội dung trả lời từ mô hình")
        or "không hỗ trợ đọc ảnh" in lowered
        or "không có model khả dụng" in lowered
        or answer.startswith("Gemini API lỗi")
        or answer.startswith("Không gọi được Gemini API")
    )
