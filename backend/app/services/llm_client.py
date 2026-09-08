from __future__ import annotations

from collections.abc import Iterable
import base64
import json
from dataclasses import dataclass

import httpx

from app.schemas import SourceChunk
from app.services.rag_tools import ToolPlan, fallback_tool_plan, quick_tool_plan

TOOL_NAMES = {"search_pdf", "summarize_pdf", "list_pdfs"}

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
Bạn là trợ lý AI đọc PDF trong một hệ thống RAG.
Bạn chỉ được trả lời dựa trên nội dung tool trả về từ tài liệu.
Trả lời bằng tiếng Việt, ngắn gọn, trực tiếp, không lan man.
Không tự ghi nguồn, số trang, tên file, hoặc citation trong câu trả lời.
Nếu dữ liệu từ tool không đủ, nói ngắn gọn rằng tài liệu không cung cấp đủ thông tin.
""".strip()


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

        text = str((message or {}).get("content") or "").strip()
        if text:
            return ChatState(answer=text)

        return ChatState(tool=fallback_tool_plan(question))

    def finalize_with_sources(
        self,
        question: str,
        sources: list[SourceChunk],
        tool_plan: ToolPlan | None,
        tool_call_id: str | None,
    ) -> str:
        if not self.api_key:
            return (
                "Backend chưa có OPENROUTER_API_KEY. Hãy thêm key vào backend/.env rồi khởi động lại server. "
                "Mình vẫn đã tìm được các nguồn liên quan bên dưới để bạn kiểm tra."
            )
        if tool_plan is None or tool_call_id is None:
            return self._answer_fallback(question, sources)

        call_id = tool_call_id or "call_search_1"
        plan = tool_plan
        tool_result_payload = json.dumps(
            {"document_count": len(sources), "documents": [source.text for source in sources]},
            ensure_ascii=False,
        )
        if len(tool_result_payload) > 60000:
            tool_result_payload = tool_result_payload[:60000]

        tool_name = plan.name if tool_name_supported(plan.name) else "search_pdf"
        messages = [
            system_message(SYSTEM_PROMPT),
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
        message, error = self._post_chat(messages)
        if error:
            return self._answer_fallback(question, sources)

        text = str((message or {}).get("content") or "").strip()
        return text or "Không nhận được nội dung trả lời từ OpenRouter."

    def list_documents_answer(self, documents: list) -> str:
        if not documents:
            return "Chưa có PDF nào được upload vào hệ thống."
        lines = [f"{index}. {document.filename} ({document.pages} trang, {document.chunks} đoạn)" for index, document in enumerate(documents, start=1)]
        return "Các PDF đã upload:\n" + "\n".join(lines)

    def _answer_fallback(self, question: str, sources: list[SourceChunk]) -> str:
        return self._generate_text(build_prompt(question, sources))

    def verify_answer(self, question: str, answer: str, sources: list[SourceChunk]) -> tuple[str, str]:
        if not self.api_key or not sources or is_api_error(answer):
            return answer, "skipped"

        raw_text = self._generate_text(build_verification_prompt(question, answer, sources))
        try:
            payload = parse_json_object(raw_text)
        except ValueError:
            return answer, "unverified"

        is_supported = bool(payload.get("is_supported"))
        fixed_answer = str(payload.get("fixed_answer") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        if is_supported:
            return answer, f"supported: {reason}" if reason else "supported"
        if fixed_answer:
            return fixed_answer, f"revised: {reason}" if reason else "revised"
        return "Tài liệu không cung cấp đủ thông tin để trả lời chắc chắn.", "unsupported"

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
        return text

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
            "HTTP-Referer": "http://localhost:5173",
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
                return None, (
                    f"Không gọi được OpenRouter API: {exc.__class__.__name__}. Hãy kiểm tra mạng hoặc thử lại sau."
                )

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


def build_api_error_message(response: httpx.Response) -> str:
    status = response.status_code
    fallback = response.reason_phrase or "Unknown error"
    try:
        payload = response.json()
        detail = payload.get("error", {}).get("message") or fallback
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


def build_prompt(question: str, sources: Iterable[SourceChunk]) -> str:
    context = "\n\n".join(
        f"[Đoạn {index}]\n{source.text}"
        for index, source in enumerate(sources, start=1)
    )
    return f"""
Bạn là trợ lý AI đọc PDF. Chỉ trả lời dựa trên phần NGỮ CẢNH.
Tập trung trả lời đúng câu hỏi của người dùng.
Trả lời bằng tiếng Việt, ngắn gọn, trực tiếp, không lan man.
Không tự ghi nguồn, số trang, tên file, hoặc citation trong câu trả lời.
Nếu ngữ cảnh không đủ dữ liệu, hãy nói ngắn gọn rằng tài liệu không cung cấp đủ thông tin.

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
""".strip()


def build_verification_prompt(question: str, answer: str, sources: Iterable[SourceChunk]) -> str:
    context = "\n\n".join(
        f"[Đoạn {index}]\n{source.text}"
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
        or answer.startswith("Không gọi được OpenRouter API")
        or "không hỗ trợ đọc ảnh" in lowered
        or "không có model khả dụng" in lowered
        or answer.startswith("Gemini API lỗi")
        or answer.startswith("Không gọi được Gemini API")
    )
