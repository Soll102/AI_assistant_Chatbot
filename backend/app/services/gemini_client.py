from collections.abc import Iterable
import base64
import json

import httpx

from app.schemas import SourceChunk
from app.services.rag_tools import ToolPlan, fallback_tool_plan


class GeminiClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def answer(self, question: str, sources: list[SourceChunk]) -> str:
        if not self.api_key:
            return (
                "Backend chưa có GEMINI_API_KEY. Hãy thêm key vào backend/.env rồi khởi động lại server. "
                "Mình vẫn đã tìm được các nguồn liên quan bên dưới để bạn kiểm tra."
            )

        prompt = build_prompt(question, sources)
        return self._generate_text([{"text": prompt}])

    def plan_tool_call(self, question: str, has_document: bool) -> ToolPlan:
        if not self.api_key or not has_document:
            return fallback_tool_plan(question)

        prompt = build_tool_planning_prompt(question)
        raw_text = self._generate_text([{"text": prompt}])
        try:
            payload = parse_json_object(raw_text)
        except ValueError:
            return fallback_tool_plan(question)

        name = str(payload.get("tool") or "search_pdf")
        query = str(payload.get("query") or question).strip() or question
        reason = str(payload.get("reason") or "")
        if name not in {"search_pdf", "summarize_pdf"}:
            return fallback_tool_plan(question)
        return ToolPlan(name=name, query=query, reason=reason)

    def verify_answer(self, question: str, answer: str, sources: list[SourceChunk]) -> tuple[str, str]:
        if not self.api_key or not sources or is_api_error(answer):
            return answer, "skipped"

        prompt = build_verification_prompt(question, answer, sources)
        raw_text = self._generate_text([{"text": prompt}])
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
        if not self.api_key:
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
        return self._generate_text(
            [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    }
                },
            ]
        )

    def _generate_text(self, parts: list[dict]) -> str:
        payload = {"contents": [{"role": "user", "parts": parts}]}
        try:
            with httpx.Client(timeout=90) as client:
                response = client.post(self.endpoint, params={"key": self.api_key}, json=payload)
        except httpx.HTTPError as exc:
            return f"Không gọi được Gemini API: {exc.__class__.__name__}. Hãy kiểm tra mạng hoặc thử lại sau."

        if response.status_code >= 400:
            return build_api_error_message(response)

        data = response.json()

        candidates = data.get("candidates") or []
        if not candidates:
            return "Không nhận được nội dung trả lời từ Gemini."

        response_parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in response_parts).strip()
        return text or "Không nhận được nội dung trả lời từ Gemini."


def build_api_error_message(response: httpx.Response) -> str:
    status = response.status_code
    fallback = response.reason_phrase or "Unknown error"
    try:
        payload = response.json()
        detail = payload.get("error", {}).get("message") or fallback
    except ValueError:
        detail = fallback

    detail = detail.replace("\n", " ")[:260]
    if status == 429:
        return (
            "Gemini API đang bị giới hạn quota/rate limit (429). "
            "Hãy thử lại sau, giảm số lần hỏi, hoặc đổi GEMINI_MODEL trong backend/.env. "
            f"Chi tiết: {detail}"
        )
    return f"Gemini API lỗi {status}: {detail}"


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


def build_tool_planning_prompt(question: str) -> str:
    return f"""
Bạn là bộ điều phối tool cho một PDF RAG chatbot.
Chọn đúng 1 tool và trả về JSON thuần, không markdown.

TOOLS:
- search_pdf: dùng cho câu hỏi cụ thể cần tìm thông tin trong PDF.
- summarize_pdf: dùng khi người dùng muốn tóm tắt, overview, ý chính, kết luận.

JSON schema:
{{"tool":"search_pdf|summarize_pdf","query":"câu truy vấn tối ưu để retrieval","reason":"lý do ngắn"}}

Câu hỏi:
{question}
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
    return answer.startswith("Gemini API lỗi") or answer.startswith("Không gọi được Gemini API")
