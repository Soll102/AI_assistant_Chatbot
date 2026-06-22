from collections.abc import Iterable
import base64

import httpx

from app.schemas import SourceChunk


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
