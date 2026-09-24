"""Generate a small multi-file corpus for exercising the fan-out path by hand.

The unit tests cover multi-document retrieval against fixtures. This makes it
possible to check the same thing through the real API -- upload, scope a
question to several documents, and read ``documents_used`` -- without hunting
for PDFs.

The three documents deliberately share vocabulary ("MUA-07", "bảy ngày làm
việc", "năm mươi triệu đồng", "giám đốc phê duyệt"). Documents that share
nothing make every retriever look perfect, so they prove nothing.

    cd backend
    .\\.venv\\Scripts\\python.exe tests/make_test_pdfs.py

Writes to ``<temp>/rag-multifile-demo/`` and prints the paths. Upload them with:

    curl -X POST http://127.0.0.1:8000/api/documents/batch \\
      -F "files=@<path>/kiem-thu-quy-trinh-mua-hang.pdf" \\
      -F "files=@<path>/kiem-thu-quy-trinh-nhap-kho.pdf" \\
      -F "files=@<path>/kiem-thu-quy-trinh-thanh-toan.pdf"

Then ask something that needs all three, scoped to their ids:

    "Mẫu MUA-07 được dùng ở những bước nào, và thời hạn thanh toán là bao lâu?"

Expect ``documents_skipped: []`` and evidence from three documents. Before the
identifier-escape fix this returned two documents and wrongly skipped the one
holding MUA-07.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

PAGES: dict[str, list[str]] = {
    "kiem-thu-quy-trinh-mua-hang.pdf": [
        "QUY TRINH MUA HANG CUA CONG TY",
        "Buoc 1: Truong phong lap phieu de nghi mua hang theo mau MUA-07.",
        "Buoc 2: Phong mua hang bao gia it nhat ba nha cung cap.",
        "Buoc 3: Giam doc phe duyet neu gia tri don hang tren nam muoi trieu dong.",
        "Buoc 4: Phong ke toan thanh toan trong vong bay ngay lam viec.",
    ],
    "kiem-thu-quy-trinh-nhap-kho.pdf": [
        "QUY TRINH NHAP KHO",
        "Buoc 1: Thu kho kiem tra phieu giao hang theo mau MUA-07.",
        "Buoc 2: Kiem dem so luong va doi chieu voi don hang.",
        "Buoc 3: Ghi nhan vao he thong kho trong vong hai gio.",
        "Buoc 4: Bao cao chenh lech cho phong mua hang trong ngay.",
    ],
    "kiem-thu-quy-trinh-thanh-toan.pdf": [
        "QUY TRINH THANH TOAN NHA CUNG CAP",
        "Phong ke toan chi thanh toan khi co du ba loai chung tu.",
        "Thoi han thanh toan la bay ngay lam viec ke tu khi nhan du chung tu.",
        "Moi khoan thanh toan tren nam muoi trieu dong can giam doc phe duyet.",
    ],
}


def main() -> int:
    out = Path(tempfile.gettempdir()) / "rag-multifile-demo"
    out.mkdir(parents=True, exist_ok=True)

    for name, lines in PAGES.items():
        document = fitz.open()
        for line in lines:
            page = document.new_page()
            page.insert_text((60, 90), line, fontsize=13)
        path = out / name
        document.save(path)
        document.close()
        print(f"{path}  ({os.path.getsize(path)} bytes)")

    print(f"\nUpload with: curl -X POST http://127.0.0.1:8000/api/documents/batch \\")
    for name in PAGES:
        print(f'  -F "files=@{out / name}" \\')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
