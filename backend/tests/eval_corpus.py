"""Vietnamese evaluation corpus and labelled queries for the retrieval benchmark.

Why this file exists
--------------------
The original quality test used three synthetic PDFs with a single rare token
each ("Meowtron", "Barkformer", "Photron"). Lexical retrieval separates those
trivially, so the reported precision told you nothing about real documents.

This corpus is deliberately harder:
  - Six business documents that SHARE vocabulary ("quy trình", "phê duyệt",
    "trưởng phòng", "thời hạn", "ngày làm việc", "hệ thống", "khách hàng").
    A query like "thời hạn phê duyệt" matches several documents, so the
    ranker actually has to discriminate.
  - Real Vietnamese with full diacritics, plus unaccented queries to exercise
    the diacritic-folding index.
  - Paraphrase queries that share few words with the source text, to expose
    the honest limit of keyword retrieval.
  - Refusal queries about topics that appear in NO document, so we can measure
    whether the system refuses instead of inventing an answer.

Every label below was written by reading the corpus text, not by running the
retriever — otherwise the benchmark would just be measuring itself.

Known blind spot: the identifier escape
---------------------------------------
The four ``identifier`` cases do NOT test the identifier escape in the evidence
gate, even though they look like they do. All four reach coverage 0.59–0.73
against a threshold of 0.25, so the gate never blocks them and the escape's
return value is irrelevant. Verified by disabling the escape entirely: every
metric stayed identical.

The reason is structural, not accidental: a query that is just a code ("HR-01")
is inherently high-coverage, because the code is itself a content word and the
matching chunk contains it. No question built from this corpus can trip the
gate, because the documents are short enough that a matching chunk covers most
of any question about it.

So the escape is covered by a UNIT test instead —
``test_identifier_escape_scans_past_the_top_chunk``, which sets
``min_query_coverage=0.9`` to force the gate to matter. Two real bugs lived in
that escape (hyphenated codes never matched; only rank 0 was inspected) and this
benchmark reported a clean 1.000 for ``identifier`` throughout. A green
benchmark is not the same as coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalDocument:
    document_id: str
    filename: str
    paragraphs: list[str]


@dataclass(frozen=True)
class EvalCase:
    """One labelled query.

    Attributes:
        question: the user question.
        kind: bucket used for the per-category breakdown.
        expected_document: document_id that must rank first. ``None`` for
            refusal cases and for multi-document cases.
        expected_documents: for ``compare`` cases, every document_id that must
            appear in the returned pool.
        keywords: strings that must appear somewhere in the returned pool.
        should_refuse: True when no document can answer the question.
        scope: restrict retrieval to these document ids (None = all).
    """

    question: str
    kind: str
    expected_document: str | None = None
    expected_documents: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    should_refuse: bool = False
    scope: tuple[str, ...] | None = None


EVAL_DOCUMENTS: list[EvalDocument] = [
    EvalDocument(
        document_id="quy-trinh-tuyen-dung",
        filename="quy-trinh-tuyen-dung.pdf",
        paragraphs=[
            "Quy trình tuyển dụng tại Công ty gồm bốn bước chính: xác định nhu cầu nhân sự, "
            "đăng tin tuyển dụng, phỏng vấn và ra quyết định tuyển dụng. Mọi bước đều phải "
            "được ghi nhận trên hệ thống quản lý nhân sự.",
            "Bước một, trưởng phòng gửi phiếu đề nghị tuyển dụng theo mẫu HR-01. Phiếu phải "
            "nêu rõ số lượng vị trí, mức lương dự kiến và thời hạn cần tuyển. Phiếu chỉ được "
            "xử lý khi có chữ ký của giám đốc khối.",
            "Bước hai, phòng nhân sự đăng tin trong vòng năm ngày làm việc kể từ khi phiếu được "
            "duyệt. Thời gian nhận hồ sơ tối thiểu là mười ngày làm việc.",
            "Bước ba, ứng viên đạt vòng sơ loại sẽ tham gia phỏng vấn chuyên môn với trưởng "
            "phòng và phỏng vấn văn hóa với phòng nhân sự. Kết quả phỏng vấn phải được gửi lại "
            "trong hai ngày làm việc.",
            "Bước bốn, giám đốc khối ra quyết định tuyển dụng. Thư mời nhận việc được gửi qua "
            "email trong vòng ba ngày làm việc sau khi có quyết định.",
            "Thời gian thử việc tiêu chuẩn là hai tháng đối với nhân viên và ba tháng đối với "
            "vị trí quản lý. Trong thời gian thử việc, nhân viên được đánh giá theo bảng KPI-02.",
            "Hồ sơ tuyển dụng phải được lưu trữ tối thiểu hai năm kể từ ngày ký hợp đồng chính thức.",
        ],
    ),
    EvalDocument(
        document_id="chinh-sach-nghi-phep",
        filename="chinh-sach-nghi-phep.pdf",
        paragraphs=[
            "Chính sách nghỉ phép áp dụng cho toàn bộ nhân viên ký hợp đồng chính thức. Nhân "
            "viên được hưởng mười hai ngày phép năm, cộng thêm một ngày cho mỗi năm thâm niên.",
            "Nhân viên gửi đơn xin nghỉ phép trên hệ thống quản lý nhân sự trước ít nhất ba "
            "ngày làm việc. Đơn nghỉ từ ba ngày liên tiếp trở lên cần trưởng phòng phê duyệt.",
            "Nghỉ ốm không quá hai ngày chỉ cần thông báo cho trưởng phòng, không cần giấy tờ "
            "y tế. Nghỉ ốm từ ba ngày trở lên phải có giấy xác nhận của cơ sở y tế.",
            "Nhân viên nữ được nghỉ thai sản sáu tháng theo quy định của pháp luật. Nhân viên "
            "nam được nghỉ năm ngày khi vợ sinh con.",
            "Phép năm chưa sử dụng được chuyển tối đa năm ngày sang năm sau. Số phép còn lại "
            "sau ngày ba mươi mốt tháng ba sẽ bị hủy.",
            "Công ty hỗ trợ ăn trưa ba mươi nghìn đồng mỗi ngày làm việc và đóng bảo hiểm sức "
            "khỏe cho nhân viên chính thức sau hai tháng thử việc.",
            "Lương tháng mười ba được chi trả cùng kỳ lương tháng mười hai cho nhân viên đã "
            "làm việc đủ mười hai tháng.",
        ],
    ),
    EvalDocument(
        document_id="quy-trinh-hoan-tien",
        filename="quy-trinh-hoan-tien.pdf",
        paragraphs=[
            "Quy trình hoàn tiền áp dụng khi khách hàng yêu cầu hủy đơn hoặc sản phẩm không "
            "đúng mô tả. Khách hàng phải gửi yêu cầu trong vòng bảy ngày kể từ ngày nhận hàng.",
            "Nhân viên chăm sóc khách hàng tiếp nhận yêu cầu và tạo phiếu hoàn tiền theo mẫu "
            "CS-07. Phiếu phải ghi rõ mã đơn hàng, số tiền và lý do hoàn.",
            "Trưởng nhóm chăm sóc khách hàng phê duyệt phiếu trong vòng hai ngày làm việc. Với "
            "số tiền trên mười triệu đồng, phiếu cần thêm chữ ký của kế toán trưởng.",
            "Sau khi phiếu được duyệt, bộ phận kế toán thực hiện hoàn tiền trong vòng năm ngày "
            "làm việc. Tiền được chuyển về tài khoản ngân hàng mà khách hàng đã dùng để thanh toán.",
            "Khách hàng sẽ nhận được email xác nhận khi lệnh hoàn tiền được thực hiện. Nếu sau "
            "mười ngày làm việc vẫn chưa nhận được tiền, khách hàng liên hệ tổng đài để tra soát.",
            "Đơn hàng thuộc nhóm hàng khuyến mãi giảm giá trên năm mươi phần trăm không được "
            "hoàn tiền. Sản phẩm đã qua sử dụng cũng không thuộc diện hoàn tiền.",
            "Mỗi tháng bộ phận chăm sóc khách hàng lập báo cáo hoàn tiền gửi cho ban giám đốc, "
            "trong đó nêu rõ tỷ lệ hoàn tiền và các nguyên nhân chính.",
        ],
    ),
    EvalDocument(
        document_id="chinh-sach-bao-mat",
        filename="chinh-sach-bao-mat.pdf",
        paragraphs=[
            "Chính sách bảo mật quy định cách phân loại và xử lý dữ liệu trong toàn công ty. "
            "Dữ liệu được chia thành bốn mức: công khai, nội bộ, mật và tối mật.",
            "Dữ liệu mức nội bộ chỉ được truy cập bởi nhân viên chính thức. Dữ liệu mức mật yêu "
            "cầu phê duyệt của trưởng phòng và phải được mã hóa khi lưu trữ.",
            "Dữ liệu tối mật chỉ được truy cập bởi ban giám đốc. Mọi lần truy cập dữ liệu tối "
            "mật đều được ghi log và lưu trong mười hai tháng.",
            "Nhân viên không được sao chép dữ liệu khách hàng ra thiết bị cá nhân. Việc sử dụng "
            "ổ cứng ngoài phải được phòng công nghệ thông tin cấp phép trước.",
            "Mật khẩu hệ thống phải có tối thiểu mười hai ký tự, bao gồm chữ hoa, chữ thường, "
            "số và ký tự đặc biệt. Mật khẩu phải đổi định kỳ chín mươi ngày.",
            "Khi phát hiện sự cố rò rỉ dữ liệu, nhân viên phải báo ngay cho phòng công nghệ "
            "thông tin trong vòng một giờ. Báo cáo sự cố phải được gửi lên ban giám đốc trong "
            "vòng hai mươi bốn giờ.",
            "Mọi nhân viên phải hoàn thành khóa đào tạo bảo mật hằng năm. Nhân viên không hoàn "
            "thành sẽ bị tạm khóa quyền truy cập hệ thống cho đến khi hoàn thành.",
        ],
    ),
    EvalDocument(
        document_id="huong-dan-phan-mem",
        filename="huong-dan-phan-mem.pdf",
        paragraphs=[
            "Phần mềm quản lý công việc nội bộ cho phép tạo dự án, giao việc và theo dõi tiến "
            "độ. Mỗi nhân viên được cấp một tài khoản riêng theo email công ty.",
            "Để tạo dự án mới, chọn mục Dự án rồi bấm nút Tạo mới. Hệ thống yêu cầu nhập tên dự "
            "án, người phụ trách và ngày kết thúc dự kiến.",
            "Mỗi công việc trong dự án có bốn trạng thái: chưa bắt đầu, đang làm, chờ duyệt và "
            "hoàn thành. Chỉ người phụ trách dự án mới được chuyển trạng thái sang hoàn thành.",
            "Tệp đính kèm tối đa hai mươi lăm megabyte cho mỗi công việc. Định dạng được hỗ trợ "
            "gồm PDF, DOCX, XLSX và PNG.",
            "Hệ thống tự động gửi thông báo qua email khi công việc được giao hoặc khi đến hạn. "
            "Người dùng có thể tắt thông báo trong phần cài đặt cá nhân.",
            "Dữ liệu dự án được sao lưu tự động mỗi ngày lúc hai giờ sáng. Bản sao lưu được lưu "
            "giữ trong ba mươi ngày.",
            "Khi quên mật khẩu, người dùng bấm vào liên kết Quên mật khẩu ở trang đăng nhập. "
            "Liên kết đặt lại mật khẩu có hiệu lực trong mười lăm phút.",
        ],
    ),
    EvalDocument(
        document_id="quy-trinh-su-co-ky-thuat",
        filename="quy-trinh-su-co-ky-thuat.pdf",
        paragraphs=[
            "Quy trình xử lý sự cố kỹ thuật phân loại sự cố theo bốn mức độ: nghiêm trọng, cao, "
            "trung bình và thấp. Mức độ quyết định thời hạn phản hồi.",
            "Sự cố mức nghiêm trọng phải được phản hồi trong vòng ba mươi phút và khắc phục "
            "trong bốn giờ. Sự cố mức cao phải được phản hồi trong hai giờ và khắc phục trong "
            "một ngày làm việc.",
            "Sự cố mức trung bình được xử lý trong ba ngày làm việc. Sự cố mức thấp được xử lý "
            "trong bảy ngày làm việc.",
            "Mọi sự cố phải được ghi nhận thành phiếu theo mẫu IT-09, bao gồm thời điểm phát "
            "hiện, ảnh hưởng và người xử lý.",
            "Sau khi khắc phục, người xử lý phải viết báo cáo nguyên nhân gốc trong vòng hai "
            "ngày làm việc. Báo cáo được lưu trong hệ thống quản lý sự cố.",
            "Sự cố lặp lại từ ba lần trở lên trong một tháng phải được chuyển thành vấn đề cần "
            "cải tiến quy trình. Trưởng phòng công nghệ thông tin chịu trách nhiệm theo dõi.",
            "Nhân viên có thể báo sự cố qua tổng đài nội bộ hoặc qua cổng hỗ trợ. Thời gian hỗ "
            "trợ qua tổng đài là từ tám giờ sáng đến sáu giờ chiều các ngày làm việc.",
        ],
    ),
]


EVAL_CASES: list[EvalCase] = [
    # ---- factual: the answer sits in exactly one document -----------------
    EvalCase(
        question="Nhân viên được hưởng bao nhiêu ngày phép năm?",
        kind="factual",
        expected_document="chinh-sach-nghi-phep",
        keywords=("mười hai ngày phép",),
    ),
    EvalCase(
        question="Thời gian thử việc tiêu chuẩn đối với nhân viên là bao lâu?",
        kind="factual",
        expected_document="quy-trinh-tuyen-dung",
        keywords=("hai tháng",),
    ),
    EvalCase(
        question="Khách hàng phải gửi yêu cầu hoàn tiền trong vòng bao nhiêu ngày?",
        kind="factual",
        expected_document="quy-trinh-hoan-tien",
        keywords=("bảy ngày",),
    ),
    EvalCase(
        question="Mật khẩu hệ thống phải có tối thiểu bao nhiêu ký tự?",
        kind="factual",
        expected_document="chinh-sach-bao-mat",
        keywords=("mười hai ký tự",),
    ),
    EvalCase(
        question="Tệp đính kèm trong phần mềm quản lý công việc tối đa bao nhiêu megabyte?",
        kind="factual",
        expected_document="huong-dan-phan-mem",
        keywords=("hai mươi lăm megabyte",),
    ),
    EvalCase(
        question="Sự cố mức nghiêm trọng phải được khắc phục trong bao lâu?",
        kind="factual",
        expected_document="quy-trinh-su-co-ky-thuat",
        keywords=("bốn giờ",),
    ),
    EvalCase(
        question="Phép năm chưa sử dụng được chuyển tối đa bao nhiêu ngày sang năm sau?",
        kind="factual",
        expected_document="chinh-sach-nghi-phep",
        keywords=("năm ngày",),
    ),
    EvalCase(
        question="Bản sao lưu dữ liệu dự án được lưu giữ trong bao nhiêu ngày?",
        kind="factual",
        expected_document="huong-dan-phan-mem",
        keywords=("ba mươi ngày",),
    ),
    EvalCase(
        question="Nhân viên nam được nghỉ bao nhiêu ngày khi vợ sinh con?",
        kind="factual",
        expected_document="chinh-sach-nghi-phep",
        keywords=("năm ngày",),
    ),
    EvalCase(
        question="Sự cố mức trung bình được xử lý trong bao nhiêu ngày làm việc?",
        kind="factual",
        expected_document="quy-trinh-su-co-ky-thuat",
        keywords=("ba ngày làm việc",),
    ),
    # ---- identifier: queries carrying codes / form numbers -----------------
    EvalCase(
        question="Phiếu đề nghị tuyển dụng dùng mẫu HR-01, vậy phiếu đó do ai gửi?",
        kind="identifier",
        expected_document="quy-trinh-tuyen-dung",
        keywords=("HR-01",),
    ),
    EvalCase(
        question="Phiếu hoàn tiền được tạo theo mẫu CS-07, ai phê duyệt phiếu này?",
        kind="identifier",
        expected_document="quy-trinh-hoan-tien",
        keywords=("CS-07",),
    ),
    EvalCase(
        question="Sự cố kỹ thuật được ghi nhận theo mẫu IT-09 gồm những thông tin gì?",
        kind="identifier",
        expected_document="quy-trinh-su-co-ky-thuat",
        keywords=("IT-09",),
    ),
    EvalCase(
        question="Bảng KPI-02 được dùng để đánh giá điều gì?",
        kind="identifier",
        expected_document="quy-trinh-tuyen-dung",
        keywords=("KPI-02",),
    ),
    # ---- unaccented: exercises diacritic folding ---------------------------
    EvalCase(
        question="Du lieu toi mat chi duoc truy cap boi ai?",
        kind="unaccented",
        expected_document="chinh-sach-bao-mat",
        keywords=("ban giám đốc",),
    ),
    EvalCase(
        question="Thoi han phe duyet phieu hoan tien la bao lau?",
        kind="unaccented",
        expected_document="quy-trinh-hoan-tien",
        keywords=("hai ngày làm việc",),
    ),
    EvalCase(
        question="Bao lau phai doi mat khau he thong mot lan?",
        kind="unaccented",
        expected_document="chinh-sach-bao-mat",
        keywords=("chín mươi ngày",),
    ),
    EvalCase(
        question="So ngay phep nam cua nhan vien chinh thuc la bao nhieu?",
        kind="unaccented",
        expected_document="chinh-sach-nghi-phep",
        keywords=("mười hai ngày phép",),
    ),
    # ---- paraphrase: few shared words with the source text ----------------
    EvalCase(
        question="Tôi bị cảm nhẹ thì cần làm thủ tục gì để được nghỉ?",
        kind="paraphrase",
        expected_document="chinh-sach-nghi-phep",
        keywords=("thông báo cho trưởng phòng",),
    ),
    EvalCase(
        question="Muốn đòi lại tiền vì hàng giao không giống quảng cáo thì làm thế nào?",
        kind="paraphrase",
        expected_document="quy-trinh-hoan-tien",
        keywords=("hoàn tiền",),
    ),
    EvalCase(
        question="Hệ thống bị sập hoàn toàn thì bao lâu mới có người xử lý?",
        kind="paraphrase",
        expected_document="quy-trinh-su-co-ky-thuat",
        keywords=("ba mươi phút",),
    ),
    EvalCase(
        question="Ai phải chịu trách nhiệm khi dữ liệu công ty bị lộ ra ngoài?",
        kind="paraphrase",
        expected_document="chinh-sach-bao-mat",
        keywords=("phòng công nghệ thông tin",),
    ),
    EvalCase(
        question="Nhân viên mới phải trải qua giai đoạn đánh giá nào trước khi ký hợp đồng chính thức?",
        kind="paraphrase",
        expected_document="quy-trinh-tuyen-dung",
        keywords=("thử việc",),
    ),
    # ---- compare: multi-document, must cover every requested document -----
    EvalCase(
        question="So sánh thời hạn phê duyệt giữa quy trình tuyển dụng và quy trình hoàn tiền",
        kind="compare",
        expected_documents=("quy-trinh-tuyen-dung", "quy-trinh-hoan-tien"),
        scope=("quy-trinh-tuyen-dung", "quy-trinh-hoan-tien"),
    ),
    EvalCase(
        question="Điểm khác nhau về thời hạn xử lý giữa chính sách bảo mật và quy trình xử lý sự cố",
        kind="compare",
        expected_documents=("chinh-sach-bao-mat", "quy-trinh-su-co-ky-thuat"),
        scope=("chinh-sach-bao-mat", "quy-trinh-su-co-ky-thuat"),
    ),
    EvalCase(
        question="Tổng hợp các mốc thời hạn cần tuân thủ trong chính sách nghỉ phép và quy trình tuyển dụng",
        kind="compare",
        expected_documents=("chinh-sach-nghi-phep", "quy-trinh-tuyen-dung"),
        scope=("chinh-sach-nghi-phep", "quy-trinh-tuyen-dung"),
    ),
    # ---- refusal: no document covers these topics -------------------------
    EvalCase(
        question="Quy trình xin visa đi công tác nước ngoài gồm những bước nào?",
        kind="refusal",
        should_refuse=True,
    ),
    EvalCase(
        question="Công ty có bán cà phê rang xay không?",
        kind="refusal",
        should_refuse=True,
    ),
    EvalCase(
        question="Chính sách cho thuê xe đạp điện cho nhân viên là gì?",
        kind="refusal",
        should_refuse=True,
    ),
    EvalCase(
        question="Giá vé máy bay đi Tokyo tháng sau là bao nhiêu?",
        kind="refusal",
        should_refuse=True,
    ),
    EvalCase(
        question="Có lớp dạy nấu ăn chay nào vào cuối tuần không?",
        kind="refusal",
        should_refuse=True,
    ),
    EvalCase(
        question="Thủ tục đăng ký kết hôn tại phường mất bao lâu?",
        kind="refusal",
        should_refuse=True,
    ),
]
