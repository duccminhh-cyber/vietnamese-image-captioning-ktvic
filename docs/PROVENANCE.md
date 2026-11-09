# Nguồn và phạm vi bản tuần 1

Ngày xuất: 09/11/2026. Bản này lấy có chọn lọc từ workspace KTVIC có sẵn.
Commit HEAD của repo nguồn khi đọc: cdb0289; workspace nguồn có thay đổi chưa
commit nên HEAD không định danh đầy đủ snapshot. SHA256 từng file sao chép nằm
trong [source_manifest.json](source_manifest.json). Hash gốc ghi byte lúc sao chép;
canonical_lf_sha256 hỗ trợ đối chiếu sau khi Git đổi CRLF/LF.

Giữ lại: annotation/split/audit, tokenizer, Dataset/collator, seed, script chuẩn bị
dữ liệu, kiểm thử dữ liệu tổng hợp và requirements. Thêm mới: check_week1.py,
notebook EDA sạch output, hướng dẫn và proposal cho hai người.

Không đưa vào bộ này: dữ liệu gốc, trọng số, log/metrics huấn luyện, kết quả test,
mã demo hay lịch sử commit của repo nguồn. Dự án nguồn đã có các phần này;
bản tuần 1 không chứng minh rằng nhóm chưa làm chúng hoặc đã nộp trước đây.

Không thay đổi source hay trạng thái staging của repo cha. Repo con có lịch sử
khởi tạo ở thời điểm xuất thật. Không tuyên bố quyền sở hữu hoặc cấp license mới
cho toàn bộ code/template khi chưa có xác nhận của nhóm.

Template ACL: https://github.com/acl-org/acl-style-files .
Thư viện và ý tưởng bài báo được khai báo ở README, requirements và proposal.
