# Kế hoạch nhóm hai người

Ngày soạn: 09/11/2026. Mã nhóm 10 theo hồ sơ local; danh tính được xác nhận trong phiên.
Đây là kế hoạch phân công, không phải nhật ký chứng minh đã làm trong sáu tuần.

| Tuần | Minh | Mạnh | Bàn giao |
|---|---|---|---|
| 1 | Rà dữ liệu, tokenizer, kiểm tra batch | Đọc Transformer, thiết kế đánh giá, rà proposal | Proposal ≤2 trang nội dung, repo, ba bài báo |
| 2 | Chạy baseline, kiểm tra loss và overfit batch nhỏ | Rà metric nhiều reference và log | Baseline chạy được, log thật |
| 3 | LSTM attention | Transformer decoder | Milestone ≤4 trang, kết quả baseline |
| 4 | Ablation attention, phân tích lỗi | So sánh decoder/decoding, bảng chi phí | Bảng thí nghiệm có kiểm soát |
| 5 | Rà lỗi định tính, trực quan hóa | Tổng hợp kết quả, nháp báo cáo | Nháp; mở rộng CLIP nếu đủ thời gian |
| 6 | Demo, rà nguồn dữ liệu và mô hình | Đóng gói tái lập, slide | PDF ACL ≤8 trang nội dung, source, slide 10–12 phút |

Mỗi người rà ít nhất một phần mã của người còn lại. Tỷ lệ đóng góp ghi theo việc
thực sự hoàn thành, không tự gán 50/50. Báo cáo cuối cần Limitations và khai báo AI.

## Hoàn thành tuần 1

1. Chạy tests bằng dữ liệu tổng hợp.
2. Đọc được một batch ảnh/caption thật với padding/BOS/EOS đúng.
3. Thống kê số ảnh, số caption, độ dài câu, thiếu ảnh và OOV.
4. Hai thành viên đọc ba tài liệu và có thể giải thích encoder, attention, causal mask.
5. Proposal được hai người rà, nộp qua kênh môn học; ghi thời điểm nộp thực tế.
6. Mời giảng viên vào repo; chia sẻ TensorBoard/W&B khi bắt đầu có log huấn luyện.

Không đặt ngày deadline cụ thể khi chưa có lịch môn học. Không cần huấn luyện full
cho mốc này; kết quả học phần được đánh giá qua thí nghiệm và phân tích đúng.
