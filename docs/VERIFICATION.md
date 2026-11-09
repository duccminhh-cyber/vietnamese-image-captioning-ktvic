# Kiểm tra bản tuần 1

Thực hiện ngày 09/11/2026 trên Windows, Python 3.13.5, CPU.
torch 2.14.1+cpu; torchvision 0.29.1+cpu; Pillow 12.3.0; NumPy 2.5.3.
Các kiểm tra dùng môi trường có sẵn; chưa xác minh clean install qua Internet.

## Kiểm thử dữ liệu tổng hợp

`python -m unittest discover -s tests -v`: **11/11 PASS**.
Bao gồm split theo ảnh, ngăn leakage, vocabulary train-only, ảnh thiếu/hỏng,
giải nén an toàn, giữ reference captions, truncation và padding mask.

## Dữ liệu thật

`python scripts/check_week1.py --data-dir <data> --split-file <split>`: **PASS**.
Không đọc file ảnh/annotation official test; kiểm tra test bằng manifest đã khóa.

| Chỉ số | Train | Validation |
|---|---:|---:|
| Ảnh | 3.202 | 565 |
| Captions | 16.010 | 2.825 |
| Captions/ảnh | 5 | 5 |
| Caption trống | 0 | 0 |
| Ảnh thiếu/không đọc được | 0 | 0 |
| Độ dài trung bình (token) | 11,008 | 11,071 |
| Phân vị 95 độ dài | 15 | 15 |
| Tỷ lệ token OOV | 0,325% | 1,180% |
| Batch ảnh | 4 × 3 × 224 × 224 | 4 × 3 × 224 × 224 |
| Padding, BOS, EOS | PASS | PASS |

Vocabulary 1.627 token gồm bốn token đặc biệt; fit train với ngưỡng tần suất 2.
OOV train khác 0 vì có ngưỡng tần suất. Báo cáo JSON đầy đủ được lưu local ở
outputs/week1_check.json, không chứa raw captions hoặc ảnh.
Image.verify kiểm tra cấu trúc toàn bộ ảnh train/val; decode/transform được chạy
trên batch mẫu. Đây không phải kiểm định chất lượng mô hình hay kiểm định decode mọi ảnh.

## Proposal

XeLaTeX + BibTeX biên dịch thành 2 trang A4, chế độ ACL final. Đã render và xem
cả hai trang; không có cảnh báo overfull, thiếu glyph hoặc citation chưa định nghĩa.
Không thay đổi lề/cỡ chữ template để giảm số trang.

## Notebook và gói Git

Đã thực thi tuần tự toàn bộ code cell của notebook trên dữ liệu thật bằng backend
đồ thị headless; PASS. Notebook bàn giao vẫn không có output ảnh/caption hay execution count.
Đã kiểm tra 34 tệp được chọn: không có raw dataset, trọng số, cache hoặc mẫu
credential phổ biến; liên kết Markdown nội bộ và hash nguồn đều hợp lệ.
Không coi kiểm tra mẫu credential là chứng minh tuyệt đối không có mọi loại bí mật.

## Phần chưa xác nhận

GPU và clean install; hai thành viên đọc bài báo/hiểu mã; giảng viên chấp nhận
hai người; deadline và việc nộp bài. Việc tạo repo/push GitHub cần kết nối tài khoản.
