# Tài liệu nền tảng

Ghi chú do AI hỗ trợ soạn ngày 09/11/2026 từ nguồn tác giả; không xác nhận thay
hai thành viên rằng đã đọc toàn văn. Sau khi đọc, mỗi người bổ sung nhận xét riêng.

## 1. KTVIC

Pham, Nguyen, Vuong và Ha (2024), *KTVIC: A Vietnamese Image Captioning Dataset on the Life Domain*.
[Bài báo](https://arxiv.org/abs/2401.08100), [repo tác giả](https://github.com/pacman-ctm/ktvic).

Bộ dữ liệu chú thích ảnh tiếng Việt thuộc miền đời sống là nguồn dữ liệu của dự án.
Cần đối chiếu định dạng annotation và quy trình đánh giá của bài báo với dữ liệu local.
Áp dụng: dùng caption tiếng Việt có sẵn, chia theo ảnh và giữ toàn bộ references cùng ảnh.
Câu hỏi khi đọc: số ảnh trong mỗi phần, nguồn ảnh, cách viết caption và tách từ là gì?
Nhóm dùng split validation riêng nên không coi điểm số là so sánh trực tiếp với bảng bài báo.

## 2. Show Attend and Tell

Xu và cộng sự (2015), *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*, ICML.
[Trang PMLR và toàn văn](https://proceedings.mlr.press/v37/xuc15.html).

Mô hình học chú ý vào các vùng ảnh khi sinh từng từ; bài báo trình bày attention
dạng xác định và ngẫu nhiên. Dự án chọn soft attention để huấn luyện bằng backpropagation.
Áp dụng: CNN giữ đặc trưng theo vùng; dùng trạng thái LSTM để tính trọng số,
tổng hợp context rồi sinh token. Đây là vận dụng ý tưởng, không phải tái lập nguyên xi bài báo.
Câu hỏi khi đọc: trọng số được chuẩn hóa trên chiều nào, context đi vào decoder ra sao?

## 3. Attention Is All You Need

Vaswani và cộng sự (2017), *Attention Is All You Need*.
[Bài báo gốc](https://arxiv.org/abs/1706.03762).

Transformer thay recurrence bằng attention trong mô hình chuỗi. Dự án vận dụng
decoder cho caption: masked self-attention trên token đã sinh, cross-attention
vào đặc trưng ảnh, cùng thông tin vị trí. Bài báo gốc tập trung dịch máy;
việc ghép CNN với decoder là thiết kế của dự án.
Câu hỏi khi đọc: vì sao cần causal mask, padding mask và positional encoding?

## 4. Show and Tell (baseline bổ sung)

Vinyals, Toshev, Bengio và Erhan (2015), *Show and Tell: A Neural Image Caption Generator*, CVPR.
[Bài báo](https://arxiv.org/abs/1411.4555).

Mô hình kết hợp biểu diễn ảnh và bộ sinh câu hồi quy để học xác suất chú thích
theo ảnh. Đây là tài liệu tham chiếu cho baseline CNN–LSTM không attention;
baseline dự án dùng ResNet50 nên không phải tái lập nguyên kiến trúc của bài báo.
Đối chiếu với Show, Attend and Tell để hiểu khác biệt giữa vector toàn cục
và việc chọn vùng ảnh theo từng token.

## Nhật ký đọc của nhóm

| Thành viên | Tài liệu | Ngày đọc | Nhận xét/câu hỏi |
|---|---|---|---|
| Nguyễn Đức Minh | Chưa xác nhận | — | Bổ sung sau khi đọc |
| Đoàn Quang Mạnh | Chưa xác nhận | — | Bổ sung sau khi đọc |
