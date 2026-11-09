# Sinh chú thích ảnh tiếng Việt trên KTVIC

Bộ mã và tài liệu cho mốc tuần 1, môn Học sâu, đề tài 05, nhóm 10.
Ảnh RGB đầu vào được mô tả bằng một câu tiếng Việt. So sánh **CNN + LSTM có attention**
với **CNN + Transformer decoder**, đánh giá BLEU và CIDEr.

| Thành viên | MSSV | Phụ trách dự kiến |
|---|---|---|
| Nguyễn Đức Minh | 24022405 | Dữ liệu, tokenizer, CNN, LSTM attention |
| Đoàn Quang Mạnh | 24022339 | Transformer, đánh giá, tổng hợp thí nghiệm |

Cả hai cùng rà mã, phân tích lỗi và viết báo cáo. Đề ghi nhóm ba người;
nhóm cần xác nhận ngoại lệ hai người với giảng viên. Phân công chưa phải xác nhận đóng góp thực tế.


## Cài đặt

Yêu cầu Python 3.13 (môi trường đã kiểm tra). Từ gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Trên Linux dùng `source .venv/bin/activate`. `requirements.txt` cố định các phiên bản
đã có trong môi trường kiểm tra; cài mới từ Internet chưa được nghiệm thu.


## Chuẩn bị dữ liệu một lần

Tải dữ liệu từ nguồn tác giả trong [data/README.md](data/README.md), đặt vào `data/`:
`train_data.json`, `test_data.json`, `train-images.zip`, `public-test-images.zip`.
Thư mục ảnh đã giải nén cũng được hỗ trợ.

```powershell
python scripts/prepare_data.py --content-safe
python scripts/check_week1.py
```

Lệnh đầu audit dữ liệu và tái tạo split theo nội dung; có đọc dữ liệu test để
kiểm tra tính toàn vẹn/trùng ảnh, không tính metric hay chọn mô hình trên test.
Không chạy lại bước này để thay đổi split sau khi đã khóa protocol.
Lệnh thứ hai chỉ đọc train/validation, tạo vocabulary từ train và kiểm tra batch ảnh,
padding, BOS/EOS; lưu báo cáo EDA vào `outputs/week1_check.json`.

Nếu dữ liệu và split đã nằm ở chỗ khác:

```powershell
python scripts/check_week1.py --data-dir 'DUONG_DAN_DU_LIEU' --split-file 'DUONG_DAN_SPLIT_JSON'
```

Notebook [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) hiển thị
ảnh/caption và phân bố độ dài trên train/validation. Output notebook được xóa
trước khi đưa lên Git để tránh phát tán ảnh/caption gốc.
Để dùng notebook trong VS Code/Jupyter, cài thêm `python -m pip install -r requirements-notebook.txt`.
Lệnh kiểm tra từ chối ghi đè báo cáo; khi chạy lại, truyền `--output outputs/week1_check_02.json`.

## Thiết kế thí nghiệm

- A1: ResNet50 spatial encoder + LSTM với additive attention, yêu cầu tối thiểu.
- T1: cùng encoder + Transformer decoder, mô hình so sánh bắt buộc.
- B0: CNN global + LSTM không attention, baseline phụ nếu đủ thời gian.
- Cùng split theo ảnh, vocabulary train-only và quy tắc tính BLEU/CIDEr.
- Chọn checkpoint/decoding trên validation, khóa protocol trước test.
- Ablation attention và phân tích lỗi; CLIP là mở rộng tùy thời gian.

## Cấu trúc

```text
docs/          proposal, bài báo, phân công, kiểm tra và khai báo AI
src/data/      annotation, split, tokenizer, Dataset và collator
src/utils/     seed và tái lập
scripts/       chuẩn bị dữ liệu, kiểm tra tuần 1
tests/         kiểm thử bằng dữ liệu tổng hợp, không cần tải dataset
notebooks/     EDA không lưu output chứa dữ liệu gốc
data/          hướng dẫn tải; dữ liệu thật bị Git bỏ qua
```


