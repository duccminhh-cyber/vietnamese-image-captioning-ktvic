# Dữ liệu KTVIC

Nguồn: Pham và cộng sự (2024), [bài báo](https://arxiv.org/abs/2401.08100),
[repo tác giả](https://github.com/pacman-ctm/ktvic).
Repo tác giả cung cấp train_data.json, test_data.json và link thư mục ảnh
Google Drive trong README. Link EVJVQA cũ đã bị gỡ theo ghi chú của tác giả;
dùng đường dẫn ảnh được cập nhật trong repo đó.

Đặt train_data.json, test_data.json, train-images.zip và public-test-images.zip
trong data/. Thư mục train-images/ và public-test-images/ đã giải nén cũng dùng được.

Từ gốc repo, chạy `python scripts/prepare_data.py --content-safe` một lần.
Script tạo split theo ảnh, gom ảnh trùng byte cùng partition và cách ly bản
train có nội dung trùng test. Nó không tính điểm test. Dữ liệu raw không được sửa.
Với bản local hiện có: train 3.202 ảnh, val 565 ảnh, test 558 ảnh; cách ly 2 ảnh.
Mỗi ảnh có 5 reference captions. Raw official train có 3.769 ảnh.

Ảnh gần trùng nhưng khác bytes không được loại bằng SHA256. Audit lịch sử ở dự án
gốc ghi nhận 13 ảnh val gần trùng train; phải công bố hạn chế, không đổi split
sau khi biết kết quả để nâng điểm số.

Vocab fit trên train, không trên val/test. Chuẩn Unicode NFC, lowercase, whitespace,
giữ underscore trong segment_caption; BOS/EOS/PAD/UNK có ID cố định.

Raw JSON, ảnh, ZIP và split manifest bị Git bỏ qua; truy cập qua nguồn tác giả,
không phân phối lại dataset trong repo này. Quyền phân phối lại chưa được xác minh.
