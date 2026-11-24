# PROTOCOL E8 — đối chứng không đồ thị (MLP trên đặc trưng flow)

**Trạng thái: viết trước khi chạy E8.** Đây là đối chứng cần thiết để trả lời câu hỏi topology có giúp ích hay không. Không phải thí nghiệm thăm dò kiểu E7.

## 1. Câu hỏi

Trên `cse_cic` với split S-A, ResACAG có tốt hơn cùng mô hình nhưng bỏ hoàn toàn đồ thị hay không? Nếu không, claim "topology hữu ích" không được ủng hộ trên dataset này.

## 2. Thiết kế

- Biến thể `mlp` có sẵn trong `resacag/models.py`. Chỉ dùng đặc trưng của chính flow (77 đặc trưng đã chuẩn hóa), qua MLP hai lớp ẩn 32 chiều. Không dùng cạnh lân cận, không dùng embedding topology.
- Huấn luyện đúng pipeline của ResACAG: cùng siêu tham số (lr 0,007, batch 500, dropout 0,5, tối đa 12 epoch, patience 3, clip 5), cùng checkpoint theo macro-F1 validation, cùng cross-entropy không trọng số.
- Cùng split S-A (stratified 50/30/20, `random_state` = seed), cùng seed 42–46, cùng test set với E6.
- Biểu diễn node không được dùng vì `mlp` không đọc topology. Để tiết kiệm thời gian, node2vec được thay bằng vector 0 (đường dẫn `node_embedding = none`). Điều này không ảnh hưởng vì `mlp` không truy cập embedding.

**Lưu ý:** ResACAG cũng nối đặc trưng gốc của flow vào lớp phân loại (`external=True`). Vì vậy so sánh ResACAG với MLP đo tác động của phần đồ thị (tổng hợp lân cận và topology), không phải tác động của đặc trưng.

## 3. So sánh

- Chính: E6 ResACAG − E8 MLP. Đây là tác động của đồ thị trên cùng dữ liệu, cùng test set.
- Phụ: E8 MLP − E6 DecisionTree và E8 MLP − các baseline cây khác. Cho biết MLP có cạnh tranh được với baseline bảng không.
- Họ hiệu chỉnh Holm riêng cho E8.
- Bootstrap cụm theo IP nguồn, 2.000 lần, như các thí nghiệm khác.

## 4. Đọc kết quả

- Nếu CI của (E6 ResACAG − E8 MLP) nằm hoàn toàn dương sau Holm: đồ thị có đóng góp đo được trên dataset này. Claim "topology hữu ích" được ủng hộ, chỉ trong phạm vi này.
- Nếu CI chứa 0: không phát hiện đóng góp của đồ thị. Claim "topology hữu ích" không được ủng hộ trên dataset này.
- Nếu CI âm: MLP tốt hơn ResACAG. Ghi là kết quả âm đầy đủ.
- Nếu MLP gần DT: nghĩa là đặc trưng flow chứa phần lớn thông tin, và ResACAG không khai thác thêm được điều gì đáng kể từ đồ thị.

## 5. Không được nói

- "Đồ thị giúp" nếu CI chứa 0.
- "Đồ thị không có ích" như một kết luận tổng quát. Chỉ nói trên dataset này, split S-A, và 5 seed.
- Điều gì về khả năng tổng quát cho nút mới (xem E7c và giới hạn của E7b).

## 6. Ghi chép

Ghi commit, hash file này, `manifest.json`, và `config.json` của biến thể `mlp` trong `lab\6\E8\runs\cse_cic\seed_<s>\mlp\`.
