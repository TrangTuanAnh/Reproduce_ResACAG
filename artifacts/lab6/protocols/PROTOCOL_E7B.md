# PROTOCOL E7b — biểu diễn node không phụ thuộc ID IP:port (THĂM DÒ, phát triển)

**Trạng thái: viết trước khi chạy E7b.** Phát sinh sau khi đã thấy E0–E7a. Đây là một cải tiến có kiểm chứng, nhưng vẫn là thăm dò trên một dataset, một split, và không gọi là novelty hay SOTA.

## 1. Câu hỏi

Điểm yếu đã đo được ở E5: khi embedding học từ train, khoảng 88% node trong test có embedding bằng 0 vì chưa thấy trong train. Nếu thay embedding theo ID bằng đặc trưng tổng hợp từ các flow đi qua node, ResACAG có còn bị phụ thuộc vào việc đã thấy IP:port trước đó không, và có vượt được DecisionTree không?

## 2. Thiết kế

Thay node2vec bằng biểu diễn node từ đặc trưng flow, giữ nguyên mọi thứ khác của ResACAG (kiến trúc, siêu tham số, split S-A, seed 42–46, cross-entropy không trọng số như E6).

Với mỗi IP:port:
1. Lấy tất cả flow có node đó làm nguồn hoặc đích (cả train, validation và test).
2. Tính trung bình của 77 đặc trưng đã chuẩn hóa (scaler fit trên train, như pipeline gốc) trên các flow đó.
3. Thêm log(1 + số flow chứa node).
4. Chuẩn hóa cột theo các node có ít nhất một flow, rồi PCA xuống 32 chiều (bằng `hidden`) để khớp kích thước embedding của ResACAG.

Node không có flow nào nhận vector 0.

## 3. Tính hợp lệ và rò rỉ

- **Không dùng nhãn** ở bất kỳ bước nào. Scaler fit trên train như cũ. PCA và chuẩn hóa cột fit trên các node, không dùng nhãn.
- **Dùng flow của test để tính đặc trưng node** (không nhãn). Đây là cùng loại transductive như node2vec ở E6, và phải nêu khi báo cáo.
- Điểm khác E6: embedding không đến từ ID mà từ đặc trưng. Về nguyên tắc, node chưa thấy trong train vẫn có vector khác 0 nếu có flow, không phụ thuộc vào việc đã học ID của nó.
- Cần lưu ý: khi triển khai thật, node mới chỉ có vector khi có flow của nó. Đây là giả định không được kiểm chứng ở đây.

## 4. Thí nghiệm

- E7b: ResACAG với biểu diễn đặc trưng node, S-A, seed 42–46.
- So sánh chính: E7b − E6 ResACAG (tác động của thay đổi biểu diễn node).
- So sánh phụ: E7b − từng baseline E6 (DT, RF, ET, HGB, CatBoost), cùng test set.
- Họ hiệu chỉnh Holm riêng cho E7 (không gộp với E6).
- Bootstrap cụm theo IP nguồn, 2.000 lần.

## 5. Đọc kết quả

- Nếu E7b − DecisionTree có CI không chứa 0 và dương sau Holm: claim "GNN với biểu diễn node không ID vượt baseline bảng" được ủng hộ, chỉ trên dataset này.
- Nếu CI chứa 0 hoặc âm: không ủng hộ claim. Ghi kết quả âm đầy đủ.
- Nếu E7b vượt E6 nhưng vẫn thấp hơn DT: ghi là cải thiện nội bộ, chưa đủ để vượt baseline.

## 6. Không được nói

- "Vượt SOTA", "novelty thuật toán", "IDS vận hành".
- Rằng E7b chứng minh khả năng tổng quát cho IP mới khi triển khai. Chỉ được nói là đo trên split có 88% node chưa thấy.
- So sánh trực tiếp E7b với E5 như thể cùng điều kiện (E5 dùng node2vec train-only, không có flow test trong embedding).

## 7. Ghi chép

Ghi commit, hash file này, và `node2vec.npy` (thực chất là embedding đặc trưng) cùng `scaler.npz` của mỗi seed trong `lab\6\E7b\runs\cse_cic\seed_<s>\`. Lưu ý: tên file `node2vec.npy` và khóa `node2vec` trong metadata là tên cũ; nội dung là embedding đặc trưng.
