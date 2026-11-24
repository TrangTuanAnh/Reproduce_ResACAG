# PROTOCOL E7a — ResACAG với trọng số lớp (THĂM DÒ)

**Trạng thái: viết trước khi chạy E7a.** Đây là thí nghiệm thăm dò, phát sinh sau khi đã thấy E0, E1, E2, E5, E6. Không gọi là cải tiến thuật toán hay novelty.

## 1. Câu hỏi

Gap macro-F1 giữa ResACAG và DecisionTree trong E6 có phải chủ yếu đến từ việc ResACAG không dùng trọng số lớp, trong khi baseline cây có không?

## 2. Lý do

- E6: ResACAG có weighted-F1 cao nhất (96,3) nhưng macro-F1 thấp hơn DecisionTree (83,5 so với 85,7). Dạng này gợi ý yếu ở lớp hiếm.
- Baseline cây dùng `class_weight='balanced'`, ResACAG dùng cross-entropy không trọng số. Đây là khác biệt thiết lập, không phải khác biệt kiến trúc.

## 3. Thay đổi duy nhất

Cross-entropy trong huấn luyện và đánh giá của ResACAG dùng trọng số lớp kiểu sklearn `balanced`, tính trên tập train của từng seed:

w_c = n_train / (số lớp có mặt trong train × n_train,c)

Không đổi kiến trúc, siêu tham số (hidden 32, batch 500, dropout 0,5, Adam lr 0,007, tối đa 12 epoch, patience 3, clip 5), split S-A, seed 42–46, node2vec transductive.

**Lưu ý kỹ thuật:** trọng số tác động đến huấn luyện, nên macro-F1 của E7a khác E6 là điều được kỳ vọng. Riêng `val_loss` và `test_loss` được ghi log bằng loss có trọng số (vì cùng hàm F trong repo), và tie-break chọn checkpoint theo −val_loss cũng dùng loss này. Khi báo cáo, `val_loss`/`test_loss` của E7a không so trực tiếp với E6.

## 4. So sánh

- Chính: E7a ResACAG − E6 ResACAG (tác động riêng của trọng số lớp).
- Phụ: E7a ResACAG − từng baseline E6 (DT, RF, ET, HGB, CatBoost), cùng test set.
- Họ so sánh E7a hiệu chỉnh Holm riêng, không gộp với họ E6.
- Bootstrap cụm theo IP nguồn, 2.000 lần, như các thí nghiệm khác.

## 5. Đọc kết quả

- Nếu CI của (E7a − E6 ResACAG) không chứa 0 và dương: trọng số lớp có ảnh hưởng đến macro-F1 của ResACAG. Báo cáo là phát hiện thăm dò trên dataset này.
- Nếu CI chứa 0: không tìm thấy tác động của trọng số lớp. Gap với DecisionTree vẫn có thể do yếu tố khác.
- Dù kết quả thế nào, E7a **không** thay thế E6. E6 là kết quả chính đã khóa.

## 6. Không được nói

- "ResACAG được cải thiện" hay "vượt DecisionTree" nếu CI vẫn chứa 0 hoặc âm sau Holm.
- "Đóng góp thuật toán" hay "novelty".
- Dùng E7a làm kết quả chính để thay E6.

## 7. Ghi chép

Ghi ngày giờ chạy, hash file này, commit repo, và `class_weights.json` của mỗi seed trong `lab\6\E7a\runs\cse_cic\seed_<s>\`.
