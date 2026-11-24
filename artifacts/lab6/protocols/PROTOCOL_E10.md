# PROTOCOL E10 — thông tin đồ thị có bổ sung giá trị cho cây quyết định không (THĂM DÒ)

**Trạng thái: viết trước khi chạy E10.** Phát sinh sau khi đã thấy E0–E9. Thăm dò, không phải kết quả chính.

## 1. Câu hỏi

E9 cho thấy cấu trúc đồ thị thật có ích cho ResACAG, nhưng DecisionTree chỉ dùng đặc trưng flow (E6) vẫn ngang hoặc hơn ResACAG. Đặc trưng lấy từ đồ thị có cộng thêm giá trị cho các mô hình cây không?

## 2. Thiết kế

Ba biến thể đầu vào cho DecisionTree (`dt`) và HistGradientBoosting (`hgb`), tham số khóa như E6:
- **flow** (đã có ở E6): 77 đặc trưng flow. Không chạy lại.
- **graph** (`dt_graph`, `hgb_graph`): 77 đặc trưng flow + vector node của nút nguồn (32 chiều) + vector node của nút đích (32 chiều). Vector node tính đúng như E7b: trung bình 77 đặc trưng đã chuẩn hóa (scaler fit trên train) của mọi flow đi qua nút, cộng log(1 + số flow), chuẩn hóa cột, PCA xuống 32 chiều. Không dùng nhãn.
- **rewired** (`dt_rewired`, `hgb_rewired`): như graph nhưng cột nút nguồn và nút đích được hoán vị giữa các flow bằng đúng hoán vị của E9 (`seed + 777`). Giữ bậc nút, phá quan hệ.

Split S-A, seed 42–46, cùng test set với E6.

## 3. Tính hợp lệ

- Vector node dùng flow của cả train, validation và test (không nhãn): transductive, cùng loại với E7b/E6. Phải nêu khi báo cáo.
- Vector node của một nút có thể bao gồm chính flow đang dự đoán. Đây không rò rỉ nhãn nhưng làm vector node gần với đặc trưng của flow ở nút có ít flow.

## 4. So sánh (họ Holm riêng cho E10)

- `dt_graph` − E6 `dt`, `hgb_graph` − E6 `hgb`: đồ thị có bổ sung giá trị cho cây không.
- `dt_graph` − `dt_rewired`, `hgb_graph` − `hgb_rewired`: lợi ích (nếu có) đến từ cấu trúc thật hay chỉ từ việc thêm cột.
- `dt_rewired` − E6 `dt`: thêm cột nhiễu có làm hại không.
- `dt_graph` − E6 ResACAG: cây có đồ thị so với GNN.
- Bootstrap cụm theo IP nguồn, 2.000 lần.

## 5. Đọc kết quả

- Nếu `dt_graph` − `dt` dương, CI không chứa 0 sau Holm, và `dt_graph` − `dt_rewired` cũng dương: topology thật bổ sung thông tin vượt trên đặc trưng flow, kể cả cho cây.
- Nếu CI chứa 0: thông tin đồ thị phần lớn dư thừa với đặc trưng flow đối với cây trên dataset này. Ghi là kết quả âm.
- Nếu `dt_rewired` thấp hơn `dt`: thêm cột nhiễu làm hại; ghi lại.

## 6. Không được nói

- Novelty, SOTA, IDS vận hành.
- Rằng kết quả tổng quát cho nút mới: vector node ở đây dùng flow của cả split.
- Rằng đồ thị "giúp" nếu chỉ `graph` hơn `flow` mà `graph` không hơn `rewired`.
