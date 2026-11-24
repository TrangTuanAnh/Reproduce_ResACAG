# PROTOCOL E9 — đối chứng topology ngẫu nhiên (giữ kiến trúc, phá cấu trúc)

**Trạng thái: viết trước khi chạy E9.** Đối chứng cho E8: E8 cho thấy ResACAG vượt MLP, nhưng MLP là đối chứng yếu nên chưa tách được "đồ thị" khỏi "kiến trúc và dung lượng".

## 1. Câu hỏi

Với cùng kiến trúc ResACAG, topology IP:port thật có tốt hơn topology ngẫu nhiên có cùng bậc nút hay không?

## 2. Thiết kế

- Giữ nguyên toàn bộ ResACAG, siêu tham số, split S-A, seed 42–46, cross-entropy không trọng số, node2vec transductive.
- Chỉ thay đổi cấu trúc: với mỗi seed, hoán vị ngẫu nhiên cột IP:port nguồn giữa các flow, và hoán vị riêng cột IP:port đích. Đặc trưng flow và nhãn không đổi.
- Hệ quả: mỗi nút có cùng tổng số flow làm nguồn và làm đích như trước (bậc được giữ), nhưng quan hệ giữa các nút bị phá vỡ. Đây là đồ thị ngẫu nhiên bảo toàn bậc.
- Hoán vị được thực hiện trước khi chia split, trên toàn bộ dataset, bằng hạt giống cố định theo seed (`seed + 777`), nên mọi lần chạy lặp lại được.

## 3. So sánh

- Chính: E6 ResACAG (topology thật) − E9 ResACAG (topology ngẫu nhiên). Đây là tác động của cấu trúc thật, tách khỏi kiến trúc.
- Phụ: E9 − E8 MLP, để xem topology ngẫu nhiên có còn hơn MLP không.
- Họ hiệu chỉnh Holm riêng cho E9.
- Bootstrap cụm theo IP nguồn, 2.000 lần.

## 4. Đọc kết quả

- Nếu CI của (E6 − E9) dương sau Holm: cấu trúc thật có giá trị vượt quá kiến trúc. Claim "topology hữu ích" được ủng hộ mạnh hơn, trên dataset này.
- Nếu CI chứa 0: gain so với MLP trong E8 phần lớn đến từ kiến trúc, không phải từ cấu trúc thật. Phải ghi rõ điều này và không dùng E8 để kết luận về topology.
- Nếu E9 gần E6 hoặc tốt hơn: kết quả E8 phụ thuộc vào thiết kế ResACAG, không phải vào đồ thị.

## 5. Không được nói

- "Topology thật vượt topology ngẫu nhiên" nếu CI chứa 0.
- Rằng E9 là bằng chứng về mọi kiểu đồ thị. Chỉ là một cách phá cấu trúc (hoán vị bậc giữ nguyên).

## 6. Ghi chép

Ghi commit, hash file này, và `config.json` của `resacag` trong `lab\6\E9\runs\cse_cic\seed_<s>\`. Metadata ghi `topology = rewired_random_control`.
