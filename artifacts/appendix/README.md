# Hai phép thử thăm dò

[Cổng cosine](gate/scores.csv) đối chiếu hai cách lấy trung bình lân cận của ResACAG trên ba seed. [Giao thức lượt gốc](gate/protocol.json) ghi cấu hình, môi trường và SHA-256 của embedding được dùng chung. Mỗi thư mục theo seed và cách tổng hợp lưu điểm validation từng vòng và chỉ số test.

[Lịch sử luồng](history/scores.csv) so sánh hai mô hình cây với đầu vào luồng đơn và đầu vào có thêm lịch sử. [Độ phủ](history/coverage.csv) ghi tỷ lệ luồng test có lịch sử ở ít nhất một đầu IP và tỷ lệ theo từng phía. [Giao thức lượt gốc](history/protocol.json) ghi cách chia GRP và cửa sổ 300 giây. Các tệp theo seed ghi số luồng theo lớp và kết quả theo mô hình, đặc trưng.

Các tệp CSV dùng tỷ lệ từ 0 đến 1 cho macro-F1, weighted-F1, recall và precision. Bảng trong báo cáo tính trung bình theo seed rồi nhân 100 để trình bày theo phần trăm. Chênh lệch là hiệu của biến thể và đối chứng trên chính các seed đó, lấy trước khi làm tròn.

Mã đã đóng gói nằm trong [clean_code/lab/9](../../clean_code/lab/9/code/run_improvements.py). Mã băm trong `protocol.json` thuộc các lượt chạy gốc. Bản đóng gói đổi đường dẫn nhập mã, cho phép tạo lại cache node2vec và xuất CSV độ phủ trực tiếp. [Hướng dẫn chạy lại](../../README.md#phép-thử-ở-phụ-lục-a)
