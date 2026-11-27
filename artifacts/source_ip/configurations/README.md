# Đối chứng topology trên bốn cấu hình IP nguồn

Brute-force và DoS mỗi lớp có hai IP nguồn. Mỗi mã `Cij` chọn IP test thứ `i` của Brute-force và IP test thứ `j` của DoS theo thứ tự chuỗi. Với mỗi cấu hình, chia các IP còn lại bằng ba seed 301 đến 303. Cách chia giữ IP nguồn giữa train và test rời nhau.

- [Giao thức lượt gốc](protocol.json) ghi seed, SHA-256 bộ dữ liệu và phiên bản thư viện.
- [Số luồng theo lớp](class_support.csv) được dựng lại từ dữ liệu đã kiểm SHA-256. Cả 12 split có đủ bảy lớp ở train và test.
- [Điểm từng lượt](metrics.csv) là bản lưu của 12 split × 3 mô hình × 5 bộ đặc trưng.
- [Bảng chênh lệch](summary.csv) so sánh theo cặp cùng mô hình và split. `mean_pp` là trung bình đều của bốn cấu hình theo điểm phần trăm. `wide_lo` và `wide_hi` lấy cận ngoài của hai khoảng xấp xỉ 95%, tính lần lượt trên bốn trung bình cấu hình và trên 12 split có điều chỉnh theo tỷ lệ test/train.

`flow+topo_c` thêm tám thống kê IP/cổng tính từ các luồng không nhãn đã bắt đầu trong 300 giây trước đó, lấy lịch sử của cả hai tập. `flow+topo_r` tính cùng thống kê sau khi hoán vị IP nguồn và đích. `flow+topo_t` dùng toàn tập theo thời gian. Chênh lệch trung bình dương trong mỗi mô hình đi kèm với khoảng xấp xỉ rộng còn chứa 0 cho so sánh quá khứ trừ chỉ luồng.

Từ gốc kho mã, sau khi chuẩn bị dataset, chạy `python reproduce_report.py source-ip-configurations --data-root ./data`. Lệnh `python reproduce_report.py report-tables` dựng lại bảng Phụ lục D từ kết quả đã lưu.
