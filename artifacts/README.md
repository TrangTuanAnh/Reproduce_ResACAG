# Kết quả và hình của báo cáo ResACAG

Thư mục này lưu số liệu đã dùng trong báo cáo. Có thể đọc bảng, đối chiếu lượt chạy và vẽ lại hình từ các tệp tổng hợp.

| Nội dung | Tệp và thư mục |
| --- | --- |
| **Dữ liệu và EDA** | [Thống kê dữ liệu và đồ thị](eda/dataset_summary.json), [kiểm tra chất lượng đặc trưng](eda/feature_diagnostics.json) và [baseline tra nhãn theo đầu kết nối](eda/endpoint_lookup.json). [Thông tin nguồn và checksum của CIC-Darknet](inputs/darknet_provenance.json) |
| **Hai lượt tái hiện chính** | [CIC-Darknet2020](reproduction/darknet/) và [CSE-CIC-IDS2018](reproduction/cse_cic/), cùng dùng seed 42. Trong mỗi thư mục, `data_audit.json` ghi dữ liệu và cách chia tập. `resacag/` chứa cấu hình, lịch sử huấn luyện, epoch được chọn và kết quả đánh giá |
| **Đối chứng E1–E8** | [Bảng kết quả](lab6/lab6_summary.csv), [so sánh ghép cặp](lab6/lab6_comparisons.csv), [biên bản thí nghiệm](lab6/protocols/) và [bản ghi theo seed](lab6/runs/). E8 trong báo cáo tương ứng Lab 6 E10 và có hai bảng tổng hợp riêng ([chỉ số](lab6/e10_reported_summary.csv), [so sánh](lab6/e10_reported_comparisons.csv)) |
| **Chia theo IP nguồn (E9)** | [Giao thức](source_ip/protocol.md), [chỉ số từng lượt chạy](source_ip/metrics.csv) và [bảng tổng hợp](source_ip/summary.csv). `RAND` là chia ngẫu nhiên theo luồng, `GRP` là chia theo IP nguồn |
| **Hình báo cáo** | [Các hình PDF](report_figures/) về phân bố lớp, huấn luyện, ma trận nhầm lẫn, mức đồng nhãn của luồng lân cận và kết quả đối chứng |

Tên thư mục trong `lab6/runs/` dùng ID của Lab 6. Ví dụ, lượt so sánh chính E1 trong báo cáo nằm ở `lab6/runs/E6/`. Xem [bảng đối chiếu ID](../README.md#chạy-lại-theo-phạm-vi-báo-cáo) để tìm các lượt khác. Các tệp này lưu kết quả ở cấp lớp, lượt chạy và phép so sánh. Dataset để chạy lại được tải riêng theo [hướng dẫn dữ liệu](../data/README.md).

## Kiểm tra và vẽ lại hình

[manifest.json](manifest.json) ghi đường dẫn, kích thước và SHA-256 của các tệp trong gói. Từ thư mục chứa `reproduce_report.py`, chạy hai lệnh sau. Lệnh vẽ hình cần các thư viện trong `requirements.txt` và ghi hình mới vào `runs/figures/`.

```powershell
python reproduce_report.py verify
python reproduce_report.py figures
```
