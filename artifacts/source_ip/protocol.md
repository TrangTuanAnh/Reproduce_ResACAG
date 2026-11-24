# Lab 7 – PROTOCOL (viết TRƯỚC khi chạy; commit riêng để chứng minh không chỉnh theo kết quả)

## Câu hỏi
Trên CSE-CIC-IDS2018 (244.776 flow, endpoint khôi phục từ pcap), đặc trưng **topology** (thống kê đồ thị IP theo cửa sổ quá khứ) có thêm giá trị dự báo cho mô hình cây **ngoài (a) đặc trưng flow và (b) danh tính endpoint** hay không, khi đánh giá chống đường tắt? Và phần lợi của đặc trưng topology "transductive" (kiểu E10) có biến mất khi tính nhân quả không?
Không tuyên bố phương pháp mới; đóng góp chỉ là phép đo có đối chứng (kể cả kết quả âm). Novelty chỉ ở mức "chưa thấy" (tài liệu mới đọc abstract).

## Dữ kiện nền (đếm trước khi thiết kế, từ parquet)
Mỗi lớp tấn công nằm trong 1–2 ngày; số IP nguồn/lớp: benign 9.333, c1: 2, c2: 2, c3: 10, c4: 6, c5: 11, c6: 545. Nhãn ≈ ngày ⇒ split thời gian toàn cục suy biến (E1). 168 dòng không có `start_us`.

## Dữ liệu / đặc trưng
- Nguồn: `code/dataset/cse_cic/cse_cic.parquet`. Bỏ 168 dòng thiếu `start_us`, BÁO số dòng theo lớp bị bỏ.
- Họ đặc trưng (tất cả KHÔNG dùng nhãn):
  - `flow`: 77 đặc trưng CICFlowMeter.
  - `id`: 4 octet Src IP, 4 octet Dst IP, Src Port, Dst Port (số). Là "trần đường tắt", lưu ý: thứ tự số có thể tạo tín hiệu subnet, không phải danh tính thuần.
  - `topo_c` (nhân quả): với flow i tại thời điểm t=start_us, chỉ dùng flow có start_us < t và ≥ t−W (W=300 s, cố định, KHÔNG chỉnh). 8 đặc trưng: số flow cùng src IP; số dst IP khác nhau của src IP; số dst port khác nhau của src IP; số flow cùng dst IP; số src IP khác nhau tới dst IP; số flow cùng cặp (src,dst) IP; số dst port khác nhau tại dst IP; số flow cùng (src IP, dst port). Tính trên TOÀN BỘ flow theo thời gian (kể cả flow thuộc test) nhưng chỉ từ quá khứ, không nhãn: mô phỏng cảm biến triển khai thực tế.
  - `topo_t` (transductive, tham chiếu kiểu E10): cùng 8 đặc trưng nhưng đếm trên toàn bộ dữ liệu không giới hạn thời gian (trừ chính flow).
  - `topo_r` (đối chứng hoán vị): tính `topo_c` sau khi hoán vị độc lập cột Src IP và cột Dst IP giữa các flow (seed+777), giữ phân phối bậc, phá quan hệ cấu trúc.
- Tổ hợp mô hình: `flow`, `id`, `topo_c`, `flow+topo_c`, `flow+topo_t`, `flow+topo_r`, `flow+id`.

## Split (3 loại; kết luận CHÍNH chỉ từ GRP)
- `RAND` (chẩn đoán): stratified 70/30, seed.
- `GRP` (chính): theo nhóm Src IP trong từng lớp: các IP của mỗi lớp chia 70/30 (≥1 IP mỗi phía; lớp c1, c2 chỉ có 2 IP ⇒ 1 IP train, 1 IP test, cấu hình do seed chọn). Test chỉ chứa IP nguồn chưa thấy lúc train. Báo số flow/IP theo lớp và split.
- `TFC` (phụ): mỗi lớp sắp theo start_us, 70% đầu train, 30% cuối test (giữ hỗ trợ lớp; IP có thể lặp).
- Mọi lớp phải có ≥1 flow ở test; nếu không, ghi vào báo cáo là "không đánh giá được". Không lặng lẽ bỏ lớp.

## Mô hình & siêu tham số (CỐ ĐỊNH, không dò, giống lab 6 `tabular_model`)
DT(class_weight=balanced), RF(100 cây, balanced), HGB(200 iter, 31 leaf, lr 0,1, l2=1, balanced, không early stopping). Không chọn checkpoint/ngưỡng; không có tập validation dùng để chọn. Không đụng tập test để tinh chỉnh.

## Metric
Macro-F1 trên các lớp có mặt ở test (chính); recall từng lớp; log loss (RF/HGB; DT cắt xác suất 1e-15). CI: cluster-bootstrap 2000 lần theo Src IP trên dự đoán test, cho hiệu cặp (cùng seed/split).

## Seed
5 seed: 42–46 (seed chọn split GRP/RAND và khởi tạo mô hình). TFC xác định nên chỉ đổi seed mô hình.

## Giả thuyết & tiêu chí (GRP, mô hình HGB và RF; DT báo cáo phụ)
- Biên tương đương thực tiễn: 1,0 điểm macro-F1.
- **H1a (topology có ích ngoài flow):** Δ = F1(flow+topo_c) − F1(flow) có CI 95% dưới > +1,0 điểm ⇒ ủng hộ. CI chứa [−1,+1] ⇒ "không phân biệt được/tương đương". CI trên < −1,0 ⇒ hại.
- **H1b (do cấu trúc, không phải phân phối bậc):** F1(flow+topo_c) − F1(flow+topo_r) CI dưới > 0 (nếu H1a ủng hộ).
- **H1c (ngoài danh tính):** F1(flow+topo_c) − F1(flow+id) CI dưới > 0; và F1(topo_c) so với F1(id) báo cáo.
- **H1d (transductive thổi phồng):** F1(flow+topo_t) − F1(flow+topo_c) báo cáo; > +1,0 ⇒ "lợi E10 một phần do transductive".
- **Bác bỏ H1 (kết luận âm) nếu:** H1a không đạt trên cả HGB lẫn RF trong GRP. Kết quả âm sẽ được báo cáo nguyên văn, không đổi protocol.
- RAND và TFC chỉ để chẩn đoán độ lạm phát; không dùng để cứu giả thuyết.
- Nếu GRP với c1/c2 (2 IP) cho recall ≈ 0, báo riêng; macro-F1 chính vẫn gồm mọi lớp có test.

## Điều kiện thực thi
Pilot trên mẫu 20% (seed 42) để kiểm lỗi/thời gian, KHÔNG dùng số pilot để đổi protocol. Chạy đầy đủ: `timeout 3h` mỗi lệnh, không bắt đầu lượt mới sau 8h sáng. Ghi lệnh/seed/commit vào WORKLOG.md. Kết quả thô vào `lab/7/runs/`. Thay đổi protocol sau commit này phải ghi thành AMENDMENT có lý do và dấu thời gian.
