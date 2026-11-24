# AMENDMENT 1 — lab/6 (viết trước khi chạy lần đầu)

Ghi các chỗ code khác PROTOCOL.md. Viết trước khi có bất kỳ số liệu nào của lab 6, nên không phải đổi protocol sau khi đã thấy kết quả.

1. **TabM bị loại khỏi lab 6.** `phase3_study.py` gọi `.cuda()` cứng trong nhánh TabM; máy không có CUDA NVIDIA nên không chạy được. Không sửa repo gốc. Ghi `excluded` trong `manifest.json`. Hệ quả: E1 và E6 chỉ có DT, RF, ET, HGB, CatBoost làm baseline bảng.
2. **Chia theo thời gian dùng `start_us`, không dùng `Timestamp`.** `Timestamp` là đồng hồ 12 giờ không có AM/PM (README dataset), sắp xếp trực tiếp sẽ sai. `start_us` là thời điểm bắt đầu flow lấy từ PCAP.
3. **CatBoost không dùng early stopping.** Dùng đúng tham số của `phase3_study.py` (500 vòng cố định). Không chọn số vòng bằng nhãn validation.
4. **Thêm baseline tốt nhất theo validation vào so sánh.** Đúng yêu cầu E1/E6 trong PROTOCOL; baseline chọn theo macro-F1 validation trung bình qua seed.
5. **E0: A = chạy E6.** Cùng dữ liệu, split, seed và mô hình với B nên không chạy lại A.
6. **Lớp `class_weight='balanced'` cho DT, RF, ET** (khớp HGB và CatBoost trong `phase3_study.py`). ResACAG dùng cross-entropy không trọng số (theo repo). Đây là bất đối xứng cần ghi khi báo cáo.
7. **E2 near-duplicate** chưa chạy. Chỉ chạy exact-duplicate (PROTOCOL cho phép báo cáo tỷ lệ trước khi quyết định loại thêm).
8. **E1 loại 168 dòng không có `start_us`.** Kiểm tra trên máy người dùng: cả 168 dòng đều có `join_tier = distrinet_exact` và `pcap_member = None`, nghĩa là được ghép endpoint đúng nhưng không có trong PCAP nên không có thời gian bắt đầu. Phân bố theo nhãn: Benign 163 (trên 171.301), Infiltration 5 (trên 5.514, khoảng 0,09%). Phân bố theo ngày: 14/02 (36), 15/02 (53), 28/02 (41), 02/03 (38). Quyết định: E1 bỏ 168 dòng này; E2, E5, E6, E0 giữ nguyên toàn bộ 244.776 dòng. Báo cáo số dòng bị bỏ trong E1. Quyết định này được đưa ra trước khi có bất kỳ kết quả test nào của E1.
