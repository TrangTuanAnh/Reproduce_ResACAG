# Protocol vòng 6: giá trị của topology IP:port thật dưới split chống leakage

**Trạng thái: CHƯA CHẠY.** Trước khi chạy bất kỳ thí nghiệm nào, ghi ngày giờ, hash SHA-256 của file này và commit hash của repo vào `lab/6/WORKLOG.md`. Sau khi đã chạy, không sửa protocol này; nếu cần đổi, tạo `lab/6/AMENDMENT_<n>.md` và nêu lý do, kèm việc đã nhìn thấy kết quả nào.

## 1. Câu hỏi

Trên `cse_cic` (topology IP:port thật, 244.776 flow), GNN cạnh-flow có vượt các baseline bảng dùng chính đặc trưng flow không, khi chia dữ liệu theo thời gian hoặc loại bỏ dòng trùng lặp?

Đây không phải đề xuất kiến trúc mới. Mục tiêu là một phép đo có đối chứng, trong đó kết quả âm được báo cáo đầy đủ.

## 2. Dữ liệu

- Dataset: `cse_cic`, file `code/dataset/cse_cic/cse_cic.parquet`, SHA-256 `ea1736a132a5ffb5d1b93eae48b5ba4c17433655dbc621236dbf9b3a9fcee8c7`.
- 77 feature CICFlowMeter-V3 thô, cột `label` (7 lớp), 4 cột endpoint (`Src IP`, `Src Port`, `Dst IP`, `Dst Port`) chỉ dùng để dựng đồ thị.
- Cột audit dùng trong protocol này: `Timestamp`, `day`, `join_candidates`, `join_tier`.
- `feature_names.json` là danh sách 77 feature đưa vào mô hình. Endpoint, timestamp, label và cột audit không đưa vào feature.
- Số liệu đã biết (từ repo): homophily 0,952 trên topology thật, 0,517 trên `adj_random`. Các con số này phải được tính lại trên bản dataset của máy này và ghi vào WORKLOG.

## 3. Cách chia dữ liệu

| Mã | Cách chia | Chi tiết |
|---|---|---|
| S-A | Stratified 50/30/20 (train/test/validation) | `random_state` = seed, khớp `resacag/data.py`. Seeds 42–46. |
| S-B | Theo thời gian | Sắp theo `Timestamp`. Train = 50% đầu, validation = 20% kế tiếp, test = 30% cuối. Chia này xác định, nên chỉ seed của mô hình thay đổi. |

Lưu ý: nhiều seed trên cùng một test population không phải nhiều dataset độc lập. Chỉ báo cáo bất định theo bootstrap của test set, không gọi là bất định của quần thể.

Chia theo host nguồn (group split) không nằm trong lab này. Lý do: DDoS có 10 IP nguồn nhưng chỉ 1 endpoint đích, nên split theo host làm mất lớp. Ghi là giới hạn, không thử.

## 4. Mô hình

**Nhóm GNN:**
- `ResACAG`: variant `resacag` trong `resacag/models.py`, siêu tham số giữ nguyên như `lab/5` (hidden 32, batch 500, dropout 0,5, Adam lr 0,007, tối đa 12 epoch, patience 3, clip 5).
- Node2vec mặc định transductive cho `cse_cic` (học trên cấu trúc không nhãn của mọi split). Thiết lập train-only chỉ dùng trong E5.

**Nhóm baseline bảng (chỉ dùng feature, không dùng topology):**
- DecisionTree, RandomForest (100 cây), ExtraTrees, HistGradientBoosting, CatBoost, TabM.
- Tham số khóa trước khi chạy và ghi trong `lab/6/config_tabular.json`. Không tuning trên test. Nếu có tuning, chỉ dùng validation với lưới khóa trước.
- Các mô hình này đã được dùng trong `lab/3` và `lab/5/study`, nhưng kết quả cũ chạy trên bản dựng 244.427 flow. Lab 6 chạy lại trên bản 244.776 flow để không trộn số liệu.

## 5. Chỉ số

- F1 weighted và macro (đơn vị %), là chỉ số chính.
- Log loss, ECE, recall theo từng lớp.
- PR-AUC (average precision một-vs-còn-lại) cho Infiltration và Web Attack, vì hai lớp này hiếm.
- Accuracy không báo cáo đơn lẻ.

## 6. Thống kê

- Với mỗi cặp so sánh (ResACAG so với một baseline), tính chênh lệch macro-F1 trên test set.
- Paired bootstrap, 2.000 lần lấy mẫu lại, lấy mẫu theo cụm IP nguồn (cluster bootstrap) để giữ tương quan giữa các flow cùng host. Khoảng tin cậy 95% là phân vị 2,5% và 97,5%.
- Holm correction cho họ các so sánh trong cùng một thí nghiệm.
- Không dùng p-value permutation như bằng chứng quần thể.

## 7. Thí nghiệm

### E0: nhạy cảm với bước ghép endpoint

- Mục tiêu: kiểm tra kết luận có phụ thuộc vào các dòng ghép mơ hồ không.
- Thiết lập: so hai tập con của dataset. A = toàn bộ dòng. B = bỏ dòng có `join_candidates > 1` (các dòng có nhiều candidate feature-equivalent).
- Chạy ResACAG và DecisionTree trên S-A, seed 42–46, cho cả A và B.
- Đo: Δ = macro-F1(ResACAG) − macro-F1(DecisionTree) trên A và trên B.
- **Ngưỡng bác bỏ:** nếu |Δ(A) − Δ(B)| > 2 điểm, kết luận về topology phải ghi là phụ thuộc bước ghép endpoint.

### E1: chia theo thời gian

- Mục tiêu: kiểm tra gain có còn khi test là dữ liệu tương lai không.
- Thiết lập: S-B. Chạy ResACAG, DecisionTree, RandomForest, HGB, CatBoost, TabM với 5 seed huấn luyện (42–46).
- Đo: Δ = ResACAG − baseline tốt nhất (theo macro-F1 trên validation).
- **Ngưỡng bác bỏ:** nếu khoảng tin cậy 95% của Δ với baseline tốt nhất chứa 0 hoặc âm, claim "ResACAG vượt baseline bảng trên split theo thời gian" bị bác bỏ.
- Lưu ý: khoảng tin cậy chỉ mô tả bất định trên test set của một lần chia thời gian.

### E2: kiểm soát dòng trùng lặp

- Mục tiêu: kiểm tra gain có phụ thuộc vào dòng lặp lại giữa train và test không.
- Thiết lập: băm 77 feature của mỗi dòng. Loại khỏi test các dòng có hash xuất hiện trong train (exact). Chạy ResACAG và DecisionTree trên S-A, seed 42–46. Ghi tỷ lệ dòng bị loại.
- Near-duplicate: băm 77 feature đã làm tròn (4 chữ số thập phân) kết hợp endpoint. Báo cáo tỷ lệ, không loại thêm nếu chưa có quyết định trước.
- **Ngưỡng bác bỏ:** nếu Δ(ResACAG − DecisionTree) sau khi loại exact không còn dương trong khoảng tin cậy 95%, H1 bị bác bỏ ở mức exact-duplicate.

### E3: nhãn và lớp hiếm

- Mục tiêu: báo cáo độ nhạy theo lớp, không thay đổi nhãn.
- Thiết lập: lấy kết quả từ E6. Báo cáo macro-F1 có và không có Infiltration, và recall/PR-AUC của Web Attack và Infiltration.
- Không loại nhãn nào. Không gán lại nhãn.
- **Không có ngưỡng bác bỏ.** Đây là báo cáo mô tả.

### E4: nhiều capture hoặc nhiều site

- Ngoài phạm vi vòng này. Dataset chỉ có một capture (CSE-CIC-IDS2018), không có site độc lập.
- Ghi là giới hạn trong báo cáo, không suy diễn.

### E5: hợp đồng triển khai (node2vec)

- Mục tiêu: tách hai chế độ node2vec để không gộp.
- Thiết lập: ResACAG trên S-A, seed 42–46, với hai chế độ riêng: (a) transductive, (b) train-only. Báo cáo trong hai bảng khác nhau.
- Ghi số node test chưa thấy trong train (dự kiến khoảng 88%, theo repo, cần tính lại) và tỷ lệ node nhận embedding 0.
- **Không có ngưỡng bác bỏ.** Quy tắc báo cáo: không gộp hai chế độ.

### E6: so sánh chính

- Mục tiêu: trả lời câu hỏi chính trên S-A.
- Thiết lập: ResACAG (transductive), DecisionTree, RandomForest, ExtraTrees, HGB, CatBoost, TabM. S-A, seed 42–46. Mỗi mô hình một lượt cho mỗi seed.
- Đo: macro-F1, weighted-F1, log loss, ECE, recall theo lớp. Cặp so sánh: ResACAG với từng baseline, và ResACAG với baseline tốt nhất theo validation.
- **Ngưỡng bác bỏ:** nếu ResACAG không vượt baseline bảng tốt nhất về macro-F1 trong khoảng tin cậy 95% (sau Holm), claim "GNN vượt baseline bảng" bị bác bỏ trên dataset này.

## 8. Quy tắc

1. Chọn mô hình và siêu tham số trên validation. Test chỉ đánh giá một lần cho mỗi lượt.
2. Không đổi protocol sau khi đã thấy số liệu test. Đổi thì viết AMENDMENT và nêu rõ đã thấy gì.
3. Mỗi lượt ghi: lệnh, seed, commit hash, `environment.json`, và đường dẫn kết quả trong `lab/6/<mã>/`.
4. Output luôn vào thư mục mới. Không ghi đè `lab/1`–`lab/5`.
5. Báo kết quả âm đầy đủ, kể cả khi không có lợi cho ResACAG.
6. Không dùng từ "vượt SOTA", "tái lập chính xác", "novelty thuật toán", hay "IDS vận hành" trong báo cáo.

## 9. Thứ tự chạy

1. E6 và E0 trước (chi phí thấp, quyết định hướng).
2. E1, E2, E5.
3. E3 và E4 là báo cáo mô tả, làm cùng lúc với viết báo cáo.

## 10. Claim được phép và không được phép

**Được phép (nếu E1, E2, E6 xác nhận):** mô tả phép đo có đối chứng trên `cse_cic`; báo cáo gain có điều kiện theo dataset, split và seed; báo cáo kết quả âm.

**Không được phép:** nói ResACAG vượt SOTA; nói tái lập bài báo; nói novelty cấp thuật toán; nói IDS vận hành; coi 5 seed trên cùng test set là 5 dataset độc lập; báo cáo residual hoặc gain như cải thiện mọi metric; dùng số lab 5 study (bản 244.427) lẫn với lab 6 (bản 244.776).

## 11. Việc chưa rõ, cần quyết định trước khi chạy

- Lưới tuning cho baseline bảng: để mặc định thư viện hay khóa lưới nhỏ? Protocol mặc định là **mặc định thư viện, không tuning**, để tránh tranh cãi về ngân sách tuning.
- Số lượng resample bootstrap và cách xử lý cụm nguồn khi một cụm rất lớn (ví dụ host DDoS). Mặc định: 2.000 lần, cụm theo IP nguồn.
