# CSE-CIC-IDS2018 với topology thật (`cse_cic`)

Bộ dữ liệu CSE dùng từ vòng 5 của bản dựng lại ResACAG. Nó gồm các dòng của subset mà E-ResGAT công bố, có số dòng và tỷ lệ lớp khớp Table 1 của ResACAG (việc ResACAG dùng đúng subset này là suy luận), với:

- **77 feature CICFlowMeter-V3 thô**, lấy nguyên giá trị trong CSV gốc của CIC (không chuẩn hóa sẵn).
- **IP/port thật** của từng flow, khôi phục từ **PCAP gốc** của CSE-CIC-IDS2018.

Vòng 1–4 dùng dataset key `cse_reference` (thư mục dữ liệu `reference_cse`): mảng z-score của E-ResGAT với đồ thị hai phía **ngẫu nhiên** (`adj_random.npy`). Đồ thị đó không mang thông tin nhãn: edge homophily 0,517, bằng mức ngẫu nhiên. Từ vòng 5, CSE dùng bộ `cse_cic` này. Loader vẫn hỗ trợ `cse_reference` để chạy lại các vòng cũ.

Để chạy thí nghiệm, có thể tải bản Parquet đã dựng theo [hướng dẫn tải dữ liệu](../../../../data/README.md). Để kiểm tra toàn bộ quá trình khôi phục IP:cổng, làm theo mục **Cách dựng** rồi đặt ba tệp `cse_cic.parquet`, `feature_names.json` và `provenance.json` vào `data/cse_cic/`.

## Nội dung

| File | Ý nghĩa |
|---|---|
| `cse_cic.parquet` (không có trong repo) | 244.776 flow. Gồm 77 feature, cột `label` và 4 cột endpoint (`Src IP`, `Src Port`, `Dst IP`, `Dst Port`); các cột audit gồm `eres_row`, `day`, `aws_file_row`, `Timestamp`, `aws_label`, `join_tier`, `join_candidates`, `join_ip_pairs`, `pcap_member`, `start_us`. SHA-256 `ea1736a132a5ffb5d1b93eae48b5ba4c17433655dbc621236dbf9b3a9fcee8c7`. |
| `feature_names.json` | Thứ tự 77 feature đưa vào mô hình. Không gồm IP, port, timestamp hay nhãn. |
| `provenance.json` | Hash, số dòng theo lớp, độ phủ, nguồn và kết quả kiểm chứng chéo. `verify_data.py --datasets cse_cic` so file parquet với hash ghim trong `build/expected_dataset.json`, và so `feature_names.json`, `provenance.json` trong thư mục dữ liệu với bản trong thư mục này. |
| `graph_stats.json` | Thống kê đồ thị: IP:port thật, chỉ IP, và `adj_random` cũ. |
| `build/` | Script dựng, dấu kiểm input và hash đầu ra mong đợi. |

Nhãn: 0 Benign, 1 Brute-force, 2 DoS, 3 DDoS, 4 Web Attack, 5 Botnet, 6 Infiltration. Đây là nhãn `label_mul` của E-ResGAT, tức nhãn bài báo.

| Lớp | Subset gốc | Có endpoint | Độ phủ |
|---|---:|---:|---:|
| Benign | 177.192 | 171.301 | 96,7% |
| Brute-force | 18.256 | 18.256 | 100% |
| DoS | 2.534 | 2.534 | 100% |
| DDoS | 32.730 | 32.730 | 100% |
| Web Attack | 928 | 903 | 97,3% |
| Botnet | 13.600 | 13.538 | 99,5% |
| Infiltration | 5.688 | 5.514 | 96,9% |
| **Tổng** | **250.928** | **244.776** | **97,5%** |

## Dùng trong repo

- Key dataset là `cse_cic`. Loader trong `clean_code/code/reproduce/resacag/data.py` đọc thư mục con `cse_cic/` của đường dẫn truyền qua `--data-root`.
- Endpoint chỉ dùng để dựng đồ thị, không đưa vào feature. Node nguồn là `Src IP:Src Port`, node đích là `Dst IP:Dst Port`, hai tập tách riêng nên đồ thị là hai phía.
- Lượt CSE-CIC trong báo cáo học node2vec trên **cấu trúc đồ thị của cả ba tập, dùng luồng không nhãn**.
- Từ thư mục chứa `reproduce_report.py`, chạy `python reproduce_report.py cse-cic --data-root ./data`. Xem [hướng dẫn tái hiện](../../../../README.md) để đối chiếu với kết quả đã lưu.

## Cách dựng

### Ý tưởng

1. E-ResGAT lấy mẫu từ 7 file CSV ngày, nhưng đã bỏ Timestamp và Dst Port rồi z-score 77 feature. Dòng 150000–150927 là toàn bộ dòng Web của 02-22/23 theo đúng thứ tự file, nên giải được (μ, σ) cho 73/77 cột; 4 cột không đổi giá trị trên các dòng này không được đảo và không dùng để ghép. Đảo z-score rồi ghép từng dòng về dòng tương ứng trong CSV AWS: 250.924/250.928 dòng khớp.
2. CSV AWS không có IP. Mỗi CSV là output CICFlowMeter-V3 của các host chụp gói, ghép nối tiếp nhau. Vì vậy chỉ cần tải đúng các host đó trong `pcap.zip` của từng ngày, tách lại flow theo luật CICFlowMeter-V3, rồi ghép với dòng AWS bằng các khóa sau:
   - Dst Port, Protocol.
   - Giây UTC bắt đầu flow. Timestamp AWS là đồng hồ 12 giờ không có AM/PM, giờ AST: +12 h nếu giờ từ 1–7, rồi +4 h để ra UTC.
   - Số gói và tổng byte payload mỗi chiều.
   - Flow Duration, khớp chính xác hoặc lệch ≤ 20 µs.
3. Trước hết, mỗi flow PCAP chỉ gán cho một dòng: ưu tiên thời lượng khớp nhất, gặp ứng viên trùng nhau thì bốc theo seed cố định. Dòng nào vẫn chưa có flow thì được dùng lại một flow đã gán; trường hợp này xảy ra khi file AWS liệt kê một flow hai lần (tier `pcap_reused_flow`). Các dòng E-ResGAT trỏ cùng một dòng AWS dùng chung flow đó. Chỉ nhận flow từ host có block dày đặc trong CSV bao quanh dòng đó, để loại các trùng khớp ngẫu nhiên từ host không thuộc CSV. Dòng không khớp PCAP được bổ sung từ khớp exact với bản trích xuất lại của DistriNet (IEEE CNS 2022).

### Yêu cầu

- Python 3.10 với `numpy`, `pandas`, `pyarrow`, `scikit-learn` và `requests` theo [requirements.txt](../../../../requirements.txt). Muốn parquet ra đúng SHA-256 đã ghim thì cần đúng pandas 2.3.3 và pyarrow 23.0.1.
- Mạng: khoảng 2,3 GB CSV AWS, khoảng 49 GB PCAP nén (đọc theo luồng, không lưu lại; riêng 02-14 và 03-02 đọc tới giới hạn 18 GB mỗi ngày) và khoảng 7 GB DistriNet (đọc theo luồng).
- Đĩa trống khoảng 5 GB. Thời gian khoảng 1,5–2 giờ ở tốc độ 10–20 MB/s.
- Không chạy notebook, không load pickle. File tải về chỉ được đọc như dữ liệu.

### Lệnh

Chạy các lệnh PowerShell dưới đây từ thư mục chứa `reproduce_report.py`. Các tệp trung gian được ghi vào `cse_build/` cạnh thư mục hiện tại.

```powershell
$Project = (Get-Location).Path
$W = Join-Path (Split-Path -Parent $Project) 'cse_build'
Push-Location (Join-Path $Project 'clean_code/code/reproduce/dataset_cse_cic/build')
# 1. Input E-ResGAT đã ghim commit 3e0ceb8f, kiểm SHA-256 (eresgat_inputs.json)
python -I fetch_eresgat.py $W\eresgat

# 2. Bảy CSV "Processed Traffic Data for ML Algorithms" (S3 công khai)
$S3 = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms"
foreach ($d in "Wednesday-14-02-2018","Thursday-15-02-2018","Wednesday-21-02-2018","Thursday-22-02-2018","Friday-23-02-2018","Wednesday-28-02-2018","Friday-02-03-2018") {
  python -I fetch_ranges.py "$S3/${d}_TrafficForML_CICFlowMeter.csv" "$W\aws\${d}_TrafficForML_CICFlowMeter.csv" 16 }

# 3. Ghép dòng E-ResGAT về dòng AWS
python -I recover_rows.py $W\aws $W\eresgat $W\row_map

# 4. DistriNet: nguồn dự phòng và kiểm chứng chéo. distrinet_index.json là chỉ mục zip; tạo lại bằng:
New-Item -ItemType Directory -Force "$W" | Out-Null
python -I zip_remote.py https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip "$W\distrinet_index.json"
$dn = @{ "Wednesday-14-02-2018.csv"="02-14"; "Thursday-15-02-2018.csv"="02-15"; "Wednesday-21-02-2018.csv"="02-21";
         "Thursday-22-02-2018.csv"="02-22"; "Friday-23-02-2018.csv"="02-23"; "Wednesday-28-02-2018.csv"="02-28"; "Friday-02-03-2018.csv"="03-02" }
foreach ($m in $dn.Keys) { python -I fetch_distrinet.py $m $dn[$m] "$W\distrinet_index.json" $W\row_map\matched_raw.parquet $W\distrinet }
python -I join_endpoints.py $W\row_map\matched_raw.parquet $W\distrinet $W\join

# 5. Tách flow từ PCAP. pcap_index\ là chỉ mục zip từng ngày; tạo lại một ngày bằng:
#    python -I pcap_index.py "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Original%20Network%20Traffic%20and%20Log%20data/<Ngày>/pcap.zip" "$I\cd_<Ngày>.json"
$A = "$W\aws"; $M = "$W\row_map\row_map.csv"; $P = "$W\pcap"; $I = "$W\pcap_index"
New-Item -ItemType Directory -Force "$I" | Out-Null
$PCAP = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Original%20Network%20Traffic%20and%20Log%20data"
foreach ($d in "Wednesday-14-02-2018","Thursday-15-02-2018","Wednesday-21-02-2018","Thursday-22-02-2018","Friday-23-02-2018","Wednesday-28-02-2018","Friday-02-03-2018") {
  python -I pcap_index.py "$PCAP/$d/pcap.zip" "$I\cd_$d.json" }
python -I pcap_flows.py 02-14 Wednesday-14-02-2018 "$I\cd_Wednesday-14-02-2018.json" $A\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv $M $P UCAP172.31.69.25 --more
python -I pcap_flows.py 02-15 Thursday-15-02-2018 "$I\cd_Thursday-15-02-2018.json" $A\Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv $M $P UCAP172.31.69.25 --more
python -I pcap_flows.py 02-21 Wednesday-21-02-2018 "$I\cd_Wednesday-21-02-2018.json" $A\Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv $M $P "UCAP172.31.69.28 part 1"
python -I pcap_flows.py 02-21 Wednesday-21-02-2018 "$I\cd_Wednesday-21-02-2018.json" $A\Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv $M $P "UCAP172.31.69.28 part 2"
python -I pcap_flows.py 02-22 Thursday-22-02-2018 "$I\cd_Thursday-22-02-2018.json" $A\Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv $M $P UCAP172.31.69.28
python -I pcap_flows.py 02-23 Friday-23-02-2018 "$I\cd_Friday-23-02-2018.json" $A\Friday-23-02-2018_TrafficForML_CICFlowMeter.csv $M $P UCAP172.31.69.28
python -I pcap_flows.py 02-28 Wednesday-28-02-2018 "$I\cd_Wednesday-28-02-2018.json" $A\Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv $M $P
python -I pcap_flows.py 03-02 Friday-02-03-2018 "$I\cd_Friday-02-03-2018.json" $A\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv $M $P

# 6. Gán endpoint, kiểm chứng chéo, dựng dataset
python -I assign_endpoints.py $W\row_map\matched_raw.parquet $P $W\join\joined.parquet $W\endpoints
python -I crosscheck.py $W\endpoints $W\join
python -I build_dataset.py $W\endpoints\joined.parquet $W\eresgat $W\row_map $W\endpoints $W\dataset
python -I graph_stats.py $W\dataset\cse_cic.parquet $W\eresgat $W\dataset\graph_stats.json
# build_dataset.py in ra MATCH khi file dựng được trùng hash với file của vòng 5 (build\expected_dataset.json).
# Sau khi thấy MATCH, chép Parquet và hai tệp metadata đã ghim vào thư mục dữ liệu
$Dataset = Join-Path $Project 'data/cse_cic'
New-Item -ItemType Directory -Force -Path $Dataset | Out-Null
Copy-Item -LiteralPath "$W\dataset\cse_cic.parquet", '..\feature_names.json', '..\provenance.json' -Destination $Dataset
Pop-Location
python clean_code/code/reproduce/verify_data.py --data-root ./data --datasets cse_cic
```

`build_dataset.py` so SHA-256 của file vừa dựng với `build/expected_dataset.json`. Dựng từ cùng input với cùng phiên bản pandas/pyarrow phải in ra `MATCH`. Lệnh kiểm tra cuối chỉ cần dữ liệu CSE-CIC.

### Chọn host PCAP theo ngày

`pcap_flows.py` tải từng member của `pcap.zip` qua HTTP Range. Nó đọc qua mirror Hugging Face `bencorn/CICIDS2018`, giống hệt từng byte với S3, và tự lùi về S3 nếu mirror lỗi. Mỗi member được giải nén theo luồng, kiểm CRC32, tách flow, rồi xóa ngay. Log ghi `match_rate` của từng host.

| Ngày | Host cần đọc |
|---|---|
| 02-14, 02-15 | Server nạn nhân `UCAP172.31.69.25`, rồi các host capDESKTOP theo thứ tự zip (02-15 thêm capEC2AMAZ). Dừng khi các block đã xử lý phủ hết dòng mục tiêu, chỉ còn tối đa max(5, số mục tiêu/5000) dòng chưa khớp phía sau, hoặc khi đọc đủ 18 GB (02-14 dừng ở giới hạn này). |
| 02-21 | Chỉ `UCAP172.31.69.28`: part 1 (Benign, LOIC-UDP) và part 2 (HOIC). |
| 02-22, 02-23 | Chỉ `UCAP172.31.69.28`, vì subset chỉ lấy dòng Web và chúng nằm ở block đầu file. |
| 02-28 | Toàn bộ host dải 172.31.69.x. |
| 03-02 | Các host 172.31.69.x (nạn nhân bot) trước, rồi capDESKTOP; dừng ở giới hạn 18 GB. |

Các script tạo log trong thư mục làm việc khi chạy.

### Luật CICFlowMeter-V3 được tái hiện

- Flow ID chuẩn tắc, so byte IP như kiểu byte có dấu của Java. Chiều xuôi là chiều của gói đầu tiên.
- FIN đầu tiên ở bất kỳ chiều nào kết thúc flow.
- Timeout 120 s tính từ lúc flow bắt đầu. Flow kết thúc vì timeout hoặc hết file chỉ được ghi nếu có nhiều hơn 1 gói.
- Payload tính theo IP total length.
- Một số member là nhiều file pcap nối liền, nên bộ đọc tự đồng bộ lại bản ghi.
- Lượt ghép "swap" (cùng tổng số gói, cùng tổng byte mỗi chiều, thời lượng chính xác tới µs) bắt các flow mà V3 ghi một gói không payload theo chiều khác. Phần lớn là flow tiếp tục sau timeout.

## Kiểm chứng

- **Cặp IP:** trùng **99,96%** với DistriNet trên 98.119 dòng mà cả hai nguồn cùng khớp; bộ 4 (gồm Src Port) trùng 92,4%. Phần chồng lấn chủ yếu là Benign và FTP; DDoS-HOIC, Bot và SSH không có kiểm chứng chéo độc lập.
- **Edge homophily:** 0,952 trên đồ thị IP:port thật, so với 0,517 trên `adj_random`.
- **Tính tất định:** file dùng cho vòng 5 chính là output của các script trong `build/` chạy đúng các lệnh trên (lần dựng ngày 25/10/2026). Dựng lại cho ra `MATCH` với `expected_dataset.json` (cùng pandas/pyarrow).

## Giới hạn

- **Dòng bị loại:** 6.152 dòng của subset (2,5%) bị loại: 4 dòng không ghép được về CSV và 6.148 dòng không có endpoint. Trong đó 2.884 dòng có Protocol 0 (CICFlowMeter-V3 ghi ICMP và khung không phải IPv4 thành Protocol 0, còn bộ tách chỉ dựng flow TCP/UDP), phần còn lại chủ yếu là Benign.
- **Dòng gốc chỉ xác định tới mức tương đương feature:** E-ResGAT đã bỏ Timestamp và Dst Port, nên khoảng 36% số dòng (90.970/250.924 dòng ghép được, theo `build/logs/row_map_summary.json`) có nhiều dòng CSV khớp ngang nhau, khác Dst Port hoặc Timestamp. Endpoint gán cho các dòng này là của một flow thật có cùng feature, cùng nhãn và cùng ngày, nhưng có thể không phải đúng flow E-ResGAT đã lấy.
- **Ghép chưa tối ưu:** các nhóm dưới đây được đếm riêng và có thể chồng nhau.
  - 540 dòng chọn ứng viên có nhãn AWS khác, vì KD-tree chỉ xét 32 ứng viên.
  - 108 dòng lệch khoảng 1–120 µs ở các cột thời gian.
  - 1.437 dòng (1.421 dòng ở 02-28) dồn vào chỉ 713 dòng CSV, vì bước gom nhóm dùng vector float64 chính xác còn bước khớp dùng dung sai float32. Ít nhất 724 dòng trong số này bị ghép sai và dùng trùng flow của dòng khác.
  - Ảnh hưởng của các dòng này lên topology **chưa đo**. Bản v2 nên mở rộng xét toàn bộ nhóm khớp ở mức float32.
- **Nhiễu nhãn có sẵn của CSE-CIC-IDS2018:**
  - Infiltration gán theo dải mạng và khung giờ, nên F1 gần 0 ở mọi mô hình.
  - 17.136 dòng Benign ngày 02-21 thực chất là chiều server → attacker của kết nối HOIC.
  - 131 dòng Web ngày 02-23 thực chất là UDP/500, DHCP hoặc SSH.
- **Danh tính endpoint mang nhiều tín hiệu:** chỉ tra nhãn đa số theo `Dst IP:Dst Port` của train đã đạt F1 96,04 / 71,25 (weighted / macro; trung bình 3 seed, đo trên bản dựng đầu `a2469a4f` trong `lab/5/study`). Split phân tầng ngẫu nhiên vì vậy không chứng minh khả năng tổng quát sang máy hay thời điểm mới.
