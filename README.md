# Tái lập ResACAG

Repo cung cấp bản triển khai độc lập của ResACAG, mã tái lập thí nghiệm trên CIC-Darknet2020 và CSE-CIC-IDS2018, các đối chứng và kết quả tổng hợp của báo cáo. Bản CSE-CIC dùng IP:cổng đã khôi phục để dựng đồ thị luồng. [Bài báo ResACAG](https://doi.org/10.1016/j.compeleceng.2024.109956)

- [Tải dữ liệu từ Google Drive](https://drive.google.com/drive/folders/1MHJjQueTZs80G7Q2eimJFLOy4kvaodgN?usp=sharing)
- [Đọc mô tả dữ liệu và cách khôi phục IP:cổng](data/README.md)
- [Tra cứu kết quả và hình của báo cáo](artifacts/README.md)

## Bố cục

| Thành phần | Nội dung |
| --- | --- |
| `reproduce_report.py` | Lệnh kiểm tra gói, tái hiện hai bộ dữ liệu, chạy đối chứng, tính EDA và vẽ hình |
| `clean_code/versions/darknet_lab2/` | Mã nguồn đúng phiên bản của lượt CIC-DarkNet có giới hạn 30 epoch |
| `clean_code/code/reproduce/` | Mô hình ResACAG, kiểm tra dữ liệu và mã khôi phục kết nối CSE-CIC |
| `clean_code/lab/6/code/run_cse_cic_controls.py` | Thực nghiệm chính và các đối chứng có kiểm soát |
| `clean_code/lab/6/code/analyze_cse_cic_controls.py` | Tổng hợp chỉ số, bootstrap theo IP nguồn và hiệu chỉnh Holm |
| `clean_code/lab/7/code/run_source_ip_experiments.py` | Đối chiếu chia ngẫu nhiên và chia theo IP nguồn |
| `clean_code/report_analysis/` | Khảo sát dữ liệu, baseline tra nhãn theo đầu kết nối và mã vẽ hình |
| `data/` | Hướng dẫn tải dữ liệu, checksum của hai tệp ZIP và hình minh họa quy trình dựng đồ thị |
| `artifacts/` | Kết quả tổng hợp, cấu hình, lịch sử huấn luyện, hình của báo cáo và checksum |

## Môi trường

Sử dụng Python 3.10. Chạy các lệnh từ thư mục chứa `reproduce_report.py`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce_report.py verify
```

Có thể dựng lại các hình từ số liệu tổng hợp đã lưu mà không cần tải dataset. Các lệnh ghi kết quả vào `runs/` hoặc thư mục chỉ định qua `--out`.

```powershell
python reproduce_report.py figures
```

## Chuẩn bị dữ liệu

Hai tệp ZIP được chia sẻ riêng với gói mã nguồn. `CIC-Darknet2020.zip` chứa bản CSV đã dùng trong báo cáo. `CSE-CIC-IDS2018-recovered-endpoints.zip` chứa bản Parquet 244.776 luồng đã khôi phục IP:cổng. Mã đọc trực tiếp các cột này để dựng đồ thị hai phía, nên người chạy có thể bỏ qua bước khôi phục từ PCAP. Mỗi ZIP kèm thông tin nguồn và trích dẫn.

Tải hai ZIP vào `data/`. Đối chiếu SHA-256 với [danh sách checksum](data/SHA256SUMS.txt), rồi giải nén cả hai ZIP vào cùng thư mục. [Thư mục tải dữ liệu trên Google Drive](https://drive.google.com/drive/folders/1MHJjQueTZs80G7Q2eimJFLOy4kvaodgN?usp=sharing)

```powershell
Get-FileHash ./data/*.zip -Algorithm SHA256
python -m zipfile -e ./data/CIC-Darknet2020.zip ./data
python -m zipfile -e ./data/CSE-CIC-IDS2018-recovered-endpoints.zip ./data
python clean_code/code/reproduce/verify_data.py --data-root ./data --datasets darknet cse_cic
```

`--data-root` trỏ tới thư mục chứa hai thư mục con sau.

```text
data/
  darknet_original/
    Darknet.csv
    provenance.json
  cse_cic/
    cse_cic.parquet
    feature_names.json
    provenance.json
```

Có thể lấy hoặc dựng lại dữ liệu từ nguồn công khai.

- **CIC-DarkNet.** Chuẩn bị `Darknet.csv` từ [nguồn CIC-DarkNet](https://www.unb.ca/cic/datasets/darknet2020.html) hoặc bản mirror đã ghim trong [darknet_provenance.json](artifacts/inputs/darknet_provenance.json). Chép tệp provenance đã ghim vào cùng thư mục với tên `provenance.json`.
- **CSE-CIC-IDS2018.** Làm theo [hướng dẫn khôi phục IP:cổng](clean_code/code/reproduce/dataset_cse_cic/README.md) để dựng đúng bản Parquet dùng trong báo cáo. Chép hai tệp metadata đi kèm từ [thư mục dựng CSE-CIC](clean_code/code/reproduce/dataset_cse_cic/).

Các lệnh tái hiện kiểm tra hash dữ liệu trước khi chạy. Nếu chỉ chạy một bộ, hãy chuẩn bị các tệp của bộ đó.

## Chạy lại theo phạm vi báo cáo

```powershell
$Data = './data'
python reproduce_report.py darknet --data-root $Data
python reproduce_report.py cse-cic --data-root $Data
python reproduce_report.py eda --data-root $Data
```

Hai lượt chính dùng seed 42. CIC-DarkNet dùng mã lưu của Lab 2, giới hạn 30 epoch, patience 30 và node2vec học trên đồ thị huấn luyện. CSE-CIC dùng lượt E6 của Lab 6 với giới hạn 12 epoch, patience 3 và node2vec học từ cấu trúc không nhãn của cả ba tập. Khi chạy lại, so kết quả với các bản ghi trong [artifacts/reproduction](artifacts/reproduction/). Thiết bị và thư viện có thể tạo khác biệt số học giữa các lần chạy.

```powershell
python reproduce_report.py controls --data-root $Data
python reproduce_report.py analyze-controls
python reproduce_report.py source-ip --data-root $Data
```

`controls` chạy E1 đến E8 của Phần C với các seed 42–46. Để chạy riêng một đối chứng, thêm `--experiment E8` hoặc ID tương ứng. Phép so sánh với E1 cần kết quả E1 được tạo trong cùng thư mục đầu ra.

| ID trong báo cáo | ID trong mã hoặc bản ghi gốc | Nội dung |
| --- | --- | --- |
| E1 | Lab 6 E6 | So sánh chính |
| E2 | Lab 6 E0 | Bỏ dòng ghép đầu kết nối mơ hồ |
| E3 | Lab 6 E2 | Bỏ dòng kiểm tra trùng đặc trưng với tập huấn luyện |
| E4 | Lab 6 E7a | Trọng số lớp |
| E5 | Lab 6 E7b | Biểu diễn nút từ đặc trưng luồng |
| E6 | Lab 6 E8 | MLP chỉ dùng đặc trưng luồng |
| E7 | Lab 6 E9 | Đồ thị hoán vị giữ bậc |
| E8 | Lab 6 E10 | Thêm vector nút vào mô hình cây |
| E9 | Lab 7, RAND và GRP | Chia theo IP nguồn, với đặc trưng luồng và đặc trưng danh tính |

Các CSV `e10_reported_*.csv` ghi các giá trị E8 đã trình bày trong báo cáo. Bản ghi hiện có của E10 là kết quả tổng hợp. Mã chạy lại nằm trong `run_cse_cic_controls.py`.

## Kiểm tra gói

```powershell
cd clean_code
python -m pytest -q code/reproduce/tests lab/7/tests
```

Các kiểm thử dùng dữ liệu tổng hợp nhỏ tạo trong lúc chạy. Gói chỉ phân phối mã, metadata, kết quả tổng hợp và hình do nhóm tạo. Dataset, checkpoint, vector đặc trưng, chỉ số dòng chia tập, dự đoán từng luồng và hình lấy từ bài báo được lưu ngoài gói.
