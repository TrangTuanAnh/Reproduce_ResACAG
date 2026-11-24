# Phương pháp và giới hạn tái hiện

Đây là **triển khai độc lập có ghi rõ giả định**, không phải source code chính thức của ResACAG. Không tuyên bố các chỉ số sẽ bằng bài báo.

## Dữ liệu và bằng chứng

- Bài gốc: Zhang et al., *ResACAG: A graph neural network based intrusion detection*, Computers and Electrical Engineering 122 (2025) 109956, DOI [10.1016/j.compeleceng.2024.109956](https://doi.org/10.1016/j.compeleceng.2024.109956).
- CIC-Darknet: [CIC/UNB](https://www.unb.ca/cic/datasets/darknet2020.html); CSV lấy từ [kho nghiên cứu public](https://github.com/mateuscmarim/Darknet-traffic-classification), commit và SHA-256 lưu trong `dataset/darknet_original/provenance.json`. 141.530 dòng, 77 feature, 9 lớp. NonVPN/Non-Tor → Benign; Tor/VPN → subtype. Phân loại traffic VPN/Tor không đồng nghĩa mọi flow đó là hành vi độc hại ngoài thực tế.
- CSE-CIC: [kho tác giả E-ResGAT](https://github.com/George730/E-ResGAT), reference [24] của bài. Numeric `.npy` 250.928 dòng, 77 feature, 7 lớp; Git LFS SHA-256 được kiểm chứng. `adj_random.npy` là đồ thị ngẫu nhiên đã công bố, **không phải IP/port quan sát**. Notebook chuẩn bị dữ liệu còn gán IP ngẫu nhiên; không dùng notebook đó để bịa khôi phục endpoint. Artifacts này khớp số lượng và phân bố Table 1 sau làm tròn nhưng chưa chứng minh là artifacts chính xác tác giả ResACAG sử dụng.
- CSE-CIC từ vòng 5 (`cse_cic`): đúng các dòng của subset trên, với feature CICFlowMeter-V3 thô và IP/port thật khôi phục từ PCAP gốc của CSE-CIC-IDS2018: 244.776/250.928 dòng có endpoint, edge homophily 0,952 (so với 0,517 của `adj_random`). Cách dựng và giới hạn: `code/reproduce/dataset_cse_cic/README.md`; kết quả: `lab/5`.
- Phiên bản mã: vòng 1–4 chạy với package `resacag` hash `5295c99a…`, tức tag git `rounds-1-4` (commit `b1d5a7d`). Vòng 5 thêm loader `cse_cic` và cờ `--transductive-node2vec`, nên hash của package đổi; đường huấn luyện của Darknet và `cse_reference` giữ nguyên. Muốn chạy lại hoặc kiểm vòng 1–4 đúng mã thì checkout tag đó. Ở HEAD, `extend_legacy.py` gặp các run vòng 2 sẽ dừng vì khác code/config thay vì bỏ qua (SKIP).
- Bộ private 8.267 mẫu, 42 features, 3 lớp: không có trong workspace và không tìm thấy bản phát hành công khai. Bài ghi data available on request. Mọi hàng kết quả private phải để `unavailable`, tuyệt đối không điền số giả.
- Không chạy code/notebook tải từ mạng; chỉ đọc, kiểm tra numeric arrays `allow_pickle=False`. Không load `.dict`, `.pkl` hoặc numpy object arrays không tin cậy.

Darknet khớp số lượng, phân bố gần khớp Table 1: VOIP có 1.465/141.530 mẫu = 1,0351%, làm tròn thành 1,04%, trong khi bảng gốc ghi 1,03%. Các tỷ lệ còn lại khớp ở 2 chữ số thập phân. Không tự điều chỉnh nhãn hoặc bỏ dòng để ép khớp tỷ lệ đã in.

## Ánh xạ công thức sang mã

| Bài gốc | Triển khai | Quyết định bổ sung |
|---|---|---|
| §3.1, IP+port là node, flow là edge | `data.endpoint_ids`, `graph.FlowGraph` | Tách namespace source/destination để đảm bảo bipartite; giữ parallel flows. CSE reference dùng adjacency đã công bố. |
| Algorithm 1, K=2 | `FlowGraph.sample` | Hai block lấy mẫu từ endpoint batch, không hoàn lại, không nhân bản hàng xóm khi thiếu. |
| Eq. 2, S=mean(d_u+d_v−2) | `FlowGraph.fanouts` | Ceil, tối thiểu 1, cap [8,4]; tính trên graph train và cố định cho val/test. Paper không cho cap hoặc fanout hop 2. |
| Eq. 3 mean aggregation | `masked_mean` | Mask padding; degree 0 cho vector 0. |
| Eq. 4–7 Q/K/V, dropout attention | `GlobalAttention` | Một head, attention trên các node được tính ở mỗi sampled block; không phải mọi node toàn graph. Paper không nêu trục token/head count. |
| node2vec x_v | `graph.node2vec` | SGNS, p=q=1, 2 walks/node, length 10, window 2, 3 negatives, 2 epochs; uniform negatives; dim=32, L2 normalization. Chỉ train topology ở vòng 1–4 (Darknet và `cse_reference`). Với `cse_cic` (vòng 5), mặc định walk trên cấu trúc mọi split, không dùng nhãn (transductive); truyền `--transductive-node2vec` không kèm giá trị để quay về chỉ train. |
| Eq. 8–9 gated fusion | `ResACAGLayer.gate` | z=sigmoid(W a), g=z*a+(1−z)*x. |
| Eq. 10 context broadcasting | `context_broadcast` | CB(x)=0.5*(x+mean(x)), không tham số. Context chỉ trong sampled block hiện tại. |
| Eq. 11 residual I | `ResACAGLayer.forward` | Concatenate CB output và attention, **không cộng**. |
| Eq. 12 hidden update | `ResACAGLayer.update` | Linear + ReLU; giữ graph autograd xuyên suốt. |
| Algorithm 2 line 12 residual E | `EdgeClassifier.forward` | Concatenate hai endpoint hidden và raw normalized flow features. |

### Điểm chưa xác định trong bài

Algorithm 2 sử dụng `e^(k−1)` nhưng không định nghĩa cập nhật edge state. Mặc định `--edge-update neighbor` làm message lớp 2 từ `[raw edge, previous hidden of opposite endpoint]`, sau linear projection; giúp thông tin thực sự truyền 2-hop. Đây là **giả định kiến trúc**, không phải công thức được tác giả nêu. `--edge-update static` giữ raw features ở mỗi lớp, là cách diễn giải bảo thủ khác. Các baseline SAGE dùng neighbor propagation. Hai cách cần được nghiên cứu riêng nếu có mã gốc để đối chiếu.

Node2vec SGNS trong bản triển khai lấy trung bình riêng cho positive loss và negative loss rồi cộng hai hạng; ba negatives làm giảm phương sai của hạng negative, không nhân trọng số loss lên ba lần. Đây cũng là một lựa chọn optimization bổ sung vì bài không nêu cấu hình node2vec.

Với attention và context trong sampled block, kết quả của một cạnh có thể phụ thuộc batch và thứ tự đánh giá. Batch size/order/sampling seed được cố định giữa các mô hình; đây là giới hạn thực của cách diễn giải, không coi mô hình là predictor bất biến theo batch.

## Giao thức thực nghiệm

1. Feature 77 chiều; bỏ label, IP/port, timestamp, Flow ID khỏi predictor input. Endpoint chỉ làm topology.
2. Chia **stratified** train/test/validation 50/30/20, seed=42. Bài mô tả chia “in sequence” nhưng không phát hành chỉ số split. Trong CSV Darknet gốc, train/test tuần tự chỉ có Benign; trong CSE, train thiếu 3 lớp. `data_audit.json` ghi đủ số đếm cho cả hai cách. CLI hỗ trợ sequential và chặn missing classes trừ khi cho phép rõ ràng.
3. Các split có edge ID rời nhau và graph rời nhau. Không dùng nhãn test/val trong fit. Với Darknet và `cse_reference` (vòng 1–4), node2vec chỉ học trên train và node chưa thấy nhận zero; đây là giao thức độc lập, khắt khe hơn transductive full-graph mà bài không làm rõ. Với `cse_cic` (vòng 5), node2vec mặc định học trên cấu trúc mọi split, không dùng nhãn, vì khoảng 88% node test là IP:cổng tạm chưa thấy trong train; chỉ embedding không giám sát này thấy cấu trúc val/test, còn graph lấy mẫu của GNN vẫn tách theo split.
4. Thay ±inf/NaN bằng median train; MinMax train-only về [0,1], clip ngoài khoảng; constant features→0. Với `cse_reference`, upstream artifacts đã standardized/imputed trên toàn bộ dữ liệu, nên không thể bảo đảm không có leakage upstream; báo cáo nêu rõ. `cse_cic` dùng giá trị CSV thô, inf/NaN do bước thay median train-only ở trên xử lý. Không thực hiện feature selection hoặc class balancing.
5. Adam: lr=0.01 Darknet / 0.007 CSE; dropout=0.5, batch=500 theo Table 2; hidden=32, tối đa 12 epochs, patience=3, gradient norm clip=5 là lựa chọn bổ sung. Chọn checkpoint theo **validation macro-F1**, tie-break validation loss. Test chỉ dùng sau khi chọn checkpoint.
6. Tất cả batch kể cả batch cuối được đánh giá. Macro-F1 tính trên danh sách lớp cố định, kể cả lớp không được dự đoán. F1 weighted, macro, accuracy, confusion matrix và từng lớp đều được lưu.
7. Một seed cho ma trận đầy đủ; không suy luận ý nghĩa thống kê từ một lượt chạy. Có thể truyền nhiều `--seeds` để tính biến thiên; `report.py` tổng hợp mean/std khi có nhiều seed.
8. Split phân tầng có thể đặt các flow trùng/lặp gần nhau vào nhiều split; đây không phải kiểm chứng khái quát theo thời gian/host. Không tuyên bố kết quả production hoặc khả năng phát hiện zero-day.

Cụm “in sequence” của bài có thể chỉ thứ tự liệt kê train/test/validation, không nhất thiết có nghĩa tác giả chia các hàng CSV liên tiếp. Diagnostic sequential chỉ chứng minh việc chia liên tiếp **các file đang có** làm mất lớp; không chứng minh tác giả đã làm như vậy. Split stratified ở đây là lựa chọn bổ sung chưa xác minh với split gốc.

## Ma trận thí nghiệm

- Table 3: ResACAG, E-GraphSAGE, E-GraphSAGE M, GAT, E-ResGAT, Graph-Transformer.
- Table 5: A không residual, B chỉ I, C chỉ E, D=ResACAG. D dùng lại cùng kết quả ResACAG, không huấn luyện lại tạo sai lệch.
- Table 6: S1 fixed [4,2], S2 ceil(mean node degree) cap [8,4], S3=ResACAG adaptive. Paper không nêu fanout S1; [4,2] là giả định. Cap S2 để giới hạn tài nguyên, nên không thể tái hiện memory của S2 không cap.
- Table 4: mean train-batch wall time (GPU nếu có), sampled process RSS, GPU peak allocation; tùy chọn CPU operator profiling forward/backward một batch. Không so sánh trực tiếp % memory toàn hệ thống trên máy đang chạy app khác với bảng bài gốc. CPU profile không gồm optimizer, báo cáo khác biệt.
- Thời gian train/validation trong metrics không gồm chuẩn bị dữ liệu và pretraining node2vec. Parameter count chỉ đếm classifier/GNN, không đếm embedding node2vec đã đóng băng. Đây không phải benchmark end-to-end hoàn toàn cách ly các tiến trình khác.
- Figure 6: t-SNE raw vs learned edge embedding trên tối đa 2.000 test samples, cùng indices/seed; labels chỉ dùng tô màu.
- Figure 7: offline replacement noise 0/10/20/30%, uniform [0,1] sau normalization, masks nested và giống nhau giữa các models. Giữ topology; sửa cả feature của flow khi dùng làm hàng xóm. Không train lại, không tạo traffic tấn công. Đây là random feature corruption, không phải bằng chứng chống adaptive adversarial attacks.
- Thêm MLP làm kiểm tra tác dụng của topology so với chỉ dùng features.

## Baseline không phải bản sao nguyên xi

SAGE/M: mean edge-aware aggregation và endpoint concat, M thêm raw edge residual. GAT/E-ResGAT: additive edge-aware neighborhood attention, E-ResGAT thêm edge residual. Transformer: tokens own/aggregated-neighborhood/structural, 4-head transformer và attention readout, lặp 2 lớp. Baseline Transformer **không triển khai đầy đủ model landslide mapping của reference [26]**. Các baseline chia sẻ hidden/split/training; kết quả phải gọi là *independent baseline implementations*.

Không copy các lỗi quan sát trong reference E-ResGAT: chuyển encoder embedding sang numpy và detach gradient; nối node ID vào edge ID; bỏ batch cuối. Kiểm thử gradient và coverage bảo vệ các lỗi này.

## Tái lập và artifact

`environment.json`: Python/library/hardware/arguments/code hash. Mỗi seed có `data_audit.json`, scaler, split indices, node2vec. Mỗi variant có config, history, best epoch, tensor-only checkpoint, clean/noise predictions, metrics. `metrics.json` chỉ được viết sau khi hoàn tất toàn bộ đánh giá; chạy lại bỏ qua chỉ khi signature config/code khớp. Nguồn thay đổi phải chọn output mới. CUDA attention có thể nondeterministic dù đã seed; log có cảnh báo và báo cáo không hứa bitwise identity.
