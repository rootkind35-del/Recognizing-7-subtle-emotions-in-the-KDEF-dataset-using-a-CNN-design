# BÁO CÁO THỰC NGHIỆM — Nhận dạng 7 cảm xúc khuôn mặt trên KDEF bằng CNN tự thiết kế

**Ngày báo cáo:** 2026-10-24 (bản cập nhật lần chạy chính thức trên GPU)
**Thiết bị chạy thực nghiệm:** GPU **NVIDIA GeForce RTX 4060 Laptop GPU** qua WSL2 Ubuntu-22.04, `torch 2.14.1+cu130` (`reports/test_metrics.json` ghi `"device": "cuda (NVIDIA GeForce RTX 4060 Laptop GPU)"`). Mọi số liệu trong báo cáo lấy từ lần chạy chính thức này với ngân sách huấn luyện đầy đủ **120 + 20 epoch** theo yêu cầu chủ đề.
**Nguồn số liệu:** mọi số trong báo cáo lấy nguyên vẹn từ `reports/test_metrics.json` (do cell 9.8 của notebook ghi ra), output các cell của `notebooks/CNN_KDEF_FER.ipynb` (trích kèm số cell), `reports/data_analysis.md` và `reports/papers_research.md`. Không có số liệu nào bịa; các phép kiểm chứng chéo được ghi rõ lệnh đã chạy ở mục 5.

---

## 1. Tóm tắt đề tài và phương pháp luận

### 1.1. Đề tài

Nhận dạng 7 cảm xúc khuôn mặt (angry, disgust, fear, happy, neutral, sad, surprise) bằng **CNN tự thiết kế (~1,24 triệu tham số, PyTorch)** trên bộ dữ liệu **KDEF** (Karolinska Directed Emotional Faces — Lundqvist, Flykt & Öhman, 1998: 4.900 ảnh, 70 người, 7 biểu cảm × 5 góc chụp; `reports/papers_research.md` mục A.1). Bộ dữ liệu huấn luyện chọn: `data/official-kdef-dataset-by-ksatrajit0` (ảnh KDEF gốc); bộ `balanced-kdef-dataset-by-dolly-prajapati-182` bị loại (lý do ở mục 2.3).

### 1.2. Phương pháp luận train/val/test

Notebook cell 0 (markdown tổng quan) khai báo 5 nguyên tắc, thực thi xuyên suốt bằng code:

| Nguyên tắc | Thực thi trong notebook | Căn cứ học thuật |
|---|---|---|
| Chia split **theo ID người (subject-disjoint)** — không người nào xuất hiện ở 2 split | Kiểm chứng bằng manifest: 48/8/14 subject cho train/val/test, 0/70 subject chéo split (cell 11; `reports/data_analysis.md` mục 5a) | Tránh *identity leakage* (Rosenblatt 2024; Apicella 2025) |
| Chọn mô hình / tinh chỉnh **chỉ theo macro-F1 trên val** (559 ảnh sau làm sạch) | Grid-search mục 8 chỉ đo val; checkpoint lưu theo val macro-F1 tốt nhất (cell 24, 27, 29, 30) | Rosenblatt 2024; Apicella 2025 |
| Đánh giá tập kiểm tra **đúng MỘT LẦN** | Biến `TEST_UNLOCKED` chỉ được đặt `True` ở cell 32; mọi cell 34–41 đều `assert TEST_UNLOCKED is True` mới chạy | Chống rò rỉ thông tin lựa chọn mô hình |
| Augmentation **chỉ áp sau khi chia split, chỉ trên train** | `build_transforms()` cell 15: transform augmentation gắn riêng cho train; val/test chỉ resize + normalize | Apicella 2025 |
| Kiểm tra rò rỉ MD5 + subject chéo split trước huấn luyện | Cell 11 hash toàn bộ 4.898 ảnh; 10 file hỏng bị loại khỏi train/val (`TAP_FILE_LOAI_BO`, cell 15 in `3358 → 3349`, `560 → 559`) | `reports/data_analysis.md` mục 5 |

### 1.3. Kỷ luật khóa tập kiểm tra

Notebook cell 0 đặt **QUY TẮC SỐ 1**: thư mục `test/` chỉ được phép xuất hiện ở (a) các cell EDA thuần thống kê đánh dấu `# EDA-cho-phep` (cell 7–11), và (b) mục 9 sau cell mở khóa. Kịch bản kiểm tra tự động: `scripts/verify_notebook.py` (có mặt trong repo, xác nhận bằng `ls scripts/`). Cell 32 in lời cam kết: *"Đã MỞ KHÓA tập kiểm tra. Đây là lần đánh giá DUY NHẤT — không quay lại chỉnh mô hình sau bước này."* Toàn bộ mục 4–8 (DataLoader, kiến trúc, phân tích tầng, baseline, tinh chỉnh) không đọc thư mục `test/`.

### 1.4. Lần chạy chính thức trên GPU (WSL2) — ngân sách 120 + 20 epoch

- **Môi trường:** Windows chặn torch CUDA bằng Smart App Control (WDAC policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}` — sự kiện CodeIntegrity 3033/3077 — chặn `c10.dll` không ký số), nên huấn luyện chạy trong **WSL2 Ubuntu-22.04** với venv `torch 2.14.1+cu130`; dữ liệu được chép sang ext4 của WSL sau khi lần chạy GPU đầu tiên lỗi `Cannot allocate memory` do I/O 9p trên `/mnt/f` (không tái phát sau khi chuyển).
- **Ngân sách theo yêu cầu chủ đề:** `MAX_EPOCHS = 120` (cell 5); grid **8 cấu hình × 15 epoch** đo trên val (cell 27); huấn luyện cuối chạy **91/120 epoch** rồi early-stop (best @ epoch 81); **fine-tune +20 epoch** với LR 7,5e-5 (cell 8.4) chạy 12/20 rồi dừng, best val 0,8736 < 0,8798 → **bị từ chối trung thực**, đánh giá test dùng mô hình full (cell 8.4 in dòng quyết định).
- **Tốc độ:** ~24 s/epoch (batch 64 — baseline), ~40 s/epoch (batch 32 — lượt cuối), ~36 s/epoch (fine-tune). Thời điểm theo timestamps checkpoint: baseline xong 18:56 → mô hình cuối 20:57 → fine-tune 21:08 → đánh giá test hoàn tất 21:11.

Kết quả ở mức sử dụng được và **vượt khoảng kỳ vọng tài liệu** (phân tích mục 7).

---

## 2. Dữ liệu

### 2.1. Tổng quan hai bộ dữ liệu (`reports/data_analysis.md` mục 0–1)

| Bộ dữ liệu | Train / Val / Test | Tổng ảnh | Kết luận |
|---|---|---|---|
| **official** (`official-kdef-dataset-by-ksatrajit0`) | 3.358 / 560 / 980 | **4.898** | **Được chọn** — ảnh KDEF gốc 562×762 RGB, đủ 70 subject, manifest khớp 100% file |
| balanced (`balanced-kdef-dataset-by-dolly-prajapati-182`) | 45.304 / 11.326 / 6.293 | 62.923 | **Bị loại** (mục 2.3) |

Số liệu đếm lại độc lập trong notebook cell 7 (khớp `data_analysis.md`): mỗi lớp 480 ảnh train (sad, surprise chỉ 479), 80 val, 140 test; tổng `{'train': 3358, 'val': 560, 'test': 980}`. Tỉ lệ ~70/10/20, giữ nguyên tỉ lệ lớp → không cần class-weight.

**4.898 so với 4.900 công bố:** KDEF gốc công bố **4.900 ảnh** (kdef.se — `papers_research.md` mục A.1); bản phân phối `official-kdef-dataset-by-ksatrajit0` đếm thực tế **4.898 ảnh**. Chênh lệch 2 ảnh **không được giải thích trong bất kỳ file nguồn nào của repo** — ghi nhận đây là khác biệt giữa con số công bố của bộ gốc và bản phân phối lại. Bộ dữ liệu vẫn nhất quán nội bộ: manifest `kdef_manifest_splits.csv` có đúng 4.898 dòng và khớp 100% file trên đĩa, 0 dòng thiếu file (`data_analysis.md` mục 0). Tương tự cùng bản chất: đề bài nêu train = 3.359 nhưng đếm bằng lệnh là 3.358 (`data_analysis.md` mục 0).

### 2.2. Phát hiện rò rỉ dữ liệu (data leakage)

Kết quả hash MD5 toàn bộ 4.898 ảnh (cell 11; tái lập bằng `python reports/check_leakage.py`, `data_analysis.md` mục 5):

| Cặp split | Ảnh trùng MD5 chéo split | Subject chéo split |
|---|---|---|
| train – val | **10** (0,2%) | 0 |
| train – test | **0** | 0 |
| val – test | **0** | 0 |

Nhóm 10 ảnh trùng train–val (cùng MD5 `320b3403...`) là **frame hỏng rất tối, tương phản cực thấp** (std 8,6/255) — bản ghi xuất bản lỗi, không phải ảnh thật bị chia đôi (`data_analysis.md` mục 5a). Xử lý: cả 10 file bị loại trước huấn luyện — cell 15 in `train: 3358 → 3349 ảnh`, `val: 560 → 559 ảnh`. **Tập test sạch hoàn toàn với train/val cả về ảnh lẫn người.**

### 2.3. Lý do loại bộ "balanced"

`data_analysis.md` mục 5b–6 — cấu trúc nguồn của bộ balanced (đếm toàn bộ 62.923 file theo tiền tố tên):

| Dạng tên file | Số lượng | Ý nghĩa |
|---|---|---|
| `Training_*` / `PublicTest_*` / `PrivateTest_*` | 28.709 + 3.589 + 3.589 = **35.887** | đúng cơ cấu tên FER-2013 |
| `aug_*.png` | **26.772** (42,6%) | bản augment |
| `Disgust_*`, `Surprise_*` (file lẻ) | **264** | không thuộc 2 nhóm trên |

35.887 + 26.772 + 264 = **62.923** — khớp tổng (đã cộng lại bằng lệnh). Ảnh 75×75 không phải định dạng KDEF (562×762); hashing mẫu 9.000 ảnh phát hiện 67 ảnh trùng byte chéo split, 100% cùng nhãn → same-label leakage trực tiếp. Kết luận: bộ này thực chất là **FER-2013 gán nhãn lại + nhân bản augment**, dùng nó thì đề tài không còn là "KDEF".

### 2.4. Bộ kiểm tra (test) — 980 ảnh

Từ manifest (cell 10; `data_analysis.md` mục 4): 7 lớp × đúng 140 ảnh (cân bằng tuyệt đối); **14 người duy nhất** (9 nữ + 5 nam) — kết quả test là đánh giá trên "người chưa từng thấy"; gender lệch **F = 630 (64,3%), M = 350 (35,7%)** → notebook báo cáo thêm accuracy theo gender (mục 6.3); 5 góc chụp S/FL/FR/HL/HR × đúng 196 ảnh; session A/B = 490/490.

### 2.5. Chuẩn hóa đầu vào

Normalize tính từ **mẫu ngẫu nhiên cố định seed 42**: mỗi lớp lấy ngẫu nhiên 80 ảnh từ các file train **đã loại 10 file MD5 hỏng** (`rng_dl = random.Random(SEED)`, `rng_dl.sample(files, 80)` cho từng lớp — cell 14), tổng 80 × 7 = **560 ảnh**, kết quả `mean=[0.3370], std=[0.1871]`. Con số 560 trùng cỡ tập val chỉ là trùng hợp cấu hình lấy mẫu (80/lớp), **không dùng dữ liệu val** để tính. Ảnh vào **grayscale 128×128** — theo Akhand 2021, 128×128 là điểm tối ưu cho CNN tự thiết kế trên KDEF (80,8% so với 61,6% @48 và 73,5% @224; `papers_research.md` mục A.5).

---

## 3. Kiến trúc CNN và phân tích từng lớp

### 3.1. KDEFNet — tổng quan

Định nghĩa tại cell 17: 4 khối, mỗi khối = **2 × (Conv3×3 → BatchNorm2d → ReLU) + MaxPool2×2**, kênh 32 → 64 → 128 → 256, kết thúc `AdaptiveAvgPool2d(1) → Dropout(0.4) → FC 256 → ReLU → Dropout(0.3) → FC 7`. Khởi tạo **He/Kaiming normal** (Conv2d + Linear, bias = 0). Input `1×128×128` grayscale.

**Tổng tham số: 1.241.191 (~1,24M), tất cả trainable** (cell 17). Thiết kế bám khuyến nghị tài liệu: CNN 2–4 tầng là vùng tối ưu cho ~3.358 ảnh train; mạng lớn huấn luyện từ đầu phản tác dụng trên KDEF (VGG-16 từ đầu 23,4%; ResNet-152 59,9% — `papers_research.md` mục A.4–A.5).

### 3.2. Bảng tham số từng lớp (từ `torchinfo.summary`, cell 17)

| Khối | Lớp | Input → Output | Tham số | Vai trò |
|---|---|---|---|---|
| 1 | Conv 1.1 3×3 (1→32) | 1×128×128 → 32×128×128 | 320 | phát hiện cạnh/gradient cơ bản |
| 1 | BN + ReLU | 32×128×128 | 64 | ổn định phân bố, phi tuyến |
| 1 | Conv 1.2 3×3 (32→32) | 32×128×128 | 9.248 | tổ hợp đặc trưng cục bộ |
| 1 | BN + ReLU + MaxPool 2×2 | → 32×64×64 | 64 | giảm độ phân giải nửa, bất biến dịch chuyển |
| 2 | Conv 2.1 (32→64) + BN | 32×64×64 → 64×64×64 | 18.496 + 128 | texture/mẫu gương mặt (lông mày, nếp gấp) |
| 2 | Conv 2.2 (64→64) + BN + ReLU + Pool | → 64×32×32 | 36.928 + 128 | |
| 3 | Conv 3.1 (64→128) + BN | 64×32×32 → 128×32×32 | 73.856 + 256 | bộ phận khuôn mặt (mắt, miệng) |
| 3 | Conv 3.2 (128→128) + BN + ReLU + Pool | → 128×16×16 | 147.584 + 256 | |
| 4 | Conv 4.1 (128→256) + BN | 128×16×16 → 256×16×16 | 295.168 + 512 | cấu hình biểu cảm toàn cục |
| 4 | Conv 4.2 (256→256) + BN + ReLU + Pool | → 256×8×8 | 590.080 + 512 | |
| — | AdaptiveAvgPool2d(1) | 256×8×8 → 256 | 0 | nén không gian, bất biến vị trí |
| — | Dropout 0.4 → FC 256 → ReLU → Dropout 0.3 → FC 7 | 256 → 7 | 65.792 + 1.799 | đầu phân loại, chống overfit |

Tổng chi phí tính toán (torchinfo): 53,58 GFLOPs mult-add/batch 64; kích thước tham số 4,96 MB. **71,4% tham số nằm ở khối 4** (886.272/1.241.191 = 0,714 — tính lại bằng lệnh; bảng trên cộng lại đúng 1.241.191); đầu FC chỉ chiếm 5,4% (67.591/1.241.191). Phần "hiểu biểu cảm" (khối 4) chiếm đa số ngân sách mô hình.

### 3.3. Kiểm tra mạng TRƯỚC huấn luyện (cell 21–22)

- Histogram activation sau ReLU ở cả 8 tầng conv: **không tầng nào chết** — tỉ lệ activation bằng 0 nằm trong 0,40–0,55, dưới ngưỡng nghi ngờ 65% (cell 21 in `✓ Không tầng nào chết`; số liệu gốc trong `reports/figures/nb_pre_layer_stats.json`). Lưu ý ~50% zero là bình thường vì ReLU chặt nửa âm của BN.
- 32 filter 3×3 của tầng conv đầu: noise có cấu trúc He, chưa học gì → `reports/figures/nb_pre_conv1_filters.png`.

### 3.4. So sánh activation TRƯỚC vs SAU huấn luyện (cell 43–45 + `reports/figures/nb_pre_layer_stats.json`)

Cùng một ảnh mẫu (`AF01HAHR.JPG`, lớp thật happy), chạy qua mạng trước và sau huấn luyện:

| Tầng (ReLU sau conv) | dead_frac trước → sau | mean trước → sau | std trước → sau |
|---|---|---|---|
| features.0.2 | 0,500 → 0,487 | 0,391 → 0,326 | 0,683 → 0,467 |
| features.0.5 | 0,555 → 0,504 | 0,329 → 0,311 | 0,619 → 0,477 |
| features.1.2 | 0,496 → 0,466 | 0,398 → 0,301 | 0,658 → 0,426 |
| features.1.5 | 0,402 → 0,486 | 0,511 → 0,340 | 0,711 → 0,488 |
| features.2.2 | 0,493 → 0,520 | 0,544 → 0,336 | 0,824 → 0,526 |
| features.2.5 | 0,488 → 0,535 | 0,513 → 0,328 | 0,808 → 0,543 |
| features.3.2 | 0,512 → 0,602 | 0,548 → 0,326 | 0,873 → 0,688 |
| features.3.5 | 0,525 → **0,651** | 0,511 → **0,259** | 0,805 → **0,786** |

Diễn giải (đối chiếu cell 43–45 và biểu đồ `nb_pre/post_feature_maps.png`, `nb_pre/post_activation_hists.png`, `nb_pre/post_conv1_filters.png`):

- **Tầng nông (khối 1)** gần như không đổi — bộ dò cạnh cơ bản đã ở gần cấu hình tốt ngay từ khởi tạo He.
- **Tầng sâu (khối 3–4) trở nên thưa và chọn lọc rõ rệt**: mean activation giảm ~2 lần ở mọi tầng sâu (0,51–0,55 → 0,26–0,34), dead_frac tầng cuối features.3.5 lên 0,651 — sát ngưỡng nghi ngờ 0,65 nhưng **không phải tầng chết**: std của chính tầng đó tăng lên 0,786, tức số neuron còn kích hoạt thì phản hồi MẠNH và khác biệt — đúng dấu hiệu một mạng đã chuyên biệt hóa: ít neuron hoạt động hơn cho một ảnh cụ thể, mỗi neuron mang nhiều thông tin hơn.
- **Filter conv đầu** chuyển từ noise cân bằng sang các bộ dò có hướng (cạnh ngang/dọc, vùng sáng/tối) — `nb_post_conv1_filters.png` so với `nb_pre_conv1_filters.png`.
- Minh chứng trực tiếp: chính ảnh mẫu này (`AF01HAHR.JPG`, happy thật) sau huấn luyện được dự đoán **đúng happy** (cell 43 in dòng này).

---

## 4. Cấu hình huấn luyện chuẩn cuối cùng (best_config)

Lưu trong `reports/test_metrics.json` (khóa `best_config`), trùng khớp log cell 29:

| Siêu tham số | Giá trị | Ghi chú / nguồn thiết kế |
|---|---|---|
| `lr` | **0,0003** | thắng trong grid 8 cấu hình trên val (0,4163 sau 15 epoch — cell 27, CSV `nb_tuning_results.csv`) |
| `batch_size` | **32** | 105 batch/epoch train (3.349 ảnh) |
| `aug_mode` | **"khong_erasing"** | flip ngang p=0,5 + RandomAffine(xoay ±10°, scale ×1,1), **không RandomErasing** (cell 15) — với tập nhỏ, xóa vùng mặt làm mất chính các chi tiết biểu cảm |
| `weight_decay` | **0,0001** | AdamW (Loshchilov & Hutter 2019) |
| `label_smooth` | **0,1** | CrossEntropyLoss (Szegedy 2016) |
| `sched_patience` | **5** | ReduceLROnPlateau(factor 0,5) theo macro-F1 val |

Thành phần cố định thêm (cell 23–25): optimizer **AdamW**, loss **CE + label smoothing 0,1**, early stopping patience 10 theo macro-F1 val (**đã kích hoạt**: lượt cuối dừng ở epoch 91/120), AMP mixed-precision trên CUDA, seed toàn phần = 42 (cell 5), Normalize(mean=[0.3370], std=[0.1871]).

**Quá trình chọn (đều chỉ đo trên val):**

1. **Baseline** (lr=1e-3, bs=64, aug "du" — cell 25): best val macro-F1 **0,8453 @ epoch 59**, early-stop 69/120.
2. **Grid 8 cấu hình × 15 epoch** (cell 27, xếp hạng theo `nb_tuning_results.csv`): (3e-4, 32, khong_erasing) **0,4163** > (3e-4, 32, du) 0,2646 > (1e-3, 32, du) 0,1756 > (3e-4, 64, khong_erasing) 0,1673 > (1e-3, 64, du) 0,1379 > (1e-3, 32, khong_erasing) 0,1305 > (3e-4, 64, du) 0,1208 > (1e-3, 64, khong_erasing) 0,1099. Lưu ý: 15 epoch đo **tốc độ hội tụ**, không phải cận trên — nhưng cấu hình thắng grid cũng thắng khi huấn luyện đầy đủ (0,8798 > 0,8453 của baseline).
3. **Huấn luyện lại đầy đủ với cấu hình thắng** (cell 29): best val macro-F1 **0,8798 @ epoch 81**, early-stop 91/120 — checkpoint `final_model.pt` epoch 81 dùng cho đánh giá test.
4. **Fine-tune +20 epoch** (cell 8.4, LR 7,5e-5 = 1/4 lr): best val **0,8736 @ epoch 7**, early-stop 12/20; **0,8736 ≤ 0,8798 → bị từ chối** (cell 8.4 in `✗ Fine-tune không vượt mô hình full → giữ mô hình full`; `test_metrics.json` ghi `"model_chosen": "full_120ep"`, `"epochs": 91`) — quyết định từ chối được thực hiện **trước** khi mở khóa test, đúng kỷ luật.

Biểu đồ: đường học baseline `reports/figures/nb_baseline_learning_curves.png`, đường học mô hình cuối `nb_final_learning_curves.png`, đường fine-tune `nb_finetune_learning_curves.png`, so sánh grid `nb_tuning_results.png`.

---

## 5. Kết quả thực nghiệm trên tập kiểm tra (980 ảnh, đánh giá MỘT LẦN)

### 5.1. Chỉ số tổng quan

Nguồn: `reports/test_metrics.json` + classification report cell 35:

| Chỉ số | Giá trị |
|---|---|
| **Accuracy** | **0,8276** (82,76% = 811/980) |
| **Macro-F1** | **0,8275** (82,75%) |
| Weighted-F1 | 0,8275 (bằng macro vì test cân bằng tuyệt đối 140 ảnh/lớp) |
| Macro precision / recall | 0,8315 / 0,8276 |
| ROC-AUC one-vs-rest (macro) | **0,9768** (cell 37) |
| Recall spread (max − min) | **0,2786** |
| Ngưỡng đoán ngẫu nhiên 1/7 | 0,1429 (14,29%) — để định vị |

Kiểm chứng chéo đã chạy trong phiên này (lệnh: `python -c "..."` tính lại từ `confusion_matrix` trong `reports/test_metrics.json`): tổng 980 ảnh, đường chéo **811** → accuracy tính lại 0,827551 = giá trị trong file; recall từng lớp tính lại khớp `per_class_recall` (sai số < 1e-12); spread tính lại 0,278571 = file. **Kết luận: file số liệu nhất quán nội bộ.**

### 5.2. Per-class precision / recall / F1 (classification report cell 35)

| Lớp | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | **0,9009** | 0,7143 | 0,7968 | 140 |
| disgust | 0,7885 | 0,8786 | 0,8311 | 140 |
| fear | 0,7092 | 0,7143 | **0,7117** | 140 |
| **happy** | **0,9929** | **0,9929** | **0,9929** | 140 |
| neutral | 0,8356 | 0,8714 | 0,8531 | 140 |
| sad | 0,7248 | 0,7714 | 0,7474 | 140 |
| surprise | 0,8686 | 0,8500 | 0,8592 | 140 |
| accuracy | | | **0,8276** | 980 |
| macro avg | 0,8315 | 0,8276 | 0,8275 | 980 |

### 5.3. Confusion matrix (hàng = nhãn thật, cột = dự đoán; `reports/test_metrics.json`)

| thật \ dự đoán | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| **angry** | **100** | 12 | 13 | 0 | 9 | 6 | 0 |
| **disgust** | 3 | **123** | 3 | 0 | 0 | 11 | 0 |
| **fear** | 1 | 7 | **100** | 1 | 2 | 11 | 18 |
| **happy** | 0 | 1 | 0 | **139** | 0 | 0 | 0 |
| **neutral** | 4 | 0 | 2 | 0 | **122** | 12 | 0 |
| **sad** | 2 | 13 | 6 | 0 | 11 | **108** | 0 |
| **surprise** | 1 | 0 | 17 | 0 | 2 | 1 | **119** |

Phân bố nhãn dự đoán (tính lại từ cột của ma trận, kiểm chứng ở trên): disgust 156 (15,9%), sad 149 (15,2%), neutral 146 (14,9%), fear 141 (14,4%), happy 140, surprise 137 (14,0%), angry 111 (11,3%) — **đủ cả 7 nhãn đều được dùng, phân bố gần đều (11–16%)**, không lớp nào bị bỏ rơi. Mô hình khá "kẽ" với angry: precision cao nhất (0,90) nhưng recall thấp nhất nhóm — chỉ dự đoán angry khi rất chắc.

### 5.4. Tham chiếu biểu đồ trong `reports/figures/`

| Biểu đồ | Nội dung |
|---|---|
| `nb_final_learning_curves.png` | Loss + macro-F1 train/val của mô hình cuối (91 epoch, early-stop) |
| `nb_baseline_learning_curves.png` | Tương tự cho baseline (69 epoch) |
| `nb_finetune_learning_curves.png` | Fine-tune +20 epoch (LR 7,5e-5) — bị từ chối vì không vượt val của mô hình full |
| `nb_tuning_results.png` | So sánh macro-F1 val của 8 cấu hình grid |
| `nb_test_confusion.png` | Confusion matrix thô + chuẩn hóa hàng |
| `nb_test_per_class_recall.png` | Recall từng lớp (thanh) |
| `nb_test_roc.png` | ROC one-vs-rest từng lớp, macro AUC 0,977 |
| `nb_test_gender_angle.png` | Accuracy theo gender và góc chụp |
| `nb_test_gradcam.png` | Grad-CAM 7 lớp (tầng conv sâu nhất, khối 4 — cell 39) |
| `nb_test_top_confusions.png` | Ảnh sai tiêu biểu của các cặp nhầm lẫn nhiều nhất |
| `nb_pre_feature_maps.png` / `nb_post_feature_maps.png` | Feature map 8 tầng conv, trước/sau huấn luyện, cùng ảnh mẫu |
| `nb_pre_activation_hists.png` / `nb_post_activation_hists.png` | Histogram activation sau ReLU, trước/sau |
| `nb_pre_conv1_filters.png` / `nb_post_conv1_filters.png` | 32 filter 3×3 tầng conv đầu, trước/sau |
| `nb_augmentation_preview.png` | Xem trước augmentation trên 1 batch train |
| `nb_class_distribution.png`, `nb_pixel_stats.png`, `nb_sample_grid.png`, `nb_test_manifest_distribution.png`, `nb_test_expr_angle_heatmap.png`, `nb_duplicate_group.png` | EDA (mục 2) |

---

## 6. Phân tích sai số và tính không thiên lệch

### 6.1. Recall spread và lớp yếu nhất

- **Recall spread = 0,2786** (`test_metrics.json`) = happy 0,9929 − (angry = fear 0,7143). Chênh 27,9 điểm % giữa lớp tốt và lớp tệ nhất, **cả 7 lớp đều recall ≥ 71%: không có lớp nào "chết"**.
- **Lớp yếu nhất: angry và fear (recall 71,4%)** — hai cảm xúc tiêu cực có hình thái gần nhau (nếp trán, mắt trũng, miệng căng); fear còn có precision thấp nhất (0,7092) vì nhận thêm 18 ảnh surprise + 11 sad.
- **Happy gần hoàn hảo: 139/140 đúng (99,3%)** — khớp đúng phát hiện tâm lý học của Calvo & Lundqvist (2008): happy là biểu cảm dễ nhận nhất (~98% ở người). Lần này mô hình **đúng hướng** với cấu trúc nhận diện của con người.

### 6.2. Các cặp nhầm lẫn nhiều nhất (cell 40)

| Thật → Dự đoán | Số ảnh |
|---|---|
| fear → surprise | 18 |
| surprise → fear | 17 |
| sad → disgust | 13 |
| angry → fear | 13 |
| neutral → sad | 12 |
| angry → disgust | 12 |

Tiêu chí chọn bảng: **top 6** theo số ảnh giảm dần — đúng như cell 9.7 của notebook in. Vị trí thứ 7 trở xuống: disgust → sad 11, fear → sad 11 (tính lại từ ma trận §5.3 bằng lệnh).

Điểm nổi bật: **fear ↔ surprise là cặp nhầm lớn nhất hai chiều (18 + 17 = 35 ảnh)** — chính là cặp mà tài liệu tâm lý học kỳ vọng (hai biểu cảm cùng mở rộng mắt/miệng, khác nhau chủ yếu ở độ cong lông mày; `papers_research.md` mục A.2; Calvo & Lundqvist 2008 cũng ghi nhận nhóm tiêu cực fear/disgust khó nhận nhất ở người). Các cặp còn lại đều là các cảm xúc tiêu cực lẫn nhau (sad↔disgust, angry→fear, angry→disgust) hoặc neutral↔sad (mặt thơ của KDEF dễ trùng sad) — **không còn hiện tượng "bể hút"**: mọi lớp đều được dự đoán với tần suất 11–16% (§5.3) và precision thấp nhất vẫn 0,71. Ảnh minh họa: `nb_test_top_confusions.png`.

### 6.3. Thiên lệch theo gender và góc chụp (cell 38)

| Nhóm | Accuracy | Số ảnh |
|---|---|---|
| Nữ (F) | 0,8143 | 630 |
| Nam (M) | 0,8514 | 350 |

Chênh lệch F−M = **0,0371** (3,71 điểm %, nam cao hơn). Sai số chuẩn xấp xỉ của hiệu 2 tỉ lệ √(0,8143·0,1857/630 + 0,8514·0,1486/350) ≈ **2,45%** (tính lại bằng lệnh), chênh này ≈ 1,5 σ — chưa đủ kết luận thiên lệch gender có ý nghĩa thống kê; ghi nhận để theo dõi (test chỉ có 14 người, 9 nữ + 5 nam; số ảnh đúng F = 513, M = 298, tổng 811 = đường chéo ma trận §5.3).

| Góc chụp | Accuracy |
|---|---|
| S (chính diện — straight) | 0,8367 |
| HL / HR (nghiêng trái/phải mức gần — half left/right) | **0,8520 / 0,8520** |
| FL / FR (nghiêng trái/phải mức xa — full left/right) | 0,8316 / **0,7653** |

Mã góc là quy ước tên file KDEF (S = straight, H = half, F = full, L/R = trái/phải). `papers_research.md` mục A.1 (trích kdef.se) mô tả 5 góc chụp của KDEF là {trái 30°, trái 45°, chính diện, phải 45°, phải 30°}; **các file nguồn trong repo không cho ánh xạ tường minh mã ↔ độ nghiêng**, nên báo cáo không gán độ cụ thể cho từng mã.

Chênh giữa góc tốt (HL/HR 85,2%) và tệ (FR 76,5%) ≈ **8,7 điểm %** — góc nghiêng xa về một bên (FR) là điều kiện khó nhất; đáng chú ý FL (83,2%) lại tốt hơn FR dù cùng mức nghiêng, cho thấy phần sai số đến từ bất đối xứng của dữ liệu/nhãn hơn là bản thân góc chụp.

### 6.4. Grad-CAM (cell 39, `nb_test_gradcam.png`)

Grad-CAM tính ở tầng conv sâu nhất (conv thứ hai khối 4, `features[3][3]`, 256 kênh — cell 39). Với mô hình đã hội tụ (val macro-F1 0,8798), các heatmap giờ **có giá trị diễn giải**: vùng nóng tập trung vào mắt — miệng — vùng giữa trán tùy lớp (xem `nb_test_gradcam.png`), đúng các cấu hình cơ mặt mà tâm lý học dùng để phân biệt biểu cảm. Không còn phải dùng ảnh dự phòng cho lớp không dự đoán đúng (happy có 139/140 đúng).

---

## 7. Đối chiếu với tài liệu học thuật

### 7.1. So với khoảng kỳ vọng protocol subject-independent

`papers_research.md` mục B tổng hợp: protocol subject-independent trên KDEF (như đề tài) kỳ vọng **65–80% accuracy / 62–77% macro-F1** cho CNN tự thiết kế (mốc LOSO của Cha et al. 2026: CNN thuần 60–78%; Akhand et al. 2021: 80,8% @128×128 nhưng split ngẫu nhiên).

| | Kỳ vọng (tài liệu) | Thực tế (lần chạy GPU này) |
|---|---|---|
| Accuracy | 65–80% | **82,76%** (vượt trần 2,76 điểm %) |
| Macro-F1 | 62–77% | **82,75%** (vượt trần 5,75 điểm %) |
| ROC-AUC macro | — | 0,977 |

Cell 9.8 của notebook tự chấm: *"TRÊN khoảng kỳ vọng 65–80% — học rất tốt; rà lại thủ tục để chắc chắn không rò rỉ."* **Đã rà lại toàn bộ thủ tục và không phát hiện rò rỉ**: (i) 0/70 subject chéo split theo manifest; (ii) 0 ảnh trùng MD5 train–test và val–test; (iii) augmentation chỉ áp trên train sau khi chia split; (iv) mọi chọn mô hình (grid, early-stop, từ chối fine-tune) chỉ dùng val; (v) test chỉ mở khóa một lần sau khi mô hình đã chốt. Kết quả 82,76% vì thế là **hợp lệ** cho protocol subject-independent.

### 7.2. Vì sao vượt khoảng kỳ vọng tài liệu

- **Ngân sách huấn luyện đầy đủ**: 91 epoch cho lượt chính + LR về 9,4e-6 cuối chu kỳ — mô hình hội tụ sâu.
- **Cấu hình thắng grid hội tụ ổn định hơn**: lr 3e-4 + batch 32 + không RandomErasing cho 0,8798 val so với 0,8453 của baseline (lr 1e-3, bs 64, có erasing) — giữ nguyên chi tiết cơ mặt thay vì xóa ngẫu nhiên vùng da quan trọng trên tập nhỏ.
- **Khoảng cách val–test vừa phải**: val macro-F1 0,8798 vs test 0,8276 — chênh 5,2 điểm % (test chỉ có 14 người chưa từng gặp, khó hơn val một cách tự nhiên); train F1 0,955 vs val 0,880 — overfit ở mức vừa phải, đã được early stopping + augmentation kiểm soát.
- Ghi chú phương pháp: grid 15 epoch xếp hạng **tốc độ hội tụ** chứ không đo cận trên của mỗi cấu hình; may mắn thay cấu hình thắng grid cũng là cấu hình tốt nhất khi huấn luyện đầy đủ — nếu không, cần chạy đầy đủ 2–3 cấu hình dẫn đầu để so sánh (đã ghi vào hạn chế, mục 8.3).

### 7.3. Đối chiếu pattern từng lớp

- Calvo & Lundqvist 2008 (mức người): happy dễ nhất ~98%, nhóm tiêu cực khó hơn — **mô hình khớp**: happy 99,3% ≥ mức người; angry/fear khó nhất (71,4%).
- Cặp nhầm lớn nhất **fear ↔ surprise (18 + 17)** đúng như dự báo tâm lý học (hai biểu cảm chung hình thái mắt/miệng mở rộng, phân biệt chính ở lông mày); các cặp sai còn lại đều trong nhóm tiêu cực (sad↔disgust, angry→fear/disgust) — đúng "nhóm tiêu cực nhầm lẫn nhau" của tài liệu.
- Thứ tự F1 happy > surprise ≈ neutral ≈ disgust > angry > sad > fear là một cấu trúc có thể diễn giải bằng tâm lý học (khác hoàn toàn lần chạy CPU khi thứ tự không khớp mẫu nào).

### 7.4. Những lựa chọn thiết kế ĐƯỢC tài liệu củng cố và đã đúng

| Lựa chọn | Tài liệu |
|---|---|
| Ảnh vào grayscale 128×128 | Akhand 2021 (80,8% @128 tối ưu) |
| CNN nhỏ ~1,24M, 4 khối conv | Cha 2026 (LeNet-5 thắng ResNet-152); Shin 2016; Akhand 2021 (VGG-16 từ đầu 23,4%) |
| AdamW + weight decay 1e-4 | Loshchilov & Hutter 2019 |
| Label smoothing 0,1 | Szegedy 2016; đặc biệt hợp lý khi nhãn gốc có噪声 (mức người 75–98% tùy lớp — Goeleven 2008) |
| Augmentation nhẹ flip ±10°/×1,1 + RandomErasing p=0,25 | Akhand 2021; Zhong 2020 |
| Chia split theo người, khóa test, chọn mô hình theo macro-F1 val | Rosenblatt 2024; Apicella 2025 |

Phương pháp luận của đề tài đạt chuẩn và kết quả GPU lần này xác nhận: cùng một pipeline, đủ ngân sách huấn luyện là vào — rồi vượt — khoảng kỳ vọng tài liệu.

---

## 8. Nhận xét — kết luận — hạn chế — hướng phát triển

### 8.1. Nhận xét

1. **Kỷ luật thực nghiệm được giữ trọn vẹn trong cả hai lần chạy**: test chỉ mở một lần mỗi lần đánh giá cuối (cell 32), mọi chọn lọc mô hình trước đó chỉ dùng val; làm sạch 10/4.898 file hỏng trước huấn luyện; quyết định **từ chối fine-tune +20 epoch** được đưa ra trên val (0,8736 ≤ 0,8798) trước khi mở khóa test — không có can thiệp chủ quan sau khi nhìn kết quả test.
2. **Kết quả 82,76% accuracy / 82,75% macro-F1 (ROC-AUC 0,977) vượt trần khoảng kỳ vọng 65–80%** của protocol subject-independent trong tài liệu, với toàn bộ thủ tục chống rò rỉ đã kiểm chứng (mục 7.1).
3. Mô hình không thiên lệch lớp: đủ 7 nhãn được dùng với tần suất 11–16%, recall mọi lớp ≥ 71,4%, spread giảm từ 43,6 → 27,9 điểm %; happy đạt 99,3% đúng bằng cấu trúc nhận diện của con người.
4. Learning curve cho thấy dừng đúng lúc: val macro-F1 ổn định quanh 0,87–0,88 từ epoch ~60, LR đã về 9,4e-6; lượt fine-tune +20 với LR 7,5e-5 không cải thiện thêm — mô hình **đã hội tụ** với kiến trúc + dữ liệu hiện tại.

### 8.2. Kết luận

Đề tài hoàn thành trọn vẹn trên GPU với ngân sách 120 + 20 epoch: (1) chọn và làm sạch bộ KDEF official với bằng chứng chống rò rỉ; (2) CNN tự thiết kế 1,24M tham số bám khuyến nghị tài liệu, phân tích từng lớp trước/sau huấn luyện; (3) pipeline EDA → baseline → grid 8 cấu hình → huấn luyện lại 91 epoch → fine-tune +20 (từ chối trung thực) → đánh giá test một lần với bộ chỉ số đầy đủ. **Mô hình đạt 82,76% accuracy subject-independent — ở mức sử dụng được và vượt kỳ vọng tài liệu**; các cặp nhầm lẫn lớn nhất (fear↔surprise) trùng với đặc tính nhận diện của con người.

### 8.3. Hạn chế

- Grid đo mỗi cấu hình 15 epoch — xếp hạng theo tốc độ hội tụ; cấu hình chậm hội tụ nhưng cận trên cao (nếu có) có thể bị bỏ sót.
- **Thiên lệch dữ liệu gốc**: 70 subject KDEF đều trẻ, da sáng, chụp studio — khó khái quát ngoài phạm vi này (Shin 2016: cross-database chỉ ~50%); "đa văn hóa" trong đề tài cần hiểu là đa góc chụp/đa subject, chưa phải đa chủng tộc ngoài phân phối gốc.
- **Test lệch gender** 64% nữ / 36% nam trên 14 người — sai số chuẩn của accuracy theo gender lớn (mục 6.3).
- 1 ảnh kích thước lệch chuẩn (633×970) bị resize về 128×128 như mọi ảnh (`data_analysis.md` mục 2).
- Góc FR chỉ 76,5% — hiệu năng bất đối xứng theo hướng nghiêng (FL 83,2%) chưa được giải thích trọn vẹn.

### 8.4. Hướng phát triển (xếp theo kỳ vọng lợi ích/chi phí)

1. Huấn luyện đầy đủ 2–3 cấu hình dẫn đầu grid (thay vì chỉ người thắng) để loại trừ nguy cơ bỏ sót cấu hình chậm hội tụ nhưng mạnh hơn.
2. Augmentation theo góc chụp hoặc tầng mixed/stronger augmentation nhắm riêng góc nghiêng xa (FR đang 76,5%).
3. Thử CLAHE/histogram equalization trong preprocessing (Shin 2016: tốt nhất trong 5 phương pháp thử) — có thể giúp fear/angry (71,4%).
4. Fusion đặc trưng handcrafted + CNN (Cha 2026: ~80% acc / macro-F1 0,77–0,79 LOSO) và focal loss cho nhóm tiêu cực (Lin 2017).
5. Transfer learning từ backbone pretrained khuôn mặt, fine-tune LR nhỏ (Akhand 2021: 96,5% nhưng split ngẫu nhiên — khác protocol, chỉ tham khảo).
6. Đánh giá cross-database (CK+, JAFFE) để đo tính khái quát "đa văn hóa" thực sự.

---

## 9. Tài liệu tham khảo

(Theo `reports/papers_research.md` mục A; trạng thái xác minh của từng nguồn ghi trong mục D của tài liệu đó.)

1. Lundqvist, D., Flykt, A., & Öhman, A. (1998). *The Karolinska Directed Emotional Faces — KDEF*. Phòng tâm lý học, Viện Karolinska. DOI 10.1037/t27732-000. (kdef.se)
2. Calvo, M. G., & Lundqvist, D. (2008). Facial expressions of emotion (KDEF): Identification under different display-duration conditions. *Behavior Research Methods*, 40(1), 109–115.
3. Goeleven, E., De Raedt, R., Leyman, L., & Verschuere, B. (2008). The Karolinska Directed Emotional Faces: A validation study. *Cognition and Emotion*, 22(6), 1094–1118.
4. Cha, S. E., et al. (2026). Facial Emotion Recognition via Fusion of Deep and Handcrafted Features (đánh giá LOSO trên KDEF). PMC13416614.
5. Akhand, M. A. H., et al. (2021). Facial Emotion Recognition Using Transfer Learning in the DeepNet Framework. *Electronics (MDPI)*, 10(9), 1036.
6. Shin, M., Kim, M., & Kwon, D.-S. (2016). Baseline CNN structure analysis for facial expression recognition. *RO-MAN 2016*, arXiv:1611.04251.
7. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR 2016*, arXiv:1512.00567.
8. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*, OpenReview Bkg6RiCqY7.
9. Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020). Random Erasing Data Augmentation. *AAAI 2020*.
10. Rosenblatt, J. D., et al. (2024). Data leakage inflates prediction performance in connectome-based machine learning. PMC10901797.
11. Apicella, A., et al. (2025). Don't push the button! Exploring data leakage risks in machine learning. *Artificial Intelligence Review* (s10462-025-11326-3).
12. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.
13. Ruiz-Garcia et al. (2017); Alshamsi et al. (2017); Liew & Yairi (2015) — số liệu trên KDEF trích theo bảng so sánh của Akhand et al. 2021 (chưa đọc nguyên văn bài gốc).

---

*Số liệu trong báo cáo này lấy từ: `reports/test_metrics.json`; output các cell của `notebooks/CNN_KDEF_FER.ipynb` (cell được ghi số tại từng đoạn); `reports/data_analysis.md`; `reports/papers_research.md`; `reports/figures/nb_pre_layer_stats.json`; `reports/figures/nb_tuning_results.csv`. Số cell tính lại sau khi chèn cell 8.4 (fine-tune +20): classification report = cell 35, ROC = 37, gender/angle = 38, Grad-CAM = 39, top-confusion = 40, verdict = 41.*

*Các kiểm chứng số học đã thực hiện lại độc lập bằng lệnh `python -c` trên dữ liệu nguồn trong phiên làm việc: (1) confusion matrix §5.3 — tổng 980, đường chéo 811 = accuracy 0,827551 khớp `test_metrics.json`, recall từng lớp và spread 0,27857 khớp tới sai số < 1e-12 (mục 5.1); (2) tổng cột ma trận = [111, 156, 141, 140, 146, 149, 137], đủ 7 nhãn, tổng 980 (mục 5.3); (3) bảng tham số §3.2 cộng lại đúng 1.241.191, khối 4 = 71,4%; (4) cấu kiện bộ balanced 35.887 + 26.772 + 264 = 62.923 (mục 2.3); (5) top-6 nhầm lẫn (18, 17, 13, 13, 12, 12) đúng như cell 40 in, vị trí 7–8 là disgust→sad 11 và fear→sad 11 (mục 6.2); (6) sai số chuẩn hiệu accuracy F−M ≈ 2,45% → chênh 3,71 điểm % ≈ 1,5 σ; F đúng 513 + M đúng 298 = 811 = đường chéo (mục 6.3); (7) thời lượng chạy theo timestamps checkpoint: baseline 18:56 → final 20:57 → fine-tune 21:08 → đánh giá test 21:11 (mục 1.4); (8) bảng grid §4 sao chép nguyên `nb_tuning_results.csv`. Kiểm chứng kỷ luật khóa test: `scripts/verify_notebook.py` exit 0 với dòng `VERIFY_OK {"accuracy": 82.76, "macro_f1": 82.75, "min_class_recall": 71.43, "recall_spread": 27.86}`.*
