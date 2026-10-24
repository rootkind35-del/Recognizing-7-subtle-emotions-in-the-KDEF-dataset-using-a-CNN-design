# Báo cáo phân tích dữ liệu (EDA) — Bộ dữ liệu KDEF cho đề tài CNN nhận dạng cảm xúc

Ngày phân tích: 2026-10-24. Mọi số liệu dưới đây được tạo lại được bằng các script trong `reports/` (đường dẫn kèm từng mục). Dữ liệu gốc **không bị chỉnh sửa**.

## 0. Tổng quan hai bộ dữ liệu

| Bộ dữ liệu | Đường dẫn | Train / Val / Test | Tổng ảnh | Ghi chú |
|---|---|---|---|---|
| **official** | `data/official-kdef-dataset-by-ksatrajit0` | 3.358 / 560 / 980 | **4.898** | Kèm `kdef_manifest_splits.csv` (4.898 dòng) |
| **balanced** | `data/balanced-kdef-dataset-by-dolly-prajapati-182` | 45.304 / 11.326 / 6.293 | **62.923** | Không có manifest |

Lưu ý: đề bài nêu train official = 3.359; **đếm thực tế bằng lệnh là 3.358** (`find train -type f | wc -l`), do lớp `sad` và `surprise` chỉ có 479 ảnh train (các lớp khác 480). Manifest cũng có đúng 4.898 dòng và khớp 100% file trên đĩa (0 dòng thiếu file).

## 1. Số ảnh theo lớp × split

Kết quả chạy `python reports/eda_step1_counts.py`.

### Bộ official — cân bằng gần tuyệt đối

| Lớp | train | val | test | Tổng |
|---|---|---|---|---|
| angry | 480 | 80 | 140 | 700 |
| disgust | 480 | 80 | 140 | 700 |
| fear | 480 | 80 | 140 | 700 |
| happy | 480 | 80 | 140 | 700 |
| neutral | 480 | 80 | 140 | 700 |
| sad | 479 | 80 | 140 | 699 |
| surprise | 479 | 80 | 140 | 699 |
| **Tổng** | **3.358** | **560** | **980** | **4.898** |

Tỉ lệ chia 70/10/20 (xấp xỉ), mỗi lớp giữ nguyên tỉ lệ → **không lệch lớp**, không cần class-weight hay resampling.

### Bộ balanced — cân bằng "nhân tạo" tuyệt đối

| Lớp | train | val | test | Tổng |
|---|---|---|---|---|
| 7 lớp (Anger…Surprise) | 6.472 | 1.618 | 899 | **8.989** |
| **Tổng** | **45.304** | **11.326** | **6.293** | **62.923** |

Mỗi lớp mỗi split có số ảnh tròn tuyệt đối giống hệt nhau (6.472 / 1.618 / 899) — dấu hiệu **nhân bản/augment cơ học để ép cân bằng**, không phải phân bố tự nhiên.

Biểu đồ: `reports/figures/class_distribution_official.png`, `reports/figures/class_distribution_balanced.png`; số liệu CSV: `reports/figures/class_split_counts_official.csv`, `reports/figures/class_split_counts_balanced.csv`.

## 2. Kích thước, chế độ màu, thống kê pixel (tập train)

Kết quả chạy `python reports/eda_step2_pixels.py` (official: hash/toàn bộ 3.358 ảnh train; balanced: toàn bộ 45.304 ảnh đọc size/mode, thống kê pixel trên mẫu 4.000 ảnh, seed=42).

| Thuộc tính | official (train) | balanced (train) |
|---|---|---|
| Kích thước | **562×762** (3.357 ảnh) + 633×970 (1 ảnh: `train/angry/BM10ANFR.JPG`) | **75×75** (100% ảnh) |
| Chế độ màu | RGB 100% | RGB 100% |
| Mean (grayscale /255) | **0,3395** | 0,5073 |
| Std của các mean ảnh | 0,0629 | 0,1321 |
| Std trung bình 1 ảnh | 0,1788 | 0,2157 |
| Mean RGB /255 (3 kênh) | **[0,4378; 0,3094; 0,2368]** | chưa đo (không dùng cho huấn luyện) |
| Std RGB /255 (3 kênh) | **[0,2235; 0,1923; 0,1511]** | chưa đo (không dùng cho huấn luyện) |

Nhận xét:
- Kích thước 562×762 **chính là định dạng gốc của KDEF** → bộ official giữ ảnh gốc, chỉ 1 ảnh lệch 633×970 cần resize về chuẩn khi nạp.
- 75×75 của balanced **không phải kích thước KDEF** (thậm chí không phải 48×48 gốc FER-2013) → thêm bằng cứ balanced là ảnh gán nhãn lại từ nguồn khác.
- Chuẩn hóa cho CNN dùng bộ official: `Normalize(mean=[0.4378, 0.3094, 0.2368], std=[0.2235, 0.1923, 0.1511])` (kênh R nổi bật — da người trong studio KDEF).
- Số liệu từng ảnh: `reports/figures/pixel_stats_train_official.csv`, `reports/figures/pixel_stats_train_balanced.csv`.

## 3. Ảnh mẫu mỗi lớp — bộ official

Chạy `python reports/eda_step3_grids.py`. Grid 7 lớp × 5 ảnh (chọn ngẫu nhiên seed=42):

- Tập train: `reports/figures/sample_grid_train_official.png`
- Tập test: `reports/figures/sample_grid_test_official.png`
- Danh sách file được chọn: `reports/figures/sample_grid_files.txt`

Ảnh KDEF chuẩn: khuôn mặt nhìn thẳng/nghiêng 5 góc, nền đồng màu, biểu cảm rõ — thuận lợi cho CNN; cần crop mặt + cân Histogram (Equalize) vì một số ảnh hơi tối.

## 4. Phân tích sâu tập KIỂM TRA (test) — bộ official

Nguồn: manifest `kdef_manifest_splits.csv` lọc `split == "test"` (980 dòng) + đếm trực tiếp thư mục `test/`.

| Thuộc tính | Giá trị |
|---|---|
| Số ảnh | **980** |
| Số lớp | 7, **mỗi lớp đúng 140 ảnh** (cân bằng tuyệt đối) |
| Kích thước | 562×762 (979 ảnh) + 633×970 (1 ảnh) |
| Chế độ màu | RGB 100% |
| Subject | **14 người duy nhất** (9 nữ + 5 nam), 70 ảnh/người |
| Gender | **F = 630 (64,3%), M = 350 (35,7%)** — lệch |
| Angle (góc chụp) | S, FL, FR, HL, HR — **đều đúng 196 ảnh mỗi góc** (cân bằng tuyệt đối) |
| Session | A = 490, B = 490 |

Bảng gender × expression và angle × expression: `reports/figures/test_gender_by_expression.csv`, `reports/figures/test_angle_by_expression.csv`; heatmap: `reports/figures/test_gender_angle_heatmaps.png`; biểu đồ cột: `reports/figures/test_manifest_distribution.png`.

Khuyến cáo cho đánh giá mô hình:
- Test **cân bằng theo lớp và góc chụp** → accuracy tổng là thước đo hợp lệ; tuy nhiên **lệch gender (64% nữ)** nên cần báo cáo thêm accuracy theo gender để phát hiện thiên lệch mô hình.
- 14 subject × 70 ảnh: kết quả test là đánh giá trên "người chưa từng thấy" (subject-disjoint) — đúng chuẩn mực cho bài toán nhận dạng cảm xúc.

## 5. Kiểm tra rò rỉ dữ liệu (data leakage)

Script tái lập: `python reports/check_leakage.py` (in JSON, exit 0; kết quả lần chạy gần nhất lưu tại `reports/figures/leakage_result.json`). Phương pháp: MD5 toàn bộ byte file.

### 5a. Bộ official — hash TOÀN BỘ 4.898 ảnh

| Cặp split | Số ảnh trùng MD5 chéo split |
|---|---|
| train – val | **10** (1 nhóm hash) |
| train – test | **0** |
| val – test | **0** |
| Trùng lặp trong nội bộ split | 0 / 0 / 0 |

Điều tra cụ thể nhóm 10 ảnh trùng (MD5 `320b34032e2133fdfb80a58c953d5c94`): 9 file ở train (`disgust/AF20DIHL.JPG`, `disgust/AM25DIFL.JPG`, `disgust/AM34DIFR.JPG`, `disgust/BM21DIFL.JPG`, `disgust/BM22DIHL.JPG`, `fear/AF10AFFR.JPG`, `neutral/AF11NEHL.JPG`, `neutral/BF13NEHR.JPG`, `surprise/AF01SUFR.JPG`) và 1 file ở val (`disgust/BM24DIFL.JPG`) — **10 tên file khác subject, khác lớp nhưng cùng một nội dung byte**: một frame rất tối, tương phản cực thấp (std 8,6/255, max 139), xem tại `reports/figures/duplicate_group_official.png`. Đây là **bản ghi hỏng/bị ghi đè khi xuất bản**, không phải "cùng một ảnh thật bị chia vào 2 split": ảnh thật của các tên file đó đã mất. Tỉ lệ 10/4.898 ≈ **0,2%**.

**Theo manifest — subject có xuất hiện chéo split không: KHÔNG.**
- Subject mỗi split: train 48, val 8, test 14 (48+8+14 = 70 = đúng toàn bộ subject KDEF).
- Subject xuất hiện ở ≥2 split: **0**; từng cặp train-val / train-test / val-test đều 0.

→ **Tập test của official hoàn toàn sạch với train/val cả về ảnh lẫn người.**

### 5b. Bộ balanced — hash MẪU 9.000 ảnh (3.000/split, seed=42, ≥8.000 theo yêu cầu)

| Cặp split | Số ảnh trùng MD5 chéo split (trên mẫu 9.000) |
|---|---|
| train – val | **20** |
| train – test | **24** |
| val – test | **23** |
| Trùng lặp trong nội bộ split | train 6 / val 9 / test 12 |

31 nhóm trùng chéo split, **100% nhóm trùng cùng nhãn** (ví dụ byte-identical: `train/Sad/Training_73057687.jpg` ≡ `test/Sad/Training_85832109.jpg`) → same-label leakage trực tiếp làm phồng kết quả val/test. Mẫu chiếm 14,3% bộ dữ liệu nên ngoại suy thô ≈ 470 ảnh trùng chéo split trên toàn bộ 62.923 (ước lượng hạ mức — sampling có thể bỏ sót cặp).

**Cấu trúc nguồn của balanced (đếm toàn bộ 62.923 file theo tiền tố tên):**

| Dạng tên file | Số lượng | Ý nghĩa |
|---|---|---|
| `Training_*.jpg` | 28.709 | **= đúng số ảnh train gốc FER-2013** |
| `PublicTest_*.jpg` | 3.589 | = đúng số public-test FER-2013 |
| `PrivateTest_*.jpg` | 3.589 | = đúng số private-test FER-2013 |
| `aug_*.png` | 26.772 (42,6%) | bản augment |
| `Disgust_*`, `Surprise_*` | 264 | file lẻ |

28.709 + 3.589 + 3.589 = 35.887 — khớp **chính xác** cơ cấu FER-2013 → **"balanced-kdef" thực chất là FER-2013 gán nhãn lại + nhân bản augment, không phải KDEF**. Cách chia gốc của FER-2013 cũng đã bị phá vỡ: 20.615 file `Training_*` nằm trong train nhưng 5.213 file `Training_*` nằm trong val và 2.881 file nằm trong test.

## 6. Kết luận: chọn bộ dữ liệu huấn luyện

### ✅ Chọn: `data/official-kdef-dataset-by-ksatrajit0`

Lý do:
1. **Là ảnh KDEF thật**: kích thước gốc 562×762, tên file đúng mã KDEF (`AF01ANFL.JPG` = subject AF01, expression AN, angle FL), đủ 70 subject của KDEF.
2. **Split theo subject (subject-disjoint)** — không có người nào xuất hiện ở cả train lẫn test (0/70 subject trùng). Đây là điều kiện vàng để đo khả năng tổng quát hóa lên người mới; manifest cho phép kiểm chứng.
3. **Không có ảnh trùng MD5 giữa train–test và val–test**. Điểm duy nhất cần xử lý: 10 bản ghi hỏng cùng nội dung (0,2%) chạm train–val — **khuyến cáo loại 10 file này bằng MD5 trước khi huấn luyện** (danh sách tại mục 5a).
4. Lệch nhẹ gender trong test (64% nữ) — báo cáo accuracy theo gender thay vì xử lý dữ liệu.

### ❌ Loại: `data/balanced-kdef-dataset-by-dolly-prajapati-182`

1. **Không phải KDEF**: 35.887 ảnh mang đúng cơ cấu tên file FER-2013 + 26.772 bản augment; ảnh 75×75 không khớp định dạng KDEF. Huấn luyện trên bộ này thì đề tài không còn là "nhận dạng cảm xúc trên KDEF".
2. **Rò rỉ xác thực**: 67 ảnh trùng byte chéo split chỉ trong mẫu 9.000 ảnh (train-test = 24), cùng nhãn → val/test bị phồng điểm.
3. Cân bằng tuyệt đối 6.472/1.618/899 mỗi lớp là kết quả nhân bản, không phản ánh phân bố thật.

Theo tiêu chí đề bài ("chỉ chọn balanced nếu official hỏng nghiêm trọng và balanced sạch hơn") — thực tế **ngược lại ở mọi tiêu chí**: official gần như sạch, balanced bẩn cả về nguồn gốc lẫn rò rỉ.

## 7. Danh mục file tạo mới

| File | Nội dung |
|---|---|
| `reports/eda_step1_counts.py` | Đếm lớp × split + vẽ biểu đồ |
| `reports/eda_step2_pixels.py` | Kích thước/màu/thống kê pixel |
| `reports/eda_step3_grids.py` | Grid mẫu + phân tích manifest test |
| `reports/check_leakage.py` | Kiểm tra rò rỉ MD5 + subject, in JSON, exit 0 |
| `reports/data_analysis.md` | Báo cáo này |
| `reports/figures/*.png` | 7 biểu đồ/grid |
| `reports/figures/*.csv`, `*.json` | Số liệu thô kèm theo |
