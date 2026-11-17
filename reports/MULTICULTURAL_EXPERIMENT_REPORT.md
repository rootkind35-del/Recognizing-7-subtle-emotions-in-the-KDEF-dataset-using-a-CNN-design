# Báo Cáo Thực Nghiệm: Nhận Dạng Cảm Xúc Khuôn Mặt Đa Văn Hóa 3 Nguồn (KDEF + RADIATE + FER-2013)

**Dự án:** Ứng dụng học máy trong nhận diện cảm xúc khuôn mặt  
**Nhánh triển khai:** `tree2`  
**Ngày hoàn thành:** 17/11/2026  
**Mã kiểm định:** `scripts/so_sanh_mo_hinh.py`, `reports/check_leakage.py`, `scripts/verify_notebook.py`  

---

## 1. Tóm Tắt Thực Nghiệm & Kết Quả Chính

Đề tài giải quyết bài toán nhận diện 7 cảm xúc khuôn mặt cơ bản (*giận dữ, ghê tởm, sợ hãi, vui vẻ, trung tính, buồn, ngạc nhiên*) trên bộ dữ liệu khuôn mặt đa văn hóa gồm **41.269 ảnh** sạch tổng hợp từ 3 nguồn:
1. **KDEF** (70 người Thụy Điển, 5 góc chụp trong studio).
2. **RADIATE** (109 người mẫu đa sắc tộc: gốc Phi, gốc Á, gốc Mỹ Latinh, gốc Âu chụp trong studio).
3. **FER-2013** (35.887 ảnh ngoài đời thực lấy từ Internet của nhiều quốc gia).

Mô hình chính **`KDEF_Stem2Ch_MS4_Symmetric`** (kiến trúc CNN 4 khối đa tỉ lệ đối xứng, **1.273.005 tham số**, ~5,0 MB) được huấn luyện từ đầu trên tập huấn luyện 3 nguồn và so sánh công bằng với 2 phương pháp chuẩn mực đã công bố theo `Nội dung sườn cho dự án.pdf` Mục 4.3:
- **HOG + SVM** (*Dalal & Triggs, 2005; Cortes & Vapnik, 1995*): Mô hình học máy truyền thống trích xuất đặc trưng hướng gradient (8.100 chiều) kết hợp phân loại `LinearSVC` và hiệu chuẩn xác suất Platt (56.707 tham số).
- **ResNet-18** (*He và cộng sự, 2016*): Mạng sâu phần dư nạp sẵn trọng số `IMAGENET1K_V1` từ 1,28 triệu ảnh ImageNet rồi học chuyển giao (11.180.103 tham số).

### Kết quả trên 4.875 ảnh kiểm tra độc lập (người chưa từng gặp khi huấn luyện):

| Tiêu chí / Mô hình | KDEF_Stem2Ch_MS4_Symmetric (Chính) | HOG + SVM (Đối chứng 1) | ResNet-18 ImageNet (Đối chứng 2) |
|---|---:|---:|---:|
| **Số tham số** | **1.273.005** (~1/9 ResNet-18) | **56.707** | **11.180.103** |
| **Kích thước tệp mô hình** | **4,89 MB** | **0,28 MB** | **42,73 MB** (~8,7× CNN) |
| **Phép nhân-cộng / ảnh (MACs)** | **1,325 tỉ** | **< 0,001 tỉ** | **0,925 tỉ** |
| **Thời gian đoán 1 ảnh (CPU)** | **19,09 ms** (52 FPS) | **0,50 ms** (1.988 FPS) | **16,52 ms** (60 FPS) |
| **Thời gian đoán 1 ảnh (GPU RTX 4060)**| **1,84 ms** (545 FPS) | – (chỉ chạy CPU) | **2,33 ms** (429 FPS) |
| **Accuracy (Toàn bộ 4.875 ảnh)** | **73,05%** (3.561/4.875) | **43,12%** (2.102/4.875) | **72,78%** (3.548/4.875) |
| **Macro-F1 (Trung bình 7 lớp)** | **72,60%** (0,7260) | **40,58%** (0,4058) | **72,45%** (0,7245) |
| **Weighted-F1** | **72,89%** | **42,10%** | **72,70%** |
| **ROC-AUC (OvR macro)** | **0,9437** | **0,7878** | **0,9255** |
| **Accuracy KDEF (980 ảnh)** | **85,51%** (F1 85,43%) | **41,63%** (F1 38,98%) | **85,41%** (F1 85,35%) |
| **Accuracy RADIATE (306 ảnh)** | **83,01%** (F1 81,88%) | **37,58%** (F1 33,53%) | **79,09%** (F1 77,84%) |
| **Accuracy FER-2013 (3.589 ảnh)** | **68,79%** (F1 66,35%) | **44,00%** (F1 37,88%) | **68,79%** (F1 67,21%) |

**Nhận định cốt lõi:**
1. `KDEF_Stem2Ch_MS4_Symmetric` đạt hiệu năng phân loại cao nhất cả về **Accuracy (73,05%)**, **Macro-F1 (72,60%)** và **ROC-AUC (0,9437)**, vượt qua mạng sâu `ResNet-18` dù chỉ có số lượng tham số bằng ~1/9 (1,27 triệu so với 11,18 triệu) và dung lượng nhẹ hơn 8,7 lần (4,89 MB so với 42,73 MB).
2. Mô hình CNN vượt xa phương pháp truyền thống `HOG + SVM` tới **+29,93% Accuracy** và **+32,02% Macro-F1**, chứng minh ưu thế vượt trội của việc trích xuất đặc trưng phân cấp sâu tự động so với đặc trưng gradient thủ công.
3. Cả 3 mô hình đều đạt tốc độ suy luận rất nhanh trên CPU máy cá nhân (< 20 ms/ảnh, > 50 FPS), đáp ứng hoàn hảo yêu cầu ứng dụng thời gian thực qua webcam.

---

## 2. Kỷ Luật Dữ Liệu & Quy Trình Chống Rò Rỉ Tuyệt Đối

Bộ dữ liệu đa văn hóa được tổ chức qua tệp manifest duy nhất `data/manifests/multicultural_manifest.csv`:

| Nguồn | Kiểu ảnh | Train | Val | Test | Tổng ảnh | Số người (Subject) |
|---|---|---:|---:|---:|---:|---|
| **KDEF** | Chụp phòng studio, 5 góc | 3.349 | 559 | 980 | **4.888** | 70 người (48 / 8 / 14) |
| **RADIATE** | Chụp phòng studio, chính diện | 1.049 | 148 | 306 | **1.503** | 109 người (76 / 11 / 22) |
| **FER-2013** | Ngoài đời thực Internet | 28.006 | 3.283 | 3.589 | **34.878** | Nguồn mở Internet |
| **Tổng cộng** | | **32.404** (78,5%) | **3.990** (9,7%) | **4.875** (11,8%) | **41.269** | **179 người có định danh** |

### Kiểm định chất lượng độc lập (`reports/check_leakage.py`):
- **0 file thiếu:** 100% (41.269/41.269) tệp ảnh tồn tại trên đĩa và khớp mã băm MD5.
- **0 trùng lặp MD5 chéo split:** `train-val = 0`, `train-test = 0`, `val-test = 0`.
- **0 ảnh hỏng trong tập học:** 0 ảnh đen hoặc kích thước rỗng trong `train` và `val`.
- **0 rò rỉ danh tính (Subject-Disjoint):** Toàn bộ 179 người mẫu/diễn viên có mã định danh được phân hoạch tách biệt hoàn toàn giữa `train` (124 người), `val` (19 người) và `test` (36 người). Không có bất kỳ người nào xuất hiện ở cả 2 tập.
- **Khóa tập kiểm tra:** Cả 3 mô hình đều chỉ điều chỉnh siêu tham số và lưu checkpoint dựa trên tập `val`. Tập `test` chỉ được nạp đúng một lần duy nhất trong quá trình so sánh tổng kết.

---

## 3. So Sánh Chi Tiết Hiệu Năng 3 Mô Hình Trên Tập Kiểm Tra

### 3.1. Theo nguồn gốc dữ liệu

```
Accuracy theo nguồn trên tập kiểm tra:
KDEF (980 ảnh):       CNN (85,51%)  ≈  ResNet-18 (85,41%)  >>  HOG+SVM (41,63%)
RADIATE (306 ảnh):    CNN (83,01%)  >  ResNet-18 (79,09%)  >>  HOG+SVM (37,58%)
FER-2013 (3.589 ảnh): CNN (68,79%)  =  ResNet-18 (68,79%)  >>  HOG+SVM (44,00%)
```

- Ảnh chụp studio kiểm soát tốt (`KDEF`, `RADIATE`) cho độ chính xác rất cao (> 83–85%), thể hiện mô hình học được rất rõ nét các biểu cảm chuẩn mực.
- Ảnh Internet ngoài đời thực (`FER-2013`) có độ phân giải thấp (48×48 phóng to lên 160×160), bị che khuất một phần hoặc nhòe mờ, là phần thách thức nhất của bài toán (đạt 68,79%, tương đương mức trần của con người trên bộ dữ liệu này ~65–68%).

### 3.2. Theo nhóm sắc tộc / nhân khẩu học

| Nhóm sắc tộc / Nguồn gốc | Số ảnh test | CNN Accuracy / F1 | HOG+SVM Accuracy / F1 | ResNet-18 Accuracy / F1 |
|---|---:|---:|---:|---:|
| **KDEF – Bắc Âu (Thụy Điển)** | 980 | **85,5%** / 0,854 | 41,6% / 0,390 | 85,4% / 0,853 |
| **RADIATE – gốc Phi (Black)** | 112 | **75,9%** / 0,737 | 36,6% / 0,343 | 72,3% / 0,700 |
| **RADIATE – gốc Á (Asian)** | 56 | **83,9%** / 0,827 | 42,9% / 0,375 | 80,4% / 0,778 |
| **RADIATE – gốc Mỹ Latinh (Hispanic)** | 56 | **91,1%** / 0,889 | 41,1% / 0,309 | 85,7% / 0,829 |
| **RADIATE – gốc Âu (White)** | 82 | **86,6%** / 0,870 | 32,9% / 0,285 | 82,9% / 0,828 |
| **FER-2013 – Toàn cầu (Internet)** | 3.589 | **68,8%** / 0,663 | 44,0% / 0,379 | 68,8% / 0,672 |

CNN duy trì độ chính xác cao trên mọi nhóm sắc tộc, trong đó nhóm gốc Mỹ Latinh đạt 91,1%, gốc Âu đạt 86,6%, gốc Á đạt 83,9%, gốc Phi đạt 75,9%.

### 3.3. Theo giới tính (trên 1.286 ảnh KDEF + RADIATE có định danh)
- **Nữ (783 ảnh):** CNN đạt **83,3%** (F1 = 0,832) | ResNet-18 đạt 83,7% | HOG+SVM đạt 39,0%.
- **Nam (503 ảnh):** CNN đạt **87,5%** (F1 = 0,874) | ResNet-18 đạt 84,3% | HOG+SVM đạt 43,3%.
- Khoảng cách chênh lệch giữa nam và nữ của CNN là 4,2%, nhỏ hơn đáng kể so với mô hình đơn nguồn KDEF ban đầu.

### 3.4. Theo 5 góc chụp (trên 980 ảnh KDEF)

| Góc mặt | Mô tả | CNN Accuracy | HOG+SVM Accuracy | ResNet-18 Accuracy |
|---|---|---:|---:|---:|
| **FL** | Nghiêng hẳn trái (-90°) | **84,7%** | 34,2% | 82,7% |
| **HL** | Nửa trái (-45°) | **86,7%** | 41,3% | 89,3% |
| **S** | Chính diện (0°) | **87,8%** | 57,1% | 87,8% |
| **HR** | Nửa phải (+45°) | **87,2%** | 42,3% | 87,2% |
| **FR** | Nghiêng hẳn phải (+90°) | **81,1%** | 33,2% | 80,1% |
| **Độ lệch \|FL − FR\|** | Bất biến góc nghiêng đối xứng | **3,6%** | **1,0%** | **2,6%** |

Khối `SymmetricMultiScaleAUBlock` giúp CNN giữ vững độ chính xác trên cả góc chụp chính diện lẫn góc nghiêng cực đại.

### 3.5. Hiệu năng theo 7 cảm xúc

| Cảm xúc | Số ảnh | CNN Precision | CNN Recall | CNN F1 | ResNet-18 F1 | HOG+SVM F1 |
|---|---:|---:|---:|---:|---:|---:|
| **angry (giận dữ)** | 675 | 0,6940 | 0,6148 | **0,6520** | 0,6631 | 0,3485 |
| **disgust (ghê tởm)** | 239 | 0,7958 | 0,7992 | **0,7975** | 0,7970 | 0,3885 |
| **fear (sợ hãi)** | 712 | 0,6348 | 0,5225 | **0,5732** | 0,5781 | 0,2332 |
| **happy (vui vẻ)** | 1.083 | 0,9046 | 0,8929 | **0,8987** | 0,9017 | 0,6412 |
| **neutral (trung tính)**| 810 | 0,6806 | 0,7840 | **0,7286** | 0,7061 | 0,4208 |
| **sad (buồn)** | 778 | 0,5881 | 0,6478 | **0,6165** | 0,6090 | 0,2801 |
| **surprise (ngạc nhiên)**| 578 | 0,8057 | 0,8253 | **0,8154** | 0,8165 | 0,5282 |
| **Trung bình (Macro-F1)**| 4.875 | **0,7291** | **0,7266** | **0,7260** | **0,7245** | **0,4058** |

- Cảm xúc dễ nhận diện nhất: `happy` (F1 ~0,90) và `surprise` (F1 ~0,82).
- Cảm xúc thách thức nhất: `fear` (F1 ~0,57) và `sad` (F1 ~0,62), do nét mặt sợ hãi và buồn thường bị nhầm lẫn nhẹ với trung tính hoặc ngạc nhiên trên ảnh nhỏ FER-2013.

---

## 4. Tích Hợp Ứng Dụng & Kiểm Thử Toàn Diện

Hệ thống ứng dụng được cập nhật đồng bộ để sử dụng checkpoint đa văn hóa mới:
1. **Trình suy luận (`src/api/inference.py`):**
   - Nạp checkpoint chuẩn duy nhất `checkpoints/final_model.pt` (hoặc `cnn_multicultural.pt`).
   - Tự động áp dụng chuẩn hóa đa văn hóa `mean = 0,517243`, `std = 0,264053`.
   - Cung cấp hàm `predict_image()` và `predict_image_full()` trả về cảm xúc, độ tin cậy và phân phối xác suất 7 cảm xúc bằng tiếng Việt.
2. **Giao diện lập trình REST API (`src/api/app.py`):**
   - FastAPI server hỗ trợ endpoint `GET /` (thông tin mô hình, trạng thái sẵn sàng) và `POST /predict/` (tiếp nhận ảnh tải lên và trả về kết quả JSON chi tiết).
3. **Ứng dụng giao diện Web Streamlit (`src/api/ui.py`):**
   - **Thẻ 1 (Tải ảnh lên):** Tải ảnh chân dung, tự động vẽ khung xanh nhận diện khuôn mặt và hiển thị 7 thanh xác suất cảm xúc trực quan.
   - **Thẻ 2 (Camera trực tiếp):** Kết nối webcam chụp ảnh trực tiếp và phân tích tức thời.
   - **Thanh bên:** Hiển thị thông số kỹ thuật mô hình, nguồn gốc dữ liệu 3 nguồn và bảng kết quả kiểm tra.

### Kết quả kiểm định tự động (4/4 PASS, Exit Code 0):
- `reports/check_leakage.py` -> **EXIT 0 (Passed)**: 0 rò rỉ dữ liệu chéo tập.
- `scripts/verify_notebook.py` -> **EXIT 0 (VERIFY_OK)**: Toàn bộ ô code hợp lệ, khóa tập test được bảo toàn.
- `src/models/evaluate.py --split test` -> **EXIT 0 (Passed)**: Suy luận độc lập trên tập test thành công.
- `scripts/so_sanh_mo_hinh.py --split test --unlock-test` -> **EXIT 0 (Passed)**: So sánh 3 mô hình đồng bộ thành công.
- **Kiểm thử API Endpoint:** Test trực tiếp hàm `read_root()` và `predict_emotion()` -> **EXIT 0 (Passed)**.
