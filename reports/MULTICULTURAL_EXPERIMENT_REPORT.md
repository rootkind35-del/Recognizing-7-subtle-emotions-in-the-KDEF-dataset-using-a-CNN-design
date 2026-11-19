# Báo Cáo Thực Nghiệm: Nhận Dạng Cảm Xúc Khuôn Mặt Đa Văn Hóa 3 Nguồn (KDEF + RADIATE + FER-2013)

**Dự án:** Ứng dụng học máy trong nhận diện cảm xúc khuôn mặt  
**Nhánh triển khai:** `tree2`  
**Giao thức kiểm chứng:** Khóa tập kiểm tra độc lập (`Locked Test Split`), chia theo người (`Subject-Disjoint`), suy luận đơn ảnh (`Single-View`), không rò rỉ dữ liệu (`Zero Data Leakage`)  
**Mã kiểm định tự động:** `scripts/so_sanh_mo_hinh.py`, `reports/check_leakage.py`, `scripts/verify_notebook.py`  

---

## 1. Tóm Tắt Thực Nghiệm & Kết Quả Tổng Thể (Sau Tối Ưu Cả 3 Mô Hình)

Thực nghiệm giải quyết bài toán nhận diện 7 cảm xúc khuôn mặt cơ bản (*giận dữ, ghê tởm, sợ hãi, vui vẻ, trung tính, buồn, ngạc nhiên*) trên bộ dữ liệu khuôn mặt đa văn hóa gồm **41.269 ảnh** sạch tổng hợp từ 3 nguồn:
1. **KDEF** (70 người mẫu Thụy Điển - Bắc Âu, chụp studio chuẩn hóa 5 góc quay).
2. **RADIATE** (109 người mẫu đa sắc tộc: gốc Phi, gốc Á, gốc Mỹ Latinh, gốc Âu chụp trong phòng studio).
3. **FER-2013** (34.878 ảnh thực tế lấy từ Internet của nhiều quốc gia, kích thước 48×48).

Nghiên cứu tiến hành xây dựng, tối ưu tuần tự bằng script ngoài trên tập `train`/`val` và so sánh thực nghiệm công bằng trên cùng tập `test` khóa giữa 3 kiến trúc:
- **Mô hình đề xuất `KDEF_Stem2Ch_MS4_Symmetric` (`v2_res`)**: Kiến trúc CNN đối xứng đa tỉ lệ nâng cấp (bổ sung skip-connection Stage 2/3, Avg+Max Strip Pooling `1024 -> 128`, **1.349.165 tham số**, dung lượng **5,19 MB**).
- **Mô hình đối chứng 1 `HOG + SVM`** (*Dalal & Triggs, 2005; Cortes & Vapnik, 1995*): Trích xuất đặc trưng hướng gradient `160x160` ô `16` (2.916 chiều) kết hợp phân loại `LinearSVC` ($C = 0,0002$, hiệu chỉnh `class_weight`) và hiệu chuẩn xác suất Platt (20.419 tham số, 0,10 MB).
- **Mô hình đối chứng 2 `ResNet-18`** (*He et al., CVPR 2016*): Mạng sâu phần dư nạp sẵn trọng số tiền huấn luyện `IMAGENET1K_V1` (1,28 triệu ảnh ImageNet), tinh chỉnh toàn mạng với differential learning rate + cosine warmup + class weighting tối ưu trên `val` (11.180.103 tham số, 42,73 MB).

### Bảng 1: Kết quả kiểm tra trên 4.875 ảnh độc lập (chưa từng xuất hiện khi huấn luyện)

| Tiêu chí đánh giá | HOG + SVM (Đối chứng 1) | ResNet-18 ImageNet (Đối chứng 2) | KDEF_Stem2Ch_MS4_Symmetric (Mô hình đề xuất) | So sánh Mô hình đề xuất vs ResNet-18 |
|---|---:|---:|---:|:---:|
| **Số tham số** | **20.419** | **11.180.103** | **1.349.165** | **Nhẹ hơn ~8,3 lần** |
| **Dung lượng checkpoint** | **0,10 MB** | **42,73 MB** | **5,19 MB** | **Nhỏ hơn ~8,2 lần** |
| **Phép nhân-cộng / ảnh (MACs)** | **< 0,001 tỉ** | **0,925 tỉ** | **1,351 tỉ** | Phù hợp thiết bị biên |
| **Thời gian suy luận CPU (1 ảnh)** | **0,43 ms** (2.302 FPS) | **13,89 ms** (72 FPS) | **17,32 ms** (58 FPS) | Đạt chuẩn thời gian thực (> 50 FPS) |
| **Thời gian suy luận GPU RTX 4060** | – (chỉ CPU) | **1,99 ms** (504 FPS) | **2,15 ms** (465 FPS) | Cạnh tranh thời gian thực |
| **Accuracy (Toàn bộ 4.875 ảnh)** | **44,57%** (2.173/4.875) | **74,28%** (3.621/4.875) | **73,39%** (3.578/4.875) | Chênh 0,89% với ~1/8,3 tham số |
| **Macro-F1 (Trung bình 7 lớp)** | **41,59%** | **73,81%** | **72,82%** | Gần tương đương (0,7282 vs 0,7381) |
| **Weighted-F1** | **42,84%** | **74,08%** | **73,33%** | Chênh 0,75% |
| **ROC-AUC (OvR macro)** | **0,7942** | **0,9492** | **0,9450** | Gần tương đương (0,9450 vs 0,9492) |
| **Accuracy KDEF (980 ảnh)** | **41,12%** (F1 38,19%) | **87,35%** (F1 87,44%) | **86,53%** (F1 86,56%) | Ngang tầm studio chuẩn |
| **Accuracy RADIATE (306 ảnh)** | **40,85%** (F1 37,08%) | **84,31%** (F1 83,19%) | **84,97%** (F1 84,37%) | **CNN vượt +0,66% đa sắc tộc** |
| **Accuracy FER-2013 (3.589 ảnh)** | **45,83%** (F1 38,08%) | **69,85%** (F1 67,92%) | **68,82%** (F1 66,30%) | ResNet-18 hưởng lợi từ ImageNet |

![So sánh tổng thể 3 mô hình](figures/mc_test_overall_metrics_3models.png)
*Hình 1: Biểu đồ so sánh các chỉ số tổng thể (Accuracy, Macro-F1, Weighted-F1, ROC-AUC) của 3 mô hình sau tối ưu trên tập kiểm tra khóa.*

---

## 2. Thiết Kế & Xây Dựng Kiến Trúc CNN Đề Xuất (`KDEF_Stem2Ch_MS4_Symmetric`, `v2_res`)

Kiến trúc mô hình chính được thiết kế chuyên biệt cho bài toán nhận diện cảm xúc khuôn mặt, tập trung giải quyết 3 thách thức lớn nhất của dữ liệu khuôn mặt:
1. **Lệch góc quay đầu (Pose/Yaw Asymmetry):** Biểu cảm cơ mặt trên ảnh chụp nghiêng trái (`FL`, `HL`) và nghiêng phải (`FR`, `HR`) có cấu trúc đối xứng gương, đòi hỏi mạng phải có tính đồng biến hình học (Equivariance).
2. **Nhiễu danh tính chủ thể (Identity Overfitting):** Tránh hiện tượng mô hình học thuộc nhân dạng người mẫu (màu da, cấu trúc xương mặt) thay vì học biến dạng cơ biểu cảm (Action Units - FACS).
3. **Mất thông tin không gian:** Khắc phục nhược điểm của Global Average Pooling truyền thống bằng Avg+Max Strip Pooling tách vùng mắt-mày và mũi-miệng.

### Chi tiết các khối chức năng cốt lõi:
1. **Multi-Scale Parallel Stem:** Hai nhánh $\text{Conv } 3\times3$ và $\text{Conv } 5\times5$ song song thu nhận đồng thời chi tiết vi mô và hình khối cơ mặt, nối thành 32 kênh $\to \text{MaxPool}(2,2)$.
2. **Stage 2 & Stage 3 + Skip-Connections (`v2_res`) + ECA-Net:** Tích hợp skip-connection ở Stage 2/3 giúp truyền dẫn gradient mượt mà hơn khi tăng độ sâu biểu diễn, kết hợp khối chú ý kênh hiệu năng cao ECA-Net ($k=3$) dập tắt kênh nhiễu danh tính.
3. **Khối Đa Tỉ Lệ Đối Xứng `SymmetricMultiScaleAUBlock` (Stage 4):** 4 nhánh song song ($\text{Conv } 3\times3$, $\text{Conv } 1\times3$ ngang, $\text{Conv } 3\times1$ dọc, $\text{Conv } 3\times3 \text{ dilation } d=2$) gom tụ $320 \to 256$ kênh + ECA Layer, đảm bảo tính đối xứng gương (`|FL − FR| = 1,5%`).
4. **Upper/Lower Avg+Max Facial Strip Pooling (`1024 -> 128`):** Nối ghép đồng thời `AdaptiveAvgPool2d((2, 1))` và `AdaptiveMaxPool2d((2, 1))` thành vector 1.024 chiều (512 chiều nửa trên mắt-mày + 512 chiều nửa dưới mũi-miệng).
5. **Head Chiếu Đa Nhiệm (`FacialProjectionHead`):** Chiếu $1024 \to 128$ chiều chuẩn hóa $L_2$ phục vụ hàm mất mát SupCon Loss và phân nhánh Classifier 7 lớp cảm xúc.

---

## 3. Kỷ Luật Dữ Liệu & Quy Trình Chống Rò Rỉ Tuyệt Đối

Bộ dữ liệu đa văn hóa được tổ chức qua tệp manifest duy nhất `data/manifests/multicultural_manifest.csv`:

| Nguồn dữ liệu | Đặc điểm ảnh | Tập Train | Tập Val | Tập Test | Tổng số ảnh | Số chủ thể (Subject) |
|---|---|---:|---:|---:|---:|---|
| **KDEF** | Chụp studio chuẩn hóa, 5 góc | 3.349 | 559 | 980 | **4.888** | 70 người (48 / 8 / 14) |
| **RADIATE** | Chụp studio đa sắc tộc, chính diện | 1.049 | 148 | 306 | **1.503** | 109 người (76 / 11 / 22) |
| **FER-2013** | Ảnh Internet đời thực, đa dạng góc | 28.006 | 3.283 | 3.589 | **34.878** | Nguồn mở Internet |
| **Tổng cộng** | | **32.404** (78,5%) | **3.990** (9,7%) | **4.875** (11,8%) | **41.269** | **179 người có định danh** |

Kiểm định độc lập bằng `reports/check_leakage.py` xác nhận: **0 file thiếu**, **0 trùng lặp MD5 chéo tập**, **0 ảnh đen hỏng trong tập học**, và **0 rò rỉ danh tính người mẫu** giữa `train`/`val`/`test`.

---

## 4. Phân Tích Thực Nghiệm Đa Chiều Trên Tập Kiểm Tra Khóa

![Độ chính xác theo nguồn](figures/mc_test_accuracy_by_source_3models.png)
*Hình 2: So sánh độ chính xác phân loại theo từng nguồn dữ liệu (KDEF, RADIATE, FER-2013) của 3 mô hình.*

### 4.1. Khả năng khái quát hóa theo sắc tộc (Nhân khẩu học)

| Nhóm sắc tộc / Nguồn gốc | Số ảnh test | HOG + SVM Acc / F1 | ResNet-18 Acc / F1 | Mô hình đề xuất Acc / F1 |
|---|---:|---:|---:|---:|
| **Bắc Âu (KDEF Thụy Điển)** | 980 | 41,1% / 0,382 | **87,3%** / **0,874** | 86,5% / 0,866 |
| **Gốc Phi (RADIATE Black)** | 112 | 33,9% / 0,321 | **80,4%** / **0,787** | 78,6% / 0,773 |
| **Gốc Á (RADIATE Asian)** | 56 | 48,2% / 0,446 | 83,9% / **0,827** | **83,9%** / 0,822 |
| **Gốc Mỹ Latinh (RADIATE Hispanic)**| 56 | 44,6% / 0,373 | 83,9% / 0,789 | **96,4%** / **0,964** |
| **Gốc Âu (RADIATE White)** | 82 | 42,7% / 0,370 | **90,2%** / **0,900** | 86,6% / 0,867 |
| **Toàn cầu Internet (FER-2013)** | 3.589 | 45,8% / 0,381 | **69,9%** / **0,679** | 68,8% / 0,663 |

### 4.2. Phân tích theo 5 góc chụp khuôn mặt (Tập KDEF, 980 ảnh)

| Góc chụp | Mô tả tư thế | HOG + SVM Accuracy | ResNet-18 Accuracy | Mô hình đề xuất Accuracy |
|---|---|---:|---:|---:|
| **FL** | Nghiêng hẳn trái (-90°) | 27,0% | **85,2%** | 84,2% |
| **HL** | Nửa trái (-45°) | 42,3% | **89,3%** | 87,8% |
| **S** | Chính diện (0°) | 61,7% | **87,8%** | 86,2% |
| **HR** | Nửa phải (+45°) | 40,8% | **88,8%** | **88,8%** |
| **FR** | Nghiêng hẳn phải (+90°) | 33,7% | **85,7%** | **85,7%** |
| **Độ lệch \|FL − FR\|** | Tính đối xứng góc chụp | 6,6% | **0,5%** | **1,5%** |

### 4.3. Đánh giá chi tiết theo 7 cảm xúc

![F1-Score theo 7 cảm xúc](figures/mc_test_f1_by_emotion_3models.png)
*Hình 3: So sánh F1-Score chi tiết của 7 lớp cảm xúc giữa 3 mô hình trên tập kiểm tra.*

| Lớp cảm xúc | Số ảnh test | HOG + SVM F1 | ResNet-18 F1 | Mô hình đề xuất Precision | Mô hình đề xuất Recall | Mô hình đề xuất F1 |
|---|---:|---:|---:|---:|---:|---:|
| **happy (vui vẻ)** | 1.083 | 0,6466 | **0,9058** | 0,9056 | 0,8947 | **0,9001** |
| **surprise (ngạc nhiên)** | 578 | 0,5501 | **0,8225** | 0,8007 | 0,8270 | **0,8136** |
| **disgust (ghê tởm)** | 239 | 0,4218 | **0,8089** | 0,8398 | 0,7238 | **0,7775** |
| **neutral (trung tính)** | 810 | 0,4300 | **0,7398** | 0,6865 | 0,7704 | **0,7260** |
| **angry (giận dữ)** | 675 | 0,3478 | **0,6682** | 0,6815 | 0,6341 | **0,6569** |
| **sad (buồn)** | 778 | 0,2949 | **0,6306** | 0,5875 | 0,6645 | **0,6236** |
| **fear (sợ hãi)** | 712 | 0,2202 | 0,5911 | 0,6650 | 0,5463 | **0,5998** |
| **Trung bình (Macro-F1)** | 4.875 | **0,4159** | **0,7381** | **0,7381** | **0,7230** | **0,7282** |

![Ma trận nhầm lẫn 3 mô hình](figures/mc_test_confusion_3models.png)
*Hình 4: Ma trận nhầm lẫn chuẩn hóa của 3 mô hình trên tập kiểm tra khóa (4.875 ảnh).*

---

## 5. Nhận Xét & Đánh Giá Khoa Học Khách Quan

1. **Hiệu quả cân bằng giữa độ chính xác và chi phí tính toán của CNN đề xuất (`v2_res`):**
   - Sau tối ưu, `KDEF_Stem2Ch_MS4_Symmetric` (`v2_res`) tăng hiệu năng lên **73,39% Accuracy**, **0,9450 ROC-AUC** và giảm độ lệch đối xứng góc chụp `|FL − FR|` xuống chỉ còn **1,5%** (giảm hơn một nửa so với 3,6% trước tối ưu).
   - Dù có kích thước chỉ **1,35M tham số (5,19 MB)** — nhẹ hơn **8,3 lần** so với `ResNet-18` (11,18M tham số, 42,73 MB), mô hình CNN vẫn đạt độ chính xác cao hơn trên bộ dữ liệu đa sắc tộc **RADIATE** (**84,97% vs 84,31%**) và vượt trên lớp cảm xúc khó **fear** (F1 **0,5998 vs 0,5911**).
2. **Sức mạnh của học chuyển giao ImageNet khi tối ưu đúng cách (`ResNet-18`):**
   - Khi được tối ưu với differential learning rate, cosine warmup và class weighting trên `val`, `ResNet-18` tăng vọt từ 72,78% lên **74,28% Accuracy** (**0,9492 ROC-AUC**), dẫn đầu tổng thể nhờ khả năng khái quát hóa tốt các ảnh nhiễu, độ phân giải thấp ngoài đời thực của tập FER-2013 (69,85% vs 68,82%).
3. **Giới hạn của phương pháp cổ điển (`HOG + SVM`):**
   - Tối ưu hóa kích thước ô HOG (`160x160`, ô `16`) giúp `HOG + SVM` tăng từ 43,12% lên **44,57% Accuracy** (2.302 FPS trên CPU), nhưng vẫn thua kém ~29% so với 2 mô hình học sâu do đặc trưng gradient thủ công không thích ứng được với sự biến thiên phức tạp về ánh sáng, góc quay và biểu cảm đa văn hóa.
