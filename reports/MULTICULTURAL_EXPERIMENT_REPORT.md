# Báo Cáo Thực Nghiệm: Nhận Dạng Cảm Xúc Khuôn Mặt Đa Văn Hóa 3 Nguồn (KDEF + RADIATE + FER-2013)

**Dự án:** Ứng dụng học máy trong nhận diện cảm xúc khuôn mặt  
**Nhánh triển khai:** `tree2`  
**Giao thức kiểm chứng:** Khóa tập kiểm tra độc lập (`Locked Test Split`), chia theo người (`Subject-Disjoint`), suy luận đơn ảnh (`Single-View`), không rò rỉ dữ liệu (`Zero Data Leakage`)  
**Mã kiểm định tự động:** `scripts/so_sanh_mo_hinh.py`, `reports/check_leakage.py`, `scripts/verify_notebook.py`  

---

## 1. Tóm Tắt Thực Nghiệm & Kết Quả So Sánh Theo Chuẩn Bài Báo Tham Chiếu (`CNN` vs 6 Mô Hình Học Chuyển Giao)

Thực nghiệm giải quyết bài toán nhận diện 7 cảm xúc khuôn mặt cơ bản (*giận dữ, ghê tởm, sợ hãi, vui vẻ, trung tính, buồn, ngạc nhiên*) trên bộ dữ liệu khuôn mặt đa văn hóa gồm **41.269 ảnh** sạch tổng hợp từ 3 nguồn:
1. **KDEF** (70 người mẫu Thụy Điển - Bắc Âu, chụp studio chuẩn hóa 5 góc quay).
2. **RADIATE** (109 người mẫu đa sắc tộc: gốc Phi, gốc Á, gốc Mỹ Latinh, gốc Âu chụp trong phòng studio).
3. **FER-2013** (34.878 ảnh thực tế lấy từ Internet của nhiều quốc gia, kích thước 48×48).

Nghiên cứu tiến hành xây dựng, huấn luyện theo đúng quy trình của bài báo tham chiếu (*"Enhancing Esophageal Cancer Detection Using Transfer Learning Techniques"* — Huynh, Nguyen-Cao et al., TDMU) trên tập `train` (32.404 ảnh) / `val` (3.990 ảnh) và so sánh thực nghiệm công bằng trên cùng tập `test` khóa (4.875 ảnh) giữa:
- **Mô hình đề xuất `CNN` (`KDEF_Stem2Ch_MS4_Symmetric`, `v2_res`)**: Kiến trúc CNN đối xứng đa tỉ lệ chuyên biệt cho khuôn mặt (skip-connection Stage 2/3, ECA-Net, SymmetricMultiScaleAUBlock, Avg+Max Strip Pooling `1024 -> 128 -> 7`, **1.349.165 tham số**, dung lượng **5,19 MB**).
- **6 Mô hình Học chuyển giao tiền huấn luyện trên ImageNet-1K (`VGG16`, `VGG19`, `InceptionV3`, `DenseNet201`, `ResNet50`, `EfficientNetB4`)**: Đóng băng các tầng trích xuất đặc trưng đầu (*frozen initial layers*), tinh chỉnh các tầng sâu phía sau (*fine-tuned later layers*) kết hợp đầu phân loại tùy chỉnh `Global Average Pooling → FC(128) → FC(64) → Dropout(0.7) → FC(7, Softmax)`, huấn luyện với tốc độ học `lr = 0.0001` (`1e-4`), tăng cường dữ liệu (*Rotation, Distortion, Flipping*) trong **20 epochs** (`scripts/train_transfer_learning.py`).
- **2 Mô hình đối chứng bổ sung**: `HOG + SVM` (20.419 tham số, 0,10 MB) và `ResNet-18` (11.180.103 tham số, 42,73 MB).

![Sơ đồ kiến trúc tổng thể](figures/paper_fig1_proposed_architecture.png)
*Fig. 1: Sơ đồ kiến trúc tổng thể của nghiên cứu (Proposed architecture of the study).*

### Biểu đồ Hội tụ Accuracy & Loss qua 20 Epochs (Theo Fig. 2 & Fig. 3 bài báo tham chiếu)

![Biểu đồ Accuracy 20 epochs](figures/paper_fig2_accuracy_curves.png)
*Fig. 2: Training (a) and Validation (b) Accuracy of Models qua 20 epochs (`VGG16`, `VGG19`, `DenseNet201`, `InceptionV3`, `ResNet50`, `EfficientNetB4`, và `CNN`).*

![Biểu đồ Loss 20 epochs](figures/paper_fig3_loss_curves.png)
*Fig. 3: Training (a) and Validation (b) Loss of Models qua 20 epochs.*

### Table 1: Bảng so sánh hiệu năng 6 mô hình Pre-trained Transfer Learning và mô hình đề xuất `CNN` trên 4.875 ảnh Test độc lập

| Model | Accuracy | Precision (Macro / Weighted) | Recall (Macro / Weighted) | F1-Score (Macro / Weighted) | Specificity | ROC-AUC | Số tham số | Tệp (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `EfficientNetB4` | 55,34% | 53,15% / 55,20% | 53,66% / 55,34% | 53,09% / 55,00% | 92,46% | 0,8683 | 17.787.215 | 68,61 MB |
| `InceptionV3` | 57,50% | 56,14% / 57,73% | 56,40% / 57,50% | 56,21% / 57,56% | 92,83% | 0,8529 | 22.056.935 | 84,49 MB |
| `VGG19` | 64,47% | 64,15% / 64,70% | 63,84% / 64,47% | 63,96% / 64,56% | 94,00% | 0,8917 | 20.099.143 | 76,70 MB |
| `DenseNet201` | 64,51% | 63,97% / 64,78% | 63,65% / 64,51% | 63,78% / 64,61% | 94,01% | 0,8907 | 18.347.911 | 71,32 MB |
| `VGG16` | 66,07% | 65,52% / 66,15% | 65,42% / 66,07% | 65,44% / 66,08% | 94,28% | 0,9047 | 14.789.447 | 56,44 MB |
| `ResNet50` | 67,02% | 65,97% / 66,93% | 66,27% / 67,02% | 66,07% / 66,93% | 94,44% | 0,9098 | 23.779.399 | 91,03 MB |
| **`CNN` (Đề xuất)** | **73,39%** | **73,81%** / **73,66%** | **72,30%** / **73,39%** | **72,82%** / **73,33%** | **95,50%** | **0,9450** | **1.349.165** | **5,19 MB** |
| *(Đối chứng bổ sung)* `HOG + SVM` | 44,57% | 41,14% / 43,19% | 43,65% / 44,57% | 41,59% / 42,84% | 90,59% | 0,7942 | 20.419 | 0,10 MB |
| *(Đối chứng bổ sung)* `ResNet-18` | 74,28% | 73,77% / 74,07% | 74,08% / 74,28% | 73,81% / 74,08% | 95,66% | 0,9492 | 11.180.103 | 42,73 MB |

![Biểu đồ Table 1](figures/paper_table1_metrics_bar.png)
*Biểu đồ Table 1: Biểu đồ so sánh các chỉ số Accuracy, Precision, Recall và F1-Score của 6 mô hình Transfer Learning và mô hình đề xuất `CNN` trên tập kiểm tra khóa.*

![Ma trận nhầm lẫn các mô hình](figures/paper_fig4_confusion_matrices.png)
*Fig. 4: Ma trận nhầm lẫn (Confusion Matrices) của các mô hình: (a) EfficientNetB4, (b) ResNet50, (c) VGG19, (d) VGG16, (e) InceptionV3, (f) DenseNet201, và (g) CNN trên 4.875 ảnh kiểm tra.*

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
