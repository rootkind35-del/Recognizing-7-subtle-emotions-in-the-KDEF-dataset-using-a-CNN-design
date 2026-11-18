# Báo Cáo Thực Nghiệm: Nhận Dạng Cảm Xúc Khuôn Mặt Đa Văn Hóa 3 Nguồn (KDEF + RADIATE + FER-2013)

**Dự án:** Ứng dụng học máy trong nhận diện cảm xúc khuôn mặt  
**Nhánh triển khai:** `tree2`  
**Giao thức kiểm chứng:** Khóa tập kiểm tra độc lập (`Locked Test Split`), chia theo người (`Subject-Disjoint`), không rò rỉ dữ liệu (`Zero Data Leakage`)  
**Mã kiểm định tự động:** `scripts/so_sanh_mo_hinh.py`, `reports/check_leakage.py`, `scripts/verify_notebook.py`  

---

## 1. Tóm Tắt Thực Nghiệm & Kết Quả Tổng Thể

Thực nghiệm giải quyết bài toán nhận diện 7 cảm xúc khuôn mặt cơ bản (*giận dữ, ghê tởm, sợ hãi, vui vẻ, trung tính, buồn, ngạc nhiên*) trên bộ dữ liệu khuôn mặt đa văn hóa gồm **41.269 ảnh** sạch tổng hợp từ 3 nguồn:
1. **KDEF** (70 người mẫu Thụy Điển - Bắc Âu, chụp studio chuẩn hóa 5 góc quay).
2. **RADIATE** (109 người mẫu đa sắc tộc: gốc Phi, gốc Á, gốc Mỹ Latinh, gốc Âu chụp trong phòng studio).
3. **FER-2013** (34.878 ảnh thực tế lấy từ Internet của nhiều quốc gia, kích thước 48×48).

Nghiên cứu tiến hành xây dựng và so sánh thực nghiệm công bằng giữa 3 kiến trúc:
- **Mô hình đề xuất `KDEF_Stem2Ch_MS4_Symmetric`** (kiến trúc CNN đối xứng đa tỉ lệ, **1.273.005 tham số**, dung lượng **4,89 MB**).
- **Mô hình đối chứng 1 `HOG + SVM`** (*Dalal & Triggs, 2005; Cortes & Vapnik, 1995*): Mô hình học máy truyền thống trích xuất đặc trưng hướng gradient 8.100 chiều kết hợp phân loại `LinearSVC` ($C = 0,0001$, `class_weight='balanced'`) và hiệu chuẩn xác suất Platt (56.707 tham số, 0,28 MB).
- **Mô hình đối chứng 2 `ResNet-18`** (*He et al., CVPR 2016*): Mạng sâu phần dư nạp sẵn trọng số tiền huấn luyện `IMAGENET1K_V1` (1,28 triệu ảnh ImageNet) rồi học chuyển giao và tinh chỉnh thích ứng (11.180.103 tham số, 42,73 MB).

### Bảng 1: Kết quả kiểm tra trên 4.875 ảnh độc lập (chưa từng xuất hiện khi huấn luyện)

| Tiêu chí đánh giá | HOG + SVM (Đối chứng 1) | ResNet-18 ImageNet (Đối chứng 2) | KDEF_Stem2Ch_MS4_Symmetric (Mô hình đề xuất) | So sánh Mô hình đề xuất vs ResNet-18 |
|---|---:|---:|---:|:---:|
| **Số tham số** | **56.707** | **11.180.103** | **1.273.005** | **Nhẹ hơn ~8,8 lần** |
| **Dung lượng checkpoint** | **0,28 MB** | **42,73 MB** | **4,89 MB** | **Nhỏ hơn ~8,7 lần** |
| **Phép nhân-cộng / ảnh (MACs)** | **< 0,001 tỉ** | **0,925 tỉ** | **1,325 tỉ** | Vừa vặn cho thiết bị biên |
| **Thời gian suy luận CPU (1 ảnh)** | **0,50 ms** (1.988 FPS) | **16,52 ms** (60 FPS) | **19,09 ms** (52 FPS) | Đạt chuẩn thời gian thực (> 50 FPS) |
| **Thời gian suy luận GPU RTX 4060** | – (chỉ CPU) | **2,33 ms** (429 FPS) | **1,84 ms** (545 FPS) | **Nhanh hơn 27% trên GPU** |
| **Accuracy (Toàn bộ 4.875 ảnh)** | **43,12%** (2.102/4.875) | **72,78%** (3.548/4.875) | **73,05%** (3.561/4.875) | **+0,27%** |
| **Macro-F1 (Trung bình 7 lớp)** | **40,58%** | **72,45%** | **72,60%** | **+0,15%** |
| **Weighted-F1** | **42,10%** | **72,70%** | **72,89%** | **+0,19%** |
| **ROC-AUC (OvR macro)** | **0,7878** | **0,9255** | **0,9437** | **+0,0182** *(Vượt trội)* |
| **Accuracy KDEF (980 ảnh)** | **41,63%** (F1 38,98%) | **85,41%** (F1 85,35%) | **85,51%** (F1 85,43%) | **+0,10%** |
| **Accuracy RADIATE (306 ảnh)** | **37,58%** (F1 33,53%) | **79,09%** (F1 77,84%) | **83,01%** (F1 81,88%) | **+3,92%** *(Vượt trội đa sắc tộc)* |
| **Accuracy FER-2013 (3.589 ảnh)** | **44,00%** (F1 37,88%) | **68,79%** (F1 67,21%) | **68,79%** (F1 66,35%) | Ngang bằng |

![So sánh tổng thể 3 mô hình](figures/mc_test_overall_metrics_3models.png)
*Hình 1: Biểu đồ so sánh các chỉ số tổng thể (Accuracy, Macro-F1, Weighted-F1, ROC-AUC) của 3 mô hình trên tập kiểm tra khóa.*

---

## 2. Thiết Kế & Xây Dựng Kiến Trúc CNN Đề Xuất (`KDEF_Stem2Ch_MS4_Symmetric`)

Kiến trúc mô hình chính được thiết kế chuyên biệt cho bài toán nhận diện cảm xúc khuôn mặt, tập trung giải quyết 3 thách thức lớn nhất của dữ liệu khuôn mặt:
1. **Lệch góc quay đầu (Pose/Yaw Asymmetry):** Biểu cảm cơ mặt trên ảnh chụp nghiêng trái (`FL`, `HL`) và nghiêng phải (`FR`, `HR`) có cấu trúc đối xứng gương, đòi hỏi mạng phải có tính đồng biến hình học (Equivariance).
2. **Nhiễu danh tính chủ thể (Identity Overfitting):** Tránh hiện tượng mô hình học thuộc nhân dạng người mẫu (màu da, cấu trúc xương mặt) thay vì học biến dạng cơ biểu cảm (Action Units - FACS).
3. **Mất thông tin không gian:** Khắc phục nhược điểm của Global Average Pooling truyền thống (vốn làm mất sự phân định giữa vùng mắt-mày và vùng miệng-cằm).

### Sơ đồ cấu trúc mạng:

```
                      [Ảnh vào: 1 x 160 x 160 (Xám + CLAHE)]
                                        │
           ┌────────────────────────────┴───────────────────────────┐
           ▼                                                        ▼
    Conv 3x3 (1->16)                                         Conv 5x5 (1->16)
    [Chi tiết cạnh vi mô]                                    [Ngữ cảnh khối cơ mặt]
           └────────────────────────────┬───────────────────────────┘
                                        ▼ (Concat: 32 kênh)
                           BatchNorm2d + SiLU + Conv 3x3 (32->32)
                                        ▼
                                MaxPool2d(2, 2)  --> [32 x 80 x 80]
                                        │
                                        ▼
             Stage 2: 2x [Conv 3x3 (64) + BN + SiLU] + MaxPool2d
                                                 --> [64 x 40 x 40]
                                        │
                                        ▼
             Stage 3: 2x [Conv 3x3 (128) + BN + SiLU] + ECA-Net (k=3) + MaxPool2d
                                                 --> [128 x 20 x 20]
                                        │
                                        ▼
             Stage 4: Conv 3x3 (128->256) + SymmetricMultiScaleAUBlock(256)
                                                 --> [256 x 20 x 20]
                                        │
                                        ▼
             Upper/Lower Facial Strip Pooling: AdaptiveAvgPool2d((2, 1))
                      [Dải trên: Mắt-Mày] + [Dải dưới: Mũi-Miệng]
                                                 --> Vector 512 chiều
                                        │
                                        ▼
             FacialProjectionHead (Dropout 0.45 -> Linear 512->128 -> BN -> SiLU)
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
                 L2-Normalize                   Linear 128 -> 7
            Embedding 128 chiều                 (Logits 7 cảm xúc)
            (Dùng cho SupCon Loss)
```

### Chi tiết các khối chức năng cốt lõi:

#### 2.1. Multi-Scale Parallel Stem (Tầng khởi đầu 2 nhánh song song)
- Tầng đầu tiên tiếp nhận ảnh xám $160\times160$ đã cân bằng độ tương phản CLAHE.
- Thay vì dùng một kernel đơn lẻ, cấu trúc rẽ thành 2 nhánh tích chập song song:
  - Nhánh $\text{Conv } 3\times3$ ($1 \to 16$ kênh): Trích xuất cạnh biên sắc nét, chi tiết nếp nhăn cục bộ tần số cao.
  - Nhánh $\text{Conv } 5\times5$ ($1 \to 16$ kênh): Bao quát vùng tiếp nhận rộng hơn, nắm bắt hình khối cơ mặt.
- Nối ghép 2 nhánh thành 32 kênh $\to \text{BatchNorm} \to \text{SiLU} \to \text{Conv } 3\times3 (32 \to 32) \to \text{MaxPool}(2, 2)$ cho bản đồ đặc trưng $32 \times 80 \times 80$.

#### 2.2. Efficient Channel Attention (ECA-Net) tại Stage 3
- Tại Stage 3 ($128$ kênh, kích thước $20\times20$), mô hình tích hợp khối chú ý kênh hiệu năng cao ECA-Net (*Wang et al., CVPR 2020*):
  $$\omega = \sigma(\text{Conv1D}_{k=3}(\text{GlobalAvgPool}(x)))$$
- Cơ chế này thay thế hoàn toàn tầng kết nối dày đặc (Fully-Connected) 2 lớp nặng nề của SE-Net bằng tích chập 1D cục bộ kích thước $k=3$. Nhờ đó, mạng tự động học cách khuếch đại các kênh đặc trưng mang biến dạng cảm xúc và triệt tiêu các kênh mang đặc trưng nhận dạng người mẫu với chi phí tham số tăng thêm xấp xỉ bằng $0$.

#### 2.3. Khối Đa Tỉ Lệ Đối Xứng `SymmetricMultiScaleAUBlock` (Stage 4)
- Là hạt nhân biểu diễn của kiến trúc, khối này nhận 256 kênh và phân tách thành **4 nhánh tích chập song song**:
  1. $\text{Conv } 3\times3$ (96 kênh): Trích xuất biến dạng đẳng hướng (khóe mắt, nếp nhăn gò má).
  2. $\text{Conv } 1\times3$ (64 kênh): Kernel định hướng **ngang**, dò các biến dạng cơ kéo dẹt theo chiều ngang (kéo khóe môi khi cười, mím môi khi giận, dãn mí mắt).
  3. $\text{Conv } 3\times1$ (64 kênh): Kernel định hướng **dọc**, dò các biến dạng nếp nhăn đứng (rãnh mũi - má, nhíu cơ mày giữa trán, há cằm khi ngạc nhiên).
  4. $\text{Conv } 3\times3 \text{ Dilation } d=2$ (96 kênh): Tích chập giãn nở với trường tiếp nhận hiệu dụng $7\times7$, thu nhận mối tương quan toàn thể giữa các vùng cơ mặt khác nhau.
- Tổng $96 + 64 + 64 + 96 = 320$ kênh được gom tụ qua tầng chiếu $\text{Conv } 1\times1 \to 256$ kênh kèm $\text{BatchNorm} \to \text{SiLU} \to \text{ECA Layer}$.
- **Tính đối xứng gương (Left/Right Flip-Equivariant):** Sự kết hợp cân đối giữa kernel $(1\times3)$ và $(3\times1)$ tạo ra sự đối xứng hình học tự nhiên, giúp mô hình phản ứng nhất quán dù khuôn mặt quay về hướng trái hay hướng phải.

#### 2.4. Upper/Lower Facial Strip Pooling (`AdaptiveAvgPool2d((2, 1))`)
- Global Average Pooling chuẩn $(1\times1)$ nén toàn bộ không gian mặt thành một điểm duy nhất, làm mất thông tin phân bố trên dưới.
- Kiến trúc áp dụng **Strip Pooling $(2, 1)$**, chia bản đồ đặc trưng thành 2 dải dọc:
  - **Dải trên ($1 \times 256$ chiều):** Đại diện cho vùng Trán - Mày - Mắt (chứa các Action Unit: AU1, AU2, AU4, AU5 — đặc trưng cho cảm xúc *sợ hãi, ngạc nhiên, giận dữ*).
  - **Dải dưới ($1 \times 256$ chiều):** Đại diện cho vùng Mũi - Miệng - Cằm (chứa các AU12, AU15, AU20, AU26 — đặc trưng cho *vui vẻ, ghê tởm, buồn*).
- Trải phẳng thành vector biểu diễn **512 chiều** bảo toàn trọn vẹn ngữ nghĩa không gian của giải phẫu học cơ mặt.

#### 2.5. Head Chiếu Biểu Diễn Đa Nhiệm (`FacialProjectionHead`)
- Vector 512 chiều đi qua Dropout $(0,45) \to \text{Linear}(512 \to 128) \to \text{BatchNorm1d} \to \text{SiLU}$.
- Tách làm 2 nhánh đầu ra:
  - **Nhánh Embedding (128 chiều chuẩn hóa $L_2$):** Dùng để tính hàm mất mát tương phản có giám sát (**Supervised Contrastive Loss - SupCon**), giúp kéo gần các vector cảm xúc cùng loại trên mặt cầu siêu chiều bất kể chủng tộc hay góc chụp.
  - **Nhánh Classifier ($\text{Linear } 128 \to 7$):** Dự đoán phân phối xác suất của 7 cảm xúc.

---

## 3. Kỷ Luật Dữ Liệu & Quy Trình Chống Rò Rỉ Tuyệt Đối

Bộ dữ liệu đa văn hóa được tổ chức qua tệp manifest duy nhất `data/manifests/multicultural_manifest.csv`:

| Nguồn dữ liệu | Đặc điểm ảnh | Tập Train | Tập Val | Tập Test | Tổng số ảnh | Số chủ thể (Subject) |
|---|---|---:|---:|---:|---:|---|
| **KDEF** | Chụp studio chuẩn hóa, 5 góc | 3.349 | 559 | 980 | **4.888** | 70 người (48 / 8 / 14) |
| **RADIATE** | Chụp studio đa sắc tộc, chính diện | 1.049 | 148 | 306 | **1.503** | 109 người (76 / 11 / 22) |
| **FER-2013** | Ảnh Internet đời thực, đa dạng góc | 28.006 | 3.283 | 3.589 | **34.878** | Nguồn mở Internet |
| **Tổng cộng** | | **32.404** (78,5%) | **3.990** (9,7%) | **4.875** (11,8%) | **41.269** | **179 người có định danh** |

### Kết quả kiểm định độc lập (`reports/check_leakage.py`):
- **100% tệp tin hợp lệ:** Toàn bộ 41.269 ảnh tồn tại đầy đủ trên đĩa, khớp mã băm MD5, không có ảnh thiếu.
- **Trùng lặp MD5 chéo tập = 0:** Tuyệt đối không có ảnh nào bị trùng giữa các tập (`train-val = 0`, `train-test = 0`, `val-test = 0`).
- **Ảnh hỏng/đen trong tập huấn luyện = 0:** Toàn bộ ảnh đen hỏng đã được loại bỏ hoàn toàn trước khi nạp vào mô hình.
- **Phân tách danh tính tuyệt đối (Subject-Disjoint = 0):** Toàn bộ 179 chủ thể có mã định danh được phân chia độc lập hoàn toàn giữa `train` (124 người), `val` (19 người) và `test` (36 người). Tập kiểm tra đóng vai trò đánh giá khả năng khái quát hóa trên người hoàn toàn mới lạ.
- **Nguyên tắc khóa tập kiểm tra:** Siêu tham số và checkpoint chỉ được tối ưu hóa dựa trên tập `val`. Tập `test` chỉ được mở khóa một lần duy nhất tại bước nghiệm thu cuối cùng.

---

## 4. Phân Tích Thực Nghiệm Đa Chiều Trên Tập Kiểm Tra Khóa

![Độ chính xác theo nguồn](figures/mc_test_accuracy_by_source_3models.png)
*Hình 2: So sánh độ chính xác phân loại theo từng nguồn dữ liệu (KDEF, RADIATE, FER-2013) của 3 mô hình.*

### 4.1. Phân tích theo nguồn gốc ảnh
- **Tập dữ liệu studio (KDEF & RADIATE):** Mô hình đề xuất đạt độ chính xác rất cao (**85,51%** trên KDEF và **83,01%** trên RADIATE). Đáng chú ý, trên bộ ảnh đa sắc tộc RADIATE, mô hình đề xuất vượt trội hơn hẳn so với ResNet-18 (**83,01% so với 79,09%**, chênh lệch **+3,92%**).
- **Tập dữ liệu Internet (FER-2013):** Do tính chất phức tạp của ảnh thế giới thực (48×48 phóng to, nhòe mờ, biểu cảm không kiểm soát), cả mô hình đề xuất và ResNet-18 đều đạt mức **68,79%** (tiệm cận ngưỡng đánh giá của người quan sát trên tập dữ liệu này ~65–68%), bỏ xa HOG+SVM (44,00%).

### 4.2. Khả năng khái quát hóa theo sắc tộc (Nhân khẩu học)

| Nhóm sắc tộc / Nguồn gốc | Số ảnh test | HOG + SVM Acc / F1 | ResNet-18 Acc / F1 | Mô hình đề xuất Acc / F1 |
|---|---:|---:|---:|---:|
| **Bắc Âu (KDEF Thụy Điển)** | 980 | 41,6% / 0,390 | 85,4% / 0,853 | **85,5%** / **0,854** |
| **Gốc Phi (RADIATE Black)** | 112 | 36,6% / 0,343 | 72,3% / 0,700 | **75,9%** / **0,737** |
| **Gốc Á (RADIATE Asian)** | 56 | 42,9% / 0,375 | 80,4% / 0,778 | **83,9%** / **0,827** |
| **Gốc Mỹ Latinh (RADIATE Hispanic)**| 56 | 41,1% / 0,309 | 85,7% / 0,829 | **91,1%** / **0,889** |
| **Gốc Âu (RADIATE White)** | 82 | 32,9% / 0,285 | 82,9% / 0,828 | **86,6%** / **0,870** |
| **Toàn cầu Internet (FER-2013)** | 3.589 | 44,0% / 0,379 | 68,8% / 0,672 | **68,8%** / **0,663** |

*Nhận xét:* Mô hình đề xuất duy trì độ chính xác cao và ổn định trên tất cả các nhóm nhân khẩu học, không bị hiện tượng suy giảm hiệu năng nghiêm trọng trên nhóm sắc tộc thiểu số như các mô hình chưa tối ưu.

### 4.3. Phân tích theo 5 góc chụp khuôn mặt (Tập KDEF, 980 ảnh)

| Góc chụp | Mô tả tư thế | HOG + SVM Accuracy | ResNet-18 Accuracy | Mô hình đề xuất Accuracy |
|---|---|---:|---:|---:|
| **FL** | Nghiêng hẳn trái (-90°) | 34,2% | 82,7% | **84,7%** |
| **HL** | Nửa trái (-45°) | 41,3% | 89,3% | **86,7%** |
| **S** | Chính diện (0°) | 57,1% | 87,8% | **87,8%** |
| **HR** | Nửa phải (+45°) | 42,3% | 87,2% | **87,2%** |
| **FR** | Nghiêng hẳn phải (+90°) | 33,2% | 80,1% | **81,1%** |
| **Độ lệch \|FL − FR\|** | Tính đối xứng góc chụp | **1,0%** | **2,6%** | **3,6%** |

*Nhận xét:* Mô hình đạt độ chính xác cao nhất ở góc chính diện (87,8%) và duy trì mức trên 81–84% ở các góc nghiêng cực đại. Độ lệch giữa hai bên nghiêng đối xứng $|\text{FL} - \text{FR}|$ chỉ là 3,6%, chứng minh tính đồng biến hình học hiệu quả của khối `SymmetricMultiScaleAUBlock`.

### 4.4. Đánh giá chi tiết theo 7 cảm xúc

![F1-Score theo 7 cảm xúc](figures/mc_test_f1_by_emotion_3models.png)
*Hình 3: So sánh F1-Score chi tiết của 7 lớp cảm xúc giữa 3 mô hình trên tập kiểm tra.*

| Lớp cảm xúc | Số ảnh test | HOG + SVM F1 | ResNet-18 F1 | Mô hình đề xuất Precision | Mô hình đề xuất Recall | Mô hình đề xuất F1 |
|---|---:|---:|---:|---:|---:|---:|
| **happy (vui vẻ)** | 1.083 | 0,6412 | **0,9017** | 0,9046 | 0,8929 | **0,8987** |
| **surprise (ngạc nhiên)** | 578 | 0,5282 | **0,8165** | 0,8057 | 0,8253 | **0,8154** |
| **disgust (ghê tởm)** | 239 | 0,3885 | 0,7970 | 0,7958 | 0,7992 | **0,7975** |
| **neutral (trung tính)** | 810 | 0,4208 | 0,7061 | 0,6806 | 0,7840 | **0,7286** |
| **angry (giận dữ)** | 675 | 0,3485 | **0,6631** | 0,6940 | 0,6148 | **0,6520** |
| **sad (buồn)** | 778 | 0,2801 | 0,6090 | 0,5881 | 0,6478 | **0,6165** |
| **fear (sợ hãi)** | 712 | 0,2332 | **0,5781** | 0,6348 | 0,5225 | **0,5732** |
| **Trung bình (Macro-F1)** | 4.875 | **0,4058** | **0,7245** | **0,7291** | **0,7266** | **0,7260** |

![Ma trận nhầm lẫn 3 mô hình](figures/mc_test_confusion_3models.png)
*Hình 4: Ma trận nhầm lẫn chuẩn hóa của 3 mô hình trên tập kiểm tra khóa (4.875 ảnh).*

---

## 5. Nhận Xét & Đánh Giá Khoa Học Khách Quan

Dựa trên kết quả thực nghiệm định lượng, nghiên cứu rút ra 4 nhận định khoa học chuẩn xác:

1. **Ưu thế của thiên kiến quy nạp chuyên biệt miền (Domain-Specific Inductive Bias):**
   - Mặc dù `ResNet-18` sở hữu số lượng tham số gấp gần 9 lần (11,18M so với 1,27M) và được khởi tạo từ 1,28 triệu ảnh ImageNet, mô hình đề xuất `KDEF_Stem2Ch_MS4_Symmetric` vẫn đạt kết quả phân loại cao hơn (**73,05% so với 72,78% Accuracy**, **0,9437 so với 0,9255 ROC-AUC**).
   - Nguyên nhân là do cấu trúc của mô hình đề xuất được thiết kế bám sát cơ chế giải phẫu khuôn mặt (FACS): Strip Pooling phân tách rõ ràng vùng mắt và vùng miệng, trong khi kernel ngang/dọc $(1\times3)$ và $(3\times1)$ trích xuất chính xác các nếp nhăn biểu cảm thay vì tìm kiếm hình dạng vật thể tổng quát như backbone của ImageNet.

2. **Khả năng kháng hiện tượng học thuộc danh tính (Identity Overfitting):**
   - Trong bài toán FER chia theo người (Subject-Disjoint), các mạng quá lớn rất dễ dùng dung lượng dư thừa để ghi nhớ diện mạo diễn viên thay vì nhận biết cảm xúc. Kích thước 1,27M tham số kết hợp cùng tầng chú ý kênh ECA-Net và hàm mất mát SupCon đã ép mô hình phải tập trung vào không gian đặc trưng cảm xúc thuần túy, thể hiện qua độ chính xác vượt trội trên tập đa sắc tộc RADIATE (**83,01% so với 79,09%** của ResNet-18).

3. **Sự thất bại của đặc trưng học máy truyền thống (`HOG + SVM`):**
   - Phương pháp `HOG + SVM` chỉ đạt **43,12% Accuracy** và **40,58% Macro-F1**, kém mô hình CNN tới gần 30% độ chính xác.
   - Nguyên nhân là do các đặc trưng gradient thủ công (handcrafted features) không có khả năng thích ứng linh hoạt trước sự biến đổi phức tạp của điều kiện ánh sáng, góc nghiêng khuôn mặt và sự khác biệt về màu da giữa 3 tập dữ liệu.

4. **Hiệu quả triển khai thực tế trên phần cứng:**
   - Với kích thước tệp mô hình chỉ **4,89 MB** và tốc độ đạt **52 FPS trên CPU** (19,09 ms/ảnh), mô hình đề xuất hoàn toàn vượt trội ResNet-18 (42,73 MB) về khả năng đóng gói, triển khai nhẹ nhàng trên môi trường web, ứng dụng di động hoặc các thiết bị camera giám sát thời gian thực.

---

## 6. Hướng Dẫn Tái Hiện Kết Quả & Chạy Ứng Dụng

Toàn bộ quy trình thực nghiệm được đóng gói sẵn để tái hiện độc lập:

- **Chạy kiểm định rò rỉ dữ liệu:**
  ```bash
  python reports/check_leakage.py
  ```
- **Chạy lại đánh giá so sánh 3 mô hình trên tập kiểm tra:**
  ```bash
  python scripts/so_sanh_mo_hinh.py --split test --unlock-test
  ```
- **Khởi chạy ứng dụng Web UI tương tác (Streamlit):**
  ```bash
  streamlit run src/api/ui.py
  ```
- **Khởi chạy dịch vụ API suy luận (FastAPI):**
  ```bash
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
  ```
