# Báo Cáo Thực Nghiệm: Nhận Dạng 7 Cảm Xúc Khuôn Mặt Trên Bộ Dữ Liệu KDEF (`KDEF_Stem2Ch_ECA_Strip`)

## 1. Tóm Tắt Dự Án

- **Mục tiêu:** Thiết kế, huấn luyện và triển khai mạng CNN thuần tự thiết kế nhận dạng 7 cảm xúc (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) trên bộ dữ liệu studio **KDEF** (*Karolinska Directed Emotional Faces* — Lundqvist, Flykt & Öhman, 1998) theo giao thức nghiêm ngặt **chia theo người (subject-disjoint)**.
- **Kiến trúc chốt & Checkpoint duy nhất:** **`KDEF_Stem2Ch_ECA_Strip`** (**1.239.597 tham số**) tại [`checkpoints/final_model.pt`](../checkpoints/final_model.pt), đồng bộ 100% giữa [`notebooks/CNN_KDEF_FER.ipynb`](../notebooks/CNN_KDEF_FER.ipynb) và [`src/models/model.py`](../src/models/model.py).
- **Kết quả chính thức trên tập kiểm tra độc lập 14 diễn viên (980 ảnh, suy luận đơn ảnh `Single-View`):**
  - **Accuracy:** **86,22%**
  - **Macro-F1:** **86,29%** (Macro Precision **87,05%**, Macro Recall **86,22%**)
  - **Weighted-F1:** **86,29%**
  - **ROC-AUC (OvR macro):** **0,9762**

---

## 2. Dữ Liệu & Kỷ Luật Chống Rò Rỉ (Subject-Disjoint)

Nguồn dữ liệu sử dụng: `data/official-kdef-dataset-by-ksatrajit0` (chi tiết EDA xem tại [`reports/data_analysis.md`](data_analysis.md) và [`reports/check_leakage.py`](check_leakage.py)).

| Tập dữ liệu | Số người (Subject) | Ảnh gốc | Ảnh hỏng loại bỏ | Ảnh sạch sử dụng | Tỷ lệ |
|---|---:|---:|---:|---:|---:|
| **Train** | 48 | 3.358 | 9 | **3.349** | 68,51% |
| **Validation** | 8 | 560 | 1 | **559** | 11,44% |
| **Test (khóa)** | 14 | 980 | 0 | **980** | 20,05% |
| **Tổng cộng** | **70** | **4.898** | **10** | **4.888** | **100%** |

- **Kiểm định rò rỉ (`check_leakage.py` & `verify_notebook.py`):** **0** ảnh trùng MD5 giữa `train–test` và `val–test`; **0/70** diễn viên xuất hiện chéo nhiều split. 10 file đen hỏng trùng MD5 `320b34032e2133fdfb80a58c953d5c94` (9 ở train, 1 ở val) được tự động loại bỏ trước khi nạp RAM.
- **Mở khóa một lần:** Tập `test/` bị cấm truy cập tuyệt đối trong toàn bộ quá trình chuẩn bị dữ liệu, huấn luyện 80 epoch và kiểm chứng fine-tune +20 epoch; chỉ được mở khóa duy nhất một lần tại Mục 8 của notebook để đo lường cuối.

---

## 3. Tiền Xử Lý Chuẩn Hóa & Kiến Trúc `KDEF_Stem2Ch_ECA_Strip`

### 3.1. Chuẩn hóa đầu vào ảnh khuôn mặt
1. **Grayscale 1 kênh:** Giảm phụ thuộc vào tông da và màu phông studio (Shin et al., 2016).
2. **Aspect-Aware Studio Crop `[0.04W:0.96W, 0.16H:0.86H]`:** Loại bỏ viền nền trống và phần vai áo trên ảnh KDEF gốc `562×762`, đưa vùng đầu–cằm về khung hình tỷ lệ chuẩn `1:1` không bị méo khi thu nhỏ.
3. **Downsampling `cv2.INTER_AREA` về `128×128`:** Giảm răng cưa mượt hơn nội suy mặc định; cỡ ảnh `128×128` khớp điểm tối ưu cho CNN tự thiết kế trên KDEF (Akhand et al., 2021).
4. **CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`, `blend=0.8`):** Trộn 80% ảnh cân bằng thích ứng cục bộ với 20% ảnh gốc để làm nổi vi nếp nhăn vùng chân mày, khóe mắt và miệng mà không khuếch đại nhiễu nền.
5. **Normalize:** Tính trực tiếp từ 560 ảnh `train` sạch (`SEED=42`): **`mean = 0.4057`** (`0.4056757390499115`), **`std = 0.2407`** (`0.24074745178222656`).

### 3.2. Thiết kế mạng `KDEF_Stem2Ch_ECA_Strip` (1.239.597 tham số)

| Thành phần | Cấu hình lớp | Kích thước đầu ra | Vai trò kiến trúc | Tham số |
|---|---|---:|---|---:|
| **Multi-Scale Stem** | `Conv3×3(1→16)` ∥ `Conv5×5(1→16)` → `Concat(32)` → `BN + SiLU` → `Conv3×3(32→32) + BN + SiLU` → `MaxPool 2×2` | `32×64×64` | Kết hợp bộ lọc cạnh cục bộ (`3×3`) và cấu trúc cơ diện rộng (`5×5`) | `9.888` |
| **Stage 2** | `2 × [Conv3×3(32→64→64) + BN + SiLU]` → `MaxPool 2×2` | `64×32×32` | Đặc trưng vùng lông mày, mắt, rãnh mũi-má | `55.552` |
| **Stage 3 + ECA** | `2 × [Conv3×3(64→128→128) + BN + SiLU]` → **`ECALayer(k=3)`** → `MaxPool 2×2` | `128×16×16` | Tái hiệu chỉnh trọng số kênh không giảm chiều (3 tham số ECA) | `221.699` |
| **Stage 4 + ECA** | `2 × [Conv3×3(128→256→256) + BN + SiLU]` → **`ECALayer(k=3)`** | `256×16×16` | Giữ nguyên độ phân giải `16×16` cho Action Units & Grad-CAM | `885.763` |
| **Strip Pooling & Head** | `AdaptiveAvgPool2d((2, 1))` (`512`-d) → `Dropout(0.4)` → `Linear(512→128) + BN1d + SiLU` → `Dropout(0.3)` → `Linear(128→7)` | `7` | Tách đặc trưng nửa mặt trên (mắt/mày) & nửa mặt dưới (mũi/miệng) | `66.695` |
| **Tổng cộng** | | | **100% khớp [`src/models/model.py`](../src/models/model.py)** | **1.239.597** |

---

## 4. Công Thức Huấn Luyện & Độ Ổn Định Kích Hoạt

### 4.1. Siêu tham số & Kỹ thuật điều chuẩn
- **Phần cứng & Thời gian chạy:** `cuda (NVIDIA GeForce RTX 4060 Laptop GPU)`, `batch_size = 32`, `epochs = 100` (80 epoch Cosine + kiểm chứng 20 epoch fine-tune).
- **Optimizer & Scheduler:** `AdamW(lr=1e-3 → 1e-6, weight_decay=5e-4)`, Linear Warmup 5 epoch + Cosine Annealing 80 epoch (`CrossEntropyLoss` với `label_smoothing=0.08`).
- **Augmentation GPU (`enhanced`):** Lật ngang (`p=0.5`), xoay `±10°`, scale `0.95–1.05`, tịnh tiến `±4%`, điều chỉnh sáng/tương phản (`gain 0.92–1.08`, `bias ±0.04`).
- **Điều chuẩn nâng cao:**
  - **Flip Consistency KL (`weight=0.4`):** Ép phân phối xác suất giữa ảnh gốc và ảnh lật ngang hội tụ nhất quán, tăng độ bền vững với góc mặt nghiêng trái/phải.
  - **Curriculum Mixup (`alpha=0.2`, epoch `6–65`):** Pha trộn tuyến tính trong giai đoạn học biểu diễn giữa chu kỳ, rồi tắt ở giai đoạn cuối để mô hình tinh chỉnh biên phân lớp sắc nét.
  - **Exponential Moving Average (`decay=0.996`):** Duy trì trọng số trung bình trượt dùng khi đánh giá `val` và lưu checkpoint.
- **Kết quả Validation (559 ảnh):** Giai đoạn 80 epoch đạt `val` macro-F1 = **0,9134** (`val` accuracy = **0,9141**) tại epoch 42; kiểm chứng `+20` epoch (`lr=1e-4`) đạt đỉnh **0,9137** (`val` accuracy = **0,9141**) tại epoch 5 và được chọn lưu vào [`checkpoints/final_model.pt`](../checkpoints/final_model.pt).

### 4.2. Phân tích kích hoạt trước và sau huấn luyện (`SiLU` vs Khởi tạo Kaiming)
Đo trên ảnh mẫu `AF01HAHR.JPG` (`happy`) qua 8 tầng kích hoạt chính `stem_act1 .. stage4_act2`:
- **Tỷ lệ nơ-ron chết (`|a| < 1e-6`):** **`0,00%`** (`0.0`) ở cả 8/8 tầng cả trước và sau khi huấn luyện.
- **Độ lệch chuẩn `std` (Trước → Sau):** `stem_act1`: `0,852 → 0,626` | `stem_act2`: `0,775 → 0,499` | `stage2.2`: `0,694 → 0,573` | `stage2.5`: `0,683 → 0,476` | `stage3.2`: `0,647 → 0,491` | `stage3.5`: `0,543 → 0,469` | `stage4_act1`: `0,263 → 0,464` | `stage4_act2`: `0,189 → 0,660`.
- **Tỷ lệ phương sai cuối/đầu (`stage4_act2 / stem_act1`):** **1,05x** (`0,660 / 0,626`), hoàn toàn ổn định không bùng nổ gradient hay suy giảm tín hiệu.

Biểu đồ huấn luyện và kích hoạt lưu tại: [`nb_final_learning_curves.png`](figures/nb_final_learning_curves.png), [`nb_finetune_learning_curves.png`](figures/nb_finetune_learning_curves.png), [`nb_post_activation_hists.png`](figures/nb_post_activation_hists.png), [`nb_post_conv1_filters.png`](figures/nb_post_conv1_filters.png).

---

## 5. Kết Quả Đánh Giá Tập Test Độc Lập (980 Ảnh, 14 Người)

Dữ liệu chi tiết lưu trong [`reports/test_metrics.json`](test_metrics.json), đánh giá chuẩn bằng chế độ đơn ảnh **`Single-View`** (khớp pipeline triển khai camera/app):

### 5.1. Chỉ số tổng quan
- **Accuracy:** **86,22%** (`0.8622`) *(so với Flip-TTA đối chiếu: `85,20%` Acc / `85,26%` Macro-F1)*
- **Macro Precision / Recall / F1:** **0,8705 / 0,8622 / 0,8629**
- **Weighted Precision / Recall / F1:** **0,8705 / 0,8622 / 0,8629**
- **ROC-AUC One-vs-Rest (Macro):** **0,9762** (`nb_test_roc.png`)
- **Recall Spread (`max − min`):** **27,14%** (`0,2714`)

### 5.2. Hiệu năng chi tiết theo 7 lớp cảm xúc

| Cảm xúc | Precision | Recall | F1-score | Số ảnh đúng / tổng |
|---|---:|---:|---:|---:|
| `angry` | 0,9358 | 0,7286 (72,86%) | 0,8193 | 102 / 140 |
| `disgust` | 0,8897 | 0,8643 (86,43%) | 0,8768 | 121 / 140 |
| `fear` | 0,7842 | 0,7786 (77,86%) | 0,7814 | 109 / 140 |
| `happy` | **1,0000** | **1,0000 (100,00%)** | **1,0000** | **140 / 140** |
| `neutral` | 0,8958 | 0,9214 (92,14%) | 0,9085 | 129 / 140 |
| `sad` | 0,7062 | 0,8929 (89,29%) | 0,7886 | 125 / 140 |
| `surprise` | 0,8815 | 0,8500 (85,00%) | 0,8655 | 119 / 140 |

- **Các cặp nhầm lẫn cao nhất (`nb_test_confusion.png`, `nb_test_top_confusions.png`):** `surprise → fear` (**16** ảnh), `disgust → sad` (**16** ảnh), `fear → sad` (**14** ảnh), `fear → surprise` (**12** ảnh), `angry → sad` (**10** ảnh), `angry → disgust` (**9** ảnh).
- **Grad-CAM (`nb_test_gradcam.png`):** Phản ứng tập trung đúng vùng khóa miệng/mắt-mày đặc trưng trên từng biểu cảm (`stage4_act2`).

### 5.3. Phân tích theo góc chụp (Pose) & giới tính (Gender)
- **5 góc chụp (196 ảnh/góc, `Single-View`):**
  - `FL` (-45°): **86,73%** | `FR` (+45°): **84,18%** | `HL` (-15°): **87,76%** | `HR` (+15°): **88,27%** | `S` (0°): **84,18%**
  - Độ lệch góc nghiêng đối xứng `|FL − FR|` = **2,55%**; độ lệch tối đa `|mean(HL, HR) − FR|` = **3,83%** (`0.0383`).
- **Giới tính (`nb_test_gender_angle.png`):**
  - Nam (`350` ảnh, 5 người): **93,71%**
  - Nữ (`630` ảnh, 9 người): **82,06%** (chênh lệch `11,65%`, phản ánh sự phân bố lệch 64,3% nữ của 14 diễn viên trong split `test/`).

---

## 6. Đối Chiếu Tài Liệu Nghiên Cứu KDEF

Số liệu trích xuất xác thực từ [`reports/papers_research.md`](papers_research.md):

| Nghiên cứu | Protocol | Cấu hình dữ liệu KDEF | Kiến trúc | Độ chính xác |
|---|---|---|---|---|
| **Cha et al. (2026)** | **LOSO subject-independent** | 840 ảnh chính diện, 6 lớp | CNN thuần (`LeNet-5`, `ResNet-18`, `MobileNetV2`) / `ResNet-152` | 60–78% (ResNet-152: 59,9%) |
| **Cha et al. (2026)** | **LOSO subject-independent** | 840 ảnh chính diện, 6 lớp | `CNN + handcrafted fusion` | **77–80%** (Macro-F1 **0,76–0,79**) |
| **Akhand et al. (2021)** | 10% random *(subject-dependent)* | 4.900 ảnh 5 góc, 7 lớp | CNN 2 tầng (`48` / `64` / **`128`** / `224`) | 61,6% / 69,4% / **80,8%** / 73,5% |
| **Akhand et al. (2021)** | 10% random *(subject-dependent)* | 4.900 ảnh 5 góc, 7 lớp | `VGG-16` từ đầu / Transfer `VGG-16` | 23,4% / 96,5% |
| **Shin et al. (2016)** | Cross-database (`FER2013 → KDEF`) | 2.961 ảnh KDEF | CNN 3 tầng + Histogram Eq. | 50,7% |
| **Dự án (`KDEF_Stem2Ch_ECA_Strip`)** | **Subject-disjoint (48 / 8 / 14)** | **4.888 ảnh sạch (5 góc, 7 lớp)** | **CNN tự thiết kế 1,24M tham số (`Single-View`)** | **86,22% Acc / 0,8629 Macro-F1 / 0,9762 AUC** |

---

## 7. Hướng Dẫn Sử Dụng & Kiểm Thử Ngoài Notebook

Toàn bộ phần huấn luyện nằm gọn trong [`notebooks/CNN_KDEF_FER.ipynb`](../notebooks/CNN_KDEF_FER.ipynb). Các module Python bên ngoài chỉ phục vụ đánh giá, kiểm định chống rò rỉ và chạy ứng dụng/camera thời gian thực:

- **Kiểm định notebook & chống rò rỉ:**
  - `python scripts/verify_notebook.py`
  - `python reports/check_leakage.py`
- **Đánh giá lại tập test từ checkpoint:**
  - `python src/models/evaluate.py --checkpoint checkpoints/final_model.pt --split test`
- **Chạy API / Web UI / Camera trực tiếp:**
  - FastAPI server: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`
  - Streamlit UI: `streamlit run src/api/ui.py`
  - Webcam demo: `python src/api/inference.py`
