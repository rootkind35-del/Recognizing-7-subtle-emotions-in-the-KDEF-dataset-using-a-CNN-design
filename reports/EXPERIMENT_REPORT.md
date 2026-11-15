# Báo Cáo Thực Nghiệm: Nhận Dạng 7 Cảm Xúc Khuôn Mặt Trên Bộ Dữ Liệu KDEF (`KDEF_Stem2Ch_MS4_Symmetric`)

## 1. Tóm Tắt Dự Án

- **Mục tiêu:** Thiết kế, huấn luyện và triển khai mạng CNN thuần tự thiết kế nhận dạng 7 cảm xúc (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) trên bộ dữ liệu studio **KDEF** (*Karolinska Directed Emotional Faces* — Lundqvist, Flykt & Öhman, 1998) theo giao thức nghiêm ngặt **chia theo người (subject-disjoint)**.
- **Kiến trúc chốt & Checkpoint duy nhất:** **`KDEF_Stem2Ch_MS4_Symmetric`** (`stem2ch_ms4_fuse`, **1.273.005 tham số**) + tiền xử lý **`ssd_pad_160_1ch`** + công thức huấn luyện **`asam_focal_supcon_60ep`**, lưu tại [`checkpoints/final_model.pt`](../checkpoints/final_model.pt) và đồng bộ 100% giữa [`notebooks/CNN_KDEF_FER.ipynb`](../notebooks/CNN_KDEF_FER.ipynb), [`src/models/model.py`](../src/models/model.py), [`src/preprocessing/alignment.py`](../src/preprocessing/alignment.py), [`src/data/dataset.py`](../src/data/dataset.py) và [`src/api/inference.py`](../src/api/inference.py).
- **Kết quả xác thực đa phân hoạch 5-Fold Subject-Disjoint GroupKFold CV (56 người `train+val`, Single-View):**
  - **Macro-F1:** **0,8804 ± 0,0213** (`88,04%`)
  - **Accuracy:** **0,8813 ± 0,0222** (`88,13%`)
  - **Độ lệch bất biến góc mặt `|FL − FR|`:** **2,43%** (`0,0243`)
- **Kết quả chính thức trên tập kiểm tra độc lập 14 diễn viên (980 ảnh, suy luận đơn ảnh `Single-View`):**
  - **Accuracy:** **86,84%** (`0,8684`)
  - **Macro-F1:** **86,87%** (`0,8687`; Macro Precision **87,37%**, Macro Recall **86,84%**)
  - **Weighted-F1:** **86,87%** (`0,8687`)
  - **ROC-AUC (OvR macro):** **0,9843**
  - **Recall Spread (`max − min`):** **22,86%** (`0,2286`)

---

## 2. Dữ Liệu & Kỷ Luật Chống Rò Rỉ (Subject-Disjoint)

Nguồn dữ liệu sử dụng: `data/official-kdef-dataset-by-ksatrajit0` (chi tiết EDA xem tại [`reports/data_analysis.md`](data_analysis.md) và [`reports/check_leakage.py`](check_leakage.py)).

| Tập dữ liệu | Số người (Subject) | Ảnh gốc | Ảnh hỏng loại bỏ | Ảnh sạch sử dụng | Tỷ lệ |
|---|---:|---:|---:|---:|---:|
| **Train** | 48 | 3.358 | 9 | **3.349** | 68,51% |
| **Validation** | 8 | 560 | 1 | **559** | 11,44% |
| **Test (khóa)** | 14 | 980 | 0 | **980** | 20,05% |
| **Tổng cộng** | **70** | **4.898** | **10** | **4.888** | **100%** |

- **Kiểm định rò rỉ (`check_leakage.py` & `verify_notebook.py`):** **0** ảnh trùng MD5 giữa `train–test` và `val–test`; **0/70** diễn viên xuất hiện chéo nhiều split. 10 file đen hỏng trùng MD5 `320b34032e2133fdfb80a58c953d5c94` (9 ở `train`, 1 ở `val`) được tự động loại bỏ trước khi nạp RAM.
- **Mở khóa một lần:** Tập `test/` bị cấm truy cập tuyệt đối trong toàn bộ quá trình chuẩn bị dữ liệu, huấn luyện 60 epoch và lựa chọn mô hình trên `val/` + 5-fold GroupKFold CV; chỉ được mở khóa duy nhất một lần tại Mục 8 (`TEST_UNLOCKED = True`) của notebook để đo lường cuối.

---

## 3. Tiền Xử Lý `ssd_pad_160_1ch` & Kiến Trúc `KDEF_Stem2Ch_MS4_Symmetric`

### 3.1. Chuẩn hóa đầu vào khuôn mặt (`ssd_pad_160_1ch`)
1. **Grayscale 1 kênh:** Giảm phụ thuộc vào tông da và màu phông studio (Shin et al., 2016).
2. **SSD Square-Pad (`crop_mode="ssd_square_pad"`, `ssd_margin=1.25`):** Dò vùng khuôn mặt bằng OpenCV Res10 Caffe SSD (`min_conf=0.20`), mở rộng khung vuông đối xứng quanh tâm khuôn mặt (`side = max(w, h) × 1.25`) và điền biên bằng `cv2.BORDER_REPLICATE` để giữ khuôn mặt cân giữa trên cả 5 góc chụp (`FL`, `HL`, `S`, `HR`, `FR`) mà không bị méo tỷ lệ; tự động fallback về `KDEF_STUDIO_CROP_BOX = (0.04, 0.16, 0.96, 0.86)` nếu không dò thấy mặt.
3. **Downsampling `cv2.INTER_AREA` về `160×160`:** Bảo toàn chi tiết vi nếp nhăn vùng chân mày, khóe mắt và khóe miệng tốt hơn nội suy mặc định và cung cấp feature map Stage 4 kích thước `20×20`.
4. **CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`, `blend=0.8`):** Trộn 80% ảnh cân bằng thích ứng cục bộ với 20% ảnh gốc.
5. **Normalize:** Tính trên toàn bộ **3.349** ảnh `train/` sạch (`SEED=42`): **`mean = 0,407255`** (`0.4072545532585531`), **`std = 0,209657`** (`0.20965699302693627`).

### 3.2. Thiết kế mạng `KDEF_Stem2Ch_MS4_Symmetric` (`stem2ch_ms4_fuse`, 1.273.005 tham số)

| Thành phần | Cấu hình lớp | Kích thước đầu ra (`160×160`) | Vai trò kiến trúc | Tham số |
|---|---|---:|---|---:|
| **Multi-Scale Stem** | `Conv3×3(1→16)` ∥ `Conv5×5(1→16)` → `Concat(32)` → `BN + SiLU` → `Conv3×3(32→32) + BN + SiLU` → `MaxPool 2×2` | `32×80×80` | Kết hợp bộ lọc cạnh cục bộ (`3×3`) và cấu trúc cơ diện rộng (`5×5`) | `9.888` |
| **Stage 2** | `2 × [Conv3×3(32→64→64) + BN + SiLU]` → `MaxPool 2×2` | `64×40×40` | Đặc trưng vùng lông mày, mắt, rãnh mũi-má; kiến trúc tuần tự lọc nhiễu danh tính | `55.552` |
| **Stage 3 + ECA** | `2 × [Conv3×3(64→128→128) + BN + SiLU]` → **`ECALayer(k=3)`** → `MaxPool 2×2` | `128×20×20` | Tái hiệu chỉnh trọng số kênh không giảm chiều (3 tham số ECA) | `221.699` |
| **Stage 4 (`SymmetricMultiScaleAUBlock`)** | `Conv3×3(128→256) + BN + SiLU` → `[Conv3×3(96)` ∥ `Conv(1,3)(64)` ∥ `Conv(3,1)(64)` ∥ `Conv3×3_d2(96)]` (`320` kênh) → `Conv1×1(320→256) + BN + SiLU(inplace=False)` → **`ECALayer(k=3)`** | `256×20×20` | Đẳng biến lật ngang L/R tuyệt đối; nắm bắt đồng thời độ mở mày/môi `(1,3)`, rãnh dọc `(3,1)`, AU cục bộ & toàn cục (`d=2`); an toàn tuyệt đối cho Grad-CAM | `919.171` |
| **Strip Pooling & Projection Head** | `AdaptiveAvgPool2d((2, 1))` (`512`-d) → `Dropout(0.45)` → `Linear(512→128) + BN1d + SiLU` (`embed` L2-norm 128-d cho SupCon) → `Dropout(0.30)` → `Linear(128→7)` | `7` (và `128`-d) | Tách biệt nửa mặt trên (mắt/mày) & nửa mặt dưới (mũi/miệng), hỗ trợ SupCon trực tiếp | `66.695` |
| **Tổng cộng** | | | **100% khớp [`src/models/model.py`](../src/models/model.py)** | **1.273.005** |

---

## 4. Công Thức Huấn Luyện `asam_focal_supcon_60ep` & Độ Ổn Định Kích Hoạt

### 4.1. Siêu tham số & Kỹ thuật điều chuẩn
- **Phần cứng & Lịch học:** `cuda (NVIDIA GeForce RTX 4060 Laptop GPU)`, `batch_size = 32`, `epochs = 60` (`warmup = 4` epoch + Cosine Annealing `lr = 1e-3 → 1e-6`, `weight_decay = 5e-4`, `grad_clip = 2.0`).
- **Tối ưu hóa phẳng biên (ASAM):** `ASAM(rho = 0.07)` trên nền `AdamW` giúp tìm vùng cực tiểu phẳng, cải thiện khả năng tổng quát hóa sang diễn viên chưa từng thấy.
- **Hàm mất mát đa mục tiêu:**
  - **Soft Label-Smoothed Focal PolyLoss (`label_smoothing = 0.06`, `focal_gamma = 1.0`, `poly_eps = 0.3`):** Tập trung vào các mẫu ranh giới khó (`angry`, `fear`, `disgust`, `sad`) đồng thời tránh over-confidence trên nhãn gốc KDEF.
  - **Supervised Contrastive Loss (`SupConLoss`, `weight = 0.20`, `temp = 0.08`):** Kéo các biểu diễn 128-d chuẩn hóa L2 của cùng cảm xúc lại gần nhau bất kể khác biệt danh tính người mẫu.
  - **Flip Consistency (`kl_weight = 0.40`, `feat_weight = 0.15`):** Ép phân phối xác suất và vector đặc trưng giữa ảnh gốc và ảnh lật ngang đồng nhất trên các góc mặt nghiêng.
- **Augmentation GPU (`enhanced` + Occlusion + Mixing):** Lật ngang (`p=0.5`), xoay/affine nhẹ (`±10°`), độ sáng/tương phản, `FacialRandomErasing (p=0.30)`, và `CutMix/Mixup/SplitMix` (`mix_prob=0.50`, `mixup_alpha=0.25`, `cutmix_alpha=0.80`, `splitmix_prob=0.25`) áp dụng có chu trình từ epoch `4` đến `57`.
- **Exponential Moving Average (`decay = 0.996`):** Lựa chọn checkpoint `best_ema` theo `val` Macro-F1.
- **Kết quả Validation (559 ảnh sạch):** Đỉnh `best_ema` tại **epoch 47 / 60** đạt `val` Macro-F1 = **0,9282** (`Acc = 0,9284`, `bf16`); đánh giá lại toàn bộ `val/` ở `fp32` (`Single-View` & `Flip-TTA`) đạt **Acc = 0,9302 / Macro-F1 = 0,9299**, lưu vào [`checkpoints/final_model.pt`](../checkpoints/final_model.pt).

### 4.2. Phân tích kích hoạt trước và sau huấn luyện (`SiLU` vs Khởi tạo Kaiming)
Đo trên ảnh mẫu `AF01HAHR.JPG` (`happy`) qua **8 tầng kích hoạt chính** (`stem_act1 .. s4_ms.fuse.2`):
- **Tỷ lệ nơ-ron chết (`|a| < 1e-6`):** **`0,00%`** (`0.0`) ở toàn bộ 8/8 tầng cả trước và sau khi huấn luyện.
- **Độ lệch chuẩn `std` (Trước → Sau):** `stem_act1`: `1,031 → 0,543` | `stem_act2`: `1,277 → 0,451` | `stage2.2`: `1,078 → 0,440` | `stage2.5`: `1,221 → 0,439` | `s3_act1`: `1,227 → 0,452` | `s3_act2`: `0,864 → 0,291` | `s4_act1`: `0,841 → 0,510` | `s4_ms.fuse.2`: `0,610 → 0,796`.
- **Tỷ lệ phương sai cuối/đầu (`s4_ms.fuse.2 / stem_act1`):** **1,47x** (`0,796 / 0,543`), chứng tỏ tín hiệu và gradient lưu thông ổn định qua toàn mạng.

Biểu đồ huấn luyện và kích hoạt lưu tại: [`nb_final_learning_curves.png`](figures/nb_final_learning_curves.png), [`nb_post_feature_maps.png`](figures/nb_post_feature_maps.png), [`nb_post_activation_hists.png`](figures/nb_post_activation_hists.png), [`nb_post_conv1_filters.png`](figures/nb_post_conv1_filters.png).

---

## 5. Kết Quả Kiểm Định 5-Fold GroupKFold CV & Đánh Giá Tập Test Độc Lập

### 5.1. Độ ổn định đa phân hoạch (5-Fold Subject-Disjoint GroupKFold CV, 56 người `train+val`, 60 epoch)
Ghi nhận từ `.zcode/results/train_cv_stem2ch_ms4_fuse_ssd_pad_160_1ch_asam_focal_supcon_60ep_best_ema.json` (`Single-View`, `best_ema`):
- **Macro-F1 (5-Fold):** **0,8804 ± 0,0213** (từng fold: `0,8774 / 0,8959 / 0,8414 / 0,8851 / 0,9021`)
- **Accuracy (5-Fold):** **0,8813 ± 0,0222** (từng fold: `0,8802 / 0,8973 / 0,8401 / 0,8856 / 0,9034`)
- **Recall trung bình theo lớp (5-Fold):** `happy` **0,9620** | `surprise` **0,9311** | `neutral` **0,9302** | `disgust` **0,8943** | `angry` **0,8570** | `sad` **0,8144** | `fear` **0,7803**.
- **Chính xác theo góc mặt (5-Fold):** `FL` **0,8620** | `FR` **0,8782** | `HL` **0,8871** | `HR` **0,8951** | `S` **0,8840** (`|FL − FR| = 0,0243`).

### 5.2. Đánh giá một lần trên tập Test độc lập (980 ảnh, 14 diễn viên)
Dữ liệu chi tiết lưu trong [`reports/test_metrics.json`](test_metrics.json), đánh giá chuẩn bằng chế độ đơn ảnh **`Single-View`** (khớp 1:1 với pipeline camera/app):
- **Accuracy:** **86,84%** (`0,8684`) *(đối chiếu Flip-TTA: `87,55%` Acc / `0,8755` Macro-F1)*
- **Macro Precision / Recall / F1:** **0,8737 / 0,8684 / 0,8687**
- **Weighted-F1:** **0,8687**
- **ROC-AUC One-vs-Rest (Macro):** **0,9843** (`nb_test_roc.png`)
- **Recall Spread (`max − min`):** **22,86%** (`0,2286`)

### 5.3. Hiệu năng chi tiết theo 7 lớp cảm xúc (`test/`, Single-View)

| Cảm xúc | Precision | Recall | F1-score | Số ảnh đúng / tổng |
|---|---:|---:|---:|---:|
| `angry` | 0,9130 | 0,7500 (75,00%) | 0,8235 | 105 / 140 |
| `disgust` | 0,9000 | 0,8357 (83,57%) | 0,8667 | 117 / 140 |
| `fear` | 0,7817 | 0,7929 (79,29%) | 0,7872 | 111 / 140 |
| `happy` | **0,9928** | **0,9786 (97,86%)** | **0,9856** | **137 / 140** |
| `neutral` | 0,9097 | 0,9357 (93,57%) | 0,9225 | 131 / 140 |
| `sad` | 0,7365 | 0,8786 (87,86%) | 0,8013 | 123 / 140 |
| `surprise` | 0,8819 | 0,9071 (90,71%) | 0,8944 | 127 / 140 |

- **Các cặp nhầm lẫn cao nhất (`nb_test_confusion.png`, `nb_test_top_confusions.png`):** `disgust → sad` (**19** ảnh), `fear → surprise` (**15** ảnh), `surprise → fear` (**13** ảnh), `angry → fear` (**11** ảnh), `fear → sad` (**9** ảnh), `angry → sad` (**9** ảnh).
- **Grad-CAM (`nb_test_gradcam.png`):** Phản ứng từ tầng `s4_ms.eca` (`get_gradcam_target_layers()`) tập trung chính xác vào vùng khóa mắt/mày và khóe miệng trên từng biểu cảm ở kích thước đặc trưng `20×20`.

### 5.4. Phân tích theo góc chụp (Pose) & giới tính (Gender)
- **5 góc chụp (196 ảnh/góc, `Single-View`):**
  - `FL` (-45°): **85,71%** | `FR` (+45°): **83,16%** | `HL` (-15°): **90,82%** | `HR` (+15°): **88,27%** | `S` (0°): **86,22%** *(Flip-TTA: `FL 85,71% / FR 84,18% / HL 91,33% / HR 89,29% / S 87,24%`)*.
  - Độ lệch góc nghiêng đối xứng `|FL − FR|` = **2,55%**; độ lệch cực đại `|mean(HL, HR) − FR|` = **6,38%** (`6,12%` với Flip-TTA).
- **Giới tính (`nb_test_gender_angle.png`):**
  - Nam (`350` ảnh, 5 người): **92,00%**
  - Nữ (`630` ảnh, 9 người): **83,97%** (chênh lệch `8,03%` trên tập `test/` có phân bố không cân bằng `64,3%` ảnh nữ / `35,7%` ảnh nam).

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
| **Dự án (`KDEF_Stem2Ch_MS4_Symmetric`)** | **Subject-disjoint (48 / 8 / 14)** | **4.888 ảnh sạch (5 góc, 7 lớp)** | **CNN tự thiết kế 1,27M tham số (`Single-View`)** | **86,84% Acc / 0,8687 Macro-F1 / 0,9843 AUC** |

---

## 7. Hướng Dẫn Sử Dụng & Kiểm Thử Ngoài Notebook

Toàn bộ phần huấn luyện nằm gọn trong [`notebooks/CNN_KDEF_FER.ipynb`](../notebooks/CNN_KDEF_FER.ipynb). Các module Python bên ngoài phục vụ đánh giá, kiểm định chống rò rỉ và chạy ứng dụng/camera thời gian thực:

- **Kiểm định notebook & chống rò rỉ:**
  - `python scripts/verify_notebook.py`
  - `python reports/check_leakage.py`
- **Đánh giá lại tập test từ checkpoint:**
  - `python src/models/evaluate.py --checkpoint checkpoints/final_model.pt --split test`
- **Chạy API / Web UI / Camera trực tiếp:**
  - FastAPI server: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`
  - Streamlit UI: `streamlit run src/api/ui.py`
  - Webcam demo: `python src/api/inference.py`
