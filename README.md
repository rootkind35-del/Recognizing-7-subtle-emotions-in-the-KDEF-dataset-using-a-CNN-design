# Nhận Dạng Cảm Xúc Khuôn Mặt (Facial Emotion Recognition — KDEF)

Dự án nhận dạng 7 cảm xúc khuôn mặt (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) trên bộ dữ liệu **KDEF official** (`data/official-kdef-dataset-by-ksatrajit0`, chia nghiêm ngặt theo người **subject-disjoint**: **3.349 train / 559 val / 980 test** sau khi loại 10 khung đen hỏng MD5 `320b34032e2133fdfb80a58c953d5c94`) bằng kiến trúc CNN tự thiết kế **`KDEF_Stem2Ch_ECA_Strip`** (**1.239.597** tham số).

> **Quy chuẩn kho mã nguồn:** Toàn bộ quy trình EDA, tiền xử lý, huấn luyện 80 epoch + kiểm chứng fine-tune +20 epoch và đánh giá 1 lần trên tập test nằm gọn trong [`notebooks/CNN_KDEF_FER.ipynb`](notebooks/CNN_KDEF_FER.ipynb). Bên ngoài notebook chỉ giữ tài liệu, công cụ kiểm định chống rò rỉ và các module chạy đánh giá / API / Web UI / Camera.

## Kết Quả Chính Thức (`KDEF_Stem2Ch_ECA_Strip` — `Single-View`)

| Chỉ số | Tập Validation (559 ảnh, 8 người) | Tập Test khóa (980 ảnh, 14 người) |
|---|---|---|
| **Accuracy** | **91,41%** (`0.9141`) | **86,22%** (`0.8622`) |
| **Macro-F1** | **91,37%** (`0.9137`) | **86,29%** (`0.8629`) |
| **ROC-AUC (OvR Macro)** | — | **0,9762** |
| **Nơ-ron chết (`\|a\| < 1e-6`)** | — | **0,00%** trên cả 8 tầng `SiLU` |
| **Tỷ lệ `std` Stage 4 / Stem 1** | — | **1,05x** (`0.660 / 0.626`) |

*(Chi tiết phân lớp, ma trận nhầm lẫn, phân tích góc chụp/giới tính, Grad-CAM và đối chiếu tài liệu tham khảo xem tại [`reports/EXPERIMENT_REPORT.md`](reports/EXPERIMENT_REPORT.md) và [`reports/test_metrics.json`](reports/test_metrics.json)).*

## Cấu Trúc Dự Án

- [`notebooks/CNN_KDEF_FER.ipynb`](notebooks/CNN_KDEF_FER.ipynb): Notebook toàn trình duy nhất cho EDA, chuẩn hóa Aspect Crop `[0.04W:0.96W, 0.16H:0.86H]` + CLAHE, huấn luyện `KDEF_Stem2Ch_ECA_Strip` (`Warmup Cosine + Flip KL + Curriculum Mixup + EMA`), mở khóa test 1 lần, Grad-CAM & phân tích kích hoạt trước/sau huấn luyện.
- `src/models/`: [`model.py`](src/models/model.py) (định nghĩa kiến trúc `KDEF_Stem2Ch_ECA_Strip`, hàm dựng `build_model` & `load_kdef_checkpoint`) và [`evaluate.py`](src/models/evaluate.py) (đánh giá checkpoint chuẩn `Single-View` hoặc tùy chọn Flip-TTA).
- `src/preprocessing/` & `src/data/`: [`alignment.py`](src/preprocessing/alignment.py) (phát hiện mặt & Studio Aspect Crop + CLAHE) và [`dataset.py`](src/data/dataset.py) (nạp dữ liệu KDEF sạch, lọc MD5 hỏng).
- `src/api/`: [`app.py`](src/api/app.py) (FastAPI server), [`ui.py`](src/api/ui.py) (Streamlit web UI), [`inference.py`](src/api/inference.py) (`EmotionRecognizer` & demo webcam thời gian thực).
- `scripts/verify_notebook.py` & `reports/check_leakage.py`: Kiểm tra tự động anti-leakage, toàn vẹn cell output và không còn widget lỗi.
- `reports/`: [`EXPERIMENT_REPORT.md`](reports/EXPERIMENT_REPORT.md), [`test_metrics.json`](reports/test_metrics.json), [`data_analysis.md`](reports/data_analysis.md), [`papers_research.md`](reports/papers_research.md) và biểu đồ trong `reports/figures/`.

## Cài Đặt Môi Trường

```bash
python -m venv venv
source venv/Scripts/activate  # hoặc venv/bin/activate
pip install -r requirements.txt
```

## Hướng Dẫn Chạy Kiểm Thử, Ứng Dụng & Camera

- **Kiểm định toàn vẹn notebook & chống rò rỉ dữ liệu:**
  ```bash
  python scripts/verify_notebook.py
  python reports/check_leakage.py
  ```
- **Chạy đánh giá mô hình trên tập test từ checkpoint (`checkpoints/final_model.pt`):**
  ```bash
  python src/models/evaluate.py --checkpoint checkpoints/final_model.pt --split test
  ```
- **Chạy FastAPI Server nhận diện qua API:**
  ```bash
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
  ```
- **Chạy giao diện web tải ảnh (Streamlit):**
  ```bash
  streamlit run src/api/ui.py
  ```
- **Chạy demo Camera thời gian thực:**
  ```bash
  python src/api/inference.py
  ```
