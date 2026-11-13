# Nhận Dạng Cảm Xúc Khuôn Mặt (Facial Emotion Recognition — KDEF)

Dự án nhận dạng 7 cảm xúc khuôn mặt trên bộ dữ liệu **KDEF official** (`data/official-kdef-dataset-by-ksatrajit0`, chia **subject-disjoint**: 3.349 train / 559 val / 980 test, loại 10 khung đen hỏng MD5 `320b34032e2133fdfb80a58c953d5c94`) bằng mạng CNN tự thiết kế **`KDEF_Stem2Ch_ECA_Strip`** (1.239.597 tham số).

> **Quy chuẩn kho mã nguồn:** Toàn bộ quy trình EDA, huấn luyện, so sánh ablation và đánh giá 1 lần trên tập test nằm gọn trong [`notebooks/CNN_KDEF_FER.ipynb`](notebooks/CNN_KDEF_FER.ipynb). Bên ngoài chỉ giữ tài liệu/báo cáo, công cụ kiểm định anti-leakage và ứng dụng chạy API / UI / Camera / Đánh giá mô hình.

## Cấu Trúc Dự Án

- [`notebooks/CNN_KDEF_FER.ipynb`](notebooks/CNN_KDEF_FER.ipynb): Sổ tay chính (EDA, tiền xử lý Aspect Crop + CLAHE, kiến trúc `KDEF_Stem2Ch_ECA_Strip`, Flip Consistency KL + Curriculum Mixup + EMA, mở khóa test 1 lần, Grad-CAM & phân tích kích hoạt).
- `src/models/`: [`model.py`](src/models/model.py) (kiến trúc `KDEF_Stem2Ch_ECA_Strip`, `KDEFNet`, `KDEFResSENet`, nạp checkpoint) và [`evaluate.py`](src/models/evaluate.py) (đánh giá Single-View / Flip-TTA / MS4-TTA).
- `src/data/` & `src/preprocessing/`: [`dataset.py`](src/data/dataset.py) và [`alignment.py`](src/preprocessing/alignment.py) (lọc ảnh hỏng MD5, Studio Aspect Crop `[0.16H:0.86H, 0.04W:0.96W]` + CLAHE).
- `src/api/`: [`app.py`](src/api/app.py) (FastAPI server), [`ui.py`](src/api/ui.py) (Streamlit UI tải ảnh), [`inference.py`](src/api/inference.py) (suy luận & demo webcam trực tiếp).
- `scripts/verify_notebook.py`: Công cụ kiểm tra notebook sau khi chạy (0 lỗi cell, 0 widget hỏng, 0 rò rỉ tập `test` trước mục 9).
- `reports/`: Kết quả kiểm thử ([`test_metrics.json`](reports/test_metrics.json), [`optimization_comparison.md`](reports/optimization_comparison.md), [`deep_data_discrepancy_research.md`](reports/deep_data_discrepancy_research.md), [`papers_research.md`](reports/papers_research.md)) và [`check_leakage.py`](reports/check_leakage.py).

## Cài Đặt Môi Trường

```bash
python -m venv venv
source venv/Scripts/activate  # hoặc venv/bin/activate
pip install -r requirements.txt
```

## Hướng Dẫn Chạy Ứng Dụng, Camera & Kiểm Thử Mô Hình

- **Chạy kiểm định notebook & chống rò rỉ:**
  ```bash
  python scripts/verify_notebook.py
  python reports/check_leakage.py
  ```
- **Chạy đánh giá mô hình trên tập test (`checkpoints/final_model.pt`):**
  ```bash
  python src/models/evaluate.py --checkpoint checkpoints/final_model.pt --split test --tta_mode ms4
  ```
- **Chạy API nhận diện ảnh (FastAPI):**
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

## Kết Quả Mới Nhất (`KDEF_Stem2Ch_ECA_Strip`, Tập Test 980 Ảnh)

- **MS4-TTA (Chế độ chính):** **85.82% Accuracy / 85.88% Macro-F1** (Single-View: **86.22% Accuracy / 86.29% Macro-F1**)
- **Sức khỏe kích hoạt:** `0.0000` nơ-ron chết tuyệt đối trên toàn bộ 8 lớp kích hoạt; tỷ lệ phương sai Stage 4 / Stage 1 = **1.05x** (triệt tiêu hiện tượng bùng nổ 138x–677x của ReLU baseline).
