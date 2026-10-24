# Project Memory: Hoc-May

## 1. Project Identity
- **Name**: Hoc-May (Nhận Dạng Cảm Xúc Khuôn Mặt - Facial Emotion Recognition)
- **Type**: Machine Learning / Computer Vision Application
- **Language**: Python
- **Branch**: master (Latest commit: 7fce5fb7)

## 2. Directory Map
- `data/` — Contains KDEF datasets (67,800+ image files)
- `src/` — Main source code directory
  - `src/api/` — FastAPI backend and real-time inference scripts
  - `src/data/` — Dataset loaders (PyTorch) and EDA scripts
  - `src/models/` — Neural network architectures, training, and evaluation scripts
  - `src/preprocessing/` — Face alignment and cropping logic (MediaPipe)
  - `src/utils/` — Helper utilities (e.g., GPU/Device management)

## 3. Key Files
- `requirements.txt` — Project dependencies
- `src/api/app.py` — FastAPI application entry point
- `src/api/inference.py` — Real-time webcam inference pipeline
- `src/models/train.py` — Model training script (PyTorch, ResNet18)
- `src/models/evaluate.py` — Model evaluation script (Classification metrics, Confusion Matrix)
- `src/utils/device.py` — CUDA/CPU device management utility

## 4. Tech Stack
- **Languages**: Python (3.x)
- **Core ML Frameworks**: PyTorch, Torchvision
- **Computer Vision**: OpenCV, MediaPipe
- **Data Science**: Scikit-learn, Pandas, Matplotlib, Seaborn
- **Backend/API**: FastAPI, Uvicorn
- **UI**: Streamlit (planned/supported)

## 5. Conventions
- **Code Style**: Python standard PEP-8 style.
- **Language**: Comments and Git commits are in Vietnamese (`Tiếng Việt`).
- **Imports**: Absolute imports are used (e.g., `from src.models.model import ...`). Entry point scripts modify `sys.path` to allow direct execution.
- **Encoding**: Stdout is explicitly reconfigured to `utf-8` on Windows (`sys.stdout.reconfigure(encoding='utf-8')`).

## 6. Important Patterns
- **Device Management**: Always use `src.utils.device.get_device()` to check and log GPU availability rather than hardcoding `torch.device("cuda")`.
- **Face Preprocessing**: Faces are aligned and cropped dynamically before passing to the CNN using `FaceAligner` in `src.preprocessing.alignment`.

## 7. Dev Commands
- **Install dependencies**: `pip install -r requirements.txt`
- **Chạy toàn bộ đề tài (khuyến nghị)**: `jupyter nbconvert --to notebook --execute --inplace notebooks/CNN_KDEF_FER.ipynb` — notebook tự cài gói, EDA, huấn luyện (train chỉ dùng `train/`, tinh chỉnh chỉ dùng `val/`), đánh giá test MỘT LẦN ở mục 9
- **Kiểm chứng tự động sau chạy**: `python scripts/verify_notebook.py` (kiểm tra khóa test + số liệu)
- **Kiểm tra rò rỉ dữ liệu**: `python reports/check_leakage.py`
- **Run EDA cũ**: `python src/data/eda.py`
- **Train model (script cũ)**: `python src/models/train.py`
- **Evaluate (script cũ)**: `python src/models/evaluate.py`
- **Run API**: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`
- **Run Webcam demo**: `python src/api/inference.py`

## 8. Kết quả chính (lần chạy GPU chính thức 2026-10-24)
- Notebook `notebooks/CNN_KDEF_FER.ipynb` (GPU RTX 4060 qua WSL2, ngân sách 120+20 epoch): **accuracy 82.76% / macro-F1 82.75%** trên tập test 980 ảnh (protocol subject-independent), vượt kỳ vọng tài liệu 65–80%.
- Bộ dữ liệu huấn luyện: `data/official-kdef-dataset-by-ksatrajit0` (ảnh KDEF gốc, 0 subject chéo split); bộ `balanced` bị loại vì là FER-2013 relabel + rò rỉ chéo split.
- Windows chặn torch CUDA bằng Smart App Control → huấn luyện GPU phải chạy trong WSL2 (venv `~/kdef_env`, dữ liệu chép sang ext4 để tránh lỗi 9p).

## 9. Last Updated
- 2026-10-24 (cập nhật sau lần chạy GPU 120+20 epoch)

