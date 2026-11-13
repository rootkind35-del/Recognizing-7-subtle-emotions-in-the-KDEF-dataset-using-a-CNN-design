# Project Memory: Hoc-May

## 1. Project Identity
- **Name**: Hoc-May (Nhận Dạng Cảm Xúc Khuôn Mặt - Facial Emotion Recognition)
- **Type**: Machine Learning / Computer Vision Application
- **Language**: Python
- **Branch**: master

## 2. Directory Map
- `notebooks/CNN_KDEF_FER.ipynb` — Complete research, EDA, ablation, training, and one-shot test evaluation notebook
- `data/` — KDEF datasets (`data/official-kdef-dataset-by-ksatrajit0`, ignored in git)
- `checkpoints/` — Trained checkpoints (`final_model.pt`, `optimized_best_model.pt`, ignored in git)
- `src/` — Inference, preprocessing, model architecture, and app/camera modules
  - `src/api/` — FastAPI backend (`app.py`), Streamlit UI (`ui.py`), webcam/frame recognizer (`inference.py`)
  - `src/data/` — PyTorch datasets & deterministic studio aspect crop + CLAHE transforms (`dataset.py`)
  - `src/models/` — Network architectures (`model.py`: `KDEF_Stem2Ch_ECA_Strip`, `KDEFNet`, `KDEFResSENet`) & evaluation (`evaluate.py`)
  - `src/preprocessing/` — Face detection/alignment and studio crop + CLAHE preprocessing (`alignment.py`)
  - `src/utils/` — CUDA/CPU device helper (`device.py`)
- `scripts/verify_notebook.py` — Anti-leakage and executed-cell verification script
- `reports/` — Technical reports, metrics JSONs, visualizations (`figures/`), and `check_leakage.py`

## 3. Conventions & Architecture Notes
- **Notebook-Only Training**: All training, hyperparameter tuning, and model selection happen exclusively inside `notebooks/CNN_KDEF_FER.ipynb`. Outside scripts focus strictly on model inference, evaluation, verification, and serving (FastAPI/Streamlit/webcam).
- **Anti-Leakage Rule**: `test/` (980 images, 14 independent subjects) is strictly locked until Section 9 (`MỞ KHÓA TẬP TEST`). The 10 corrupt black MD5 `320b34032e2133fdfb80a58c953d5c94` images are excluded (9 train, 1 val -> 3,349 train / 559 val / 980 test = 4,888 clean images).
- **Winning Model (`KDEF_Stem2Ch_ECA_Strip`, 1,239,597 params)**: Parallel 3x3+5x5 stem, 4 SiLU stages, ECA-Net (`k=3`) in stages 3–4, Upper/Lower Facial Strip Pool `AdaptiveAvgPool2d((2,1))`, trained with Studio Aspect Crop `[0.16H:0.86H, 0.04W:0.96W]`, CLAHE `(clip=2.0, blend=0.8)`, Flip Consistency KL (`0.4`), Curriculum Mixup (`alpha=0.2`), and EMA (`0.996`).

## 4. Dev & App Commands
- **Install dependencies**: `pip install -r requirements.txt`
- **Execute full notebook (WSL2 GPU)**: `/home/dhp01/kdef_env/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 --ExecutePreprocessor.kernel_name=python3 notebooks/CNN_KDEF_FER.ipynb`
- **Verify notebook & anti-leakage**: `python scripts/verify_notebook.py` & `python reports/check_leakage.py`
- **Evaluate checkpoint on test split**: `python src/models/evaluate.py --checkpoint checkpoints/final_model.pt --split test --tta_mode ms4`
- **Run API**: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`
- **Run Streamlit UI**: `streamlit run src/api/ui.py`
- **Run Webcam demo**: `python src/api/inference.py`

## 5. Latest Verified Results (Phase 2 Deep Upgrade, Subject-Disjoint 980 Test Images)
- **MS4-TTA (Official Mode)**: **85.82% Accuracy / 85.88% Macro-F1** (ROC-AUC 0.9771)
- **Single-View**: **86.22% Accuracy / 86.29% Macro-F1** | **Flip-TTA**: **85.20% Accuracy / 85.26% Macro-F1**
- **Validation Best**: **0.9068** (30ep Grid 4) -> **0.9134** (80ep) -> **0.9137** (5ep fine-tune)
- **Discrepancy Eliminated**: `exact_dead_frac = 0.0000` across all 8 activation layers (`stem_act1`..`stage4_act2`), cross-stage `std(stage4_act2)/std(stem_act1) = 1.05x`, pose asymmetry `|FL - FR| = 2.04%` (Single-View).
