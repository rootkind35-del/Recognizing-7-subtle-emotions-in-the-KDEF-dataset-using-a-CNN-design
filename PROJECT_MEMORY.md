# Project Memory: Hoc-May

## 1. Project Identity
- **Name**: Hoc-May (Nhận Dạng Cảm Xúc Khuôn Mặt — Facial Emotion Recognition on KDEF)
- **Type**: Machine Learning / Computer Vision Application
- **Architecture**: `KDEF_Stem2Ch_ECA_Strip` (`1,239,597` params — single highest model across notebook, inference, evaluation, and docs)
- **Branch**: `master`

## 2. Directory Map
- `notebooks/CNN_KDEF_FER.ipynb` — Unified end-to-end research, EDA, training, one-shot test evaluation, Grad-CAM, and activation audit notebook
- `data/` — KDEF dataset (`data/official-kdef-dataset-by-ksatrajit0`, ignored in git)
- `checkpoints/` — Production checkpoint (`final_model.pt`, ignored in git)
- `src/` — Model definition, inference, preprocessing, evaluation, and app/camera entrypoints
  - `src/api/` — FastAPI server (`app.py`), Streamlit UI (`ui.py`), frame & webcam recognizer (`inference.py`)
  - `src/data/` — Clean KDEF dataset & studio aspect crop + CLAHE transforms (`dataset.py`)
  - `src/models/` — Single canonical architecture (`model.py`: `ECALayer`, `KDEF_Stem2Ch_ECA_Strip`) & evaluator (`evaluate.py`)
  - `src/preprocessing/` — Face alignment + Studio Aspect Crop & CLAHE (`alignment.py`)
  - `src/utils/` — CUDA/CPU device helper (`device.py`)
- `scripts/verify_notebook.py` — Executed-notebook anti-leakage and widget/error verification utility
- `reports/` — Core reports (`EXPERIMENT_REPORT.md`, `data_analysis.md`, `papers_research.md`), `test_metrics.json`, `check_leakage.py`, and `figures/`

## 3. Conventions & Architecture Notes
- **Notebook-Only Training**: All EDA, preprocessing calibration, model training, and one-shot test evaluation live inside `notebooks/CNN_KDEF_FER.ipynb`. Outside Python modules cover only inference, evaluation, verification, and serving.
- **Anti-Leakage Rule**: `test/` (`980` images, `14` independent subjects) stays strictly locked until Section 8 (`MỞ KHÓA TẬP TEST`) of the notebook. The `10` corrupt MD5 `320b34032e2133fdfb80a58c953d5c94` images are excluded (`9` train, `1` val -> `3,349` train / `559` val / `980` test = `4,888` clean images).
- **Canonical Pipeline (`KDEF_Stem2Ch_ECA_Strip`, `Single-View`)**:
  - Input: Aspect-Aware Studio Crop `[0.04W:0.96W, 0.16H:0.86H]` -> Grayscale `cv2.INTER_AREA` `128x128` -> CLAHE `(clip=2.0, blend=0.8)` -> Normalize (`mean=0.4056757390499115`, `std=0.24074745178222656`).
  - Network: Parallel `3x3 + 5x5` Stem + 4 `SiLU` Stages (`32->64->128->256`) + `ECALayer(k=3)` in Stages 3 & 4 + `StripPool AdaptiveAvgPool2d((2, 1))` + `512->128->7` BN/Dropout head.
  - Training: `AdamW(lr=1e-3 -> 1e-6, wd=5e-4)`, `CrossEntropyLoss(label_smoothing=0.08)`, Warmup 5ep + Cosine 80ep, Flip Consistency KL (`0.4`), Curriculum Mixup (`alpha=0.2`, ep `6-65`), EMA (`0.996`), +20ep fine-tune check (`lr=1e-4`).

## 4. Verified Metrics (`Single-View`, 980 Test Images / 14 Subjects)
- **Validation Best (`final_model.pt`, 80ep + ep 5/20 FT)**: **91.37% Macro-F1 / 91.41% Accuracy**
- **Test Accuracy**: **86.22%** | **Macro-F1**: **86.29%** | **Weighted-F1**: **86.29%** | **ROC-AUC**: **0.9762**
- **Per-Class Recall**: `happy 100.00%`, `neutral 92.14%`, `sad 89.29%`, `disgust 86.43%`, `surprise 85.00%`, `fear 77.86%`, `angry 72.86%` (`recall_spread = 27.14%`)
- **Angle Symmetry (`Single-View`)**: `FL 86.73%`, `FR 84.18%`, `HL 87.76%`, `HR 88.27%`, `S 84.18%` (`|FL-FR| = 2.55%`, `|mean(HL,HR)-FR| = 3.83%`)
- **Activation Health**: `0.00%` dead neurons across all 8 `SiLU` layers (`stem_act1 .. stage4_act2`); `std(stage4_act2)/std(stem_act1) = 1.05x`.

## 5. Dev & App Commands
- **Verify notebook & leakage**: `python scripts/verify_notebook.py` & `python reports/check_leakage.py`
- **Evaluate checkpoint**: `python src/models/evaluate.py --checkpoint checkpoints/final_model.pt --split test`
- **Run API / UI / Webcam**: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000` | `streamlit run src/api/ui.py` | `python src/api/inference.py`
