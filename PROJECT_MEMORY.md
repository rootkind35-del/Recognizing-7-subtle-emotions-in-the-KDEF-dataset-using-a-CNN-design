# Project Memory: Hoc-May

## 1. Project Identity
- **Name**: Hoc-May (Nhận Dạng Cảm Xúc Khuôn Mặt — Facial Emotion Recognition on KDEF)
- **Type**: Machine Learning / Computer Vision Application
- **Architecture**: `KDEF_Stem2Ch_MS4_Symmetric` (`stem2ch_ms4_fuse`, `1,273,005` params — single canonical model across notebook, inference, evaluation, and docs)
- **Branch**: `master`

## 2. Directory Map
- `notebooks/CNN_KDEF_FER.ipynb` — Unified end-to-end research, EDA, training, one-shot test evaluation, Grad-CAM, and activation audit notebook
- `data/` — KDEF dataset (`data/official-kdef-dataset-by-ksatrajit0`, ignored in git)
- `checkpoints/` — Production checkpoint (`final_model.pt`, ignored in git)
- `src/` — Model definition, inference, preprocessing, evaluation, and app/camera entrypoints
  - `src/api/` — FastAPI server (`app.py`), Streamlit UI (`ui.py`), frame & webcam recognizer (`inference.py`)
  - `src/data/` — Clean KDEF dataset & `ssd_pad_160_1ch` + CLAHE transforms (`dataset.py`)
  - `src/models/` — Single canonical architecture (`model.py`: `ECALayer`, `SymmetricMultiScaleAUBlock`, `FacialProjectionHead`, `KDEF_Stem2Ch_MS4_Symmetric`) & evaluator (`evaluate.py`)
  - `src/preprocessing/` — OpenCV Res10 SSD face detector + symmetric square-pad crop & CLAHE (`alignment.py`, `models/`)
  - `src/utils/` — CUDA/CPU device helper (`device.py`)
- `scripts/verify_notebook.py` — Executed-notebook anti-leakage and widget/error verification utility
- `reports/` — Core reports (`EXPERIMENT_REPORT.md`, `data_analysis.md`, `papers_research.md`), `test_metrics.json`, `check_leakage.py`, and `figures/`

## 3. Conventions & Architecture Notes
- **Notebook-Only Training**: All EDA, preprocessing calibration, model training, and one-shot test evaluation live inside `notebooks/CNN_KDEF_FER.ipynb`. Outside Python modules cover only inference, evaluation, verification, and serving.
- **Anti-Leakage Rule**: `test/` (`980` images, `14` independent subjects) stays strictly locked until Section 8 (`MỞ KHÓA TẬP TEST`) of the notebook. The `10` corrupt MD5 `320b34032e2133fdfb80a58c953d5c94` images are excluded (`9` train, `1` val -> `3,349` train / `559` val / `980` test = `4,888` clean images).
- **Canonical Pipeline (`KDEF_Stem2Ch_MS4_Symmetric`, `Single-View`)**:
  - Input (`ssd_pad_160_1ch`): OpenCV Res10 SSD face detection (`min_conf=0.20`, `KDEF_STUDIO_CROP_BOX` fallback) -> symmetric square-pad crop (`margin=1.25`, `cv2.BORDER_REPLICATE`) -> Grayscale `cv2.INTER_AREA` `160x160` -> CLAHE (`clip=2.0, blend=0.8`) -> Normalize (`mean=0.407255`, `std=0.209657`).
  - Network: Parallel `3x3 + 5x5` Stem -> Plain Stages 2–3 (`ECALayer` in Stage 3) -> Stage 4 `SymmetricMultiScaleAUBlock` (`3x3 + (1,3) + (3,1) + 3x3_d2` + `1x1` 320->256 fusion + `ECALayer`, Grad-CAM target `s4_ms.eca`) -> `StripPool AdaptiveAvgPool2d((2, 1))` -> `FacialProjectionHead(512->128->7, drop 0.45/0.30)`.
  - Training (`asam_focal_supcon_60ep`): `ASAM(rho=0.07)` + `AdamW(lr=1e-3 -> 1e-6, wd=5e-4, grad_clip=2.0)`, Warmup 4ep + Cosine 60ep, `SoftLabelSmoothedFocalCE(ls=0.06, gamma=1.0, poly_eps=0.3)` + `SupConLoss(w=0.20, T=0.08)` + `FlipConsistency(kl=0.40, feat=0.15)` + `FacialRandomErasing(p=0.30)` + `CutMix/Mixup/SplitMix` (ep `4..57`) + `EMA(decay=0.996)` (`best_ema` @ epoch `47`).

## 4. Verified Metrics (`Single-View`, 980 Test Images / 14 Subjects)
- **5-Fold Subject-Disjoint GroupKFold CV (56 subjects, `train+val`)**: **88.04% ± 2.13% Macro-F1 / 88.13% ± 2.22% Accuracy** (`|FL-FR|=2.43%`).
- **Validation (`final_model.pt`, `best_ema` @ ep 47/60, `fp32` Single-View)**: **92.99% Macro-F1 / 93.02% Accuracy**.
- **Test Accuracy**: **86.84%** | **Macro-F1**: **86.87%** | **Weighted-F1**: **86.87%** | **ROC-AUC**: **0.9843** *(Flip-TTA ref: 87.55% Acc / 87.55% Macro-F1)*.
- **Per-Class Recall**: `happy 97.86%`, `neutral 93.57%`, `surprise 90.71%`, `sad 87.86%`, `disgust 83.57%`, `fear 79.29%`, `angry 75.00%` (`recall_spread = 22.86%`).
- **Angle Symmetry (`Single-View`)**: `FL 85.71%`, `FR 83.16%`, `HL 90.82%`, `HR 88.27%`, `S 86.22%` (`|FL-FR| = 2.55%`, `pose_asymmetry_gap = 6.38%`).
- **Activation Health**: `0.00%` dead neurons across all 8 `SiLU` layers (`stem_act1 .. s4_ms.fuse.2`); `std(s4_ms.fuse.2)/std(stem_act1) = 1.47x` (`0.796 / 0.543`).

## 5. Dev & App Commands
- **Verify notebook & leakage**: `python scripts/verify_notebook.py` & `python reports/check_leakage.py`
- **Evaluate checkpoint**: `python src/models/evaluate.py --checkpoint checkpoints/final_model.pt --split test`
- **Run API / UI / Webcam**: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000` | `streamlit run src/api/ui.py` | `python src/api/inference.py`
