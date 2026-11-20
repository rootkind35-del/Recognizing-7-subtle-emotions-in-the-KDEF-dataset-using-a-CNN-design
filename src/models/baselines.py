"""
Baseline Models & Evaluation Suite for Multicultural Facial Emotion Recognition (Milestone M3).

Implements the two benchmark architectures from `Nội dung sườn cho dự án.pdf` Section 4.3:
  1. `HOG + SVM` (Dalal & Triggs, 2005 + Cortes & Vapnik, 1995):
     - OpenCV `cv2.HOGDescriptor` feature extraction + `StandardScaler` + `LinearSVC` (LIBLINEAR)
     - Calibrated probabilities (`CalibratedClassifierCV` / sigmoid Platt scaling or softmax decision scores)
       for ROC-AUC, Accuracy, Macro-F1, Weighted-F1, and per-emotion/per-source breakdowns.
     - Hyperparameter search strictly on `val` split (`img_size`, `cell_size`, `nbins`, `C`, `class_weight`).
  2. `ResNet-18` Transfer Learning (`torchvision.models.resnet18` with `ResNet18_Weights.IMAGENET1K_V1`):
     - Pretrained on ImageNet-1K, adapted to the 3-source multicultural input pipeline (grayscale repeated
       to 3 channels + ImageNet normalization, with 512 -> 7 classification head).
     - Class imbalance handling via inverse-frequency balanced class weights + label smoothing / focal / EMA.
     - Hyperparameter search strictly on `val` split (`lr`, `weight_decay`, freeze/unfreeze schedule, `img_size`).

STRICT ANTI-LEAKAGE RULE:
- Hyperparameter tuning and checkpoint selection use ONLY `train` (32,404 images) and `val` (3,990 images)
  from `data/manifests/multicultural_manifest.csv`.
- The `test` split (4,875 images) is strictly locked during all training and tuning functions.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tv_models

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import (  # noqa: E402
    ANGLE_NAMES,
    CLASS_NAMES,
    CLASS_TO_IDX,
    DEFAULT_MULTICULTURAL_MANIFEST,
    DEFAULT_MULTICULTURAL_MEAN,
    DEFAULT_MULTICULTURAL_STD,
    compute_class_weights,
)
from src.preprocessing.alignment import (  # noqa: E402
    DEFAULT_CROP_MODE,
    DEFAULT_SSD_MARGIN,
    KDEF_STUDIO_CROP_BOX,
    _preprocess_to_uint8,
)

# ImageNet canonical normalization constants (for 3-channel ResNet-18 transfer learning)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CACHE_DIR = REPO_ROOT / ".zcode" / "cache"


def seed_everything(seed: int = 42) -> None:
    """Fixes all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_not_test_split(split: str) -> None:
    """Hard guard preventing accidental use of `test` split during training or tuning."""
    if str(split).strip().lower() == "test":
        raise RuntimeError(
            "ANTI-LEAKAGE VIOLATION: `test` split is strictly locked during baseline training and tuning!"
        )


# ---------------------------------------------------------------------------
# Fast Cached Multicultural Split Loader (uint8 in-memory cache)
# ---------------------------------------------------------------------------

def load_multicultural_split_cache(
    split: str,
    img_size: int = 160,
    manifest_path: Optional[Union[str, Path]] = None,
    use_crop: bool = True,
    crop_mode: str = DEFAULT_CROP_MODE,
    ssd_margin: float = DEFAULT_SSD_MARGIN,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_blend: float = 0.8,
    rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads (or builds and caches) uint8 grayscale images `(N, img_size, img_size)`,
    integer labels `(N,)`, and metadata DataFrame for a given split (`train`, `val`, or `test`)
    from `data/manifests/multicultural_manifest.csv`.
    """
    split_norm = split.strip().lower()
    m_path = Path(manifest_path) if manifest_path is not None else DEFAULT_MULTICULTURAL_MANIFEST
    if not m_path.is_absolute():
        m_path = REPO_ROOT / m_path
    if not m_path.exists():
        raise FileNotFoundError(f"Multicultural manifest not found: {m_path}")

    df_all = pd.read_csv(m_path)
    df = df_all[df_all["split"].astype(str).str.strip().str.lower() == split_norm].copy()
    df.reset_index(drop=True, inplace=True)
    if len(df) == 0:
        raise ValueError(f"No rows found for split='{split_norm}' in {m_path}")

    clahe_tag = f"clahe{int(use_clahe)}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw_img_cache = CACHE_DIR / f"mc_{split_norm}_{crop_mode}_{img_size}_{clahe_tag}_imgs.npy"
    raw_lbl_cache = CACHE_DIR / f"mc_{split_norm}_{crop_mode}_{img_size}_{clahe_tag}_lbls.npy"
    cache_file = CACHE_DIR / f"mc_{split_norm}_{crop_mode}_{img_size}_{clahe_tag}.npz"

    if raw_img_cache.exists() and raw_lbl_cache.exists() and not rebuild:
        imgs = np.load(raw_img_cache)
        labels = np.load(raw_lbl_cache)
        if len(imgs) == len(df):
            return imgs, labels, df

    if cache_file.exists() and not rebuild:
        data = np.load(cache_file)
        imgs = data["images"]
        labels = data["labels"]
        if len(imgs) == len(df):
            try:
                np.save(raw_img_cache, imgs)
                np.save(raw_lbl_cache, labels)
            except Exception:
                pass
            return imgs, labels, df

    # Fast-path: If a 160x160 cache exists, downsample directly in ~1s instead of reading all raw files from disk
    raw_160_img = CACHE_DIR / f"mc_{split_norm}_{crop_mode}_160_{clahe_tag}_imgs.npy"
    raw_160_lbl = CACHE_DIR / f"mc_{split_norm}_{crop_mode}_160_{clahe_tag}_lbls.npy"
    cache_160 = CACHE_DIR / f"mc_{split_norm}_{crop_mode}_160_{clahe_tag}.npz"
    if img_size < 160 and (raw_160_img.exists() or cache_160.exists()):
        if raw_160_img.exists() and raw_160_lbl.exists():
            imgs160 = np.load(raw_160_img)
            labels = np.load(raw_160_lbl)
        else:
            data160 = np.load(cache_160)
            imgs160 = data160["images"]
            labels = data160["labels"]
        if len(imgs160) == len(df):
            t_tensor = torch.from_numpy(imgs160).unsqueeze(1).float()
            resized_t = F.interpolate(t_tensor, size=(img_size, img_size), mode="area")
            imgs = resized_t.squeeze(1).byte().numpy()
            try:
                np.save(raw_img_cache, imgs)
                np.save(raw_lbl_cache, labels)
                if not cache_file.exists():
                    np.savez_compressed(cache_file, images=imgs, labels=labels)
            except Exception:
                pass
            return imgs, labels, df

    n = len(df)
    imgs = np.empty((n, img_size, img_size), dtype=np.uint8)
    labels = df["label_idx"].astype(np.int64).to_numpy()

    for i, row in enumerate(df.itertuples(index=False)):
        rel_fp = Path(row.filepath)
        fpath = rel_fp if rel_fp.is_absolute() else (REPO_ROOT / rel_fp)
        imgs[i] = _preprocess_to_uint8(
            fpath,
            source=str(row.source),
            use_aspect_crop=use_crop,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            clahe_blend=clahe_blend,
            img_size=img_size,
            crop_mode=crop_mode,
            ssd_margin=ssd_margin,
            crop_box=KDEF_STUDIO_CROP_BOX,
        )

    try:
        np.save(raw_img_cache, imgs)
        np.save(raw_lbl_cache, labels)
    except Exception:
        pass
    np.savez_compressed(cache_file, images=imgs, labels=labels)
    return imgs, labels, df


# ---------------------------------------------------------------------------
# Unified Multicultural Metrics Computation (All PDF Section 1.4 & 5.4 Metrics)
# ---------------------------------------------------------------------------

def _softmax_np(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def compute_multicultural_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    meta_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics on any split:
      - Overall Accuracy, Macro-F1, Weighted-F1, ROC-AUC (OVR Macro & Weighted)
      - Per-class Precision, Recall, F1, ROC-AUC, Recall Spread, 7x7 Confusion Matrix
      - Per-source breakdown (`kdef`, `radiate`, `fer2013`: Accuracy, Macro-F1, Weighted-F1, ROC-AUC, Per-class Recall/F1)
      - Per-demographic/ethnic group breakdown (`KDEF`, `B`, `W`, `A`, `H`, `FER`)
      - Per-gender breakdown (`female`, `male` on tracked subjects)
      - Per-pose angle breakdown (`FL`, `HL`, `S`, `HR`, `FR` on KDEF)
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    # Ensure rows sum to 1.0 for roc_auc_score
    row_sums = y_prob.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        y_prob = _softmax_np(y_prob)
    else:
        y_prob = y_prob / np.maximum(row_sums, 1e-12)

    n_classes = len(CLASS_NAMES)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    try:
        roc_auc_macro = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        roc_auc_weighted = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
    except Exception:
        roc_auc_macro = 0.0
        roc_auc_weighted = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    per_class_precision: Dict[str, float] = {}
    per_class_recall: Dict[str, float] = {}
    per_class_f1: Dict[str, float] = {}
    per_class_specificity: Dict[str, float] = {}
    per_class_auc: Dict[str, float] = {}
    total_n = float(cm.sum())

    for c_idx, c_name in enumerate(CLASS_NAMES):
        tp = float(cm[c_idx, c_idx])
        support = float(cm[c_idx, :].sum())
        pred_pos = float(cm[:, c_idx].sum())
        fn = support - tp
        fp = pred_pos - tp
        tn = max(total_n - tp - fn - fp, 0.0)
        r = tp / support if support > 0 else 0.0
        p = tp / pred_pos if pred_pos > 0 else 0.0
        f = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class_precision[c_name] = round(p, 4)
        per_class_recall[c_name] = round(r, 4)
        per_class_f1[c_name] = round(f, 4)
        per_class_specificity[c_name] = round(spec, 4)

        bin_true = (y_true == c_idx).astype(np.int32)
        if 0 < bin_true.sum() < len(bin_true):
            per_class_auc[c_name] = round(float(roc_auc_score(bin_true, y_prob[:, c_idx])), 4)
        else:
            per_class_auc[c_name] = 0.0

    recalls = list(per_class_recall.values())
    recall_spread = round(float(max(recalls) - min(recalls)), 4) if recalls else 0.0
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_specificity = float(np.mean(list(per_class_specificity.values()))) if per_class_specificity else 0.0

    result: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "n_correct": int((y_true == y_pred).sum()),
        "accuracy": round(acc, 6),
        "macro_precision": round(macro_precision, 6),
        "weighted_precision": round(weighted_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "weighted_recall": round(weighted_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "macro_specificity": round(macro_specificity, 6),
        "roc_auc_ovr_macro": round(roc_auc_macro, 6),
        "roc_auc_ovr_weighted": round(roc_auc_weighted, 6),
        "recall_spread": recall_spread,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "per_class_specificity": per_class_specificity,
        "per_class_auc": per_class_auc,
        "confusion_matrix": cm.tolist(),
    }

    if meta_df is not None and len(meta_df) == len(y_true):
        # 1. Per-source breakdown (kdef, radiate, fer2013)
        sources = meta_df["source"].astype(str).str.lower().to_numpy()
        per_source: Dict[str, Any] = {}
        for src in ("kdef", "radiate", "fer2013"):
            mask = sources == src
            if mask.sum() == 0:
                continue
            yt_s, yp_s, ypr_s = y_true[mask], y_pred[mask], y_prob[mask]
            try:
                s_auc = float(roc_auc_score(yt_s, ypr_s, multi_class="ovr", average="macro"))
            except Exception:
                s_auc = 0.0
            cm_s = confusion_matrix(yt_s, yp_s, labels=list(range(n_classes)))
            s_rec = {}
            s_f1 = {}
            for ci, cn in enumerate(CLASS_NAMES):
                tp_s = float(cm_s[ci, ci])
                sup_s = float(cm_s[ci, :].sum())
                pp_s = float(cm_s[:, ci].sum())
                r_s = tp_s / sup_s if sup_s > 0 else 0.0
                p_s = tp_s / pp_s if pp_s > 0 else 0.0
                f_s = (2.0 * p_s * r_s / (p_s + r_s)) if (p_s + r_s) > 0 else 0.0
                s_rec[cn] = round(r_s, 4)
                s_f1[cn] = round(f_s, 4)
            per_source[src] = {
                "n_samples": int(mask.sum()),
                "n_correct": int((yt_s == yp_s).sum()),
                "accuracy": round(float(accuracy_score(yt_s, yp_s)), 6),
                "macro_f1": round(float(f1_score(yt_s, yp_s, average="macro", zero_division=0)), 6),
                "weighted_f1": round(float(f1_score(yt_s, yp_s, average="weighted", zero_division=0)), 6),
                "roc_auc_ovr_macro": round(s_auc, 6),
                "per_class_recall": s_rec,
                "per_class_f1": s_f1,
            }
        result["per_source"] = per_source

        # 2. Per-ethnic/demographic group breakdown
        ethnic_arr = meta_df["ethnic_group"].astype(str).to_numpy()
        per_ethnic: Dict[str, Any] = {}
        for grp in ("KDEF", "B", "A", "H", "W", "O", "FER"):
            mask = ethnic_arr == grp
            if mask.sum() == 0:
                continue
            yt_g, yp_g = y_true[mask], y_pred[mask]
            per_ethnic[grp] = {
                "n_samples": int(mask.sum()),
                "n_correct": int((yt_g == yp_g).sum()),
                "accuracy": round(float(accuracy_score(yt_g, yp_g)), 6),
                "macro_f1": round(float(f1_score(yt_g, yp_g, average="macro", zero_division=0)), 6),
            }
        result["per_ethnic_group"] = per_ethnic

        # 3. Per-gender breakdown (tracked subjects: KDEF + RADIATE)
        gender_arr = meta_df["gender"].astype(str).str.lower().to_numpy()
        per_gender: Dict[str, Any] = {}
        for g in ("female", "male"):
            mask = gender_arr == g
            if mask.sum() == 0:
                continue
            yt_gn, yp_gn = y_true[mask], y_pred[mask]
            per_gender[g] = {
                "n_samples": int(mask.sum()),
                "accuracy": round(float(accuracy_score(yt_gn, yp_gn)), 6),
                "macro_f1": round(float(f1_score(yt_gn, yp_gn, average="macro", zero_division=0)), 6),
            }
        result["per_gender"] = per_gender

        # 4. Per-angle breakdown (KDEF 5 angles: FL, HL, S, HR, FR)
        angle_arr = meta_df["angle"].astype(str).str.upper().to_numpy()
        angle_acc: Dict[str, float] = {}
        for ang in ANGLE_NAMES:
            mask = (sources == "kdef") & (angle_arr == ang)
            if mask.sum() > 0:
                angle_acc[ang] = round(float((y_true[mask] == y_pred[mask]).mean()), 6)
        result["angle_accuracy"] = angle_acc

    return result


# ===========================================================================
# BASELINE 1: HOG + LinearSVC (with Calibrated / Softmax Probabilities)
# ===========================================================================

class HOGSVMClassifier:
    """
    HOG + LinearSVM baseline matching `Nội dung sườn cho dự án.pdf` Section 4.3.1:
      - Extracts HOG features via `cv2.HOGDescriptor` on `hog_img_size x hog_img_size` grayscale images
      - Standardizes each feature dimension via `StandardScaler` (fit strictly on `train`)
      - Trains `LinearSVC` (LIBLINEAR one-vs-rest) with `class_weight`
      - Provides probability outputs `predict_proba` via temperature-calibrated softmax or Platt sigmoid
        for ROC-AUC calculation.
    """

    def __init__(
        self,
        hog_img_size: int = 128,
        cell_size: int = 8,
        block_cells: int = 2,
        block_stride_cells: int = 1,
        nbins: int = 9,
        C: float = 0.0001,
        class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
        loss: str = "squared_hinge",
        dual: Union[bool, str] = "auto",
        max_iter: int = 3000,
        seed: int = 42,
    ):
        self.hog_img_size = int(hog_img_size)
        self.cell_size = int(cell_size)
        self.block_cells = int(block_cells)
        self.block_stride_cells = int(block_stride_cells)
        self.nbins = int(nbins)
        self.C = float(C)
        self.class_weight = class_weight
        self.loss = loss
        self.dual = dual
        self.max_iter = int(max_iter)
        self.seed = int(seed)

        self.scaler = StandardScaler()
        self.svm = LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            loss=self.loss,
            dual=self.dual,
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        # Platt calibration parameters (per-class slope A_c and intercept B_c fitted on decision scores)
        self.calib_a: Optional[np.ndarray] = None
        self.calib_b: Optional[np.ndarray] = None
        self.feature_dim: int = 0

    def _create_hog_descriptor(self) -> cv2.HOGDescriptor:
        win = (self.hog_img_size, self.hog_img_size)
        cell = (self.cell_size, self.cell_size)
        block = (self.cell_size * self.block_cells, self.cell_size * self.block_cells)
        stride = (self.cell_size * self.block_stride_cells, self.cell_size * self.block_stride_cells)
        return cv2.HOGDescriptor(
            _winSize=win,
            _blockSize=block,
            _blockStride=stride,
            _cellSize=cell,
            _nbins=self.nbins,
        )

    def extract_hog_features(self, images_u8: np.ndarray) -> np.ndarray:
        """
        Extracts HOG descriptors for a batch of uint8 grayscale images `(N, H, W)`.
        Returns float32 matrix `(N, D)`.
        """
        hog = self._create_hog_descriptor()
        n = len(images_u8)
        target_sz = (self.hog_img_size, self.hog_img_size)

        first_img = images_u8[0]
        if first_img.shape[:2] != target_sz:
            first_img = cv2.resize(first_img, target_sz, interpolation=cv2.INTER_AREA)
        first_feat = hog.compute(first_img).ravel()
        feat_dim = int(first_feat.shape[0])
        self.feature_dim = feat_dim

        feats = np.empty((n, feat_dim), dtype=np.float32)
        feats[0] = first_feat
        for i in range(1, n):
            im = images_u8[i]
            if im.shape[:2] != target_sz:
                im = cv2.resize(im, target_sz, interpolation=cv2.INTER_AREA)
            feats[i] = hog.compute(im).ravel()
        return feats

    def _fit_platt_scaling(self, margins: np.ndarray, y: np.ndarray) -> None:
        """
        Fits 1D logistic calibration `P(y=c | m_c) = sigmoid(a_c * m_c + b_c)` per class on training margins
        so `predict_proba` produces well-calibrated probabilities summing to 1.0 for ROC-AUC.
        """
        from sklearn.linear_model import LogisticRegression

        n_classes = margins.shape[1]
        a_arr = np.ones(n_classes, dtype=np.float64)
        b_arr = np.zeros(n_classes, dtype=np.float64)
        for c in range(n_classes):
            bin_y = (y == c).astype(np.int32)
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
            lr.fit(margins[:, c : c + 1], bin_y)
            a_arr[c] = float(lr.coef_[0, 0])
            b_arr[c] = float(lr.intercept_[0])
        self.calib_a = a_arr
        self.calib_b = b_arr

    def fit_from_features(self, X_train_hog: np.ndarray, y_train: np.ndarray) -> "HOGSVMClassifier":
        self.feature_dim = int(X_train_hog.shape[1])
        X_scaled = self.scaler.fit_transform(X_train_hog)
        self.svm.fit(X_scaled, y_train)
        margins = self.svm.decision_function(X_scaled)
        self._fit_platt_scaling(margins, y_train)
        return self

    def fit(self, images_u8: np.ndarray, y_train: np.ndarray) -> "HOGSVMClassifier":
        X_hog = self.extract_hog_features(images_u8)
        return self.fit_from_features(X_hog, y_train)

    def decision_function_from_features(self, X_hog: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_hog)
        return self.svm.decision_function(X_scaled)

    def predict_from_features(self, X_hog: np.ndarray) -> np.ndarray:
        margins = self.decision_function_from_features(X_hog)
        return np.argmax(margins, axis=1)

    def predict_proba_from_features(self, X_hog: np.ndarray) -> np.ndarray:
        margins = self.decision_function_from_features(X_hog)
        if self.calib_a is not None and self.calib_b is not None:
            logits = margins * self.calib_a[None, :] + self.calib_b[None, :]
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            probs_sum = np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            return probs / probs_sum
        return _softmax_np(margins)

    def predict(self, images_u8: np.ndarray) -> np.ndarray:
        X_hog = self.extract_hog_features(images_u8)
        return self.predict_from_features(X_hog)

    def predict_proba(self, images_u8: np.ndarray) -> np.ndarray:
        X_hog = self.extract_hog_features(images_u8)
        return self.predict_proba_from_features(X_hog)

    def count_parameters(self) -> int:
        """Total learnable SVM weights + biases: `7 * feature_dim + 7`."""
        if hasattr(self.svm, "coef_") and self.svm.coef_ is not None:
            return int(self.svm.coef_.size + self.svm.intercept_.size)
        return int(len(CLASS_NAMES) * self.feature_dim + len(CLASS_NAMES))

    def save(self, path: Union[str, Path], extra_meta: Optional[Dict[str, Any]] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "hog_svm",
            "config": {
                "hog_img_size": self.hog_img_size,
                "cell_size": self.cell_size,
                "block_cells": self.block_cells,
                "block_stride_cells": self.block_stride_cells,
                "nbins": self.nbins,
                "C": self.C,
                "class_weight": self.class_weight,
                "loss": self.loss,
                "max_iter": self.max_iter,
                "seed": self.seed,
                "feature_dim": self.feature_dim,
                "n_parameters": self.count_parameters(),
            },
            "scaler_mean": self.scaler.mean_.astype(np.float32),
            "scaler_scale": self.scaler.scale_.astype(np.float32),
            "svm_coef": self.svm.coef_.astype(np.float32),
            "svm_intercept": self.svm.intercept_.astype(np.float32),
            "svm_classes": self.svm.classes_,
            "calib_a": self.calib_a,
            "calib_b": self.calib_b,
            "extra_meta": extra_meta or {},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> Tuple["HOGSVMClassifier", Dict[str, Any]]:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        cfg = payload["config"]
        obj = cls(
            hog_img_size=cfg["hog_img_size"],
            cell_size=cfg["cell_size"],
            block_cells=cfg.get("block_cells", 2),
            block_stride_cells=cfg.get("block_stride_cells", 1),
            nbins=cfg["nbins"],
            C=cfg["C"],
            class_weight=cfg["class_weight"],
            loss=cfg.get("loss", "squared_hinge"),
            max_iter=cfg.get("max_iter", 3000),
            seed=cfg.get("seed", 42),
        )
        obj.feature_dim = int(cfg["feature_dim"])
        obj.scaler.mean_ = payload["scaler_mean"].astype(np.float64)
        obj.scaler.scale_ = payload["scaler_scale"].astype(np.float64)
        obj.scaler.var_ = obj.scaler.scale_ ** 2
        obj.scaler.n_features_in_ = obj.feature_dim
        obj.svm.coef_ = payload["svm_coef"].astype(np.float64)
        obj.svm.intercept_ = payload["svm_intercept"].astype(np.float64)
        obj.svm.classes_ = payload["svm_classes"]
        obj.calib_a = payload.get("calib_a")
        obj.calib_b = payload.get("calib_b")
        return obj, payload.get("extra_meta", {})


# ===========================================================================
# BASELINE 2: ResNet-18 Transfer Learning (IMAGENET1K_V1)
# ===========================================================================

class ResNet18Emotion(nn.Module):
    """
    ResNet-18 transfer learning architecture from `Nội dung sườn cho dự án.pdf` Section 4.3.2:
      - Uses official `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`
      - Adapts 1-channel grayscale input `[B, 1, H, W]` (or `[B, 3, H, W]`) to 3-channel ImageNet
        normalized input `(x - IMAGENET_MEAN) / IMAGENET_STD` inside the forward pass so it works
        seamlessly with both 1-channel and 3-channel evaluation pipelines.
      - Replaces final `fc` layer (`512 -> 1000`) with optional dropout + `Linear(512, 7)`
        (`11,180,103` trainable parameters).
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.2,
        input_is_normalized_1ch: bool = False,
        src_mean: float = DEFAULT_MULTICULTURAL_MEAN,
        src_std: float = DEFAULT_MULTICULTURAL_STD,
    ):
        super().__init__()
        self.model_name = "resnet18_imagenet1k_v1"
        self.num_classes = num_classes
        self.dropout_p = float(dropout)
        self.input_is_normalized_1ch = bool(input_is_normalized_1ch)
        self.src_mean = float(src_mean)
        self.src_std = float(src_std)

        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        in_features = backbone.fc.in_features  # 512
        if self.dropout_p > 0.0:
            backbone.fc = nn.Sequential(
                nn.Dropout(p=self.dropout_p),
                nn.Linear(in_features, num_classes),
            )
        else:
            backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def set_backbone_frozen(self, freeze: bool = True) -> None:
        """Freezes or unfreezes all conv/bn layers except `backbone.fc`."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = not freeze

    def get_gradcam_target_layers(self) -> List[nn.Module]:
        return [self.backbone.layer4[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both [0, 1] float tensors and pre-normalized 1-ch tensors
        if self.input_is_normalized_1ch:
            x = torch.clamp(x * self.src_std + self.src_mean, 0.0, 1.0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.imagenet_mean) / self.imagenet_std
        return self.backbone(x)


def count_resnet18_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_resnet18_checkpoint(
    model: ResNet18Emotion,
    checkpoint_path: Union[str, Path],
    config: Dict[str, Any],
    val_metrics: Dict[str, Any],
    epoch: int,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "resnet18",
        "model_name": "resnet18_imagenet1k_v1",
        "state_dict": model.state_dict(),
        "config": config,
        "normalization": {
            "mode": "imagenet_rgb_from_gray",
            "imagenet_mean": list(IMAGENET_MEAN),
            "imagenet_std": list(IMAGENET_STD),
            "multicultural_mean": DEFAULT_MULTICULTURAL_MEAN,
            "multicultural_std": DEFAULT_MULTICULTURAL_STD,
        },
        "epoch": int(epoch),
        "val_macro_f1": float(val_metrics.get("macro_f1", 0.0)),
        "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
        "val_roc_auc": float(val_metrics.get("roc_auc_ovr_macro", 0.0)),
        "val_metrics": val_metrics,
        "history": history or [],
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_resnet18_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    input_is_normalized_1ch: bool = False,
) -> Tuple[ResNet18Emotion, Dict[str, Any]]:
    if map_location is not None:
        device = torch.device(map_location)
    elif device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    dropout = float(cfg.get("dropout", 0.0))
    model = ResNet18Emotion(
        num_classes=int(cfg.get("num_classes", 7)),
        pretrained=False,
        dropout=dropout,
        input_is_normalized_1ch=input_is_normalized_1ch,
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()
    meta = {
        "model_name": ckpt.get("model_name", "resnet18_imagenet1k_v1"),
        "img_size": int(cfg.get("img_size", 128)),
        "crop_mode": str(cfg.get("crop_mode", DEFAULT_CROP_MODE)),
        "ssd_margin": float(cfg.get("ssd_margin", DEFAULT_SSD_MARGIN)),
        "use_clahe": bool(cfg.get("use_clahe", True)),
        "dropout": dropout,
        "epoch": ckpt.get("epoch"),
        "val_macro_f1": ckpt.get("val_macro_f1"),
        "val_accuracy": ckpt.get("val_accuracy"),
        "val_roc_auc": ckpt.get("val_roc_auc"),
        "config": cfg,
    }
    return model, meta


@torch.no_grad()
def predict_resnet18_split(
    model: ResNet18Emotion,
    images_u8: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    use_tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs inference on `images_u8` `(N, H, W)` uint8 array in `[0, 255]`.
    Returns `(y_pred, y_prob)`.
    """
    model.eval()
    n = len(images_u8)
    preds_list: List[np.ndarray] = []
    probs_list: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        batch_u8 = images_u8[start : start + batch_size]
        xb = torch.from_numpy(batch_u8).to(device=device, dtype=torch.float32).unsqueeze(1).div_(255.0)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(xb)
            if use_tta:
                logits_flip = model(torch.flip(xb, dims=[3]))
                logits = 0.5 * (logits + logits_flip)
            probs = F.softmax(logits.float(), dim=1)
        probs_np = probs.cpu().numpy()
        preds_np = np.argmax(probs_np, axis=1)
        preds_list.append(preds_np)
        probs_list.append(probs_np)

    return np.concatenate(preds_list, axis=0), np.concatenate(probs_list, axis=0)


# ===========================================================================
# TRANSFER LEARNING SUITE (6 Pre-trained Models from Reference Paper)
#   VGG16, VGG19, InceptionV3, DenseNet201, ResNet50, EfficientNetB4
#   Architecture (Section 3.2 & Fig. 1):
#     - Pre-trained ImageNet feature extractor (initial layers frozen to retain
#       general features; later layers fine-tuned on target dataset)
#     - Global Average Pooling -> FC(128) -> FC(64) -> Dropout(0.7) -> FC(num_classes)
# ===========================================================================

PAPER_TL_ARCHITECTURES: Dict[str, str] = {
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "inception_v3": "InceptionV3",
    "densenet201": "DenseNet201",
    "resnet50": "ResNet50",
    "efficientnet_b4": "EfficientNetB4",
}


def normalize_tl_arch_key(arch: str) -> str:
    k = str(arch).strip().lower().replace("-", "_")
    aliases = {
        "vgg_16": "vgg16",
        "vgg_19": "vgg19",
        "inceptionv3": "inception_v3",
        "densenet_201": "densenet201",
        "resnet_50": "resnet50",
        "efficientnetb4": "efficientnet_b4",
    }
    k = aliases.get(k, k)
    if k not in PAPER_TL_ARCHITECTURES:
        raise ValueError(f"Unsupported transfer learning arch '{arch}'. Expected one of {list(PAPER_TL_ARCHITECTURES)}")
    return k


class TransferLearningEmotion(nn.Module):
    """
    Pre-trained Transfer Learning model implementing the architecture in Section 3.2 & Fig. 1
    of the reference paper ('Enhancing Esophageal Cancer Detection Using Transfer Learning Techniques'):
      - Pre-trained on ImageNet-1K (`VGG16`, `VGG19`, `InceptionV3`, `DenseNet201`, `ResNet50`, `EfficientNetB4`).
      - Initial layers (`frozen_features`) are frozen to retain general features (edges, textures, patterns);
        later layers (`trainable_features`) are fine-tuned on the target facial emotion dataset.
      - Custom classification head:
        `Global Average Pooling` -> `FC(128)` -> `FC(64)` -> `Dropout(dropout)` -> `FC(num_classes)`.
      - Seamlessly accepts 1-channel grayscale `[B, 1, H, W]` or 3-channel `[B, 3, H, W]` in `[0, 1]`
        (or normalized 1-ch if `input_is_normalized_1ch=True`) and applies ImageNet RGB normalization.
    """

    def __init__(
        self,
        arch: str = "densenet201",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.7,
        freeze_initial: bool = True,
        input_is_normalized_1ch: bool = False,
        src_mean: float = DEFAULT_MULTICULTURAL_MEAN,
        src_std: float = DEFAULT_MULTICULTURAL_STD,
    ):
        super().__init__()
        self.arch_key = normalize_tl_arch_key(arch)
        self.display_name = PAPER_TL_ARCHITECTURES[self.arch_key]
        self.model_name = f"{self.arch_key}_tl_imagenet1k"
        self.num_classes = int(num_classes)
        self.dropout_p = float(dropout)
        self.freeze_initial = bool(freeze_initial)
        self.input_is_normalized_1ch = bool(input_is_normalized_1ch)
        self.src_mean = float(src_mean)
        self.src_std = float(src_std)

        if self.arch_key == "vgg16":
            weights = tv_models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            feats = tv_models.vgg16(weights=weights).features
            # Freeze initial block 1 (edges/textures); fine-tune blocks 2..5
            self.frozen_features = feats[:5]
            self.trainable_features = feats[5:]
            out_ch = 512
            default_in_sz = 160
        elif self.arch_key == "vgg19":
            weights = tv_models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            feats = tv_models.vgg19(weights=weights).features
            # Freeze initial block 1 (edges/textures); fine-tune blocks 2..5
            self.frozen_features = feats[:5]
            self.trainable_features = feats[5:]
            out_ch = 512
            default_in_sz = 160
        elif self.arch_key == "inception_v3":
            if pretrained:
                inc = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1)
            else:
                inc = tv_models.inception_v3(weights=None, aux_logits=False, init_weights=False)
            # Freeze initial conv stem (edges/textures); fine-tune Mixed_5b..7c
            self.frozen_features = nn.Sequential(
                inc.Conv2d_1a_3x3,
                inc.Conv2d_2a_3x3,
                inc.Conv2d_2b_3x3,
                inc.maxpool1,
                inc.Conv2d_3b_1x1,
                inc.Conv2d_4a_3x3,
                inc.maxpool2,
            )
            self.trainable_features = nn.Sequential(
                inc.Mixed_5b,
                inc.Mixed_5c,
                inc.Mixed_5d,
                inc.Mixed_6a,
                inc.Mixed_6b,
                inc.Mixed_6c,
                inc.Mixed_6d,
                inc.Mixed_6e,
                inc.Mixed_7a,
                inc.Mixed_7b,
                inc.Mixed_7c,
            )
            out_ch = 2048
            default_in_sz = 192
        elif self.arch_key == "densenet201":
            weights = tv_models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
            dn = tv_models.densenet201(weights=weights).features
            # Freeze stem + denseblock1 + transition1; fine-tune denseblock2..4 + norm5
            self.frozen_features = nn.Sequential(
                dn.conv0,
                dn.norm0,
                dn.relu0,
                dn.pool0,
                dn.denseblock1,
                dn.transition1,
            )
            self.trainable_features = nn.Sequential(
                dn.denseblock2,
                dn.transition2,
                dn.denseblock3,
                dn.transition3,
                dn.denseblock4,
                dn.norm5,
                nn.ReLU(inplace=True),
            )
            out_ch = 1920
            default_in_sz = 160
        elif self.arch_key == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            rn = tv_models.resnet50(weights=weights)
            # Freeze stem + layer1; fine-tune layer2..layer4
            self.frozen_features = nn.Sequential(
                rn.conv1,
                rn.bn1,
                rn.relu,
                rn.maxpool,
                rn.layer1,
            )
            self.trainable_features = nn.Sequential(
                rn.layer2,
                rn.layer3,
                rn.layer4,
            )
            out_ch = 2048
            default_in_sz = 160
        elif self.arch_key == "efficientnet_b4":
            weights = tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            ef = tv_models.efficientnet_b4(weights=weights).features
            # Freeze stem + stage 1; fine-tune stages 2..8
            self.frozen_features = ef[:2]
            self.trainable_features = ef[2:]
            out_ch = 1792
            default_in_sz = 160
        else:
            raise ValueError(f"Unknown arch_key: {self.arch_key}")

        self.out_channels = out_ch
        self.backbone_input_size = default_in_sz

        # Paper Section 3.2 Custom Layers:
        # Global Average Pooling -> FC(128) -> FC(64) -> Dropout(0.7) -> FC(num_classes)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(out_ch, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(64, self.num_classes),
        )

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.set_initial_frozen(self.freeze_initial)

    def set_initial_frozen(self, freeze_initial: bool = True) -> None:
        self.freeze_initial = bool(freeze_initial)
        for p in self.frozen_features.parameters():
            p.requires_grad = not self.freeze_initial
        for p in self.trainable_features.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True
        if self.freeze_initial:
            self.frozen_features.eval()

    def train(self, mode: bool = True) -> "TransferLearningEmotion":
        super().train(mode)
        if mode and self.freeze_initial:
            self.frozen_features.eval()
        return self

    def get_gradcam_target_layers(self) -> List[nn.Module]:
        return [self.trainable_features[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_is_normalized_1ch:
            x = torch.clamp(x * self.src_std + self.src_mean, 0.0, 1.0)
        if x.shape[-1] < self.backbone_input_size or x.shape[-2] < self.backbone_input_size:
            x = F.interpolate(
                x,
                size=(self.backbone_input_size, self.backbone_input_size),
                mode="bilinear",
                align_corners=False,
            )
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.imagenet_mean) / self.imagenet_std
        if self.freeze_initial and not any(p.requires_grad for p in self.frozen_features.parameters()):
            with torch.no_grad():
                f = self.frozen_features(x)
        else:
            f = self.frozen_features(x)
        f = self.trainable_features(f)
        return self.head(f)


def save_tl_checkpoint(
    model: TransferLearningEmotion,
    checkpoint_path: Union[str, Path],
    config: Dict[str, Any],
    val_metrics: Dict[str, Any],
    epoch: int,
    history: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "transfer_learning",
        "arch": model.arch_key,
        "display_name": model.display_name,
        "model_name": model.model_name,
        "state_dict": model.state_dict(),
        "config": config,
        "normalization": {
            "mode": "imagenet_rgb_from_gray",
            "imagenet_mean": list(IMAGENET_MEAN),
            "imagenet_std": list(IMAGENET_STD),
            "multicultural_mean": DEFAULT_MULTICULTURAL_MEAN,
            "multicultural_std": DEFAULT_MULTICULTURAL_STD,
        },
        "epoch": int(epoch),
        "val_macro_f1": float(val_metrics.get("macro_f1", 0.0)),
        "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
        "val_roc_auc": float(val_metrics.get("roc_auc_ovr_macro", 0.0)),
        "val_metrics": val_metrics,
        "history": history or {},
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_tl_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    input_is_normalized_1ch: bool = False,
) -> Tuple[TransferLearningEmotion, Dict[str, Any]]:
    if map_location is not None:
        device = torch.device(map_location)
    elif device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    arch = ckpt.get("arch") or cfg.get("arch", "densenet201")
    dropout = float(cfg.get("dropout", 0.7))
    freeze_initial = bool(cfg.get("freeze_initial", True))
    model = TransferLearningEmotion(
        arch=arch,
        num_classes=int(cfg.get("num_classes", 7)),
        pretrained=False,
        dropout=dropout,
        freeze_initial=freeze_initial,
        input_is_normalized_1ch=input_is_normalized_1ch,
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()
    meta = {
        "arch": model.arch_key,
        "display_name": model.display_name,
        "model_name": ckpt.get("model_name", model.model_name),
        "img_size": int(cfg.get("img_size", 160)),
        "crop_mode": str(cfg.get("crop_mode", DEFAULT_CROP_MODE)),
        "ssd_margin": float(cfg.get("ssd_margin", DEFAULT_SSD_MARGIN)),
        "use_clahe": bool(cfg.get("use_clahe", True)),
        "dropout": dropout,
        "freeze_initial": freeze_initial,
        "epoch": ckpt.get("epoch"),
        "val_macro_f1": ckpt.get("val_macro_f1"),
        "val_accuracy": ckpt.get("val_accuracy"),
        "val_roc_auc": ckpt.get("val_roc_auc"),
        "config": cfg,
        "history": ckpt.get("history", {}),
    }
    return model, meta


@torch.no_grad()
def predict_tl_split(
    model: nn.Module,
    images_u8: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    use_tta: bool = False,
    eval_img_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs inference on `images_u8` `(N, H, W)` uint8 array in `[0, 255]`.
    Returns `(y_pred, y_prob)`.
    """
    model.eval()
    n = len(images_u8)
    preds_list: List[np.ndarray] = []
    probs_list: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        batch_u8 = images_u8[start : start + batch_size]
        xb = torch.from_numpy(batch_u8).to(device=device, dtype=torch.float32).unsqueeze(1).div_(255.0)
        if eval_img_size is not None and xb.shape[-1] != eval_img_size:
            xb = F.interpolate(xb, size=(eval_img_size, eval_img_size), mode="area")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(xb)
            if use_tta:
                logits_flip = model(torch.flip(xb, dims=[3]))
                logits = 0.5 * (logits + logits_flip)
            probs = F.softmax(logits.float(), dim=1)
        probs_np = probs.cpu().numpy()
        preds_np = np.argmax(probs_np, axis=1)
        preds_list.append(preds_np)
        probs_list.append(probs_np)

    return np.concatenate(preds_list, axis=0), np.concatenate(probs_list, axis=0)


# ---------------------------------------------------------------------------
# Latency Benchmarking Utility (CPU & GPU single-image & batch latency)
# ---------------------------------------------------------------------------

def benchmark_inference_latency(
    model_type: str,
    model_or_obj: Any,
    sample_images_u8: np.ndarray,
    device: Optional[torch.device] = None,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> Dict[str, float]:
    """
    Measures single-image inference latency (ms/image and FPS) across `n_runs` images.
    For `hog_svm`, includes HOG descriptor extraction + StandardScaler + LinearSVC margin + probability.
    For PyTorch models (`resnet18` or `cnn`), measures tensor conversion + forward pass + softmax on `device`.
    """
    n_avail = len(sample_images_u8)
    if n_avail == 0:
        return {"ms_per_image": 0.0, "fps": 0.0}

    if model_type == "hog_svm":
        svm_model: HOGSVMClassifier = model_or_obj
        for i in range(min(n_warmup, n_avail)):
            _ = svm_model.predict_proba(sample_images_u8[i : i + 1])
        t0 = time.perf_counter()
        for i in range(n_runs):
            idx = i % n_avail
            _ = svm_model.predict_proba(sample_images_u8[idx : idx + 1])
        elapsed = time.perf_counter() - t0
        ms_per_img = (elapsed / n_runs) * 1000.0
        return {
            "ms_per_image": round(ms_per_img, 3),
            "fps": round(1000.0 / max(ms_per_img, 1e-6), 1),
            "device": "CPU",
            "n_runs": n_runs,
        }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net: nn.Module = model_or_obj.to(device)
    net.eval()

    with torch.no_grad():
        for i in range(min(n_warmup, n_avail)):
            im = sample_images_u8[i : i + 1]
            xb = torch.from_numpy(im).to(device=device, dtype=torch.float32).unsqueeze(1).div_(255.0)
            _ = net(xb)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for i in range(n_runs):
            idx = i % n_avail
            im = sample_images_u8[idx : idx + 1]
            xb = torch.from_numpy(im).to(device=device, dtype=torch.float32).unsqueeze(1).div_(255.0)
            out = net(xb)
            if isinstance(out, tuple):
                out = out[0]
            _ = F.softmax(out, dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

    ms_per_img = (elapsed / n_runs) * 1000.0
    return {
        "ms_per_image": round(ms_per_img, 3),
        "fps": round(1000.0 / max(ms_per_img, 1e-6), 1),
        "device": str(device),
        "n_runs": n_runs,
    }
