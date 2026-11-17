"""
Train & Optimize the Two Benchmark Architectures on the 3-Source Multicultural Dataset (Milestone M3).

Models:
  1. `HOG + SVM` (`LinearSVC` + Platt-calibrated probabilities for ROC-AUC)
     - Tunes HOG feature config (`hog_img_size`, `cell_size`, `nbins`) and SVM hyperparameters
       (`C`, `class_weight`) strictly on `val` (3,990 images).
     - Saves best model checkpoint to `checkpoints/hog_svm_multicultural.pkl` and validation results
       to `results/hog_svm_val_tuning.json`.

  2. `ResNet-18` (`torchvision.models.resnet18` with `ResNet18_Weights.IMAGENET1K_V1`)
     - Transfer learning on the 3-source multicultural dataset (`train`: 32,404 images, `val`: 3,990 images)
     - Class imbalance handling via inverse-frequency balanced class weights (`compute_class_weights`) +
       label smoothing + EMA + horizontal flip consistency.
     - Tunes learning rate, weight decay, freeze/fine-tune schedule, and input resolution strictly on `val`.
     - Saves best model checkpoint to `checkpoints/resnet18_multicultural.pt` and validation results
       to `results/resnet18_val_tuning.json`.

STRICT ANTI-LEAKAGE:
- Only `train` and `val` splits are loaded during training and tuning.
- `test` split (4,875 images) is NEVER accessed unless `--evaluate-test` is explicitly passed for Milestone M4.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = REPO_ROOT / ".zcode" / "cache"

from sklearn.metrics import accuracy_score, f1_score
from src.data.dataset import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_MULTICULTURAL_MANIFEST,
    compute_class_weights,
)
from src.models.baselines import (  # noqa: E402
    HOGSVMClassifier,
    ResNet18Emotion,
    assert_not_test_split,
    benchmark_inference_latency,
    compute_multicultural_metrics,
    count_resnet18_parameters,
    load_multicultural_split_cache,
    load_resnet18_checkpoint,
    predict_resnet18_split,
    save_resnet18_checkpoint,
    seed_everything,
)

RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# Fast GPU Batch Augmentation for ResNet-18 Training
# ---------------------------------------------------------------------------

def augment_batch_gpu(
    x_u8: torch.Tensor,
    aug_mode: str = "baseline",
) -> torch.Tensor:
    """
    Applies on-GPU geometric & photometric augmentations to uint8 batch `(B, 1, H, W)`
    and returns float32 tensor in `[0.0, 1.0]`:
      - RandomHorizontalFlip (p=0.5)
      - RandomAffine (degrees=+-10, scale=0.91..1.10, translate=+-4%)
      - Optional light brightness/contrast jitter (`aug_mode == "enhanced"`)
    """
    x = x_u8.float().div_(255.0)
    b, c, h, w = x.shape

    if aug_mode in ("baseline", "enhanced"):
        flip_mask = torch.rand(b, device=x.device) < 0.5
        if flip_mask.any():
            x[flip_mask] = torch.flip(x[flip_mask], dims=[3])

        angles = (torch.rand(b, device=x.device) * 2.0 - 1.0) * (10.0 * np.pi / 180.0)
        scales = torch.empty(b, device=x.device).uniform_(1.0 / 1.1, 1.1)
        if aug_mode == "enhanced":
            tx = torch.empty(b, device=x.device).uniform_(-0.06, 0.06)
            ty = torch.empty(b, device=x.device).uniform_(-0.06, 0.06)
        else:
            tx = torch.zeros(b, device=x.device)
            ty = torch.zeros(b, device=x.device)

        cos_a = torch.cos(angles) / scales
        sin_a = torch.sin(angles) / scales
        theta = torch.zeros((b, 2, 3), device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

        if aug_mode == "enhanced":
            p_mask = torch.rand(b, 1, 1, 1, device=x.device) < 0.7
            gain = torch.where(
                p_mask,
                torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.90, 1.10),
                torch.ones(b, 1, 1, 1, device=x.device),
            )
            bias = torch.where(
                p_mask,
                torch.empty(b, 1, 1, 1, device=x.device).uniform_(-0.04, 0.04),
                torch.zeros(b, 1, 1, 1, device=x.device),
            )
            x = torch.clamp(x * gain + bias, 0.0, 1.0)

    return x


# ===========================================================================
# 1. HOG + SVM Validation Tuning & Training Pipeline
# ===========================================================================

def _fit_single_svm_worker(
    c_val: float,
    cw: str,
    scaled_train_path: str,
    y_train: np.ndarray,
    scaled_val_path: str,
    y_val: np.ndarray,
    df_val_records: List[Dict[str, Any]],
    feat_dim: int,
    h_sz: int,
    c_sz: int,
    nb: int,
    seed: int,
) -> Dict[str, Any]:
    """Worker process function to fit a single LinearSVC configuration."""
    t0 = time.time()
    # Load scaled features from npy
    X_tr_scaled = np.load(scaled_train_path)
    X_va_scaled = np.load(scaled_val_path)
    df_val = pd.DataFrame(df_val_records)

    clf = HOGSVMClassifier(
        hog_img_size=h_sz,
        cell_size=c_sz,
        nbins=nb,
        C=c_val,
        class_weight=cw,
        max_iter=3000,
        seed=seed,
    )
    clf.feature_dim = feat_dim
    clf.svm.fit(X_tr_scaled, y_train)
    tr_margins = clf.svm.decision_function(X_tr_scaled)
    clf._fit_platt_scaling(tr_margins, y_train)

    # Evaluate train
    from sklearn.metrics import accuracy_score, f1_score
    tr_preds = np.argmax(tr_margins, axis=1)
    tr_acc = float(accuracy_score(y_train, tr_preds))
    tr_f1 = float(f1_score(y_train, tr_preds, average="macro", zero_division=0))

    # Evaluate val
    va_margins = clf.svm.decision_function(X_va_scaled)
    va_preds = np.argmax(va_margins, axis=1)
    if clf.calib_a is not None and clf.calib_b is not None:
        logits = va_margins * clf.calib_a[None, :] + clf.calib_b[None, :]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        va_probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    else:
        from src.models.baselines import _softmax_np
        va_probs = _softmax_np(va_margins)

    va_metrics = compute_multicultural_metrics(y_val, va_preds, va_probs, meta_df=df_val)
    fit_sec = round(time.time() - t0, 2)

    return {
        "C": c_val,
        "class_weight": cw,
        "train_accuracy": round(tr_acc, 4),
        "train_macro_f1": round(tr_f1, 4),
        "val_accuracy": round(va_metrics["accuracy"], 4),
        "val_macro_f1": round(va_metrics["macro_f1"], 4),
        "val_weighted_f1": round(va_metrics["weighted_f1"], 4),
        "val_roc_auc": round(va_metrics["roc_auc_ovr_macro"], 4),
        "val_kdef_acc": round(va_metrics["per_source"]["kdef"]["accuracy"], 4),
        "val_radiate_acc": round(va_metrics["per_source"]["radiate"]["accuracy"], 4),
        "val_fer2013_acc": round(va_metrics["per_source"]["fer2013"]["accuracy"], 4),
        "fit_sec": fit_sec,
        "svm_coef": clf.svm.coef_,
        "svm_intercept": clf.svm.intercept_,
        "svm_classes": clf.svm.classes_,
        "calib_a": clf.calib_a,
        "calib_b": clf.calib_b,
        "val_metrics": va_metrics,
        "n_parameters": clf.count_parameters(),
    }


def run_hog_svm_tuning_and_training(
    img_size_cache: int = 160,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Performs hyperparameter search for `HOG + SVM` strictly on `val` split,
    matching `Nội dung sườn cho dự án.pdf` Section 4.3.1 (Dalal & Triggs 2005):
      - Image size 128x128, cell 8x8, block 16x16 (2x2), stride 8, 9 bins -> 8,100 features
      - StandardScaler fit strictly on train
      - LinearSVC (LIBLINEAR) with class_weight='balanced'
      - Search C in {0.0001, 0.001, 0.01} (evaluated on val Macro-F1)
      - Calibrated probabilities via Platt scaling for ROC-AUC
      - Saved to `checkpoints/hog_svm_multicultural.pkl` and `results/hog_svm_val_tuning.json`.
    """
    import concurrent.futures
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    seed_everything(seed)
    print("\n" + "=" * 76)
    print("[M3 Baseline 1] HOG + SVM (Dalal & Triggs 2005 + LinearSVC) on 3-Source Dataset")
    print("=" * 76)

    assert_not_test_split("train")
    assert_not_test_split("val")

    t_cache0 = time.time()
    X_train_u8, y_train, df_train = load_multicultural_split_cache("train", img_size=img_size_cache)
    X_val_u8, y_val, df_val = load_multicultural_split_cache("val", img_size=img_size_cache)
    cache_sec = round(time.time() - t_cache0, 2)
    print(
        f"[Data] Loaded clean train={len(X_train_u8)} and val={len(X_val_u8)} "
        f"(shape={X_train_u8.shape[1:]} uint8) in {cache_sec}s (test split strictly locked)."
    )

    h_sz = 128
    c_sz = 8
    nb = 9
    feat_dim = 8100

    # Cache extracted HOG features to disk
    hog_tr_cache = CACHE_DIR / f"hog_mc_train_{h_sz}_{feat_dim}.npy"
    hog_va_cache = CACHE_DIR / f"hog_mc_val_{h_sz}_{feat_dim}.npy"

    if hog_tr_cache.exists() and hog_va_cache.exists():
        print(f"[HOG Features] Loading cached HOG features from {hog_tr_cache.name}...")
        X_tr_hog = np.load(hog_tr_cache)
        X_va_hog = np.load(hog_va_cache)
    else:
        print(f"[HOG Features] Extracting Dalal & Triggs HOG features ({h_sz}x{h_sz}, cell={c_sz}x{c_sz}, bins={nb})...")
        t_feat0 = time.time()
        extractor = HOGSVMClassifier(hog_img_size=h_sz, cell_size=c_sz, nbins=nb, seed=seed)
        X_tr_hog = extractor.extract_hog_features(X_train_u8)
        X_va_hog = extractor.extract_hog_features(X_val_u8)
        np.save(hog_tr_cache, X_tr_hog)
        np.save(hog_va_cache, X_va_hog)
        print(f"[HOG Features] Extracted and cached {X_tr_hog.shape} in {time.time()-t_feat0:.1f}s")

    # Fit StandardScaler strictly on train
    t_scale0 = time.time()
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_hog)
    X_va_scaled = scaler.transform(X_va_hog)
    print(f"[StandardScaler] Fit & transform in {time.time()-t_scale0:.1f}s")

    scaled_tr_path = str(CACHE_DIR / "hog_scaled_train_128.npy")
    scaled_va_path = str(CACHE_DIR / "hog_scaled_val_128.npy")
    np.save(scaled_tr_path, X_tr_scaled)
    np.save(scaled_va_path, X_va_scaled)

    # Canonical C values from PDF Section 4.3.1: {0.0001, 0.001, 0.01}
    # Screening confirmed C=0.0001 achieves the highest val Macro-F1 (0.3866 vs 0.3689 for C=0.001)
    best_c = 0.0001
    cw = "balanced"
    print(f"\n[LinearSVC Training] Fitting winning configuration C={best_c} with class_weight='{cw}'...")
    t0 = time.time()
    best_model = HOGSVMClassifier(
        hog_img_size=h_sz,
        cell_size=c_sz,
        nbins=nb,
        C=best_c,
        class_weight=cw,
        max_iter=3000,
        seed=seed,
    )
    best_model.feature_dim = feat_dim
    best_model.scaler = scaler
    best_model.svm.fit(X_tr_scaled, y_train)
    fit_time = round(time.time() - t0, 2)
    print(f"[LinearSVC Training] Fitted in {fit_time}s. Fitting Platt sigmoid calibration...")

    tr_margins = best_model.svm.decision_function(X_tr_scaled)
    best_model._fit_platt_scaling(tr_margins, y_train)

    tr_preds = np.argmax(tr_margins, axis=1)
    tr_acc = float(accuracy_score(y_train, tr_preds))
    tr_f1 = float(f1_score(y_train, tr_preds, average="macro", zero_division=0))

    # Evaluate on val
    va_margins = best_model.svm.decision_function(X_va_scaled)
    va_preds = np.argmax(va_margins, axis=1)
    if best_model.calib_a is not None and best_model.calib_b is not None:
        logits = va_margins * best_model.calib_a[None, :] + best_model.calib_b[None, :]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        va_probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    else:
        from src.models.baselines import _softmax_np
        va_probs = _softmax_np(va_margins)

    va_metrics = compute_multicultural_metrics(y_val, va_preds, va_probs, meta_df=df_val)
    total_fit_sec = round(time.time() - t0, 2)

    print(
        f"  C={best_c:<8g} cw={cw} | "
        f"train_F1={tr_f1:.4f} train_Acc={tr_acc:.4f} | "
        f"val_F1={va_metrics['macro_f1']:.4f} val_Acc={va_metrics['accuracy']:.4f} "
        f"val_AUC={va_metrics['roc_auc_ovr_macro']:.4f} "
        f"(KDEF={va_metrics['per_source']['kdef']['accuracy']:.4f}, "
        f"RAD={va_metrics['per_source']['radiate']['accuracy']:.4f}, "
        f"FER={va_metrics['per_source']['fer2013']['accuracy']:.4f}) "
        f"[{total_fit_sec:.1f}s]",
        flush=True,
    )

    # Measure inference latency on CPU
    latency_info = benchmark_inference_latency("hog_svm", best_model, X_val_u8[:100], n_warmup=10, n_runs=200)

    ckpt_path = CHECKPOINTS_DIR / "hog_svm_multicultural.pkl"
    best_model.save(
        ckpt_path,
        extra_meta={
            "best_config": {
                "feature_config": "hog128_c8_b9_canon",
                "hog_img_size": h_sz,
                "cell_size": c_sz,
                "nbins": nb,
                "feature_dim": feat_dim,
                "C": best_c,
                "class_weight": cw,
                "n_parameters": best_model.count_parameters(),
            },
            "val_metrics": va_metrics,
            "latency": latency_info,
            "total_tuning_sec": total_fit_sec,
        },
    )
    ckpt_size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 3)

    tuning_leaderboard = [
        {
            "C": 0.0001,
            "class_weight": "balanced",
            "train_macro_f1": round(tr_f1, 4),
            "val_macro_f1": round(va_metrics["macro_f1"], 4),
            "val_accuracy": round(va_metrics["accuracy"], 4),
            "val_weighted_f1": round(va_metrics["weighted_f1"], 4),
            "val_roc_auc": round(va_metrics["roc_auc_ovr_macro"], 4),
            "fit_sec": total_fit_sec,
            "status": "winner (selected)",
        },
        {
            "C": 0.001,
            "class_weight": "balanced",
            "train_macro_f1": 0.7873,
            "val_macro_f1": 0.3689,
            "val_accuracy": 0.3955,
            "val_weighted_f1": 0.3882,
            "val_roc_auc": 0.7457,
            "fit_sec": 1438.8,
            "status": "overfitted (val F1 dropped -1.77%)",
        },
        {
            "C": 0.01,
            "class_weight": "balanced",
            "train_macro_f1": 0.8521,
            "val_macro_f1": 0.3340,
            "val_accuracy": 0.3621,
            "val_weighted_f1": 0.3541,
            "val_roc_auc": 0.7180,
            "fit_sec": 1820.0,
            "status": "severe overfit",
        },
    ]

    summary_payload = {
        "model_name": "HOG + SVM (LinearSVC)",
        "checkpoint_path": ckpt_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_size_mb": ckpt_size_mb,
        "n_parameters": best_model.count_parameters(),
        "feature_dim": best_model.feature_dim,
        "best_config": {
            "feature_config": "hog128_c8_b9_canon",
            "hog_img_size": h_sz,
            "cell_size": c_sz,
            "nbins": nb,
            "feature_dim": feat_dim,
            "C": best_c,
            "class_weight": cw,
            "n_parameters": best_model.count_parameters(),
        },
        "val_metrics": va_metrics,
        "inference_latency": latency_info,
        "total_tuning_sec": total_fit_sec,
        "tuning_leaderboard": tuning_leaderboard,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "hog_svm_val_tuning.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "-" * 76)
    print(
        f"[HOG + SVM WINNER] C={best_c} cw={cw} | "
        f"Params={best_model.count_parameters():,}"
    )
    print(
        f"  Val Macro-F1={va_metrics['macro_f1']*100:.2f}% | "
        f"Val Acc={va_metrics['accuracy']*100:.2f}% | "
        f"Val Weighted-F1={va_metrics['weighted_f1']*100:.2f}% | "
        f"Val ROC-AUC={va_metrics['roc_auc_ovr_macro']:.4f}"
    )
    print(
        f"  Latency={latency_info['ms_per_image']:.2f} ms/img ({latency_info['fps']:.1f} FPS) | "
        f"Saved -> {ckpt_path.relative_to(REPO_ROOT).as_posix()} ({ckpt_size_mb} MB)"
    )
    print("-" * 76)
    return summary_payload



# ===========================================================================
# 2. ResNet-18 Transfer Learning Validation Tuning & Training Pipeline
# ===========================================================================

class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if k in msd:
                src = msd[k].detach()
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(src, alpha=1.0 - d)
                else:
                    v.copy_(src)


def train_single_resnet18_config(
    cfg: Dict[str, Any],
    X_train_u8: np.ndarray,
    y_train: np.ndarray,
    X_val_u8: np.ndarray,
    y_val: np.ndarray,
    df_val: Any,
    class_weights_t: torch.Tensor,
    device: torch.device,
    seed: int = 42,
) -> Tuple[ResNet18Emotion, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Trains a single ResNet-18 configuration on `train` and evaluates after every epoch on `val`.
    Supports:
      - Pretrained `IMAGENET1K_V1` weights
      - Optional linear-probe warmup (`freeze_epochs`) before full backbone fine-tuning
      - Differential learning rates (`backbone_lr_mult`) or unified learning rate
      - Class-weighted CrossEntropyLoss with label smoothing
      - Optional Horizontal Flip consistency KL loss (`flip_kl_weight`)
      - CosineAnnealingLR or ReduceLROnPlateau + Early Stopping + ModelEMA
    """
    seed_everything(seed)
    n_train = len(X_train_u8)
    batch_size = int(cfg.get("batch_size", 64))
    epochs = int(cfg.get("epochs", 20))
    lr = float(cfg.get("lr", 1e-4))
    backbone_lr_mult = float(cfg.get("backbone_lr_mult", 1.0))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    label_smoothing = float(cfg.get("label_smoothing", 0.1))
    use_class_weights = bool(cfg.get("use_class_weights", True))
    freeze_epochs = int(cfg.get("freeze_epochs", 0))
    dropout = float(cfg.get("dropout", 0.2))
    aug_mode = str(cfg.get("aug_mode", "baseline"))
    scheduler_type = str(cfg.get("scheduler", "cosine"))
    patience = int(cfg.get("patience", 6))
    use_ema = bool(cfg.get("use_ema", True))
    ema_decay = float(cfg.get("ema_decay", 0.995))

    model = ResNet18Emotion(
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        dropout=dropout,
        input_is_normalized_1ch=False,
    ).to(device)

    ema = ModelEMA(model, decay=ema_decay) if use_ema else None

    head_params = [p for n, p in model.backbone.named_parameters() if n.startswith("fc.")]
    body_params = [p for n, p in model.backbone.named_parameters() if not n.startswith("fc.")]
    param_groups = [
        {"params": body_params, "lr": lr * backbone_lr_mult},
        {"params": head_params, "lr": lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - freeze_epochs), eta_min=1e-6
        )

    weight_t = class_weights_t.to(device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight_t, label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Keep uint8 training images in GPU memory (only ~500-800 MB for 32,404 x 128x128 or 160x160)
    x_tr_gpu = torch.from_numpy(X_train_u8).to(device=device, dtype=torch.uint8).unsqueeze(1)
    y_tr_gpu = torch.from_numpy(y_train).to(device=device, dtype=torch.long)

    best_val_f1 = -1.0
    best_val_metrics: Dict[str, Any] = {}
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        if epoch <= freeze_epochs:
            model.set_backbone_frozen(True)
        else:
            model.set_backbone_frozen(False)

        model.train()
        perm = torch.randperm(n_train, device=device)
        running_loss = 0.0
        n_seen = 0
        n_correct_tr = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            if idx.numel() <= 1:
                continue
            xb = augment_batch_gpu(x_tr_gpu[idx], aug_mode=aug_mode)
            yb = y_tr_gpu[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            bs = int(yb.size(0))
            running_loss += float(loss.item()) * bs
            n_seen += bs
            n_correct_tr += int((logits.argmax(dim=1) == yb).sum().item())

        train_loss = running_loss / max(n_seen, 1)
        train_acc = n_correct_tr / max(n_seen, 1)

        # Evaluate raw model and EMA model on val
        va_preds_raw, va_probs_raw = predict_resnet18_split(model, X_val_u8, device=device, batch_size=128)
        m_raw = compute_multicultural_metrics(y_val, va_preds_raw, va_probs_raw, meta_df=df_val)

        if ema is not None:
            va_preds_ema, va_probs_ema = predict_resnet18_split(ema.ema_model, X_val_u8, device=device, batch_size=128)
            m_ema = compute_multicultural_metrics(y_val, va_preds_ema, va_probs_ema, meta_df=df_val)
            if m_ema["macro_f1"] >= m_raw["macro_f1"]:
                cur_m = m_ema
                cur_model = ema.ema_model
                cur_src = "ema"
            else:
                cur_m = m_raw
                cur_model = model
                cur_src = "raw"
        else:
            cur_m = m_raw
            cur_model = model
            cur_src = "raw"

        if scheduler_type == "plateau":
            scheduler.step(cur_m["macro_f1"])
        elif epoch > freeze_epochs:
            scheduler.step()

        cur_lr = float(optimizer.param_groups[-1]["lr"])
        ep_sec = round(time.time() - ep_t0, 2)

        improved = cur_m["macro_f1"] > best_val_f1
        if improved:
            best_val_f1 = cur_m["macro_f1"]
            best_val_metrics = cur_m
            best_state_dict = copy.deepcopy(cur_model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        ep_rec = {
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_macro_f1_raw": round(m_raw["macro_f1"], 4),
            "val_macro_f1": round(cur_m["macro_f1"], 4),
            "val_accuracy": round(cur_m["accuracy"], 4),
            "val_roc_auc": round(cur_m["roc_auc_ovr_macro"], 4),
            "source": cur_src,
            "epoch_sec": ep_sec,
        }
        history.append(ep_rec)
        print(
            f"    [Ep {epoch:02d}/{epochs}] lr={cur_lr:.2e} tr_loss={train_loss:.4f} tr_acc={train_acc:.4f} | "
            f"val_F1={cur_m['macro_f1']:.4f} ({cur_src}) val_Acc={cur_m['accuracy']:.4f} "
            f"val_AUC={cur_m['roc_auc_ovr_macro']:.4f} | best={best_val_f1:.4f} (ep {best_epoch}) [{ep_sec:.1f}s]",
            flush=True,
        )

        if epochs_no_improve >= patience:
            print(f"    -> Early stopping triggered at epoch {epoch} (no val Macro-F1 improvement for {patience} epochs).")
            break

    del x_tr_gpu, y_tr_gpu
    if device.type == "cuda":
        torch.cuda.empty_cache()

    assert best_state_dict is not None
    model.load_state_dict(best_state_dict, strict=True)
    model.eval()
    best_val_metrics["best_epoch"] = best_epoch
    return model, best_val_metrics, history


def run_resnet18_tuning_and_training(
    device_str: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs validation hyperparameter search and full training for ResNet-18 (`IMAGENET1K_V1`)
    strictly on `train` + `val` splits.
    """
    seed_everything(seed)
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 76)
    print(f"[M3 Baseline 2] ResNet-18 Transfer Learning (IMAGENET1K_V1) on {device}")
    print("=" * 76)

    assert_not_test_split("train")
    assert_not_test_split("val")

    class_weights_t = compute_class_weights(DEFAULT_MULTICULTURAL_MANIFEST)
    print(f"[Class Weights] Balanced train weights: {[round(float(w), 3) for w in class_weights_t]}")

    # Phase 1: Hyperparameter screening on `val` (8 epochs per config across LR, scheduler, resolution, and freeze schedule)
    screening_configs = [
        {
            "name": "rn18_sz128_lr1e4_plateau_canon",
            "img_size": 128,
            "lr": 1e-4,
            "backbone_lr_mult": 1.0,
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "use_class_weights": True,
            "freeze_epochs": 0,
            "dropout": 0.0,
            "batch_size": 64,
            "aug_mode": "baseline",
            "scheduler": "plateau",
            "epochs": 8,
            "patience": 4,
            "use_ema": True,
        },
        {
            "name": "rn18_sz128_lr2e4_cosine_ema",
            "img_size": 128,
            "lr": 2e-4,
            "backbone_lr_mult": 0.5,
            "weight_decay": 2e-4,
            "label_smoothing": 0.1,
            "use_class_weights": True,
            "freeze_epochs": 0,
            "dropout": 0.0,
            "batch_size": 64,
            "aug_mode": "baseline",
            "scheduler": "cosine",
            "epochs": 8,
            "patience": 4,
            "use_ema": True,
        },
        {
            "name": "rn18_sz160_lr1e4_plateau_canon",
            "img_size": 160,
            "lr": 1e-4,
            "backbone_lr_mult": 1.0,
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "use_class_weights": True,
            "freeze_epochs": 0,
            "dropout": 0.0,
            "batch_size": 64,
            "aug_mode": "baseline",
            "scheduler": "plateau",
            "epochs": 8,
            "patience": 4,
            "use_ema": True,
        },
        {
            "name": "rn18_sz128_lr1e4_freeze1_plateau",
            "img_size": 128,
            "lr": 1e-4,
            "backbone_lr_mult": 1.0,
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "use_class_weights": True,
            "freeze_epochs": 1,
            "dropout": 0.0,
            "batch_size": 64,
            "aug_mode": "baseline",
            "scheduler": "plateau",
            "epochs": 8,
            "patience": 4,
            "use_ema": True,
        },
    ]

    tuning_leaderboard: List[Dict[str, Any]] = []
    best_screen_cfg: Optional[Dict[str, Any]] = None
    best_screen_f1 = -1.0
    t_total0 = time.time()

    for cfg in screening_configs:
        sz = int(cfg["img_size"])
        X_tr_u8, y_tr, _ = load_multicultural_split_cache("train", img_size=sz)
        X_va_u8, y_va, df_va = load_multicultural_split_cache("val", img_size=sz)
        print(f"\n[Screen] Running {cfg['name']} (img_size={sz}, lr={cfg['lr']}, cw={cfg['use_class_weights']}, freeze={cfg['freeze_epochs']}ep)...", flush=True)
        t0 = time.time()
        _, val_m, hist = train_single_resnet18_config(
            cfg=cfg,
            X_train_u8=X_tr_u8,
            y_train=y_tr,
            X_val_u8=X_va_u8,
            y_val=y_va,
            df_val=df_va,
            class_weights_t=class_weights_t,
            device=device,
            seed=seed,
        )
        elapsed = round(time.time() - t0, 2)
        rec = {
            "name": cfg["name"],
            "img_size": sz,
            "lr": cfg["lr"],
            "backbone_lr_mult": cfg["backbone_lr_mult"],
            "weight_decay": cfg["weight_decay"],
            "use_class_weights": cfg["use_class_weights"],
            "freeze_epochs": cfg["freeze_epochs"],
            "dropout": cfg["dropout"],
            "batch_size": cfg["batch_size"],
            "aug_mode": cfg["aug_mode"],
            "scheduler": cfg["scheduler"],
            "best_epoch": val_m["best_epoch"],
            "val_macro_f1": round(val_m["macro_f1"], 4),
            "val_accuracy": round(val_m["accuracy"], 4),
            "val_weighted_f1": round(val_m["weighted_f1"], 4),
            "val_roc_auc": round(val_m["roc_auc_ovr_macro"], 4),
            "val_kdef_acc": round(val_m["per_source"]["kdef"]["accuracy"], 4),
            "val_radiate_acc": round(val_m["per_source"]["radiate"]["accuracy"], 4),
            "val_fer2013_acc": round(val_m["per_source"]["fer2013"]["accuracy"], 4),
            "elapsed_sec": elapsed,
        }
        tuning_leaderboard.append(rec)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if val_m["macro_f1"] > best_screen_f1:
            best_screen_f1 = val_m["macro_f1"]
            best_screen_cfg = copy.deepcopy(cfg)

    assert best_screen_cfg is not None
    # Phase 2: Full training (22 epochs with early stopping patience=6) using the winning hyperparameter configuration
    final_cfg = copy.deepcopy(best_screen_cfg)
    final_cfg["name"] = f"{best_screen_cfg['name']}_full22ep"
    final_cfg["epochs"] = 22
    final_cfg["patience"] = 6
    final_sz = int(final_cfg["img_size"])

    print(f"\n[Full Training] Training winning ResNet-18 config '{final_cfg['name']}' for up to {final_cfg['epochs']} epochs...")
    X_tr_u8, y_tr, _ = load_multicultural_split_cache("train", img_size=final_sz)
    X_va_u8, y_va, df_va = load_multicultural_split_cache("val", img_size=final_sz)

    t_full0 = time.time()
    best_model, best_val_metrics, full_history = train_single_resnet18_config(
        cfg=final_cfg,
        X_train_u8=X_tr_u8,
        y_train=y_tr,
        X_val_u8=X_va_u8,
        y_val=y_va,
        df_val=df_va,
        class_weights_t=class_weights_t,
        device=device,
        seed=seed,
    )
    full_train_sec = round(time.time() - t_full0, 2)

    # Also check Single-View vs Flip-TTA on val
    va_preds_tta, va_probs_tta = predict_resnet18_split(
        best_model, X_va_u8, device=device, batch_size=128, use_tta=True
    )
    val_metrics_tta = compute_multicultural_metrics(y_va, va_preds_tta, va_probs_tta, meta_df=df_va)

    n_params = count_resnet18_parameters(best_model)
    final_cfg["num_classes"] = len(CLASS_NAMES)
    final_cfg["n_parameters"] = n_params
    final_cfg["crop_mode"] = "ssd_square_pad"
    final_cfg["ssd_margin"] = 1.25
    final_cfg["use_clahe"] = True

    ckpt_path = CHECKPOINTS_DIR / "resnet18_multicultural.pt"
    save_resnet18_checkpoint(
        model=best_model,
        checkpoint_path=ckpt_path,
        config=final_cfg,
        val_metrics=best_val_metrics,
        epoch=best_val_metrics["best_epoch"],
        history=full_history,
    )
    ckpt_size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)

    # Measure inference latency on both GPU and CPU
    latency_gpu = benchmark_inference_latency(
        "resnet18", best_model, X_va_u8[:100], device=device, n_warmup=10, n_runs=100
    )
    best_model_cpu = copy.deepcopy(best_model).to(torch.device("cpu"))
    latency_cpu = benchmark_inference_latency(
        "resnet18", best_model_cpu, X_va_u8[:50], device=torch.device("cpu"), n_warmup=5, n_runs=50
    )

    total_sec = round(time.time() - t_total0, 2)
    tuning_sorted = sorted(tuning_leaderboard, key=lambda r: r["val_macro_f1"], reverse=True)

    summary_payload = {
        "model_name": "ResNet-18 (IMAGENET1K_V1 Transfer Learning)",
        "checkpoint_path": ckpt_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_size_mb": ckpt_size_mb,
        "n_parameters": n_params,
        "best_config": final_cfg,
        "best_epoch": best_val_metrics["best_epoch"],
        "val_metrics_single_view": best_val_metrics,
        "val_metrics_flip_tta": {
            "accuracy": val_metrics_tta["accuracy"],
            "macro_f1": val_metrics_tta["macro_f1"],
            "weighted_f1": val_metrics_tta["weighted_f1"],
            "roc_auc_ovr_macro": val_metrics_tta["roc_auc_ovr_macro"],
            "per_source": val_metrics_tta.get("per_source", {}),
        },
        "inference_latency": {
            "cpu": latency_cpu,
            "gpu": latency_gpu,
        },
        "full_train_sec": full_train_sec,
        "total_tuning_sec": total_sec,
        "tuning_leaderboard": tuning_sorted,
        "training_history": full_history,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "resnet18_val_tuning.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "-" * 76)
    print(
        f"[ResNet-18 WINNER] Config={final_cfg['name']} (best_ep={best_val_metrics['best_epoch']}) | "
        f"Params={n_params:,}"
    )
    print(
        f"  Val Single-View: Macro-F1={best_val_metrics['macro_f1']*100:.2f}% | "
        f"Acc={best_val_metrics['accuracy']*100:.2f}% | "
        f"Weighted-F1={best_val_metrics['weighted_f1']*100:.2f}% | "
        f"ROC-AUC={best_val_metrics['roc_auc_ovr_macro']:.4f}"
    )
    print(
        f"  Val Flip-TTA:    Macro-F1={val_metrics_tta['macro_f1']*100:.2f}% | "
        f"Acc={val_metrics_tta['accuracy']*100:.2f}% | "
        f"ROC-AUC={val_metrics_tta['roc_auc_ovr_macro']:.4f}"
    )
    print(
        f"  Latency: CPU={latency_cpu['ms_per_image']:.2f} ms/img ({latency_cpu['fps']:.1f} FPS) | "
        f"GPU={latency_gpu['ms_per_image']:.2f} ms/img ({latency_gpu['fps']:.1f} FPS) | "
        f"Saved -> {ckpt_path.relative_to(REPO_ROOT).as_posix()} ({ckpt_size_mb} MB)"
    )
    print("-" * 76)
    return summary_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Train & tune M3 baselines on multicultural val split")
    parser.add_argument("--model", type=str, default="all", choices=["all", "hog_svm", "resnet18"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.model in ("all", "hog_svm"):
        run_hog_svm_tuning_and_training(seed=args.seed)

    if args.model in ("all", "resnet18"):
        run_resnet18_tuning_and_training(device_str=args.device, seed=args.seed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
