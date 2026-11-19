# -*- coding: utf-8 -*-
"""
Finalize Baseline 1 (`HOG + SVM`) with exact L-BFGS convex solver on GPU
for the LinearSVC OvR squared-hinge objective on the winning `hog160_c16_b9` (2,916 dims).
Saves `checkpoints/hog_svm_multicultural.pkl` and `results/hog_svm_val_tuning.json`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import CLASS_NAMES, compute_class_weights
from src.models.baselines import (
    HOGSVMClassifier,
    assert_not_test_split,
    benchmark_inference_latency,
    compute_multicultural_metrics,
    load_multicultural_split_cache,
    seed_everything,
)

RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
CACHE_DIR = REPO_ROOT / ".zcode" / "cache"


def fit_linearsvc_lbfgs_gpu(
    X_tr_gpu: torch.Tensor,
    y_tr_gpu: torch.Tensor,
    c_val: float,
    cw_dict: Optional[Dict[int, float]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, dim = X_tr_gpu.shape
    n_classes = len(CLASS_NAMES)
    y_pm = torch.full((n_samples, n_classes), -1.0, device=device, dtype=torch.float32)
    y_pm.scatter_(1, y_tr_gpu.unsqueeze(1), 1.0)
    if cw_dict is not None:
        cw_t = torch.tensor([cw_dict[i] for i in range(n_classes)], device=device, dtype=torch.float32)
        sw = cw_t[y_tr_gpu].unsqueeze(1)
    else:
        sw = None

    lin = nn.Linear(dim, n_classes).to(device)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)

    # Exact LinearSVC OvR squared-hinge objective (divided by N for numerical conditioning)
    reg = 1.0 / (2.0 * float(c_val) * float(n_samples))
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=60, history_size=15, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        margins = lin(X_tr_gpu)
        hinge = torch.clamp(1.0 - y_pm * margins, min=0.0).pow(2)
        if sw is not None:
            hinge = hinge * sw
        loss = hinge.sum(dim=1).mean() + reg * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lin.weight.detach().cpu().numpy().astype(np.float64), lin.bias.detach().cpu().numpy().astype(np.float64)


def main() -> None:
    seed_everything(42)
    assert_not_test_split("train")
    assert_not_test_split("val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_u8, y_train, df_train = load_multicultural_split_cache("train", img_size=160)
    X_val_u8, y_val, df_val = load_multicultural_split_cache("val", img_size=160)
    base_cw = compute_class_weights(df_train).numpy()
    cw35 = {i: round(float((base_cw[i] ** 0.35) / np.mean(base_cw ** 0.35)), 4) for i in range(len(CLASS_NAMES))}
    cw50 = {i: round(float((base_cw[i] ** 0.50) / np.mean(base_cw ** 0.50)), 4) for i in range(len(CLASS_NAMES))}

    h_sz, c_sz, nb = 160, 16, 9
    p_tr = CACHE_DIR / "hog_hog160_c16_b9_train.npy"
    p_va = CACHE_DIR / "hog_hog160_c16_b9_val.npy"
    if p_tr.exists() and p_va.exists():
        X_tr_hog, X_va_hog = np.load(p_tr), np.load(p_va)
    else:
        ext = HOGSVMClassifier(hog_img_size=h_sz, cell_size=c_sz, nbins=nb, seed=42)
        X_tr_hog = ext.extract_hog_features(X_train_u8)
        X_va_hog = ext.extract_hog_features(X_val_u8)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_hog).astype(np.float32)
    X_va_s = scaler.transform(X_va_hog).astype(np.float32)

    X_tr_gpu = torch.from_numpy(X_tr_s).to(device)
    y_tr_gpu = torch.from_numpy(y_train).long().to(device)

    candidates = [
        ("cw35", cw35, 0.0001),
        ("cw35", cw35, 0.0002),
        ("cw35", cw35, 0.0003),
        ("cw35", cw35, 0.0005),
        ("cw50", cw50, 0.0003),
        ("none", None, 0.0001),
    ]

    best_f1 = -1.0
    best_model: Optional[HOGSVMClassifier] = None
    best_meta: Dict[str, Any] = {}
    leaderboard: List[Dict[str, Any]] = []

    for cw_name, cw_val, c_val in candidates:
        t0 = time.time()
        w, b = fit_linearsvc_lbfgs_gpu(X_tr_gpu, y_tr_gpu, c_val=c_val, cw_dict=cw_val, device=device)

        clf = HOGSVMClassifier(
            hog_img_size=h_sz,
            cell_size=c_sz,
            nbins=nb,
            C=c_val,
            class_weight=cw_val,
            max_iter=3000,
            seed=42,
        )
        clf.feature_dim = int(X_tr_hog.shape[1])
        clf.scaler = scaler
        clf.svm.coef_ = w
        clf.svm.intercept_ = b
        clf.svm.classes_ = np.arange(len(CLASS_NAMES), dtype=np.int64)

        tr_margins = clf.svm.decision_function(X_tr_s)
        clf._fit_platt_scaling(tr_margins, y_train)

        va_margins = clf.svm.decision_function(X_va_s)
        va_preds = np.argmax(va_margins, axis=1)
        logits = va_margins * clf.calib_a[None, :] + clf.calib_b[None, :]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        va_probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

        va_m = compute_multicultural_metrics(y_val, va_preds, va_probs, meta_df=df_val)
        fit_sec = round(time.time() - t0, 2)
        rec = {
            "hog_img_size": h_sz,
            "cell_size": c_sz,
            "nbins": nb,
            "feature_dim": clf.feature_dim,
            "C": c_val,
            "cw_name": cw_name,
            "val_macro_f1": va_m["macro_f1"],
            "val_accuracy": va_m["accuracy"],
            "val_weighted_f1": va_m["weighted_f1"],
            "val_roc_auc": va_m["roc_auc_ovr_macro"],
            "fit_sec": fit_sec,
        }
        leaderboard.append(rec)
        print(
            f"[LBFGS-SVM] hog160_c16_b9 | cw={cw_name} C={c_val} -> "
            f"val_F1={va_m['macro_f1']:.4f} val_Acc={va_m['accuracy']:.4f} val_AUC={va_m['roc_auc_ovr_macro']:.4f} ({fit_sec}s)",
            flush=True,
        )
        if va_m["macro_f1"] > best_f1:
            best_f1 = va_m["macro_f1"]
            best_model = clf
            best_meta = {"rec": rec, "va_metrics": va_m, "cw_name": cw_name, "c_val": c_val}

    del X_tr_gpu, y_tr_gpu
    if device.type == "cuda":
        torch.cuda.empty_cache()

    assert best_model is not None
    latency_info = benchmark_inference_latency("hog_svm", best_model, X_val_u8[:100], n_warmup=10, n_runs=200)
    ckpt_path = CHECKPOINTS_DIR / "hog_svm_multicultural.pkl"
    best_cfg = {
        "feature_config": "hog160_c16_b9_cw35",
        "hog_img_size": h_sz,
        "cell_size": c_sz,
        "nbins": nb,
        "feature_dim": best_model.feature_dim,
        "C": best_meta["c_val"],
        "class_weight": best_meta["cw_name"],
        "n_parameters": best_model.count_parameters(),
    }
    best_model.save(
        ckpt_path,
        extra_meta={
            "best_config": best_cfg,
            "val_metrics": best_meta["va_metrics"],
            "latency": latency_info,
            "total_tuning_sec": sum(r["fit_sec"] for r in leaderboard),
        },
    )
    ckpt_size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 3)
    summary = {
        "model_name": "HOG + SVM (LinearSVC)",
        "checkpoint_path": ckpt_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_size_mb": ckpt_size_mb,
        "n_parameters": best_model.count_parameters(),
        "feature_dim": best_model.feature_dim,
        "best_config": best_cfg,
        "val_metrics": best_meta["va_metrics"],
        "inference_latency": latency_info,
        "tuning_leaderboard": sorted(leaderboard, key=lambda x: x["val_macro_f1"], reverse=True),
    }
    with open(RESULTS_DIR / "hog_svm_val_tuning.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[HOG+SVM FINALIZED] Saved {ckpt_path} ({ckpt_size_mb} MB) | Val Macro-F1={best_f1:.4f} Acc={best_meta['va_metrics']['accuracy']:.4f} AUC={best_meta['va_metrics']['roc_auc_ovr_macro']:.4f}")


if __name__ == "__main__":
    main()
