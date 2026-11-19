# -*- coding: utf-8 -*-
"""
Fast GPU-Accelerated Research & Tuning Script for Baseline 1 (`HOG + SVM`).
Strictly uses `train` and `val` splits only (`test` split is locked).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, normalize

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
    load_multicultural_split_cache,
    seed_everything,
)

RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / ".zcode" / "cache"


def get_hog_cached(tag: str, hog_img_size: int, cell_size: int, nbins: int, X_tr_u8: np.ndarray, X_va_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p_tr = CACHE_DIR / f"hog_{tag}_train.npy"
    p_va = CACHE_DIR / f"hog_{tag}_val.npy"
    if p_tr.exists() and p_va.exists():
        return np.load(p_tr), np.load(p_va)
    ext = HOGSVMClassifier(hog_img_size=hog_img_size, cell_size=cell_size, nbins=nbins, seed=42)
    X_tr = ext.extract_hog_features(X_tr_u8)
    X_va = ext.extract_hog_features(X_va_u8)
    np.save(p_tr, X_tr)
    np.save(p_va, X_va)
    return X_tr, X_va


def transform_features(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    use_roothog: bool = False,
    use_l2: bool = False,
    use_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    tr = X_tr.astype(np.float32, copy=True)
    va = X_va.astype(np.float32, copy=True)
    if use_roothog:
        tr = np.sqrt(np.maximum(tr, 0.0))
        va = np.sqrt(np.maximum(va, 0.0))
    if use_l2:
        tr = normalize(tr, norm="l2", axis=1, copy=False)
        va = normalize(va, norm="l2", axis=1, copy=False)
    if use_scaler:
        scaler = StandardScaler()
        tr = scaler.fit_transform(tr)
        va = scaler.transform(va)
    return tr, va


def fit_ovr_svm_gpu(
    X_tr_gpu: torch.Tensor,
    y_tr_gpu: torch.Tensor,
    X_va_gpu: torch.Tensor,
    y_va_np: np.ndarray,
    c_val: float,
    cw_vec: Optional[torch.Tensor],
    epochs: int = 45,
) -> Dict[str, Any]:
    """Solves LinearSVC OvR squared-hinge objective on GPU in ~1.5s."""
    n_samples, dim = X_tr_gpu.shape
    n_classes = len(CLASS_NAMES)
    # OvR targets in {-1, +1}
    y_onehot = torch.full((n_samples, n_classes), -1.0, device=X_tr_gpu.device, dtype=torch.float32)
    y_onehot.scatter_(1, y_tr_gpu.unsqueeze(1), 1.0)
    if cw_vec is not None:
        sample_w = cw_vec[y_tr_gpu].unsqueeze(1)  # (N, 1)
    else:
        sample_w = None

    linear = nn.Linear(dim, n_classes).to(X_tr_gpu.device)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)

    opt = torch.optim.AdamW(linear.parameters(), lr=0.02, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    bs = 2048
    reg_coeff = 1.0 / (2.0 * float(c_val) * float(n_samples))

    for _ in range(epochs):
        perm = torch.randperm(n_samples, device=X_tr_gpu.device)
        for s in range(0, n_samples, bs):
            idx = perm[s : s + bs]
            xb = X_tr_gpu[idx]
            yb_pm = y_onehot[idx]
            opt.zero_grad(set_to_none=True)
            margins = linear(xb)
            hinge = torch.clamp(1.0 - yb_pm * margins, min=0.0).pow(2)
            if sample_w is not None:
                hinge = hinge * sample_w[idx]
            loss = hinge.sum(dim=1).mean() + reg_coeff * (linear.weight.pow(2).sum())
            loss.backward()
            opt.step()
        sched.step()

    with torch.no_grad():
        va_margins = linear(X_va_gpu).cpu().numpy()
    va_preds = np.argmax(va_margins, axis=1)
    va_f1 = float(f1_score(y_va_np, va_preds, average="macro", zero_division=0))
    va_acc = float(accuracy_score(y_va_np, va_preds))
    return {"val_macro_f1": round(va_f1, 6), "val_accuracy": round(va_acc, 6)}


def main() -> None:
    seed_everything(42)
    assert_not_test_split("train")
    assert_not_test_split("val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_u8, y_train, df_train = load_multicultural_split_cache("train", img_size=160)
    X_val_u8, y_val, _ = load_multicultural_split_cache("val", img_size=160)
    base_cw = compute_class_weights(df_train).numpy()

    cw_map = {
        "balanced": torch.tensor(base_cw, device=device, dtype=torch.float32),
        "cw50": torch.tensor((base_cw ** 0.5) / np.mean(base_cw ** 0.5), device=device, dtype=torch.float32),
        "cw35": torch.tensor((base_cw ** 0.35) / np.mean(base_cw ** 0.35), device=device, dtype=torch.float32),
        "none": None,
    }
    y_tr_gpu = torch.from_numpy(y_train).long().to(device)

    feat_configs = [
        {"tag": "hog128_c8_b9", "hog_img_size": 128, "cell_size": 8, "nbins": 9},
        {"tag": "hog96_c8_b9", "hog_img_size": 96, "cell_size": 8, "nbins": 9},
        {"tag": "hog160_c16_b9", "hog_img_size": 160, "cell_size": 16, "nbins": 9},
    ]
    norm_specs = [
        ("std", False, False, True, [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]),
        ("root_std", True, False, True, [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]),
        ("root_l2", True, True, False, [0.01, 0.05, 0.1, 0.3, 1.0]),
        ("root_l2_std", True, True, True, [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]),
    ]

    leaderboard: List[Dict[str, Any]] = []
    for fcfg in feat_configs:
        t0 = time.time()
        X_tr_raw, X_va_raw = get_hog_cached(
            fcfg["tag"], fcfg["hog_img_size"], fcfg["cell_size"], fcfg["nbins"], X_train_u8, X_val_u8
        )
        print(f"\n[Feature {fcfg['tag']}] dim={X_tr_raw.shape[1]} ready in {time.time()-t0:.1f}s", flush=True)

        for norm_tag, use_root, use_l2, use_sc, c_grid in norm_specs:
            X_tr_n, X_va_n = transform_features(X_tr_raw, X_va_raw, use_roothog=use_root, use_l2=use_l2, use_scaler=use_sc)
            X_tr_gpu = torch.from_numpy(X_tr_n).to(device)
            X_va_gpu = torch.from_numpy(X_va_n).to(device)
            for cw_name, cw_vec in cw_map.items():
                for c_val in c_grid:
                    t1 = time.time()
                    m = fit_ovr_svm_gpu(X_tr_gpu, y_tr_gpu, X_va_gpu, y_val, c_val=c_val, cw_vec=cw_vec)
                    rec = {
                        **fcfg,
                        "feature_dim": int(X_tr_raw.shape[1]),
                        "norm_tag": norm_tag,
                        "use_roothog": use_root,
                        "use_l2": use_l2,
                        "use_scaler": use_sc,
                        "cw_name": cw_name,
                        "C": c_val,
                        **m,
                        "sec": round(time.time() - t1, 2),
                    }
                    leaderboard.append(rec)
            del X_tr_gpu, X_va_gpu
            torch.cuda.empty_cache()

        top_so_far = max(leaderboard, key=lambda x: x["val_macro_f1"])
        print(f"  Best so far: {top_so_far['tag']} | {top_so_far['norm_tag']} | {top_so_far['cw_name']} | C={top_so_far['C']} -> F1={top_so_far['val_macro_f1']:.4f}, Acc={top_so_far['val_accuracy']:.4f}", flush=True)

    leaderboard_sorted = sorted(leaderboard, key=lambda x: x["val_macro_f1"], reverse=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_search = RESULTS_DIR / "hog_svm_tuning_experiments.json"
    with open(out_search, "w", encoding="utf-8") as f:
        json.dump({"best": leaderboard_sorted[0], "top20": leaderboard_sorted[:20], "total_runs": len(leaderboard)}, f, indent=2)
    print(f"\n[tune_hog_svm] Completed {len(leaderboard)} GPU SVM runs! Top 5:")
    for r in leaderboard_sorted[:5]:
        print(" ", r)


if __name__ == "__main__":
    main()
