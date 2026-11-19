# -*- coding: utf-8 -*-
"""
External Research & Tuning Script for Main CNN (`KDEF_Stem2Ch_MS4_Symmetric`).
Strictly uses `train` and `val` splits only (test split is locked).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import CLASS_NAMES, compute_class_weights
from src.models.model import KDEF_Stem2Ch_MS4_Symmetric, count_parameters
from src.models.train import (
    ModelEMA,
    SnapshotSoup,
    SoftLabelSmoothedFocalCE,
    SupConLoss,
    FlipConsistencyLoss,
    TrainConfig,
    apply_batch_mix,
    compute_full_val_metrics,
    gpu_augment_batch,
    load_multicultural_split_cached,
    optimize_val_logit_bias,
    predict_cnn_on_images,
    seed_everything,
    train_cnn_run,
)

RESULTS_DIR = REPO_ROOT / "results"
CKPT_DIR = REPO_ROOT / "checkpoints"


def run_cnn_experiments(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tune_cnn] Device: {device}")

    t0 = time.time()
    train_imgs, train_lbls, train_df = load_multicultural_split_cached("train", img_size=160)
    val_imgs, val_lbls, val_df = load_multicultural_split_cached("val", img_size=160)
    base_cw = compute_class_weights(train_df)
    print(f"[tune_cnn] Loaded train={len(train_imgs)}, val={len(val_imgs)} in {time.time()-t0:.2f}s")
    print(f"[tune_cnn] Base class weights: {[round(float(x), 3) for x in base_cw]}")

    experiments: List[TrainConfig] = []

    if args.mode in ("v2_res", "all"):
        experiments.append(
            TrainConfig(
                name="cnn_v2res_bs128_lr2e3_cw50_erase15",
                variant="v2_res",
                epochs=args.epochs,
                warmup_epochs=4,
                patience=12,
                batch_size=64,
                grad_accum_steps=2,
                lr=2.0e-3,
                min_lr=1e-6,
                weight_decay=3e-4,
                class_weight_power=0.50,
                label_smoothing=0.05,
                focal_gamma=0.50,
                poly_eps=0.10,
                two_view_flip=True,
                flip_kl_weight=0.25,
                flip_feat_weight=0.10,
                supcon_weight=0.12,
                supcon_temp=0.08,
                aug_mode="enhanced",
                facial_erase_prob=0.15,
                mix_prob=0.25,
                mix_start_epoch=4,
                mix_end_epoch=max(10, args.epochs - 7),
                finetune_epochs=8,
                tune_val_bias=True,
                seed=args.seed,
            )
        )

    if args.mode in ("screen", "all"):
        # Exp 1: Faster batch_size=128, lr=1.5e-3, tempered cw=0.50, lighter reg
        experiments.append(
            TrainConfig(
                name="cnn_exp1_bs128_lr15e4_cw50_lightreg",
                epochs=args.epochs,
                warmup_epochs=4,
                patience=12,
                batch_size=128,
                lr=1.5e-3,
                min_lr=1e-6,
                weight_decay=2e-4,
                class_weight_power=0.50,
                label_smoothing=0.05,
                focal_gamma=0.50,
                poly_eps=0.15,
                two_view_flip=True,
                flip_kl_weight=0.25,
                flip_feat_weight=0.10,
                supcon_weight=0.12,
                supcon_temp=0.08,
                aug_mode="enhanced",
                mix_prob=0.25,
                mix_start_epoch=4,
                mix_end_epoch=max(10, args.epochs - 7),
                finetune_epochs=8,
                tune_val_bias=True,
                seed=args.seed,
            )
        )
        # Exp 2: cw=0.35 (less extreme minority upweighting -> better overall Acc & fear/sad balance), no poly_eps
        experiments.append(
            TrainConfig(
                name="cnn_exp2_bs128_lr15e4_cw35_ls05",
                epochs=args.epochs,
                warmup_epochs=4,
                patience=12,
                batch_size=128,
                lr=1.5e-3,
                min_lr=1e-6,
                weight_decay=2e-4,
                class_weight_power=0.35,
                label_smoothing=0.05,
                focal_gamma=0.50,
                poly_eps=0.0,
                two_view_flip=True,
                flip_kl_weight=0.20,
                flip_feat_weight=0.10,
                supcon_weight=0.15,
                supcon_temp=0.07,
                aug_mode="enhanced",
                mix_prob=0.20,
                mix_start_epoch=4,
                mix_end_epoch=max(10, args.epochs - 7),
                finetune_epochs=8,
                tune_val_bias=True,
                seed=args.seed,
            )
        )
        # Exp 3: Higher LR 2e-3 + cosine + cw=0.50 + facial_erase=0.15
        experiments.append(
            TrainConfig(
                name="cnn_exp3_bs128_lr2e3_cw50_erase15",
                epochs=args.epochs,
                warmup_epochs=4,
                patience=12,
                batch_size=64,
                grad_accum_steps=2,
                lr=2.0e-3,
                min_lr=1e-6,
                weight_decay=3e-4,
                class_weight_power=0.50,
                label_smoothing=0.05,
                focal_gamma=0.50,
                poly_eps=0.10,
                two_view_flip=True,
                flip_kl_weight=0.25,
                flip_feat_weight=0.10,
                supcon_weight=0.12,
                supcon_temp=0.08,
                aug_mode="enhanced",
                facial_erase_prob=0.15,
                mix_prob=0.25,
                mix_start_epoch=4,
                mix_end_epoch=max(10, args.epochs - 7),
                finetune_epochs=8,
                tune_val_bias=True,
                seed=args.seed,
            )
        )

    results_summary: List[Dict[str, Any]] = []
    best_f1 = -1.0
    best_res: Optional[Dict[str, Any]] = None
    best_ckpt: Optional[Path] = None

    for cfg in experiments:
        print("\n" + "=" * 72, flush=True)
        print(f"Running {cfg.name} (epochs={cfg.epochs}, bs={cfg.batch_size}, lr={cfg.lr}, cw_pow={cfg.class_weight_power})", flush=True)
        print("=" * 72, flush=True)
        ckpt_p = CKPT_DIR / f"_tune_{cfg.name}.pt"
        res = train_cnn_run(
            cfg=cfg,
            train_imgs_u8=train_imgs,
            train_labels=train_lbls,
            val_imgs_u8=val_imgs,
            val_labels=val_lbls,
            val_df=val_df,
            base_class_weights=base_cw,
            device=device,
            save_ckpt_path=ckpt_p,
            verbose=True,
        )
        sv = res["val_metrics_single_view"]
        tta = res["val_metrics_flip_tta"]
        row = {
            "name": cfg.name,
            "best_epoch": res["best_epoch"],
            "elapsed_sec": res["elapsed_sec"],
            "val_sv_macro_f1": sv["macro_f1"],
            "val_sv_acc": sv["accuracy"],
            "val_sv_roc_auc": sv["roc_auc_ovr_macro"],
            "val_tta_macro_f1": tta["macro_f1"],
            "val_tta_acc": tta["accuracy"],
            "per_source_acc": {k: v["accuracy"] for k, v in sv.get("per_source", {}).items()},
            "per_class_f1": sv["per_class_f1"],
        }
        results_summary.append(row)
        print(
            f"[RESULT {cfg.name}] SV Macro-F1={sv['macro_f1']:.4f} Acc={sv['accuracy']:.4f} AUC={sv['roc_auc_ovr_macro']:.4f} | "
            f"TTA Macro-F1={tta['macro_f1']:.4f} | time={res['elapsed_sec']}s",
            flush=True,
        )
        if sv["macro_f1"] > best_f1:
            best_f1 = sv["macro_f1"]
            best_res = res
            best_ckpt = ckpt_p

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "cnn_tuning_experiments.json"
    existing_map: Dict[str, Dict[str, Any]] = {}
    if out_json.exists():
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            for item in old_data.get("experiments", []):
                if isinstance(item, dict) and "name" in item:
                    existing_map[item["name"]] = item
        except Exception:
            pass
    for item in results_summary:
        existing_map[item["name"]] = item
    merged_experiments = list(existing_map.values())
    best_overall = max(merged_experiments, key=lambda x: float(x.get("val_sv_macro_f1", -1.0))) if merged_experiments else None
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"experiments": merged_experiments, "best_name": best_overall["name"] if best_overall else None},
            f,
            indent=2,
        )
    print(f"\n[tune_cnn] Saved tuning summary to {out_json}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CNN on multicultural train/val splits")
    parser.add_argument("--mode", type=str, default="screen", choices=["screen", "v2_res", "all"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    run_cnn_experiments(args)


if __name__ == "__main__":
    main()
