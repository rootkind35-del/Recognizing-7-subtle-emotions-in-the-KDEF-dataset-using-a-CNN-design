# -*- coding: utf-8 -*-
"""
External Research & Tuning Script for Baseline 2 (`ResNet-18` IMAGENET1K_V1).
Strictly uses `train` and `val` splits only (`test` split is locked).
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_baselines import ModelEMA, augment_batch_gpu
from src.data.dataset import CLASS_NAMES, DEFAULT_MULTICULTURAL_MANIFEST, compute_class_weights
from src.models.baselines import (
    ResNet18Emotion,
    assert_not_test_split,
    benchmark_inference_latency,
    compute_multicultural_metrics,
    count_resnet18_parameters,
    load_multicultural_split_cache,
    predict_resnet18_split,
    save_resnet18_checkpoint,
    seed_everything,
)
from src.models.train import apply_batch_mix

RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def train_resnet18_exp(
    cfg: Dict[str, Any],
    X_train_u8: np.ndarray,
    y_train: np.ndarray,
    X_val_u8: np.ndarray,
    y_val: np.ndarray,
    df_val: Any,
    base_cw: torch.Tensor,
    device: torch.device,
    seed: int = 42,
) -> Tuple[ResNet18Emotion, Dict[str, Any], List[Dict[str, Any]]]:
    seed_everything(seed)
    n_train = len(X_train_u8)
    bs = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    warmup_epochs = int(cfg.get("warmup_epochs", 3))
    patience = int(cfg.get("patience", 8))
    lr = float(cfg["lr"])
    backbone_mult = float(cfg.get("backbone_lr_mult", 1.0))
    wd = float(cfg.get("weight_decay", 5e-4))
    ls = float(cfg.get("label_smoothing", 0.05))
    cw_pow = float(cfg.get("cw_power", 0.5))
    dropout = float(cfg.get("dropout", 0.2))
    aug_mode = str(cfg.get("aug_mode", "enhanced"))
    mix_prob = float(cfg.get("mix_prob", 0.20))
    flip_kl_w = float(cfg.get("flip_kl_weight", 0.15))
    ema_decay = float(cfg.get("ema_decay", 0.996))

    model = ResNet18Emotion(
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        dropout=dropout,
        input_is_normalized_1ch=False,
    ).to(device)
    ema = ModelEMA(model, decay=ema_decay)

    head_params = [p for n, p in model.backbone.named_parameters() if n.startswith("fc.")]
    body_params = [p for n, p in model.backbone.named_parameters() if not n.startswith("fc.")]
    optimizer = torch.optim.AdamW(
        [{"params": body_params, "lr": lr * backbone_mult}, {"params": head_params, "lr": lr}],
        lr=lr,
        weight_decay=wd,
    )

    def lr_lambda(ep_idx: int) -> float:
        if warmup_epochs > 0 and ep_idx < warmup_epochs:
            return float(ep_idx + 1) / float(max(1, warmup_epochs))
        prog = float(ep_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.005 + 0.995 * 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if cw_pow > 0.0:
        cw = torch.pow(base_cw.float(), cw_pow)
        cw = (cw / cw.mean()).to(device)
    else:
        cw = None

    x_tr_gpu = torch.from_numpy(X_train_u8).to(device=device, dtype=torch.uint8).unsqueeze(1)
    y_tr_gpu = torch.from_numpy(y_train).to(device=device, dtype=torch.long)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_f1 = -1.0
    best_val_metrics: Dict[str, Any] = {}
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        perm = torch.randperm(n_train, device=device)
        running_loss = 0.0
        n_seen = 0
        use_mix = mix_prob > 0.0 and (3 <= epoch <= epochs - 5)

        for start in range(0, n_train, bs):
            idx = perm[start : start + bs]
            if idx.numel() <= 1:
                continue
            xb = augment_batch_gpu(x_tr_gpu[idx], aug_mode=aug_mode)
            yb = y_tr_gpu[idx]
            if use_mix:
                xb_in, y_soft, was_mixed = apply_batch_mix(
                    xb, yb, num_classes=len(CLASS_NAMES), mix_prob=mix_prob, mixup_alpha=0.2, cutmix_alpha=0.8, splitmix_prob=0.0
                )
            else:
                xb_in, y_soft, was_mixed = xb, F.one_hot(yb, len(CLASS_NAMES)).float(), False

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(xb_in)
                log_p = F.log_softmax(logits.float(), dim=1)
                if was_mixed:
                    y_smooth = y_soft * (1.0 - ls) + ls / len(CLASS_NAMES)
                    loss = -(y_smooth * log_p).sum(dim=1).mean()
                else:
                    y_oh = F.one_hot(yb, len(CLASS_NAMES)).float()
                    y_smooth = y_oh * (1.0 - ls) + ls / len(CLASS_NAMES)
                    per_sample = -(y_smooth * log_p).sum(dim=1)
                    if cw is not None:
                        per_sample = per_sample * cw[yb]
                    loss = per_sample.mean()
                if flip_kl_w > 0.0 and not was_mixed:
                    logits_f = model(torch.flip(xb_in, dims=[3]))
                    p1 = F.softmax(logits.float(), dim=1)
                    p2 = F.softmax(logits_f.float(), dim=1)
                    sym_kl = 0.5 * (
                        F.kl_div(F.log_softmax(logits.float(), dim=1), p2.detach(), reduction="batchmean")
                        + F.kl_div(F.log_softmax(logits_f.float(), dim=1), p1.detach(), reduction="batchmean")
                    )
                    loss = loss + flip_kl_w * sym_kl

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_loss += float(loss.item()) * int(yb.size(0))
            n_seen += int(yb.size(0))

        scheduler.step()
        train_loss = running_loss / max(n_seen, 1)

        va_preds_raw, va_probs_raw = predict_resnet18_split(model, X_val_u8, device=device, batch_size=128)
        m_raw = compute_multicultural_metrics(y_val, va_preds_raw, va_probs_raw, meta_df=df_val)
        va_preds_ema, va_probs_ema = predict_resnet18_split(ema.ema_model, X_val_u8, device=device, batch_size=128)
        m_ema = compute_multicultural_metrics(y_val, va_preds_ema, va_probs_ema, meta_df=df_val)
        if m_ema["macro_f1"] >= m_raw["macro_f1"]:
            cur_m, cur_model, cur_src = m_ema, ema.ema_model, "ema"
        else:
            cur_m, cur_model, cur_src = m_raw, model, "raw"

        ep_sec = round(time.time() - ep_t0, 2)
        improved = cur_m["macro_f1"] > best_val_f1
        if improved:
            best_val_f1 = cur_m["macro_f1"]
            best_val_metrics = cur_m
            best_state_dict = copy.deepcopy(cur_model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        history.append({
            "epoch": epoch,
            "lr": float(optimizer.param_groups[-1]["lr"]),
            "train_loss": round(train_loss, 4),
            "val_macro_f1": round(cur_m["macro_f1"], 4),
            "val_accuracy": round(cur_m["accuracy"], 4),
            "val_roc_auc": round(cur_m["roc_auc_ovr_macro"], 4),
            "source": cur_src,
            "epoch_sec": ep_sec,
        })
        print(
            f"  [{cfg['name']}] Ep {epoch:02d}/{epochs} | tr_loss={train_loss:.4f} | "
            f"val_F1={cur_m['macro_f1']:.4f} ({cur_src}) val_Acc={cur_m['accuracy']:.4f} "
            f"val_AUC={cur_m['roc_auc_ovr_macro']:.4f} | best={best_val_f1:.4f} [{ep_sec:.1f}s]",
            flush=True,
        )
        if no_improve >= patience:
            print(f"  -> Early stopping at epoch {epoch}", flush=True)
            break

    del x_tr_gpu, y_tr_gpu
    if device.type == "cuda":
        torch.cuda.empty_cache()

    assert best_state_dict is not None
    model.load_state_dict(best_state_dict, strict=True)
    model.eval()
    best_val_metrics["best_epoch"] = best_epoch
    return model, best_val_metrics, history


def main() -> None:
    seed_everything(42)
    assert_not_test_split("train")
    assert_not_test_split("val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_u8, y_tr, _ = load_multicultural_split_cache("train", img_size=160)
    X_va_u8, y_va, df_va = load_multicultural_split_cache("val", img_size=160)
    base_cw = compute_class_weights(DEFAULT_MULTICULTURAL_MANIFEST)

    configs = [
        {
            "name": "rn18_cw50_cos_lr2e4_enh",
            "img_size": 160,
            "batch_size": 64,
            "epochs": 28,
            "warmup_epochs": 3,
            "patience": 8,
            "lr": 2e-4,
            "backbone_lr_mult": 0.5,
            "weight_decay": 5e-4,
            "label_smoothing": 0.05,
            "cw_power": 0.50,
            "dropout": 0.2,
            "aug_mode": "enhanced",
            "mix_prob": 0.20,
            "flip_kl_weight": 0.15,
            "ema_decay": 0.996,
        },
        {
            "name": "rn18_cw50_cos_lr3e4_full_enh",
            "img_size": 160,
            "batch_size": 64,
            "epochs": 28,
            "warmup_epochs": 3,
            "patience": 8,
            "lr": 3e-4,
            "backbone_lr_mult": 1.0,
            "weight_decay": 5e-4,
            "label_smoothing": 0.05,
            "cw_power": 0.50,
            "dropout": 0.2,
            "aug_mode": "enhanced",
            "mix_prob": 0.25,
            "flip_kl_weight": 0.20,
            "ema_decay": 0.996,
        },
        {
            "name": "rn18_cw35_cos_lr2e4_enh",
            "img_size": 160,
            "batch_size": 64,
            "epochs": 28,
            "warmup_epochs": 3,
            "patience": 8,
            "lr": 2e-4,
            "backbone_lr_mult": 1.0,
            "weight_decay": 3e-4,
            "label_smoothing": 0.05,
            "cw_power": 0.35,
            "dropout": 0.15,
            "aug_mode": "enhanced",
            "mix_prob": 0.15,
            "flip_kl_weight": 0.15,
            "ema_decay": 0.996,
        },
    ]

    best_f1 = -1.0
    best_bundle: Optional[Dict[str, Any]] = None
    leaderboard: List[Dict[str, Any]] = []

    for cfg in configs:
        print("\n" + "=" * 72, flush=True)
        print(f"Running {cfg['name']}", flush=True)
        print("=" * 72, flush=True)
        t0 = time.time()
        model, val_m, hist = train_resnet18_exp(cfg, X_tr_u8, y_tr, X_va_u8, y_va, df_va, base_cw, device, seed=42)
        elapsed = round(time.time() - t0, 2)
        rec = {
            "name": cfg["name"],
            "best_epoch": val_m["best_epoch"],
            "val_macro_f1": val_m["macro_f1"],
            "val_accuracy": val_m["accuracy"],
            "val_weighted_f1": val_m["weighted_f1"],
            "val_roc_auc": val_m["roc_auc_ovr_macro"],
            "elapsed_sec": elapsed,
        }
        leaderboard.append(rec)
        print(f"[RESULT {cfg['name']}] Val Macro-F1={val_m['macro_f1']:.4f} Acc={val_m['accuracy']:.4f} AUC={val_m['roc_auc_ovr_macro']:.4f}", flush=True)
        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            best_bundle = {"cfg": copy.deepcopy(cfg), "model": model, "val_m": val_m, "hist": hist, "elapsed": elapsed}
        else:
            del model
            torch.cuda.empty_cache()

    assert best_bundle is not None
    best_model = best_bundle["model"]
    best_cfg = best_bundle["cfg"]
    best_val_m = best_bundle["val_m"]

    # Only promote if it beats previous 0.704291
    prev_f1 = 0.704291
    if best_f1 > prev_f1:
        va_preds_tta, va_probs_tta = predict_resnet18_split(best_model, X_va_u8, device=device, batch_size=128, use_tta=True)
        val_m_tta = compute_multicultural_metrics(y_va, va_preds_tta, va_probs_tta, meta_df=df_va)
        n_params = count_resnet18_parameters(best_model)
        best_cfg["num_classes"] = len(CLASS_NAMES)
        best_cfg["n_parameters"] = n_params
        best_cfg["crop_mode"] = "ssd_square_pad"
        best_cfg["ssd_margin"] = 1.25
        best_cfg["use_clahe"] = True
        ckpt_path = CHECKPOINTS_DIR / "resnet18_multicultural.pt"
        save_resnet18_checkpoint(
            model=best_model,
            checkpoint_path=ckpt_path,
            config=best_cfg,
            val_metrics=best_val_m,
            epoch=best_val_m["best_epoch"],
            history=best_bundle["hist"],
        )
        lat_gpu = benchmark_inference_latency("resnet18", best_model, X_va_u8[:100], device=device, n_warmup=10, n_runs=100)
        lat_cpu = benchmark_inference_latency("resnet18", copy.deepcopy(best_model).cpu(), X_va_u8[:50], device=torch.device("cpu"), n_warmup=5, n_runs=50)
        payload = {
            "model_name": "ResNet-18 (IMAGENET1K_V1 Transfer Learning)",
            "checkpoint_path": ckpt_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
            "n_parameters": n_params,
            "best_config": best_cfg,
            "best_epoch": best_val_m["best_epoch"],
            "val_metrics_single_view": best_val_m,
            "val_metrics_flip_tta": {
                "accuracy": val_m_tta["accuracy"],
                "macro_f1": val_m_tta["macro_f1"],
                "weighted_f1": val_m_tta["weighted_f1"],
                "roc_auc_ovr_macro": val_m_tta["roc_auc_ovr_macro"],
                "per_source": val_m_tta.get("per_source", {}),
            },
            "inference_latency": {"cpu": lat_cpu, "gpu": lat_gpu},
            "tuning_leaderboard": sorted(leaderboard, key=lambda x: x["val_macro_f1"], reverse=True),
            "training_history": best_bundle["hist"],
        }
        with open(RESULTS_DIR / "resnet18_val_tuning.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[ResNet-18 PROMOTED] {best_cfg['name']} beat {prev_f1:.4f} -> {best_f1:.4f} (Acc={best_val_m['accuracy']:.4f})")
    else:
        print(f"\n[ResNet-18 KEPT] Best {best_f1:.4f} did not beat previous {prev_f1:.4f}; keeping existing checkpoint.")


if __name__ == "__main__":
    main()
