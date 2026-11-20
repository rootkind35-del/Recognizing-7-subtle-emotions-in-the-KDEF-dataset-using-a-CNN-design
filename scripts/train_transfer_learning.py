# -*- coding: utf-8 -*-
"""
Train & Evaluate the 6 Pre-trained Transfer Learning Models from the Reference Paper
("Enhancing Esophageal Cancer Detection Using Transfer Learning Techniques", Section 3.2 & Fig. 1)
on the 3-Source Multicultural Facial Emotion Recognition Dataset (KDEF + RADIATE + FER-2013).

Models trained (all pre-trained on ImageNet-1K):
  1. VGG16          -> checkpoints/vgg16_multicultural.pt
  2. VGG19          -> checkpoints/vgg19_multicultural.pt
  3. InceptionV3    -> checkpoints/inception_v3_multicultural.pt
  4. DenseNet201    -> checkpoints/densenet201_multicultural.pt
  5. ResNet50       -> checkpoints/resnet50_multicultural.pt
  6. EfficientNetB4 -> checkpoints/efficientnet_b4_multicultural.pt

Paper Architecture & Setup (Section 3.1, 3.2 & Figs. 1-3):
  - Input resized to 224x224, converted to 3-channel RGB, normalized with ImageNet mean/std.
  - Pre-trained backbone used as feature extractor (initial stem/block1 layers frozen to retain
    general features like edges, textures, and patterns; later layers fine-tuned on target dataset).
  - Custom classification head:
    Global Average Pooling -> FC(128) -> FC(64) -> Dropout(0.7) -> FC(num_classes=7) with Softmax.
  - Online GPU data augmentation every epoch (rotation, affine distortion, horizontal flipping,
    light brightness/contrast jitter) combined with tempered inverse-frequency class weights.
  - Fine-tuning hyperparameters: lr = 0.0001 (1e-4), batch_size = 32, 20 epochs (Paper Section 3.2 & Figs. 2-3).
  - Strictly uses `train` (32,404 images) and `val` (3,990 images) splits; `test` split (4,875 images) is locked.
"""

from __future__ import annotations

import argparse
import copy
import gc
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
from sklearn.metrics import accuracy_score, f1_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_MULTICULTURAL_MANIFEST,
    compute_class_weights,
)
from src.models.baselines import (  # noqa: E402
    PAPER_TL_ARCHITECTURES,
    TransferLearningEmotion,
    assert_not_test_split,
    compute_multicultural_metrics,
    load_multicultural_split_cache,
    normalize_tl_arch_key,
    predict_tl_split,
    save_tl_checkpoint,
    seed_everything,
)

RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


class ModelEMA:
    """Exponential Moving Average of trainable weights and BatchNorm buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.decay = float(decay)
        self.updates = 0
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        msd = dict(model.state_dict())
        self._pairs = [
            (v, msd[k])
            for k, v in self.ema_model.state_dict().items()
            if k in msd
        ]

    @torch.no_grad()
    def update(self, _model: nn.Module = None) -> None:
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for v, src in self._pairs:
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src.detach(), alpha=1.0 - d)
            else:
                v.copy_(src)


def apply_online_paper_aug_gpu(
    x_u8: torch.Tensor,
    target_size: int = 224,
) -> torch.Tensor:
    """
    Applies online GPU augmentations following Paper Section 3.1 (horizontal flip,
    rotation p=0.7, affine distortion) + light photometric jitter for cross-dataset
    robustness, returning float32 `(B, 1, target_size, target_size)` in `[0, 1]`.
    """
    x = x_u8.float().div_(255.0)
    if x.shape[-1] != target_size or x.shape[-2] != target_size:
        x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)

    b, c, h, w = x.shape
    flip_mask = torch.rand(b, device=x.device) < 0.5
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])

    rot_mask = (torch.rand(b, device=x.device) < 0.7).float()
    angles = rot_mask * ((torch.rand(b, device=x.device) * 2.0 - 1.0) * (8.0 * math.pi / 180.0))
    scales = torch.empty(b, device=x.device).uniform_(0.92, 1.08)
    shear = torch.empty(b, device=x.device).uniform_(-0.04, 0.04)
    tx = torch.empty(b, device=x.device).uniform_(-0.05, 0.05)
    ty = torch.empty(b, device=x.device).uniform_(-0.05, 0.05)

    cos_a = torch.cos(angles) / scales
    sin_a = torch.sin(angles) / scales
    theta = torch.zeros((b, 2, 3), device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a + shear
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

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
    return torch.clamp(x * gain + bias, 0.0, 1.0)


@torch.no_grad()
def evaluate_tl_val(
    model: TransferLearningEmotion,
    x_val_gpu_u8: torch.Tensor,
    y_val_gpu: torch.Tensor,
    y_val_np: np.ndarray,
    df_val: Any,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
    model.eval()
    n = len(x_val_gpu_u8)
    total_loss = 0.0
    probs_list: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = x_val_gpu_u8[start:end].float().div_(255.0)
        yb = y_val_gpu[start:end]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(xb)
            loss = criterion(logits.float(), yb)
            probs = F.softmax(logits.float(), dim=1)
        bs = int(yb.size(0))
        total_loss += float(loss.item()) * bs
        probs_list.append(probs.cpu().numpy())

    val_loss = total_loss / max(n, 1)
    y_prob = np.concatenate(probs_list, axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = compute_multicultural_metrics(y_val_np, y_pred, y_prob, meta_df=df_val)
    return val_loss, y_pred, y_prob, metrics


def train_single_tl_model(
    arch_key: str,
    X_train_u8: np.ndarray,
    y_train: np.ndarray,
    X_val_u8: np.ndarray,
    y_val: np.ndarray,
    df_val: Any,
    base_cw: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 32,
    dropout: float = 0.7,
    cw_power: float = 0.35,
    label_smoothing: float = 0.05,
    img_size: int = 224,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains one pre-trained transfer learning model for `epochs` (20 epochs as in Paper Figs. 2 & 3)
    using `lr = 0.0001` and `batch_size = 32` (Paper Section 3.2) with online GPU augmentation,
    logging training/validation accuracy & loss at every epoch.
    """
    seed_everything(seed)
    arch_key = normalize_tl_arch_key(arch_key)
    display_name = PAPER_TL_ARCHITECTURES[arch_key]

    print("\n" + "=" * 78)
    print(
        f"[Transfer Learning] Training {display_name} ({arch_key}) | "
        f"epochs={epochs}, lr={lr}, batch_size={batch_size}, dropout={dropout}, img_size={img_size}x{img_size}"
    )
    print("=" * 78, flush=True)

    model = TransferLearningEmotion(
        arch=arch_key,
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        dropout=dropout,
        freeze_initial=True,
        input_is_normalized_1ch=False,
    ).to(device=device)
    if img_size and img_size > 0:
        model.backbone_input_size = int(img_size)
    else:
        img_size = int(model.backbone_input_size)

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  [{display_name}] Total params: {n_total_params:,} | "
        f"Trainable (fine-tuned later layers + head): {n_trainable_params:,} | "
        f"Frozen initial params: {n_total_params - n_trainable_params:,}",
        flush=True,
    )

    t_start = time.time()
    ema = ModelEMA(model, decay=0.995)

    # Differential LR: backbone fine-tuning at lr, new custom head at 3x lr (within AdamW)
    backbone_params = [p for p in model.trainable_features.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    trainable_params = backbone_params + head_params
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr},
            {"params": head_params, "lr": lr * 3.0},
        ],
        lr=lr,
        weight_decay=2e-4,
    )

    warmup_epochs = 2
    def lr_lambda(ep_idx: int) -> float:
        if ep_idx < warmup_epochs:
            return float(ep_idx + 1) / float(warmup_epochs)
        prog = float(ep_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if cw_power > 0.0:
        cw = torch.pow(base_cw.float(), cw_power)
        cw = (cw / cw.mean()).to(device)
    else:
        cw = None

    train_criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    x_tr_gpu = torch.from_numpy(X_train_u8).to(device=device, dtype=torch.uint8).unsqueeze(1)
    y_tr_gpu = torch.from_numpy(y_train).to(device=device, dtype=torch.long)
    x_va_gpu = torch.from_numpy(X_val_u8).to(device=device, dtype=torch.uint8).unsqueeze(1)
    y_va_gpu = torch.from_numpy(y_val).to(device=device, dtype=torch.long)

    n_train = len(y_tr_gpu)
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "val_roc_auc": [],
        "lr": [],
    }
    epoch_records: List[Dict[str, Any]] = []

    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_epoch = 0
    best_metrics: Dict[str, Any] = {}
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_source = "raw"

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        perm = torch.randperm(n_train, device=device)
        running_loss = 0.0
        n_seen = 0
        tr_preds_all: List[np.ndarray] = []
        tr_targets_all: List[np.ndarray] = []

        for step_idx, start in enumerate(range(0, n_train, batch_size)):
            idx = perm[start : start + batch_size]
            if idx.numel() <= 1:
                continue
            xb = apply_online_paper_aug_gpu(x_tr_gpu[idx], target_size=img_size)
            yb = y_tr_gpu[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = train_criterion(logits.float(), yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
            optimizer.step()
            if (step_idx & 3) == 0:
                ema.update()

            bs_cur = int(yb.size(0))
            running_loss += float(loss.detach().item()) * bs_cur
            n_seen += bs_cur
            tr_preds_all.append(logits.detach().argmax(dim=1).cpu().numpy())
            tr_targets_all.append(yb.detach().cpu().numpy())

        ema.update()
        cur_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        train_loss = running_loss / max(n_seen, 1)
        tr_p_np = np.concatenate(tr_preds_all, axis=0)
        tr_y_np = np.concatenate(tr_targets_all, axis=0)
        train_acc = float(accuracy_score(tr_y_np, tr_p_np))
        train_f1 = float(f1_score(tr_y_np, tr_p_np, average="macro", zero_division=0))

        val_loss_raw, _, _, m_raw = evaluate_tl_val(
            model, x_va_gpu, y_va_gpu, y_val, df_val, val_criterion, device=device
        )
        val_loss_ema, _, _, m_ema = evaluate_tl_val(
            ema.ema_model, x_va_gpu, y_va_gpu, y_val, df_val, val_criterion, device=device
        )

        if m_ema["macro_f1"] >= m_raw["macro_f1"]:
            cur_m = m_ema
            cur_val_loss = val_loss_ema
            cur_model = ema.ema_model
            cur_src = "ema"
        else:
            cur_m = m_raw
            cur_val_loss = val_loss_raw
            cur_model = model
            cur_src = "raw"

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(cur_val_loss, 5))
        history["train_acc"].append(round(train_acc, 5))
        history["val_acc"].append(round(cur_m["accuracy"], 5))
        history["train_f1"].append(round(train_f1, 5))
        history["val_f1"].append(round(cur_m["macro_f1"], 5))
        history["val_roc_auc"].append(round(cur_m["roc_auc_ovr_macro"], 5))
        history["lr"].append(cur_lr)

        is_best = cur_m["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = cur_m["macro_f1"]
            best_val_acc = cur_m["accuracy"]
            best_epoch = epoch
            best_metrics = cur_m
            best_source = cur_src
            best_state_dict = {k: v.detach().cpu().clone() for k, v in cur_model.state_dict().items()}

        ep_sec = round(time.time() - ep_t0, 2)
        epoch_records.append(
            {
                "epoch": epoch,
                "lr": cur_lr,
                "train_loss": round(train_loss, 4),
                "val_loss": round(cur_val_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_accuracy": round(cur_m["accuracy"], 4),
                "train_f1": round(train_f1, 4),
                "val_macro_f1": round(cur_m["macro_f1"], 4),
                "val_roc_auc": round(cur_m["roc_auc_ovr_macro"], 4),
                "source": cur_src,
                "epoch_sec": ep_sec,
            }
        )

        print(
            f"  [{display_name}] Ep {epoch:02d}/{epochs:02d} | "
            f"tr_loss={train_loss:.4f} tr_acc={train_acc*100:5.2f}% tr_f1={train_f1:.4f} | "
            f"val_loss={cur_val_loss:.4f} val_acc={cur_m['accuracy']*100:5.2f}% "
            f"val_f1={cur_m['macro_f1']:.4f} ({cur_src}) auc={cur_m['roc_auc_ovr_macro']:.4f} | "
            f"best_f1={best_val_f1:.4f} (ep {best_epoch}) [{ep_sec:.1f}s]"
            + (" <- best" if is_best else ""),
            flush=True,
        )

    del x_tr_gpu, y_tr_gpu, x_va_gpu, y_va_gpu
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    assert best_state_dict is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()}, strict=True)
    model.eval()

    # End-to-end verification on raw X_val_u8 images through full model(x)
    e2e_preds, e2e_probs = predict_tl_split(model, X_val_u8, device=device, batch_size=128)
    best_metrics = compute_multicultural_metrics(y_val, e2e_preds, e2e_probs, meta_df=df_val)
    best_val_f1 = best_metrics["macro_f1"]
    best_val_acc = best_metrics["accuracy"]

    total_sec = round(time.time() - t_start, 2)
    cfg = {
        "arch": arch_key,
        "display_name": display_name,
        "num_classes": len(CLASS_NAMES),
        "img_size": img_size,
        "crop_mode": "ssd_square_pad",
        "ssd_margin": 1.25,
        "use_clahe": True,
        "dropout": dropout,
        "freeze_initial": True,
        "head_layers": f"GlobalAveragePooling -> FC(128) -> FC(64) -> Dropout({dropout}) -> FC(7)",
        "lr": lr,
        "head_lr": lr * 3.0,
        "batch_size": batch_size,
        "epochs": epochs,
        "cw_power": cw_power,
        "label_smoothing": label_smoothing,
        "n_parameters": n_total_params,
        "n_trainable_parameters": n_trainable_params,
        "selected_artifact": best_source,
        "seed": seed,
    }

    ckpt_path = CHECKPOINTS_DIR / f"{arch_key}_multicultural.pt"
    save_tl_checkpoint(
        model=model,
        checkpoint_path=ckpt_path,
        config=cfg,
        val_metrics=best_metrics,
        epoch=best_epoch,
        history=history,
    )
    ckpt_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)

    summary = {
        "arch": arch_key,
        "display_name": display_name,
        "model_name": f"{display_name} (Transfer Learning)",
        "checkpoint_path": ckpt_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_size_mb": ckpt_mb,
        "n_parameters": n_total_params,
        "n_trainable_parameters": n_trainable_params,
        "best_epoch": best_epoch,
        "epochs_run": epochs,
        "train_sec": total_sec,
        "config": cfg,
        "val_metrics_single_view": best_metrics,
        "history": history,
        "epoch_records": epoch_records,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"{arch_key}_val_tuning.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"  ==> [{display_name} DONE (E2E verified)] Best Ep={best_epoch}/{epochs} | "
        f"Val Acc={best_val_acc*100:.2f}% | Val Macro-F1={best_val_f1*100:.2f}% | "
        f"Val AUC={best_metrics['roc_auc_ovr_macro']:.4f} | "
        f"Saved -> {ckpt_path.relative_to(REPO_ROOT).as_posix()} ({ckpt_mb} MB) in {total_sec:.1f}s",
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train 6 Pre-trained Transfer Learning models from the reference paper on multicultural dataset."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(PAPER_TL_ARCHITECTURES.keys()),
        help="Subset or all of: vgg16 vgg19 inception_v3 densenet201 resnet50 efficientnet_b4",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs (Paper Figs. 2 & 3 use 20 epochs)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Fine-tuning learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.7, help="Dropout rate in custom head (Paper Section 3.2: 0.7)")
    parser.add_argument("--img-size", type=int, default=0, help="Backbone input resolution (0 = model default: 160 or 192)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip models whose checkpoint already exists")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert_not_test_split("train")
    assert_not_test_split("val")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    # Load 160x160 uint8 cache into memory; online GPU augmentation upsamples to args.img_size (224x224)
    X_train_u8, y_train, _ = load_multicultural_split_cache("train", img_size=160)
    X_val_u8, y_val, df_val = load_multicultural_split_cache("val", img_size=160)
    base_cw = compute_class_weights(DEFAULT_MULTICULTURAL_MANIFEST)
    print(
        f"[Data] Loaded train={len(X_train_u8):,} and val={len(X_val_u8):,} "
        f"(160x160 uint8 -> {args.img_size}x{args.img_size}) in {time.time()-t0:.2f}s on {device} (test split strictly locked).",
        flush=True,
    )

    all_summaries: Dict[str, Any] = {}
    for m_name in args.models:
        arch_key = normalize_tl_arch_key(m_name)
        ckpt_p = CHECKPOINTS_DIR / f"{arch_key}_multicultural.pt"
        json_p = RESULTS_DIR / f"{arch_key}_val_tuning.json"
        if args.skip_existing and ckpt_p.exists() and json_p.exists():
            print(f"[Skip] {arch_key}: checkpoint already exists at {ckpt_p}", flush=True)
            all_summaries[arch_key] = json.loads(json_p.read_text(encoding="utf-8"))
            continue

        summary = train_single_tl_model(
            arch_key=arch_key,
            X_train_u8=X_train_u8,
            y_train=y_train,
            X_val_u8=X_val_u8,
            y_val=y_val,
            df_val=df_val,
            base_cw=base_cw,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            dropout=args.dropout,
            img_size=args.img_size,
            seed=args.seed,
        )
        all_summaries[arch_key] = summary

    combined_path = RESULTS_DIR / "transfer_learning_val_summary.json"
    if combined_path.exists() and len(all_summaries) < len(PAPER_TL_ARCHITECTURES):
        existing = json.loads(combined_path.read_text(encoding="utf-8"))
        existing.update(all_summaries)
        all_summaries = existing
    combined_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[All Done] Saved combined validation summary -> {combined_path.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
