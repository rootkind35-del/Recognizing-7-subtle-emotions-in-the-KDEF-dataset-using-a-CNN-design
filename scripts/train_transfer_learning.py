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
  - Pre-trained backbone used as feature extractor (initial layers frozen to retain general features
    like edges, textures, and patterns; later layers fine-tuned on the target dataset).
  - Custom classification head:
    Global Average Pooling -> FC(128) -> FC(64) -> Dropout(0.7) -> FC(num_classes=7) with Softmax.
  - Data augmentation to mitigate class imbalance: rotation (p=0.7, +-3..6 deg), distortion, horizontal flipping,
    combined with inverse-frequency class weights.
  - Fine-tuning hyperparameters: lr = 0.0001 (1e-4), 20 epochs (matching Figs. 2 & 3 of the paper).
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

    def __init__(self, model: nn.Module, decay: float = 0.992):
        self.decay = float(decay)
        self.updates = 0
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if k in msd:
                src = msd[k].detach()
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(src, alpha=1.0 - d)
                else:
                    v.copy_(src)


def apply_paper_image_transform_gpu(x_u8: torch.Tensor, mode: str = "rot_distort") -> torch.Tensor:
    """
    Applies the paper's Section 3.1 data augmentations (Rotation +-3..5 deg with p=0.7,
    Distortion grid/affine warp, and Horizontal Flipping) to uint8 image batch `(B, 1, H, W)`
    and returns float32 in `[0, 1]`.
    """
    x = x_u8.float().div_(255.0)
    if mode == "orig":
        return x
    if mode == "flip":
        return torch.flip(x, dims=[3])

    b, c, h, w = x.shape
    flip_mask = torch.rand(b, device=x.device) < 0.5
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])

    rot_mask = (torch.rand(b, device=x.device) < 0.7).float()
    angles = rot_mask * ((torch.rand(b, device=x.device) * 2.0 - 1.0) * (4.0 * math.pi / 180.0))
    scales = torch.empty(b, device=x.device).uniform_(0.94, 1.06)
    shear = torch.empty(b, device=x.device).uniform_(-0.035, 0.035)
    tx = torch.empty(b, device=x.device).uniform_(-0.035, 0.035)
    ty = torch.empty(b, device=x.device).uniform_(-0.035, 0.035)

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
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


@torch.no_grad()
def extract_frozen_features(
    model: TransferLearningEmotion,
    images_u8: np.ndarray,
    device: torch.device,
    transform_mode: str = "orig",
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Runs the frozen initial feature extractor (`model.frozen_features`) once over `images_u8`
    and returns a CPU `bfloat16` tensor `(N, C, H', W')`.
    Since `model.frozen_features` has `requires_grad=False` and stays in `.eval()` mode,
    its output is invariant across epochs for a given view.
    """
    model.eval()
    n = len(images_u8)
    out_tensor: Optional[torch.Tensor] = None

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb_u8 = torch.from_numpy(images_u8[start:end]).to(device=device, dtype=torch.uint8).unsqueeze(1)
        xb = apply_paper_image_transform_gpu(xb_u8, mode=transform_mode)
        if xb.shape[1] == 1:
            xb = xb.repeat(1, 3, 1, 1)
        xb = (xb - model.imagenet_mean) / model.imagenet_std
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            fb = model.frozen_features(xb).to(dtype=torch.bfloat16)
        if out_tensor is None:
            out_tensor = torch.empty((n, *fb.shape[1:]), dtype=torch.bfloat16, device="cpu")
        out_tensor[start:end].copy_(fb.cpu())

    assert out_tensor is not None
    return out_tensor


def augment_feature_batch_gpu(f_orig: torch.Tensor, f_aug: torch.Tensor) -> torch.Tensor:
    """
    Combines original and augmented (rotated/distorted/flipped) frozen feature maps
    with stochastic interpolation and subtle Multiplicative Distortion on GPU.
    """
    b = f_orig.shape[0]
    use_aug = (torch.rand(b, 1, 1, 1, device=f_orig.device) < 0.65).to(f_orig.dtype)
    f = f_orig * (1.0 - use_aug) + f_aug * use_aug
    # Subtle multiplicative feature distortion (simulates fine grid/lighting variation)
    if f.training if hasattr(f, "training") else True:
        noise = torch.empty_like(f).uniform_(0.95, 1.05)
        f = f * noise
    return f


@torch.no_grad()
def evaluate_trainable_on_val_features(
    model: TransferLearningEmotion,
    f_val_cpu: torch.Tensor,
    y_val_gpu: torch.Tensor,
    y_val_np: np.ndarray,
    df_val: Any,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
    model.eval()
    n = len(f_val_cpu)
    total_loss = 0.0
    probs_list: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        fb = f_val_cpu[start:end].to(device=device, non_blocking=True)
        yb = y_val_gpu[start:end]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model.head(model.trainable_features(fb))
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
    batch_size: int = 128,
    dropout: float = 0.7,
    cw_power: float = 0.5,
    label_smoothing: float = 0.05,
    img_size: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains one pre-trained transfer learning model for `epochs` (20 epochs as in Paper Figs. 2 & 3)
    using `lr = 0.0001` (Paper Section 3.2), logging training/validation accuracy & loss at every epoch.
    """
    seed_everything(seed)
    arch_key = normalize_tl_arch_key(arch_key)
    display_name = PAPER_TL_ARCHITECTURES[arch_key]

    print("\n" + "=" * 78)
    print(
        f"[Transfer Learning] Training {display_name} ({arch_key}) | "
        f"epochs={epochs}, lr={lr}, dropout={dropout}, img_size={img_size}x{img_size}"
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

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  [{display_name}] Total params: {n_total_params:,} | "
        f"Trainable (fine-tuned later layers + head): {n_trainable_params:,} | "
        f"Frozen initial params: {n_total_params - n_trainable_params:,}",
        flush=True,
    )

    t_start = time.time()
    t_ext = time.time()
    print(f"  [{display_name}] Extracting frozen initial stage features (orig + paper-augmented views)...", flush=True)
    f_tr_orig = extract_frozen_features(model, X_train_u8, device=device, transform_mode="orig", batch_size=256)
    f_tr_aug = extract_frozen_features(model, X_train_u8, device=device, transform_mode="rot_distort", batch_size=256)
    f_va_orig = extract_frozen_features(model, X_val_u8, device=device, transform_mode="orig", batch_size=256)
    print(
        f"  [{display_name}] Extracted frozen features {tuple(f_tr_orig.shape)} in {time.time()-t_ext:.1f}s. "
        f"Fine-tuning later layers + head (GAP->128->64->Dropout({dropout})->7) for {epochs} epochs...",
        flush=True,
    )

    ema = ModelEMA(model, decay=0.992)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    if cw_power > 0.0:
        cw = torch.pow(base_cw.float(), cw_power)
        cw = (cw / cw.mean()).to(device)
    else:
        cw = None

    train_criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    y_tr_gpu = torch.from_numpy(y_train).to(device=device, dtype=torch.long)
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
        perm_cpu = torch.randperm(n_train)
        running_loss = 0.0
        n_seen = 0
        tr_preds_all: List[np.ndarray] = []
        tr_targets_all: List[np.ndarray] = []

        for start in range(0, n_train, batch_size):
            idx_cpu = perm_cpu[start : start + batch_size]
            if idx_cpu.numel() <= 1:
                continue
            f_o = f_tr_orig[idx_cpu].to(device=device, non_blocking=True)
            f_a = f_tr_aug[idx_cpu].to(device=device, non_blocking=True)
            fb = augment_feature_batch_gpu(f_o, f_a)
            yb = y_tr_gpu[idx_cpu.to(device)]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model.head(model.trainable_features(fb))
                loss = train_criterion(logits.float(), yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
            optimizer.step()
            ema.update(model)

            bs_cur = int(yb.size(0))
            running_loss += float(loss.detach().item()) * bs_cur
            n_seen += bs_cur
            tr_preds_all.append(logits.detach().argmax(dim=1).cpu().numpy())
            tr_targets_all.append(yb.detach().cpu().numpy())

        cur_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        train_loss = running_loss / max(n_seen, 1)
        tr_p_np = np.concatenate(tr_preds_all, axis=0)
        tr_y_np = np.concatenate(tr_targets_all, axis=0)
        train_acc = float(accuracy_score(tr_y_np, tr_p_np))
        train_f1 = float(f1_score(tr_y_np, tr_p_np, average="macro", zero_division=0))

        val_loss_raw, _, _, m_raw = evaluate_trainable_on_val_features(
            model, f_va_orig, y_va_gpu, y_val, df_val, val_criterion, device=device
        )
        val_loss_ema, _, _, m_ema = evaluate_trainable_on_val_features(
            ema.ema_model, f_va_orig, y_va_gpu, y_val, df_val, val_criterion, device=device
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

    del f_tr_orig, f_tr_aug, f_va_orig, y_tr_gpu, y_va_gpu
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
        "head_layers": "GlobalAveragePooling -> FC(128) -> FC(64) -> Dropout(0.7) -> FC(7)",
        "lr": lr,
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
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate (Paper Section 3.2: 0.0001)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.7, help="Dropout rate in custom head (Paper Section 3.2: 0.7)")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true", help="Skip models whose checkpoint already exists")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert_not_test_split("train")
    assert_not_test_split("val")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    X_train_u8, y_train, _ = load_multicultural_split_cache("train", img_size=args.img_size)
    X_val_u8, y_val, df_val = load_multicultural_split_cache("val", img_size=args.img_size)
    base_cw = compute_class_weights(DEFAULT_MULTICULTURAL_MANIFEST)
    print(
        f"[Data] Loaded train={len(X_train_u8):,} and val={len(X_val_u8):,} "
        f"({args.img_size}x{args.img_size} uint8) in {time.time()-t0:.2f}s on {device} (test split strictly locked).",
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
    combined_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[All Done] Saved combined validation summary -> {combined_path.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
