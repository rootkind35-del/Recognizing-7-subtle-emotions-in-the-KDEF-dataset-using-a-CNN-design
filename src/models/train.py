"""
Training Pipeline for KDEF Facial Emotion Recognition.
Core Integrator Implementation.

Features:
- Clean dataset loading with automatic corrupt MD5 frame exclusion
- Aspect-aware facial crop and CLAHE preprocessing
- Linear Warmup (5 epochs) + Cosine Annealing LR schedule
- AdamW optimizer with decoupled weight decay
- Exponential Moving Average (EMA) model weight tracking to prevent degradation
- Cross-Entropy with Label Smoothing
- Mixed Precision (AMP) training for fast RTX 4060 GPU execution
- Strict validation-only model checkpoint selection (zero test leakage)
"""

import copy
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.dataset import (
    CLASS_NAMES,
    DEFAULT_KDEF_MEAN,
    DEFAULT_KDEF_STD,
    KDEFDataset,
    get_transforms
)
from src.models.model import KDEFResSENet, build_model


# ---------------------------------------------------------------------------
# 1. Exponential Moving Average (EMA) Helper
# ---------------------------------------------------------------------------
class ModelEMA:
    """
    Maintains moving average of model parameters:
    shadow_v = decay * shadow_v + (1 - decay) * current_v.
    Smooths optimization trajectories and prevents single-epoch validation drops.
    """
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone()

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.shadow.items():
                if k in msd and msd[k].dtype.is_floating_point:
                    v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k in msd:
                msd[k].copy_(v)
        model.load_state_dict(msd)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


# ---------------------------------------------------------------------------
# 2. Training Loop
# ---------------------------------------------------------------------------
def train_model(
    data_dir: Optional[Union[str, Path]] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    model_name: str = "kdef_res_se_net",
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    min_lr: float = 1e-6,
    weight_decay: float = 1e-3,
    warmup_epochs: int = 5,
    label_smoothing: float = 0.1,
    use_ema: bool = True,
    ema_decay: float = 0.995,
    use_crop: bool = True,
    use_clahe: bool = True,
    img_size: int = 128,
    aug_mode: str = "enhanced",
    num_workers: int = 2,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Trains the emotion recognition model strictly on train split,
    monitoring validation performance on val split, and saves the best checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using compute device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Resolve paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    if checkpoint_dir is None:
        checkpoint_dir = repo_root / "checkpoints"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "optimized_best_model.pt"

    # 1. Datasets & DataLoaders
    train_tf = get_transforms(split="train", mean=DEFAULT_KDEF_MEAN, std=DEFAULT_KDEF_STD, aug_mode=aug_mode)
    val_tf = get_transforms(split="val", mean=DEFAULT_KDEF_MEAN, std=DEFAULT_KDEF_STD, aug_mode=aug_mode)

    train_ds = KDEFDataset(
        data_dir=data_dir,
        split="train",
        use_crop=use_crop,
        use_clahe=use_clahe,
        img_size=img_size,
        in_channels=1,
        transform=train_tf,
        exclude_corrupt=True
    )
    val_ds = KDEFDataset(
        data_dir=data_dir,
        split="val",
        use_crop=use_crop,
        use_clahe=use_clahe,
        img_size=img_size,
        in_channels=1,
        transform=val_tf,
        exclude_corrupt=True
    )

    print(f"[Train] Clean train samples: {len(train_ds)} | Clean val samples: {len(val_ds)}")
    assert len(train_ds) == 3349, f"Expected 3,349 clean train images, got {len(train_ds)}"
    assert len(val_ds) == 559, f"Expected 559 clean val images, got {len(val_ds)}"

    # Determine num_workers based on OS
    effective_workers = num_workers if os.name == "posix" else 0
    pin_mem = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_mem
    )

    # 2. Build Model
    model = build_model(
        model_name=model_name,
        num_classes=len(CLASS_NAMES),
        in_channels=1
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model: {model.__class__.__name__} | Total Parameters: {num_params:,}")

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        ratio = min_lr / lr
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # EMA
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None

    # 4. Training Loop
    best_val_f1 = -1.0
    best_epoch = -1
    best_val_acc = 0.0
    best_train_f1 = 0.0
    history = {
        "train_loss": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_acc": [],
        "lr": []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            if ema is not None:
                ema.update(model)

            running_loss += loss.item() * xb.size(0)
            train_preds.append(logits.detach().argmax(dim=1).cpu().numpy())
            train_targets.append(yb.cpu().numpy())

        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        train_loss = running_loss / len(train_ds)
        train_preds_all = np.concatenate(train_preds)
        train_targets_all = np.concatenate(train_targets)
        train_f1 = float(f1_score(train_targets_all, train_preds_all, average="macro"))

        # Validation phase: evaluate with EMA weights if enabled
        val_model = copy.deepcopy(model)
        if ema is not None:
            ema.apply_to(val_model)
        val_model.eval()

        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = val_model(xb)
                        loss = criterion(logits, yb)
                else:
                    logits = val_model(xb)
                    loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)
                val_preds.append(logits.argmax(dim=1).cpu().numpy())
                val_targets.append(yb.cpu().numpy())

        val_loss = val_loss / len(val_ds)
        val_preds_all = np.concatenate(val_preds)
        val_targets_all = np.concatenate(val_targets)
        val_acc = float(accuracy_score(val_targets_all, val_preds_all))
        val_f1 = float(f1_score(val_targets_all, val_preds_all, average="macro"))

        history["train_loss"].append(round(train_loss, 4))
        history["train_macro_f1"].append(round(train_f1, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_macro_f1"].append(round(val_f1, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["lr"].append(curr_lr)

        epoch_time = time.time() - epoch_t0

        # Check for best model strictly on validation Macro-F1
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_train_f1 = train_f1

            # Save best checkpoint
            checkpoint_payload = {
                "epoch": epoch,
                "model_name": model_name,
                "state_dict": val_model.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_accuracy": best_val_acc,
                "train_macro_f1": best_train_f1,
                "num_params": num_params,
                "config": {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "label_smoothing": label_smoothing,
                    "use_ema": use_ema,
                    "ema_decay": ema_decay,
                    "use_crop": use_crop,
                    "use_clahe": use_clahe,
                    "img_size": img_size
                }
            }
            torch.save(checkpoint_payload, best_model_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs or is_best:
            best_mark = " *** BEST ***" if is_best else ""
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} || "
                f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} "
                f"(LR: {curr_lr:.2e}, {epoch_time:.1f}s){best_mark}"
            )

    total_time = time.time() - start_time
    print(f"\n[Train] Training complete in {total_time:.1f}s.")
    print(f"[Train] Best Validation Macro-F1: {best_val_f1:.4f} (Acc: {best_val_acc:.4f}) at Epoch {best_epoch}")
    print(f"[Train] Best checkpoint saved to: {best_model_path}")

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "train_macro_f1_at_best": round(best_train_f1, 4),
        "train_val_gap": round(best_train_f1 - best_val_f1, 4),
        "checkpoint_path": str(best_model_path),
        "history": history,
        "total_time_s": round(total_time, 2)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train KDEF Facial Emotion Recognition Model")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
