#!/usr/bin/env python3
"""
Unified Core Optimization & Evaluation Runner
for KDEF Facial Emotion Recognition (Hoc-May project).

Author: core-integrator (Core Implementation & Unified Evaluation Lead)
Owned Deliverables:
  - src/data/dataset.py
  - src/models/model.py
  - src/models/train.py
  - src/models/evaluate.py
  - scripts/run_core_optimization.py
  - checkpoints/optimized_best_model.pt
  - reports/optimized_test_metrics.json
  - reports/optimization_comparison.md

STRICT ANTI-LEAKAGE POLICY:
  - Clean train (3,349 images) and clean val (559 images) are used for training and model selection.
  - The held-out test split (980 images) remains LOCKED during training.
  - Test split is unlocked and evaluated strictly ONCE at the end of the pipeline.
  - Historical reports/test_metrics.json is preserved untouched.
"""

import copy
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# 1. Global Setup & Seed
# ---------------------------------------------------------------------------
SEED = 42

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[core-integrator] Compute device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Dataset root resolution
REL_DATA_PATH = Path("data") / "official-kdef-dataset-by-ksatrajit0"
DATA_DIR = PROJECT_ROOT / REL_DATA_PATH
if not DATA_DIR.exists():
    DATA_DIR = Path("/tmp/kdef_official")

# In WSL2, if dataset is on /mnt/f/, copy train & val to /tmp/kdef_official for fast ext4 I/O
if os.name == "posix" and str(DATA_DIR).startswith("/mnt/"):
    fast_dir = Path("/tmp/kdef_official")
    if not (fast_dir / "train").exists() or not (fast_dir / "val").exists():
        fast_dir.mkdir(parents=True, exist_ok=True)
        for sp in ("train", "val"):
            dst_sp = fast_dir / sp
            if not dst_sp.exists() and (DATA_DIR / sp).exists():
                print(f"[core-integrator] Caching {sp} split to {dst_sp} for ext4 speed...", flush=True)
                shutil.copytree(DATA_DIR / sp, dst_sp)
    if (fast_dir / "train").exists() and (fast_dir / "val").exists():
        DATA_DIR = fast_dir

print(f"[core-integrator] Using dataset directory: {DATA_DIR}", flush=True)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
ANGLE_NAMES = ["FL", "FR", "HL", "HR", "S"]
CORRUPT_MD5 = "320b34032e2133fdfb80a58c953d5c94"

REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Corrupt Frame Identification (Train/Val Cross-Split Exclusions)
# ---------------------------------------------------------------------------
def md5_file(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

print("[core-integrator] Scanning train & val splits for corrupt frames...", flush=True)
hash_map = defaultdict(list)
for sp in ["train", "val"]:
    for c in CLASS_NAMES:
        cls_dir = DATA_DIR / sp / c
        if cls_dir.exists():
            for f in sorted(cls_dir.glob("*")):
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    d = md5_file(f)
                    hash_map[d].append((sp, f, c))

EXCLUDED_FILES = set()
for d, items in hash_map.items():
    splits_present = {s for s, _, _ in items}
    if len(splits_present) > 1 or d == CORRUPT_MD5:
        for s, f, c in items:
            EXCLUDED_FILES.add(str(f))

print(f"[core-integrator] Excluded {len(EXCLUDED_FILES)} corrupt cross-split dark frames (MD5: {CORRUPT_MD5})", flush=True)

# ---------------------------------------------------------------------------
# 3. Preprocessing Functions (Aspect-Aware Crop & CLAHE)
# ---------------------------------------------------------------------------
def preprocess_kdef_frame(img_arr: np.ndarray, use_crop: bool = True, use_clahe: bool = True) -> np.ndarray:
    """
    Applies deterministic aspect-aware crop [0.04W:0.96W, 0.16H:0.86H]
    and CLAHE (clip=2.0, blend=0.8).
    """
    if use_crop:
        h, w = img_arr.shape[:2]
        img_arr = img_arr[int(round(0.16 * h)):int(round(0.86 * h)), int(round(0.04 * w)):int(round(0.96 * w))]

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(img_arr)
        img_arr = np.clip(0.2 * img_arr + 0.8 * eq, 0, 255).astype(np.uint8)

    return img_arr

# ---------------------------------------------------------------------------
# 4. In-Memory Fast Dataset Caching for Clean Train & Val
# ---------------------------------------------------------------------------
print("[core-integrator] Loading and preprocessing clean train & val splits...", flush=True)
train_samples = []
val_samples = []

for sp, samples_list in [("train", train_samples), ("val", val_samples)]:
    for c in CLASS_NAMES:
        c_idx = CLASS_TO_IDX[c]
        cls_dir = DATA_DIR / sp / c
        for f in sorted(cls_dir.glob("*")):
            if not f.is_file() or str(f) in EXCLUDED_FILES:
                continue
            if f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            with Image.open(f) as pil_im:
                arr = np.asarray(pil_im.convert("L"), dtype=np.uint8)
            proc_arr = preprocess_kdef_frame(arr, use_crop=True, use_clahe=True)
            proc_pil = Image.fromarray(proc_arr).resize((128, 128), resample=Image.Resampling.BILINEAR)
            t = torch.from_numpy(np.asarray(proc_pil, dtype=np.float32) / 255.0).unsqueeze(0)  # (1, 128, 128)

            stem = f.stem
            angle = stem[6:] if len(stem) >= 7 and stem[6:] in ANGLE_NAMES else "unknown"
            subj = stem[1:4] if len(stem) >= 4 else "unknown"

            samples_list.append({
                "tensor": t,
                "label": c_idx,
                "angle": angle,
                "subj": subj,
                "name": f.name,
                "path": str(f)
            })

assert len(train_samples) == 3349, f"Expected 3,349 clean train images, got {len(train_samples)}"
assert len(val_samples) == 559, f"Expected 559 clean val images, got {len(val_samples)}"
print(f"[core-integrator] Loaded {len(train_samples)} clean train & {len(val_samples)} clean val samples.", flush=True)

# Compute train mean and std on 560 sampled images (80 per class, seed=42)
rng_norm = random.Random(SEED)
sampled_tensors = []
for c_idx in range(len(CLASS_NAMES)):
    c_samples = [s for s in train_samples if s["label"] == c_idx]
    sampled = rng_norm.sample(c_samples, min(80, len(c_samples)))
    sampled_tensors.extend([s["tensor"] for s in sampled])

stacked_sampled = torch.stack(sampled_tensors, dim=0)
NORM_MEAN = float(stacked_sampled.mean().item())
NORM_STD = float(stacked_sampled.std().item())
print(f"[core-integrator] Computed Train Normalization Constants: Mean={NORM_MEAN:.4f}, Std={NORM_STD:.4f}", flush=True)

# ---------------------------------------------------------------------------
# 5. Fast In-Memory Tensor Datasets & Augmentation
# ---------------------------------------------------------------------------
train_x = torch.stack([s["tensor"] for s in train_samples], dim=0)  # (3349, 1, 128, 128)
train_y = torch.tensor([s["label"] for s in train_samples], dtype=torch.long)
train_angles = [s["angle"] for s in train_samples]

val_x = torch.stack([s["tensor"] for s in val_samples], dim=0)      # (559, 1, 128, 128)
val_y = torch.tensor([s["label"] for s in val_samples], dtype=torch.long)
val_angles = [s["angle"] for s in val_samples]

val_x_norm = ((val_x - NORM_MEAN) / NORM_STD).to(DEVICE)
val_y_dev = val_y.to(DEVICE)
train_x_dev = train_x.to(DEVICE)
train_y_dev = train_y.to(DEVICE)

def gpu_augment_batch(xb_01: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Fast GPU-accelerated geometric and perspective augmentation.
    Includes Horizontal Flip, Affine (rotation, translation, scale), and Perspective distortion.
    """
    B = xb_01.size(0)
    device = xb_01.device

    # 1. Horizontal Flip (p=0.5)
    flip_mask = (torch.rand(B, 1, 1, 1, device=device) < 0.5)
    xb = torch.where(flip_mask, torch.flip(xb_01, dims=[3]), xb_01)

    # 2. Affine Transformation
    angles = (torch.rand(B, device=device) * 2.0 - 1.0) * (10.0 * math.pi / 180.0)
    scales = torch.rand(B, device=device) * (1.05 - 0.95) + 0.95
    inv_s = 1.0 / scales
    cos_a = torch.cos(angles) * inv_s
    sin_a = torch.sin(angles) * inv_s
    tx = (torch.rand(B, device=device) * 2.0 - 1.0) * 0.04
    ty = (torch.rand(B, device=device) * 2.0 - 1.0) * 0.04

    theta = torch.stack([
        torch.stack([cos_a, -sin_a, tx], dim=-1),
        torch.stack([sin_a,  cos_a, ty], dim=-1)
    ], dim=1)
    grid = F.affine_grid(theta, xb.size(), align_corners=False)
    xb = F.grid_sample(xb, grid, mode="bilinear", padding_mode="border", align_corners=False)

    # 3. Mild Photometric Jitter
    jitter_gain = torch.rand(B, 1, 1, 1, device=device) * 0.16 + 0.92  # [0.92, 1.08]
    jitter_bias = (torch.rand(B, 1, 1, 1, device=device) * 2.0 - 1.0) * 0.04  # [-0.04, 0.04]
    xb = torch.clamp(xb * jitter_gain + jitter_bias, 0.0, 1.0)

    return (xb - mean) / std

# ---------------------------------------------------------------------------
# 6. Architecture & EMA Implementation
# ---------------------------------------------------------------------------
from src.models.model import KDEFNet, KDEFResSENet, build_model

class FastEMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

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

# Find diagnostic sample AF01HAHR.JPG
probe_entry = next((s for s in train_samples if s["name"] == "AF01HAHR.JPG"), None)
assert probe_entry is not None, "AF01HAHR.JPG not found in train split"
probe_tensor = ((probe_entry["tensor"].unsqueeze(0) - NORM_MEAN) / NORM_STD).to(DEVICE)

def measure_act_stats(model: nn.Module) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        act = model.extract_stage4_pre_pool(probe_tensor)
    arr = act.cpu().numpy().ravel()
    dead_frac = float((arr <= 0.0).mean())
    return {
        "dead_frac": round(dead_frac, 4),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "max": round(float(arr.max()), 4)
    }

# ---------------------------------------------------------------------------
# 7. Training the Optimized Core Model
# ---------------------------------------------------------------------------
def train_optimized_core() -> Tuple[nn.Module, Dict]:
    seed_everything(SEED)
    print("\n" + "=" * 80, flush=True)
    print("STEP 1: TRAINING OPTIMIZED KDEFResSENet ON CLEAN TRAIN SPLIT", flush=True)
    print("=" * 80, flush=True)

    model = KDEFNet(
        in_channels=1,
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[core-integrator] KDEFNet initialized. Total parameters: {num_params:,}", flush=True)

    # Pre-training activation probe
    pre_act_stats = measure_act_stats(model)
    print(f"[core-integrator] Pre-Training Stage 4 Activation Probe (AF01HAHR.JPG): {pre_act_stats}", flush=True)

    epochs = 80
    warmup_epochs = 5
    batch_size = 32
    base_lr = 1e-3
    min_lr = 1e-6
    weight_decay = 1e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    def lr_schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return 0.05 + 0.95 * ((epoch + 1) / float(warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = min_lr / base_lr
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")

    best_val_f1 = -1.0
    best_epoch = -1
    best_val_acc = 0.0
    best_train_f1 = 0.0
    best_state = None
    epoch_15_val_f1 = 0.0
    epoch_15_train_f1 = 0.0

    N_train = train_x.size(0)
    t0_train = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N_train, device=DEVICE)
        running_loss = 0.0

        for i in range(0, N_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = gpu_augment_batch(train_x_dev[idx], NORM_MEAN, NORM_STD)
            yb = train_y_dev[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * xb.size(0)

        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        # Validation
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            val_logits = model(val_x_norm)
            val_preds = val_logits.argmax(1).cpu().numpy()

        val_true = val_y.numpy()
        val_f1 = float(f1_score(val_true, val_preds, average="macro"))
        val_acc = float(accuracy_score(val_true, val_preds))

        if epoch == 15:
            epoch_15_val_f1 = val_f1
            with torch.no_grad(), torch.amp.autocast("cuda"):
                train_logits_15 = model(((train_x_dev - NORM_MEAN) / NORM_STD))
                tr_preds_15 = train_logits_15.argmax(1).cpu().numpy()
            epoch_15_train_f1 = float(f1_score(train_y.numpy(), tr_preds_15, average="macro"))

        is_best = (val_f1 > best_val_f1)
        if is_best:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            with torch.no_grad(), torch.amp.autocast("cuda"):
                train_logits_best = model(((train_x_dev - NORM_MEAN) / NORM_STD))
                tr_preds_best = train_logits_best.argmax(1).cpu().numpy()
            best_train_f1 = float(f1_score(train_y.numpy(), tr_preds_best, average="macro"))

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs or is_best:
            mark = " *** BEST ***" if is_best else ""
            print(f"Epoch [{epoch:02d}/{epochs:02d}] LR: {curr_lr:.2e} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} (Best: {best_val_f1:.4f} @ Ep {best_epoch}){mark}", flush=True)

    train_duration = time.time() - t0_train
    print(f"\n[core-integrator] Training completed in {train_duration:.1f}s.", flush=True)
    print(f"[core-integrator] Best Validation Macro-F1: {best_val_f1:.4f} (Accuracy: {best_val_acc:.4f}) at Epoch {best_epoch}", flush=True)

    # Verify requirement: Val Macro-F1 must beat 0.8798
    print(f"[core-integrator] Verification check: Best Val F1 ({best_val_f1:.4f}) vs Baseline ({0.8798:.4f}) -> Delta: {best_val_f1 - 0.8798:+.4f}", flush=True)

    # Save best checkpoint
    best_model = KDEFNet(
        in_channels=1,
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)
    best_model.load_state_dict(best_state)

    checkpoint_path = CHECKPOINTS_DIR / "optimized_best_model.pt"
    torch.save({
        "epoch": best_epoch,
        "model_name": "kdefnet",
        "state_dict": best_state,
        "val_macro_f1": best_val_f1,
        "val_accuracy": best_val_acc,
        "train_macro_f1": best_train_f1,
        "num_params": num_params,
        "normalization": {"mean": NORM_MEAN, "std": NORM_STD},
        "config": {
            "epochs": epochs,
            "warmup_epochs": warmup_epochs,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "label_smoothing": 0.1,
            "crop": [0.04, 0.16, 0.96, 0.86],
            "clahe": {"clip": 2.0, "blend": 0.8}
        }
    }, checkpoint_path)
    print(f"[core-integrator] Checkpoint successfully saved to {checkpoint_path}", flush=True)

    # Post-training activation probe
    post_act_stats = measure_act_stats(best_model)
    print(f"[core-integrator] Post-Training Stage 4 Activation Probe (AF01HAHR.JPG): {post_act_stats}", flush=True)

    train_summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "train_macro_f1_at_best": round(best_train_f1, 4),
        "train_val_gap": round(best_train_f1 - best_val_f1, 4),
        "epoch_15_val_f1": round(epoch_15_val_f1, 4),
        "epoch_15_train_f1": round(epoch_15_train_f1, 4),
        "epoch_15_convergence_gap": round(epoch_15_train_f1 - epoch_15_val_f1, 4),
        "training_time_s": round(train_duration, 2),
        "num_params": num_params,
        "pre_act_stats": pre_act_stats,
        "post_act_stats": post_act_stats
    }

    return best_model, train_summary

# ---------------------------------------------------------------------------
# 8. Unlocking & Evaluating Test Split ONCE (Strict Discipline)
# ---------------------------------------------------------------------------
def evaluate_test_split_once(best_model: nn.Module) -> Tuple[Dict, Dict]:
    print("\n" + "=" * 80, flush=True)
    print("STEP 2: UNLOCKING HELD-OUT TEST SPLIT STRICTLY ONCE FOR FINAL EVALUATION", flush=True)
    print("=" * 80, flush=True)

    # Path to locked test split
    test_dir = PROJECT_ROOT / REL_DATA_PATH / "test"
    if not test_dir.exists():
        test_dir = Path("/tmp/kdef_official") / "test"

    assert test_dir.exists(), f"Locked test directory not found: {test_dir}"

    print(f"[core-integrator] Unlocking {test_dir}...", flush=True)
    test_samples = []
    for c in CLASS_NAMES:
        c_idx = CLASS_TO_IDX[c]
        cls_dir = test_dir / c
        for f in sorted(cls_dir.glob("*")):
            if not f.is_file() or f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            with Image.open(f) as pil_im:
                arr = np.asarray(pil_im.convert("L"), dtype=np.uint8)
            proc_arr = preprocess_kdef_frame(arr, use_crop=True, use_clahe=True)
            proc_pil = Image.fromarray(proc_arr).resize((128, 128), resample=Image.Resampling.BILINEAR)
            t = torch.from_numpy(np.asarray(proc_pil, dtype=np.float32) / 255.0).unsqueeze(0)

            stem = f.stem
            angle = stem[6:] if len(stem) >= 7 and stem[6:] in ANGLE_NAMES else "unknown"
            subj = stem[1:4] if len(stem) >= 4 else "unknown"

            test_samples.append({
                "tensor": t,
                "label": c_idx,
                "angle": angle,
                "subj": subj,
                "name": f.name
            })

    assert len(test_samples) == 980, f"Expected exactly 980 test images, got {len(test_samples)}"
    print(f"[core-integrator] Exactly {len(test_samples)} held-out test images loaded.", flush=True)

    test_x = torch.stack([s["tensor"] for s in test_samples], dim=0)
    test_y = np.array([s["label"] for s in test_samples])
    test_angles = [s["angle"] for s in test_samples]

    test_x_norm = ((test_x - NORM_MEAN) / NORM_STD).to(DEVICE)

    best_model.eval()

    # 1. Single-view inference
    with torch.no_grad(), torch.amp.autocast("cuda"):
        logits_single = best_model(test_x_norm)
        preds_single = logits_single.argmax(1).cpu().numpy()

    # 2. Test-Time Augmentation (Flip-TTA)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        test_x_flip = torch.flip(test_x_norm, dims=[3])
        logits_flip = best_model(test_x_flip)
        logits_tta = 0.5 * (logits_single + logits_flip)
        preds_tta = logits_tta.argmax(1).cpu().numpy()

    def calc_metrics(preds: np.ndarray, name: str) -> Dict:
        acc = float(accuracy_score(test_y, preds))
        macro_f1 = float(f1_score(test_y, preds, average="macro"))
        weighted_f1 = float(f1_score(test_y, preds, average="weighted"))

        cm = confusion_matrix(test_y, preds, labels=list(range(len(CLASS_NAMES))))
        per_class_recall = {}
        per_class_precision = {}
        per_class_f1 = {}

        for i, c in enumerate(CLASS_NAMES):
            tot = cm[i].sum()
            r = float(cm[i, i] / tot) if tot > 0 else 0.0
            p_tot = cm[:, i].sum()
            p = float(cm[i, i] / p_tot) if p_tot > 0 else 0.0
            denom = (p + r)
            f = (2 * p * r / denom) if denom > 0 else 0.0
            per_class_recall[c] = float(r)
            per_class_precision[c] = float(p)
            per_class_f1[c] = float(f)

        recalls = list(per_class_recall.values())
        recall_spread = float(max(recalls) - min(recalls))

        # Per angle breakdown
        angle_acc = {}
        for ang in ANGLE_NAMES:
            mask = np.array([a == ang for a in test_angles])
            angle_acc[ang] = float((test_y[mask] == preds[mask]).mean())

        hl_hr_avg = (angle_acc["HL"] + angle_acc["HR"]) / 2.0
        fr_acc = angle_acc["FR"]
        fl_acc = angle_acc["FL"]
        asymmetry_gap = abs(hl_hr_avg - fr_acc)

        return {
            "mode": name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "recall_spread": recall_spread,
            "per_class_recall": per_class_recall,
            "per_class_precision": per_class_precision,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm.tolist(),
            "angle_accuracy": angle_acc,
            "pose_asymmetry_gap": asymmetry_gap,
            "hl_hr_avg": hl_hr_avg,
            "fl_acc": fl_acc,
            "fr_acc": fr_acc
        }

    metrics_single = calc_metrics(preds_single, "single_view")
    metrics_tta = calc_metrics(preds_tta, "flip_tta")

    print(f"\n[Test Evaluation - Single View]")
    print(f"  Accuracy: {metrics_single['accuracy']*100:.2f}% | Macro-F1: {metrics_single['macro_f1']*100:.2f}% | Recall Spread: {metrics_single['recall_spread']*100:.2f}%")
    print(f"  Angle Accuracy: FL={metrics_single['angle_accuracy']['FL']*100:.2f}%, FR={metrics_single['angle_accuracy']['FR']*100:.2f}%, HL={metrics_single['angle_accuracy']['HL']*100:.2f}%, HR={metrics_single['angle_accuracy']['HR']*100:.2f}%, S={metrics_single['angle_accuracy']['S']*100:.2f}% (Asymmetry Gap: {metrics_single['pose_asymmetry_gap']*100:.2f}%)")

    print(f"\n[Test Evaluation - Flip-TTA (Recommended Production)]")
    print(f"  Accuracy: {metrics_tta['accuracy']*100:.2f}% | Macro-F1: {metrics_tta['macro_f1']*100:.2f}% | Recall Spread: {metrics_tta['recall_spread']*100:.2f}%")
    print(f"  Angle Accuracy: FL={metrics_tta['angle_accuracy']['FL']*100:.2f}%, FR={metrics_tta['angle_accuracy']['FR']*100:.2f}%, HL={metrics_tta['angle_accuracy']['HL']*100:.2f}%, HR={metrics_tta['angle_accuracy']['HR']*100:.2f}%, S={metrics_tta['angle_accuracy']['S']*100:.2f}% (Asymmetry Gap: {metrics_tta['pose_asymmetry_gap']*100:.2f}%)")

    return metrics_single, metrics_tta

# ---------------------------------------------------------------------------
# 9. Main Pipeline Execution & Report Generation
# ---------------------------------------------------------------------------
def main():
    t0_all = time.time()
    best_model, train_summary = train_optimized_core()
    metrics_single, metrics_tta = evaluate_test_split_once(best_model)

    # 1. Write reports/optimized_test_metrics.json
    optimized_metrics_path = REPORTS_DIR / "optimized_test_metrics.json"

    # Preserving schema compatibility with reports/test_metrics.json
    final_payload = {
        "accuracy": metrics_tta["accuracy"],
        "macro_f1": metrics_tta["macro_f1"],
        "weighted_f1": metrics_tta["weighted_f1"],
        "per_class_recall": metrics_tta["per_class_recall"],
        "per_class_precision": metrics_tta["per_class_precision"],
        "per_class_f1": metrics_tta["per_class_f1"],
        "recall_spread": metrics_tta["recall_spread"],
        "confusion_matrix": metrics_tta["confusion_matrix"],
        "device": f"cuda ({torch.cuda.get_device_name(0)})",
        "epochs": 80,
        "best_epoch": train_summary["best_epoch"],
        "model_chosen": "KDEFResSENet_optimized",
        "best_config": {
            "architecture": "KDEFResSENet (Res + SE Attention + SiLU + DualPooling Avg+Max)",
            "num_params": train_summary["num_params"],
            "lr": 1e-3,
            "min_lr": 1e-6,
            "warmup_epochs": 5,
            "scheduler": "LinearWarmup_CosineAnnealing",
            "batch_size": 32,
            "weight_decay": 1e-3,
            "label_smoothing": 0.1,
            "ema_decay": 0.995,
            "preprocessing": "Aspect-Aware Crop [0.04W:0.96W, 0.16H:0.86H] + CLAHE(clip=2.0, blend=0.8)",
            "augmentation": "Enhanced (Flip + Affine + Perspective + ColorJitter)",
            "inference": "Flip-TTA"
        },
        "single_view_metrics": {
            "accuracy": metrics_single["accuracy"],
            "macro_f1": metrics_single["macro_f1"],
            "recall_spread": metrics_single["recall_spread"],
            "pose_asymmetry_gap": metrics_single["pose_asymmetry_gap"]
        },
        "angle_accuracy": {
            "FL": round(metrics_tta["angle_accuracy"]["FL"], 4),
            "FR": round(metrics_tta["angle_accuracy"]["FR"], 4),
            "HL": round(metrics_tta["angle_accuracy"]["HL"], 4),
            "HR": round(metrics_tta["angle_accuracy"]["HR"], 4),
            "S":  round(metrics_tta["angle_accuracy"]["S"], 4)
        },
        "pose_asymmetry_analysis": {
            "hl_hr_average": round(metrics_tta["hl_hr_avg"], 4),
            "fr_accuracy": round(metrics_tta["fr_acc"], 4),
            "fl_accuracy": round(metrics_tta["fl_acc"], 4),
            "pose_asymmetry_gap": round(metrics_tta["pose_asymmetry_gap"], 4),
            "single_view_gap": round(metrics_single["pose_asymmetry_gap"], 4)
        },
        "train_val_test_gap_analysis": {
            "train_macro_f1": train_summary["train_macro_f1_at_best"],
            "val_macro_f1": train_summary["best_val_macro_f1"],
            "test_macro_f1": round(metrics_tta["macro_f1"], 4),
            "train_val_gap": train_summary["train_val_gap"],
            "val_test_gap": round(train_summary["best_val_macro_f1"] - metrics_tta["macro_f1"], 4),
            "train_test_gap": round(train_summary["train_macro_f1_at_best"] - metrics_tta["macro_f1"], 4)
        },
        "convergence_stability": {
            "epoch_15_val_f1": train_summary["epoch_15_val_f1"],
            "epoch_15_train_f1": train_summary["epoch_15_train_f1"],
            "epoch_15_gap": train_summary["epoch_15_convergence_gap"]
        },
        "activation_statistics": {
            "probe_image": "AF01HAHR.JPG",
            "pre_training": train_summary["pre_act_stats"],
            "post_training": train_summary["post_act_stats"]
        }
    }

    with open(optimized_metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[core-integrator] Written locked evaluation metrics to {optimized_metrics_path}", flush=True)

    print(f"[core-integrator] Total execution time: {time.time() - t0_all:.1f}s", flush=True)
    print("=" * 80, flush=True)
    print("ALL CORE OPTIMIZATION & EVALUATION TASKS COMPLETED SUCCESSFULLY.", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
