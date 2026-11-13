#!/usr/bin/env python3
"""
Controlled Architecture & Preprocessing Ablation Study
for KDEF Facial Emotion Recognition (f:/Hoc-May).

Author: arch-opt (Core Architecture & Preprocessing Researcher)
Owned Deliverables:
  - scripts/exp_arch_preproc.py
  - reports/exp_arch_results.json

STRICT ANTI-LEAKAGE POLICY:
  This script ONLY accesses the clean `train` (3,349 images) and `val` (559 images)
  splits of `data/official-kdef-dataset-by-ksatrajit0`. The held-out `test` split (980 images)
  is NEVER loaded, touched, or evaluated in any shape or form.
"""

import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Global Setup & Seed
# ---------------------------------------------------------------------------
SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[arch-opt] Using compute device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

# Dataset path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/tmp/kdef_official")
if not (DATA_DIR / "train").exists():
    DATA_DIR = PROJECT_ROOT / "data" / "official-kdef-dataset-by-ksatrajit0"

assert (DATA_DIR / "train").exists(), f"Train directory not found at {DATA_DIR / 'train'}"
assert (DATA_DIR / "val").exists(), f"Val directory not found at {DATA_DIR / 'val'}"

CLASS_NAMES = sorted([p.name for p in (DATA_DIR / "train").iterdir() if p.is_dir()])
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
print(f"[arch-opt] Found {NUM_CLASSES} classes: {CLASS_NAMES}", flush=True)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_JSON_PATH = REPORTS_DIR / "exp_arch_results.json"

# ---------------------------------------------------------------------------
# 2. Corrupt Frame Identification (Train/Val Cross-Split MD5 Exclusions)
# ---------------------------------------------------------------------------
# Identifies the 10 corrupt pitch-black frames with MD5 = 320b34032e2133fdfb80a58c953d5c94
hmap = defaultdict(list)
for split in ["train", "val"]:
    for c in CLASS_NAMES:
        for f in sorted((DATA_DIR / split / c).glob("*")):
            h = hashlib.md5(f.read_bytes()).hexdigest()
            hmap[h].append((split, f))

EXCLUDED_FILES = set()
for h, items in hmap.items():
    splits = {s for s, _ in items}
    if len(splits) > 1:
        for s, f in items:
            EXCLUDED_FILES.add(str(f))

print(f"[arch-opt] Total corrupt cross-split MD5 files excluded: {len(EXCLUDED_FILES)}", flush=True)

# ---------------------------------------------------------------------------
# 3. In-Memory Raw Image Caching (Fast Loading)
# ---------------------------------------------------------------------------
RAW_CACHE_FILE = Path("/tmp/kdef_raw_cache_arch.pt")
if RAW_CACHE_FILE.exists():
    t0 = time.time()
    raw_dataset = torch.load(RAW_CACHE_FILE, weights_only=False)
    print(f"[arch-opt] Loaded in-memory raw dataset from {RAW_CACHE_FILE} in {time.time()-t0:.2f}s", flush=True)
else:
    t0 = time.time()
    raw_dataset = {"train": [], "val": []}
    for split in ["train", "val"]:
        for c in CLASS_NAMES:
            c_idx = CLASS_TO_IDX[c]
            files = [f for f in sorted((DATA_DIR / split / c).glob("*")) if str(f) not in EXCLUDED_FILES]
            for f in files:
                im = Image.open(f).convert("L")
                arr = np.asarray(im, dtype=np.uint8)
                raw_dataset[split].append({
                    "arr": arr,
                    "label": c_idx,
                    "name": f.name,
                    "path": str(f)
                })
    torch.save(raw_dataset, RAW_CACHE_FILE)
    print(f"[arch-opt] Cached {len(raw_dataset['train'])} train & {len(raw_dataset['val'])} val images in {time.time()-t0:.2f}s", flush=True)

assert len(raw_dataset["train"]) == 3349, f"Expected 3,349 clean train images, got {len(raw_dataset['train'])}"
assert len(raw_dataset["val"]) == 559, f"Expected 559 clean val images, got {len(raw_dataset['val'])}"
print("[arch-opt] Strict anti-leakage check passed: exactly 3,349 clean train & 559 clean val images loaded.", flush=True)

SAMPLE_NAME = "AF01HAHR.JPG"
sample_entry = None
for item in raw_dataset["train"]:
    if item["name"] == SAMPLE_NAME:
        sample_entry = item
        break
assert sample_entry is not None, f"Could not find diagnostic sample {SAMPLE_NAME} in train split"
print(f"[arch-opt] Diagnostic activation probe image: {sample_entry['name']} (class {CLASS_NAMES[sample_entry['label']]})", flush=True)

# ---------------------------------------------------------------------------
# 4. Preprocessing Functions (Crop & CLAHE)
# ---------------------------------------------------------------------------
def prepare_preprocessed_dataset(use_crop=False, use_clahe=False, clahe_clip=2.0, clahe_blend=0.8, img_size=128):
    """
    Builds CPU float tensors in [0, 1] for train and val splits.
    Aspect-aware crop: [0.04W:0.96W, 0.16H:0.86H] eliminates excess studio background,
    top hair, and neck/shoulders, yielding a near 1:1 facial region.
    """
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)) if use_clahe else None
    tensors = {}
    for split in ["train", "val"]:
        imgs, labels = [], []
        for item in raw_dataset[split]:
            img = item["arr"]
            if use_crop:
                h, w = img.shape
                y1, y2 = int(round(0.16 * h)), int(round(0.86 * h))
                x1, x2 = int(round(0.04 * w)), int(round(0.96 * w))
                img = img[y1:y2, x1:x2]
            if clahe is not None:
                eq = clahe.apply(img)
                if clahe_blend < 1.0:
                    img = np.clip((1.0 - clahe_blend) * img + clahe_blend * eq, 0, 255).astype(np.uint8)
                else:
                    img = eq
            pil_img = Image.fromarray(img).resize((img_size, img_size), resample=Image.Resampling.BILINEAR)
            imgs.append(np.asarray(pil_img, dtype=np.float32) / 255.0)
            labels.append(item["label"])
        tensors[split] = (
            torch.from_numpy(np.stack(imgs, axis=0)).unsqueeze(1),
            torch.tensor(labels, dtype=torch.long)
        )

    # Compute mean and std strictly on 560 sampled train images (80 per class, seed=42)
    rng = random.Random(SEED)
    train_x, train_y = tensors["train"]
    sampled_indices = []
    for c_idx in range(NUM_CLASSES):
        class_indices = (train_y == c_idx).nonzero(as_tuple=True)[0].tolist()
        sampled_indices.extend(rng.sample(class_indices, 80))
    sampled_x = train_x[sampled_indices]
    norm_mean = float(sampled_x.mean().item())
    norm_std = float(sampled_x.std().item())

    # Preprocessed probe sample
    probe_raw = sample_entry["arr"]
    if use_crop:
        h, w = probe_raw.shape
        probe_raw = probe_raw[int(round(0.16 * h)):int(round(0.86 * h)), int(round(0.04 * w)):int(round(0.96 * w))]
    if clahe is not None:
        eq = clahe.apply(probe_raw)
        probe_raw = np.clip((1.0 - clahe_blend) * probe_raw + clahe_blend * eq, 0, 255).astype(np.uint8) if clahe_blend < 1.0 else eq
    probe_pil = Image.fromarray(probe_raw).resize((img_size, img_size), resample=Image.Resampling.BILINEAR)
    probe_tensor = torch.from_numpy(np.asarray(probe_pil, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    probe_tensor = (probe_tensor - norm_mean) / norm_std

    return tensors["train"], tensors["val"], norm_mean, norm_std, probe_tensor

# GPU geometric data augmentation matching baseline khong_erasing parameters
def gpu_augment_batch(xb_01, mean, std, deg=10.0, scale_range=(1.0/1.1, 1.1), translate=0.03):
    B = xb_01.size(0)
    device = xb_01.device
    flip_mask = (torch.rand(B, 1, 1, 1, device=device) < 0.5)
    xb = torch.where(flip_mask, torch.flip(xb_01, dims=[3]), xb_01)

    angles = (torch.rand(B, device=device) * 2.0 - 1.0) * (deg * math.pi / 180.0)
    scales = torch.rand(B, device=device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    inv_s = 1.0 / scales
    cos_a = torch.cos(angles) * inv_s
    sin_a = torch.sin(angles) * inv_s
    tx = (torch.rand(B, device=device) * 2.0 - 1.0) * translate if translate > 0 else torch.zeros(B, device=device)
    ty = (torch.rand(B, device=device) * 2.0 - 1.0) * translate if translate > 0 else torch.zeros(B, device=device)

    theta = torch.stack([
        torch.stack([cos_a, -sin_a, tx], dim=-1),
        torch.stack([sin_a,  cos_a, ty], dim=-1)
    ], dim=1)
    grid = F.affine_grid(theta, xb.size(), align_corners=False)
    xb = F.grid_sample(xb, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return (xb - mean) / std

# ---------------------------------------------------------------------------
# 5. Modular Neural Network Architecture Family
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, act_type="relu", use_res=False, use_se=False, drop2d=0.0):
        super().__init__()
        self.use_res = use_res
        self.use_se = use_se

        def make_act():
            if act_type == "silu": return nn.SiLU(inplace=True)
            if act_type == "leaky_relu": return nn.LeakyReLU(0.1, inplace=True)
            if act_type == "gelu": return nn.GELU()
            return nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = make_act()

        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c, reduction=8) if use_se else nn.Identity()

        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
            if (use_res and in_c != out_c) else (nn.Identity() if use_res else None)
        )
        self.act2 = make_act()
        self.drop = nn.Dropout2d(drop2d) if drop2d > 0 else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.use_res:
            out = out + self.shortcut(x)
        out = self.act2(out)
        out = self.drop(out)
        return self.pool(out)

class KDEFNetVariant(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), num_classes=7,
                 act_type="relu", use_res=False, use_se=False, pool_type="avg",
                 p1=0.4, p2=0.3, drop2d=0.0):
        super().__init__()
        self.pool_type = pool_type
        stages = []
        in_c = 1
        for i, out_c in enumerate(channels):
            d2 = drop2d if i >= 2 else 0.0
            stages.append(ConvBlock(in_c, out_c, act_type=act_type, use_res=use_res, use_se=use_se, drop2d=d2))
            in_c = out_c
        self.features = nn.Sequential(*stages)

        if pool_type == "dual":
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            feat_dim = channels[-1] * 2
        elif pool_type == "strip":
            self.strip_pool = nn.AdaptiveAvgPool2d((2, 1))
            feat_dim = channels[-1] * 2
        else: # "avg"
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = channels[-1]

        def make_head_act():
            if act_type == "silu": return nn.SiLU(inplace=True)
            if act_type == "leaky_relu": return nn.LeakyReLU(0.1, inplace=True)
            if act_type == "gelu": return nn.GELU()
            return nn.ReLU(inplace=True)

        self.head = nn.Sequential(
            nn.Dropout(p1),
            nn.Linear(feat_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            make_head_act(),
            nn.Dropout(p2),
            nn.Linear(256, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.features(x)
        if self.pool_type == "dual":
            p = torch.cat([self.avg_pool(f).flatten(1), self.max_pool(f).flatten(1)], dim=1)
        elif self.pool_type == "strip":
            p = self.strip_pool(f).flatten(1)
        else:
            p = self.avg_pool(f).flatten(1)
        return self.head(p)

    def extract_stage4_pre_pool(self, x):
        """Extract activation of Stage 4 before MaxPool2d (equivalent to features.3.5 in baseline KDEFNet)"""
        s0 = self.features[0](x)
        s1 = self.features[1](s0)
        s2 = self.features[2](s1)
        blk = self.features[3]
        out = blk.act1(blk.bn1(blk.conv1(s2)))
        out = blk.bn2(blk.conv2(out))
        out = blk.se(out)
        if blk.use_res:
            out = out + blk.shortcut(s2)
        act = blk.act2(out)
        return act

def measure_activation_stats(model, probe_tensor):
    model.eval()
    with torch.no_grad():
        act = model.extract_stage4_pre_pool(probe_tensor.to(DEVICE))
    arr = act.cpu().numpy().ravel()
    dead_frac = float((arr <= 0).mean())
    return {
        "dead_frac": round(dead_frac, 4),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "max": round(float(arr.max()), 4),
        "shape": list(act.shape)
    }

def evaluate_batched(model, x_data, y_data, norm_mean, norm_std, batch_size=64):
    model.eval()
    all_preds = []
    N = x_data.size(0)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = x_data[i:i + batch_size].to(DEVICE)
            xb = (xb - norm_mean) / norm_std
            with torch.amp.autocast("cuda"):
                logits = model(xb)
            all_preds.append(logits.argmax(1).cpu())
    preds = torch.cat(all_preds).numpy()
    targets = y_data.numpy()
    macro_f1 = float(f1_score(targets, preds, average="macro"))
    acc = float(accuracy_score(targets, preds))
    return macro_f1, acc, preds

# ---------------------------------------------------------------------------
# 6. Controlled Training & Evaluation Harness
# ---------------------------------------------------------------------------
def run_ablation_experiment(cfg, preproc_data):
    """
    Runs a single controlled ablation for 45 epochs with:
    - Linear Warmup (3 epochs) + Cosine Annealing (lr: 1e-3 -> 1e-5)
    - Batch size 32, label smoothing 0.1, weight decay 2e-4
    - Batched evaluation with low peak VRAM
    - Logs validation Macro-F1, Accuracy, Recall Spread, Epoch 15 convergence,
      Train-Val Gap, and Pre vs Post activation stats.
    """
    seed_everything(SEED)
    (train_x, train_y), (val_x, val_y), norm_mean, norm_std, probe_tensor = preproc_data

    model = KDEFNetVariant(
        act_type=cfg["act_type"],
        use_res=cfg["use_res"],
        use_se=cfg["use_se"],
        pool_type=cfg["pool_type"],
        drop2d=cfg.get("drop2d", 0.0)
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())

    # Pre-training activation stats
    pre_act_stats = measure_activation_stats(model, probe_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    max_epochs = 45
    warmup_epochs = 3

    def lr_schedule(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    best_preds = None
    epoch_15_val_f1 = 0.0

    N_train = train_x.size(0)
    batch_size = 32

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(N_train)
        for i in range(0, N_train, batch_size):
            idx = perm[i:i + batch_size]
            xb_01 = train_x[idx].to(DEVICE)
            xb = gpu_augment_batch(xb_01, norm_mean, norm_std)
            yb = train_y[idx].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Batched validation evaluation
        val_f1, val_acc, val_preds = evaluate_batched(model, val_x, val_y, norm_mean, norm_std, batch_size=64)

        if epoch == 15:
            epoch_15_val_f1 = val_f1

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_preds = val_preds.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == max_epochs or epoch == 1:
            print(f"    [Ep {epoch:02d}/{max_epochs}] Val F1: {val_f1:.4f} | Best: {best_val_f1:.4f} (@ep {best_epoch:02d}) | LR: {scheduler.get_last_lr()[0]:.1e}", flush=True)

    elapsed = time.time() - t0

    # Load best checkpoint for post-training metrics
    model.load_state_dict(best_state)
    post_act_stats = measure_activation_stats(model, probe_tensor)

    # Evaluate train F1 at best checkpoint
    best_train_f1, _, _ = evaluate_batched(model, train_x, train_y, norm_mean, norm_std, batch_size=64)

    val_true = val_y.numpy()
    val_acc = float(accuracy_score(val_true, best_preds))
    cm = confusion_matrix(val_true, best_preds, labels=list(range(NUM_CLASSES)))
    per_class_recall = {}
    for i, c in enumerate(CLASS_NAMES):
        total = cm[i].sum()
        r = float(cm[i, i] / total) if total > 0 else 0.0
        per_class_recall[c] = round(r, 4)

    recalls = list(per_class_recall.values())
    recall_spread = round(max(recalls) - min(recalls), 4)
    train_val_gap = round(best_train_f1 - best_val_f1, 4)

    return {
        "name": cfg["name"],
        "description": cfg["description"],
        "num_params": num_params,
        "best_epoch": best_epoch,
        "val_macro_f1": round(best_val_f1, 4),
        "val_accuracy": round(val_acc, 4),
        "val_recall_spread": recall_spread,
        "per_class_recall": per_class_recall,
        "train_f1": round(best_train_f1, 4),
        "train_val_gap": train_val_gap,
        "epoch_15_val_f1": round(epoch_15_val_f1, 4),
        "training_time_s": round(elapsed, 2),
        "pre_act_stats": pre_act_stats,
        "post_act_stats": post_act_stats,
        "config": {k: v for k, v in cfg.items() if k != "preproc"}
    }

# ---------------------------------------------------------------------------
# 7. Ablation Plan Execution
# ---------------------------------------------------------------------------
def main():
    print("=" * 80, flush=True)
    print("STARTING CONTROLLED ARCHITECTURE & PREPROCESSING ABLATIONS", flush=True)
    print("=" * 80, flush=True)

    # 1. Preload Preprocessed Datasets
    print("[arch-opt] Generating preprocessed datasets for ablations...", flush=True)
    preproc_raw = prepare_preprocessed_dataset(use_crop=False, use_clahe=False)
    print(f"  - Raw Squash: mean={preproc_raw[2]:.4f}, std={preproc_raw[3]:.4f}", flush=True)

    preproc_crop = prepare_preprocessed_dataset(use_crop=True, use_clahe=False)
    print(f"  - Aspect Crop: mean={preproc_crop[2]:.4f}, std={preproc_crop[3]:.4f}", flush=True)

    preproc_crop_clahe = prepare_preprocessed_dataset(use_crop=True, use_clahe=True, clahe_clip=2.0, clahe_blend=0.8)
    print(f"  - Crop + CLAHE: mean={preproc_crop_clahe[2]:.4f}, std={preproc_crop_clahe[3]:.4f}", flush=True)

    # 2. Define Ablation Matrix
    ablation_configs = [
        # --- Preprocessing Ablations (on baseline plain KDEFNet) ---
        {
            "name": "1_plain_kdefnet_raw_squash",
            "description": "Baseline: Plain KDEFNet on raw 562x762 squashed directly to 128x128 (no crop, no CLAHE)",
            "preproc": preproc_raw,
            "act_type": "relu",
            "use_res": False,
            "use_se": False,
            "pool_type": "avg",
            "drop2d": 0.0
        },
        {
            "name": "2_plain_kdefnet_aspect_crop",
            "description": "Plain KDEFNet with Aspect-Aware Studio Face Crop (eliminating hair top/shoulders background)",
            "preproc": preproc_crop,
            "act_type": "relu",
            "use_res": False,
            "use_se": False,
            "pool_type": "avg",
            "drop2d": 0.0
        },
        {
            "name": "3_plain_kdefnet_crop_clahe",
            "description": "Plain KDEFNet with Aspect-Aware Crop + CLAHE (micro-contrast on brows/mouth creases)",
            "preproc": preproc_crop_clahe,
            "act_type": "relu",
            "use_res": False,
            "use_se": False,
            "pool_type": "avg",
            "drop2d": 0.0
        },
        # --- Core Architecture Ablations (on winning Crop+CLAHE preprocessing) ---
        {
            "name": "4_res_kdefnet",
            "description": "ResKDEFNet: Adding Residual Skip Connections across Stages 1-4 (ReLU, AvgPool)",
            "preproc": preproc_crop_clahe,
            "act_type": "relu",
            "use_res": True,
            "use_se": False,
            "pool_type": "avg",
            "drop2d": 0.0
        },
        {
            "name": "5_res_se_kdefnet",
            "description": "ResSENet: Residual Connections + Squeeze-and-Excitation (SE) Channel Attention (ReLU, AvgPool)",
            "preproc": preproc_crop_clahe,
            "act_type": "relu",
            "use_res": True,
            "use_se": True,
            "pool_type": "avg",
            "drop2d": 0.0
        },
        {
            "name": "6_res_se_silu",
            "description": "ResSENet with SiLU (Swish) Activation (eliminating ReLU dead neurons in Stage 4)",
            "preproc": preproc_crop_clahe,
            "act_type": "silu",
            "use_res": True,
            "use_se": True,
            "pool_type": "avg",
            "drop2d": 0.0
        },
        {
            "name": "7_res_se_leakyrelu",
            "description": "ResSENet with LeakyReLU(0.1) Activation (non-zero negative gradient flow)",
            "preproc": preproc_crop_clahe,
            "act_type": "leaky_relu",
            "use_res": True,
            "use_se": True,
            "pool_type": "avg",
            "drop2d": 0.0
        },
        {
            "name": "8_kdef_res_se_net_dualpool_final",
            "description": "KDEFResSENet (Final Core Architecture): Res + SE Attention + SiLU + Dual Pooling (Avg+Max) + Crop + CLAHE",
            "preproc": preproc_crop_clahe,
            "act_type": "silu",
            "use_res": True,
            "use_se": True,
            "pool_type": "dual",
            "drop2d": 0.05
        }
    ]

    all_results = []
    for idx, cfg in enumerate(ablation_configs, 1):
        print(f"\n[{idx}/{len(ablation_configs)}] RUNNING ABLATION: {cfg['name']}", flush=True)
        print(f"  Description: {cfg['description']}", flush=True)
        preproc_data = cfg["preproc"]
        res = run_ablation_experiment(cfg, preproc_data)
        all_results.append(res)

        print(f"  --> Best Val Macro-F1: {res['val_macro_f1']:.4f} (@ epoch {res['best_epoch']}) | Val Acc: {res['val_accuracy']:.4f}", flush=True)
        print(f"  --> Epoch 15 Val F1: {res['epoch_15_val_f1']:.4f} | Recall Spread: {res['val_recall_spread']:.4f} | Train-Val Gap: {res['train_val_gap']:.4f}", flush=True)
        print(f"  --> Stage 4 Act Dead Frac: {res['pre_act_stats']['dead_frac']:.4f} (pre) -> {res['post_act_stats']['dead_frac']:.4f} (post) | Time: {res['training_time_s']:.1f}s", flush=True)

        # Progressive saving after each ablation
        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "benchmark_baseline_val_f1": 0.8798,
                "dataset": "official-kdef-dataset-by-ksatrajit0 (clean 3349 train / 559 val)",
                "seed": SEED,
                "ablations": all_results
            }, f, indent=2, ensure_ascii=False)
        print(f"  --> Updated {RESULTS_JSON_PATH}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("ALL ABLATIONS COMPLETED SUCCESSFULLY. SUMMARY TABLE:", flush=True)
    print("=" * 80, flush=True)
    header = f"{'Ablation Name':<35} | {'Val F1':<8} | {'Val Acc':<8} | {'Ep 15 F1':<9} | {'Spread':<8} | {'Gap':<8} | {'Dead Frac (Post)':<16}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['name']:<35} | {r['val_macro_f1']:<8.4f} | {r['val_accuracy']:<8.4f} | {r['epoch_15_val_f1']:<9.4f} | {r['val_recall_spread']:<8.4f} | {r['train_val_gap']:<8.4f} | {r['post_act_stats']['dead_frac']:<16.4f}")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
