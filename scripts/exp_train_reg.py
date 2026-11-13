#!/usr/bin/env python3
"""
Controlled Optimization, Loss, Regularization & Inference Ablation Study
for KDEF Facial Emotion Recognition (f:/Hoc-May).

Author: train-opt
Owned Deliverables:
  - scripts/exp_train_reg.py
  - reports/exp_train_results.json

STRICT ANTI-LEAKAGE POLICY:
  This script ONLY accesses the `train` (3,349 clean images) and `val` (559 clean images)
  splits of `data/official-kdef-dataset-by-ksatrajit0`. The held-out evaluation split
  is NEVER referenced, loaded, or touched.
"""

import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# 1. Reproducibility & Configuration
# ---------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 128
NUM_CLASSES = 7
CORRUPT_MD5 = "320b34032e2133fdfb80a58c953d5c94"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
ANGLE_NAMES = ["FL", "FR", "HL", "HR", "S"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 2. Clean Train/Val Split Discovery & In-Memory Caching (Anti-Leakage)
# ---------------------------------------------------------------------------
def resolve_repo_and_data():
    repo_root = Path(__file__).resolve().parent.parent
    rel_data = Path("data") / "official-kdef-dataset-by-ksatrajit0"
    src_data = repo_root / rel_data
    assert src_data.exists(), f"Dataset directory not found: {src_data}"

    # If running on Linux/WSL2 over /mnt/f, cache train & val to /tmp/kdef_official_trainval
    # for native ext4 I/O speed.
    if os.name == "posix" and str(src_data).startswith("/mnt/"):
        fast_dir = Path("/tmp/kdef_official")
        if not (fast_dir / "train").exists() or not (fast_dir / "val").exists():
            fast_dir.mkdir(parents=True, exist_ok=True)
            for sp in ("train", "val"):
                dst_sp = fast_dir / sp
                if not dst_sp.exists():
                    shutil.copytree(src_data / sp, dst_sp)
        data_dir = fast_dir
    else:
        data_dir = src_data

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    assert train_dir.exists() and val_dir.exists(), "train/ or val/ missing"
    return repo_root, train_dir, val_dir


def load_clean_train_val(train_dir: Path, val_dir: Path):
    """
    Replicates the exact EDA 3.5 & 4.1-4.2 clean split logic from
    notebooks/CNN_KDEF_FER.ipynb using ONLY train and val directories.
    """
    hash_map = defaultdict(list)
    raw_counts = {"train": 0, "val": 0}
    for split_name, split_dir in [("train", train_dir), ("val", val_dir)]:
        for cls_name in CLASS_NAMES:
            cls_dir = split_dir / cls_name
            for fpath in sorted(cls_dir.glob("*")):
                if fpath.is_file():
                    raw_counts[split_name] += 1
                    digest = md5_file(fpath)
                    hash_map[digest].append((split_name, fpath, cls_name))

    # Exclude cross-split duplicate MD5 groups (the 10 corrupt dark frames)
    excluded_files = set()
    excluded_rel_paths = []
    for digest, items in hash_map.items():
        splits_present = {s for s, _, _ in items}
        if len(splits_present) > 1 or digest == CORRUPT_MD5:
            for s, fp, c in items:
                excluded_files.add(str(fp))
                excluded_rel_paths.append(f"{s}/{c}/{fp.name}")

    excluded_rel_paths.sort()
    assert len(excluded_files) == 10, f"Expected 10 corrupt files excluded, got {len(excluded_files)}"

    # Compute exact MEAN_TRAIN and STD_TRAIN from 560 sampled train images (SEED=42)
    rng_dl = random.Random(SEED)
    sampled_train_files = []
    for cls_name in CLASS_NAMES:
        cls_files = [
            f for f in sorted((train_dir / cls_name).glob("*"))
            if f.is_file() and str(f) not in excluded_files
        ]
        sampled_train_files.extend(rng_dl.sample(cls_files, 80))

    base_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.ToTensor(),
    ])

    total_sum, total_sq_sum, total_px = 0.0, 0.0, 0
    for fpath in sampled_train_files:
        with Image.open(fpath) as img:
            x = base_tf(img)
        total_sum += x.sum().item()
        total_sq_sum += (x ** 2).sum().item()
        total_px += x.numel()

    mean_train = total_sum / total_px
    std_train = float(np.sqrt(max(total_sq_sum / total_px - mean_train ** 2, 1e-12)))

    # Pre-decode and resize all clean train (3,349) and val (559) images in RAM
    # as (1, 128, 128) float32 tensors in [0, 1] (matching base_tf output).
    def cache_split(split_dir: Path):
        tensors = []
        labels = []
        angles = []
        subjects = []
        for cls_name in CLASS_NAMES:
            cls_idx = CLASS_TO_IDX[cls_name]
            for fpath in sorted((split_dir / cls_name).glob("*")):
                if not fpath.is_file() or str(fpath) in excluded_files:
                    continue
                with Image.open(fpath) as img:
                    t = base_tf(img)  # (1, 128, 128) float32 in [0, 1]
                stem = fpath.stem  # e.g. AF01ANFL
                angle = stem[6:]
                subj = stem[1:4]
                assert angle in ANGLE_NAMES, f"Unexpected angle {angle} in {fpath.name}"
                tensors.append(t)
                labels.append(cls_idx)
                angles.append(angle)
                subjects.append(subj)
        return torch.stack(tensors, dim=0), torch.tensor(labels, dtype=torch.long), angles, subjects

    train_x, train_y, train_angles, train_subjects = cache_split(train_dir)
    val_x, val_y, val_angles, val_subjects = cache_split(val_dir)

    assert len(train_x) == 3349, f"Expected 3349 clean train images, got {len(train_x)}"
    assert len(val_x) == 559, f"Expected 559 clean val images, got {len(val_x)}"

    split_info = {
        "raw_train_count": raw_counts["train"],
        "raw_val_count": raw_counts["val"],
        "clean_train_count": int(len(train_x)),
        "clean_val_count": int(len(val_x)),
        "excluded_corrupt_count": len(excluded_rel_paths),
        "excluded_corrupt_md5": CORRUPT_MD5,
        "excluded_corrupt_files": excluded_rel_paths,
        "mean_train": round(mean_train, 6),
        "std_train": round(std_train, 6),
        "train_subjects": len(set(train_subjects)),
        "val_subjects": len(set(val_subjects)),
        "subject_overlap_train_val": len(set(train_subjects).intersection(set(val_subjects))),
    }
    return (train_x, train_y, train_angles), (val_x, val_y, val_angles), mean_train, std_train, split_info


class InMemoryKDEFDataset(Dataset):
    def __init__(self, x_tensors: torch.Tensor, y_tensors: torch.Tensor, transform=None):
        self.x = x_tensors
        self.y = y_tensors
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        img = self.x[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[idx]


# ---------------------------------------------------------------------------
# 3. Augmentation Pipelines (Operating on [0, 1] 1x128x128 Tensor -> Normalized)
# ---------------------------------------------------------------------------
def build_transforms(aug_mode: str, mean_train: float, std_train: float):
    norm = transforms.Normalize([mean_train], [std_train])
    eval_tf = transforms.Compose([norm])

    if aug_mode == "du":
        # Notebook baseline with RandomErasing(p=0.25)
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, scale=(1 / 1.1, 1.1)),
            norm,
            transforms.RandomErasing(p=0.25),
        ])
    elif aug_mode == "khong_erasing":
        # Notebook best_config augmentation (flip + affine, no erasing)
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, scale=(1 / 1.1, 1.1)),
            norm,
        ])
    elif aug_mode == "enhanced_geom_photo":
        # Enhanced geometry (Perspective for FL/FR profiles + slight translation/rotation/scale)
        # + mild photometric jitter (brightness/contrast for dark studio frames)
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.14, p=0.30),
            transforms.RandomAffine(degrees=12, translate=(0.04, 0.04), scale=(0.90, 1.10)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            norm,
        ])
    else:
        raise ValueError(f"Unknown aug_mode: {aug_mode}")

    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# 4. Reference Architecture: KDEFNet (Exact match to notebooks/CNN_KDEF_FER.ipynb)
# ---------------------------------------------------------------------------
class KDEFNet(nn.Module):
    """
    4-block custom CNN from notebooks/CNN_KDEF_FER.ipynb (1,241,191 params).
    Kept fixed across all training/optimization/regularization ablations for
    strict apples-to-apples comparison.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, ch=(32, 64, 128, 256), drop=(0.4, 0.3)):
        super().__init__()
        blocks = []
        c_in = 1
        for c_out in ch:
            blocks.append(nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ))
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop[0]),
            nn.Linear(ch[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop[1]),
            nn.Linear(256, num_classes),
        )
        self._init_he()

    def _init_he(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 5. Loss Functions & Mixup Regularization
# ---------------------------------------------------------------------------
class Poly1FocalCrossEntropyLoss(nn.Module):
    """
    Combines:
      - Label-Smoothed Cross-Entropy
      - Focal modulation (1 - pt)^gamma to focus gradients on hard negative classes
      - Poly-1 term eps * (1 - pt) (Leng et al., ICLR 2022) to sharpen decision boundaries
      - Optional mild class-difficulty weights for hard classes (angry, fear, sad)
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        label_smoothing: float = 0.08,
        gamma: float = 1.2,
        poly_eps: float = 0.5,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.poly_eps = poly_eps
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        if targets.dim() == 1:
            with torch.no_grad():
                true_dist = torch.full_like(log_probs, self.label_smoothing / (self.num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            sample_weights = self.class_weights[targets] if self.class_weights is not None else 1.0
        else:
            # Soft targets from Mixup
            true_dist = targets
            pt = (probs * true_dist).sum(dim=-1)
            sample_weights = (true_dist * self.class_weights.unsqueeze(0)).sum(dim=-1) if self.class_weights is not None else 1.0

        ce = -(true_dist * log_probs).sum(dim=-1)
        focal_term = ((1.0 - pt).clamp(min=1e-6) ** self.gamma) * ce
        poly1_loss = focal_term + self.poly_eps * (1.0 - pt)
        return (poly1_loss * sample_weights).mean()


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2, num_classes: int = NUM_CLASSES, label_smoothing: float = 0.08):
    """Applies mild Mixup across batch samples, converting targets to smoothed soft distributions."""
    with torch.no_grad():
        y_onehot = torch.full((x.size(0), num_classes), label_smoothing / (num_classes - 1), device=x.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1.0 - label_smoothing)

    if alpha <= 0.0:
        return x, y_onehot

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    mixed_y = lam * y_onehot + (1.0 - lam) * y_onehot[perm]
    return mixed_x, mixed_y


# ---------------------------------------------------------------------------
# 6. Model EMA (Exponential Moving Average) & SWA (Stochastic Weight Averaging)
# ---------------------------------------------------------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.992):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        # Warmup EMA decay in early steps so initial random weights are forgotten quickly
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(model_v, alpha=1.0 - d)
            else:
                ema_v.copy_(model_v)


@torch.no_grad()
def update_bn_stats(model: nn.Module, loader: DataLoader, device: torch.device):
    """Recomputes clean BatchNorm running_mean/running_var over the training loader."""
    momenta = {}
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            momenta[m] = m.momentum
            m.momentum = None
            m.train()

    if not momenta:
        return

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        model(xb)

    for m, mom in momenta.items():
        m.momentum = mom
    model.eval()


# ---------------------------------------------------------------------------
# 7. Pure NumPy/PyTorch Metrics & TTA Inference Evaluation
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, angles: list = None):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    per_class_recall = {}
    per_class_precision = {}
    per_class_f1 = {}
    f1_list = []
    recall_list = []

    for i, cname in enumerate(CLASS_NAMES):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - tp)
        fp = float(cm[:, i].sum() - tp)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_recall[cname] = round(rec, 4)
        per_class_precision[cname] = round(prec, 4)
        per_class_f1[cname] = round(f1, 4)
        f1_list.append(f1)
        recall_list.append(rec)

    macro_f1 = float(np.mean(f1_list))
    acc = float((y_true == y_pred).mean())
    recall_spread = float(max(recall_list) - min(recall_list))
    hard3_recall_mean = float(np.mean([per_class_recall["angry"], per_class_recall["fear"], per_class_recall["sad"]]))

    res = {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "recall_spread": round(recall_spread, 4),
        "hard3_recall_mean": round(hard3_recall_mean, 4),
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
    }

    if angles is not None:
        angle_acc = {}
        for ang in ANGLE_NAMES:
            mask = np.array([a == ang for a in angles], dtype=bool)
            if mask.sum() > 0:
                angle_acc[ang] = round(float((y_true[mask] == y_pred[mask]).mean()), 4)
        res["per_angle_accuracy"] = angle_acc
        res["angle_spread"] = round(float(max(angle_acc.values()) - min(angle_acc.values())), 4)
        res["fr_fl_gap"] = round(float(abs(angle_acc["FL"] - angle_acc["FR"])), 4)

    return res


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_angles: list,
    mean_train: float,
    std_train: float,
    device: torch.device,
    tta_mode: str = "none",
    logit_bias: torch.Tensor = None,
    batch_size: int = 64,
):
    """
    Evaluates a model on the validation split under:
      - tta_mode="none": standard single-view evaluation
      - tta_mode="flip2": 2-view TTA (original + horizontal flip)
      - tta_mode="multiscale4": 4-view TTA (orig, hflip, 94% center-crop resized + hflip)
    """
    model.eval()
    n = len(val_y)
    all_logits = []

    for start in range(0, n, batch_size):
        xb_raw = val_x[start : start + batch_size].to(device, non_blocking=True)
        xb = (xb_raw - mean_train) / std_train

        if tta_mode == "none":
            logits = model(xb)
        elif tta_mode == "flip2":
            xb_flip = torch.flip(xb, dims=[3])
            logits = 0.5 * (model(xb) + model(xb_flip))
        elif tta_mode == "multiscale4":
            xb_flip = torch.flip(xb, dims=[3])
            # Slight inner crop (94% = 120x120 resized back to 128x128) to reduce studio border
            margin = 4
            xb_crop = F.interpolate(
                xb[:, :, margin : IMG_SIZE - margin, margin : IMG_SIZE - margin],
                size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            xb_crop_flip = torch.flip(xb_crop, dims=[3])
            logits = (
                0.35 * model(xb)
                + 0.35 * model(xb_flip)
                + 0.15 * model(xb_crop)
                + 0.15 * model(xb_crop_flip)
            )
        else:
            raise ValueError(f"Unknown tta_mode: {tta_mode}")

        if logit_bias is not None:
            logits = logits + logit_bias.unsqueeze(0)

        all_logits.append(logits.cpu())

    all_logits_t = torch.cat(all_logits, dim=0)
    val_loss = float(F.cross_entropy(all_logits_t, val_y, label_smoothing=0.1).item())
    preds = all_logits_t.argmax(dim=1).numpy()
    metrics = compute_metrics(val_y.numpy(), preds, angles=val_angles)
    metrics["val_loss"] = round(val_loss, 4)
    return metrics, all_logits_t


# ---------------------------------------------------------------------------
# 8. Controlled Ablation Runner
# ---------------------------------------------------------------------------
def run_ablation_experiment(
    exp_id: str,
    description: str,
    train_data,
    val_data,
    mean_train: float,
    std_train: float,
    device: torch.device,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    sched_type: str = "warmup_cosine",
    warmup_epochs: int = 5,
    aug_mode: str = "khong_erasing",
    loss_type: str = "ce_smooth",
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    mixup_off_last_k: int = 15,
    use_ema: bool = False,
    ema_decay: float = 0.992,
    use_swa: bool = False,
    swa_start_epoch: int = 65,
    simulate_cold_restart_ft: bool = False,
):
    seed_everything(SEED)
    train_x, train_y, _ = train_data
    val_x, val_y, val_angles = val_data

    train_tf, eval_tf = build_transforms(aug_mode, mean_train, std_train)
    ds_train = InMemoryKDEFDataset(train_x, train_y, transform=train_tf)
    ds_train_clean = InMemoryKDEFDataset(train_x, train_y, transform=eval_tf)

    g = torch.Generator()
    g.manual_seed(SEED)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, generator=g, drop_last=False)
    loader_train_clean = DataLoader(ds_train_clean, batch_size=64, shuffle=False)

    model = KDEFNet(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss setup
    if loss_type == "ce_smooth":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    elif loss_type == "poly1_focal_cb":
        # Mild class difficulty weights: boost hard negative classes (angry, fear, sad)
        # ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        cw = torch.tensor([1.15, 0.95, 1.18, 0.85, 0.95, 1.12, 0.98], dtype=torch.float32, device=device)
        criterion = Poly1FocalCrossEntropyLoss(
            num_classes=NUM_CLASSES,
            label_smoothing=0.08,
            gamma=1.2,
            poly_eps=0.5,
            class_weights=cw,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Scheduler setup
    if sched_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )
    elif sched_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(loader_train),
            pct_start=0.10,
            div_factor=15.0,
            final_div_factor=1000.0,
        )
    elif sched_type == "warmup_cosine":
        def lr_lambda(ep_idx: int):
            if ep_idx < warmup_epochs:
                return 0.05 + 0.95 * ((ep_idx + 1) / float(warmup_epochs))
            progress = (ep_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = 1e-6 / lr
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unknown sched_type: {sched_type}")

    ema_helper = ModelEMA(model, decay=ema_decay) if use_ema else None
    swa_state = None
    swa_count = 0

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    best_train_f1_at_best = 0.0
    ep15_val_f1 = 0.0
    ep15_train_f1 = 0.0
    val_f1_history = []
    train_f1_history = []

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        all_tr_preds = []
        all_tr_targets = []

        allow_mixup = (mixup_alpha > 0.0) and (ep <= epochs - mixup_off_last_k)

        for xb, yb in loader_train:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if allow_mixup and random.random() < mixup_prob:
                xb_in, yb_soft = mixup_batch(xb, yb, alpha=mixup_alpha, num_classes=NUM_CLASSES, label_smoothing=0.08)
                logits = model(xb_in)
                if isinstance(criterion, Poly1FocalCrossEntropyLoss):
                    loss = criterion(logits, yb_soft)
                else:
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss = -(yb_soft * log_probs).sum(dim=-1).mean()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            if sched_type == "onecycle":
                scheduler.step()
            if ema_helper is not None:
                ema_helper.update(model)

            all_tr_preds.append(logits.detach().argmax(dim=1).cpu().numpy())
            all_tr_targets.append(yb.cpu().numpy())

        tr_preds = np.concatenate(all_tr_preds)
        tr_targets = np.concatenate(all_tr_targets)
        tr_metrics = compute_metrics(tr_targets, tr_preds)
        tr_f1 = tr_metrics["macro_f1"]

        val_metrics, _ = evaluate_model(model, val_x, val_y, val_angles, mean_train, std_train, device, tta_mode="none")
        val_f1 = val_metrics["macro_f1"]

        if sched_type == "plateau":
            scheduler.step(val_f1)
        elif sched_type == "warmup_cosine":
            scheduler.step()

        # Accumulate SWA weights in final epochs
        if use_swa and ep >= swa_start_epoch:
            msd = model.state_dict()
            if swa_state is None:
                swa_state = {k: v.detach().clone() for k, v in msd.items()}
                swa_count = 1
            else:
                swa_count += 1
                alpha_swa = 1.0 / float(swa_count)
                for k, v in msd.items():
                    if swa_state[k].dtype.is_floating_point:
                        swa_state[k].mul_(1.0 - alpha_swa).add_(v.detach(), alpha=alpha_swa)
                    else:
                        swa_state[k].copy_(v.detach())

        val_f1_history.append(val_f1)
        train_f1_history.append(tr_f1)

        if ep == 15:
            ep15_val_f1 = val_f1
            ep15_train_f1 = tr_f1

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = ep
            best_train_f1_at_best = tr_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0

    # Evaluate best raw model on clean train and val
    model.load_state_dict(best_state)
    clean_train_metrics, _ = evaluate_model(model, train_x, train_y, None, mean_train, std_train, device, tta_mode="none")
    best_val_single, _ = evaluate_model(model, val_x, val_y, val_angles, mean_train, std_train, device, tta_mode="none")
    best_val_flip2, _ = evaluate_model(model, val_x, val_y, val_angles, mean_train, std_train, device, tta_mode="flip2")
    best_val_ms4, _ = evaluate_model(model, val_x, val_y, val_angles, mean_train, std_train, device, tta_mode="multiscale4")

    # Late-epoch oscillation metric (std of Val F1 across last 15 epochs)
    last15_std = float(np.std(val_f1_history[-15:]))
    last15_mean = float(np.mean(val_f1_history[-15:]))
    final_epoch_val_f1 = float(val_f1_history[-1])

    result = {
        "exp_id": exp_id,
        "description": description,
        "hyperparams": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "sched_type": sched_type,
            "aug_mode": aug_mode,
            "loss_type": loss_type,
            "mixup_alpha": mixup_alpha,
            "mixup_prob": mixup_prob,
            "use_ema": use_ema,
            "use_swa": use_swa,
        },
        "elapsed_sec": round(elapsed, 2),
        "ep15_val_macro_f1": round(ep15_val_f1, 4),
        "ep15_train_macro_f1": round(ep15_train_f1, 4),
        "best_epoch": best_epoch,
        "final_epoch_val_f1": round(final_epoch_val_f1, 4),
        "last15_val_f1_mean": round(last15_mean, 4),
        "last15_val_f1_std": round(last15_std, 4),
        "train_macro_f1_running_at_best": round(best_train_f1_at_best, 4),
        "train_macro_f1_clean_eval": clean_train_metrics["macro_f1"],
        "val_single_view": best_val_single,
        "val_tta_flip2": best_val_flip2,
        "val_tta_multiscale4": best_val_ms4,
        "train_val_gap_running": round(best_train_f1_at_best - best_val_single["macro_f1"], 4),
        "train_val_gap_clean": round(clean_train_metrics["macro_f1"] - best_val_single["macro_f1"], 4),
    }

    # Optional simulation of Notebook Cell 8.4 cold-restart fine-tuning degradation
    if simulate_cold_restart_ft:
        ft_model = KDEFNet(num_classes=NUM_CLASSES).to(device)
        ft_model.load_state_dict(best_state)
        ft_opt = torch.optim.AdamW(ft_model.parameters(), lr=7.5e-5, weight_decay=weight_decay)
        ft_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=15, eta_min=1e-6)
        ft_val_f1s = []
        for _ in range(15):
            ft_model.train()
            for xb, yb in loader_train:
                xb, yb = xb.to(device), yb.to(device)
                ft_opt.zero_grad(set_to_none=True)
                loss = criterion(ft_model(xb), yb)
                loss.backward()
                ft_opt.step()
            ft_sched.step()
            ft_m, _ = evaluate_model(ft_model, val_x, val_y, val_angles, mean_train, std_train, device, tta_mode="none")
            ft_val_f1s.append(ft_m["macro_f1"])
        result["cold_restart_ft_ep1_val_f1"] = round(float(ft_val_f1s[0]), 4)
        result["cold_restart_ft_best_val_f1"] = round(float(max(ft_val_f1s)), 4)
        result["cold_restart_ft_final_val_f1"] = round(float(ft_val_f1s[-1]), 4)

    # Evaluate EMA & SWA models if enabled
    if ema_helper is not None:
        update_bn_stats(ema_helper.ema, loader_train_clean, device)
        ema_single, _ = evaluate_model(ema_helper.ema, val_x, val_y, val_angles, mean_train, std_train, device, tta_mode="none")
        ema_flip2, _ = evaluate_model(ema_helper.ema, val_x, val_y, val_angles, mean_train, std_train, device, tta_mode="flip2")
        ema_ms4, _ = evaluate_model(ema_helper.ema, val_x, val_y, val_angles, mean_train, std_train, device, tta_mode="multiscale4")
        result["ema_val_single_view"] = ema_single
        result["ema_val_tta_flip2"] = ema_flip2
        result["ema_val_tta_multiscale4"] = ema_ms4

    if use_swa and swa_state is not None:
        swa_model = KDEFNet(num_classes=NUM_CLASSES).to(device)
        swa_model.load_state_dict(swa_state)
        update_bn_stats(swa_model, loader_train_clean, device)
        swa_single, _ = evaluate_model(swa_model, val_x, val_y, val_angles, mean_train, std_train, device, tta_mode="none")
        swa_flip2, _ = evaluate_model(swa_model, val_x, val_y, val_angles, mean_train, std_train, device, tta_mode="flip2")
        swa_ms4, _ = evaluate_model(swa_model, val_x, val_y, val_angles, mean_train, std_train, device, tta_mode="multiscale4")
        result["swa_val_single_view"] = swa_single
        result["swa_val_tta_flip2"] = swa_flip2
        result["swa_val_tta_multiscale4"] = swa_ms4

    print(
        f"[{exp_id}] Ep15 F1={result['ep15_val_macro_f1']:.4f} | "
        f"Best Val F1={best_val_single['macro_f1']:.4f} (Ep {best_epoch}) | "
        f"Flip-TTA F1={best_val_flip2['macro_f1']:.4f} | "
        f"MS4-TTA F1={best_val_ms4['macro_f1']:.4f} | "
        f"Spread={best_val_single['recall_spread']:.4f}->{best_val_flip2['recall_spread']:.4f} | "
        f"Gap(run)={result['train_val_gap_running']:.4f} | {elapsed:.1f}s"
    )
    if "swa_val_single_view" in result:
        print(
            f"   └─ SWA Single={result['swa_val_single_view']['macro_f1']:.4f} | "
            f"SWA Flip-TTA={result['swa_val_tta_flip2']['macro_f1']:.4f} | "
            f"SWA MS4-TTA={result['swa_val_tta_multiscale4']['macro_f1']:.4f} | "
            f"SWA Spread={result['swa_val_tta_flip2']['recall_spread']:.4f}"
        )
    if "ema_val_single_view" in result:
        print(
            f"   └─ EMA Single={result['ema_val_single_view']['macro_f1']:.4f} | "
            f"EMA Flip-TTA={result['ema_val_tta_flip2']['macro_f1']:.4f} | "
            f"EMA MS4-TTA={result['ema_val_tta_multiscale4']['macro_f1']:.4f}"
        )

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Running on device: {device} ({gpu_name})")

    repo_root, train_dir, val_dir = resolve_repo_and_data()
    t_load = time.time()
    train_data, val_data, mean_train, std_train, split_info = load_clean_train_val(train_dir, val_dir)
    print(
        f"Loaded clean splits in {time.time() - t_load:.2f}s: "
        f"train={split_info['clean_train_count']} (excluded 9), "
        f"val={split_info['clean_val_count']} (excluded 1), "
        f"mean={mean_train:.4f}, std={std_train:.4f}"
    )

    experiments = []

    # Exp 0: Baseline Reproduction (Notebook Best Config: AdamW lr=3e-4, ReduceLROnPlateau, CE smooth=0.1, khong_erasing)
    experiments.append(run_ablation_experiment(
        exp_id="exp0_baseline_plateau",
        description="Notebook Best Config: AdamW(lr=3e-4, wd=1e-4) + ReduceLROnPlateau(p=5, f=0.5) + CE(ls=0.1) + khong_erasing",
        train_data=train_data,
        val_data=val_data,
        mean_train=mean_train,
        std_train=std_train,
        device=device,
        epochs=80,
        batch_size=32,
        lr=3e-4,
        weight_decay=1e-4,
        sched_type="plateau",
        aug_mode="khong_erasing",
        loss_type="ce_smooth",
        simulate_cold_restart_ft=True,
    ))

    # Exp 1: LR Schedule — OneCycleLR (max_lr=1e-3, pct_start=0.1)
    experiments.append(run_ablation_experiment(
        exp_id="exp1_sched_onecycle",
        description="LR Schedule Ablation: AdamW + OneCycleLR(max_lr=1e-3, pct_start=0.1) + CE(ls=0.1) + khong_erasing",
        train_data=train_data,
        val_data=val_data,
        mean_train=mean_train,
        std_train=std_train,
        device=device,
        epochs=80,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        sched_type="onecycle",
        aug_mode="khong_erasing",
        loss_type="ce_smooth",
    ))

    # Exp 2: LR Schedule — Linear Warmup (5 ep) + CosineAnnealingLR (1e-3 -> 1e-6)
    experiments.append(run_ablation_experiment(
        exp_id="exp2_sched_warmup_cosine",
        description="LR Schedule Ablation: AdamW(lr=1e-3, wd=1e-4) + 5-ep Linear Warmup + CosineAnnealingLR(1e-6) + CE(ls=0.1) + khong_erasing",
        train_data=train_data,
        val_data=val_data,
        mean_train=mean_train,
        std_train=std_train,
        device=device,
        epochs=80,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        sched_type="warmup_cosine",
        warmup_epochs=5,
        aug_mode="khong_erasing",
        loss_type="ce_smooth",
    ))

    # Exp 3: Enhanced Geometry + Photometric Augmentation (RandomPerspective for FL/FR + ColorJitter + Affine)
    experiments.append(run_ablation_experiment(
        exp_id="exp3_aug_geom_photo",
        description="Augmentation Ablation: Warmup+Cosine + RandomPerspective(0.14, p=0.3) + Affine(12 deg, trans=4%) + ColorJitter(0.15)",
        train_data=train_data,
        val_data=val_data,
        mean_train=mean_train,
        std_train=std_train,
        device=device,
        epochs=80,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        sched_type="warmup_cosine",
        warmup_epochs=5,
        aug_mode="enhanced_geom_photo",
        loss_type="ce_smooth",
    ))

    # Exp 4: Loss Function for Hard Negative Classes — Poly-1 Focal Cross-Entropy + Hard-Class Weighting
    experiments.append(run_ablation_experiment(
        exp_id="exp4_loss_poly1_focal_cb",
        description="Loss Ablation: Warmup+Cosine + Enhanced Aug + Poly-1 Focal CE (gamma=1.2, eps=0.5, ls=0.08, hard-class boost)",
        train_data=train_data,
        val_data=val_data,
        mean_train=mean_train,
        std_train=std_train,
        device=device,
        epochs=80,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        sched_type="warmup_cosine",
        warmup_epochs=5,
        aug_mode="enhanced_geom_photo",
        loss_type="poly1_focal_cb",
    ))

    # Exp 5: Generalization Gap Regularization + Weight Averaging (Mixup alpha=0.2, wd=1e-3, EMA + SWA + TTA)
    experiments.append(run_ablation_experiment(
        exp_id="exp5_full_mixup_ema_swa_tta",
        description="Full Recipe: Warmup+Cosine + Enhanced Aug + Poly-1 Focal CE + Mixup(alpha=0.2, p=0.5, off last 15ep) + WD=1e-3 + EMA + SWA + TTA",
        train_data=train_data,
        val_data=val_data,
        mean_train=mean_train,
        std_train=std_train,
        device=device,
        epochs=80,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-3,
        sched_type="warmup_cosine",
        warmup_epochs=5,
        aug_mode="enhanced_geom_photo",
        loss_type="poly1_focal_cb",
        mixup_alpha=0.2,
        mixup_prob=0.5,
        mixup_off_last_k=15,
        use_ema=True,
        ema_decay=0.992,
        use_swa=True,
        swa_start_epoch=65,
    ))

    # Save comprehensive JSON report
    out_path = repo_root / "reports" / "exp_train_results.json"
    report_payload = {
        "author": "train-opt",
        "device": f"{device} ({gpu_name})",
        "anti_leakage_verified": True,
        "dataset_split_info": split_info,
        "experiments": experiments,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved all experiment results to: {out_path}")


if __name__ == "__main__":
    main()
