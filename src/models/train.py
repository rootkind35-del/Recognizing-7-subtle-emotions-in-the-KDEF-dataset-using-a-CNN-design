"""
Multicultural Training & Validation Optimization Pipeline (`src/models/train.py`)
for the canonical CNN architecture `KDEF_Stem2Ch_MS4_Symmetric` (`stem2ch_ms4_fuse`, 1,273,005 params).

Designed for Milestone M2 on branch `tree2`:
- Adapts data loading for the 3-source multicultural dataset (`data/manifests/multicultural_manifest.csv`:
  32,404 clean `train` images and 3,990 clean `val` images across KDEF, RADIATE, and FER-2013).
- STRICT ANTI-LEAKAGE: Never touches, loads, or evaluates the `test` split (4,875 images). All normalization
  constants (`mean=0.517243`, `std=0.264053`), class weights, hyperparameter searches, model selections,
  early stopping, and logit-prior calibrations use strictly `train` and `val`.
- Keeps the exact `KDEF_Stem2Ch_MS4_Symmetric` architecture from `src/models/model.py`.
- Handles severe 7-class imbalance (7.75x ratio between `happy` and `disgust`) via inverse-frequency
  class-weighted `SoftLabelSmoothedFocalCE`, `SupConLoss`, `FlipConsistencyLoss`, `ModelEMA`,
  `SnapshotSoup`, and validation-tuned prior bias calibration absorbed into `head.classifier.bias`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import (  # noqa: E402
    ANGLE_NAMES,
    CLASS_NAMES,
    CLASS_TO_IDX,
    DEFAULT_MULTICULTURAL_MANIFEST,
    DEFAULT_MULTICULTURAL_MEAN,
    DEFAULT_MULTICULTURAL_STD,
    compute_class_weights,
)
from src.models.model import (  # noqa: E402
    KDEF_Stem2Ch_MS4_Symmetric,
    count_parameters,
    load_kdef_checkpoint,
)
from src.preprocessing.alignment import (  # noqa: E402
    DEFAULT_CROP_MODE,
    DEFAULT_SSD_MARGIN,
    KDEF_STUDIO_CROP_BOX,
    _preprocess_to_uint8,
)

CACHE_DIR = REPO_ROOT / ".zcode" / "cache"
CKPT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Reproducibility & Anti-Leakage Guards
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def assert_no_test_leakage(split_name: str = "", path_str: str = "") -> None:
    s = str(split_name).lower().strip()
    p = str(path_str).lower().replace("\\", "/")
    if s == "test" or "/test/" in p or p.endswith("/test"):
        raise RuntimeError(
            f"ANTI-LEAKAGE VIOLATION: Attempted access to locked test split ({split_name}, {path_str})"
        )


# ---------------------------------------------------------------------------
# Fast Cached Multicultural Train/Val Loader
# ---------------------------------------------------------------------------

def load_multicultural_split_cached(
    split: str,
    img_size: int = 160,
    manifest_path: Optional[Union[str, Path]] = None,
    use_crop: bool = True,
    crop_mode: str = DEFAULT_CROP_MODE,
    ssd_margin: float = DEFAULT_SSD_MARGIN,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_blend: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads (or builds and caches) uint8 grayscale images `(N, img_size, img_size)`,
    integer labels `(N,)`, and metadata DataFrame strictly for `train` or `val`.
    """
    split_norm = split.strip().lower()
    assert_no_test_leakage(split_name=split_norm)
    if split_norm not in ("train", "val"):
        raise ValueError(f"Only 'train' and 'val' splits are allowed in train.py, got '{split_norm}'")

    m_path = Path(manifest_path) if manifest_path is not None else DEFAULT_MULTICULTURAL_MANIFEST
    if not m_path.is_absolute():
        m_path = REPO_ROOT / m_path
    if not m_path.exists():
        raise FileNotFoundError(f"Multicultural manifest not found: {m_path}")

    df_all = pd.read_csv(m_path)
    df = df_all[df_all["split"].astype(str).str.strip().str.lower() == split_norm].copy()
    df.reset_index(drop=True, inplace=True)
    if len(df) == 0:
        raise ValueError(f"Empty split '{split_norm}' in {m_path}")

    clahe_tag = f"clahe{int(use_clahe)}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw_img_cache = CACHE_DIR / f"mc_{split_norm}_{crop_mode}_{img_size}_{clahe_tag}_imgs.npy"
    raw_lbl_cache = CACHE_DIR / f"mc_{split_norm}_{crop_mode}_{img_size}_{clahe_tag}_lbls.npy"
    npz_cache = CACHE_DIR / f"mc_{split_norm}_{crop_mode}_{img_size}_{clahe_tag}.npz"

    if raw_img_cache.exists() and raw_lbl_cache.exists():
        imgs = np.load(raw_img_cache)
        labels = np.load(raw_lbl_cache)
        if len(imgs) == len(df):
            return imgs, labels, df

    if npz_cache.exists():
        data = np.load(npz_cache)
        imgs = data["images"]
        labels = data["labels"]
        if len(imgs) == len(df):
            try:
                np.save(raw_img_cache, imgs)
                np.save(raw_lbl_cache, labels)
            except Exception:
                pass
            return imgs, labels, df

    n = len(df)
    imgs = np.empty((n, img_size, img_size), dtype=np.uint8)
    labels = df["label_idx"].astype(np.int64).to_numpy()

    for i, row in enumerate(df.itertuples(index=False)):
        rel_fp = Path(row.filepath)
        fpath = rel_fp if rel_fp.is_absolute() else (REPO_ROOT / rel_fp)
        imgs[i] = _preprocess_to_uint8(
            fpath,
            source=str(row.source),
            use_aspect_crop=use_crop,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            clahe_blend=clahe_blend,
            img_size=img_size,
            crop_mode=crop_mode,
            ssd_margin=ssd_margin,
            crop_box=KDEF_STUDIO_CROP_BOX,
        )

    try:
        np.save(raw_img_cache, imgs)
        np.save(raw_lbl_cache, labels)
        if not npz_cache.exists():
            np.savez_compressed(npz_cache, images=imgs, labels=labels)
    except Exception:
        pass
    return imgs, labels, df


# ---------------------------------------------------------------------------
# Optimizers, EMA, Snapshot Soup & BatchNorm Protection
# ---------------------------------------------------------------------------

def disable_bn_running_stats(model: nn.Module) -> List[Tuple[nn.Module, Optional[float]]]:
    saved: List[Tuple[nn.Module, Optional[float]]] = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            saved.append((m, m.momentum))
            m.momentum = 0.0
    return saved


def restore_bn_running_stats(saved: List[Tuple[nn.Module, Optional[float]]]) -> None:
    for m, mom in saved:
        m.momentum = mom


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM / ASAM) wrapper over AdamW."""

    def __init__(
        self,
        params: Iterable,
        base_optimizer_cls: type = torch.optim.AdamW,
        rho: float = 0.05,
        adaptive: bool = True,
        **kwargs: Any,
    ):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            rho = group.get("rho", 0.05)
            adaptive = group.get("adaptive", True)
            scale = rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            ((torch.abs(p) if group.get("adaptive", True) else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2) if norms else torch.tensor(0.0, device=shared_device)


class ModelEMA:
    """Exponential Moving Average of model weights and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.996):
        self.decay = float(decay)
        self.updates = 0
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.device), alpha=1.0 - d)
            else:
                v.copy_(msd[k].detach().to(v.device))


class SnapshotSoup:
    """Collects top validation snapshots and builds Uniform or Greedy Model Soup."""

    def __init__(self, max_snapshots: int = 5, min_epoch_gap: int = 2):
        self.max_snapshots = max_snapshots
        self.min_epoch_gap = min_epoch_gap
        self.snapshots: List[Tuple[float, int, Dict[str, torch.Tensor]]] = []

    @staticmethod
    def _clone_cpu_sd(model: nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def maybe_add(self, val_f1: float, epoch: int, model: nn.Module) -> None:
        sd = self._clone_cpu_sd(model)
        for idx, (s_f1, s_ep, _) in enumerate(self.snapshots):
            if abs(epoch - s_ep) < self.min_epoch_gap:
                if val_f1 > s_f1:
                    self.snapshots[idx] = (val_f1, epoch, sd)
                    self.snapshots.sort(key=lambda x: x[0], reverse=True)
                return
        self.snapshots.append((val_f1, epoch, sd))
        self.snapshots.sort(key=lambda x: x[0], reverse=True)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[: self.max_snapshots]

    @staticmethod
    def average_state_dicts(state_dicts: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(state_dicts) == 1:
            return {k: v.clone() for k, v in state_dicts[0].items()}
        avg_sd: Dict[str, torch.Tensor] = {}
        n = float(len(state_dicts))
        for k, v0 in state_dicts[0].items():
            if v0.dtype.is_floating_point:
                acc = torch.zeros_like(v0, dtype=torch.float32)
                for sd in state_dicts:
                    acc.add_(sd[k].to(torch.float32))
                avg_sd[k] = (acc / n).to(v0.dtype)
            else:
                avg_sd[k] = v0.clone()
        return avg_sd

    def build_greedy_soup(
        self, eval_fn: Callable[[Dict[str, torch.Tensor]], float]
    ) -> Tuple[Dict[str, torch.Tensor], float, List[int]]:
        sorted_snaps = sorted(self.snapshots, key=lambda x: x[0], reverse=True)
        selected_sds = [sorted_snaps[0][2]]
        selected_epochs = [sorted_snaps[0][1]]
        best_score = eval_fn(selected_sds[0])

        for _, ep, sd in sorted_snaps[1:]:
            cand_sds = selected_sds + [sd]
            cand_avg = self.average_state_dicts(cand_sds)
            cand_score = eval_fn(cand_avg)
            if cand_score >= best_score:
                best_score = cand_score
                selected_sds = cand_sds
                selected_epochs.append(ep)

        return self.average_state_dicts(selected_sds), best_score, selected_epochs


# ---------------------------------------------------------------------------
# Vectorized GPU Batch Augmentation & Facial Mixing
# ---------------------------------------------------------------------------

@torch.no_grad()
def gpu_augment_batch(
    x_u8: torch.Tensor,
    mean_v: float,
    std_v: float,
    aug_mode: str = "enhanced",
    erase_prob: float = 0.0,
) -> torch.Tensor:
    """
    Vectorized GPU-resident geometric + photometric + optional random erasing augmentation.
    x_u8: (B, 1, H, W) uint8 tensor on device.
    Returns normalized float32 tensor (B, 1, H, W).
    """
    x = x_u8.float().div_(255.0)
    b, c, h, w = x.shape
    dev = x.device

    if aug_mode in ("enhanced", "baseline"):
        # 1. Random horizontal flip (p=0.5)
        flip_mask = torch.rand(b, device=dev) < 0.5
        if flip_mask.any():
            x[flip_mask] = torch.flip(x[flip_mask], dims=[3])

        # 2. Random affine: rotation +-10 deg, scale, translation
        angles = (torch.rand(b, device=dev) * 2.0 - 1.0) * (10.0 * math.pi / 180.0)
        if aug_mode == "enhanced":
            scales = torch.empty(b, device=dev).uniform_(0.93, 1.07)
            tx = torch.empty(b, device=dev).uniform_(-0.08, 0.08)
            ty = torch.empty(b, device=dev).uniform_(-0.08, 0.08)
        else:
            scales = torch.empty(b, device=dev).uniform_(1.0 / 1.1, 1.1)
            tx = torch.zeros(b, device=dev)
            ty = torch.zeros(b, device=dev)

        cos_a = torch.cos(angles) / scales
        sin_a = torch.sin(angles) / scales
        theta = torch.zeros((b, 2, 3), device=dev, dtype=x.dtype)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

        if aug_mode == "enhanced":
            # 3. Photometric gain + bias jitter (p=0.8)
            p_mask = torch.rand(b, 1, 1, 1, device=dev) < 0.8
            gain = torch.where(
                p_mask,
                torch.empty(b, 1, 1, 1, device=dev).uniform_(0.88, 1.12),
                torch.ones(b, 1, 1, 1, device=dev),
            )
            bias = torch.where(
                p_mask,
                torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.05, 0.05),
                torch.zeros(b, 1, 1, 1, device=dev),
            )
            x = torch.clamp(x * gain + bias, 0.0, 1.0)

    x = (x - mean_v) / std_v

    if erase_prob > 0.0:
        for i in range(b):
            if float(torch.rand(1, device=dev).item()) < erase_prob:
                eh = random.randint(int(0.12 * h), int(0.32 * h))
                ew = random.randint(int(0.12 * w), int(0.35 * w))
                top = random.randint(int(0.10 * h), max(int(0.10 * h), h - eh))
                left = random.randint(0, max(0, w - ew))
                x[i, :, top : top + eh, left : left + ew] = 0.0

    return x


def apply_batch_mix(
    xb: torch.Tensor,
    yb: torch.Tensor,
    num_classes: int = 7,
    mix_prob: float = 0.40,
    mixup_alpha: float = 0.25,
    cutmix_alpha: float = 0.80,
    splitmix_prob: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    B, C, H, W = xb.shape
    y_one = F.one_hot(yb, num_classes=num_classes).float()
    if mix_prob <= 0.0 or float(torch.rand(1).item()) >= mix_prob or B <= 1:
        return xb, y_one, False

    perm = torch.randperm(B, device=xb.device)
    y_perm = y_one[perm]
    roll = float(torch.rand(1).item())

    if roll < splitmix_prob:
        split_y = random.randint(int(0.42 * H), int(0.58 * H))
        xb_mix = xb.clone()
        xb_mix[:, :, split_y:, :] = xb[perm, :, split_y:, :]
        lam = float(split_y) / float(H)
        return xb_mix, lam * y_one + (1.0 - lam) * y_perm, True

    if roll < splitmix_prob + 0.5 * (1.0 - splitmix_prob):
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        cut_rat = math.sqrt(max(1.0 - lam, 0.0))
        cw, ch = int(round(W * cut_rat)), int(round(H * cut_rat))
        cx, cy = random.randint(0, W - 1), random.randint(0, H - 1)
        x1, y1 = max(cx - cw // 2, 0), max(cy - ch // 2, 0)
        x2, y2 = min(cx + cw // 2, W), min(cy + ch // 2, H)
        xb_mix = xb.clone()
        xb_mix[:, :, y1:y2, x1:x2] = xb[perm, :, y1:y2, x1:x2]
        lam_adj = 1.0 - float((x2 - x1) * (y2 - y1)) / float(W * H)
        return xb_mix, lam_adj * y_one + (1.0 - lam_adj) * y_perm, True

    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
    return lam * xb + (1.0 - lam) * xb[perm], lam * y_one + (1.0 - lam) * y_perm, True


# ---------------------------------------------------------------------------
# Losses: Class-Weighted SoftLabelSmoothedFocalCE, SupConLoss, FlipConsistency
# ---------------------------------------------------------------------------

class SoftLabelSmoothedFocalCE(nn.Module):
    """
    Class-weighted Label-Smoothed Focal + Poly-1 Cross-Entropy supporting both hard integer labels
    and soft mixed targets (CutMix / Mixup / SplitMix).
    """

    def __init__(
        self,
        num_classes: int = 7,
        label_smoothing: float = 0.08,
        focal_gamma: float = 0.5,
        poly_eps: float = 0.2,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ls = float(label_smoothing)
        self.gamma = float(focal_gamma)
        self.poly_eps = float(poly_eps)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.detach().clone().float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        if targets.dim() == 1:
            y_true = F.one_hot(targets.long(), num_classes=self.num_classes).float()
        else:
            y_true = targets.float()

        if self.ls > 0.0:
            y_smooth = (1.0 - self.ls) * y_true + (self.ls / float(self.num_classes))
        else:
            y_smooth = y_true

        log_probs = F.log_softmax(logits_f, dim=-1)
        probs = log_probs.exp()
        pt = (probs * y_true).sum(dim=-1).clamp(min=1e-6, max=1.0)

        ce_per_cls = -(y_smooth * log_probs)
        if self.class_weights is not None:
            ce_per_cls = ce_per_cls * self.class_weights.unsqueeze(0)
        ce = ce_per_cls.sum(dim=-1)

        if self.gamma > 0.0:
            ce = ((1.0 - pt) ** self.gamma) * ce
        if self.poly_eps > 0.0:
            sample_w = (y_true * self.class_weights.unsqueeze(0)).sum(dim=-1) if self.class_weights is not None else 1.0
            ce = ce + self.poly_eps * (1.0 - pt) * sample_w
        return ce.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020) over 1-view or 2-view embeddings."""

    def __init__(self, temperature: float = 0.08):
        super().__init__()
        self.temp = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = features if features.dim() == 3 else features.unsqueeze(1)
        B, V, D = feats.shape
        if B <= 1:
            return feats.sum() * 0.0
        feats = F.normalize(feats.float(), dim=-1)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(feats.device)
        contrast_feat = torch.cat(torch.unbind(feats, dim=1), dim=0)
        sim = torch.matmul(contrast_feat, contrast_feat.T) / self.temp
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - sim_max.detach()
        mask = mask.repeat(V, V)
        logits_mask = 1.0 - torch.eye(B * V, device=feats.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        pos_per_row = mask.sum(1)
        valid = pos_per_row > 0
        if not valid.any():
            return feats.sum() * 0.0
        return -(mask[valid] * log_prob[valid]).sum(1).div(pos_per_row[valid]).mean()


class FlipConsistencyLoss(nn.Module):
    """Symmetric KL + feature cosine alignment between x and flip(x)."""

    def __init__(self, feat_weight: float = 0.15):
        super().__init__()
        self.feat_weight = float(feat_weight)

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        feat1: Optional[torch.Tensor] = None,
        feat2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        l1, l2 = logits1.float(), logits2.float()
        p1, p2 = F.softmax(l1, dim=-1), F.softmax(l2, dim=-1)
        lp1, lp2 = F.log_softmax(l1, dim=-1), F.log_softmax(l2, dim=-1)
        loss = 0.5 * (
            F.kl_div(lp2, p1.detach(), reduction="batchmean")
            + F.kl_div(lp1, p2.detach(), reduction="batchmean")
        )
        if self.feat_weight > 0.0 and feat1 is not None and feat2 is not None:
            f1 = F.normalize(feat1.float(), dim=-1)
            f2 = F.normalize(feat2.float(), dim=-1)
            loss = loss + self.feat_weight * (1.0 - (f1 * f2).sum(dim=-1)).mean()
        return loss


# ---------------------------------------------------------------------------
# Comprehensive Validation Evaluator & Logit-Prior Calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_cnn_on_images(
    model: nn.Module,
    images_u8: np.ndarray,
    mean_v: float,
    std_v: float,
    device: torch.device,
    batch_size: int = 256,
    use_tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference on `images_u8` `(N, H, W)` uint8 array.
    Returns `(logits_np, probs_np, preds_np)`.
    """
    model.eval()
    n = len(images_u8)
    logits_list: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        batch_u8 = images_u8[start : start + batch_size]
        xb = torch.from_numpy(batch_u8).to(device=device, dtype=torch.float32).unsqueeze(1).div_(255.0)
        xb = (xb - mean_v) / std_v
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            if use_tta:
                out_f = model(torch.flip(xb, dims=[3]))
                logits_f = out_f[0] if isinstance(out_f, tuple) else out_f
                logits = 0.5 * (logits + logits_f)
        logits_list.append(logits.float().cpu().numpy())

    logits_all = np.concatenate(logits_list, axis=0)
    shifted = logits_all - np.max(logits_all, axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    probs_all = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    preds_all = np.argmax(probs_all, axis=1)
    return logits_all, probs_all, preds_all


def compute_full_val_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    meta_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Computes full validation metrics including overall, per-class, per-source, ethnic, gender, angle."""
    from src.models.baselines import compute_multicultural_metrics
    return compute_multicultural_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, meta_df=meta_df)


def optimize_val_logit_bias(
    model: KDEF_Stem2Ch_MS4_Symmetric,
    val_logits: np.ndarray,
    val_y: np.ndarray,
    max_rounds: int = 4,
) -> Tuple[np.ndarray, float, float]:
    """
    Finds an optimal additive logit bias vector `delta_b` (7,) strictly on the `val` split
    to maximize `val` Single-View Macro-F1 (while never decreasing Accuracy by more than 0.2%),
    and absorbs `delta_b` directly into `model.head.classifier.bias` so standard `model(x)`
    Single-View inference uses the calibrated threshold automatically.
    """
    base_preds = np.argmax(val_logits, axis=1)
    base_f1 = float(f1_score(val_y, base_preds, average="macro", zero_division=0))
    base_acc = float(accuracy_score(val_y, base_preds))

    best_bias = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    best_f1 = base_f1
    best_acc = base_acc

    grid = np.linspace(-0.85, 0.85, 69, dtype=np.float32)
    for _ in range(max_rounds):
        improved = False
        for c in range(len(CLASS_NAMES)):
            cur_c = best_bias[c]
            for cand in grid:
                trial_bias = best_bias.copy()
                trial_bias[c] = cand
                adj_logits = val_logits + trial_bias[None, :]
                preds = np.argmax(adj_logits, axis=1)
                f1 = float(f1_score(val_y, preds, average="macro", zero_division=0))
                acc = float(accuracy_score(val_y, preds))
                if (f1 > best_f1 + 1e-5 and acc >= base_acc - 0.003) or (
                    abs(f1 - best_f1) <= 1e-5 and acc > best_acc
                ):
                    best_f1 = f1
                    best_acc = acc
                    best_bias = trial_bias
                    improved = True
                else:
                    best_bias[c] = cur_c
        if not improved:
            break

    best_bias = best_bias - float(np.mean(best_bias))
    if best_f1 > base_f1:
        with torch.no_grad():
            bias_param = model.head.classifier.bias
            bias_param.add_(torch.tensor(best_bias, dtype=bias_param.dtype, device=bias_param.device))
    else:
        best_bias = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        best_f1 = base_f1
        best_acc = base_acc

    return best_bias, best_f1, best_acc


# ---------------------------------------------------------------------------
# Training Configuration & Core Training Loop
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    name: str = "multicultural_cnn_opt"
    arch: str = "stem2ch_ms4_fuse"
    img_size: int = 160
    crop_mode: str = DEFAULT_CROP_MODE
    ssd_margin: float = DEFAULT_SSD_MARGIN
    use_crop: bool = True
    use_clahe: bool = True
    mean: float = DEFAULT_MULTICULTURAL_MEAN
    std: float = DEFAULT_MULTICULTURAL_STD

    epochs: int = 45
    warmup_epochs: int = 3
    patience: int = 12
    batch_size: int = 64
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 3e-4
    grad_clip: float = 2.0
    scheduler: str = "warmup_cosine"  # "warmup_cosine" or "plateau"

    # Optimizer & SAM
    use_sam: bool = False
    sam_rho: float = 0.05
    sam_adaptive: bool = True

    # Loss & Class Imbalance
    class_weight_power: float = 0.65  # 0.0=unweighted, 0.5..0.75=tempered balanced, 1.0=full balanced
    label_smoothing: float = 0.06
    focal_gamma: float = 0.75
    poly_eps: float = 0.20

    # Multi-view / Contrastive Regularization
    two_view_flip: bool = True
    flip_kl_weight: float = 0.30
    flip_feat_weight: float = 0.10
    supcon_weight: float = 0.15
    supcon_temp: float = 0.08

    # Augmentation
    aug_mode: str = "enhanced"
    facial_erase_prob: float = 0.0
    mix_prob: float = 0.30
    mixup_alpha: float = 0.20
    cutmix_alpha: float = 0.80
    splitmix_prob: float = 0.25
    mix_start_epoch: int = 4
    mix_end_epoch: int = 38

    # EMA, Soup & Post-hoc Val Calibration
    use_ema: bool = True
    ema_decay: float = 0.996
    use_soup: bool = True
    tune_val_bias: bool = True
    finetune_epochs: int = 8
    seed: int = 42


def train_cnn_run(
    cfg: TrainConfig,
    train_imgs_u8: np.ndarray,
    train_labels: np.ndarray,
    val_imgs_u8: np.ndarray,
    val_labels: np.ndarray,
    val_df: pd.DataFrame,
    base_class_weights: torch.Tensor,
    device: Optional[torch.device] = None,
    save_ckpt_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Trains `KDEF_Stem2Ch_MS4_Symmetric` on `(train_imgs_u8, train_labels)` and optimizes/evaluates
    strictly on `(val_imgs_u8, val_labels)`.
    """
    seed_everything(cfg.seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KDEF_Stem2Ch_MS4_Symmetric(in_channels=1, num_classes=len(CLASS_NAMES)).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    soup = SnapshotSoup(max_snapshots=5, min_epoch_gap=2) if cfg.use_soup else None

    # Tempered or full balanced class weights: w_c^power normalized to mean 1.0
    if cfg.class_weight_power > 0.0:
        cw = torch.pow(base_class_weights.float(), cfg.class_weight_power)
        cw = cw / cw.mean()
        cw = cw.to(device)
    else:
        cw = None

    ce_fn = SoftLabelSmoothedFocalCE(
        num_classes=len(CLASS_NAMES),
        label_smoothing=cfg.label_smoothing,
        focal_gamma=cfg.focal_gamma,
        poly_eps=cfg.poly_eps,
        class_weights=cw,
    ).to(device)
    supcon_fn = SupConLoss(temperature=cfg.supcon_temp).to(device)
    flip_fn = FlipConsistencyLoss(feat_weight=cfg.flip_feat_weight).to(device)

    if cfg.use_sam:
        optimizer: torch.optim.Optimizer = SAM(
            model.parameters(),
            base_optimizer_cls=torch.optim.AdamW,
            rho=cfg.sam_rho,
            adaptive=cfg.sam_adaptive,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        sched_opt = optimizer.base_optimizer
    else:
        use_fused = device.type == "cuda"
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=use_fused,
        )
        sched_opt = optimizer

    if cfg.scheduler == "plateau":
        scheduler: Any = torch.optim.lr_scheduler.ReduceLROnPlateau(
            sched_opt, mode="max", factor=0.5, patience=4, min_lr=cfg.min_lr
        )
    else:
        def lr_lambda(ep_idx: int) -> float:
            if cfg.warmup_epochs > 0 and ep_idx < cfg.warmup_epochs:
                return float(ep_idx + 1) / float(max(1, cfg.warmup_epochs))
            prog = float(ep_idx - cfg.warmup_epochs) / float(max(1, cfg.epochs - cfg.warmup_epochs))
            cos_v = 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))
            ratio = cfg.min_lr / max(cfg.lr, 1e-12)
            return ratio + (1.0 - ratio) * cos_v

        scheduler = torch.optim.lr_scheduler.LambdaLR(sched_opt, lr_lambda)

    # Keep uint8 training images on GPU (~830 MB) for zero-bottleneck batch slicing
    train_x_u8_dev = torch.from_numpy(train_imgs_u8).unsqueeze(1).to(device)
    train_y_dev = torch.from_numpy(train_labels).long().to(device)
    val_y_dev = torch.from_numpy(val_labels).long().to(device)
    n_train = len(train_imgs_u8)
    bs = int(cfg.batch_size)
    amp_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def _forward_loss(
        xb_in: torch.Tensor, yb_hard: torch.Tensor, y_soft: torch.Tensor, was_mixed: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device.type == "cuda":
            xb_in = xb_in.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_bf16):
            if cfg.two_view_flip and (cfg.flip_kl_weight > 0.0 or cfg.supcon_weight > 0.0):
                xb_cat = torch.cat([xb_in, torch.flip(xb_in, dims=[3])], dim=0)
                logits_cat, feat_cat = model(xb_cat, return_embed=True)
                l1, l2 = logits_cat.float().chunk(2, dim=0)
                f1, f2 = feat_cat.float().chunk(2, dim=0)
                if was_mixed:
                    loss = ce_fn(l1, y_soft) + cfg.flip_kl_weight * flip_fn(l1, l2, f1, f2)
                else:
                    loss = ce_fn(l1, yb_hard)
                    if cfg.flip_kl_weight > 0.0:
                        loss = loss + cfg.flip_kl_weight * flip_fn(l1, l2, f1, f2)
                    if cfg.supcon_weight > 0.0:
                        loss = loss + cfg.supcon_weight * supcon_fn(torch.stack([f1, f2], dim=1), yb_hard)
                return loss, l1.detach()
            else:
                logits1, feat1 = model(xb_in, return_embed=True)
                l1 = logits1.float()
                if was_mixed:
                    loss = ce_fn(l1, y_soft)
                else:
                    loss = ce_fn(l1, yb_hard)
                    if cfg.supcon_weight > 0.0:
                        loss = loss + cfg.supcon_weight * supcon_fn(feat1.float(), yb_hard)
                return loss, l1.detach()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "val_acc": [],
        "val_roc_auc": [],
        "lr": [],
    }
    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = -1
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0
    t_start = time.time()

    for ep in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_seen = 0
        tr_preds_all: List[np.ndarray] = []
        tr_targets_all: List[np.ndarray] = []
        use_mix = (cfg.mix_prob > 0.0) and (cfg.mix_start_epoch <= ep <= cfg.mix_end_epoch)

        for start in range(0, n_train, bs):
            idx = perm[start : start + bs]
            if idx.numel() <= 1:
                continue
            xb = gpu_augment_batch(
                train_x_u8_dev[idx],
                mean_v=cfg.mean,
                std_v=cfg.std,
                aug_mode=cfg.aug_mode,
                erase_prob=cfg.facial_erase_prob,
            )
            yb = train_y_dev[idx]
            if use_mix:
                xb_in, y_soft, was_mixed = apply_batch_mix(
                    xb,
                    yb,
                    num_classes=len(CLASS_NAMES),
                    mix_prob=cfg.mix_prob,
                    mixup_alpha=cfg.mixup_alpha,
                    cutmix_alpha=cfg.cutmix_alpha,
                    splitmix_prob=cfg.splitmix_prob,
                )
            else:
                xb_in, y_soft, was_mixed = xb, F.one_hot(yb, len(CLASS_NAMES)).float(), False

            if cfg.use_sam:
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                optimizer.zero_grad(set_to_none=True)
                loss1, l1_det = _forward_loss(xb_in, yb, y_soft, was_mixed)
                loss1.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.first_step(zero_grad=True)

                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, device)
                saved_bn = disable_bn_running_stats(model)
                try:
                    loss2, _ = _forward_loss(xb_in, yb, y_soft, was_mixed)
                    loss2.backward()
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                finally:
                    restore_bn_running_stats(saved_bn)
                optimizer.second_step(zero_grad=True)
                b_loss = float(loss1.detach().item())
            else:
                optimizer.zero_grad(set_to_none=True)
                loss1, l1_det = _forward_loss(xb_in, yb, y_soft, was_mixed)
                loss1.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                b_loss = float(loss1.detach().item())

            if ema is not None:
                ema.update(model)

            total_loss += b_loss * xb.size(0)
            n_seen += xb.size(0)
            tr_targets_all.append(yb.cpu().numpy())
            tr_preds_all.append(l1_det.argmax(1).cpu().numpy())

        train_loss_ep = total_loss / max(1, n_seen)
        train_f1_ep = float(
            f1_score(np.concatenate(tr_targets_all), np.concatenate(tr_preds_all), average="macro", zero_division=0)
        )

        # Evaluate raw model and EMA model on val (Single-View)
        eval_target = ema.ema if ema is not None else model
        val_logits, val_probs, val_preds = predict_cnn_on_images(
            eval_target, val_imgs_u8, cfg.mean, cfg.std, device, batch_size=256, use_tta=False
        )
        if ema is not None:
            raw_logits, raw_probs, raw_preds = predict_cnn_on_images(
                model, val_imgs_u8, cfg.mean, cfg.std, device, batch_size=256, use_tta=False
            )
            f1_ema = float(f1_score(val_labels, val_preds, average="macro", zero_division=0))
            f1_raw = float(f1_score(val_labels, raw_preds, average="macro", zero_division=0))
            if f1_raw > f1_ema:
                val_logits, val_probs, val_preds = raw_logits, raw_probs, raw_preds
                eval_target = model

        with torch.no_grad():
            val_loss_ep = float(ce_fn(torch.from_numpy(val_logits).to(device), val_y_dev).item())
        val_f1_ep = float(f1_score(val_labels, val_preds, average="macro", zero_division=0))
        val_acc_ep = float(accuracy_score(val_labels, val_preds))
        try:
            val_auc_ep = float(roc_auc_score(val_labels, val_probs, multi_class="ovr", average="macro"))
        except Exception:
            val_auc_ep = 0.0

        curr_lr = float(sched_opt.param_groups[0]["lr"])
        if cfg.scheduler == "plateau":
            scheduler.step(val_f1_ep)
        else:
            scheduler.step()

        history["train_loss"].append(round(train_loss_ep, 5))
        history["val_loss"].append(round(val_loss_ep, 5))
        history["train_f1"].append(round(train_f1_ep, 5))
        history["val_f1"].append(round(val_f1_ep, 5))
        history["val_acc"].append(round(val_acc_ep, 5))
        history["val_roc_auc"].append(round(val_auc_ep, 5))
        history["lr"].append(curr_lr)

        if soup is not None and ep >= max(3, cfg.warmup_epochs):
            soup.maybe_add(val_f1_ep, ep, eval_target)

        is_best = val_f1_ep > best_val_f1
        if is_best:
            best_val_f1 = val_f1_ep
            best_val_acc = val_acc_ep
            best_epoch = ep
            no_improve = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in eval_target.state_dict().items()}
        else:
            no_improve += 1

        if verbose and (ep == 1 or ep % 5 == 0 or ep == cfg.epochs or is_best):
            print(
                f"[{cfg.name}] Ep {ep:02d}/{cfg.epochs:02d} | "
                f"tr_loss={train_loss_ep:.4f} tr_f1={train_f1_ep:.4f} | "
                f"val_loss={val_loss_ep:.4f} val_f1={val_f1_ep:.4f} val_acc={val_acc_ep:.4f} auc={val_auc_ep:.4f} | "
                f"lr={curr_lr:.1e} | {time.time() - ep_t0:.1f}s"
                + (" <- best" if is_best else ""),
                flush=True,
            )

        if no_improve >= cfg.patience:
            if verbose:
                print(f"[{cfg.name}] Early stopping at epoch {ep} (patience={cfg.patience})", flush=True)
            break

    assert best_state_dict is not None
    selected_source = "best_ema"

    # Evaluate Greedy Model Soup on val if enabled
    if soup is not None and len(soup.snapshots) >= 2:
        eval_m = KDEF_Stem2Ch_MS4_Symmetric(in_channels=1, num_classes=len(CLASS_NAMES)).to(device)

        def _eval_sd_f1(sd: Dict[str, torch.Tensor]) -> float:
            eval_m.load_state_dict({k: v.to(device) for k, v in sd.items()})
            _, _, preds = predict_cnn_on_images(eval_m, val_imgs_u8, cfg.mean, cfg.std, device)
            return float(f1_score(val_labels, preds, average="macro", zero_division=0))

        greedy_sd, greedy_f1, greedy_eps = soup.build_greedy_soup(_eval_sd_f1)
        if greedy_f1 > best_val_f1:
            if verbose:
                print(
                    f"[{cfg.name}] Greedy Soup ({greedy_eps}) improved val Macro-F1: "
                    f"{best_val_f1:.4f} -> {greedy_f1:.4f}",
                    flush=True,
                )
            best_val_f1 = greedy_f1
            best_state_dict = greedy_sd
            selected_source = f"greedy_soup({greedy_eps})"

    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    # Optional Step 4: Fine-tune with lr/4 and no mixup (accepted strictly if val Macro-F1 improves)
    finetune_history: List[Dict[str, float]] = []
    if cfg.finetune_epochs > 0:
        if verbose:
            print(
                f"[{cfg.name}] Step 4 Fine-tuning for up to {cfg.finetune_epochs} epochs at lr={cfg.lr * 0.25:.1e}...",
                flush=True,
            )
        ft_model = copy.deepcopy(model).to(device)
        ft_ema = ModelEMA(ft_model, decay=0.992)
        ft_opt = torch.optim.AdamW(ft_model.parameters(), lr=cfg.lr * 0.25, weight_decay=cfg.weight_decay)
        ft_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=cfg.finetune_epochs, eta_min=cfg.min_lr)
        ft_best_f1 = best_val_f1
        ft_best_sd: Optional[Dict[str, torch.Tensor]] = None
        ft_best_ep = best_epoch

        for ft_ep in range(1, cfg.finetune_epochs + 1):
            ft_model.train()
            perm = torch.randperm(n_train, device=device)
            ft_loss_sum = 0.0
            ft_seen = 0
            for start in range(0, n_train, bs):
                idx = perm[start : start + bs]
                if idx.numel() <= 1:
                    continue
                xb = gpu_augment_batch(
                    train_x_u8_dev[idx], mean_v=cfg.mean, std_v=cfg.std, aug_mode="baseline", erase_prob=0.0
                )
                yb = train_y_dev[idx]
                ft_opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_bf16):
                    logits, _ = ft_model(xb, return_embed=True)
                    loss = ce_fn(logits.float(), yb)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ft_model.parameters(), cfg.grad_clip)
                ft_opt.step()
                ft_ema.update(ft_model)
                ft_loss_sum += float(loss.detach().item()) * xb.size(0)
                ft_seen += xb.size(0)
            ft_sched.step()

            for cand_tag, cand_m in (("ft_ema", ft_ema.ema), ("ft_raw", ft_model)):
                _, _, ft_preds = predict_cnn_on_images(cand_m, val_imgs_u8, cfg.mean, cfg.std, device)
                ft_f1 = float(f1_score(val_labels, ft_preds, average="macro", zero_division=0))
                ft_acc = float(accuracy_score(val_labels, ft_preds))
                if ft_f1 > ft_best_f1:
                    ft_best_f1 = ft_f1
                    ft_best_sd = {k: v.detach().cpu().clone() for k, v in cand_m.state_dict().items()}
                    ft_best_ep = len(history["train_loss"]) + ft_ep
                    selected_source = f"finetuned_{cand_tag}"
            finetune_history.append({"ft_epoch": ft_ep, "val_f1": round(ft_best_f1, 5)})

        if ft_best_sd is not None and ft_best_f1 > best_val_f1:
            if verbose:
                print(
                    f"[{cfg.name}] Fine-tuning improved val Macro-F1: {best_val_f1:.4f} -> {ft_best_f1:.4f}",
                    flush=True,
                )
            best_val_f1 = ft_best_f1
            best_state_dict = ft_best_sd
            best_epoch = ft_best_ep
            model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        else:
            if verbose:
                print(
                    f"[{cfg.name}] Fine-tuning did not exceed pre-finetune val Macro-F1 ({best_val_f1:.4f}); keeping pre-finetune weights.",
                    flush=True,
                )

    # Pre-calibration val evaluation
    val_logits_pre, val_probs_pre, val_preds_pre = predict_cnn_on_images(
        model, val_imgs_u8, cfg.mean, cfg.std, device, batch_size=256, use_tta=False
    )
    pre_bias_f1 = float(f1_score(val_labels, val_preds_pre, average="macro", zero_division=0))
    pre_bias_acc = float(accuracy_score(val_labels, val_preds_pre))

    # Optional val logit-bias calibration absorbed directly into head.classifier.bias
    logit_bias = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    if cfg.tune_val_bias:
        logit_bias, cal_f1, cal_acc = optimize_val_logit_bias(model, val_logits_pre, val_labels)
        if verbose and cal_f1 > pre_bias_f1:
            print(
                f"[{cfg.name}] Val prior bias calibration improved Single-View Val Macro-F1: "
                f"{pre_bias_f1:.4f} (Acc {pre_bias_acc:.4f}) -> {cal_f1:.4f} (Acc {cal_acc:.4f})",
                flush=True,
            )
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Final Single-View and Flip-TTA evaluation on val
    val_logits_sv, val_probs_sv, val_preds_sv = predict_cnn_on_images(
        model, val_imgs_u8, cfg.mean, cfg.std, device, batch_size=256, use_tta=False
    )
    val_metrics_sv = compute_full_val_metrics(val_labels, val_preds_sv, val_probs_sv, meta_df=val_df)

    _, val_probs_tta, val_preds_tta = predict_cnn_on_images(
        model, val_imgs_u8, cfg.mean, cfg.std, device, batch_size=256, use_tta=True
    )
    val_metrics_tta = compute_full_val_metrics(val_labels, val_preds_tta, val_probs_tta, meta_df=val_df)

    elapsed_total = round(time.time() - t_start, 2)
    n_params = count_parameters(model)

    if save_ckpt_path is not None:
        save_p = Path(save_ckpt_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        ckpt_payload = {
            "model_state": best_state_dict,
            "state_dict": best_state_dict,
            "model_name": model.model_name,
            "arch": model.model_name,
            "epoch": best_epoch,
            "selected_artifact": selected_source,
            "val_f1": val_metrics_sv["macro_f1"],
            "val_macro_f1": val_metrics_sv["macro_f1"],
            "val_weighted_f1": val_metrics_sv["weighted_f1"],
            "val_accuracy": val_metrics_sv["accuracy"],
            "val_roc_auc": val_metrics_sv["roc_auc_ovr_macro"],
            "val_tta_macro_f1": val_metrics_tta["macro_f1"],
            "val_tta_accuracy": val_metrics_tta["accuracy"],
            "val_tta_roc_auc": val_metrics_tta["roc_auc_ovr_macro"],
            "pre_calibration_val_macro_f1": round(pre_bias_f1, 6),
            "pre_calibration_val_accuracy": round(pre_bias_acc, 6),
            "logit_bias_absorbed": [round(float(x), 5) for x in logit_bias.tolist()],
            "num_params": n_params,
            "img_size": cfg.img_size,
            "crop_mode": cfg.crop_mode,
            "ssd_margin": cfg.ssd_margin,
            "class_names": CLASS_NAMES,
            "dataset": "multicultural_3source (KDEF + RADIATE + FER-2013)",
            "normalization": {"mean": cfg.mean, "std": cfg.std},
            "config": {
                "model_name": model.model_name,
                "in_channels": 1,
                "crop": cfg.use_crop,
                "clahe": cfg.use_clahe,
                "img_size": cfg.img_size,
                "crop_mode": cfg.crop_mode,
                "ssd_margin": cfg.ssd_margin,
                "epochs": cfg.epochs,
                "warmup_epochs": cfg.warmup_epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "class_weight_power": cfg.class_weight_power,
                "label_smoothing": cfg.label_smoothing,
                "focal_gamma": cfg.focal_gamma,
                "poly_eps": cfg.poly_eps,
                "use_sam": cfg.use_sam,
                "sam_rho": cfg.sam_rho,
                "supcon_weight": cfg.supcon_weight,
                "flip_kl_weight": cfg.flip_kl_weight,
                "use_ema": cfg.use_ema,
            },
            "cfg": asdict(cfg),
            "history": history,
            "val_metrics": val_metrics_sv,
            "val_metrics_tta": val_metrics_tta,
        }
        torch.save(ckpt_payload, save_p)

    del train_x_u8_dev, train_y_dev, val_y_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": cfg.name,
        "selected_artifact": selected_source,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "elapsed_sec": elapsed_total,
        "num_params": n_params,
        "pre_calibration_val_macro_f1": round(pre_bias_f1, 6),
        "pre_calibration_val_accuracy": round(pre_bias_acc, 6),
        "val_metrics_single_view": val_metrics_sv,
        "val_metrics_flip_tta": val_metrics_tta,
        "history": history,
        "finetune_history": finetune_history,
        "config": asdict(cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multicultural CNN Training (KDEF_Stem2Ch_MS4_Symmetric)")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MULTICULTURAL_MANIFEST))
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.65)
    parser.add_argument("--label-smoothing", type=float, default=0.06)
    parser.add_argument("--focal-gamma", type=float, default=0.75)
    parser.add_argument("--poly-eps", type=float, default=0.20)
    parser.add_argument("--flip-kl-weight", type=float, default=0.30)
    parser.add_argument("--supcon-weight", type=float, default=0.15)
    parser.add_argument("--mix-prob", type=float, default=0.30)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--save-ckpt", type=str, default="checkpoints/final_model.pt")
    parser.add_argument("--save-alt-ckpt", type=str, default="checkpoints/ckpt_stem2ch_ms4_fuse_multicultural.pt")
    parser.add_argument("--save-results", type=str, default="results/val_metrics_cnn_multicultural.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weight_power=args.class_weight_power,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        poly_eps=args.poly_eps,
        flip_kl_weight=args.flip_kl_weight,
        supcon_weight=args.supcon_weight,
        mix_prob=args.mix_prob,
        finetune_epochs=args.finetune_epochs,
        seed=args.seed,
    )

    print(f"Loading cached multicultural train & val data...", flush=True)
    train_imgs, train_lbls, train_df = load_multicultural_split_cached("train", img_size=cfg.img_size, manifest_path=args.manifest)
    val_imgs, val_lbls, val_df = load_multicultural_split_cached("val", img_size=cfg.img_size, manifest_path=args.manifest)

    print(f"Loaded train: {len(train_imgs)} images, val: {len(val_imgs)} images.", flush=True)
    base_cw = compute_class_weights(train_df)

    dev = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {dev}", flush=True)

    result = train_cnn_run(
        cfg=cfg,
        train_imgs_u8=train_imgs,
        train_labels=train_lbls,
        val_imgs_u8=val_imgs,
        val_labels=val_lbls,
        val_df=val_df,
        base_class_weights=base_cw,
        device=dev,
        save_ckpt_path=args.save_ckpt,
        verbose=True,
    )

    if args.save_alt_ckpt:
        alt_path = Path(args.save_alt_ckpt)
        alt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.save_ckpt, alt_path)
        print(f"Checkpoint copied to {alt_path}", flush=True)

    if args.save_results:
        res_p = Path(args.save_results)
        res_p.parent.mkdir(parents=True, exist_ok=True)
        with open(res_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Validation metrics saved to {res_p}", flush=True)


if __name__ == "__main__":
    main()
