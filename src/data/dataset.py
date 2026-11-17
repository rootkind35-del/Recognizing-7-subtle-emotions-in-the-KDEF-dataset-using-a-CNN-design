"""
KDEF Dataset & Data Transformations Module.
Optimized for KDEF Facial Emotion Recognition (Hoc-May project).

Supports:
- Clean train/val/test splits of official-kdef-dataset-by-ksatrajit0
- Automatic exclusion of corrupt pitch-black frames (MD5: 320b34032e2133fdfb80a58c953d5c94)
- SSD face-centered square-pad cropping (`ssd_pad_160_1ch`, margin=1.25) with studio fallback
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Robust case-insensitive file matching (.JPG / .jpg)
- Extraction of facial metadata (subject ID, gender, emotion code, pose angle: FL, FR, HL, HR, S)
- Geometric and photometric augmentations for robust pose and expression generalization
"""

import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.preprocessing.alignment import (
    DEFAULT_CROP_MODE,
    DEFAULT_SSD_MARGIN,
    KDEF_STUDIO_CROP_BOX,
    preprocess_image,
    preprocess_kdef_image,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CORRUPT_MD5 = "320b34032e2133fdfb80a58c953d5c94"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

ANGLE_NAMES = ["FL", "FR", "HL", "HR", "S"]
ANGLE_DESCRIPTIONS = {
    "FL": "Full Left (-90° / profile)",
    "HL": "Half Left (-45° / semi-profile)",
    "S":  "Straight (0° / frontal)",
    "HR": "Half Right (+45° / semi-profile)",
    "FR": "Full Right (+90° / profile)"
}

EMOTION_CODE_MAP = {
    "AN": "angry",
    "DI": "disgust",
    "AF": "fear",
    "HA": "happy",
    "NE": "neutral",
    "SA": "sad",
    "SU": "surprise"
}

# Default dataset normalization constants for ssd_pad_160_1ch (160x160 CLAHE grayscale)
# Strictly computed on the clean train split (3,349 images, excluding 10 black frames)
DEFAULT_KDEF_MEAN = 0.407255
DEFAULT_KDEF_STD = 0.209657

# Default 3-source multicultural normalization constants for ssd_pad_160_1ch
# Strictly computed on full clean 3-source train split (32,404 images, excluding val and test)
DEFAULT_MULTICULTURAL_MEAN = 0.517243
DEFAULT_MULTICULTURAL_STD = 0.264053

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MULTICULTURAL_MANIFEST = REPO_ROOT / "data" / "manifests" / "multicultural_manifest.csv"


def parse_kdef_filename(filename: str) -> Dict[str, str]:
    """
    Parses a standard KDEF image filename (e.g., 'AF01ANFL.JPG').
    Format: [Series][Gender][ID][Emotion][Angle].[ext]
    - Series: A (series 1) or B (series 2)
    - Gender: F (female) or M (male)
    - ID: 01 to 35
    - Emotion: AN, DI, AF, HA, NE, SA, SU
    - Angle: FL, HL, S, HR, FR
    """
    stem = Path(filename).stem
    if len(stem) >= 7:
        series = stem[0]
        gender = stem[1]
        subject_id = stem[1:4]
        emo_code = stem[4:6]
        angle = stem[6:]
        return {
            "stem": stem,
            "series": series,
            "gender": "female" if gender.upper() == "F" else "male",
            "subject_id": subject_id,
            "emotion_code": emo_code,
            "emotion": EMOTION_CODE_MAP.get(emo_code.upper(), "unknown"),
            "angle": angle if angle in ANGLE_NAMES else "unknown"
        }
    return {
        "stem": stem,
        "series": "unknown",
        "gender": "unknown",
        "subject_id": "unknown",
        "emotion_code": "unknown",
        "emotion": "unknown",
        "angle": "unknown"
    }


def compute_file_md5(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """Calculates MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


class KDEFDataset(Dataset):
    """
    PyTorch Dataset for KDEF Facial Emotion Recognition.
    Handles data discovery, corrupt frame exclusion, SSD face-centered square-pad
    cropping (`ssd_pad_160_1ch` by default), CLAHE preprocessing, and augmentation.
    """
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        use_crop: bool = True,
        crop_mode: str = DEFAULT_CROP_MODE,
        ssd_margin: float = DEFAULT_SSD_MARGIN,
        crop_box: Tuple[float, float, float, float] = KDEF_STUDIO_CROP_BOX,
        use_clahe: bool = True,
        clahe_clip: float = 2.0,
        clahe_blend: float = 0.8,
        img_size: int = 160,
        in_channels: int = 1,
        transform: Optional[Callable] = None,
        exclude_corrupt: bool = True,
        return_metadata: bool = False
    ):
        super().__init__()
        self.split = split.lower()
        self.use_crop = use_crop
        self.crop_mode = crop_mode
        self.ssd_margin = ssd_margin
        self.crop_box = crop_box
        self.use_clahe = use_clahe
        self.clahe_clip = clahe_clip
        self.clahe_blend = clahe_blend
        self.img_size = img_size
        self.in_channels = in_channels
        self.transform = transform
        self.exclude_corrupt = exclude_corrupt
        self.return_metadata = return_metadata

        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)) if use_clahe else None

        # Resolve dataset root
        if data_dir is None:
            # Check standard repo locations
            repo_root = Path(__file__).resolve().parent.parent.parent
            candidate = repo_root / "data" / "official-kdef-dataset-by-ksatrajit0"
            if candidate.exists():
                data_dir = candidate
            else:
                data_dir = Path("/tmp/kdef_official")
        self.data_dir = Path(data_dir).resolve()

        self.split_dir = self.data_dir / self.split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"KDEF split directory does not exist: {self.split_dir}")

        self.samples: List[Tuple[Path, int, Dict[str, str]]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Discovers valid image files and filters corrupt dark frames."""
        valid_extensions = {".jpg", ".jpeg", ".png"}

        # Find all emotion directories (case-insensitive)
        subdirs = [p for p in self.split_dir.iterdir() if p.is_dir()]
        dir_map = {}
        for d in subdirs:
            dir_map[d.name.lower()] = d

        for cls_name in CLASS_NAMES:
            if cls_name not in dir_map:
                continue
            cls_dir = dir_map[cls_name]
            cls_idx = CLASS_TO_IDX[cls_name]

            # Discover all image files regardless of extension case
            for fpath in sorted(cls_dir.iterdir()):
                if not fpath.is_file():
                    continue
                if fpath.suffix.lower() not in valid_extensions:
                    continue

                # Filter out known corrupt pitch-black frames
                if self.exclude_corrupt:
                    digest = compute_file_md5(fpath)
                    if digest == CORRUPT_MD5:
                        continue

                meta = parse_kdef_filename(fpath.name)
                self.samples.append((fpath, cls_idx, meta))

    def __len__(self) -> int:
        return len(self.samples)

    def _preprocess_raw_image(self, img_arr: np.ndarray) -> np.ndarray:
        """
        Applies SSD face-centered cropping (or studio fallback) and CLAHE contrast enhancement.
        img_arr: 2D uint8 numpy array (H, W) or 3D BGR array.
        """
        return preprocess_kdef_image(
            img_arr,
            use_aspect_crop=self.use_crop,
            use_clahe=self.use_clahe,
            clahe_clip=self.clahe_clip,
            clahe_blend=self.clahe_blend,
            img_size=self.img_size,
            crop_mode=self.crop_mode,
            ssd_margin=self.ssd_margin,
            crop_box=self.crop_box,
        )

    def __getitem__(self, idx: int):
        fpath, label, meta = self.samples[idx]

        # Standardized preprocessing with SSD square-pad crop, cv2.INTER_AREA & CLAHE
        proc_arr = preprocess_kdef_image(
            fpath,
            use_aspect_crop=self.use_crop,
            use_clahe=self.use_clahe,
            clahe_clip=self.clahe_clip,
            clahe_blend=self.clahe_blend,
            img_size=self.img_size,
            crop_mode=self.crop_mode,
            ssd_margin=self.ssd_margin,
            crop_box=self.crop_box,
        )

        tensor = torch.from_numpy(proc_arr).unsqueeze(0).float() / 255.0  # (1, H, W)
        if self.in_channels == 3:
            tensor = tensor.repeat(3, 1, 1)

        if self.transform is not None:
            tensor = self.transform(tensor)

        if self.return_metadata:
            return tensor, label, meta
        return tensor, label


def get_transforms(
    split: str = "train",
    mean: float = DEFAULT_KDEF_MEAN,
    std: float = DEFAULT_KDEF_STD,
    aug_mode: str = "enhanced",
    in_channels: int = 1
) -> transforms.Compose:
    """
    Returns data transforms for train, val, or test splits.
    Operates on [0, 1] float tensors and applies normalization.
    """
    norm_mean = [mean] * in_channels
    norm_std = [std] * in_channels
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    if split.lower() == "train":
        if aug_mode == "enhanced":
            # State-of-the-art geometric + photometric augmentation recipe
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.04, 0.04),
                    scale=(0.95, 1.05)
                ),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
                normalize
            ])
        elif aug_mode == "baseline":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, scale=(1 / 1.1, 1.1)),
                normalize
            ])
        else:
            return transforms.Compose([normalize])
    else:
        # Validation / Test deterministic evaluation transform
        return transforms.Compose([normalize])


def compute_kdef_mean_std(
    data_dir: Optional[Union[str, Path]] = None,
    num_samples_per_class: Optional[int] = 80,
    seed: int = 42,
    use_crop: bool = True,
    crop_mode: str = DEFAULT_CROP_MODE,
    ssd_margin: float = DEFAULT_SSD_MARGIN,
    use_clahe: bool = True,
    img_size: int = 160
) -> Tuple[float, float]:
    """
    Computes exact dataset mean and standard deviation strictly on the clean
    training split (default: 80 samples per class = 560 samples, seed=42, or full
    clean train split if `num_samples_per_class is None`).
    """
    import random
    rng = random.Random(seed)

    ds = KDEFDataset(
        data_dir=data_dir,
        split="train",
        use_crop=use_crop,
        crop_mode=crop_mode,
        ssd_margin=ssd_margin,
        use_clahe=use_clahe,
        img_size=img_size,
        in_channels=1,
        transform=None,
        exclude_corrupt=True
    )

    class_indices = defaultdict(list)
    for i, (_, label) in enumerate(ds):
        class_indices[label].append(i)

    sampled_indices = []
    for c in range(len(CLASS_NAMES)):
        indices = class_indices[c]
        if num_samples_per_class is None:
            sampled_indices.extend(indices)
        else:
            k = min(num_samples_per_class, len(indices))
            sampled_indices.extend(rng.sample(indices, k))

    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0

    for idx in sampled_indices:
        t, _ = ds[idx]
        total_sum += float(t.sum().item())
        total_sq_sum += float((t ** 2).sum().item())
        total_pixels += t.numel()

    mean = total_sum / total_pixels
    var = (total_sq_sum / total_pixels) - (mean ** 2)
    std = float(np.sqrt(max(var, 1e-12)))

    return round(mean, 6), round(std, 6)


class MulticulturalFERDataset(Dataset):
    """
    PyTorch Dataset for the unified 3-source Multicultural FER manifest
    (KDEF + RADIATE + FER-2013), applying source-aware `ssd_pad_160_1ch` preprocessing.
    """

    META_COLS = ("source", "subject_id", "ethnic_group", "gender", "angle", "md5", "label", "filepath")

    def __init__(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        split: str = "train",
        img_size: int = 160,
        transform: Optional[Callable] = None,
        return_metadata: bool = True,
        in_channels: int = 1,
        use_crop: bool = True,
        crop_mode: str = DEFAULT_CROP_MODE,
        ssd_margin: float = DEFAULT_SSD_MARGIN,
        use_clahe: bool = True,
        clahe_clip: float = 2.0,
        clahe_blend: float = 0.8,
        root_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.split = split.strip().lower()
        self.img_size = img_size
        self.transform = transform
        self.return_metadata = return_metadata
        self.in_channels = in_channels
        self.use_crop = use_crop
        self.crop_mode = crop_mode
        self.ssd_margin = ssd_margin
        self.use_clahe = use_clahe
        self.clahe_clip = clahe_clip
        self.clahe_blend = clahe_blend

        self.root_dir = Path(root_dir).resolve() if root_dir is not None else REPO_ROOT
        m_path = Path(manifest_path) if manifest_path is not None else DEFAULT_MULTICULTURAL_MANIFEST
        if not m_path.is_absolute():
            m_path = self.root_dir / m_path
        if not m_path.exists():
            raise FileNotFoundError(f"Multicultural manifest not found: {m_path}")
        self.manifest_path = m_path

        df = pd.read_csv(self.manifest_path)
        if "split" not in df.columns:
            raise ValueError(f"Manifest missing 'split' column: {self.manifest_path}")
        df = df[df["split"].astype(str).str.strip().str.lower() == self.split].copy()
        df.reset_index(drop=True, inplace=True)

        for col in self.META_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("unknown").astype(str)
            else:
                df[col] = "unknown"

        if "label_idx" in df.columns:
            df["label_idx"] = df["label_idx"].astype(int)
        else:
            df["label_idx"] = df["label"].map(CLASS_TO_IDX).astype(int)

        if (df["label_idx"] < 0).any() or (df["label_idx"] >= len(CLASS_NAMES)).any():
            raise ValueError("Invalid label_idx values found in manifest.")

        self.df = df
        self.samples: List[Tuple[Path, int, Dict[str, str]]] = []
        for row in self.df.itertuples(index=False):
            rel_fp = Path(row.filepath)
            fpath = rel_fp if rel_fp.is_absolute() else (self.root_dir / rel_fp)
            meta = {
                "source": row.source,
                "subject_id": row.subject_id,
                "ethnic_group": row.ethnic_group,
                "gender": row.gender,
                "angle": row.angle,
                "md5": row.md5,
                "label": row.label,
                "filepath": row.filepath,
            }
            self.samples.append((fpath, int(row.label_idx), meta))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fpath, label_idx, meta = self.samples[idx]
        proc_arr = preprocess_image(
            fpath,
            source=meta["source"],
            use_aspect_crop=self.use_crop,
            use_clahe=self.use_clahe,
            clahe_clip=self.clahe_clip,
            clahe_blend=self.clahe_blend,
            img_size=self.img_size,
            crop_mode=self.crop_mode,
            ssd_margin=self.ssd_margin,
        )
        tensor = torch.from_numpy(proc_arr)  # float32 (1, H, W) in [0, 1]
        if self.in_channels == 3:
            tensor = tensor.repeat(3, 1, 1)

        if self.transform is not None:
            tensor = self.transform(tensor)

        if self.return_metadata:
            return tensor, label_idx, meta
        return tensor, label_idx


def compute_class_weights(
    manifest_df_train: Union[pd.DataFrame, str, Path],
) -> torch.Tensor:
    """
    Computes inverse-frequency balanced class weights `N / (7 * N_c)` strictly on the
    `train` split.
    """
    if isinstance(manifest_df_train, (str, Path)):
        df = pd.read_csv(manifest_df_train)
    else:
        df = manifest_df_train.copy()

    if "split" in df.columns:
        df = df[df["split"].astype(str).str.strip().str.lower() == "train"]

    if len(df) == 0:
        raise ValueError("Empty training dataframe passed to compute_class_weights.")

    if "label_idx" in df.columns:
        labels = df["label_idx"].astype(int).to_numpy()
    elif "label" in df.columns:
        labels = df["label"].astype(str).map(CLASS_TO_IDX).astype(int).to_numpy()
    else:
        raise ValueError("Manifest must contain 'label_idx' or 'label' column.")

    n_total = len(labels)
    n_classes = len(CLASS_NAMES)
    counts = np.bincount(labels, minlength=n_classes)
    weights = [float(n_total) / (float(n_classes) * float(max(int(counts[c]), 1))) for c in range(n_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def compute_multicultural_mean_std(
    manifest_path: Optional[Union[str, Path]] = None,
    num_samples_per_class: Optional[int] = None,
    seed: int = 42,
    use_crop: bool = True,
    crop_mode: str = DEFAULT_CROP_MODE,
    ssd_margin: float = DEFAULT_SSD_MARGIN,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_blend: float = 0.8,
    img_size: int = 160,
    batch_size: int = 64,
    num_workers: int = 0,
    root_dir: Optional[Union[str, Path]] = None,
) -> Tuple[float, float]:
    """
    Computes exact pixel mean and std strictly on the clean 3-source `train` split
    (excluding `val` and `test`). Uses full clean `train` split by default.
    """
    import random

    ds = MulticulturalFERDataset(
        manifest_path=manifest_path,
        split="train",
        img_size=img_size,
        transform=None,
        return_metadata=False,
        in_channels=1,
        use_crop=use_crop,
        crop_mode=crop_mode,
        ssd_margin=ssd_margin,
        use_clahe=use_clahe,
        clahe_clip=clahe_clip,
        clahe_blend=clahe_blend,
        root_dir=root_dir,
    )

    if num_samples_per_class is not None:
        rng = random.Random(seed)
        class_indices = defaultdict(list)
        for i, (_, lbl, _) in enumerate(ds.samples):
            class_indices[lbl].append(i)
        sampled_indices: List[int] = []
        for c in range(len(CLASS_NAMES)):
            idxs = class_indices[c]
            k = min(num_samples_per_class, len(idxs))
            sampled_indices.extend(rng.sample(idxs, k))
        subset = torch.utils.data.Subset(ds, sampled_indices)
    else:
        subset = ds

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0

    for batch_x, _ in loader:
        b64 = batch_x.to(torch.float64)
        total_sum += float(b64.sum().item())
        total_sq_sum += float((b64 * b64).sum().item())
        total_pixels += int(b64.numel())

    mean = total_sum / total_pixels
    var = (total_sq_sum / total_pixels) - (mean ** 2)
    std = float(np.sqrt(max(var, 1e-12)))
    return round(mean, 6), round(std, 6)
