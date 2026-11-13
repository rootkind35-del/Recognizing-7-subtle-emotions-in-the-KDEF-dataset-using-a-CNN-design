"""
KDEF Dataset & Data Transformations Module.
Optimized for KDEF Facial Emotion Recognition (Hoc-May project).

Supports:
- Clean train/val/test splits of official-kdef-dataset-by-ksatrajit0
- Automatic exclusion of corrupt pitch-black frames (MD5: 320b34032e2133fdfb80a58c953d5c94)
- Aspect-aware facial region cropping [0.04W:0.96W, 0.16H:0.86H]
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
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.preprocessing.alignment import preprocess_kdef_image

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

# Default dataset normalization constants for aspect-cropped CLAHE grayscale 128x128
# Strictly computed on the clean train split (3,349 images, excluding 10 black frames)
DEFAULT_KDEF_MEAN = 0.407835
DEFAULT_KDEF_STD = 0.242375


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
    Handles data discovery, corrupt frame exclusion, aspect-aware cropping,
    CLAHE preprocessing, and augmentation.
    """
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        use_crop: bool = True,
        crop_box: Tuple[float, float, float, float] = (0.04, 0.16, 0.96, 0.86),
        use_clahe: bool = True,
        clahe_clip: float = 2.0,
        clahe_blend: float = 0.8,
        img_size: int = 128,
        in_channels: int = 1,
        transform: Optional[Callable] = None,
        exclude_corrupt: bool = True,
        return_metadata: bool = False
    ):
        super().__init__()
        self.split = split.lower()
        self.use_crop = use_crop
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
        Applies aspect-aware cropping and CLAHE contrast enhancement via preprocess_kdef_image.
        img_arr: 2D uint8 numpy array (H, W).
        """
        return preprocess_kdef_image(
            img_arr,
            use_aspect_crop=self.use_crop,
            use_clahe=self.use_clahe,
            clahe_clip=self.clahe_clip,
            clahe_blend=self.clahe_blend,
            img_size=self.img_size
        )

    def __getitem__(self, idx: int):
        fpath, label, meta = self.samples[idx]

        # Standardized preprocessing with aspect crop, cv2.INTER_AREA & CLAHE
        proc_arr = preprocess_kdef_image(
            fpath,
            use_aspect_crop=self.use_crop,
            use_clahe=self.use_clahe,
            clahe_clip=self.clahe_clip,
            clahe_blend=self.clahe_blend,
            img_size=self.img_size
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
    num_samples_per_class: int = 80,
    seed: int = 42,
    use_crop: bool = True,
    use_clahe: bool = True,
    img_size: int = 128
) -> Tuple[float, float]:
    """
    Computes exact dataset mean and standard deviation strictly on the clean
    training split (default: 80 samples per class = 560 samples, seed=42).
    """
    import random
    rng = random.Random(seed)

    ds = KDEFDataset(
        data_dir=data_dir,
        split="train",
        use_crop=use_crop,
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
