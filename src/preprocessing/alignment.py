"""
KDEF Face Alignment & Standardized Preprocessing Module (`src/preprocessing/alignment.py`).

Supports:
- Cached OpenCV Res10 SSD face detection (`res10_300x300_ssd_iter_140000.caffemodel`)
  with fallback to Haar Cascade (`FaceAligner`) and studio aspect crop (`KDEF_STUDIO_CROP_BOX`)
- Face-centered symmetric square crop (`ssd_square_pad`, margin=1.25, `cv2.BORDER_REPLICATE`)
  aligning all 5 KDEF pose angles (`FL`, `HL`, `S`, `HR`, `FR`) and handling outlier dimensions
- Anti-aliased `cv2.INTER_AREA` downsampling to `160x160` + CLAHE (`clipLimit=2.0, blend=0.8`)
"""

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Aspect-aware studio portrait crop box [x1, y1, x2, y2] relative to image dimensions (fallback / legacy)
KDEF_STUDIO_CROP_BOX: Tuple[float, float, float, float] = (0.04, 0.16, 0.96, 0.86)
DEFAULT_CROP_MODE: str = "ssd_square_pad"
DEFAULT_SSD_MARGIN: float = 1.25
DEFAULT_SSD_MIN_CONF: float = 0.20

_PREPROC_DIR = Path(__file__).resolve().parent
_SSD_PROTOTXT = _PREPROC_DIR / "models" / "deploy.prototxt"
_SSD_CAFFEMODEL = _PREPROC_DIR / "models" / "res10_300x300_ssd_iter_140000.caffemodel"
_SSD_NET = None


def _get_ssd_net():
    """Lazily loads and caches the OpenCV Res10 SSD Caffe face detector."""
    global _SSD_NET
    if _SSD_NET is None:
        if not _SSD_PROTOTXT.exists() or not _SSD_CAFFEMODEL.exists():
            raise FileNotFoundError(
                f"Missing OpenCV SSD face detector weights under {_SSD_PROTOTXT.parent}"
            )
        _SSD_NET = cv2.dnn.readNetFromCaffe(str(_SSD_PROTOTXT), str(_SSD_CAFFEMODEL))
    return _SSD_NET


def detect_face_ssd(
    img_bgr: np.ndarray,
    min_conf: float = DEFAULT_SSD_MIN_CONF,
) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
    """
    Runs OpenCV Res10 SSD face detector on a BGR image.
    Returns `(normalized (x1, y1, x2, y2), confidence)` clamped to [0, 1], or `(None, conf)`.
    """
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    net = _get_ssd_net()
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    det = net.forward()[0, 0]
    best_idx = int(np.argmax(det[:, 2]))
    best = det[best_idx]
    conf = float(best[2])
    if conf < min_conf:
        return None, conf

    x1 = float(np.clip(best[3], 0.0, 1.0))
    y1 = float(np.clip(best[4], 0.0, 1.0))
    x2 = float(np.clip(best[5], 0.0, 1.0))
    y2 = float(np.clip(best[6], 0.0, 1.0))
    if x2 <= x1 or y2 <= y1:
        return None, conf
    return (x1, y1, x2, y2), conf


def _crop_studio(
    img: np.ndarray,
    crop_box: Tuple[float, float, float, float] = KDEF_STUDIO_CROP_BOX,
) -> np.ndarray:
    h, w = img.shape[:2]
    x1_ratio, y1_ratio, x2_ratio, y2_ratio = crop_box
    y1, y2 = int(round(y1_ratio * h)), int(round(y2_ratio * h))
    x1, x2 = int(round(x1_ratio * w)), int(round(x2_ratio * w))
    return img[y1:y2, x1:x2]


def _crop_ssd_square(
    img: np.ndarray,
    ssd_box: Optional[Sequence[float]],
    margin: float = DEFAULT_SSD_MARGIN,
    pad_mode: str = "pad",
    fallback_box: Tuple[float, float, float, float] = KDEF_STUDIO_CROP_BOX,
) -> np.ndarray:
    """
    Crops a face-centered square around `ssd_box` with side `max(w, h) * margin`.
    In `pad` mode, out-of-bounds margins are filled with `cv2.BORDER_REPLICATE` so the
    face remains horizontally/vertically centered across profile angles (`FL`, `FR`).
    """
    if ssd_box is None:
        return _crop_studio(img, fallback_box)

    h, w = img.shape[:2]
    x1n, y1n, x2n, y2n = ssd_box
    bw = (x2n - x1n) * w
    bh = (y2n - y1n) * h
    cx = 0.5 * (x1n + x2n) * w
    cy = 0.5 * (y1n + y2n) * h

    side = max(int(round(max(bw, bh) * margin)), 32)
    half = side / 2.0

    ix1 = int(round(cx - half))
    iy1 = int(round(cy - half))
    ix2 = ix1 + side
    iy2 = iy1 + side

    if pad_mode == "clamp":
        if side >= w:
            ix1, ix2 = 0, w
        else:
            if ix1 < 0:
                ix2 -= ix1
                ix1 = 0
            if ix2 > w:
                ix1 -= (ix2 - w)
                ix2 = w
            ix1 = max(0, ix1)
        if side >= h:
            iy1, iy2 = 0, h
        else:
            if iy1 < 0:
                iy2 -= iy1
                iy1 = 0
            if iy2 > h:
                iy1 -= (iy2 - h)
                iy2 = h
            iy1 = max(0, iy1)
        return img[iy1:iy2, ix1:ix2]

    pad_l = max(0, -ix1)
    pad_t = max(0, -iy1)
    pad_r = max(0, ix2 - w)
    pad_b = max(0, iy2 - h)
    if pad_l or pad_t or pad_r or pad_b:
        padded = cv2.copyMakeBorder(
            img, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REPLICATE
        )
        return padded[iy1 + pad_t : iy2 + pad_t, ix1 + pad_l : ix2 + pad_l]
    return img[iy1:iy2, ix1:ix2]


def preprocess_kdef_image(
    img_or_path: Union[str, Path, np.ndarray],
    use_aspect_crop: bool = True,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_blend: float = 0.8,
    img_size: int = 160,
    crop_mode: str = DEFAULT_CROP_MODE,
    ssd_margin: float = DEFAULT_SSD_MARGIN,
    crop_box: Tuple[float, float, float, float] = KDEF_STUDIO_CROP_BOX,
    ssd_box: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Standardized KDEF Image Preprocessing Pipeline (`ssd_pad_160_1ch` by default):
      1. Load BGR image (for SSD detection) and convert to grayscale.
      2. Crop facial region:
         - `ssd_square_pad` (default): OpenCV Res10 SSD face-centered square crop
           (`margin=1.25`, `cv2.BORDER_REPLICATE`), falling back to `KDEF_STUDIO_CROP_BOX`
           if no face is detected.
         - `ssd_square_clamp`: SSD square crop clamped inside image bounds.
         - `studio`: Fixed studio portrait crop `[0.04W:0.96W, 0.16H:0.86H]`.
      3. `cv2.resize` to `(img_size, img_size)` with `cv2.INTER_AREA`.
      4. CLAHE contrast enhancement (`clipLimit=2.0`, `tileGridSize=(8, 8)`) blended
         at `0.8 * enhanced + 0.2 * raw`.
    Returns uint8 array of shape `(img_size, img_size)` in `[0, 255]`.
    """
    if isinstance(img_or_path, (str, Path)):
        img_bgr = cv2.imread(str(img_or_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {img_or_path}")
    elif isinstance(img_or_path, np.ndarray):
        if img_or_path.ndim == 2:
            img_bgr = cv2.cvtColor(img_or_path, cv2.COLOR_GRAY2BGR)
        elif img_or_path.ndim == 3:
            img_bgr = img_or_path
        else:
            raise ValueError(f"Unexpected image array shape: {img_or_path.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(img_or_path)}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if use_aspect_crop and crop_mode != "none":
        if crop_mode == "ssd_square_pad":
            if ssd_box is None:
                ssd_box, _ = detect_face_ssd(img_bgr, min_conf=DEFAULT_SSD_MIN_CONF)
            gray = _crop_ssd_square(
                gray, ssd_box, margin=ssd_margin, pad_mode="pad", fallback_box=crop_box
            )
        elif crop_mode == "ssd_square_clamp":
            if ssd_box is None:
                ssd_box, _ = detect_face_ssd(img_bgr, min_conf=DEFAULT_SSD_MIN_CONF)
            gray = _crop_ssd_square(
                gray, ssd_box, margin=ssd_margin, pad_mode="clamp", fallback_box=crop_box
            )
        elif crop_mode == "studio":
            gray = _crop_studio(gray, crop_box)
        else:
            raise ValueError(f"Unsupported crop_mode: {crop_mode}")

    img = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        if clahe_blend < 1.0:
            img = cv2.addWeighted(enhanced, clahe_blend, img, 1.0 - clahe_blend, 0)
        else:
            img = enhanced

    return img.astype(np.uint8)


class FaceAligner:
    """Face detector & square-pad aligner matching the training `ssd_pad_160_1ch` pipeline."""

    def __init__(self, use_ssd: bool = True, ssd_margin: float = DEFAULT_SSD_MARGIN):
        self.use_ssd = use_ssd
        self.ssd_margin = ssd_margin
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(current_dir, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def align_and_crop(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (160, 160),
        margin: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Detects the primary face using OpenCV Res10 SSD (supporting frontal and `FL`/`FR`
        profile angles) with Haar Cascade fallback, applies symmetric square-pad cropping
        (`margin=1.25`, `cv2.BORDER_REPLICATE`), and resizes via `cv2.INTER_AREA`.
        """
        eff_margin = self.ssd_margin if margin is None else float(margin)
        if image.ndim == 2:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3:
            img_bgr = image
        else:
            return None

        if self.use_ssd:
            ssd_box, _ = detect_face_ssd(img_bgr, min_conf=DEFAULT_SSD_MIN_CONF)
            if ssd_box is not None:
                cropped = _crop_ssd_square(img_bgr, ssd_box, margin=eff_margin, pad_mode="pad")
                if cropped.size > 0:
                    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

        # Fallback to Haar Cascade if SSD misses
        gray = cv2.equalizeHist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None

        ih, iw = img_bgr.shape[:2]
        x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
        haar_box = (x / float(iw), y / float(ih), (x + w) / float(iw), (y + h) / float(ih))
        cropped_face = _crop_ssd_square(img_bgr, haar_box, margin=eff_margin, pad_mode="pad")
        if cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
            return None

        return cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)
