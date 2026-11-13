"""
Evaluation & Diagnostic Pipeline for KDEF Facial Emotion Recognition.
Core Integrator Implementation.

Features:
- Single-view and Test-Time Augmentation (Flip-TTA: original + horizontal flip)
- Comprehensive metric calculations: Accuracy, Macro-F1, Weighted-F1, Recall Spread
- Per-class Recall, Precision, and 7x7 Confusion Matrix
- Breakdown by pose angle (FL, FR, HL, HR, S) and pose asymmetry gap analysis
- Activation statistics extraction (Stage 4 pre-pool dead neuron fraction, mean, std)
- Schema strictly compatible with historical reports/test_metrics.json + audit extensions
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing.alignment import preprocess_kdef_image
from src.data.dataset import (
    ANGLE_NAMES,
    CLASS_NAMES,
    DEFAULT_KDEF_MEAN,
    DEFAULT_KDEF_STD,
    KDEFDataset,
    get_transforms
)
from src.models.model import KDEF_Stem2Ch_ECA_Strip, KDEFResSENet, build_model, load_kdef_checkpoint


def measure_activation_statistics(
    model: nn.Module,
    sample_path: Union[str, Path],
    device: torch.device,
    use_crop: bool = True,
    use_clahe: bool = True,
    img_size: int = 128,
    mean: float = DEFAULT_KDEF_MEAN,
    std: float = DEFAULT_KDEF_STD
) -> Dict[str, float]:
    """
    Measures activation stats across stages (especially Stage 4 pre-pool) on a specific probe sample (e.g., AF01HAHR.JPG).
    Supports both KDEF_Stem2Ch_ECA_Strip (probe_activations) and baseline KDEFNet (extract_stage4_pre_pool).
    """
    model.eval()
    sample_path = Path(sample_path)
    if not sample_path.exists():
        return {"dead_frac": 0.0, "mean": 0.0, "std": 0.0, "max": 0.0}

    proc_arr = preprocess_kdef_image(
        sample_path,
        use_aspect_crop=use_crop,
        use_clahe=use_clahe,
        clahe_clip=2.0,
        clahe_blend=0.8,
        img_size=img_size
    )
    tensor = torch.from_numpy(proc_arr).unsqueeze(0).unsqueeze(0).float() / 255.0
    tensor = ((tensor - mean) / std).to(device)

    with torch.no_grad():
        if hasattr(model, "probe_activations"):
            probe = model.probe_activations(tensor)
            s4 = probe.get("stage4", {})
            return {
                "dead_frac": round(s4.get("exact_dead_frac", 0.0), 4),
                "neg_frac": round(s4.get("neg_frac", 0.0), 4),
                "mean": round(s4.get("mean", 0.0), 4),
                "std": round(s4.get("std", 0.0), 4),
                "stages": probe
            }
        elif hasattr(model, "extract_stage4_pre_pool"):
            act = model.extract_stage4_pre_pool(tensor)
            arr = act.cpu().numpy().ravel()
            dead_frac = float((np.abs(arr) < 1e-6).mean())
            neg_frac = float((arr <= 0.0).mean())
            return {
                "dead_frac": round(dead_frac, 4),
                "neg_frac": round(neg_frac, 4),
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std()), 4),
                "max": round(float(arr.max()), 4)
            }
        else:
            return {"dead_frac": 0.0, "mean": 0.0, "std": 0.0, "max": 0.0}


def evaluate_split(
    model: nn.Module,
    data_dir: Optional[Union[str, Path]] = None,
    split: str = "test",
    use_tta: bool = True,
    tta_mode: str = "flip",
    use_crop: bool = True,
    use_clahe: bool = True,
    img_size: int = 128,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    probe_sample_path: Optional[Union[str, Path]] = None,
    mean: float = DEFAULT_KDEF_MEAN,
    std: float = DEFAULT_KDEF_STD
) -> Dict:
    """
    Evaluates model on specified split (default: 'test').
    Supports Single-View, Flip-TTA, and MS4-TTA (multi-scale 4-view TTA).
    Returns comprehensive metrics dictionary matching reports/test_metrics.json schema.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    eval_tf = get_transforms(split=split, mean=mean, std=std)
    dataset = KDEFDataset(
        data_dir=data_dir,
        split=split,
        use_crop=use_crop,
        use_clahe=use_clahe,
        img_size=img_size,
        in_channels=1,
        transform=eval_tf,
        exclude_corrupt=True,
        return_metadata=True
    )

    effective_workers = 2 if os.name == "posix" else 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers
    )

    all_preds = []
    all_targets = []
    all_angles = []
    all_subjects = []

    # Normalize tta_mode
    effective_tta = tta_mode.lower() if use_tta else "none"

    with torch.no_grad():
        for xb, yb, meta in loader:
            xb = xb.to(device)
            if effective_tta in ["ms4", "multiscale4"]:
                xb_flip = torch.flip(xb, dims=[3])
                margin = 4
                xb_crop = F.interpolate(
                    xb[:, :, margin : img_size - margin, margin : img_size - margin],
                    size=(img_size, img_size),
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
            elif effective_tta in ["flip", "flip2"]:
                xb_flipped = torch.flip(xb, dims=[3])
                logits = 0.5 * (model(xb) + model(xb_flipped))
            else:
                logits = model(xb)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yb.numpy())

            # Parse metadata
            if isinstance(meta, dict) and "angle" in meta:
                all_angles.extend(meta["angle"])
                all_subjects.extend(meta["subject_id"])
            elif isinstance(meta, list):
                for m in meta:
                    all_angles.append(m.get("angle", "unknown"))
                    all_subjects.append(m.get("subject_id", "unknown"))

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # 1. Core Global Metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    # 2. Per-Class Recall & Precision
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    per_class_recall = {}
    per_class_precision = {}
    per_class_f1 = {}

    for i, c in enumerate(CLASS_NAMES):
        tot = cm[i].sum()
        r = float(cm[i, i] / tot) if tot > 0 else 0.0
        pred_tot = cm[:, i].sum()
        p = float(cm[i, i] / pred_tot) if pred_tot > 0 else 0.0
        denom = (p + r)
        f = (2 * p * r / denom) if denom > 0 else 0.0
        per_class_recall[c] = float(r)
        per_class_precision[c] = float(p)
        per_class_f1[c] = float(f)

    recalls = list(per_class_recall.values())
    recall_spread = float(max(recalls) - min(recalls))

    # 3. Pose Angle Accuracy Breakdown
    angle_accuracy = {}
    if len(all_angles) == len(y_true):
        for ang in ANGLE_NAMES:
            mask = np.array([a == ang for a in all_angles])
            if mask.sum() > 0:
                ang_acc = float((y_true[mask] == y_pred[mask]).mean())
                angle_accuracy[ang] = round(ang_acc, 4)

    # Calculate HL/HR average vs FR gap
    hl_hr_acc = (angle_accuracy.get("HL", 0.0) + angle_accuracy.get("HR", 0.0)) / 2.0
    fr_acc = angle_accuracy.get("FR", 0.0)
    fl_acc = angle_accuracy.get("FL", 0.0)
    pose_asymmetry_gap = round(abs(hl_hr_acc - fr_acc), 4)

    # 4. Activation Stats on Diagnostic Sample
    if probe_sample_path is None:
        # Default probe image: AF01HAHR.JPG from train split
        default_probe = dataset.data_dir / "train" / "happy" / "AF01HAHR.JPG"
        if default_probe.exists():
            probe_sample_path = default_probe

    act_stats = None
    if probe_sample_path is not None and Path(probe_sample_path).exists():
        act_stats = measure_activation_statistics(
            model=model,
            sample_path=probe_sample_path,
            device=device,
            use_crop=use_crop,
            use_clahe=use_clahe,
            img_size=img_size,
            mean=mean,
            std=std
        )

    # Output schema matching reports/test_metrics.json + audit extensions
    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "recall_spread": recall_spread,
        "confusion_matrix": cm.tolist(),
        "angle_accuracy": angle_accuracy,
        "pose_asymmetry_metrics": {
            "hl_hr_average": round(hl_hr_acc, 4),
            "fl_accuracy": round(fl_acc, 4),
            "fr_accuracy": round(fr_acc, 4),
            "pose_asymmetry_gap": pose_asymmetry_gap
        },
        "activation_statistics": act_stats,
        "evaluation_config": {
            "split": split,
            "use_tta": use_tta,
            "use_crop": use_crop,
            "use_clahe": use_clahe,
            "img_size": img_size,
            "device": f"{device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})"
        }
    }
    return results


def evaluate_checkpoint(
    checkpoint_path: Union[str, Path],
    data_dir: Optional[Union[str, Path]] = None,
    split: str = "test",
    use_tta: bool = True,
    tta_mode: str = "flip",
    device: Optional[torch.device] = None
) -> Dict:
    """Loads a checkpoint (optimized core or notebook) and evaluates it on the specified split."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, meta = load_kdef_checkpoint(checkpoint_path, device=device, num_classes=len(CLASS_NAMES))

    return evaluate_split(
        model=model,
        data_dir=data_dir,
        split=split,
        use_tta=use_tta,
        tta_mode=tta_mode,
        use_crop=meta["use_crop"],
        use_clahe=meta["use_clahe"],
        img_size=meta["img_size"],
        device=device,
        mean=meta["mean"],
        std=meta["std"]
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate KDEF Model Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/optimized_best_model.pt")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--tta_mode", type=str, default="flip", choices=["none", "flip", "ms4", "multiscale4"])
    args = parser.parse_args()

    res = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        use_tta=not args.no_tta,
        tta_mode=args.tta_mode
    )
    print(json.dumps(res, indent=2))
