# -*- coding: utf-8 -*-
"""Unified locked-test comparison of the 3 models on the 3-source multicultural dataset (M4).

Models (all scored by ONE shared metric function on the SAME preprocessed test images):
  1. KDEF_Stem2Ch_MS4_Symmetric (main CNN)         -> checkpoints/*.pt  (src/models/model.py)
  2. HOG + SVM (LinearSVC, Platt probabilities)    -> checkpoints/hog_svm_multicultural.pkl
  3. ResNet-18 (IMAGENET1K_V1 transfer learning)   -> checkpoints/resnet18_multicultural.pt

Usage (test split is locked unless --unlock-test is passed explicitly):
  python scripts/so_sanh_mo_hinh.py --split val                    # dry run on val (safe)
  python scripts/so_sanh_mo_hinh.py --split test --unlock-test     # ONE-TIME final evaluation

Outputs (split=test):
  results/test_predictions_<model>.npz      y_true / y_pred / y_prob per model
  results/test_comparison.json              full metrics + params + MACs + size + latency
  reports/model_comparison.md               comparison tables (overall / source / group / gender / pose / emotion / cost)
  reports/figures/mc_*.png                  confusion matrices + comparison bar charts
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import CLASS_NAMES  # noqa: E402
from src.models.baselines import (  # noqa: E402
    HOGSVMClassifier,
    compute_multicultural_metrics,
    load_multicultural_split_cache,
    load_resnet18_checkpoint,
)
from src.models.model import load_kdef_checkpoint  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

GROUP_LABELS = {
    "KDEF": "KDEF – Bắc Âu (Thụy Điển)",
    "B": "RADIATE – gốc Phi",
    "A": "RADIATE – gốc Á",
    "H": "RADIATE – gốc Mỹ Latinh",
    "W": "RADIATE – gốc Âu",
    "FER": "FER-2013 – Internet",
}
SOURCE_LABELS = {"kdef": "KDEF", "radiate": "RADIATE", "fer2013": "FER-2013"}
ANGLE_LABELS = {"FL": "FL (nghiêng hẳn trái)", "HL": "HL (nửa trái)", "S": "S (chính diện)",
                "HR": "HR (nửa phải)", "FR": "FR (nghiêng hẳn phải)"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class NormalizedCNN(nn.Module):
    """Wraps the CNN so it takes [0,1] 1-ch input and applies the checkpoint's train-split mean/std."""

    def __init__(self, net: nn.Module, mean: float, std: float):
        super().__init__()
        self.net = net
        self.mean = float(mean)
        self.std = float(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net((x - self.mean) / self.std)
        return out[0] if isinstance(out, tuple) else out


def count_macs(model: nn.Module, img_size: int, device: torch.device) -> int:
    """Counts multiply-accumulates of Conv2d + Linear layers for one 1x1xHxW image."""
    total = [0]

    def conv_hook(m: nn.Conv2d, _inp, out):
        kh, kw = m.kernel_size
        total[0] += int(out.numel() // out.shape[0]) * (m.in_channels // m.groups) * kh * kw

    def lin_hook(m: nn.Linear, _inp, out):
        total[0] += int(m.in_features * m.out_features)

    hooks = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(torch.rand(1, 1, img_size, img_size, device=device))
    for h in hooks:
        h.remove()
    return total[0]


@torch.no_grad()
def predict_torch(model: nn.Module, images_u8: np.ndarray, device: torch.device, batch_size: int = 256,
                  use_flip_tta: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """fp32 inference on uint8 (N,H,W) images -> (y_pred, y_prob)."""
    model.eval().to(device)
    probs: List[np.ndarray] = []
    for s in range(0, len(images_u8), batch_size):
        xb = torch.from_numpy(images_u8[s:s + batch_size]).to(device=device, dtype=torch.float32).unsqueeze(1) / 255.0
        logits = model(xb)
        if use_flip_tta:
            logits = 0.5 * (logits + model(torch.flip(xb, dims=[3])))
        probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    y_prob = np.concatenate(probs, axis=0)
    return y_prob.argmax(axis=1), y_prob


def latency_single_image(predict_one: Callable[[np.ndarray], Any], images_u8: np.ndarray,
                         n_warmup: int = 20, n_runs: int = 300, sync: Optional[Callable] = None) -> Dict[str, float]:
    """Batch-1 latency including uint8 -> tensor -> normalize -> forward -> softmax (no face detection)."""
    n = len(images_u8)
    for i in range(n_warmup):
        predict_one(images_u8[i % n])
    if sync:
        sync()
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        predict_one(images_u8[i % n])
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    mean_ms = float(arr.mean())
    return {"ms_per_image": round(mean_ms, 3), "ms_median": round(float(np.median(arr)), 3),
            "ms_p95": round(float(np.percentile(arr, 95)), 3), "fps": round(1000.0 / max(mean_ms, 1e-9), 1),
            "n_runs": n_runs}


def torch_predict_one_fn(model: nn.Module, device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    model.eval().to(device)

    @torch.no_grad()
    def _fn(img_u8: np.ndarray) -> np.ndarray:
        xb = torch.from_numpy(img_u8).to(device=device, dtype=torch.float32).view(1, 1, *img_u8.shape) / 255.0
        return F.softmax(model(xb), dim=1).cpu().numpy()

    return _fn


def file_mb(p: Path) -> float:
    return round(p.stat().st_size / (1024 * 1024), 3)


# ---------------------------------------------------------------------------
# Model loaders -> uniform record
# ---------------------------------------------------------------------------
def evaluate_cnn(ckpt: Path, split: str, device: torch.device, flip_tta: bool) -> Dict[str, Any]:
    net, meta = load_kdef_checkpoint(ckpt, device=device, num_classes=len(CLASS_NAMES))
    model = NormalizedCNN(net, meta["mean"], meta["std"]).to(device).eval()
    X, y, df = load_multicultural_split_cache(split, img_size=meta["img_size"], crop_mode=meta["crop_mode"],
                                              ssd_margin=meta["ssd_margin"], use_clahe=meta["use_clahe"])
    y_pred, y_prob = predict_torch(model, X, device, use_flip_tta=flip_tta)
    n_params = int(sum(p.numel() for p in net.parameters()))
    macs = count_macs(model, meta["img_size"], device)
    cpu_model = NormalizedCNN(load_kdef_checkpoint(ckpt, device=torch.device("cpu"))[0], meta["mean"], meta["std"])
    lat_cpu = latency_single_image(torch_predict_one_fn(cpu_model, torch.device("cpu")), X[:200])
    lat_gpu = None
    if device.type == "cuda":
        lat_gpu = latency_single_image(torch_predict_one_fn(model, device), X[:200], sync=torch.cuda.synchronize)
    return {"key": "cnn", "name": "KDEF_Stem2Ch_MS4_Symmetric (CNN chính)", "checkpoint": ckpt,
            "y_true": y, "y_pred": y_pred, "y_prob": y_prob, "df": df, "n_parameters": n_params, "macs": macs,
            "latency_cpu": lat_cpu, "latency_gpu": lat_gpu, "img_size": meta["img_size"],
            "input": f"xám {meta['img_size']}x{meta['img_size']}, CLAHE={meta['use_clahe']}, "
                     f"mean={meta['mean']:.6f}, std={meta['std']:.6f}",
            "val_macro_f1_ckpt": meta.get("val_macro_f1"), "eval_mode": "flip_tta" if flip_tta else "single_view"}


def evaluate_hog(ckpt: Path, split: str) -> Dict[str, Any]:
    clf, extra = HOGSVMClassifier.load(ckpt)
    X, y, df = load_multicultural_split_cache(split, img_size=160)  # same 160 cache used during tuning
    feats = clf.extract_hog_features(X)
    y_prob = clf.predict_proba_from_features(feats)
    y_pred = clf.predict_from_features(feats)  # argmax of SVM margins (identical to training-time rule)
    Xh = X[:200]
    lat_cpu = latency_single_image(lambda im: clf.predict_proba(im[None]), Xh)
    return {"key": "hog_svm", "name": "HOG + SVM (LinearSVC)", "checkpoint": ckpt,
            "y_true": y, "y_pred": y_pred, "y_prob": y_prob, "df": df, "n_parameters": clf.count_parameters(),
            "macs": int(clf.count_parameters() - len(CLASS_NAMES)), "latency_cpu": lat_cpu, "latency_gpu": None,
            "img_size": clf.hog_img_size,
            "input": f"HOG {clf.hog_img_size}x{clf.hog_img_size}, ô {clf.cell_size}, {clf.nbins} hướng -> "
                     f"{clf.feature_dim} chiều, C={clf.C}, class_weight={clf.class_weight}",
            "val_macro_f1_ckpt": (extra.get("val_metrics") or {}).get("macro_f1"), "eval_mode": "single_view"}


def evaluate_resnet(ckpt: Path, split: str, device: torch.device, flip_tta: bool) -> Dict[str, Any]:
    model, meta = load_resnet18_checkpoint(ckpt, device=device, input_is_normalized_1ch=False)
    X, y, df = load_multicultural_split_cache(split, img_size=meta["img_size"], crop_mode=meta["crop_mode"],
                                              ssd_margin=meta["ssd_margin"], use_clahe=meta["use_clahe"])
    y_pred, y_prob = predict_torch(model, X, device, use_flip_tta=flip_tta)
    n_params = int(sum(p.numel() for p in model.parameters()))
    macs = count_macs(model, meta["img_size"], device)
    cpu_model = load_resnet18_checkpoint(ckpt, device=torch.device("cpu"))[0]
    lat_cpu = latency_single_image(torch_predict_one_fn(cpu_model, torch.device("cpu")), X[:200])
    lat_gpu = None
    if device.type == "cuda":
        lat_gpu = latency_single_image(torch_predict_one_fn(model, device), X[:200], sync=torch.cuda.synchronize)
    return {"key": "resnet18", "name": "ResNet-18 (IMAGENET1K_V1)", "checkpoint": ckpt,
            "y_true": y, "y_pred": y_pred, "y_prob": y_prob, "df": df, "n_parameters": n_params, "macs": macs,
            "latency_cpu": lat_cpu, "latency_gpu": lat_gpu, "img_size": meta["img_size"],
            "input": f"xám {meta['img_size']}x{meta['img_size']} lặp 3 kênh, chuẩn hóa ImageNet",
            "val_macro_f1_ckpt": meta.get("val_macro_f1"), "eval_mode": "flip_tta" if flip_tta else "single_view"}


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def pct(x: Optional[float], d: int = 2) -> str:
    return "–" if x is None else f"{x * 100:.{d}f}%"


def render_markdown(records: List[Dict[str, Any]], split: str, sanity: Dict[str, Any]) -> str:
    L: List[str] = []
    n = records[0]["metrics"]["n_samples"]
    L.append(f"# So sánh 3 mô hình trên tập `{split}` khóa của bộ dữ liệu đa văn hóa 3 nguồn\n")
    L.append(f"Sinh tự động bởi `python scripts/so_sanh_mo_hinh.py --split {split}"
             f"{' --unlock-test' if split == 'test' else ''}` — cùng {n:,} ảnh, cùng tiền xử lý "
             "`ssd_pad_1ch` (SSD square-pad 1,25 + CLAHE), cùng hàm tính độ đo "
             "`compute_multicultural_metrics`. Đoán ngẫu nhiên = 14,29%.\n")
    L.append(f"Kiểm tra toàn vẹn: y_true của 3 mô hình trùng khớp = **{sanity['labels_identical']}**, "
             f"thứ tự ảnh theo manifest = **{sanity['order_identical']}**.\n")

    L.append("## 1. Kết quả chung\n")
    L.append("| Mô hình | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC (OvR macro) | Macro-Precision | Macro-Recall | Recall spread | Ảnh đúng |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in records:
        m = r["metrics"]
        mp = float(np.mean(list(m["per_class_precision"].values())))
        mr = float(np.mean(list(m["per_class_recall"].values())))
        L.append(f"| {r['name']} | {pct(m['accuracy'])} | {pct(m['macro_f1'])} | {pct(m['weighted_f1'])} | "
                 f"{m['roc_auc_ovr_macro']:.4f} | {pct(mp)} | {pct(mr)} | {pct(m['recall_spread'])} | "
                 f"{m['n_correct']:,}/{m['n_samples']:,} |")

    L.append("\n## 2. Theo từng nguồn dữ liệu\n")
    L.append("| Mô hình | Nguồn | Số ảnh | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for r in records:
        for src, lab in SOURCE_LABELS.items():
            s = r["metrics"].get("per_source", {}).get(src)
            if s:
                L.append(f"| {r['name']} | {lab} | {s['n_samples']:,} | {pct(s['accuracy'])} | {pct(s['macro_f1'])} | "
                         f"{pct(s['weighted_f1'])} | {s['roc_auc_ovr_macro']:.4f} |")

    L.append("\n## 3. Theo nhóm người (nguồn gốc)\n")
    hdr = "| Nhóm | Số ảnh | " + " | ".join(f"{r['name']} Acc / F1" for r in records) + " |"
    L.append(hdr)
    L.append("|---|---:|" + "---:|" * len(records))
    for g, lab in GROUP_LABELS.items():
        first = records[0]["metrics"].get("per_ethnic_group", {}).get(g)
        if not first:
            continue
        cells = []
        for r in records:
            e = r["metrics"]["per_ethnic_group"][g]
            cells.append(f"{pct(e['accuracy'], 1)} / {e['macro_f1']:.3f}")
        L.append(f"| {lab} | {first['n_samples']:,} | " + " | ".join(cells) + " |")

    L.append("\n## 4. Theo giới tính (KDEF + RADIATE, có định danh)\n")
    L.append("| Giới tính | Số ảnh | " + " | ".join(f"{r['name']} Acc / F1" for r in records) + " |")
    L.append("|---|---:|" + "---:|" * len(records))
    for g, lab in (("female", "Nữ"), ("male", "Nam")):
        first = records[0]["metrics"].get("per_gender", {}).get(g)
        if not first:
            continue
        cells = [f"{pct(r['metrics']['per_gender'][g]['accuracy'], 1)} / {r['metrics']['per_gender'][g]['macro_f1']:.3f}"
                 for r in records]
        L.append(f"| {lab} | {first['n_samples']:,} | " + " | ".join(cells) + " |")

    L.append("\n## 5. Theo góc chụp (phần KDEF, 5 góc)\n")
    L.append("| Góc | " + " | ".join(r["name"] for r in records) + " |")
    L.append("|---|" + "---:|" * len(records))
    for a, lab in ANGLE_LABELS.items():
        if a not in records[0]["metrics"].get("angle_accuracy", {}):
            continue
        L.append(f"| {lab} | " + " | ".join(pct(r["metrics"]["angle_accuracy"].get(a), 1) for r in records) + " |")
    L.append("| \\|FL − FR\\| | " + " | ".join(
        pct(abs(r["metrics"]["angle_accuracy"].get("FL", 0) - r["metrics"]["angle_accuracy"].get("FR", 0)), 1)
        for r in records) + " |")

    L.append("\n## 6. Theo từng cảm xúc\n")
    L.append("### 6.1. F1 từng cảm xúc theo mô hình\n")
    L.append("| Cảm xúc | Số ảnh | " + " | ".join(r["name"] for r in records) + " |")
    L.append("|---|---:|" + "---:|" * len(records))
    cm0 = np.asarray(records[0]["metrics"]["confusion_matrix"])
    for i, c in enumerate(CLASS_NAMES):
        L.append(f"| {c} | {int(cm0[i].sum()):,} | " + " | ".join(f"{r['metrics']['per_class_f1'][c]:.4f}" for r in records) + " |")
    L.append("| **Macro-F1** | | " + " | ".join(f"**{r['metrics']['macro_f1']:.4f}**" for r in records) + " |")
    for r in records:
        m = r["metrics"]
        L.append(f"\n### 6.{records.index(r) + 2}. {r['name']}: Precision / Recall / F1 / AUC từng cảm xúc\n")
        L.append("| Cảm xúc | Precision | Recall | F1 | AUC | Recall KDEF | Recall RADIATE | Recall FER-2013 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        ps = m.get("per_source", {})
        for c in CLASS_NAMES:
            L.append(f"| {c} | {m['per_class_precision'][c]:.4f} | {m['per_class_recall'][c]:.4f} | "
                     f"{m['per_class_f1'][c]:.4f} | {m['per_class_auc'][c]:.4f} | "
                     + " | ".join(f"{ps.get(s, {}).get('per_class_recall', {}).get(c, 0):.4f}" for s in SOURCE_LABELS) + " |")

    L.append("\n## 7. Quy mô và tốc độ dự đoán\n")
    L.append("Độ trễ đo theo lô 1 ảnh, tính cả bước chuyển uint8 → tensor → chuẩn hóa → suy luận → softmax "
             "(không tính dò mặt), 300 lần sau 20 lần khởi động, fp32.\n")
    L.append("| Mô hình | Số tham số | Phép nhân-cộng / ảnh | Tệp mô hình | CPU ms/ảnh (FPS) | GPU ms/ảnh (FPS) | Ảnh vào |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for r in records:
        c, g = r["latency_cpu"], r["latency_gpu"]
        gtxt = f"{g['ms_per_image']:.2f} ({g['fps']:.0f})" if g else "– (chỉ CPU)"
        L.append(f"| {r['name']} | {r['n_parameters']:,} | {r['macs'] / 1e9:.3f} tỉ | {r['checkpoint_mb']:.2f} MB | "
                 f"{c['ms_per_image']:.2f} ({c['fps']:.0f}) | {gtxt} | {r['input']} |")
    L.append("")
    return "\n".join(L)


def save_figures(records: List[Dict[str, Any]], split: str) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    fig, axes = plt.subplots(1, len(records), figsize=(6 * len(records), 5.2))
    for ax, r in zip(np.atleast_1d(axes), records):
        cm = np.asarray(r["metrics"]["confusion_matrix"], dtype=float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, f"{int(cm[i, j])}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(7), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(7), CLASS_NAMES, fontsize=8)
        ax.set_title(f"{r['name']}\nAcc {r['metrics']['accuracy'] * 100:.2f}% | F1 {r['metrics']['macro_f1']:.4f}", fontsize=9)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thật")
    fig.tight_layout()
    p = FIG_DIR / f"mc_{split}_confusion_3models.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(p.relative_to(REPO_ROOT).as_posix())

    cats = ["Tổng"] + list(SOURCE_LABELS.values())
    x = np.arange(len(cats))
    w = 0.8 / len(records)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, r in enumerate(records):
        m = r["metrics"]
        vals = [m["accuracy"]] + [m.get("per_source", {}).get(s, {}).get("accuracy", 0) for s in SOURCE_LABELS]
        bars = ax.bar(x + (k - (len(records) - 1) / 2) * w, np.asarray(vals) * 100, w, label=r["name"])
        ax.bar_label(bars, fmt="%.1f", fontsize=7)
    ax.set_xticks(x, cats)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.set_title(f"Accuracy theo nguồn – tập {split}")
    fig.tight_layout()
    p = FIG_DIR / f"mc_{split}_accuracy_by_source_3models.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(p.relative_to(REPO_ROOT).as_posix())
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Unified 3-model comparison on the multicultural dataset.")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--unlock-test", action="store_true", help="Required to score the locked test split.")
    ap.add_argument("--cnn", default="checkpoints/cnn_multicultural.pt")
    ap.add_argument("--hog", default="checkpoints/hog_svm_multicultural.pkl")
    ap.add_argument("--resnet", default="checkpoints/resnet18_multicultural.pt")
    ap.add_argument("--flip-tta", action="store_true", help="Also use flip-TTA for the 2 CNNs (default single-view).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.split == "test" and not args.unlock_test:
        print("REFUSED: test split is locked. Re-run with --unlock-test for the one-time final evaluation.")
        return 2

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    records: List[Dict[str, Any]] = []
    for key, path in (("cnn", args.cnn), ("hog_svm", args.hog), ("resnet18", args.resnet)):
        p = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
        if not p.exists():
            print(f"[skip] {key}: checkpoint not found -> {p}")
            continue
        t0 = time.time()
        if key == "cnn":
            rec = evaluate_cnn(p, args.split, device, args.flip_tta)
        elif key == "hog_svm":
            rec = evaluate_hog(p, args.split)
        else:
            rec = evaluate_resnet(p, args.split, device, args.flip_tta)
        rec["checkpoint_mb"] = file_mb(p)
        rec["metrics"] = compute_multicultural_metrics(rec["y_true"], rec["y_pred"], rec["y_prob"], meta_df=rec["df"])
        rec["eval_sec"] = round(time.time() - t0, 1)
        m = rec["metrics"]
        print(f"[{key}] n={m['n_samples']} Acc={m['accuracy']:.4f} MacroF1={m['macro_f1']:.4f} "
              f"WF1={m['weighted_f1']:.4f} AUC={m['roc_auc_ovr_macro']:.4f} | params={rec['n_parameters']:,} "
              f"CPU={rec['latency_cpu']['ms_per_image']:.2f}ms ({rec['eval_sec']}s)")
        records.append(rec)

    if not records:
        print("No checkpoints found.")
        return 1

    ref = records[0]
    sanity = {
        "labels_identical": all(np.array_equal(r["y_true"], ref["y_true"]) for r in records),
        "order_identical": all(r["df"]["filepath"].tolist() == ref["df"]["filepath"].tolist() for r in records),
    }
    if not (sanity["labels_identical"] and sanity["order_identical"]):
        print(f"FATAL: models were not scored on identical images: {sanity}")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for r in records:
        np.savez_compressed(RESULTS_DIR / f"{args.split}_predictions_{r['key']}.npz",
                            y_true=r["y_true"], y_pred=r["y_pred"], y_prob=r["y_prob"].astype(np.float32),
                            filepath=np.asarray(r["df"]["filepath"].tolist()))

    figures = save_figures(records, args.split)
    payload = {
        "split": args.split,
        "n_samples": int(ref["metrics"]["n_samples"]),
        "device": f"{device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})",
        "torch": torch.__version__,
        "sanity": sanity,
        "figures": figures,
        "models": {
            r["key"]: {
                "name": r["name"], "checkpoint": r["checkpoint"].relative_to(REPO_ROOT).as_posix(),
                "checkpoint_mb": r["checkpoint_mb"], "n_parameters": r["n_parameters"], "macs": r["macs"],
                "input": r["input"], "eval_mode": r["eval_mode"], "val_macro_f1_ckpt": r["val_macro_f1_ckpt"],
                "latency_cpu": r["latency_cpu"], "latency_gpu": r["latency_gpu"], "metrics": r["metrics"],
            } for r in records
        },
    }
    out_json = RESULTS_DIR / f"{args.split}_comparison.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown(records, args.split, sanity)
    out_md = REPORTS_DIR / ("model_comparison.md" if args.split == "test" else f"model_comparison_{args.split}.md")
    out_md.write_text(md, encoding="utf-8")
    print(f"Saved -> {out_json.relative_to(REPO_ROOT).as_posix()}, {out_md.relative_to(REPO_ROOT).as_posix()}, {figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
