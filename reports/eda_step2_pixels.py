# -*- coding: utf-8 -*-
"""Bước 2: Kích thước ảnh, chế độ màu, thống kê pixel (mean/std) tập train."""
import os
import json
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "data", "official-kdef-dataset-by-ksatrajit0")
BALANCED = os.path.join(ROOT, "data", "balanced-kdef-dataset-by-dolly-prajapati-182")
FIGDIR = os.path.join(ROOT, "reports", "figures")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


def iter_images(split_dir):
    for cls in sorted(os.listdir(split_dir)):
        d = os.path.join(split_dir, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if os.path.splitext(f)[1].lower() in EXTS:
                yield cls, os.path.join(d, f)


def analyze(split_dir, pixel_sample_cap=None, seed=42):
    sizes, modes = Counter(), Counter()
    pixel_mean, pixel_std = [], []
    per_image_stats = []
    files = list(iter_images(split_dir))
    rng = np.random.default_rng(seed)
    idx_all = list(range(len(files)))
    pixel_idx = set(rng.choice(idx_all, size=min(pixel_sample_cap or len(files), len(files)),
                                replace=False).tolist()) if pixel_sample_cap else set(idx_all)
    for i, (cls, path) in enumerate(tqdm(files, desc=os.path.basename(os.path.dirname(split_dir)))):
        with Image.open(path) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
            if i in pixel_idx:
                arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
                pixel_mean.append(float(arr.mean()))
                pixel_std.append(float(arr.std()))
                per_image_stats.append({"class": cls, "mean": float(arr.mean()),
                                        "std": float(arr.std())})
    return {
        "n_files": len(files),
        "n_pixel_sample": len(pixel_mean),
        "sizes_top": sizes.most_common(10),
        "n_unique_sizes": len(sizes),
        "modes": dict(modes),
        "dataset_mean": float(np.mean(pixel_mean)),
        "dataset_std_of_image_means": float(np.std(pixel_mean)),
        "avg_image_std": float(np.mean(pixel_std)),
        "per_image": pd.DataFrame(per_image_stats),
    }


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    out = {}
    res_off = analyze(os.path.join(OFFICIAL, "train"), pixel_sample_cap=None)
    res_bal = analyze(os.path.join(BALANCED, "train"), pixel_sample_cap=4000)

    for name, r in [("official", res_off), ("balanced", res_bal)]:
        print(f"===== train/{name} =====")
        print("Số ảnh:", r["n_files"], "| mẫu pixel:", r["n_pixel_sample"])
        print("Kích thước duy nhất:", r["n_unique_sizes"], "| top:", r["sizes_top"][:5])
        print("Chế độ màu:", r["modes"])
        print(f"Mean( ảnh-gray/255 ) = {r['dataset_mean']:.4f}")
        print(f"Std các mean ảnh      = {r['dataset_std_of_image_means']:.4f}")
        print(f"Std trung bình 1 ảnh  = {r['avg_image_std']:.4f}")
        r["per_image"].to_csv(os.path.join(FIGDIR, f"pixel_stats_train_{name}.csv"), index=False)
        out[name] = {k: v for k, v in r.items() if k != "per_image"}

    with open(os.path.join(FIGDIR, "step2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Đã lưu JSON + CSV vào", FIGDIR)


if __name__ == "__main__":
    main()
