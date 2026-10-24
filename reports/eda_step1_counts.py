# -*- coding: utf-8 -*-
"""Bước 1: Thống kê số ảnh theo lớp x split cho 2 bộ dữ liệu KDEF."""
import os
import json
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "data", "official-kdef-dataset-by-ksatrajit0")
BALANCED = os.path.join(ROOT, "data", "balanced-kdef-dataset-by-dolly-prajapati-182")
FIGDIR = os.path.join(ROOT, "reports", "figures")
SPLITS = ["train", "val", "test"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


def count_by_class(root):
    """Đếm số file ảnh theo split x lớp."""
    rows = []
    for split in SPLITS:
        for cls in sorted(os.listdir(os.path.join(root, split))):
            d = os.path.join(root, split, cls)
            if not os.path.isdir(d):
                continue
            n = sum(1 for f in os.listdir(d) if os.path.splitext(f)[1].lower() in EXTS)
            rows.append({"split": split, "class": cls, "count": n})
    return pd.DataFrame(rows)


def plot_distribution(df, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    order = sorted(df["class"].unique())
    sns.countplot(data=df, x="class", order=order, ax=axes[0], palette="viridis",
                  hue="class", legend=False)
    axes[0].set_title(f"{title} — tổng số ảnh theo lớp")
    axes[0].set_ylabel("Số ảnh (ghi log)")
    axes[0].set_yscale("log")
    for c in axes[0].containers:
        axes[0].bar_label(c, fontsize=8)
    pivot = df.pivot(index="class", columns="split", values="count").reindex(order)[SPLITS]
    pivot.plot(kind="bar", ax=axes[1], rot=20)
    axes[1].set_title(f"{title} — số ảnh theo lớp x split")
    axes[1].set_ylabel("Số ảnh")
    axes[1].set_yscale("log")
    for c in axes[1].containers:
        axes[1].bar_label(c, fontsize=7, fmt="%d")
    axes[1].legend(title="split")
    axes[1].set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    summary = {}
    for name, root in [("official", OFFICIAL), ("balanced", BALANCED)]:
        df = count_by_class(root)
        df.to_csv(os.path.join(FIGDIR, f"class_split_counts_{name}.csv"), index=False)
        pivot = df.pivot(index="class", columns="split", values="count").fillna(0).astype(int)
        pivot = pivot[SPLITS]
        pivot["total"] = pivot.sum(axis=1)
        print(f"===== {name} =====")
        print(pivot.to_string())
        print("Tổng mỗi split:", pivot[SPLITS].sum().to_dict())
        print("Tổng toàn bộ:", int(pivot["total"].sum()))
        summary[name] = {
            "per_class_split": pivot.to_dict(),
            "split_totals": pivot[SPLITS].sum().to_dict(),
            "total": int(pivot["total"].sum()),
        }
        plot_distribution(df, f"Bộ {name}", f"class_distribution_{name}.png")

    with open(os.path.join(FIGDIR, "step1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print("\nĐã lưu biểu đồ + CSV vào", FIGDIR)


if __name__ == "__main__":
    main()
