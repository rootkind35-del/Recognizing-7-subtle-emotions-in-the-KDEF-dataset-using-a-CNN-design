# -*- coding: utf-8 -*-
"""Bước 3+4: Grid ảnh mẫu mỗi lớp (5 ảnh/lớp) cho bộ official — train và test;
kèm phân bố gender/angle của tập test theo manifest."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "data", "official-kdef-dataset-by-ksatrajit0")
FIGDIR = os.path.join(ROOT, "reports", "figures")
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
VI = {"angry": "Giận", "disgust": "Chán ghét", "fear": "Sợ", "happy": "Vui",
      "neutral": "Trung tính", "sad": "Buồn", "surprise": "Ngạc nhiên"}


def sample_grid(split, fname, n_per_class=5, seed=42):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASSES), n_per_class, figsize=(2.1 * n_per_class, 2.5 * len(CLASSES)))
    picked_log = []
    for r, cls in enumerate(CLASSES):
        d = os.path.join(OFFICIAL, split, cls)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        idx = rng.choice(len(files), size=min(n_per_class, len(files)), replace=False)
        for c, i in enumerate(idx):
            f = files[int(i)]
            picked_log.append(f"{split}/{cls}/{f}")
            with Image.open(os.path.join(d, f)) as im:
                ax = axes[r, c]
                ax.imshow(np.asarray(im.convert("RGB")))
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"{VI[cls]}\n({cls})", fontsize=10)
                ax.set_xlabel(f[:16], fontsize=6)
    fig.suptitle(f"Bộ official — ảnh mẫu tập {split} (5 ảnh/lớp, seed=42)", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(os.path.join(FIGDIR, fname), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return picked_log


def test_manifest_analysis():
    man = pd.read_csv(os.path.join(OFFICIAL, "kdef_manifest_splits.csv"))
    test = man[man["split"] == "test"].copy()
    print("Số dòng manifest split=test:", len(test))
    print("\n-- expression --");  print(test["expression"].value_counts().to_string())
    print("\n-- gender --");     print(test["gender"].value_counts().to_string())
    print("\n-- angle --");      print(test["angle"].value_counts().to_string())
    print("\n-- session --");    print(test["session"].value_counts().to_string())
    print("\n-- số subject duy nhất --", test["subject"].nunique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    order_e = CLASSES
    sns.countplot(data=test, x="expression", order=order_e, ax=axes[0], hue="expression",
                  palette="viridis", legend=False)
    axes[0].set_title("Test (official): số ảnh theo expression")
    axes[0].tick_params(axis="x", rotation=20)
    for c in axes[0].containers: axes[0].bar_label(c, fontsize=8)

    sns.countplot(data=test, x="gender", ax=axes[1], hue="gender", palette="Set2", legend=False)
    axes[1].set_title("Test (official): phân bố gender")
    for c in axes[1].containers: axes[1].bar_label(c, fontsize=9)

    angle_order = sorted(test["angle"].unique())
    sns.countplot(data=test, x="angle", order=angle_order, ax=axes[2], hue="angle",
                  palette="crest", legend=False)
    axes[2].set_title("Test (official): phân bố góc chụp (angle)")
    for c in axes[2].containers: axes[2].bar_label(c, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "test_manifest_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # gender x expression
    g = test.groupby(["expression", "gender"]).size().unstack(fill_value=0)
    g.to_csv(os.path.join(FIGDIR, "test_gender_by_expression.csv"))
    a = test.groupby(["expression", "angle"]).size().unstack(fill_value=0)
    a.to_csv(os.path.join(FIGDIR, "test_angle_by_expression.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.heatmap(g.reindex(CLASSES), annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Test: gender x expression")
    sns.heatmap(a.reindex(CLASSES), annot=True, fmt="d", cmap="Greens", ax=axes[1])
    axes[1].set_title("Test: angle x expression")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "test_gender_angle_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return test


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    log1 = sample_grid("train", "sample_grid_train_official.png")
    log2 = sample_grid("test", "sample_grid_test_official.png")
    with open(os.path.join(FIGDIR, "sample_grid_files.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log1 + log2))
    test = test_manifest_analysis()
    print("\nĐã xong bước 3+4.")


if __name__ == "__main__":
    main()
