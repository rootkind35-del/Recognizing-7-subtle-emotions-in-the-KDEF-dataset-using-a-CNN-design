# -*- coding: utf-8 -*-
"""Kiểm tra rò rỉ dữ liệu (data leakage) giữa các split của 2 bộ dữ liệu KDEF.

Chạy:  python reports/check_leakage.py
Kết quả: in JSON ra stdout, exit 0 khi chạy xong.

Nội dung kiểm tra:
1. Bộ official (data/official-kdef-dataset-by-ksatrajit0) — TOÀN BỘ ảnh:
   - MD5 từng file; đếm ảnh trùng MD5 chéo từng cặp split (train-val, train-test, val-test)
     và trùng lặp trong nội bộ từng split.
   - Theo manifest kdef_manifest_splits.csv: subject xuất hiện ở hơn một split.
2. Bộ balanced (data/balanced-kdef-dataset-by-dolly-prajapati-182) — MẪU
   (mặc định 3.000 ảnh/split, tổng 9.000 >= 8.000 theo yêu cầu):
   - MD5 từng file mẫu; đếm ảnh trùng MD5 chéo từng cặp split.
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "data", "official-kdef-dataset-by-ksatrajit0")
BALANCED = os.path.join(ROOT, "data", "balanced-kdef-dataset-by-dolly-prajapati-182")
MANIFEST = os.path.join(OFFICIAL, "kdef_manifest_splits.csv")
SPLITS = ["train", "val", "test"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
PAIRS = [("train", "val"), ("train", "test"), ("val", "test")]
BALANCED_SAMPLE_PER_SPLIT = 3000  # 3 x 3000 = 9000 >= 8000
SEED = 42


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def list_images(root, sample_per_split=None):
    """Trả về list (split, path). Nếu sample_per_split: lấy ngẫu nhiên cố định seed."""
    import random
    items = []
    for split in SPLITS:
        split_dir = os.path.join(root, split)
        split_items = []
        for cls in sorted(os.listdir(split_dir)):
            d = os.path.join(split_dir, cls)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if os.path.splitext(f)[1].lower() in EXTS:
                    split_items.append(os.path.join(d, f))
        if sample_per_split is not None and len(split_items) > sample_per_split:
            rng = random.Random(SEED)
            split_items = rng.sample(split_items, sample_per_split)
        items.extend((split, p) for p in split_items)
    return items


def md5_cross_split(root, sample_per_split=None):
    """MD5 các ảnh và đếm ảnh trùng chéo split + trùng trong nội bộ split."""
    items = list_images(root, sample_per_split)
    hash_to_splits = defaultdict(set)
    hash_count = defaultdict(int)
    for split, path in items:
        digest = md5_of(path)
        hash_to_splits[digest].add(split)
        hash_count[digest] += 1

    cross = {f"{a}-{b}": 0 for a, b in PAIRS}
    duplicated_hashes_cross = 0
    for digest, splits in hash_to_splits.items():
        for a, b in PAIRS:
            if a in splits and b in splits:
                cross[f"{a}-{b}"] += hash_count[digest]
                duplicated_hashes_cross += 1
                break  # chỉ đếm hash 1 lần cho thống kê số hash rò rỉ
    within = {s: 0 for s in SPLITS}
    for digest, splits in hash_to_splits.items():
        if len(splits) == 1 and hash_count[digest] > 1:
            within[next(iter(splits))] += hash_count[digest] - 1
    return {
        "n_images_hashed": len(items),
        "n_unique_md5": len(hash_to_splits),
        "cross_split_duplicate_images": cross,
        "n_hashes_appearing_in_multiple_splits": duplicated_hashes_cross,
        "within_split_extra_duplicates": within,
    }


def subject_overlap_manifest(manifest_path):
    man = pd.read_csv(manifest_path)
    subj_splits = man.groupby("subject")["split"].apply(lambda s: set(s))
    overlap = {}
    for a, b in PAIRS:
        n = sum(1 for s in subj_splits if a in s and b in s)
        overlap[f"{a}-{b}"] = n
    multi = sorted(subj_splits[subj_splits.apply(len) > 1].index.tolist())
    per_split = man.groupby("split")["subject"].nunique().to_dict()
    return {
        "manifest_rows": len(man),
        "subjects_per_split": {k: int(v) for k, v in per_split.items()},
        "subjects_in_multiple_splits": len(multi),
        "subject_overlap_by_pair": overlap,
        "subject_list_overlap": multi,
    }


def main():
    result = {
        "official_full": md5_cross_split(OFFICIAL, sample_per_split=None),
        "official_manifest_subjects": subject_overlap_manifest(MANIFEST),
        "balanced_sample": md5_cross_split(BALANCED, sample_per_split=BALANCED_SAMPLE_PER_SPLIT),
        "config": {
            "balanced_sample_per_split": BALANCED_SAMPLE_PER_SPLIT,
            "seed": SEED,
            "note": "official hash toàn bộ; balanced hash mẫu 3000/split",
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
