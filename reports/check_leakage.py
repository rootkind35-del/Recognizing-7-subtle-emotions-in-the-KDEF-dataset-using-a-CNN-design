# -*- coding: utf-8 -*-
"""Kiểm tra rò rỉ dữ liệu (data leakage) giữa các split của bộ dữ liệu KDEF và đa văn hóa 3 nguồn.

Chạy:  python reports/check_leakage.py
Kết quả: in JSON ra stdout, ghi `reports/leakage_result.json`, exit 0 khi sạch / exit 1 nếu phát hiện rò rỉ hoặc ảnh hỏng.

Nội dung kiểm tra:
1. Bộ official KDEF (nếu có): MD5 chéo split & subject overlap từ kdef_manifest_splits.csv.
2. Bộ balanced KDEF (nếu có): mẫu 3.000 ảnh/split.
3. Bộ đa văn hóa 3 nguồn (`data/manifests/multicultural_manifest.csv`):
   - Toàn bộ file tồn tại trên đĩa và MD5 khớp cột `md5` trong manifest.
   - 0 bản sao MD5 chéo giữa các split (train-val, train-test, val-test).
   - 0 ảnh đen/hỏng (`CORRUPT_MD5` hoặc `img is None` hoặc `max == 0`) trong `train` và `val`.
   - 0 subject overlap giữa các split trên các nguồn có định danh người (`subject_id != "unknown"`).
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "data", "official-kdef-dataset-by-ksatrajit0")
BALANCED = os.path.join(ROOT, "data", "balanced-kdef-dataset-by-dolly-prajapati-182")
MANIFEST = os.path.join(OFFICIAL, "kdef_manifest_splits.csv")
MULTICULTURAL_MANIFEST = os.path.join(ROOT, "data", "manifests", "multicultural_manifest.csv")
DEFAULT_OUT_JSON = os.path.join(ROOT, "reports", "leakage_result.json")

CORRUPT_MD5 = "320b34032e2133fdfb80a58c953d5c94"
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
        if not os.path.isdir(split_dir):
            continue
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
                break
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


def audit_multicultural_manifest(manifest_path: str) -> dict:
    """Audit the unified 3-source manifest for missing files, MD5 leakage, black frames, and subject overlap."""
    df = pd.read_csv(manifest_path)
    root_path = Path(ROOT)

    missing_files = []
    md5_mismatches = []
    corrupt_train_val = []

    hash_to_splits = defaultdict(set)
    hash_count_by_split = defaultdict(lambda: defaultdict(int))
    subject_to_splits = defaultdict(set)

    for row in df.itertuples(index=False):
        rel_fp = str(row.filepath)
        split = str(row.split).strip().lower()
        expected_md5 = str(row.md5).strip().lower()
        subj = str(row.subject_id).strip()

        abs_path = root_path / rel_fp
        if not abs_path.is_file():
            missing_files.append(rel_fp)
            continue

        actual_md5 = md5_of(abs_path)
        if actual_md5 != expected_md5:
            md5_mismatches.append({"filepath": rel_fp, "manifest_md5": expected_md5, "disk_md5": actual_md5})

        hash_to_splits[actual_md5].add(split)
        hash_count_by_split[actual_md5][split] += 1

        if subj and subj.lower() not in ("unknown", "nan", "none"):
            subject_to_splits[subj].add(split)

        if split in ("train", "val"):
            if actual_md5 == CORRUPT_MD5:
                corrupt_train_val.append({"filepath": rel_fp, "split": split, "reason": "CORRUPT_MD5"})
            else:
                img = cv2.imread(str(abs_path), cv2.IMREAD_UNCHANGED)
                if img is None or img.size == 0 or int(img.max()) == 0:
                    corrupt_train_val.append({"filepath": rel_fp, "split": split, "reason": "blank_or_unreadable"})

    cross_split_groups = 0
    cross_split_pair_counts = {f"{a}-{b}": 0 for a, b in PAIRS}
    within_split_extra = {s: 0 for s in SPLITS}

    for digest, splits in hash_to_splits.items():
        if len(splits) > 1:
            cross_split_groups += 1
            for a, b in PAIRS:
                if a in splits and b in splits:
                    cross_split_pair_counts[f"{a}-{b}"] += (
                        hash_count_by_split[digest][a] + hash_count_by_split[digest][b]
                    )
        else:
            s = next(iter(splits))
            cnt = hash_count_by_split[digest][s]
            if cnt > 1 and s in within_split_extra:
                within_split_extra[s] += cnt - 1

    multi_subjects = sorted([s for s, sp in subject_to_splits.items() if len(sp) > 1])
    subj_overlap_pairs = {
        f"{a}-{b}": sum(1 for sp in subject_to_splits.values() if a in sp and b in sp)
        for a, b in PAIRS
    }
    subjects_per_split = {
        s: len({
            str(r.subject_id).strip()
            for r in df[df["split"] == s].itertuples(index=False)
            if str(r.subject_id).strip().lower() not in ("unknown", "nan", "none", "")
        })
        for s in SPLITS
    }

    split_counts = {s: int((df["split"] == s).sum()) for s in SPLITS}
    source_split_counts = {
        src: {s: int(((df["source"] == src) & (df["split"] == s)).sum()) for s in SPLITS}
        for src in sorted(df["source"].astype(str).unique().tolist())
    }

    passed = (
        len(missing_files) == 0
        and len(md5_mismatches) == 0
        and cross_split_groups == 0
        and len(corrupt_train_val) == 0
        and len(multi_subjects) == 0
    )

    try:
        manifest_label = os.path.relpath(manifest_path, ROOT).replace("\\", "/")
    except ValueError:
        manifest_label = os.path.abspath(manifest_path).replace("\\", "/")

    return {
        "manifest_path": manifest_label,
        "passed": passed,
        "total_rows": int(len(df)),
        "split_counts": split_counts,
        "source_split_counts": source_split_counts,
        "n_unique_md5": len(hash_to_splits),
        "n_missing_files": len(missing_files),
        "missing_files_sample": missing_files[:10],
        "n_md5_mismatches": len(md5_mismatches),
        "md5_mismatches_sample": md5_mismatches[:10],
        "cross_split_md5_groups": cross_split_groups,
        "cross_split_md5_pair_images": cross_split_pair_counts,
        "within_split_extra_md5_duplicates": within_split_extra,
        "n_corrupt_or_black_in_train_val": len(corrupt_train_val),
        "corrupt_in_train_val_sample": corrupt_train_val[:10],
        "n_tracked_subjects_total": len(subject_to_splits),
        "subjects_per_split": subjects_per_split,
        "subjects_in_multiple_splits": len(multi_subjects),
        "subject_overlap_by_pair": subj_overlap_pairs,
        "subject_overlap_list": multi_subjects[:20],
    }


def main():
    parser = argparse.ArgumentParser(description="Audit KDEF and 3-source multicultural dataset leakage.")
    parser.add_argument("--manifest", default=MULTICULTURAL_MANIFEST, help="Path to 3-source manifest CSV.")
    parser.add_argument("--out", default=DEFAULT_OUT_JSON, help="Output JSON path.")
    args = parser.parse_args()

    result = {
        "config": {
            "balanced_sample_per_split": BALANCED_SAMPLE_PER_SPLIT,
            "seed": SEED,
        }
    }

    if os.path.isdir(OFFICIAL):
        result["official_full"] = md5_cross_split(OFFICIAL, sample_per_split=None)
    if os.path.isfile(MANIFEST):
        result["official_manifest_subjects"] = subject_overlap_manifest(MANIFEST)
    if os.path.isdir(BALANCED):
        result["balanced_sample"] = md5_cross_split(BALANCED, sample_per_split=BALANCED_SAMPLE_PER_SPLIT)

    exit_code = 0
    if os.path.isfile(args.manifest):
        mc_audit = audit_multicultural_manifest(args.manifest)
        result["multicultural_manifest"] = mc_audit
        if not mc_audit["passed"]:
            exit_code = 1
    elif args.manifest != MULTICULTURAL_MANIFEST:
        result["error"] = f"Requested manifest not found: {args.manifest}"
        exit_code = 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
