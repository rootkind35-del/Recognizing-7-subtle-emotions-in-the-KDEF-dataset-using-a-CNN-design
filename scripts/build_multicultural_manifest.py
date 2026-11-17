"""Build the unified 3-source multicultural manifest (KDEF + RADIATE + FER-2013).

Outputs:
  - data/manifests/multicultural_manifest.csv
  - data/manifests/manifest_summary.json

Columns (exact):
  filepath, split, source, label, label_idx, subject_id, ethnic_group, gender, angle, md5
"""

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import (  # noqa: E402
    CLASS_NAMES,
    CLASS_TO_IDX,
    CORRUPT_MD5,
    DEFAULT_MULTICULTURAL_MEAN,
    DEFAULT_MULTICULTURAL_STD,
    compute_class_weights,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = [
    "filepath",
    "split",
    "source",
    "label",
    "label_idx",
    "subject_id",
    "ethnic_group",
    "gender",
    "angle",
    "md5",
]

# Authoritative 16 NimStim expression codes used in RADIATE (Conley et al., 2018)
RADIATE_CODE_MAP: Dict[str, str] = {
    "AC": "angry",
    "AO": "angry",
    "CC": "DROP_CALM",
    "CO": "DROP_CALM",
    "DC": "disgust",
    "DO": "disgust",
    "FC": "fear",
    "FO": "fear",
    "HC": "happy",
    "HO": "happy",
    "HE": "happy",
    "NC": "neutral",
    "NO": "neutral",
    "SC": "sad",
    "SO": "sad",
    "SUR": "surprise",
}

# Stratified ~70/10/20 allocation across RADIATE ethnic groups (total 76/11/22 = 109)
RADIATE_GROUP_SPLIT_TARGETS = {
    "B": (26, 4, 8),   # 38
    "W": (19, 3, 6),   # 28
    "A": (16, 2, 4),   # 22
    "H": (14, 2, 4),   # 20
    "O": (1, 0, 0),    # 1 (cannot exist in >1 split)
}


def compute_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def is_corrupt_or_black(path: Path, md5_hex: str) -> bool:
    if md5_hex == CORRUPT_MD5:
        return True
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0 or int(img.max()) == 0:
        return True
    return False


def collect_kdef() -> List[Dict]:
    kdef_root = REPO_ROOT / "data" / "official-kdef-dataset-by-ksatrajit0"
    csv_path = kdef_root / "kdef_manifest_splits.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing KDEF split manifest: {csv_path}")

    df = pd.read_csv(csv_path)
    records: List[Dict] = []
    for row in df.itertuples(index=False):
        split = str(row.split).strip().lower()
        label = str(row.expression).strip().lower()
        filename = str(row.filename).strip()
        if label not in CLASS_TO_IDX:
            raise ValueError(f"Unknown KDEF label '{label}' for {filename}")
        if split not in SPLITS:
            raise ValueError(f"Unknown KDEF split '{split}' for {filename}")

        fpath = kdef_root / split / label / filename
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing KDEF image on disk: {fpath}")

        gender_raw = str(row.gender).strip().upper()
        gender = "female" if gender_raw == "F" else ("male" if gender_raw == "M" else "unknown")
        subj = str(row.subject).strip()
        angle = str(row.angle).strip().upper()

        records.append({
            "filepath": fpath.relative_to(REPO_ROOT).as_posix(),
            "split": split,
            "source": "kdef",
            "label": label,
            "label_idx": CLASS_TO_IDX[label],
            "subject_id": f"kdef_{subj}",
            "ethnic_group": "KDEF",
            "gender": gender,
            "angle": angle,
            "md5": compute_md5(fpath),
        })
    return records


def collect_radiate(seed: int = 42) -> tuple[List[Dict], Dict]:
    rad_root = REPO_ROOT / "data" / "radiate"
    if not rad_root.is_dir():
        raise FileNotFoundError(f"Missing RADIATE directory: {rad_root}")

    # Select only Color JPEG folders, ignoring __MACOSX, AppleDouble ._* files, and BW folders
    color_files: List[Path] = []
    for p in sorted(rad_root.rglob("*")):
        if not p.is_file():
            continue
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if any(part.lower() == "color" or "_color_" in part.lower() for part in p.parts):
            color_files.append(p)

    if not color_files:
        raise RuntimeError(f"No RADIATE color images found under {rad_root}")

    parsed = []
    dropped_calm = 0
    subjects_by_group: Dict[str, set] = defaultdict(set)

    for fpath in color_files:
        parts = fpath.stem.split("_")
        if len(parts) != 2:
            raise ValueError(f"Unexpected RADIATE filename stem format: {fpath.name}")
        subj_code, expr_token = parts[0].strip(), parts[1].strip().upper()
        if expr_token not in RADIATE_CODE_MAP:
            raise ValueError(f"Unknown RADIATE expression token '{expr_token}' in {fpath}")

        mapped = RADIATE_CODE_MAP[expr_token]
        if mapped == "DROP_CALM":
            dropped_calm += 1
            continue

        ethnic_group = subj_code[0].upper()
        gender_char = subj_code[1].upper()
        if ethnic_group not in RADIATE_GROUP_SPLIT_TARGETS:
            raise ValueError(f"Unexpected RADIATE ethnic_group '{ethnic_group}' in {fpath.name}")
        if gender_char not in ("F", "M"):
            raise ValueError(f"Unexpected RADIATE gender char '{gender_char}' in {fpath.name}")

        subjects_by_group[ethnic_group].add(subj_code)
        parsed.append({
            "fpath": fpath,
            "subj_code": subj_code,
            "ethnic_group": ethnic_group,
            "gender": "female" if gender_char == "F" else "male",
            "label": mapped,
        })

    # Assign subject-disjoint stratified splits with deterministic seed
    rng = random.Random(seed)
    subj_to_split: Dict[str, str] = {}
    group_split_subject_counts: Dict[str, Dict[str, int]] = {}

    for grp, subjs in sorted(subjects_by_group.items()):
        sorted_subjs = sorted(subjs)
        n_train, n_val, n_test = RADIATE_GROUP_SPLIT_TARGETS[grp]
        if len(sorted_subjs) != (n_train + n_val + n_test):
            raise ValueError(
                f"RADIATE group '{grp}' has {len(sorted_subjs)} subjects, "
                f"expected {n_train + n_val + n_test}"
            )
        shuffled = list(sorted_subjs)
        rng.shuffle(shuffled)
        val_subjs = shuffled[:n_val]
        test_subjs = shuffled[n_val : n_val + n_test]
        train_subjs = shuffled[n_val + n_test :]

        for s in train_subjs:
            subj_to_split[s] = "train"
        for s in val_subjs:
            subj_to_split[s] = "val"
        for s in test_subjs:
            subj_to_split[s] = "test"
        group_split_subject_counts[grp] = {
            "train": len(train_subjs),
            "val": len(val_subjs),
            "test": len(test_subjs),
            "total": len(sorted_subjs),
        }

    records: List[Dict] = []
    for item in parsed:
        fpath = item["fpath"]
        subj_code = item["subj_code"]
        label = item["label"]
        records.append({
            "filepath": fpath.relative_to(REPO_ROOT).as_posix(),
            "split": subj_to_split[subj_code],
            "source": "radiate",
            "label": label,
            "label_idx": CLASS_TO_IDX[label],
            "subject_id": f"radiate_{subj_code}",
            "ethnic_group": item["ethnic_group"],
            "gender": item["gender"],
            "angle": "frontal",
            "md5": compute_md5(fpath),
        })

    stats = {
        "total_color_images": len(color_files),
        "dropped_calm": dropped_calm,
        "retained_images": len(records),
        "n_subjects": len(subj_to_split),
        "group_split_subjects": group_split_subject_counts,
    }
    return records, stats


def collect_fer2013() -> List[Dict]:
    fer_root = REPO_ROOT / "data" / "fer2013"
    if not fer_root.is_dir():
        raise FileNotFoundError(f"Missing FER-2013 directory: {fer_root}")

    records: List[Dict] = []
    for folder_name in ("train", "test"):
        folder = fer_root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing FER-2013 subfolder: {folder}")
        for cls_dir in sorted(folder.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = cls_dir.name.strip().lower()
            if label not in CLASS_TO_IDX:
                raise ValueError(f"Unexpected FER-2013 emotion folder: {cls_dir}")
            for fpath in sorted(cls_dir.iterdir()):
                if not fpath.is_file() or fpath.suffix.lower() not in IMAGE_EXTS:
                    continue
                name = fpath.name
                if name.startswith("Training_"):
                    split = "train"
                elif name.startswith("PublicTest_"):
                    split = "val"
                elif name.startswith("PrivateTest_"):
                    split = "test"
                else:
                    raise ValueError(f"Unexpected FER-2013 filename prefix: {fpath}")

                records.append({
                    "filepath": fpath.relative_to(REPO_ROOT).as_posix(),
                    "split": split,
                    "source": "fer2013",
                    "label": label,
                    "label_idx": CLASS_TO_IDX[label],
                    "subject_id": "unknown",
                    "ethnic_group": "FER",
                    "gender": "unknown",
                    "angle": "unknown",
                    "md5": compute_md5(fpath),
                })
    return records


def main() -> int:
    print("Collecting KDEF...")
    kdef_recs = collect_kdef()
    print(f"  KDEF raw: {len(kdef_recs)}")

    print("Collecting RADIATE...")
    rad_recs, rad_stats = collect_radiate(seed=42)
    print(f"  RADIATE usable (calm dropped={rad_stats['dropped_calm']}): {len(rad_recs)} ({rad_stats['n_subjects']} subjects)")

    print("Collecting FER-2013...")
    fer_recs = collect_fer2013()
    print(f"  FER-2013 raw: {len(fer_recs)}")

    raw_df = pd.DataFrame(kdef_recs + rad_recs + fer_recs, columns=MANIFEST_COLUMNS)
    raw_split_counts = raw_df["split"].value_counts().to_dict()
    raw_source_split_counts = {
        src: {s: int(((raw_df["source"] == src) & (raw_df["split"] == s)).sum()) for s in SPLITS}
        for src in ("kdef", "radiate", "fer2013")
    }
    print(f"Total raw rows: {len(raw_df)} | splits: {raw_split_counts}")

    # Step 1: Identify corrupt / black frames
    corrupt_mask = [
        is_corrupt_or_black(REPO_ROOT / r.filepath, r.md5)
        for r in raw_df.itertuples(index=False)
    ]
    raw_df["_is_corrupt"] = corrupt_mask
    corrupt_by_split_source = (
        raw_df[raw_df["_is_corrupt"]]
        .groupby(["split", "source"])
        .size()
        .to_dict()
    )
    corrupt_in_test = int(((raw_df["_is_corrupt"]) & (raw_df["split"] == "test")).sum())
    if corrupt_in_test > 0:
        raise RuntimeError(f"Found {corrupt_in_test} corrupt frames in test split!")

    # Remove corrupt frames from train/val
    df_noncorrupt = raw_df[~((raw_df["_is_corrupt"]) & (raw_df["split"].isin(["train", "val"])))].copy()
    removed_corrupt = len(raw_df) - len(df_noncorrupt)

    # Step 2: Cross-split MD5 deduplication (remove ALL train/val copies of any MD5 in >=2 splits; keep test & within-split)
    md5_splits: Dict[str, set] = defaultdict(set)
    for r in df_noncorrupt.itertuples(index=False):
        md5_splits[r.md5].add(r.split)
    cross_split_hashes = {m for m, s in md5_splits.items() if len(s) > 1}

    drop_cross_mask = (df_noncorrupt["md5"].isin(cross_split_hashes)) & (
        df_noncorrupt["split"].isin(["train", "val"])
    )
    removed_cross_md5 = int(drop_cross_mask.sum())
    removed_cross_by_split = (
        df_noncorrupt[drop_cross_mask]["split"].value_counts().to_dict()
    )
    clean_df = df_noncorrupt[~drop_cross_mask].drop(columns=["_is_corrupt"]).reset_index(drop=True)

    # Sort deterministically: split order, source, filepath
    split_order = {"train": 0, "val": 1, "test": 2}
    clean_df["_split_ord"] = clean_df["split"].map(split_order)
    clean_df = clean_df.sort_values(["_split_ord", "source", "filepath"]).drop(columns=["_split_ord"]).reset_index(drop=True)

    # Assertions on clean manifest
    post_md5_splits: Dict[str, set] = defaultdict(set)
    for r in clean_df.itertuples(index=False):
        post_md5_splits[r.md5].add(r.split)
    remaining_cross = [m for m, s in post_md5_splits.items() if len(s) > 1]
    assert len(remaining_cross) == 0, f"Cross-split MD5 leakage remains: {len(remaining_cross)}"

    subj_splits: Dict[str, set] = defaultdict(set)
    for r in clean_df.itertuples(index=False):
        if r.subject_id != "unknown":
            subj_splits[r.subject_id].add(r.split)
    overlapping_subjs = [s for s, sp in subj_splits.items() if len(sp) > 1]
    assert len(overlapping_subjs) == 0, f"Subject leakage detected: {overlapping_subjs}"
    assert len(subj_splits) == 179, f"Expected 179 tracked subjects, got {len(subj_splits)}"

    out_dir = REPO_ROOT / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = out_dir / "multicultural_manifest.csv"
    clean_df.to_csv(manifest_csv, index=False)

    # Compute train class weights & stats
    train_df = clean_df[clean_df["split"] == "train"]
    weights_t = compute_class_weights(train_df)
    train_class_counts = {
        c: int((train_df["label"] == c).sum()) for c in CLASS_NAMES
    }
    class_weights = {
        c: round(float(weights_t[i].item()), 6) for i, c in enumerate(CLASS_NAMES)
    }

    clean_split_counts = {s: int((clean_df["split"] == s).sum()) for s in SPLITS}
    clean_source_split_counts = {
        src: {s: int(((clean_df["source"] == src) & (clean_df["split"] == s)).sum()) for s in SPLITS}
        for src in ("kdef", "radiate", "fer2013")
    }
    clean_ethnic_split_counts = {
        grp: {s: int(((clean_df["ethnic_group"] == grp) & (clean_df["split"] == s)).sum()) for s in SPLITS}
        for grp in sorted(clean_df["ethnic_group"].unique().tolist())
    }

    summary = {
        "manifest_path": manifest_csv.relative_to(REPO_ROOT).as_posix(),
        "seed": 42,
        "raw_total": int(len(raw_df)),
        "raw_split_counts": {k: int(v) for k, v in raw_split_counts.items()},
        "raw_source_split_counts": raw_source_split_counts,
        "radiate_ingestion": rad_stats,
        "cleaning": {
            "removed_corrupt_black_train_val": int(removed_corrupt),
            "corrupt_by_split_source": {f"{k[0]}_{k[1]}": int(v) for k, v in corrupt_by_split_source.items()},
            "cross_split_md5_groups": len(cross_split_hashes),
            "removed_cross_split_md5_train_val": removed_cross_md5,
            "removed_cross_split_md5_by_split": {k: int(v) for k, v in removed_cross_by_split.items()},
            "total_removed": int(removed_corrupt + removed_cross_md5),
        },
        "clean_total": int(len(clean_df)),
        "clean_split_counts": clean_split_counts,
        "clean_source_split_counts": clean_source_split_counts,
        "clean_ethnic_split_counts": clean_ethnic_split_counts,
        "tracked_subjects_total": len(subj_splits),
        "subject_overlap": len(overlapping_subjs),
        "cross_split_md5_overlap": len(remaining_cross),
        "train_class_counts": train_class_counts,
        "train_class_weights": class_weights,
        "normalization": {
            "mean": DEFAULT_MULTICULTURAL_MEAN,
            "std": DEFAULT_MULTICULTURAL_STD,
            "computed_on": "clean train split only (32,404 images, ssd_pad_160_1ch)",
        },
    }
    summary_path = out_dir / "manifest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== CLEAN MANIFEST SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
