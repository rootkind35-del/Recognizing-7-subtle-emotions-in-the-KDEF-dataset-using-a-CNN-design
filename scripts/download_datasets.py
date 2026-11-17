"""Download and extract RADIATE and FER-2013 datasets with integrity checks."""

import argparse
import sys
import time
import zipfile
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RADIATE_BASE_URL = "https://abcdstudy.org/newsletter/fMRI_files"
FER2013_URL = "https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
KAGGLE_HEADERS = {"User-Agent": "kagglehub/0.3.0"}


def download_with_retry(
    url: str,
    dest: Path,
    headers: dict,
    max_retries: int = 5,
    chunk_size: int = 1 << 20,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and dest.stat().st_size > 0:
        if zipfile.is_zipfile(dest):
            print(f"[SKIP] Already valid ZIP: {dest} ({dest.stat().st_size:,} bytes)")
            return dest
        else:
            print(f"[WARN] Existing file is not a valid ZIP, re-downloading: {dest}")
            dest.unlink()

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[GET] ({attempt}/{max_retries}) {url} -> {dest}")
            with requests.get(
                url, headers=headers, allow_redirects=True, stream=True, timeout=60
            ) as resp:
                if resp.status_code in (401, 403):
                    raise PermissionError(
                        f"HTTP {resp.status_code} auth/access blocked for {url}"
                    )
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "")
                if "text/html" in ctype.lower():
                    snippet = next(resp.iter_content(512), b"").decode("utf-8", "ignore")
                    raise PermissionError(
                        f"Unexpected HTML response (possible reCAPTCHA/auth) for {url}: {snippet[:200]}"
                    )
                total = int(resp.headers.get("Content-Length") or 0)
                downloaded = 0
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                if total and downloaded != total:
                    raise IOError(
                        f"Incomplete download for {dest.name}: {downloaded} != {total}"
                    )
            tmp_path.replace(dest)
            with zipfile.ZipFile(dest, "r") as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise zipfile.BadZipFile(f"Corrupt entry {bad} in {dest}")
            print(f"[OK] Verified ZIP: {dest} ({dest.stat().st_size:,} bytes)")
            return dest
        except PermissionError:
            raise
        except Exception as exc:
            last_err = exc
            print(f"[RETRY] Attempt {attempt} failed: {exc}")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {last_err}")


def extract_zip(zip_path: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        zf.extractall(target_dir)
    print(f"[EXTRACT] {zip_path.name} -> {target_dir} ({len(members)} entries)")
    return len(members)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download RADIATE and FER-2013 datasets.")
    parser.add_argument("--only", choices=["all", "fer2013", "radiate", "radiate_bmp"], default="all")
    args = parser.parse_args()

    fer_dir = REPO_ROOT / "data" / "fer2013"
    rad_dir = REPO_ROOT / "data" / "radiate"

    if args.only in ("all", "fer2013"):
        fer_zip = fer_dir / "fer2013.zip"
        download_with_retry(FER2013_URL, fer_zip, KAGGLE_HEADERS)
        extract_zip(fer_zip, fer_dir)

    if args.only in ("all", "radiate"):
        for name in ("RADIATE_JPEGS1.zip", "RADIATE_JPEGS2.zip"):
            zp = rad_dir / name
            download_with_retry(f"{RADIATE_BASE_URL}/{name}", zp, BROWSER_HEADERS)
            extract_zip(zp, rad_dir)

    if args.only == "radiate_bmp":
        zp = rad_dir / "RADIATE_BMP.zip"
        download_with_retry(f"{RADIATE_BASE_URL}/RADIATE_BMP.zip", zp, BROWSER_HEADERS)
        extract_zip(zp, rad_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
