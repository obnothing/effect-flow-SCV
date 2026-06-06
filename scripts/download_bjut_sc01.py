from pathlib import Path
from zipfile import BadZipFile, ZipFile

import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "BJUT_SC01"
ZIP_PATH = RAW_DIR / "BJUT_SC01.zip"
PART_PATH = RAW_DIR / "BJUT_SC01.zip.part"
EXTRACTED_DIR = RAW_DIR / "extracted"

DOWNLOAD_URLS = [
    "https://github.com/huangjing2021/BJUT_SC01/raw/main/BJUT_SC01.zip",
    "https://github.com/huangjing2021/BJUT_SC01/raw/master/BJUT_SC01.zip",
    "https://raw.githubusercontent.com/huangjing2021/BJUT_SC01/main/BJUT_SC01.zip",
    "https://raw.githubusercontent.com/huangjing2021/BJUT_SC01/master/BJUT_SC01.zip",
]


def is_valid_zip(path):
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except BadZipFile:
        return False


def has_extracted_files(path):
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


def download_with_resume(url):
    headers = {}
    mode = "wb"
    existing_size = PART_PATH.stat().st_size if PART_PATH.exists() else 0
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    with requests.get(url, stream=True, timeout=30, headers=headers) as response:
        if response.status_code == 416:
            PART_PATH.replace(ZIP_PATH)
            return
        if response.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {response.status_code}")
        if response.status_code == 200 and existing_size > 0:
            mode = "wb"
            existing_size = 0

        total = response.headers.get("content-length")
        total = int(total) + existing_size if total is not None else None
        with tqdm(
            total=total,
            initial=existing_size,
            unit="B",
            unit_scale=True,
            desc="[DOWNLOADING] BJUT_SC01.zip",
        ) as progress:
            with PART_PATH.open(mode) as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))

    PART_PATH.replace(ZIP_PATH)


def download_zip():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if is_valid_zip(ZIP_PATH):
        print(f"[OK] zip already exists: {ZIP_PATH.relative_to(PROJECT_ROOT)}")
        return

    errors = []
    for url in DOWNLOAD_URLS:
        try:
            print(f"[DOWNLOADING] BJUT_SC01.zip from {url}")
            download_with_resume(url)
            if is_valid_zip(ZIP_PATH):
                print(f"[OK] downloaded: {ZIP_PATH.relative_to(PROJECT_ROOT)}")
                return
            errors.append(f"{url}: downloaded file is not a valid zip")
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Automatic download failed. Please manually download BJUT_SC01.zip from "
        "https://github.com/huangjing2021/BJUT_SC01 and place it at "
        f"{ZIP_PATH.relative_to(PROJECT_ROOT)}.\nErrors:\n- " + "\n- ".join(errors)
    )


def extract_zip():
    if is_valid_zip(ZIP_PATH) and has_extracted_files(EXTRACTED_DIR):
        print("[OK] dataset already exists")
        return

    if not is_valid_zip(ZIP_PATH):
        raise RuntimeError(f"Invalid or missing zip file: {ZIP_PATH}")

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[EXTRACTING] {ZIP_PATH.relative_to(PROJECT_ROOT)}")
    with ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACTED_DIR)
    print(f"[OK] extracted to {EXTRACTED_DIR.relative_to(PROJECT_ROOT)}")


def main():
    if is_valid_zip(ZIP_PATH) and has_extracted_files(EXTRACTED_DIR):
        print("[OK] dataset already exists")
        return
    download_zip()
    extract_zip()


if __name__ == "__main__":
    main()
