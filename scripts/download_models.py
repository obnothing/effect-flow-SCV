from pathlib import Path
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[ERROR] huggingface_hub is not installed")
    print("        python -m pip install -U huggingface_hub")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODELS = [
    ("microsoft/codebert-base", "codebert-base"),
    ("microsoft/graphcodebert-base", "graphcodebert-base"),
    ("huggingface/CodeBERTa-small-v1", "CodeBERTa-small-v1"),
]

CONFIG_FILE = "config.json"
WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")
TOKENIZER_FILES = ("tokenizer.json", "vocab.json", "tokenizer_config.json")
ALLOW_PATTERNS = (
    "*.json",
    "*.txt",
    "*.md",
    ".gitattributes",
    "pytorch_model.bin",
    "model.safetensors",
)


def has_any_file(model_dir: Path, names: tuple[str, ...]) -> bool:
    return any((model_dir / name).is_file() for name in names)


def has_required_files(model_dir: Path) -> bool:
    return (
        (model_dir / CONFIG_FILE).is_file()
        and has_any_file(model_dir, WEIGHT_FILES)
        and has_any_file(model_dir, TOKENIZER_FILES)
    )


def describe_required_files(model_dir: Path) -> None:
    config_ok = (model_dir / CONFIG_FILE).is_file()
    weights_ok = has_any_file(model_dir, WEIGHT_FILES)
    tokenizer_ok = has_any_file(model_dir, TOKENIZER_FILES)

    print(f"[CHECK] {model_dir.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"        config.json: {'OK' if config_ok else 'MISSING'}")
    print(f"        weight file: {'OK' if weights_ok else 'MISSING'}")
    print(f"        tokenizer files: {'OK' if tokenizer_ok else 'MISSING'}")


def download_model(repo_id: str, local_name: str) -> bool:
    target_dir = MODELS_DIR / local_name

    if target_dir.exists() and has_required_files(target_dir):
        print(f"[OK] {local_name} already exists")
        return True

    print(f"[DOWNLOADING] {repo_id}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=ALLOW_PATTERNS,
            max_workers=1,
        )
    except Exception as exc:
        print(f"[ERROR] failed to download {repo_id}")
        print(f"        {exc}")
        return False

    if not has_required_files(target_dir):
        print(f"[ERROR] downloaded files incomplete for {repo_id}")
        describe_required_files(target_dir)
        return False

    print(f"[DONE] saved to {target_dir.relative_to(PROJECT_ROOT).as_posix()}")
    return True


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for repo_id, local_name in MODELS:
        all_ok = download_model(repo_id, local_name) and all_ok

    print()
    print("[SUMMARY]")
    for _, local_name in MODELS:
        describe_required_files(MODELS_DIR / local_name)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
