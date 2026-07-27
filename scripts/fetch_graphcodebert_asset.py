"""Fetch one pinned GraphCodeBERT revision and record a deterministic file-tree hash."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def tree_sha256(root):
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "asset_manifest.json"):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="models/graphcodebert-base"); parser.add_argument("--revision", required=True); args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    snapshot_download("microsoft/graphcodebert-base", revision=args.revision, local_dir=str(output), local_dir_use_symlinks=False)
    manifest = {"model": "microsoft/graphcodebert-base", "revision": args.revision, "tree_sha256": tree_sha256(output)}
    (output / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
