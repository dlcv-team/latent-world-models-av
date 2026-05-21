"""Upload embeddings from Modal volume to HuggingFace Hub.

Downloads embedding files from the Modal volume, then uploads to
the HuggingFace dataset repo.

Usage:
  # Download from Modal volume first
  modal volume get nuscenes-full /embeddings/ artifacts/full/embeddings/

  # Then upload to HuggingFace
  python scripts/upload_hf.py --repo surlac/lwm-av-embeddings --dir artifacts/full/embeddings/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload embeddings to HuggingFace Hub")
    parser.add_argument("--repo", required=True, help="HF repo ID (e.g., surlac/lwm-av-embeddings)")
    parser.add_argument("--dir", required=True, type=Path, help="Local directory with embedding files")
    parser.add_argument("--tag", default="v1.0", help="Git tag for this version")
    parser.add_argument("--token", default=None, help="HF write token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Provide --token or set HF_TOKEN environment variable")

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Upload all files in the directory
    files = sorted(args.dir.glob("*"))
    print(f"Uploading {len(files)} files to {args.repo} ...")

    for f in files:
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name} ({size_mb:.1f} MB) ...", end=" ", flush=True)
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=args.repo,
                repo_type="dataset",
            )
            print("done")

    # Create tag
    try:
        api.create_tag(
            repo_id=args.repo,
            repo_type="dataset",
            tag=args.tag,
            tag_message=f"Full-dataset embeddings ({len(files)} files)",
        )
        print(f"Tagged as {args.tag}")
    except Exception as e:
        print(f"Tag creation skipped: {e}")

    print(f"\nUpload complete: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
