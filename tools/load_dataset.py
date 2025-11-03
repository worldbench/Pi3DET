#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
import random
from pathlib import Path
from typing import Optional, List

from huggingface_hub import (
    snapshot_download,
    hf_hub_download,
    HfApi,
)
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

REPO_ID = "robosense/datasets"
REPO_TYPE = "dataset"
SUBFOLDER = "track5-cross-platform-3d-object-detection"

def _copy_tree(src: Path, dst: Path):
    """Copy all files and directories from src to dst."""
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(Path(root) / f, dst / rel / f)

def _list_repo_files_in_subfolder(repo_id: str, subfolder: str, revision: Optional[str], token: Optional[str]) -> List[str]:
    """List all files under the given subfolder in the Hugging Face repo."""
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type=REPO_TYPE)
    subfolder = subfolder.strip("/") + "/"
    return [f for f in files if f.startswith(subfolder)]

def _sequential_download_with_retry(dest_dir: Path, files: List[str], revision: Optional[str], token: Optional[str]):
    """
    Download files sequentially with exponential backoff retries.
    This minimizes the chance of hitting 429 (Too Many Requests) errors.
    """
    for rel_path in files:
        out_path = dest_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if file already exists and is non-empty
        if out_path.exists() and out_path.stat().st_size > 0:
            continue

        delay = 1.5  # initial retry delay
        for attempt in range(8):  # up to 8 retries
            try:
                cached = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=rel_path,
                    repo_type=REPO_TYPE,
                    revision=revision,
                    token=token,
                    local_files_only=False,
                    force_download=False,
                    resume_download=True,
                )
                shutil.copy2(cached, out_path)
                break
            except HfHubHTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status in (429, 500, 502, 503, 504):
                    sleep_s = delay + random.uniform(0, 0.5 * delay)
                    time.sleep(sleep_s)
                    delay = min(delay * 2, 60)
                    if attempt == 7:
                        raise
                else:
                    raise
            except LocalEntryNotFoundError:
                sleep_s = delay + random.uniform(0, 0.5 * delay)
                time.sleep(sleep_s)
                delay = min(delay * 2, 60)
                if attempt == 7:
                    raise

def download_track5(dest_dir: str,
                    revision: Optional[str] = None,
                    token: Optional[str] = None) -> Path:
    """
    Download the track5-cross-platform-3d-object-detection subfolder from
    robosense/datasets into dest_dir.
    Uses snapshot_download with limited concurrency first, then falls back to
    sequential download with retries if rate-limited.
    """
    dest_dir = Path(dest_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_dir = dest_dir / SUBFOLDER
    out_dir.mkdir(parents=True, exist_ok=True)

    # First try snapshot_download with low concurrency
    try:
        cache_dir = snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=[f"{SUBFOLDER}/**"],
            revision=revision,
            token=token,
            local_dir=None,
            local_dir_use_symlinks=False,
            max_workers=2,  # reduce concurrency to avoid 429
            resume_download=True,
        )
        src = Path(cache_dir) / SUBFOLDER
        if not src.exists():
            raise FileNotFoundError(f"Subfolder not found in snapshot: {src}")
        _copy_tree(src, out_dir)
        return out_dir
    except (HfHubHTTPError, LocalEntryNotFoundError):
        # Fallback: sequential download with retries
        files = _list_repo_files_in_subfolder(REPO_ID, SUBFOLDER, revision, token)
        if not files:
            raise RuntimeError("No files found in target subfolder; check path or permissions.")
        _sequential_download_with_retry(dest_dir=dest_dir, files=files, revision=revision, token=token)
        return out_dir

def main():
    parser = argparse.ArgumentParser(
        description=f"Download {REPO_ID}/{SUBFOLDER} to a local path."
    )
    parser.add_argument("dest", help="Local directory to save the dataset (will be created if not exists)")
    parser.add_argument("--revision", default=None, help="Optional: branch/tag/commit (default: latest)")
    parser.add_argument("--token", default=os.environ.get("HUGGINGFACE_TOKEN"), help="Optional: HF access token")
    args = parser.parse_args()

    out_dir = download_track5(args.dest, revision=args.revision, token=args.token)
    print(f"✅ Download completed, saved to: {out_dir}")

if __name__ == "__main__":
    main()
