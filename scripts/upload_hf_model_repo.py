#!/usr/bin/env python3
"""Upload the trained BinGo four-bin waste model to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import LocalTokenNotFoundError


DEFAULT_MODEL_DIR = Path("models/four-bin-waste-vit-v2-target-adapted")
DEFAULT_REPO_NAME = "bingo-thai-four-bin-waste-vit"
REQUIRED_FILES = {
    "README.md",
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "thai_waste_labels.json",
    "requirements.txt",
    "metadata.json",
    "report.zh.md",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID"),
        help="Full Hugging Face repo id, for example username/bingo-thai-four-bin-waste-vit.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the model repo as public. Default is private.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload BinGo Thai four-bin waste classifier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check local files and authentication without uploading.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    validate_model_dir(model_dir)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    try:
        username = api.whoami()["name"]
    except (LocalTokenNotFoundError, HfHubHTTPError) as exc:
        raise SystemExit(
            "Missing Hugging Face authentication. Run `.venv-ml/bin/hf auth login` "
            "or create a write token at https://huggingface.co/settings/tokens and "
            "run with HF_TOKEN=..."
        ) from exc
    repo_id = args.repo_id or f"{username}/{DEFAULT_REPO_NAME}"

    if args.dry_run:
        print(f"Dry run OK. Authenticated as {username}.")
        print(f"Model directory: {model_dir}")
        print(f"Target repo: https://huggingface.co/{repo_id}")
        print(f"Visibility: {'public' if args.public else 'private'}")
        return

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message=args.commit_message,
    )

    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    missing_remote = sorted(REQUIRED_FILES - remote_files)
    if missing_remote:
        raise SystemExit(
            "Upload finished, but remote verification failed. Missing: "
            + ", ".join(missing_remote)
        )

    print(f"Uploaded model repo: https://huggingface.co/{repo_id}")
    print("Remote verification OK.")


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise SystemExit(f"Model path is not a directory: {model_dir}")

    files = {path.name for path in model_dir.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_FILES - files)
    if missing:
        raise SystemExit("Model directory is missing files: " + ", ".join(missing))

    model_file = model_dir / "model.safetensors"
    size_mb = model_file.stat().st_size / 1024 / 1024
    if size_mb < 100:
        raise SystemExit(
            f"model.safetensors looks too small ({size_mb:.1f} MB); refusing upload."
        )


if __name__ == "__main__":
    main()
