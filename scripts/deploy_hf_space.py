#!/usr/bin/env python3
"""Create/update the BinGo Hugging Face Space inference API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, get_token


DEFAULT_SPACE_DIR = Path("spaces/bingo-thai-waste-api")
DEFAULT_SPACE_ID = "ChrisSujinlv/bingo-thai-waste-api"
DEFAULT_MODEL_REPO_ID = "ChrisSujinlv/bingo-thai-four-bin-waste-vit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-dir", default=str(DEFAULT_SPACE_DIR))
    parser.add_argument("--space-id", default=os.environ.get("HF_SPACE_ID", DEFAULT_SPACE_ID))
    parser.add_argument(
        "--model-repo-id",
        default=os.environ.get("MODEL_REPO_ID", DEFAULT_MODEL_REPO_ID),
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    space_dir = Path(args.space_dir)
    validate_space_dir(space_dir)

    upload_token = get_token()
    if not upload_token:
        raise SystemExit("Run `.venv-ml/bin/hf auth login` with a write token first.")

    runtime_token = os.environ.get("HF_RUNTIME_TOKEN")
    if not runtime_token:
        raise SystemExit("Missing HF_RUNTIME_TOKEN for the Space secret.")

    api = HfApi(token=upload_token)
    username = api.whoami()["name"]

    if args.dry_run:
        print(f"Dry run OK. Authenticated as {username}.")
        print(f"Space: https://huggingface.co/spaces/{args.space_id}")
        print(f"Visibility: {'private' if args.private else 'public'}")
        print(f"Model repo: {args.model_repo_id}")
        return

    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )
    api.add_space_secret(
        repo_id=args.space_id,
        key="HF_TOKEN",
        value=runtime_token,
        description="Read private model weights for the BinGo classifier.",
    )
    api.add_space_variable(
        repo_id=args.space_id,
        key="MODEL_REPO_ID",
        value=args.model_repo_id,
        description="Source model repo loaded by the FastAPI Space.",
    )
    api.upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=str(space_dir),
        commit_message="Deploy BinGo Thai waste inference API",
    )

    files = api.list_repo_files(repo_id=args.space_id, repo_type="space")
    required = {"README.md", "Dockerfile", "app.py", "requirements.txt"}
    missing = sorted(required - set(files))
    if missing:
        raise SystemExit("Remote Space is missing files: " + ", ".join(missing))

    print(f"Uploaded Space: https://huggingface.co/spaces/{args.space_id}")
    print(f"API base URL: https://{args.space_id.replace('/', '-')}.hf.space")
    print("Remote Space file verification OK.")


def validate_space_dir(space_dir: Path) -> None:
    if not space_dir.is_dir():
        raise SystemExit(f"Space directory does not exist: {space_dir}")

    required = {"README.md", "Dockerfile", "app.py", "requirements.txt"}
    files = {path.name for path in space_dir.iterdir() if path.is_file()}
    missing = sorted(required - files)
    if missing:
        raise SystemExit("Space directory is missing files: " + ", ".join(missing))


if __name__ == "__main__":
    main()
