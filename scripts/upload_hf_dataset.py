#!/usr/bin/env python3
"""
Upload a directory (with subfolders/files) to a Hugging Face *dataset* repo.

- Uses huggingface_hub API (no local Git repo needed)
- Include or exclude files via glob patterns
- Optional dry-run to preview what would be uploaded
- Creates the repo if it doesn't exist
- Supports branches (revision) and path_in_repo subfolder

Usage examples:

  # Preview first (no network changes)
  python upload_hf_dataset.py \
    --src hf_release/v1 \
    --repo-id reachusama0/uk-housing-data \
    --allow "**/*.parquet" \
    --dry-run

  # Upload parquet + docs to the repo root
  python upload_hf_dataset.py \
    --src hf_release/v1 \
    --repo-id reachusama0/uk-housing-data \
    --allow "**/*.parquet" "README.md" "dataset_card.md" "LICENSE" \
    --message "v1 parquet + docs"

  # Upload under a subfolder (e.g., /v1) on a branch named v1
  python upload_hf_dataset.py \
    --src hf_release/v1 \
    --repo-id reachusama0/uk-housing-data \
    --path-in-repo v1 \
    --revision v1 \
    --allow "**/*.parquet" "README.md" "dataset_card.md" "LICENSE" \
    --message "release v1"
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional
from fnmatch import fnmatch

from huggingface_hub import HfApi, create_repo, upload_folder, hf_hub_url
from huggingface_hub.utils import HfHubHTTPError


def _norm_patterns(patts: Optional[Iterable[str]]) -> List[str]:
    if not patts:
        return []
    return [str(p) for p in patts]


def _list_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def _filter_paths(paths: List[Path], root: Path,
                  allow: List[str], ignore: List[str]) -> List[Path]:
    """
    Local preview of what --allow / --ignore will do.
    This mirrors the spirit of huggingface_hub filters (not byte-for-byte identical).
    """
    rels = []
    for p in paths:
        rel = str(p.relative_to(root)).replace("\\", "/")
        rels.append((p, rel))

    def any_match(rel: str, patterns: List[str]) -> bool:
        return any(fnmatch(rel, pat) for pat in patterns)

    out = []
    if allow:
        # Keep only files that match allow AND not ignored
        for p, rel in rels:
            if any_match(rel, allow) and not any_match(rel, ignore):
                out.append(p)
    else:
        # Keep all that are not ignored
        for p, rel in rels:
            if not any_match(rel, ignore):
                out.append(p)
    return out


def upload_dir(
        src: str,
        repo_id: str,
        *,
        path_in_repo: str = "",
        revision: str = "main",
        private: bool = False,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        commit_message: str = "Upload dataset folder",
        create_pr: bool = False,
        token: Optional[str] = None,
):
    api = HfApi(token=token)

    # Create repo if missing
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )
    except HfHubHTTPError as e:
        # If it's a permission or auth issue, let it bubble with a friendly tip
        raise RuntimeError(f"Failed to create/access repo {repo_id}: {e}") from e

    # Do the upload
    res = upload_folder(
        folder_path=src,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        create_pr=create_pr,
        token=token,
    )

    url = hf_hub_url(repo_id=repo_id, filename="", repo_type="dataset", revision=revision)
    print(f"Uploaded to: https://huggingface.co/datasets/{repo_id}/tree/{revision}/{path_in_repo}".rstrip("/"))
    print(f"Commit: {res.oid}  (branch: {revision})")


def main():
    ap = argparse.ArgumentParser(description="Upload a folder tree to a Hugging Face *dataset* repo.")
    ap.add_argument("--src", required=True, help="Local source folder to upload")
    ap.add_argument("--repo-id", required=True, help="e.g., <username>/<dataset-name>")
    ap.add_argument("--path-in-repo", default="", help="Destination subfolder inside the repo (optional)")
    ap.add_argument("--revision", default="main", help="Branch name to commit to (default: main)")
    ap.add_argument("--private", action="store_true", help="Create as private (default: public)")
    ap.add_argument("--allow", nargs="*", default=None, help="Glob patterns to include (whitelist)")
    ap.add_argument("--ignore", nargs="*", default=None, help="Glob patterns to exclude (blacklist)")
    ap.add_argument("--message", default="Upload dataset folder", help="Commit message")
    ap.add_argument("--create-pr", action="store_true", help="Open a PR instead of committing to the branch")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or run `huggingface-cli login`)")
    ap.add_argument("--dry-run", action="store_true", help="List what would be uploaded and exit")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    if not src.exists() or not src.is_dir():
        raise SystemExit(f"[ERROR] Source folder not found: {src}")

    allow = _norm_patterns(args.allow)
    ignore = _norm_patterns(args.ignore) or [".DS_Store", "**/.DS_Store", "**/_tmp/**", "**/.cache/**"]

    if args.dry_run:
        files = _list_files(src)
        keep = _filter_paths(files, src, allow, ignore)
        print(f"[DRY-RUN] Repo: {args.repo_id}  branch: {args.revision}  path_in_repo: {args.path_in_repo or '/'}")
        print(f"[DRY-RUN] Include: {allow or '[ALL]'}")
        print(f"[DRY-RUN] Exclude: {ignore}")
        print(f"[DRY-RUN] {len(keep)} files will be uploaded:")
        for p in keep[:200]:
            print("  -", p.relative_to(src))
        if len(keep) > 200:
            print(f"  … and {len(keep) - 200} more")
        return

    upload_dir(
        str(src),
        args.repo_id,
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        private=args.private,
        allow_patterns=allow,
        ignore_patterns=ignore,
        commit_message=args.message,
        create_pr=args.create_pr,
        token=args.token,
    )


if __name__ == "__main__":
    main()
