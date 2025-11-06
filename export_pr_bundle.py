#!/usr/bin/env python3
"""Create a zip archive of the current working tree for easy PR handoff.

The archive contains every tracked file (as reported by `git ls-files`) so
whoever downloads it gets the exact snapshot you have locally without needing
Git.  The output lands in ``dist/`` and is named with the active branch and
short commit hash for quick reference.
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def git_ls_files(repo: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    files = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        files.append(repo / line)
    return files


def git_ref_info(repo: Path) -> tuple[str, str]:
    branch_proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    branch = branch_proc.stdout.strip().replace("/", "-")

    hash_proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    short_hash = hash_proc.stdout.strip()
    return branch, short_hash


def main() -> int:
    repo = Path(__file__).resolve().parent
    tracked_files = git_ls_files(repo)
    branch, short_hash = git_ref_info(repo)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    dist_dir = repo / "dist"
    dist_dir.mkdir(exist_ok=True)
    zip_path = dist_dir / f"alpacatrade-{branch}-{short_hash}-{timestamp}.zip"

    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for file_path in tracked_files:
            arcname = file_path.relative_to(repo)
            zf.write(file_path, arcname)

    print(f"Wrote {zip_path.relative_to(repo)} with {len(tracked_files)} files")
    print("Distribute that zip to run the PR snapshot without cloning Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
