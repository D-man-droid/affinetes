#!/usr/bin/env python3
"""
Build LiveWeb Arena image directly from its Git repository URL.

Demonstrates the URL-based build feature added to build_image_from_env:
no need to clone the repo manually.  The Docker-style fragment syntax
selects a branch and/or subfolder:

    https://github.com/org/repo.git#branch:path/to/env

Usage:
    python examples/liveweb/build_from_repo.py
    python examples/liveweb/build_from_repo.py --tag liveweb-arena:dev
    python examples/liveweb/build_from_repo.py --ref main
    python examples/liveweb/build_from_repo.py \
        --push --registry docker.io/myuser

CLI equivalent:
    afs build https://github.com/AffineFoundation/liveweb-arena.git \
        --tag liveweb-arena:latest
"""

import argparse
import shutil
import sys
from typing import Optional

import affinetes as af


LIVEWEB_REPO = "https://github.com/AffineFoundation/liveweb-arena.git"


def parse_args(argv: list = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LiveWeb Arena image from Git repo URL",
    )
    parser.add_argument(
        "--tag",
        default="liveweb-arena:latest",
        help="Image tag (default: liveweb-arena:latest)",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Git ref (branch/tag) to checkout (default: repo default)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push image to registry after build",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Registry URL for push (e.g. docker.io/myuser)",
    )
    parser.add_argument(
        "--nocache",
        action="store_true",
        help="Build without Docker cache",
    )
    return parser.parse_args(argv)


def check_git_installed() -> None:
    """Ensure git is available on PATH."""
    if shutil.which("git") is None:
        print("Error: git is not installed or not on PATH.", file=sys.stderr)
        sys.exit(1)


def build_repo_url(base: str, ref: Optional[str] = None) -> str:
    """Append #ref fragment to a repo URL when ref is given."""
    if ref:
        return f"{base}#{ref}"
    return base


def main(argv: list = None) -> int:
    args = parse_args(argv)
    check_git_installed()

    repo_url = build_repo_url(LIVEWEB_REPO, ref=args.ref)

    print(f"Building '{args.tag}' from {repo_url}")
    print("(repo will be shallow-cloned into a temp directory)")

    try:
        tag = af.build_image_from_env(
            env_path=repo_url,
            image_tag=args.tag,
            nocache=args.nocache,
            push=args.push,
            registry=args.registry,
        )
    except Exception as exc:
        print(f"\nBuild failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nImage built: {tag}")
    if not args.push:
        print(f"Run with:  afs run {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
