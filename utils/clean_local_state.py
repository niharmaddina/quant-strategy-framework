"""
Utility to clear local, non-source state.

This intentionally avoids touching git history and avoids deleting your venv by default.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_DIRS = [
    "output",
    "logs",
    "data_cache",
    ".pytest_cache",
]


def _rm_tree(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        path.unlink()
        return True
    shutil.rmtree(path)
    return True


def clear_local_state(project_root: Path, include_venv: bool = False) -> int:
    removed = 0

    for rel in DEFAULT_DIRS:
        p = project_root / rel
        if _rm_tree(p):
            removed += 1

    # Remove python bytecode/caches
    for cache_dir in project_root.rglob("__pycache__"):
        if ".venv" in cache_dir.parts or "venv" in cache_dir.parts:
            continue
        if _rm_tree(cache_dir):
            removed += 1

    for pyc in project_root.rglob("*.pyc"):
        if "venv" in pyc.parts or ".venv" in pyc.parts:
            continue
        try:
            pyc.unlink()
            removed += 1
        except FileNotFoundError:
            pass

    if include_venv:
        for rel in ["venv", ".venv"]:
            p = project_root / rel
            if _rm_tree(p):
                removed += 1

    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear local state (output/logs/cache/pyc).")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root directory (default: auto-detected).",
    )
    parser.add_argument(
        "--include-venv",
        action="store_true",
        help="Also delete ./venv and ./.venv (DANGEROUS).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    if not root.exists():
        raise SystemExit(f"Project root not found: {root}")

    removed = clear_local_state(root, include_venv=args.include_venv)
    print(f"Removed {removed} paths/items.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


