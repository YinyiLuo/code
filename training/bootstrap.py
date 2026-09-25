"""Import-path helpers for the self-contained Training BAGEL checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_BAGEL_ROOT = Path(__file__).resolve().parents[1]


def ensure_bagel_importable(bagel_root: str | os.PathLike[str] | None = None) -> Path:
    """Put the local BAGEL root on ``sys.path`` and return the resolved path.

    BAGEL uses top-level imports such as ``modeling`` and ``data``. By default,
    those packages are resolved from the parent of this ``training`` package.
    ``BAGEL_ROOT`` remains available for explicit development overrides.
    """

    configured_root = bagel_root or os.environ.get("BAGEL_ROOT") or DEFAULT_BAGEL_ROOT
    root = Path(configured_root).expanduser().resolve()
    if not (root / "modeling" / "bagel" / "qwen2_navit.py").is_file():
        raise FileNotFoundError(
            f"BAGEL_ROOT={root} does not contain modeling/bagel/qwen2_navit.py"
        )

    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    return root
