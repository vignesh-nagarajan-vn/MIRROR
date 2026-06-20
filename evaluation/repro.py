"""Reproducibility metadata for evaluation runs.

Every results JSON should be traceable back to *which code* and *which seed*
produced it — otherwise a number in the paper cannot be regenerated. This module
provides a small, dependency-light record (seed, git commit, library versions)
that the evaluation harnesses stamp into their output.
"""

from __future__ import annotations

import platform
import subprocess


def git_commit() -> str | None:
    """Return the current git commit hash, or None outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git missing
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def reproducibility_info(seed: int) -> dict:
    """A compact, JSON-serialisable provenance record for a run."""
    info: dict = {
        "seed": seed,
        "git_commit": git_commit(),
        "python": platform.python_version(),
    }
    try:
        import numpy as np

        info["numpy"] = np.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:  # pragma: no cover
        pass
    return info
