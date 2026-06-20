"""Tests for the reproducibility metadata helper (``evaluation/repro``).

Run:  pytest tests/test_repro.py
"""

from __future__ import annotations

from evaluation import repro


def test_reproducibility_info_has_core_fields():
    info = repro.reproducibility_info(seed=123)
    assert info["seed"] == 123
    assert "python" in info
    # git_commit is either a 40-char hash (inside a repo) or None.
    assert info["git_commit"] is None or len(info["git_commit"]) == 40


def test_git_commit_returns_hash_or_none():
    commit = repro.git_commit()
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)
