"""Tests for the pinned global seed.

The seed is referenced in the manuscript and in committed result tables, so it
must not drift. See ``src/ppgfeas/_seed.py`` and the project CLAUDE.md.
"""

from __future__ import annotations

import ppgfeas
from ppgfeas._seed import GLOBAL_SEED


def test_global_seed_value() -> None:
    """The pinned seed is the fixed project value and must not change."""
    assert GLOBAL_SEED == 20260426


def test_global_seed_reexported_from_package_root() -> None:
    """``ppgfeas.GLOBAL_SEED`` is the same object as ``ppgfeas._seed.GLOBAL_SEED``."""
    assert ppgfeas.GLOBAL_SEED == GLOBAL_SEED


def test_llm_inspect_seed_matches_package_seed() -> None:
    """The LLM runner's hardcoded seed mirrors the package seed.

    ``ppgfeas.llm.inspect`` hardcodes ``GLOBAL_SEED`` so the AI step is
    reproducible independent of the import path; if the package seed ever
    changes, this guards against the two drifting apart.
    """
    from ppgfeas.llm import inspect

    assert inspect.GLOBAL_SEED == GLOBAL_SEED
