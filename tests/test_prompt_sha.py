"""Tests for the prompt-body SHA-256 helper used by the SHA-stamp utility.

``compute_body_sha256`` defines the line-1 stamp convention: line 1 of a prompt
file is ``# sha256: <hex>`` and the hex is the sha256 of the file *body* (every
byte after the first newline), UTF-8 encoded. The stamp utility
(``scripts/stamp_prompt_sha.py``) writes this value and the runner
(``ppgfeas.llm.inspect``) recomputes and verifies it at load time, so the helper
must be stable.
"""

from __future__ import annotations

import hashlib

from ppgfeas.llm.inspect import compute_body_sha256


def test_body_sha_is_over_bytes_after_first_newline() -> None:
    """The digest covers only the body (everything after the first newline)."""
    body = "checklist line one\nchecklist line two\n"
    text = f"# sha256: ignored\n{body}"
    expected = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert compute_body_sha256(text) == expected


def test_body_sha_ignores_the_line_1_stamp_contents() -> None:
    """Changing only the line-1 stamp does not change the body digest."""
    body = "the morphology checklist body\n"
    text_a = f"# sha256: TBD-stamp-after-content-finalized\n{body}"
    text_b = f"# sha256: 0000000000000000000000000000000000000000000000000000000000000000\n{body}"
    assert compute_body_sha256(text_a) == compute_body_sha256(text_b)


def test_body_sha_of_empty_body_is_sha_of_empty_string() -> None:
    """A file with no newline (no body) hashes the empty byte string."""
    empty_digest = hashlib.sha256(b"").hexdigest()
    assert compute_body_sha256("# sha256: x") == empty_digest
    assert compute_body_sha256("# sha256: x\n") == empty_digest
