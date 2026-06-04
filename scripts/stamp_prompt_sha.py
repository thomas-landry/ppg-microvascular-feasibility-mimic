"""Stamp the line-1 SHA-256 for the MedGemma prompt files.

Convention: line 1 of each prompt file is ``# sha256: <hex-or-placeholder>``.
The stamped hex is the sha256 of the file *body*, defined as the bytes from
line 2 to end of file (everything after the first newline), UTF-8 encoded. This
utility recomputes that body sha for each prompt and rewrites line 1 in place.

The runner (``ppgfeas.llm.inspect``) recomputes the same body sha at load time
and refuses to run when a non-placeholder stamp does not match, so stamping must
happen whenever a prompt body changes.

By default it stamps the four inspection prompts used by the two runners:

    vmayer_inspection_system.txt
    vmayer_inspection_user.txt
    vbeatcrt_inspection_system.txt
    vbeatcrt_inspection_user.txt

Examples
--------
Stamp all four with a preview::

    python scripts/stamp_prompt_sha.py

Show what would change without writing::

    python scripts/stamp_prompt_sha.py --check
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ppgfeas.llm.inspect import (
    SHA_LINE_PREFIX,
    SHA_PLACEHOLDER,
    compute_body_sha256,
    repo_root,
)

LOGGER = logging.getLogger("stamp_prompt_sha")

DEFAULT_PROMPT_NAMES = (
    "vmayer_inspection_system.txt",
    "vmayer_inspection_user.txt",
    "vbeatcrt_inspection_system.txt",
    "vbeatcrt_inspection_user.txt",
)


def _current_stamp(text: str) -> str:
    """Return the existing line-1 stamp token, or empty string if malformed."""
    first_line = text.split("\n", 1)[0].strip()
    if not first_line.startswith(SHA_LINE_PREFIX):
        return ""
    return first_line[len(SHA_LINE_PREFIX) :].strip()


def stamp_file(path: Path, check_only: bool) -> bool:
    """Stamp one prompt file's line-1 SHA in place.

    Parameters
    ----------
    path
        Path to the prompt file. Line 1 must be a ``# sha256:`` line (which may
        carry the placeholder); the body is everything after the first newline.
    check_only
        When ``True``, report whether the stamp is current but do not write.

    Returns
    -------
    bool
        ``True`` if the file's stamp was already current (or, in write mode, is
        current after stamping); ``False`` if a change was needed in check mode.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If line 1 is not a ``# sha256:`` line.
    """
    text = path.read_text(encoding="utf-8")
    if "\n" not in text:
        raise ValueError(f"{path.name}: file has no body (no newline after line 1).")
    first_line, body = text.split("\n", 1)
    if not first_line.strip().startswith(SHA_LINE_PREFIX):
        raise ValueError(
            f"{path.name}: line 1 is not a '{SHA_LINE_PREFIX}' line: {first_line!r}"
        )

    body_sha = compute_body_sha256(text)
    existing = _current_stamp(text)
    already_current = existing == body_sha

    if check_only:
        if already_current:
            LOGGER.info("OK       %s  (stamp matches body sha256 %s)", path.name, body_sha)
        elif existing == SHA_PLACEHOLDER:
            LOGGER.info("NEEDS    %s  (placeholder -> %s)", path.name, body_sha)
        else:
            LOGGER.info("STALE    %s  (%s -> %s)", path.name, existing or "<empty>", body_sha)
        return already_current

    new_text = f"{SHA_LINE_PREFIX} {body_sha}\n{body}"
    path.write_text(new_text, encoding="utf-8")
    if already_current:
        LOGGER.info("unchanged %s  (sha256 %s)", path.name, body_sha)
    else:
        LOGGER.info("stamped   %s  (%s -> %s)", path.name, existing or "<empty>", body_sha)
    return True


def main(argv: list[str] | None = None) -> int:
    """Stamp (or check) the line-1 SHA for the prompt files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=repo_root() / "prompts",
        help="Directory containing the prompt files.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=list(DEFAULT_PROMPT_NAMES),
        help="Prompt filenames to stamp (default: the four inspection prompts).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report whether each stamp is current without writing.",
    )
    args = parser.parse_args(argv)

    all_current = True
    for name in args.files:
        path = args.prompts_dir / name
        current = stamp_file(path, check_only=args.check)
        all_current = all_current and current

    if args.check and not all_current:
        LOGGER.warning("One or more prompts need stamping; re-run without --check to write.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
