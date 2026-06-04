"""Shared runner logic for the local MedGemma morphology-inspection step.

This module (``ppgfeas.llm.inspect``) is the engine behind the two thin CLI
entry points ``scripts/42_run_medgemma_vmayer.py`` (Signal 2, Mayer-band
windows) and ``scripts/43_run_medgemma_vbeatcrt.py`` (Signal 3, per-beat
exponential decay), which import :data:`VMAYER` / :data:`VBEATCRT` and
:func:`run` from here.

A local multimodal language model (MedGemma 1.5, served on-device by oMLX
through an OpenAI-compatible HTTP endpoint) acts as a second reader against a
prespecified morphology checklist. The model never sees data over a network:
the OpenAI client speaks only to a localhost port.

Privacy and provenance invariants enforced here
------------------------------------------------
* The API key is never logged, printed, or written to any output.
* The input image header text (which carries a patient pseudonym, a
  record/segment id, and a time offset) is never written to any output. Each
  output row is keyed by an opaque exemplar id derived purely from the
  *filename* (``exemplar_07.png`` -> ``vMayer_07`` / ``vBeatCRT_07``), never
  from image content.
* Per row we record model id, prompt SHA-256s, the response-schema SHA-256 (when
  schema-constrained decoding is on), seed, decoding parameters, the run
  timestamp, and (when available) a model-weights fingerprint, so the AI step is
  reproducible per TRIPOD+AI (PMID 38626948) and TRIPOD-LLM (PMID 39779929).

Dependencies are deliberately minimal. ``openai`` is the only required external
package and is imported lazily, so ``--dry-run`` and ``python -m py_compile``
work in an environment without it. ``pandas``+``pyarrow`` are used only to emit
a parquet sidecar; if either is missing, CSV is written and parquet is skipped
with a logged note.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ppgfeas GLOBAL_SEED (see src/ppgfeas/_seed.py). Hardcoded so the AI step is
# reproducible independent of the package import path.
GLOBAL_SEED = 20260426

MODEL_ID = "medgemma-1.5-4b-it-bf16"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
TEMPERATURE = 0.0
# MedGemma 1.5 is a reasoning model: each response opens a thinking block at the
# literal Gemma token ``<unused94>thought`` and closes it at ``<unused95>``
# before emitting the final answer. At 512 tokens generation was truncated
# inside the thinking block and never reached the JSON answer, so every row was
# recorded as ``parse_error``. 2048 leaves room for the reasoning plus the JSON.
MAX_TOKENS = 2048
# MedGemma 1.5's degenerate repetition loops at temperature 0 trace to the
# reasoning block itself: the model fills its <unused94>thought ... <unused95>
# span with a sentence repeated for thousands of characters and never reaches the
# JSON answer, so the row is recorded as a parse_error. oMLX honors a per-request
# chat-template kwarg ``enable_thinking``; disabling thinking suppresses the
# reasoning block entirely and yields a short, direct JSON answer, which removes
# the failure at its source. Thinking is therefore OFF by default. This is a
# chat-template kwarg, not an OpenAI field, so it travels in ``extra_body`` under
# ``chat_template_kwargs`` (see build_extra_body).
DEFAULT_ENABLE_THINKING = False
# A repetition penalty is a blunt instrument for structured output: it penalizes
# the JSON keys and the per-signal enum label strings that the answer MUST repeat
# verbatim, so it can corrupt the very tokens we need. With thinking disabled the
# repetition loops no longer occur, so the penalty is turned OFF by default
# (1.0 == no penalty). The --repetition-penalty flag is retained so it can be
# raised later if a loop ever resurfaces. When set above 1.0 it is a non-standard
# OpenAI field that oMLX/MLX reads from the request body, so it is passed through
# ``extra_body`` rather than as a normal kwarg.
DEFAULT_REPETITION_PENALTY = 1.0
# JSON-schema-constrained decoding is the working fix for MedGemma 1.5's
# thinking-block repetition loop. Sending a strict ``response_format`` of
# ``json_schema`` makes oMLX force the first generated token to ``{``, which
# suppresses the model's thinking block and its repetition loop and returns valid
# JSON deterministically. This is a standard OpenAI field (passed as the
# ``response_format`` kwarg, not in ``extra_body``); a live probe confirmed this
# oMLX build returns HTTP 200 for it. It is therefore ON by default. The
# ``enable_thinking`` and ``repetition_penalty`` controls remain as harmless
# no-op backstops. When constrained decoding is OFF, ``response_format`` is
# omitted (legacy behavior) and the thinking-strip parser still applies.
DEFAULT_CONSTRAINED_JSON = True

# Gemma reasoning-block delimiters. The thinking block is opened by
# ``<unused94>`` (the model then writes the word ``thought`` and its reasoning)
# and closed by ``<unused95>``; the final answer follows the close marker.
THOUGHT_OPEN = "<unused94>"
THOUGHT_CLOSE = "<unused95>"

# Line 1 of each prompt file is "# sha256: <hex-or-placeholder>". This is the
# placeholder written before content is finalized; the runner warns and
# proceeds when it sees it so that --dry-run works pre-stamping.
SHA_PLACEHOLDER = "TBD-stamp-after-content-finalized"
SHA_LINE_PREFIX = "# sha256:"

# Output column order. Stable and explicit so the CSV header is deterministic.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "exemplar_id",
    "signal",
    "model_id",
    "base_url",
    "system_prompt_sha256",
    "user_prompt_sha256",
    "schema_sha256",
    "image_sha256",
    "seed",
    "temperature",
    "max_tokens",
    "run_utc",
    "call",
    "call_valid",
    "confidence",
    "failure_modes",
    "rationale",
    "observed",
    "model_weights_sha256",
    "thinking",
    "raw_response",
)

LOGGER = logging.getLogger("medgemma_inspect")

# Environment variable naming the parent directory that holds the per-signal
# subdirectories of rendered exemplar PNGs. No personal default ships: the
# exemplars are derived from credentialed MIMIC-IV-WDB data the user must obtain
# and render themselves (see data/README.md and scripts/README_medgemma.md). Each
# signal resolves its directory as (in priority order): the ``--input-dir`` CLI
# flag, then the per-signal env var (``SignalConfig.env_var``), then this single
# root joined with the signal ``name`` (``$PPGFEAS_EXEMPLAR_ROOT/<name>``).
EXEMPLAR_ROOT_ENV_VAR = "PPGFEAS_EXEMPLAR_ROOT"


# ---------------------------------------------------------------------------
# Per-signal configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for one signal's inspection run.

    Parameters
    ----------
    name
        Short signal slug used in output filenames and logging
        (``"vmayer"`` or ``"vbeatcrt"``).
    id_prefix
        Prefix for the opaque exemplar id (``"vMayer"`` or ``"vBeatCRT"``).
    system_prompt_name, user_prompt_name
        Prompt filenames within the prompts directory.
    call_enum
        The ordered per-signal enum of acceptable ``call`` values, used both for
        post-hoc ``call_valid`` checks and to constrain the ``call`` property of
        the JSON response schema. ``valid_calls`` (a frozenset over this same
        sequence) is the membership-test view.
    failure_mode_enum
        The ordered per-signal enum of acceptable ``failure_modes`` *item*
        values, used to constrain the array items of the JSON response schema.
        Not enforced post-hoc on the parsed row (failure modes are recorded
        verbatim); it exists so constrained decoding can only emit known labels.
    env_var
        Name of the per-signal environment variable that points at this signal's
        directory of input PNG exemplars (e.g. ``PPGFEAS_VMAYER_EXEMPLAR_DIR``).
        No personal default path is hardcoded: the exemplars are read in place
        from a directory the user provides, and are never copied into this repo.
        See :func:`resolve_input_dir` for the full resolution order.
    """

    name: str
    id_prefix: str
    system_prompt_name: str
    user_prompt_name: str
    call_enum: tuple[str, ...]
    failure_mode_enum: tuple[str, ...]
    env_var: str

    @property
    def valid_calls(self) -> frozenset[str]:
        """Membership-test view of ``call_enum`` (back-compat accessor)."""
        return frozenset(self.call_enum)

    def response_schema(self) -> dict[str, Any]:
        """Build the strict JSON response schema for this signal.

        The schema is a closed object with five required properties:
        ``observed`` (string), ``call`` (string constrained to ``call_enum``),
        ``confidence`` (number), ``failure_modes`` (array of strings each
        constrained to ``failure_mode_enum``), and ``rationale`` (string).
        ``additionalProperties`` is ``false`` so the model cannot emit a free
        thinking field. Constraining the ``call`` property forces oMLX to open
        the response with ``{`` (no thinking block) and return valid JSON.

        Returns
        -------
        dict[str, Any]
            The JSON Schema object (the value passed as the ``schema`` field of
            the ``json_schema`` response_format).
        """
        return {
            "type": "object",
            "properties": {
                "observed": {"type": "string"},
                "call": {"type": "string", "enum": list(self.call_enum)},
                "confidence": {"type": "number"},
                "failure_modes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(self.failure_mode_enum),
                    },
                },
                "rationale": {"type": "string"},
            },
            "required": [
                "observed",
                "call",
                "confidence",
                "failure_modes",
                "rationale",
            ],
            "additionalProperties": False,
        }

    def schema_sha256(self) -> str:
        """Return sha256 over the canonical (sorted-key, compact) schema JSON."""
        canonical = json.dumps(
            self.response_schema(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def response_format(self) -> dict[str, Any]:
        """Return the standard OpenAI ``response_format`` for constrained JSON.

        This is the standard ``{"type": "json_schema", "json_schema": {...}}``
        field (NOT ``extra_body``); a live probe confirmed this oMLX build
        returns HTTP 200 for it.
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"{self.name}_morphology",
                "strict": True,
                "schema": self.response_schema(),
            },
        }


VMAYER = SignalConfig(
    name="vmayer",
    id_prefix="vMayer",
    system_prompt_name="vmayer_inspection_system.txt",
    user_prompt_name="vmayer_inspection_user.txt",
    call_enum=("mayer_peak_present", "no_mayer_peak", "indeterminate"),
    failure_mode_enum=(
        "F1_hr_subharmonic",
        "F2_respiratory_bleedthrough",
        "F3_drift",
        "F4_nonstationary",
    ),
    env_var="PPGFEAS_VMAYER_EXEMPLAR_DIR",
)

VBEATCRT = SignalConfig(
    name="vbeatcrt",
    id_prefix="vBeatCRT",
    system_prompt_name="vbeatcrt_inspection_system.txt",
    user_prompt_name="vbeatcrt_inspection_user.txt",
    call_enum=("exponential_decay_present", "no_exponential_decay", "indeterminate"),
    failure_mode_enum=(
        "F1_cardiac_frequency_fit",
        "F2_flat_baseline_fit",
        "F3_motion_artifact",
        "F4_monotonic_linear_decay",
        "F5_tau_on_bound",
    ),
    env_var="PPGFEAS_VBEATCRT_EXEMPLAR_DIR",
)


# ---------------------------------------------------------------------------
# Repo-root resolution
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """Return the repository root.

    This module lives at ``src/ppgfeas/llm/inspect.py``, so the repository root
    is four directory levels up (``inspect.py`` -> ``llm`` -> ``ppgfeas`` ->
    ``src`` -> repo root). The runner reads the gitignored ``.env`` and the
    ``prompts`` directory relative to this root; both ``--prompts-dir`` and the
    ``OMLX_*`` environment variables let a caller override the defaults when the
    package is installed somewhere other than a source checkout.
    """
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Prompt loading and SHA verification
# ---------------------------------------------------------------------------


def compute_body_sha256(text: str) -> str:
    """Compute sha256 of a prompt's *body* (bytes after the first newline).

    The stamp convention is: line 1 carries ``# sha256: <hex>`` and the hex is
    sha256 of everything from line 2 to the end of file, UTF-8 encoded.

    Parameters
    ----------
    text
        Full file content including the line-1 stamp.

    Returns
    -------
    str
        Hex digest of the body. If the file has no newline (no body), the
        digest of the empty byte string is returned.
    """
    newline_idx = text.find("\n")
    body = "" if newline_idx == -1 else text[newline_idx + 1 :]
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _parse_stamp_line(first_line: str) -> str:
    """Extract the stamp token from a ``# sha256: <token>`` line."""
    stripped = first_line.strip()
    if not stripped.startswith(SHA_LINE_PREFIX):
        raise ValueError(
            f"Prompt line 1 does not start with '{SHA_LINE_PREFIX}': {stripped!r}"
        )
    return stripped[len(SHA_LINE_PREFIX) :].strip()


@dataclass(frozen=True)
class LoadedPrompt:
    """A prompt file loaded and verified against its line-1 stamp."""

    path: Path
    body: str
    body_sha256: str
    stamp: str
    stamp_is_placeholder: bool


def load_prompt(path: Path) -> LoadedPrompt:
    """Load a prompt file and verify its line-1 SHA stamp.

    The returned ``body`` is the text sent to the model (line 2 onward, with a
    trailing newline stripped). The verification rules:

    * placeholder stamp -> log WARNING and proceed (so ``--dry-run`` works
      before stamping);
    * real hex that does not match the recomputed body sha -> raise
      ``ValueError``;
    * real hex that matches -> proceed silently.

    Parameters
    ----------
    path
        Path to the prompt file.

    Returns
    -------
    LoadedPrompt
        Loaded body, its body sha256, and stamp metadata.

    Raises
    ------
    ValueError
        If line 1 is malformed or a non-placeholder stamp does not match.
    """
    text = path.read_text(encoding="utf-8")
    first_line = text.split("\n", 1)[0]
    stamp = _parse_stamp_line(first_line)
    body_sha = compute_body_sha256(text)

    newline_idx = text.find("\n")
    body = "" if newline_idx == -1 else text[newline_idx + 1 :]
    body = body.rstrip("\n")

    is_placeholder = stamp == SHA_PLACEHOLDER
    if is_placeholder:
        LOGGER.warning(
            "Prompt %s carries the placeholder SHA stamp; proceeding unstamped. "
            "Run scripts/stamp_prompt_sha.py before real inference.",
            path.name,
        )
    elif stamp != body_sha:
        raise ValueError(
            f"Prompt SHA mismatch for {path.name}: line-1 stamp {stamp!r} does not "
            f"match recomputed body sha256 {body_sha!r}. The prompt body changed "
            f"after stamping; re-run scripts/stamp_prompt_sha.py."
        )

    return LoadedPrompt(
        path=path,
        body=body,
        body_sha256=body_sha,
        stamp=stamp,
        stamp_is_placeholder=is_placeholder,
    )


# ---------------------------------------------------------------------------
# Config / .env loading
# ---------------------------------------------------------------------------


def parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file of ``KEY=VALUE`` lines.

    Lines that are blank or begin with ``#`` are ignored. Surrounding single or
    double quotes around a value are stripped. This intentionally does not use
    python-dotenv; the parser is small and self-contained.

    Parameters
    ----------
    path
        Path to the ``.env`` file. A missing file yields an empty mapping.

    Returns
    -------
    dict[str, str]
        Parsed key-value pairs.
    """
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            result[key] = value
    return result


@dataclass(frozen=True)
class ServerConfig:
    """Resolved server connection settings."""

    base_url: str
    api_key: str


def load_server_config(repo: Path, base_url_override: str | None = None) -> ServerConfig:
    """Resolve server config from real env vars layered over ``.env``.

    Real process environment variables take precedence over ``.env`` entries.
    The API key is never logged. ``base_url_override`` (a CLI flag) wins over
    both when provided.

    Parameters
    ----------
    repo
        Repository root containing the gitignored ``.env``.
    base_url_override
        Optional CLI-supplied base URL.

    Returns
    -------
    ServerConfig
        Resolved base URL and API key.

    Raises
    ------
    RuntimeError
        If no API key can be resolved.
    """
    import os

    dotenv = parse_dotenv(repo / ".env")

    def resolve(key: str, default: str | None = None) -> str | None:
        if key in os.environ and os.environ[key] != "":
            return os.environ[key]
        if key in dotenv and dotenv[key] != "":
            return dotenv[key]
        return default

    base_url = base_url_override or resolve("OMLX_BASE_URL", DEFAULT_BASE_URL)
    api_key = resolve("OMLX_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No OMLX_API_KEY found in environment or .env. Set it in the gitignored "
            ".env (see .env.example) or export it before running."
        )
    # base_url is guaranteed non-None here because DEFAULT_BASE_URL is the
    # fallback, but assert for the type checker.
    assert base_url is not None
    return ServerConfig(base_url=base_url, api_key=api_key)


# ---------------------------------------------------------------------------
# Model-weights fingerprint
# ---------------------------------------------------------------------------


def load_model_weights_sha(output_dir: Path) -> str | None:
    """Return the composite weights sha from the newest fingerprint file.

    Looks for ``results/medgemma/_model_fingerprint_*.json`` and reads its
    composite sha (tried keys: ``composite_sha256``, ``composite_sha``,
    ``sha256``, ``sha``). Returns ``None`` and logs a non-fatal warning when no
    fingerprint is present or none parses.

    Parameters
    ----------
    output_dir
        The ``results/medgemma`` output directory.

    Returns
    -------
    str | None
        Composite weights sha, or ``None`` if unavailable.
    """
    candidates = sorted(output_dir.glob("_model_fingerprint_*.json"))
    if not candidates:
        LOGGER.warning(
            "No _model_fingerprint_*.json under %s; model_weights_sha256 will be null.",
            output_dir,
        )
        return None
    newest = candidates[-1]
    try:
        payload = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read model fingerprint %s: %s", newest.name, exc)
        return None
    for key in ("composite_sha256", "composite_sha", "sha256", "sha"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            LOGGER.info("Model weights fingerprint loaded from %s.", newest.name)
            return value
    LOGGER.warning(
        "Model fingerprint %s has no recognizable composite sha key.", newest.name
    )
    return None


# ---------------------------------------------------------------------------
# Image discovery and id derivation
# ---------------------------------------------------------------------------


def exemplar_id_from_filename(filename: str, id_prefix: str) -> str:
    """Derive the opaque exemplar id from a filename only.

    ``exemplar_07.png`` with prefix ``vMayer`` -> ``vMayer_07``. The numeric
    suffix is preserved verbatim. When no ``exemplar_<n>`` pattern is found, the
    file stem is used after the prefix. The image *content* is never consulted.

    Parameters
    ----------
    filename
        Base filename (no directory).
    id_prefix
        Signal-specific id prefix.

    Returns
    -------
    str
        Opaque exemplar id, e.g. ``vMayer_07``.
    """
    stem = Path(filename).stem
    match = re.search(r"exemplar[_-]?(\d+)", stem, flags=re.IGNORECASE)
    suffix = match.group(1) if match else stem
    return f"{id_prefix}_{suffix}"


def discover_images(input_dir: Path) -> list[Path]:
    """Return PNG exemplars in ``input_dir`` sorted by filename.

    Parameters
    ----------
    input_dir
        Directory of input PNGs (read in place).

    Returns
    -------
    list[Path]
        PNG paths sorted by filename for deterministic order.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(input_dir.glob("*.png"), key=lambda p: p.name)


def resolve_input_dir(config: SignalConfig, cli_input_dir: Path | None) -> Path:
    """Resolve the exemplar input directory for one signal.

    No personal default path ships with this repo. The directory of rendered
    exemplar PNGs is derived from credentialed MIMIC-IV-WDB data the user must
    obtain and render themselves, so it has to be supplied at run time. The
    resolution order is:

    1. the ``--input-dir`` CLI flag, when given;
    2. the per-signal environment variable named by ``config.env_var``
       (for example ``PPGFEAS_VMAYER_EXEMPLAR_DIR``);
    3. the single-root environment variable ``PPGFEAS_EXEMPLAR_ROOT`` joined with
       the signal slug (``$PPGFEAS_EXEMPLAR_ROOT/<config.name>``).

    Parameters
    ----------
    config
        The signal configuration, which names the per-signal env var.
    cli_input_dir
        The value of the ``--input-dir`` flag, or ``None`` when it was not given.

    Returns
    -------
    Path
        The resolved input directory. Existence is not checked here;
        :func:`discover_images` validates it and raises if it is absent.

    Raises
    ------
    SystemExit
        If neither the CLI flag nor any of the environment variables is set,
        with an actionable message telling the user how to supply the directory.
    """
    import os

    if cli_input_dir is not None:
        return cli_input_dir

    per_signal = os.environ.get(config.env_var, "")
    if per_signal:
        return Path(per_signal)

    root = os.environ.get(EXEMPLAR_ROOT_ENV_VAR, "")
    if root:
        return Path(root) / config.name

    raise SystemExit(
        "No exemplar input directory configured for signal "
        f"'{config.name}'. The rendered exemplar PNGs are derived from "
        "credentialed MIMIC-IV-WDB v0.1.0 data, which is not shipped with this "
        "repo; you must obtain and render your own copy. Then point the runner "
        "at it in one of these ways:\n"
        f"  - pass --input-dir /path/to/{config.name}/exemplars, or\n"
        f"  - export {config.env_var}=/path/to/{config.name}/exemplars, or\n"
        f"  - export {EXEMPLAR_ROOT_ENV_VAR}=/path/to/exemplars "
        f"(the runner then reads ${EXEMPLAR_ROOT_ENV_VAR}/{config.name})."
    )


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_png_data_url(path: Path) -> str:
    """Encode a PNG as a base64 ``data:image/png;base64,...`` URL."""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_last_balanced_object(text: str) -> str | None:
    """Return the *last* balanced top-level ``{...}`` block in ``text``.

    Brace matching is string-aware so braces inside double-quoted JSON strings
    do not throw off the balance count. The reasoning model often emits prose
    (and sometimes an illustrative JSON fragment) before the final answer
    object, so we want the last complete top-level object, not the first.

    Parameters
    ----------
    text
        Text to scan (typically the answer region after the thinking block).

    Returns
    -------
    str | None
        The substring of the last balanced top-level ``{...}`` object, or
        ``None`` if no balanced object is present.
    """
    last: str | None = None
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    last = text[start : idx + 1]
                    start = None
    return last


def split_thinking(raw_text: str) -> tuple[str, str]:
    """Split a reasoning-model response into ``(thinking, answer_region)``.

    MedGemma 1.5 opens a thinking block at the literal token ``<unused94>``
    (after which it writes the word ``thought`` and its reasoning) and closes it
    at ``<unused95>``; the final answer follows the close marker. This function
    returns the captured reasoning text and the region in which to look for the
    JSON answer, following these rules:

    * ``thinking`` is the text between ``<unused94>`` and the first subsequent
      ``<unused95>`` (both present), with a leading ``thought`` word and
      surrounding whitespace stripped; otherwise an empty string.
    * If ``<unused95>`` is present, the answer region is everything *after* the
      *last* ``<unused95>``.
    * If ``<unused94>`` is present but never closed, the response was truncated
      inside the thinking block and there is no answer yet; the whole text is
      returned as the answer region (it will fail to parse, which is the correct
      ``parse_error`` outcome).
    * If neither marker is present, the whole text is the answer region.

    Parameters
    ----------
    raw_text
        The raw assistant message text.

    Returns
    -------
    tuple[str, str]
        ``(thinking, answer_region)``.
    """
    if not raw_text:
        return "", raw_text or ""

    open_idx = raw_text.find(THOUGHT_OPEN)
    has_open = open_idx != -1
    has_close = THOUGHT_CLOSE in raw_text

    # --- thinking capture: between the open marker and the first close ------
    thinking = ""
    if has_open and has_close:
        after_open = raw_text[open_idx + len(THOUGHT_OPEN) :]
        close_in_after = after_open.find(THOUGHT_CLOSE)
        if close_in_after != -1:
            thinking = after_open[:close_in_after]
        else:
            thinking = after_open
        thinking = thinking.strip()
        if thinking.lower().startswith("thought"):
            thinking = thinking[len("thought") :].strip()

    # --- answer region: everything after the LAST close marker --------------
    if has_close:
        last_close = raw_text.rfind(THOUGHT_CLOSE)
        answer_region = raw_text[last_close + len(THOUGHT_CLOSE) :]
    else:
        # No close marker: no thinking block, or a truncated one whose whole
        # text fails to parse (the correct parse_error outcome).
        answer_region = raw_text

    return thinking, answer_region


def parse_model_json(raw_text: str) -> tuple[dict[str, Any] | None, bool, str]:
    """Robustly parse the assistant text as a JSON object.

    Handles the MedGemma 1.5 reasoning block: the thinking text (delimited by
    ``<unused94>`` / ``<unused95>``) is captured separately and the JSON is
    sought only in the answer region after the block. Within the answer region
    the code strips ```json / ``` fences and whitespace, tries ``json.loads`` on
    the cleaned text, and otherwise extracts the *last* balanced top-level
    ``{...}`` object and retries. Never raises on parse failure.

    Parameters
    ----------
    raw_text
        The raw assistant message text.

    Returns
    -------
    tuple[dict | None, bool, str]
        ``(parsed_object, ok, thinking)``. ``ok`` is ``True`` only when a dict
        was recovered. ``thinking`` is the captured reasoning text (empty when
        no closed thinking block was present).
    """
    if raw_text is None:
        return None, False, ""

    thinking, answer_region = split_thinking(raw_text)

    cleaned = answer_region.strip()
    # Strip an opening fence such as ```json or ``` and a closing fence.
    fence = re.match(r"^```[a-zA-Z0-9]*\s*\n?(.*?)\n?```$", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    for candidate in (cleaned, _extract_last_balanced_object(cleaned)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed, True, thinking
    return None, False, thinking


def _coerce_failure_modes(value: Any) -> str:
    """Render a failure-modes field as a semicolon-joined string."""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """Static context shared by every row in a run."""

    config: SignalConfig
    base_url: str
    system_prompt: LoadedPrompt
    user_prompt: LoadedPrompt
    model_weights_sha: str | None
    seed: int = GLOBAL_SEED
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    constrained_json: bool = DEFAULT_CONSTRAINED_JSON


def _now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_row(
    ctx: RunContext,
    exemplar_id: str,
    image_sha: str,
    call: str,
    call_valid: bool,
    confidence: Any,
    failure_modes: str,
    rationale: Any,
    observed: Any,
    raw_response: str,
    thinking: str = "",
) -> dict[str, Any]:
    """Assemble one output row dict in the canonical column order."""
    return {
        "exemplar_id": exemplar_id,
        "signal": ctx.config.name,
        "model_id": MODEL_ID,
        "base_url": ctx.base_url,
        "system_prompt_sha256": ctx.system_prompt.body_sha256,
        "user_prompt_sha256": ctx.user_prompt.body_sha256,
        "schema_sha256": ctx.config.schema_sha256() if ctx.constrained_json else "",
        "image_sha256": image_sha,
        "seed": ctx.seed,
        "temperature": ctx.temperature,
        "max_tokens": ctx.max_tokens,
        "run_utc": _now_utc_iso(),
        "call": call,
        "call_valid": call_valid,
        "confidence": confidence if confidence is not None else "",
        "failure_modes": failure_modes,
        "rationale": rationale if rationale is not None else "",
        "observed": observed if observed is not None else "",
        "model_weights_sha256": ctx.model_weights_sha if ctx.model_weights_sha else "",
        "thinking": thinking,
        "raw_response": raw_response,
    }


def row_from_response(ctx: RunContext, exemplar_id: str, image_sha: str, raw_text: str) -> dict[str, Any]:
    """Parse a model response and build the corresponding output row.

    Records ``call="parse_error"`` (with the raw text preserved) when JSON
    cannot be recovered; flags out-of-enum calls with ``call_valid=False``.
    """
    parsed, ok, thinking = parse_model_json(raw_text)
    if not ok or parsed is None:
        LOGGER.warning("Could not parse JSON for %s; recording parse_error.", exemplar_id)
        return build_row(
            ctx,
            exemplar_id=exemplar_id,
            image_sha=image_sha,
            call="parse_error",
            call_valid=False,
            confidence="",
            failure_modes="",
            rationale="",
            observed="",
            raw_response=raw_text,
            thinking=thinking,
        )

    call = str(parsed.get("call", "")).strip()
    call_valid = call in ctx.config.valid_calls
    if not call_valid:
        LOGGER.warning(
            "Out-of-enum call %r for %s (valid: %s); keeping value, call_valid=False.",
            call,
            exemplar_id,
            sorted(ctx.config.valid_calls),
        )
    return build_row(
        ctx,
        exemplar_id=exemplar_id,
        image_sha=image_sha,
        call=call,
        call_valid=call_valid,
        confidence=parsed.get("confidence"),
        failure_modes=_coerce_failure_modes(parsed.get("failure_modes")),
        rationale=parsed.get("rationale"),
        observed=parsed.get("observed"),
        raw_response=raw_text,
        thinking=thinking,
    )


# ---------------------------------------------------------------------------
# Inference call
# ---------------------------------------------------------------------------


def build_extra_body(ctx: RunContext) -> dict[str, Any]:
    """Build the ``extra_body`` dict of non-standard request fields.

    Two fields ride here, neither of which is part of the OpenAI chat-completions
    schema; oMLX/MLX reads both from the raw request body, so they must travel
    through the SDK's ``extra_body`` passthrough rather than as normal kwargs:

    * ``chat_template_kwargs.enable_thinking`` -- a per-request chat-template
      kwarg oMLX honors. ``False`` (the default) suppresses MedGemma 1.5's
      ``<unused94>thought ... <unused95>`` reasoning block so the model returns a
      short, direct JSON answer instead of looping inside the block.
    * ``repetition_penalty`` -- a sampling field (default 1.0 == no penalty;
      see DEFAULT_REPETITION_PENALTY for why it defaults off).

    Any future non-standard fields should be merged into this same dict.

    Parameters
    ----------
    ctx
        The run context carrying the resolved decoding parameters.

    Returns
    -------
    dict[str, Any]
        The ``extra_body`` payload to attach to the chat-completions request.
    """
    return {
        "chat_template_kwargs": {"enable_thinking": ctx.enable_thinking},
        "repetition_penalty": ctx.repetition_penalty,
    }


def call_model(client: Any, ctx: RunContext, data_url: str) -> str:
    """Send one image to the model and return the raw assistant text.

    Decoding is deterministic: ``temperature=0.0``, ``seed=GLOBAL_SEED``, and
    ``max_tokens=ctx.max_tokens`` (default 2048, large enough to clear any
    reasoning block and reach the JSON answer). When ``ctx.constrained_json`` is
    ``True`` (the default), the standard OpenAI ``response_format`` kwarg carries
    a strict ``json_schema`` for this signal; oMLX forces the first token to
    ``{``, which suppresses the thinking block and its repetition loop and
    returns valid JSON deterministically. When ``False``, ``response_format`` is
    omitted (legacy behavior). ``extra_body`` carries two non-standard fields
    oMLX reads from the request body: ``chat_template_kwargs.enable_thinking``
    (default ``False``) and ``repetition_penalty`` (default 1.0 == off; see
    DEFAULT_REPETITION_PENALTY) -- both retained as harmless no-op backstops. If
    the server rejects the ``seed`` kwarg, the call is retried without it (the
    intended seed is still recorded in the output and logged).

    Parameters
    ----------
    client
        An ``openai.OpenAI`` client bound to the local server.
    ctx
        The run context (prompts, decoding params).
    data_url
        The ``data:image/png;base64,...`` URL for the image.

    Returns
    -------
    str
        The raw assistant message content (empty string if the server returned
        no content).
    """
    messages = [
        {"role": "system", "content": ctx.system_prompt.body},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ctx.user_prompt.body},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    common = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": ctx.temperature,
        "max_tokens": ctx.max_tokens,
        "extra_body": build_extra_body(ctx),
    }
    if ctx.constrained_json:
        # Standard OpenAI field (not extra_body); forces oMLX to open the answer
        # with "{", suppressing the thinking block and repetition loop.
        common["response_format"] = ctx.config.response_format()
    try:
        response = client.chat.completions.create(seed=ctx.seed, **common)
    except TypeError:
        # Some servers reject the seed kwarg outright at the SDK layer.
        LOGGER.warning("Server rejected 'seed' kwarg; retrying without it (intended seed=%d).", ctx.seed)
        response = client.chat.completions.create(**common)
    except Exception as exc:  # noqa: BLE001 - inspect message for seed rejection
        if "seed" in str(exc).lower():
            LOGGER.warning(
                "Server error mentions 'seed'; retrying without it (intended seed=%d).",
                ctx.seed,
            )
            response = client.chat.completions.create(**common)
        else:
            raise
    content = response.choices[0].message.content
    return content if isinstance(content, str) else ""


# ---------------------------------------------------------------------------
# Output writing (CSV always, parquet best-effort)
# ---------------------------------------------------------------------------


# Call values that mark a row as INCOMPLETE: the model never produced a valid,
# parseable answer (a degenerate repetition loop yields ``parse_error``; a thrown
# exception during inference yields ``error``). On resume these rows are
# reprocessed; any other call value is treated as a completed row and skipped.
INCOMPLETE_CALLS: frozenset[str] = frozenset({"parse_error", "error"})


def read_completed_ids(csv_path: Path) -> set[str]:
    """Return the set of ``exemplar_id`` values already present in a CSV.

    This is a content-agnostic read of every id in the file. ``classify_existing
    _rows`` is the resume-aware variant that separates completed from
    incomplete (retry) rows.
    """
    if not csv_path.exists():
        return set()
    completed: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ident = row.get("exemplar_id")
            if ident:
                completed.add(ident)
    return completed


@dataclass
class ExistingRows:
    """Existing-output classification used to drive resume behavior.

    Attributes
    ----------
    complete_ids
        Exemplar ids whose recorded ``call`` is a valid (non-retry) value;
        these are skipped on resume.
    retry_ids
        Exemplar ids whose recorded ``call`` is in ``INCOMPLETE_CALLS``
        (``parse_error`` / ``error``); these are reprocessed on resume.
    keep_rows
        The existing rows to carry forward verbatim, i.e. every row whose id is
        NOT being retried. Rewriting the CSV from these rows (then appending the
        freshly reprocessed rows) overwrites failed rows in place without
        duplicating any exemplar id.
    """

    complete_ids: set[str] = field(default_factory=set)
    retry_ids: set[str] = field(default_factory=set)
    keep_rows: list[dict[str, Any]] = field(default_factory=list)


def classify_existing_rows(csv_path: Path) -> ExistingRows:
    """Classify the rows of an existing output CSV for resume.

    A row is INCOMPLETE (and so a retry candidate) when its ``call`` is in
    ``INCOMPLETE_CALLS`` (``parse_error`` or ``error``); every other row is
    treated as complete and skipped. The last occurrence of an id wins so that a
    previously-overwritten id is classified by its most recent row.

    Parameters
    ----------
    csv_path
        Path to the existing output CSV. A missing file yields an empty
        classification.

    Returns
    -------
    ExistingRows
        Completed ids, retry ids, and the rows to keep verbatim (all non-retry
        rows, in original order).
    """
    result = ExistingRows()
    if not csv_path.exists():
        return result

    # Map id -> its latest call so a re-run that overwrote a row earlier is
    # classified by the most recent value.
    latest_call: dict[str, str] = {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ident = row.get("exemplar_id")
            if not ident:
                continue
            if ident not in rows_by_id:
                order.append(ident)
            rows_by_id[ident] = row
            latest_call[ident] = (row.get("call") or "").strip()

    for ident in order:
        if latest_call.get(ident, "") in INCOMPLETE_CALLS:
            result.retry_ids.add(ident)
        else:
            result.complete_ids.add(ident)
            result.keep_rows.append(rows_by_id[ident])
    return result


def append_rows_csv(csv_path: Path, rows: list[dict[str, Any]], write_header: bool) -> None:
    """Append rows to the output CSV, writing the header when requested."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if csv_path.exists() else "w"
    with csv_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def rewrite_rows_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    """Rewrite the output CSV from scratch with exactly ``rows`` (header + rows).

    Used on resume to drop previously-failed rows before they are reprocessed:
    the kept (good) rows are written first, then the freshly reprocessed rows are
    appended. This overwrites failed rows in place rather than duplicating the
    exemplar id.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def write_parquet_if_possible(csv_path: Path) -> bool:
    """Rewrite the full CSV as a parquet sidecar when pandas+pyarrow import.

    Reads the (now complete) CSV and writes ``<stem>.parquet`` beside it. Logs
    and returns ``False`` if pandas or pyarrow is unavailable, or on any write
    error; the CSV is always authoritative.

    Returns
    -------
    bool
        ``True`` if a parquet file was written, else ``False``.
    """
    try:
        import pandas as pd  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        LOGGER.info(
            "pandas+pyarrow not both available; parquet sidecar skipped (CSV is authoritative)."
        )
        return False
    try:
        frame = pd.read_csv(csv_path)
        parquet_path = csv_path.with_suffix(".parquet")
        frame.to_parquet(parquet_path, index=False)
        LOGGER.info("Wrote parquet sidecar %s (%d rows).", parquet_path.name, len(frame))
        return True
    except Exception as exc:  # noqa: BLE001 - parquet is best-effort
        LOGGER.warning("Parquet write failed (%s); CSV remains authoritative.", exc)
        return False


def write_manifest(
    output_dir: Path,
    ctx: RunContext,
    n_images: int,
    utcstamp: str,
) -> Path:
    """Write the per-run manifest JSON and return its path."""
    manifest = {
        "signal": ctx.config.name,
        "model_id": MODEL_ID,
        "base_url": ctx.base_url,
        "system_prompt_sha256": ctx.system_prompt.body_sha256,
        "user_prompt_sha256": ctx.user_prompt.body_sha256,
        "system_prompt_stamp_is_placeholder": ctx.system_prompt.stamp_is_placeholder,
        "user_prompt_stamp_is_placeholder": ctx.user_prompt.stamp_is_placeholder,
        "seed": ctx.seed,
        "temperature": ctx.temperature,
        "max_tokens": ctx.max_tokens,
        "enable_thinking": ctx.enable_thinking,
        "repetition_penalty": ctx.repetition_penalty,
        "constrained_json": "on" if ctx.constrained_json else "off",
        "schema_sha256": ctx.config.schema_sha256() if ctx.constrained_json else None,
        "n_images": n_images,
        "run_utc": _now_utc_iso(),
        "model_weights_sha256": ctx.model_weights_sha,
    }
    manifest_path = output_dir / f"_run_manifest_{ctx.config.name}_{utcstamp}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


@dataclass
class CliArgs:
    """Parsed CLI arguments for a runner.

    ``input_dir`` is ``None`` when the ``--input-dir`` flag was not given; the
    directory is then resolved from the environment by :func:`resolve_input_dir`
    inside :func:`run`, so that ``--help`` and other no-op invocations never
    require the variable to be set.
    """

    input_dir: Path | None
    output_dir: Path
    prompts_dir: Path
    base_url: str | None
    dry_run: bool
    resume: bool
    limit: int | None
    max_tokens: int
    repetition_penalty: float
    enable_thinking: bool
    constrained_json: bool
    extra: dict[str, Any] = field(default_factory=dict)


def build_parser(config: SignalConfig, description: str) -> argparse.ArgumentParser:
    """Construct the shared argument parser for a signal runner."""
    repo = repo_root()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory of input PNG exemplars (read in place; never copied into "
            "this repo). No default ships. When omitted, the directory is read "
            f"from the {config.env_var} environment variable, or from "
            f"{EXEMPLAR_ROOT_ENV_VAR}/{config.name} if that single root is set. "
            "The exemplars are derived from credentialed MIMIC-IV-WDB data you "
            "must obtain and render yourself."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "results" / "medgemma",
        help="Output directory for CSV, parquet, and run manifest.",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=repo / "prompts",
        help="Directory containing the prompt files.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"Override the oMLX base URL (default from env/.env, else {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify prompts, enumerate images, build one payload, and print a summary. No server call.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore any existing output CSV and restart from scratch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images (after sorting).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=(
            f"Maximum tokens to generate per response (default: {MAX_TOKENS}). "
            "MedGemma 1.5 is a reasoning model: it spends tokens on a thinking "
            "block before the JSON answer, so this must be large enough to clear "
            "that block."
        ),
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help=(
            f"Repetition penalty passed to the server via extra_body (default: "
            f"{DEFAULT_REPETITION_PENALTY}, i.e. off). A penalty above 1.0 harms "
            "structured output because it penalizes the JSON keys and enum label "
            "strings the answer must repeat, so it is off by default; raise it only "
            "if a repetition loop resurfaces. Greedy decoding at temperature 0 is "
            "preserved."
        ),
    )
    parser.add_argument(
        "--thinking",
        dest="enable_thinking",
        action="store_true",
        help=(
            "Enable MedGemma 1.5's reasoning block (sends "
            "extra_body.chat_template_kwargs.enable_thinking=true). Off by default "
            "because the reasoning block is where the degenerate repetition loops "
            "occur at temperature 0; disabling it yields a short, direct JSON answer."
        ),
    )
    parser.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable the reasoning block (the default; enable_thinking=false).",
    )
    parser.add_argument(
        "--constrained-json",
        dest="constrained_json",
        action="store_true",
        help=(
            "Constrain decoding with a strict json_schema response_format (the "
            "default). This standard OpenAI field makes oMLX force the first token "
            "to '{', which suppresses MedGemma 1.5's thinking block and its "
            "repetition loop and returns valid JSON deterministically."
        ),
    )
    parser.add_argument(
        "--no-constrained-json",
        dest="constrained_json",
        action="store_false",
        help=(
            "Disable schema-constrained decoding (legacy behavior): omit "
            "response_format and rely on the thinking-strip parser. "
            "schema_sha256 is written empty/null in this mode."
        ),
    )
    parser.set_defaults(
        resume=True,
        enable_thinking=DEFAULT_ENABLE_THINKING,
        constrained_json=DEFAULT_CONSTRAINED_JSON,
    )
    return parser


def parse_args(config: SignalConfig, description: str, argv: list[str] | None) -> CliArgs:
    """Parse argv into a ``CliArgs`` for the given signal."""
    parser = build_parser(config, description)
    ns = parser.parse_args(argv)
    return CliArgs(
        input_dir=ns.input_dir,
        output_dir=ns.output_dir,
        prompts_dir=ns.prompts_dir,
        base_url=ns.base_url,
        dry_run=ns.dry_run,
        resume=ns.resume,
        limit=ns.limit,
        max_tokens=ns.max_tokens,
        repetition_penalty=ns.repetition_penalty,
        enable_thinking=ns.enable_thinking,
        constrained_json=ns.constrained_json,
    )


def configure_logging() -> None:
    """Configure stdlib logging once with a timestamped format.

    The API key never flows through any log record; only key *names* and other
    non-secret metadata are logged anywhere in this module.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------


def _summarize_dry_run(
    config: SignalConfig,
    base_url: str,
    system_prompt: LoadedPrompt,
    user_prompt: LoadedPrompt,
    weights_sha: str | None,
    images: list[Path],
    id_prefix: str,
    max_tokens: int,
    repetition_penalty: float,
    enable_thinking: bool,
    constrained_json: bool,
) -> None:
    """Print and log a human-readable dry-run preflight summary (no server call)."""
    schema_sha = config.schema_sha256() if constrained_json else None
    lines = [
        "=== MedGemma inspection DRY RUN (no server call) ===",
        f"signal              : {config.name}",
        f"model_id            : {MODEL_ID}",
        f"base_url            : {base_url}",
        f"seed                : {GLOBAL_SEED}",
        f"temperature         : {TEMPERATURE}",
        f"max_tokens          : {max_tokens}",
        f"thinking            : {'on' if enable_thinking else 'off'} "
        f"(enable_thinking={str(enable_thinking).lower()})",
        f"repetition_penalty  : {repetition_penalty}",
        f"constrained_json    : {'on' if constrained_json else 'off'}",
        f"schema_sha256       : {schema_sha if schema_sha else 'null (constrained_json off)'}",
        f"system_prompt       : {system_prompt.path.name}",
        f"  sha256 (body)     : {system_prompt.body_sha256}",
        f"  stamp placeholder : {system_prompt.stamp_is_placeholder}",
        f"user_prompt         : {user_prompt.path.name}",
        f"  sha256 (body)     : {user_prompt.body_sha256}",
        f"  stamp placeholder : {user_prompt.stamp_is_placeholder}",
        f"model_weights_sha256: {weights_sha if weights_sha else 'null (not available)'}",
        f"n_images            : {len(images)}",
    ]
    for img in images:
        exemplar_id = exemplar_id_from_filename(img.name, id_prefix)
        lines.append(f"  {exemplar_id:<14} <- {img.name}")
    # Build (but do NOT send) one payload to confirm shape.
    if images:
        first = images[0]
        data_url = encode_png_data_url(first)
        payload_preview: dict[str, Any] = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": f"<system prompt, {len(system_prompt.body)} chars>"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"<user prompt, {len(user_prompt.body)} chars>"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,<{len(data_url)} chars>"},
                        },
                    ],
                },
            ],
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
            "seed": GLOBAL_SEED,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
                "repetition_penalty": repetition_penalty,
            },
        }
        # The standard response_format rides as a top-level field (NOT extra_body)
        # only when constrained decoding is on; show the full schema so the
        # per-signal call enum and failure-mode enum are visible in the preview.
        if constrained_json:
            payload_preview["response_format"] = config.response_format()
        lines.append("sample payload (NOT sent):")
        lines.append(json.dumps(payload_preview, indent=2))
    lines.append("=== preflight OK; no server call was made ===")
    summary = "\n".join(lines)
    print(summary)
    LOGGER.info("Dry-run preflight complete for signal=%s (%d images).", config.name, len(images))


def run(config: SignalConfig, description: str, argv: list[str] | None) -> int:
    """Execute a full or dry-run inspection pass for one signal.

    Parameters
    ----------
    config
        The signal configuration.
    description
        CLI description string.
    argv
        Argument vector (``None`` -> ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (0 success).
    """
    configure_logging()
    args = parse_args(config, description, argv)
    repo = repo_root()

    # --- prompts -----------------------------------------------------------
    system_prompt = load_prompt(args.prompts_dir / config.system_prompt_name)
    user_prompt = load_prompt(args.prompts_dir / config.user_prompt_name)
    LOGGER.info(
        "Prompts loaded: system sha256=%s user sha256=%s",
        system_prompt.body_sha256,
        user_prompt.body_sha256,
    )

    # --- images ------------------------------------------------------------
    input_dir = resolve_input_dir(config, args.input_dir)
    images = discover_images(input_dir)
    if args.limit is not None:
        images = images[: args.limit]
    LOGGER.info("Discovered %d image(s) in %s.", len(images), input_dir)

    schema_sha = config.schema_sha256() if args.constrained_json else None
    LOGGER.info(
        "Decoding params: temperature=%s seed=%d max_tokens=%d thinking=%s "
        "repetition_penalty=%s constrained_json=%s schema_sha256=%s.",
        TEMPERATURE,
        GLOBAL_SEED,
        args.max_tokens,
        "on" if args.enable_thinking else "off",
        args.repetition_penalty,
        "on" if args.constrained_json else "off",
        schema_sha if schema_sha else "null",
    )

    # --- weights fingerprint ----------------------------------------------
    weights_sha = load_model_weights_sha(args.output_dir)

    # --- dry run -----------------------------------------------------------
    if args.dry_run:
        # Resolve base_url without requiring an API key for the preflight.
        dotenv = parse_dotenv(repo / ".env")
        import os

        base_url = (
            args.base_url
            or os.environ.get("OMLX_BASE_URL")
            or dotenv.get("OMLX_BASE_URL")
            or DEFAULT_BASE_URL
        )
        _summarize_dry_run(
            config,
            base_url,
            system_prompt,
            user_prompt,
            weights_sha,
            images,
            config.id_prefix,
            args.max_tokens,
            args.repetition_penalty,
            args.enable_thinking,
            args.constrained_json,
        )
        return 0

    # --- real inference ----------------------------------------------------
    server = load_server_config(repo, base_url_override=args.base_url)
    LOGGER.info("Connecting to oMLX at %s (model=%s).", server.base_url, MODEL_ID)

    try:
        from openai import OpenAI
    except ImportError as exc:
        LOGGER.error(
            "The 'openai' package is required for inference but is not installed. "
            "Install it with: uv add openai"
        )
        raise SystemExit(2) from exc

    client = OpenAI(base_url=server.base_url, api_key=server.api_key)

    ctx = RunContext(
        config=config,
        base_url=server.base_url,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_weights_sha=weights_sha,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        enable_thinking=args.enable_thinking,
        constrained_json=args.constrained_json,
    )

    utcstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = args.output_dir / f"{config.name}_2026_05_31.csv"

    # Resume classification. On resume an existing row whose call is in
    # INCOMPLETE_CALLS (parse_error/error) is treated as INCOMPLETE and will be
    # reprocessed; rows with any other (valid) call are skipped. To overwrite the
    # failed rows in place without duplicating exemplar ids, the kept (good) rows
    # are rewritten to the CSV up front and the reprocessed rows are appended
    # after. --no-resume discards the whole file and starts fresh.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume:
        existing = classify_existing_rows(csv_path)
        completed = existing.complete_ids
        if csv_path.exists():
            n_retry = len(existing.retry_ids)
            n_skip = len(existing.complete_ids)
            LOGGER.info(
                "Resuming %s: %d row(s) complete (skipped), %d failed row(s) "
                "(parse_error/error) to retry%s.",
                csv_path.name,
                n_skip,
                n_retry,
                (
                    f": {sorted(existing.retry_ids)}"
                    if existing.retry_ids
                    else ""
                ),
            )
            if existing.retry_ids:
                # Drop the failed rows now so reprocessed rows replace them in
                # place; the good rows are preserved verbatim.
                rewrite_rows_csv(csv_path, existing.keep_rows)
    elif csv_path.exists():
        LOGGER.info("--no-resume: existing %s will be overwritten.", csv_path.name)
        csv_path.unlink()

    header_needed = not csv_path.exists()

    processed = 0
    for img in images:
        exemplar_id = exemplar_id_from_filename(img.name, config.id_prefix)
        if exemplar_id in completed:
            LOGGER.info("Skip %s (already complete).", exemplar_id)
            continue
        LOGGER.info("Processing %s <- %s", exemplar_id, img.name)
        try:
            image_sha = sha256_file(img)
            data_url = encode_png_data_url(img)
            raw_text = call_model(client, ctx, data_url)
            row = row_from_response(ctx, exemplar_id, image_sha, raw_text)
            LOGGER.info("  -> call=%s call_valid=%s", row["call"], row["call_valid"])
        except Exception as exc:  # noqa: BLE001 - one failure must not abort the batch
            LOGGER.exception("Inference failed for %s; recording error row.", exemplar_id)
            # image_sha may not have been computed; recompute defensively.
            try:
                image_sha = sha256_file(img)
            except Exception:  # noqa: BLE001
                image_sha = ""
            row = build_row(
                ctx,
                exemplar_id=exemplar_id,
                image_sha=image_sha,
                call="error",
                call_valid=False,
                confidence="",
                failure_modes="",
                rationale="",
                observed="",
                raw_response=f"{type(exc).__name__}: {exc}",
            )
        append_rows_csv(csv_path, [row], write_header=header_needed)
        header_needed = False
        processed += 1

    LOGGER.info("Processed %d image(s); output at %s.", processed, csv_path)
    write_parquet_if_possible(csv_path)
    manifest_path = write_manifest(args.output_dir, ctx, n_images=len(images), utcstamp=utcstamp)
    LOGGER.info("Wrote run manifest %s.", manifest_path.name)
    return 0


# ---------------------------------------------------------------------------
# Inline parser self-test (no server contact)
# ---------------------------------------------------------------------------


def _self_test() -> int:
    """Smoke-test the parser on five synthetic responses (no server contact).

    Run with ``python -m ppgfeas.llm.inspect``; returns 0 on success, 1 on any
    failed assertion. Cases (i)-(iv) cover the thinking-block responses, case
    (v) the pure-JSON response from schema-constrained decoding. Also checks the
    per-signal schema shape. The equivalent assertions live in
    ``tests/test_inspect_parser.py``.
    """
    failures: list[str] = []

    # (i) Closed thinking block followed by a bare JSON answer.
    raw_i = (
        "<unused94>thought\n reasoning... <unused95>\n"
        '{"call": "mayer_peak_present", "confidence": 0.8, '
        '"failure_modes": [], "observed": "x", "rationale": "y"}'
    )
    parsed_i, ok_i, thinking_i = parse_model_json(raw_i)
    if not (ok_i and parsed_i is not None and parsed_i.get("call") == "mayer_peak_present"):
        failures.append(f"(i) expected call=mayer_peak_present, got ok={ok_i} parsed={parsed_i!r}")
    if not thinking_i:
        failures.append(f"(i) expected non-empty thinking, got {thinking_i!r}")
    if thinking_i.lower().startswith("thought"):
        failures.append(f"(i) thinking should have leading 'thought' stripped, got {thinking_i!r}")

    # (ii) Closed thinking block followed by a fenced ```json {...}``` answer.
    raw_ii = (
        "<unused94>thought\n more reasoning... <unused95>\n"
        "Here is my answer:\n```json\n"
        '{"call": "no_mayer_peak", "confidence": 0.4, "failure_modes": ["a", "b"], '
        '"observed": "flat", "rationale": "no peak"}\n```'
    )
    parsed_ii, ok_ii, thinking_ii = parse_model_json(raw_ii)
    if not (ok_ii and parsed_ii is not None and parsed_ii.get("call") == "no_mayer_peak"):
        failures.append(f"(ii) expected call=no_mayer_peak, got ok={ok_ii} parsed={parsed_ii!r}")
    if not thinking_ii:
        failures.append(f"(ii) expected non-empty thinking, got {thinking_ii!r}")

    # (iii) Truncated: thinking opened, never closed, no JSON -> parse_error.
    raw_iii = "<unused94>thought\n reasoning with no close and no json"
    parsed_iii, ok_iii, thinking_iii = parse_model_json(raw_iii)
    if ok_iii or parsed_iii is not None:
        failures.append(f"(iii) expected parse failure, got ok={ok_iii} parsed={parsed_iii!r}")
    if thinking_iii != "":
        failures.append(f"(iii) expected empty thinking (no close marker), got {thinking_iii!r}")

    # (iv) Sanity: a thought block with an illustrative JSON fragment before the
    # close marker plus the real answer after must pick the LAST top-level
    # object (the real answer), not the illustrative one inside the reasoning.
    raw_iv = (
        '<unused94>thought\n I might output {"call": "indeterminate"} but actually '
        '<unused95>\n{"call": "exponential_decay_present", "confidence": 0.9}'
    )
    parsed_iv, ok_iv, _ = parse_model_json(raw_iv)
    iv_ok = ok_iv and parsed_iv is not None and parsed_iv.get("call") == "exponential_decay_present"
    if not iv_ok:
        failures.append(
            f"(iv) expected call=exponential_decay_present, got ok={ok_iv} parsed={parsed_iv!r}"
        )

    # (v) Schema-constrained decoding output: pure JSON, no thinking markers, so
    # the plain json.loads path (no balanced-object fallback, empty thinking)
    # recovers it. Also sanity-check the per-signal schemas constrain the
    # call/failure-mode enums and form the expected canonical sha256.
    raw_v = (
        '{"observed": "broad low-frequency peak near 0.1 Hz", '
        '"call": "mayer_peak_present", "confidence": 0.77, '
        '"failure_modes": ["F3_drift"], "rationale": "peak above noise floor"}'
    )
    parsed_v, ok_v, thinking_v = parse_model_json(raw_v)
    if not (ok_v and parsed_v is not None and parsed_v.get("call") == "mayer_peak_present"):
        failures.append(f"(v) expected call=mayer_peak_present, got ok={ok_v} parsed={parsed_v!r}")
    if thinking_v != "":
        failures.append(f"(v) expected empty thinking (pure JSON, no markers), got {thinking_v!r}")

    vmayer_props = VMAYER.response_schema()["properties"]
    if vmayer_props["call"].get("enum") != list(VMAYER.call_enum):
        failures.append(f"(v) vmayer call enum mismatch: {vmayer_props['call'].get('enum')!r}")
    if vmayer_props["failure_modes"]["items"].get("enum") != list(VMAYER.failure_mode_enum):
        failures.append(
            f"(v) vmayer failure_modes enum mismatch: "
            f"{vmayer_props['failure_modes']['items'].get('enum')!r}"
        )
    vbeatcrt_props = VBEATCRT.response_schema()["properties"]
    if vbeatcrt_props["call"].get("enum") != list(VBEATCRT.call_enum):
        failures.append(f"(v) vbeatcrt call enum mismatch: {vbeatcrt_props['call'].get('enum')!r}")
    if vbeatcrt_props["failure_modes"]["items"].get("enum") != list(VBEATCRT.failure_mode_enum):
        failures.append(
            f"(v) vbeatcrt failure_modes enum mismatch: "
            f"{vbeatcrt_props['failure_modes']['items'].get('enum')!r}"
        )

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SELF-TEST PASSED:")
    print(f"  (i)   call={parsed_i.get('call')!r}  thinking={thinking_i!r}")
    print(f"  (ii)  call={parsed_ii.get('call')!r}  thinking={thinking_ii!r}")
    print(f"  (iii) ok={ok_iii} (parse_error as expected)  thinking={thinking_iii!r}")
    print(f"  (iv)  call={parsed_iv.get('call')!r} (last balanced object chosen)")
    print(f"  (v)   call={parsed_v.get('call')!r} (pure JSON, plain json.loads path)")
    print(f"        vmayer   schema_sha256={VMAYER.schema_sha256()}")
    print(f"        vbeatcrt schema_sha256={VBEATCRT.schema_sha256()}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
