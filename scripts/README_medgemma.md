# MedGemma morphology-inspection runner

A self-contained second-reader step. A local multimodal language model
(MedGemma 1.5, served on-device by oMLX through an OpenAI-compatible localhost
endpoint) classifies exemplar images of two PPG-derived signals against a
prespecified morphology checklist. Nothing leaves the machine: the OpenAI
client speaks only to a localhost port.

## Files

| File | Role |
|---|---|
| `src/ppgfeas/llm/inspect.py` | Shared runner engine (prompt loading, config, inference, parsing, output). Installed as the `ppgfeas.llm.inspect` module. |
| `scripts/42_run_medgemma_vmayer.py` | Signal 2 (Mayer-band Welch windows). Thin CLI over `ppgfeas.llm.inspect`. |
| `scripts/43_run_medgemma_vbeatcrt.py` | Signal 3 (per-beat exponential decay). Thin CLI over `ppgfeas.llm.inspect`. |
| `scripts/stamp_prompt_sha.py` | Compute and write the line-1 SHA-256 stamp for the four prompt files. |

## Privacy and provenance

- The API key is never logged, printed, or written to any output.
- The input image header text (patient pseudonym, record/segment id, time
  offset) is never written to any output. Each output row is keyed by an opaque
  exemplar id derived from the **filename only**
  (`exemplar_07.png` -> `vMayer_07` / `vBeatCRT_07`), never from image content.
- Each row records model id, prompt SHA-256s, seed, decoding parameters, run
  timestamp, and (when available) a model-weights fingerprint, per TRIPOD+AI
  (PMID 38626948) and TRIPOD-LLM (PMID 39779929).

## Environment variables

The two `OMLX_*` variables are loaded from the gitignored `.env` at the repo
root (parsed directly, no python-dotenv); real process environment variables
override `.env`. The exemplar-directory variables are read from the process
environment only.

| Variable | Required | Default |
|---|---|---|
| `OMLX_API_KEY` | yes (for real inference) | none |
| `OMLX_BASE_URL` | no | `http://127.0.0.1:8000/v1` |
| `PPGFEAS_VMAYER_EXEMPLAR_DIR` | yes for vMayer, unless `--input-dir` or `PPGFEAS_EXEMPLAR_ROOT` is given | none (no default ships) |
| `PPGFEAS_VBEATCRT_EXEMPLAR_DIR` | yes for vBeatCRT, unless `--input-dir` or `PPGFEAS_EXEMPLAR_ROOT` is given | none (no default ships) |
| `PPGFEAS_EXEMPLAR_ROOT` | optional single root for both signals | none |

### Where the exemplar PNGs come from

No exemplar-directory default ships with this repo. The input PNGs are rendered
from credentialed **MIMIC-IV Waveform Database v0.1.0** data, which you must
obtain yourself under the PhysioNet credentialed-access data use agreement (see
`data/README.md`). Generate them with the pipeline's rendering step, then point
each runner at your own copy. The directory is resolved, in priority order:

1. the `--input-dir PATH` flag, when given;
2. the per-signal variable (`PPGFEAS_VMAYER_EXEMPLAR_DIR` /
   `PPGFEAS_VBEATCRT_EXEMPLAR_DIR`);
3. `PPGFEAS_EXEMPLAR_ROOT/<signal>`, i.e. `$PPGFEAS_EXEMPLAR_ROOT/vmayer` or
   `$PPGFEAS_EXEMPLAR_ROOT/vbeatcrt`.

If none of these is set, the runner exits with an actionable message naming all
three options. The PNGs are read in place and never copied into this repo.

Model id is fixed: `medgemma-1.5-4b-it-bf16`. Decoding is deterministic:
`temperature=0.0`, `max_tokens=2048`, `seed=20260426` (the ppgfeas
`GLOBAL_SEED`). If the server rejects the `seed` kwarg, the call is retried
without it and the intended seed is still recorded and logged.

## Dependencies

`openai` is the only required external package and is imported lazily, so the
`--dry-run` preflight and `python -m py_compile` work without it.

```bash
uv add openai
```

A parquet sidecar is written only if both `pandas` and `pyarrow` import;
otherwise CSV is the sole output and parquet is skipped with a logged note. To
enable parquet:

```bash
uv add pyarrow      # pandas is already a project dependency
```

## Prompts and the SHA stamp

Each prompt file's line 1 is `# sha256: <hex-or-placeholder>`. The stamped hex
is the sha256 of the file **body** (bytes from line 2 to end of file, UTF-8).

Stamp all four prompts in place:

```bash
python scripts/stamp_prompt_sha.py
```

Check stamps without writing:

```bash
python scripts/stamp_prompt_sha.py --check
```

At load time the runner recomputes the body sha and:
- placeholder stamp (`TBD-stamp-after-content-finalized`): logs a WARNING and
  proceeds, so `--dry-run` works before stamping;
- real hex that does not match: raises a clear error (re-stamp after editing);
- real hex that matches: proceeds.

## Commands

First tell each runner where your rendered exemplar PNGs are (see "Where the
exemplar PNGs come from" above). The examples below use the per-signal
environment variables; `--input-dir PATH` works in place of either.

### Preflight (no server call)

```bash
export PPGFEAS_VMAYER_EXEMPLAR_DIR=/path/to/vmayer/exemplars
export PPGFEAS_VBEATCRT_EXEMPLAR_DIR=/path/to/vbeatcrt/exemplars
python scripts/42_run_medgemma_vmayer.py --dry-run
python scripts/43_run_medgemma_vbeatcrt.py --dry-run
```

The preflight verifies the prompts, enumerates the input images, prints the
filename-to-exemplar-id mapping, builds (but does not send) one request
payload, and exits. No API key is needed, but an input directory must still be
resolvable (via the flag or an environment variable).

### Full run

```bash
export PPGFEAS_VMAYER_EXEMPLAR_DIR=/path/to/vmayer/exemplars
export PPGFEAS_VBEATCRT_EXEMPLAR_DIR=/path/to/vbeatcrt/exemplars
python scripts/42_run_medgemma_vmayer.py
python scripts/43_run_medgemma_vbeatcrt.py
```

Or pass the directory explicitly per run:

```bash
python scripts/42_run_medgemma_vmayer.py --input-dir /path/to/vmayer/exemplars
```

Useful flags (both runners):

| Flag | Effect |
|---|---|
| `--input-dir PATH` | Directory of input exemplar PNGs (read in place; no default ships). Takes priority over the environment variables. Required unless `PPGFEAS_*_EXEMPLAR_DIR` or `PPGFEAS_EXEMPLAR_ROOT` is set. |
| `--output-dir PATH` | Override the output directory (default `results/medgemma`). |
| `--prompts-dir PATH` | Override the prompts directory. |
| `--base-url URL` | Override the oMLX base URL. |
| `--limit N` | Process only the first N images (after sorting by filename). |
| `--no-resume` | Ignore any existing output CSV and restart from scratch. |
| `--dry-run` | Preflight only; no server call. |

The input PNGs are read in place and never copied into this repo.

## Outputs (under `results/medgemma/`)

- `vmayer_2026_05_31.csv` (+ `.parquet` when possible); analogous
  `vbeatcrt_2026_05_31.csv`.
- A per-run manifest `_run_manifest_<signal>_<utcstamp>.json`.

CSV columns: `exemplar_id, signal, model_id, base_url, system_prompt_sha256,
user_prompt_sha256, image_sha256, seed, temperature, max_tokens, run_utc, call,
call_valid, confidence, failure_modes, rationale, observed,
model_weights_sha256, raw_response`.

The runs are checkpoint-resumable, keyed by `exemplar_id`: on start the runner
reads any existing output CSV and skips completed ids. `--no-resume` restarts.
Each image is processed inside its own try/except, so one failure does not abort
the batch; a failed image writes a row with `call="error"` and the exception
text in `raw_response`. A response that cannot be parsed as JSON writes
`call="parse_error"` with the raw text in `raw_response`. An out-of-enum `call`
is kept but flagged `call_valid=False`.

## Model-weights fingerprint

`model_weights_sha256` is read from the newest
`results/medgemma/_model_fingerprint_*.json` (composite sha) if present; if
absent, it is left null and a non-fatal warning is logged.
