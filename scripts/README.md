# Scripts

## What is present now

These scripts exist and run in this repository. They drive the AI-assisted morphology-inspection step; the engine lives in the installed package at `ppgfeas.llm.inspect`. Run any script with `--help` for its CLI.

| Script | Purpose |
|---|---|
| `42_run_medgemma_vmayer.py` | Run the local MedGemma morphology inspection against the Signal 2 (Mayer-band Welch window) exemplars. Thin CLI over `ppgfeas.llm.inspect`. |
| `43_run_medgemma_vbeatcrt.py` | Run the local MedGemma morphology inspection against the Signal 3 (per-beat exponential decay) exemplars. Thin CLI over `ppgfeas.llm.inspect`. |
| `stamp_prompt_sha.py` | Compute and write (or `--check`) the line-1 SHA-256 stamp for the four inspection prompt files. |

The two runners need a directory of rendered exemplar PNGs (derived from your own credentialed MIMIC-IV-WDB copy; see `data/README.md`) and, for real inference, a local oMLX server. `--dry-run` and `stamp_prompt_sha.py --check` need neither a server nor an API key. See `README_medgemma.md` for full usage.

## Forthcoming (not yet in this repository)

The numbered cohort, extraction, and aggregation pipeline reads credentialed MIMIC-IV and MIMIC-IV-WDB data and is being ported from the originating repository only after a per-file data-use-agreement and secret review. None of the scripts below exists here yet; this table tracks the intended sequence and porting source.

| Script | Purpose | Status |
|---|---|---|
| `10_link_wdb_to_icustay.py` | Link MIMIC-IV-WDB records to MIMIC-IV ICU stays. | TO PORT from sibling. |
| `20_extract_cuff_events.py` | Extract candidate cuff cycles from WDB records and write per-record event parquet. | TO PORT from sibling. |
| `30_aggregate_funnel.py` | Aggregate Signal 1 funnel counts across records. | TO PORT from sibling. |
| `31_sensitivity_sweep.py` | Signal 1 sensitivity sweep across nadir depth and recovery thresholds. | TO PORT from sibling. |
| `32_alignment_split_half.py` | Signal 1 subject-clustered split-half calibration of the alignment window. | TO PORT from sibling. |
| `40_medgemma_inference.py` | Shared MedGemma client helpers (now superseded by `ppgfeas.llm.inspect`). | SUPERSEDED. |
| `41_run_medgemma_cuff.py` | Run MedGemma against the Signal 1 candidate gallery (cuff cycles). | TO PORT from sibling. |
| `50_compute_vmayer.py` | Compute Signal 2 (Welch / lf_ratio) per patient. | TO PORT from sibling. |
| `51_compute_vbeatcrt.py` | Compute Signal 3 (per-beat exponential tau) per patient. | TO PORT from sibling. |
| `52_synthetic_validation.py` | Synthetic check: known-truth recovery for Signal 3. | TO PORT from sibling. |
| `60_figures.py` | Build the remaining manuscript figures. | TO PORT from sibling. |
| `compute_model_sha.sh` | One-time MedGemma weight SHA fingerprint. | TO PORT from sibling. |
