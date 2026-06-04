# Data

This repository ships code, not data. PhysioNet credentialed access and an executed Data Use Agreement are required to obtain MIMIC-IV and MIMIC-IV-WDB v0.1.0.

## Required datasets

- **MIMIC-IV** (clinical layer): <https://physionet.org/content/mimiciv/>. We used v3.1.
- **MIMIC-IV Waveform Database (MIMIC-IV-WDB)** (bedside-monitor waveforms): <https://physionet.org/content/mimic4wdb/0.1.0/>.

Both require:

1. A free PhysioNet account.
2. Completion of one of the accepted human-subjects training courses (CITI Data or Specimens Only Research is the most common).
3. An executed PhysioNet Data Use Agreement.
4. Approval as a credentialed user.

Detailed instructions: <https://physionet.org/about/credentialing/>.

## After credentialed access

Place the downloaded `mimic-iv-3.1/` and `mimic4wdb/0.1.0/` trees anywhere on disk. The cohort, linkage, and signal-extraction scripts that consume these roots (via `--mimic-iv-root` / `--wdb-root` flags or the `MIMIC_IV_ROOT` / `WDB_ROOT` environment variables) are not yet part of this release; they are being ported only after a per-file data-use-agreement and secret review. See `scripts/README.md` for the intended pipeline and its porting status. The example below shows the planned invocation:

```bash
export MIMIC_IV_ROOT=/path/to/mimic-iv-3.1
export WDB_ROOT=/path/to/mimic4wdb/0.1.0
# forthcoming: uv run python scripts/10_link_wdb_to_icustay.py
```

## Exemplar PNGs for the MedGemma morphology-inspection step

The AI-assisted second-reader step (`scripts/42_run_medgemma_vmayer.py` and `scripts/43_run_medgemma_vbeatcrt.py`) reads a directory of rendered exemplar PNGs, one set per signal. These PNGs are **derived artifacts of credentialed MIMIC-IV-WDB v0.1.0 data and are not shipped here.** No default path is hardcoded in the code. You generate them yourself from your own credentialed copy with the pipeline's signal-extraction and rendering steps, then point each runner at the resulting directory.

Tell the runners where the PNGs are, in priority order: the `--input-dir PATH` flag, the per-signal variables `PPGFEAS_VMAYER_EXEMPLAR_DIR` and `PPGFEAS_VBEATCRT_EXEMPLAR_DIR`, or a single root `PPGFEAS_EXEMPLAR_ROOT` (the runner then reads `$PPGFEAS_EXEMPLAR_ROOT/vmayer` and `$PPGFEAS_EXEMPLAR_ROOT/vbeatcrt`). If none is set, the runner exits with an actionable message. The PNGs are read in place and never copied into this repo. See `scripts/README_medgemma.md` for full usage.

```bash
export PPGFEAS_VMAYER_EXEMPLAR_DIR=/path/to/vmayer/exemplars
export PPGFEAS_VBEATCRT_EXEMPLAR_DIR=/path/to/vbeatcrt/exemplars
uv run python scripts/42_run_medgemma_vmayer.py --dry-run
```

## What you will NOT find here

- No raw waveform records (.dat, .hea).
- No row-level clinical tables (admissions.csv, patients.csv, chartevents.csv).
- No derived per-patient or per-event tables that contain MIMIC subject IDs alongside row-level signal data.
- No clinical note text.
- No model weights.

What this repository contains is code, prompts, frozen evaluation checklists, the manuscript source, and a manifest of derived artifacts with SHA-256 hashes. Plots in `manuscript/figures/` show derived signal segments only and carry no identifying information beyond MIMIC's pseudonymized subject IDs.

## DUA compliance

The PhysioNet DUA forbids redistribution of raw MIMIC data, attempts at re-identification, and transmission of credentialed-access content to third-party services that may retain or train on it. PhysioNet maintains an explicit policy on generative-AI use: <https://physionet.org/news/post/gpt-responsible-use>. The AI-assisted morphology inspection in this work uses MedGemma 1.5 served locally on the project workstation; no MIMIC-IV-WDB content is transmitted off the workstation.
