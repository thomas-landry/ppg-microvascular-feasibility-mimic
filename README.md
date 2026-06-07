# PPG-derived microvascular reactivity signals in MIMIC-IV-WDB

A feasibility evaluation of three photoplethysmography-derived microvascular reactivity signals computed on the MIMIC-IV Waveform Database (v0.1.0), with a local multimodal language model used as a second morphology reader.

**Authors:** Thomas C. Landry, MD; Youjin Kim, MD

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20534451.svg)](https://doi.org/10.5281/zenodo.20534451)

The three signals attempted:

- **Signal 1**: a cuff-anchored perfusion-index recovery around routine noninvasive blood pressure cycles, intended to capture a reactive-hyperemia response when the blood pressure cuff happens to share a limb with the pulse oximeter probe.
- **Signal 2**: a Mayer-band power ratio on the 1-Hz perfusion-index time series, intended to index sympathetic vasomotor tone.
- **Signal 3**: a per-beat single-exponential time constant fitted to the diastolic limb of each PPG beat, intended as a heartbeat-by-heartbeat refill-like recovery time.

## What is in this release

This repository currently ships the AI-assisted morphology-inspection step and the supporting package scaffold. The waveform signal-extraction pipeline that turns raw MIMIC-IV-WDB records into the per-signal time series is not yet included; see "Forthcoming" below.

What runs today, with no credentialed data and no model server:

```bash
uv venv && uv sync --extra dev --extra llm
uv run pytest -q                              # the test suite
uv run python -m ppgfeas.llm.inspect          # parser self-test for the LLM engine
uv run python scripts/stamp_prompt_sha.py --check   # verify the prompt SHA stamps
```

What runs once you have rendered your own exemplar PNGs and a local model server (no credentialed data leaves the machine):

```bash
# Preflight only, no server call. Needs a directory of rendered exemplar PNGs
# (see data/README.md for where these come from), but no API key.
uv run python scripts/42_run_medgemma_vmayer.py --dry-run --input-dir /path/to/vmayer/exemplars
uv run python scripts/43_run_medgemma_vbeatcrt.py --dry-run --input-dir /path/to/vbeatcrt/exemplars
```

A full inference run additionally requires an Apple-silicon machine running `oMLX` with MedGemma 1.5 weights pulled and an `OMLX_API_KEY` in a gitignored `.env` (copy `.env.example`). See `scripts/README_medgemma.md` for the full runner usage and `data/README.md` for how the exemplar PNGs are produced from your own credentialed copy of MIMIC-IV-WDB.

## Package layout

```
src/ppgfeas/
  _seed.py        pinned GLOBAL_SEED used everywhere stochastic
  llm/            local MedGemma morphology-inspection engine (ships now)
  signal/         per-signal waveform extraction (forthcoming)
  cohort/         MIMIC-IV-WDB <-> MIMIC-IV linkage (forthcoming)
  analysis/       aggregation, bootstrap, sensitivity sweeps (forthcoming)
```

The local-language-model step is a plain client to a local OpenAI-compatible server; the engine is `ppgfeas.llm.inspect`, driven by the two thin CLIs `scripts/42_run_medgemma_vmayer.py` (Signal 2) and `scripts/43_run_medgemma_vbeatcrt.py` (Signal 3).

## Forthcoming

The `signal/`, `cohort/`, and `analysis/` subpackages and the numbered cohort, extraction, and aggregation scripts read credentialed MIMIC-IV and MIMIC-IV-WDB data. They are being ported from the originating repository only after a per-file data-use-agreement and secret review, and so are not part of this release. The subpackages above are placeholders for that work, and `scripts/README.md` tracks the porting status of the numbered pipeline scripts.

See `data/README.md` for the MIMIC-IV access pointer.

## Manuscript

The manuscript source is at `manuscript/tex/manuscript.tex` and compiles to `manuscript.pdf` with `latexmk -pdf manuscript.tex`; the figures are in `manuscript/figures/`. A preprint is posted on medRxiv: [doi:10.64898/2026.06.03.26354863](https://doi.org/10.64898/2026.06.03.26354863).

## Data availability

This study uses third-party data that are **not redistributed here**. MIMIC-IV (v3.1) and the MIMIC-IV Waveform Database (v0.1.0) are available to credentialed researchers from PhysioNet under a Data Use Agreement:

- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/
- MIMIC-IV-WDB v0.1.0: https://physionet.org/content/mimic4wdb/0.1.0/

## License

The code in this repository is released under the MIT License (see `LICENSE`). The manuscript and figures (`manuscript/`) are released under CC BY 4.0, matching the medRxiv preprint.

## Citation

If you use this work, please cite the preprint and the archived software. A machine-readable `CITATION.cff` is included, so GitHub shows a "Cite this repository" button.

**Preprint**

> Landry TC, Kim Y. An AI-assisted feasibility evaluation of three photoplethysmography-derived microvascular reactivity signals in MIMIC-IV-WDB v0.1.0. medRxiv. 2026. doi:10.64898/2026.06.03.26354863.

**Software (this repository, v1.0.0)**

> Landry TC, Kim Y. ppg-microvascular-feasibility-mimic (v1.0.0). Zenodo. 2026. https://doi.org/10.5281/zenodo.20534451
