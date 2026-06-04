"""Signal 3 (per-beat exponential decay): run the local MedGemma inspection.

Thin CLI entry point. All logic lives in ``ppgfeas.llm.inspect``. The model is
served on-device by oMLX over an OpenAI-compatible localhost endpoint; nothing
leaves the machine. See ``README_medgemma.md`` for usage.

The directory of input exemplar PNGs is not shipped (it is derived from
credentialed MIMIC-IV-WDB data you must obtain and render yourself). Supply it
with ``--input-dir`` or the ``PPGFEAS_VBEATCRT_EXEMPLAR_DIR`` environment
variable (or ``PPGFEAS_EXEMPLAR_ROOT/vbeatcrt``). See ``README_medgemma.md``.

Examples
--------
Preflight without contacting the server::

    python scripts/43_run_medgemma_vbeatcrt.py --dry-run --input-dir /path/to/vbeatcrt/exemplars

Full run against your rendered exemplar gallery::

    export PPGFEAS_VBEATCRT_EXEMPLAR_DIR=/path/to/vbeatcrt/exemplars
    python scripts/43_run_medgemma_vbeatcrt.py
"""

from __future__ import annotations

import sys

from ppgfeas.llm.inspect import VBEATCRT, run


def main(argv: list[str] | None = None) -> int:
    """Entry point for the Signal 3 (vBeatCRT) inspection runner."""
    return run(VBEATCRT, description=__doc__, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
