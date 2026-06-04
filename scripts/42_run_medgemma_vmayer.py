"""Signal 2 (Mayer-band windows): run the local MedGemma morphology inspection.

Thin CLI entry point. All logic lives in ``ppgfeas.llm.inspect``. The model is
served on-device by oMLX over an OpenAI-compatible localhost endpoint; nothing
leaves the machine. See ``README_medgemma.md`` for usage.

The directory of input exemplar PNGs is not shipped (it is derived from
credentialed MIMIC-IV-WDB data you must obtain and render yourself). Supply it
with ``--input-dir`` or the ``PPGFEAS_VMAYER_EXEMPLAR_DIR`` environment variable
(or ``PPGFEAS_EXEMPLAR_ROOT/vmayer``). See ``README_medgemma.md``.

Examples
--------
Preflight without contacting the server::

    python scripts/42_run_medgemma_vmayer.py --dry-run --input-dir /path/to/vmayer/exemplars

Full run against your rendered exemplar gallery::

    export PPGFEAS_VMAYER_EXEMPLAR_DIR=/path/to/vmayer/exemplars
    python scripts/42_run_medgemma_vmayer.py
"""

from __future__ import annotations

import sys

from ppgfeas.llm.inspect import VMAYER, run


def main(argv: list[str] | None = None) -> int:
    """Entry point for the Signal 2 (vMayer) inspection runner."""
    return run(VMAYER, description=__doc__, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
