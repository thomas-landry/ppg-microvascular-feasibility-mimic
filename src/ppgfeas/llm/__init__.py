"""Local-language-model subpackage for ppgfeas.

Intended scope: the AI-assisted morphology-inspection step in which a local
multimodal language model (MedGemma 1.5, served on-device by oMLX through an
OpenAI-compatible localhost endpoint) acts as a second reader for Signal 2
(Mayer-band windows) and Signal 3 (per-beat exponential decay) against a
prespecified morphology checklist. Nothing leaves the machine: the client speaks
only to a localhost port.

The engine lives in :mod:`ppgfeas.llm.inspect`. It is the only subpackage whose
code ships in this release, because it touches derived images and prompts only,
never raw waveforms or clinical notes.
"""

from ppgfeas.llm.inspect import VBEATCRT, VMAYER, SignalConfig, parse_model_json, run

__all__ = ["VBEATCRT", "VMAYER", "SignalConfig", "parse_model_json", "run"]
