"""Tests for the response parser and id derivation in ``ppgfeas.llm.inspect``.

These exercise the pure, server-free logic of the MedGemma morphology-inspection
engine: the reasoning-aware JSON parser, the filename-only exemplar id
derivation, and the per-signal response schema. None of these contacts a server
or reads credentialed data.
"""

from __future__ import annotations

from ppgfeas.llm.inspect import (
    VBEATCRT,
    VMAYER,
    exemplar_id_from_filename,
    parse_model_json,
)


def test_parse_pure_json_answer() -> None:
    """Schema-constrained output (pure JSON, no thinking markers) parses cleanly."""
    raw = (
        '{"observed": "broad low-frequency peak near 0.1 Hz", '
        '"call": "mayer_peak_present", "confidence": 0.77, '
        '"failure_modes": ["F3_drift"], "rationale": "peak above noise floor"}'
    )
    parsed, ok, thinking = parse_model_json(raw)
    assert ok is True
    assert parsed is not None
    assert parsed["call"] == "mayer_peak_present"
    assert thinking == ""


def test_parse_strips_thinking_block_and_picks_answer() -> None:
    """A closed thinking block is captured; the JSON after it is the answer."""
    raw = (
        "<unused94>thought\n reasoning... <unused95>\n"
        '{"call": "no_mayer_peak", "confidence": 0.4, "failure_modes": [], '
        '"observed": "flat", "rationale": "no peak"}'
    )
    parsed, ok, thinking = parse_model_json(raw)
    assert ok is True
    assert parsed is not None
    assert parsed["call"] == "no_mayer_peak"
    assert thinking != ""
    assert not thinking.lower().startswith("thought")


def test_parse_picks_last_balanced_object() -> None:
    """An illustrative object inside the reasoning is ignored for the real answer."""
    raw = (
        '<unused94>thought\n I might output {"call": "indeterminate"} but actually '
        '<unused95>\n{"call": "exponential_decay_present", "confidence": 0.9}'
    )
    parsed, ok, _ = parse_model_json(raw)
    assert ok is True
    assert parsed is not None
    assert parsed["call"] == "exponential_decay_present"


def test_parse_truncated_thinking_is_parse_failure() -> None:
    """A thinking block that never closes (and has no JSON) fails to parse."""
    raw = "<unused94>thought\n reasoning with no close and no json"
    parsed, ok, thinking = parse_model_json(raw)
    assert ok is False
    assert parsed is None
    assert thinking == ""


def test_parse_none_input() -> None:
    """A ``None`` response is a parse failure, not an exception."""
    parsed, ok, thinking = parse_model_json(None)  # type: ignore[arg-type]
    assert ok is False
    assert parsed is None
    assert thinking == ""


def test_exemplar_id_from_filename_uses_filename_only() -> None:
    """The opaque id is derived from the filename's exemplar number only."""
    assert exemplar_id_from_filename("exemplar_07.png", "vMayer") == "vMayer_07"
    assert exemplar_id_from_filename("exemplar-12.png", "vBeatCRT") == "vBeatCRT_12"


def test_response_schema_constrains_enums() -> None:
    """Each signal's schema constrains the call and failure-mode enums."""
    vmayer_props = VMAYER.response_schema()["properties"]
    assert vmayer_props["call"]["enum"] == list(VMAYER.call_enum)
    assert vmayer_props["failure_modes"]["items"]["enum"] == list(VMAYER.failure_mode_enum)

    vbeatcrt_props = VBEATCRT.response_schema()["properties"]
    assert vbeatcrt_props["call"]["enum"] == list(VBEATCRT.call_enum)
    assert (
        vbeatcrt_props["failure_modes"]["items"]["enum"]
        == list(VBEATCRT.failure_mode_enum)
    )


def test_schema_sha256_is_deterministic() -> None:
    """The schema sha is stable across calls (canonical, sorted-key JSON)."""
    assert VMAYER.schema_sha256() == VMAYER.schema_sha256()
    assert VMAYER.schema_sha256() != VBEATCRT.schema_sha256()
