from __future__ import annotations

import json

import pytest

from contextbank.compilation.cards import _validated_generated_card_payload
from contextbank.config.settings import AISettings


def _required() -> dict:
    return {"title": "T", "one_line_summary": "O", "summary": "S", "why_useful": "W"}


def test_coerces_scalar_list_field_to_list() -> None:
    # The real gemma3:4b failure: utility came back as a bare string, not a list.
    out = _validated_generated_card_payload(json.dumps({**_required(), "utility": "build"}))
    assert out["utility"] == ["build"]


def test_coerces_list_required_field_to_string() -> None:
    out = _validated_generated_card_payload(json.dumps({**_required(), "title": ["A", "B"]}))
    assert isinstance(out["title"], str) and "A" in out["title"]


def test_stringifies_and_drops_none_items_in_list_field() -> None:
    out = _validated_generated_card_payload(json.dumps({**_required(), "topics": ["x", None, 7]}))
    assert out["topics"] == ["x", "7"]


def test_coerces_numeric_string_confidence() -> None:
    out = _validated_generated_card_payload(json.dumps({**_required(), "confidence": "0.8"}))
    assert out["confidence"] == 0.8


def test_still_rejects_unrecoverable_missing_required_field() -> None:
    # A genuinely absent required field cannot be fabricated — must still fail.
    bad = {"one_line_summary": "O", "summary": "S", "why_useful": "W"}
    with pytest.raises(ValueError):
        _validated_generated_card_payload(json.dumps(bad))


def test_generation_timeout_is_configurable() -> None:
    assert AISettings().generation_timeout == 60.0
    assert AISettings(generation_timeout=180.0).generation_timeout == 180.0
