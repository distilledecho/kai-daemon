"""Tests for daemon_inner_thought_filter tool (§7b)."""

from __future__ import annotations

import random

import pytest

from kai_daemon.tools.inner_thought_filter import (
    DEFAULT_BYPASS_PROBABILITY,
    FilterVerdict,
    _parse_verdict,
    daemon_inner_thought_filter,
)

# ---------------------------------------------------------------------------
# FilterVerdict
# ---------------------------------------------------------------------------


def test_filter_verdict_values() -> None:
    assert FilterVerdict.KEEP == "keep"
    assert FilterVerdict.BORDERLINE == "borderline"
    assert FilterVerdict.DISCARD == "discard"


# ---------------------------------------------------------------------------
# _parse_verdict
# ---------------------------------------------------------------------------


def test_parse_verdict_keep() -> None:
    assert _parse_verdict("KEEP") == FilterVerdict.KEEP
    assert _parse_verdict("keep") == FilterVerdict.KEEP
    assert _parse_verdict("  Keep  ") == FilterVerdict.KEEP


def test_parse_verdict_borderline() -> None:
    assert _parse_verdict("BORDERLINE") == FilterVerdict.BORDERLINE
    assert _parse_verdict("borderline") == FilterVerdict.BORDERLINE


def test_parse_verdict_discard() -> None:
    assert _parse_verdict("DISCARD") == FilterVerdict.DISCARD
    assert _parse_verdict("discard") == FilterVerdict.DISCARD


def test_parse_verdict_unknown_falls_back_to_borderline() -> None:
    result = _parse_verdict("SOMETHING_ELSE")
    assert result == FilterVerdict.BORDERLINE


# ---------------------------------------------------------------------------
# Bypass valve
# ---------------------------------------------------------------------------


def test_bypass_valve_fires_at_probability_1() -> None:
    """bypass_probability=1.0 always returns KEEP without calling inference_fn."""
    rng = random.Random(0)
    result = daemon_inner_thought_filter(
        "some thought",
        inference_fn=None,
        bypass_probability=1.0,
        rng=rng,
    )
    assert result == FilterVerdict.KEEP


def test_bypass_valve_inference_fn_never_called_when_bypass_fires() -> None:
    """When the bypass valve fires, inference_fn is never called — not just ignored."""
    calls: list[str] = []

    def _fn(prompt: str) -> str:
        calls.append(prompt)
        return "DISCARD"

    rng = random.Random(0)
    result = daemon_inner_thought_filter(
        "some thought",
        inference_fn=_fn,
        bypass_probability=1.0,
        rng=rng,
    )
    assert result == FilterVerdict.KEEP
    assert calls == [], (
        "inference_fn was called despite bypass firing — "
        "bypass must short-circuit before the inference call"
    )


def test_bypass_valve_never_fires_at_probability_0() -> None:
    """bypass_probability=0.0 always calls inference_fn."""
    calls: list[str] = []

    def _fn(prompt: str) -> str:
        calls.append(prompt)
        return "KEEP"

    rng = random.Random(0)
    daemon_inner_thought_filter(
        "thought",
        inference_fn=_fn,
        bypass_probability=0.0,
        rng=rng,
    )
    assert len(calls) == 1


def test_bypass_valve_raises_without_inference_fn_when_not_bypassed() -> None:
    """If bypass does not fire and inference_fn is None, raise ValueError."""
    rng = random.Random(0)
    with pytest.raises(ValueError, match="inference_fn is required"):
        daemon_inner_thought_filter(
            "thought",
            inference_fn=None,
            bypass_probability=0.0,
            rng=rng,
        )


def test_default_bypass_probability() -> None:
    assert DEFAULT_BYPASS_PROBABILITY == 0.12


def test_bypass_rate_is_approximately_correct() -> None:
    """Over 1000 runs with p=0.12, approximately 12 % should bypass."""
    calls = 0

    def _fn(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "KEEP"

    n = 1000
    rng = random.Random(99)
    for _ in range(n):
        daemon_inner_thought_filter(
            "thought",
            inference_fn=_fn,
            bypass_probability=DEFAULT_BYPASS_PROBABILITY,
            rng=rng,
        )
    bypassed = n - calls
    rate = bypassed / n
    # Allow ±5 percentage points
    assert 0.07 <= rate <= 0.17, f"Bypass rate {rate:.2%} outside expected range"


# ---------------------------------------------------------------------------
# Verdict routing
# ---------------------------------------------------------------------------


def test_keep_verdict_returned() -> None:
    rng = random.Random(0)
    result = daemon_inner_thought_filter(
        "great thought",
        inference_fn=lambda _: "KEEP",
        bypass_probability=0.0,
        rng=rng,
    )
    assert result == FilterVerdict.KEEP


def test_discard_verdict_returned() -> None:
    rng = random.Random(0)
    result = daemon_inner_thought_filter(
        "bad thought",
        inference_fn=lambda _: "DISCARD",
        bypass_probability=0.0,
        rng=rng,
    )
    assert result == FilterVerdict.DISCARD


def test_borderline_verdict_returned() -> None:
    rng = random.Random(0)
    result = daemon_inner_thought_filter(
        "unclear thought",
        inference_fn=lambda _: "BORDERLINE",
        bypass_probability=0.0,
        rng=rng,
    )
    assert result == FilterVerdict.BORDERLINE


def test_filter_prompt_contains_raw_output() -> None:
    """The prompt sent to inference_fn must contain the raw thought text."""
    received: list[str] = []

    def _capture(prompt: str) -> str:
        received.append(prompt)
        return "KEEP"

    rng = random.Random(0)
    daemon_inner_thought_filter(
        "my specific thought text",
        inference_fn=_capture,
        bypass_probability=0.0,
        rng=rng,
    )
    assert "my specific thought text" in received[0]
