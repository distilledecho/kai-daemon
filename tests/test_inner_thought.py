"""Tests for daemon_inner_thought tool (§7a)."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from kai_daemon.state.daemon_self import (
    Fascination,
    FascinationOrigin,
    FascinationStatus,
)
from kai_daemon.tools.inner_thought import (
    PROMPT_A,
    PROMPT_E,
    daemon_inner_thought,
    select_prompt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)


def _fascination(
    topic: str = "emergence",
    status: FascinationStatus = FascinationStatus.ACTIVE,
    created_days_ago: int = 0,
    last_developed_days_ago: int | None = None,
    development_count: int = 0,
) -> Fascination:
    created = _NOW - timedelta(days=created_days_ago)
    last_developed = (
        _NOW - timedelta(days=last_developed_days_ago)
        if last_developed_days_ago is not None
        else None
    )
    return Fascination(
        topic=topic,
        what_daemon_finds_interesting="interesting aspects",
        created=created.isoformat(),
        last_updated=created.isoformat(),
        last_developed=last_developed.isoformat() if last_developed else None,
        development_count=development_count,
        status=status,
        origin=FascinationOrigin.INNER_LIFE_PIPELINE,
    )


def _seeded_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# select_prompt — PROMPT_F threshold
# ---------------------------------------------------------------------------


def test_prompt_f_fires_when_fascination_undeveloped_14_days() -> None:
    """PROMPT_F fires when active fascination has had no development for >= 14 days."""
    f = _fascination(
        topic="recursion", created_days_ago=20, last_developed_days_ago=None
    )
    # last_developed is None → reference date = created (20 days ago)
    result = select_prompt([f], now=_NOW, rng=_seeded_rng())
    assert "recursion" in result
    assert "interesting aspects" in result


def test_prompt_f_fires_when_last_developed_old() -> None:
    """PROMPT_F fires when last_developed is >= 14 days ago."""
    f = _fascination(topic="entropy", last_developed_days_ago=15)
    result = select_prompt([f], now=_NOW, rng=_seeded_rng())
    assert "entropy" in result


def test_prompt_f_does_not_fire_when_recently_developed() -> None:
    """PROMPT_F must not fire when last_developed is < 14 days ago."""
    f = _fascination(topic="entropy", last_developed_days_ago=5)
    # Run many times — PROMPT_F must never appear
    for seed in range(50):
        result = select_prompt([f], now=_NOW, rng=_seeded_rng(seed))
        assert "You've been thinking about entropy for a while" not in result


def test_prompt_f_uses_first_eligible_fascination() -> None:
    """When multiple fasciations qualify for PROMPT_F, the first eligible wins."""
    f1 = _fascination(topic="first", last_developed_days_ago=20)
    f2 = _fascination(topic="second", last_developed_days_ago=20)
    result = select_prompt([f1, f2], now=_NOW, rng=_seeded_rng())
    assert "first" in result
    assert "second" not in result


def test_prompt_f_skips_suspended_fasciations() -> None:
    """PROMPT_F must not trigger for suspended fasciations."""
    f = _fascination(
        topic="old_topic",
        status=FascinationStatus.SUSPENDED,
        last_developed_days_ago=30,
    )
    result = select_prompt([f], now=_NOW, rng=_seeded_rng())
    assert "old_topic" not in result


# ---------------------------------------------------------------------------
# select_prompt — A–E rotation
# ---------------------------------------------------------------------------


def test_prompt_a_in_pool_without_fasciations() -> None:
    """With no fasciations, PROMPT_A must appear among results."""
    seen: set[str] = set()
    for seed in range(30):
        result = select_prompt([], now=_NOW, rng=_seeded_rng(seed))
        seen.add(result)
    assert PROMPT_A in seen


def test_prompt_e_in_pool_without_fasciations() -> None:
    """PROMPT_E must appear among results over many runs."""
    seen: set[str] = set()
    for seed in range(60):
        result = select_prompt([], now=_NOW, rng=_seeded_rng(seed))
        seen.add(result)
    assert PROMPT_E in seen


def test_prompt_d_excluded_with_no_active_fasciations() -> None:
    """PROMPT_D requires an active fascination; no active → never selected."""
    suspended = _fascination(status=FascinationStatus.SUSPENDED)
    for seed in range(50):
        result = select_prompt([suspended], now=_NOW, rng=_seeded_rng(seed))
        assert "What does it make you notice" not in result


def test_prompt_d_eligible_with_active_fascination() -> None:
    """PROMPT_D may appear when active fascination exists (not in PROMPT_F window)."""
    f = _fascination(topic="recursion", last_developed_days_ago=1)
    seen: set[str] = set()
    for seed in range(100):
        result = select_prompt([f], now=_NOW, rng=_seeded_rng(seed))
        seen.add(result)
    assert any("What does it make you notice" in r for r in seen)


def test_custom_weights_can_force_single_prompt() -> None:
    """Setting weight to zero excludes a prompt; weight >> 0 dominates selection."""
    weights = {"A": 100.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
    for seed in range(20):
        result = select_prompt(
            [], now=_NOW, prompt_weights=weights, rng=_seeded_rng(seed)
        )
        assert result == PROMPT_A


def test_zero_all_weights_fallback_to_prompt_a() -> None:
    """If all weights are zero and no active fasciations, fall back to PROMPT_A."""
    weights = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
    result = select_prompt([], now=_NOW, prompt_weights=weights, rng=_seeded_rng())
    assert result == PROMPT_A


# ---------------------------------------------------------------------------
# daemon_inner_thought — integration
# ---------------------------------------------------------------------------


def test_daemon_inner_thought_returns_inference_output() -> None:
    """daemon_inner_thought returns whatever the inference_fn returns."""

    def _fn(prompt: str) -> str:
        return f"thought about: {prompt[:20]}"

    result = daemon_inner_thought([], inference_fn=_fn, now=_NOW)
    assert result.startswith("thought about:")


def test_daemon_inner_thought_passes_prompt_to_inference_fn() -> None:
    """The prompt passed to inference_fn is non-empty and a string."""
    received: list[str] = []

    def _capture(prompt: str) -> str:
        received.append(prompt)
        return "ok"

    daemon_inner_thought([], inference_fn=_capture, now=_NOW)
    assert len(received) == 1
    assert isinstance(received[0], str)
    assert len(received[0]) > 0


def test_daemon_inner_thought_with_prompt_f_eligible_fascination() -> None:
    """Prompt contains fascination data when PROMPT_F-eligible fascination exists."""
    f = _fascination(
        topic="emergence",
        last_developed_days_ago=20,
        created_days_ago=20,
    )
    received: list[str] = []

    def _capture(prompt: str) -> str:
        received.append(prompt)
        return "thought"

    daemon_inner_thought([f], inference_fn=_capture, now=_NOW)
    assert "emergence" in received[0]
