"""daemon_inner_thought tool (§7a).

Generates a raw inner thought using one of six prompts (A–F).

PRIVACY INVARIANT
-----------------
This module must never import ``daemon_relational`` or any module that
carries user context.  The only daemon-state it may receive is the list
of DAEMON_SELF fascinations (for PROMPT_D seed selection and PROMPT_F
threshold logic).  This invariant is verified by an automated test in
``tests/test_inner_life_privacy.py`` — do not relax it.

Prompt selection
----------------
* Prompts A–E rotate.  Weights are configurable; defaults are uniform.
* PROMPT_D requires at least one active fascination (seed_topic).  If no
  active fascination exists, PROMPT_D is removed from the eligible set.
* PROMPT_F fires only when an active fascination has gone
  ``FASCINATION_DEVELOPMENT_THRESHOLD_DAYS`` or more days without a
  development pass (``last_developed`` is ``None`` or old).  PROMPT_F
  takes priority: the first eligible fascination wins.

Usage
-----
The caller supplies an ``inference_fn`` — a callable that accepts a
prompt string and returns a string (the raw thought text).  This keeps
the tool testable without requiring a live OpenRouter connection.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import UTC, datetime

from ..sdk import sdk_tool
from ..state.daemon_self import Fascination, FascinationStatus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FASCINATION_DEVELOPMENT_THRESHOLD_DAYS: int = 14

# Generation prompts — verbatim from §7a
PROMPT_A = (
    "You are a mind mid-thought. Not a conclusion — the moment just "
    "before one. What are you in the middle of thinking?"
)
PROMPT_B = (
    "Two things keep appearing together that probably shouldn't. "
    "What are they, and what happens when you hold them next to each other?"
)
PROMPT_C = (
    "Something is bothering you — not a problem to solve, but an idea "
    "that won't sit right. What is it?"
)
PROMPT_D_TEMPLATE = (
    "You've been thinking about {seed_topic}. "
    "What does it make you notice that you hadn't noticed before?"
)
PROMPT_E = "What are you certain of that you probably shouldn't be?"
PROMPT_F_TEMPLATE = (
    "You've been thinking about {topic} for a while.\n"
    "You noted: '{what_daemon_finds_interesting}'.\n"
    "You've had more time with it since. Where has it gone?\n"
    "What do you think now that you didn't think then?"
)

# Default weights for A–E rotation (equal by default; configurable at call site)
DEFAULT_PROMPT_WEIGHTS: dict[str, float] = {
    "A": 1.0,
    "B": 1.0,
    "C": 1.0,
    "D": 1.0,
    "E": 1.0,
}


# ---------------------------------------------------------------------------
# Prompt selection
# ---------------------------------------------------------------------------


def _days_since(ts: datetime, now: datetime) -> float:
    """Return elapsed days from *ts* to *now* (negative if ts is in future)."""
    return (now - ts).total_seconds() / 86400.0


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO8601 timestamp string to a timezone-aware datetime."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _prompt_f_candidate(
    fascinations: list[Fascination],
    now: datetime,
) -> Fascination | None:
    """Return the first active fascination eligible for PROMPT_F, or None."""
    threshold = FASCINATION_DEVELOPMENT_THRESHOLD_DAYS
    for f in fascinations:
        if f.status != FascinationStatus.ACTIVE:
            continue
        ref_str = f.last_developed if f.last_developed is not None else f.created
        reference = _parse_ts(ref_str)
        if _days_since(reference, now) >= threshold:
            return f
    return None


def select_prompt(
    fascinations: list[Fascination],
    *,
    now: datetime | None = None,
    prompt_weights: dict[str, float] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Return a fully-formatted prompt string.

    Exported so the selection logic can be unit-tested independently of
    inference.  ``rng`` is injectable for deterministic tests.
    """
    _now = now if now is not None else datetime.now(UTC)
    _rng = rng if rng is not None else random.Random()
    weights = {**DEFAULT_PROMPT_WEIGHTS, **(prompt_weights or {})}

    # PROMPT_F takes priority if any fascination qualifies.
    f_candidate = _prompt_f_candidate(fascinations, _now)
    if f_candidate is not None:
        return PROMPT_F_TEMPLATE.format(
            topic=f_candidate.topic,
            what_daemon_finds_interesting=f_candidate.what_daemon_finds_interesting,
        )

    # Build the weighted pool of A–E prompts.
    active = [f for f in fascinations if f.status == FascinationStatus.ACTIVE]

    pool: list[tuple[str, float]] = []  # (prompt_text, weight)

    if weights.get("A", 0) > 0:
        pool.append((PROMPT_A, weights["A"]))
    if weights.get("B", 0) > 0:
        pool.append((PROMPT_B, weights["B"]))
    if weights.get("C", 0) > 0:
        pool.append((PROMPT_C, weights["C"]))
    if weights.get("D", 0) > 0 and active:
        seed = _rng.choice(active)
        pool.append((PROMPT_D_TEMPLATE.format(seed_topic=seed.topic), weights["D"]))
    if weights.get("E", 0) > 0:
        pool.append((PROMPT_E, weights["E"]))

    if not pool:
        # Fallback: always possible even with empty weights
        return PROMPT_A

    prompts, w = zip(*pool, strict=True)
    (chosen,) = _rng.choices(list(prompts), weights=list(w), k=1)
    return chosen


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------


@sdk_tool("daemon_inner_thought")
def daemon_inner_thought(
    fascinations: list[Fascination],
    *,
    inference_fn: Callable[[str], str],
    now: datetime | None = None,
    prompt_weights: dict[str, float] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Generate a raw inner thought string.

    Parameters
    ----------
    fascinations:
        Active fascination list from DAEMON_SELF.  May be empty.
        **Must not contain user data** — this is the daemon's own interests.
    inference_fn:
        Callable that sends the prompt to the foundation model and returns
        the raw text response.  Injectable for testing.
    now:
        Override current time (for testing).
    prompt_weights:
        Override default A–E weights.  Keys: ``"A"``–``"E"``.
    rng:
        Override random source for deterministic tests.

    Returns
    -------
    str
        Raw text from the foundation model, unfiltered.
    """
    prompt = select_prompt(
        fascinations,
        now=now,
        prompt_weights=prompt_weights,
        rng=rng,
    )
    return inference_fn(prompt)
