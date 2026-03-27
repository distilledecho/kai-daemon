"""inner_life_push_evaluation workflow (§2F).

Evaluates whether an inner-life insight should be surfaced to the user.

The 7-day push frequency ceiling is checked **before** the inference
prompt runs — enforced in code, not by prompt.  If the ceiling has not
been cleared the workflow returns immediately with ``outcome=SILENCE``.

When the ceiling is clear the four-question evaluation prompt from §2F
is run.  Possible outcomes:

- ``SILENCE``      — default; no state change
- ``HOLDING_ITEM`` — insight written to holding store; not yet surfaced
- ``PUSH``         — insight written to holding store with ``urgency=high``;
  push event recorded in push history

Priority: 8 / Preemption: restart / Model: presentation
Condition: push_signal_written (only triggered if pollination emitted a signal)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from ..state._types import EpistemicOrigin
from ..state.holding import (
    HoldingItem,
    HoldingStore,
    HoldingType,
    RegisterNeeded,
    Urgency,
)
from ..state.push_history import PUSH_CEILING_DAYS, PushHistoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt (§2F verbatim)
# ---------------------------------------------------------------------------

_PUSH_EVALUATION_PROMPT_TEMPLATE = """\
You just noticed something. Before deciding whether to surface it, ask:

1. Has this person been carrying this unresolved question for a meaningful
   amount of time? Days don't count. Months do.
2. Does what you noticed genuinely reframe it — or does it add another
   angle to something already well-examined?
3. Is this the kind of thing they would want to hear now, or better when
   they bring it up themselves?
4. Have you pushed recently? If yes, default to silence.

If all four are clear and strong: write a holding item, not a push.
If 1 and 2 are yes and 3 is genuinely uncertain: consider a push.
If the connection spans multiple threads and reveals something structural
neither of you has named: this may be the rare case for a push.

Default is silence. The push is for when silence would be a failure
of the relationship, not an exercise of restraint.

Context:
Thought: {thought}
Fascination: {fascination}
Threads touched: {thread_count}

Reply with exactly one of:
SILENCE
HOLDING_ITEM: <brief description of what to hold>
PUSH: <brief description of what to surface>
"""

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class PushOutcome(StrEnum):
    """Outcome of the push evaluation."""

    SILENCE = "silence"
    HOLDING_ITEM = "holding_item"
    PUSH = "push"


@dataclass
class PushEvaluationResult:
    """Result of the inner_life_push_evaluation workflow."""

    outcome: PushOutcome
    skipped_ceiling: bool
    """``True`` if the workflow returned early due to the 7-day ceiling."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_outcome(response: str) -> tuple[PushOutcome, str]:
    """Parse the push evaluation response.

    Returns ``(outcome, description)``.  Description is empty for SILENCE.
    Falls back to SILENCE on unrecognised input.
    """
    stripped = response.strip()
    upper = stripped.upper()

    if upper.startswith("PUSH:"):
        desc = stripped[len("PUSH:") :].strip()
        return PushOutcome.PUSH, desc
    if upper.startswith("HOLDING_ITEM:"):
        desc = stripped[len("HOLDING_ITEM:") :].strip()
        return PushOutcome.HOLDING_ITEM, desc
    if upper == "SILENCE":
        return PushOutcome.SILENCE, ""

    # Handle multi-word PUSH / HOLDING_ITEM without colon
    first_token = upper.split()[0] if upper else ""
    if first_token == "PUSH":
        return PushOutcome.PUSH, stripped[len("PUSH") :].strip(": ")
    if first_token == "HOLDING_ITEM":
        return PushOutcome.HOLDING_ITEM, stripped[len("HOLDING_ITEM") :].strip(": ")

    logger.warning(
        "Unrecognised push evaluation response %r — defaulting to SILENCE",
        stripped,
    )
    return PushOutcome.SILENCE, ""


# ---------------------------------------------------------------------------
# Public workflow function
# ---------------------------------------------------------------------------


def inner_life_push_evaluation(
    thought_content: str,
    fascination_topic: str | None,
    threads_pollinated: list[str],
    *,
    push_history: PushHistoryStore,
    holding_store: HoldingStore,
    inference_fn: Callable[[str], str],
    now: datetime | None = None,
    push_ceiling_days: int = PUSH_CEILING_DAYS,
) -> PushEvaluationResult:
    """Evaluate whether an inner-life insight should be surfaced.

    Parameters
    ----------
    thought_content:
        The original inner thought text.
    fascination_topic:
        The fascination topic involved, or ``None``.
    threads_pollinated:
        List of thread IDs that were pollinated upstream.
    push_history:
        Push history store for ceiling check and recording.
    holding_store:
        Holding store to write items when not silenced.
    inference_fn:
        Callable that sends a prompt to the presentation model and returns
        the raw text response.  Injectable for testing.
    now:
        Override current time (for testing).
    push_ceiling_days:
        Ceiling window in days (default 7).

    Returns
    -------
    PushEvaluationResult
        Outcome and whether the ceiling gate fired.
    """
    _now = now if now is not None else datetime.now(UTC)

    # Gate 1: 7-day ceiling — enforced in code before any inference
    if push_history.within_ceiling(days=push_ceiling_days, now=_now):
        logger.info(
            "Push evaluation skipped — within %d-day ceiling", push_ceiling_days
        )
        return PushEvaluationResult(outcome=PushOutcome.SILENCE, skipped_ceiling=True)

    # Gate 2: inference evaluation using §2F prompt
    prompt = _PUSH_EVALUATION_PROMPT_TEMPLATE.format(
        thought=thought_content,
        fascination=fascination_topic or "(none)",
        thread_count=len(threads_pollinated),
    )
    response = inference_fn(prompt)
    outcome, description = _parse_outcome(response)

    if outcome == PushOutcome.SILENCE:
        return PushEvaluationResult(outcome=PushOutcome.SILENCE, skipped_ceiling=False)

    # Determine urgency based on outcome
    urgency = Urgency.HIGH if outcome == PushOutcome.PUSH else Urgency.MEDIUM
    content = description or thought_content

    holding_item = HoldingItem(
        content=content,
        type=HoldingType.CONNECTION,
        relevance_trigger=(
            f"Inner life fascination: {fascination_topic}"
            if fascination_topic
            else "Inner life insight"
        ),
        register_needed=RegisterNeeded.REFLECTIVE,
        urgency=urgency,
        source_workflow="inner_life_push_evaluation",
        epistemic_origin=EpistemicOrigin.INNER_LIFE_PIPELINE,
        thread_ids=threads_pollinated,
    )
    holding_store.write(holding_item)

    # Record push in history when outcome is PUSH
    if outcome == PushOutcome.PUSH:
        push_history.record_push(
            content_summary=content[:200] if len(content) > 200 else content
        )

    return PushEvaluationResult(outcome=outcome, skipped_ceiling=False)
