"""contradiction_detection workflow (§3F).

Detects contradictions among daemon-space semantic items and surfaces
confirmed conflicts to the holding store as ``type: reasoned_disagreement``
items, ready for discharge through the Stage 4 conversation turn (§8d).

Trigger scopes
--------------
cron_nightly
    All semantic items written since the last detection run.
write_threshold
    The triggering batch of N items (default N=10, configurable in
    ``daemon-memory-server.yaml`` as ``contradiction.detection_batch_size``).
inquiry_completion
    All items with ``inquiry_id == completed_inquiry_id`` — threaded
    through from ``commissioned_inquiry``'s trigger call.

Detection algorithm
-------------------
1. Find candidate pairs: similarity >= ``similarity_threshold`` (default 0.85,
   configurable in ``daemon-memory-server.yaml``).  Similarity is *necessary*
   but not sufficient.
2. Assess each candidate pair with model inference (``items_conflict()``).
3. Confirmed conflicts: write a contradiction record via
   ``create_contradiction_fn``, then write a ``HoldingItem`` with
   ``type: reasoned_disagreement`` and a non-null ``contradiction_id``.

Register gate
-------------
Contradiction items are written with ``register_needed: reflective``.
The Stage 4 discharge logic will not surface them when the inferred register
is ``urgent`` — enforced at discharge time, not here.

Priority: 5 / Preemption: **suspend** / Model: presentation
Trigger: cron_nightly_or_write_threshold (also: inquiry_completion)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from ..state._types import EpistemicOrigin
from ..state.holding import (
    HoldingItem,
    HoldingStore,
    HoldingType,
    RegisterNeeded,
    Urgency,
)
from .preemption import PreemptionContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DetectionTrigger(StrEnum):
    """How this contradiction detection run was initiated."""

    CRON_NIGHTLY = "cron_nightly"
    WRITE_THRESHOLD = "write_threshold"
    INQUIRY_COMPLETION = "inquiry_completion"


# ---------------------------------------------------------------------------
# Input / output types
# ---------------------------------------------------------------------------


@dataclass
class SemanticItemSummary:
    """Minimum representation of a daemon-space semantic item.

    The caller fetches these from daemon-memory-server and scopes them to
    the correct trigger window before passing them to the workflow.  All
    items must belong to the ``daemon`` knowledge space.
    """

    id: str
    """UUID from daemon-memory-server."""

    content: str
    """Prose content of the item."""

    content_type: str
    """``finding | conclusion | observation | ...``"""

    inquiry_id: str | None = None
    """Set when the item was produced by a commissioned inquiry."""


@dataclass
class CandidatePair:
    """Two items that passed the similarity threshold."""

    item_a: SemanticItemSummary
    item_b: SemanticItemSummary
    similarity: float


@dataclass
class ContradictionDetectionResult:
    """Returned on successful completion of a detection run."""

    trigger: DetectionTrigger
    items_assessed: int
    """Number of ``new_items`` passed in."""

    candidate_pairs: int
    """Number of pairs that passed the similarity threshold."""

    contradictions_written: int
    """Number of confirmed contradictions written to the holding store."""

    inquiry_id: str | None = None
    """Populated only for ``inquiry_completion`` trigger."""

    contradiction_ids: list[str] = field(default_factory=lambda: list[str]())
    """IDs of contradiction records created during this run."""


# ---------------------------------------------------------------------------
# Callable type aliases
# ---------------------------------------------------------------------------

FindCandidatePairsFn = Callable[[list[SemanticItemSummary]], list[CandidatePair]]
"""Find candidate pairs above the similarity threshold.

The memory server performs ANN lookup internally; this callable returns only
pairs whose similarity is at or above the configured threshold.  Items with
the same ``id`` are never returned as a pair.

Signature: ``(new_items: list[SemanticItemSummary]) → list[CandidatePair]``.
"""

CreateContradictionFn = Callable[[str, str, str], str]
"""Record a contradiction in daemon-memory-server.

Signature: ``(item_a_id, item_b_id, conflict_summary) → contradiction_id``.

Returns a UUID string for the new contradiction record.
"""

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CONFLICT_ASSESSMENT_PROMPT = """\
You are reviewing two statements from the daemon's knowledge base.

Item A ({id_a}):
{content_a}

Item B ({id_b}):
{content_b}

Do these two statements meaningfully conflict with each other?

Reply with exactly one of:
CONFLICT: <one sentence explaining the conflict>
NO_CONFLICT: <one sentence explaining why they are compatible>
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _noop() -> None:
    pass


def _assess_conflict(
    pair: CandidatePair,
    inference_fn: Callable[[str], str],
) -> tuple[bool, str]:
    """Ask the model whether *pair* represents a genuine conflict.

    Returns ``(conflict_confirmed, summary_text)``.  Falls back to
    ``NO_CONFLICT`` when the response cannot be parsed (conservative).
    """
    prompt = _CONFLICT_ASSESSMENT_PROMPT.format(
        id_a=pair.item_a.id,
        content_a=pair.item_a.content,
        id_b=pair.item_b.id,
        content_b=pair.item_b.content,
    )
    response = inference_fn(prompt).strip()
    match = re.match(r"^(CONFLICT|NO_CONFLICT):\s*(.+)$", response, re.DOTALL)
    if match:
        verdict = match.group(1)
        summary = match.group(2).strip()
    else:
        logger.warning(
            "contradiction_detection: unparseable response; treating as NO_CONFLICT "
            "item_a=%s item_b=%s response=%r",
            pair.item_a.id,
            pair.item_b.id,
            response[:120],
        )
        verdict = "NO_CONFLICT"
        summary = response

    return verdict == "CONFLICT", summary


def _write_contradiction_holding_item(
    contradiction_id: str,
    pair: CandidatePair,
    conflict_summary: str,
    holding_store: HoldingStore,
) -> HoldingItem:
    """Write a ``reasoned_disagreement`` holding item for a confirmed contradiction.

    ``register_needed`` is ``reflective`` — contradictions want a thoughtful
    conversation, never ``urgent``.  This satisfies the register gate
    constraint from §3F acceptance criteria.
    """
    item = HoldingItem(
        content=conflict_summary,
        type=HoldingType.REASONED_DISAGREEMENT,
        relevance_trigger=(
            f"Items {pair.item_a.id[:8]}... and {pair.item_b.id[:8]}... "
            f"contradict each other (similarity {pair.similarity:.2f})"
        ),
        register_needed=RegisterNeeded.REFLECTIVE,
        urgency=Urgency.MEDIUM,
        source_workflow="contradiction_detection",
        epistemic_origin=EpistemicOrigin.INTERNAL,
        thread_ids=[],
        contradiction_id=contradiction_id,
    )
    return holding_store.write(item)


# ---------------------------------------------------------------------------
# Public workflow function
# ---------------------------------------------------------------------------


def contradiction_detection(
    new_items: list[SemanticItemSummary],
    trigger: DetectionTrigger,
    *,
    inference_fn: Callable[[str], str],
    find_candidate_pairs_fn: FindCandidatePairsFn,
    create_contradiction_fn: CreateContradictionFn,
    holding_store: HoldingStore | None = None,
    inquiry_id: str | None = None,
    preemption_ctx: PreemptionContext | None = None,
    checkpoint_fn: Callable[[], None] | None = None,
    rollback_fn: Callable[[], None] | None = None,
) -> ContradictionDetectionResult:
    """Detect contradictions among semantic items and surface via holding store (§3F).

    Parameters
    ----------
    new_items:
        Items to check, already scoped by the caller to the trigger window.
        For ``inquiry_completion``, the caller filters to items whose
        ``inquiry_id`` matches the completed inquiry.  Items must belong to
        the ``daemon`` knowledge space.
    trigger:
        How this run was initiated.
    inference_fn:
        Callable that sends a prompt to the presentation model and returns text.
    find_candidate_pairs_fn:
        Returns pairs above the similarity threshold.  Items with identical
        ``id`` are never returned as a pair.
    create_contradiction_fn:
        Records a contradiction in daemon-memory-server and returns its UUID.
        Signature: ``(item_a_id, item_b_id, conflict_summary) → contradiction_id``.
    holding_store:
        Holding store to write ``reasoned_disagreement`` items to.
        ``None`` → uses the default store at ``data/daemon_state/holding.yaml``.
    inquiry_id:
        The inquiry UUID for ``inquiry_completion`` trigger.  Threaded through
        from ``commissioned_inquiry``'s trigger call.  ``None`` for other
        trigger types.
    preemption_ctx:
        Preemption context from the workflow engine.  ``None`` = no preemption.
    checkpoint_fn:
        Called on checkpoint if ``preemption_ctx`` fires.  Defaults to noop.
    rollback_fn:
        Called on resume if ``preemption_ctx`` fires.  Defaults to noop.

    Returns
    -------
    ContradictionDetectionResult
        Summary of the detection run.
    """
    store = holding_store if holding_store is not None else HoldingStore()
    ckpt_fn = checkpoint_fn if checkpoint_fn is not None else _noop
    rb_fn = rollback_fn if rollback_fn is not None else _noop

    logger.info(
        "contradiction_detection: starting trigger=%s items=%d inquiry_id=%s",
        trigger,
        len(new_items),
        inquiry_id,
    )

    if not new_items:
        logger.info("contradiction_detection: no items to check; returning early")
        return ContradictionDetectionResult(
            trigger=trigger,
            inquiry_id=inquiry_id,
            items_assessed=0,
            candidate_pairs=0,
            contradictions_written=0,
        )

    # ------------------------------------------------------------------
    # Step 1: Find candidate pairs above similarity threshold
    # ------------------------------------------------------------------
    logger.debug("contradiction_detection: finding candidate pairs")
    candidates = find_candidate_pairs_fn(new_items)
    logger.debug("contradiction_detection: %d candidate pair(s) found", len(candidates))

    # ------------------------------------------------------------------
    # Step 2: Assess each candidate pair; cooperate at safe points
    # ------------------------------------------------------------------
    contradictions_written = 0
    contradiction_ids: list[str] = []

    for pair in candidates:
        if preemption_ctx is not None:
            preemption_ctx.cooperate(checkpoint_fn=ckpt_fn, rollback_fn=rb_fn)

        conflict_confirmed, conflict_summary = _assess_conflict(pair, inference_fn)

        if not conflict_confirmed:
            logger.debug(
                "contradiction_detection: not a conflict item_a=%s item_b=%s",
                pair.item_a.id,
                pair.item_b.id,
            )
            continue

        logger.info(
            "contradiction_detection: conflict confirmed item_a=%s item_b=%s",
            pair.item_a.id,
            pair.item_b.id,
        )

        # ------------------------------------------------------------------
        # Step 3: Write contradiction record + holding item
        # ------------------------------------------------------------------
        contradiction_id = create_contradiction_fn(
            pair.item_a.id,
            pair.item_b.id,
            conflict_summary,
        )
        _write_contradiction_holding_item(
            contradiction_id=contradiction_id,
            pair=pair,
            conflict_summary=conflict_summary,
            holding_store=store,
        )
        contradiction_ids.append(contradiction_id)
        contradictions_written += 1

    logger.info(
        "contradiction_detection: complete candidates=%d contradictions=%d",
        len(candidates),
        contradictions_written,
    )

    return ContradictionDetectionResult(
        trigger=trigger,
        inquiry_id=inquiry_id,
        items_assessed=len(new_items),
        candidate_pairs=len(candidates),
        contradictions_written=contradictions_written,
        contradiction_ids=contradiction_ids,
    )
