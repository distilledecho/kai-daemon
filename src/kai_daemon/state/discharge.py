"""Holding store discharge logic — §8d.

Two conditions required for an item to discharge:

1. ``relevance_trigger`` similarity to the current message exceeds
   ``threshold`` (default 0.72, configurable in ``user.yaml`` under
   ``holding: discharge_threshold``).
2. ``register_needed`` matches the inferred register, or is ``any``.

Additional constraints:

- At most one item per turn (highest similarity score wins).
- Items already surfaced (``surfaced`` is not ``None``) are never candidates.
- ``type: reasoned_disagreement`` is **never** discharged when
  ``inferred_register`` is ``"urgent"``.

The caller is responsible for:

- Pre-computing similarity scores between the message and each item's
  ``relevance_trigger`` text (usually via the embedding model).
- Calling ``hydrate_contradiction`` when the returned item has a non-null
  ``contradiction_id``, before surfacing the content.
- Marking the item as surfaced (``HoldingStore.discharge()``) after use.

The selection function ``select_discharge_candidate`` is pure: it performs
no I/O, no writes, and has no side effects.  Only ``hydrate_contradiction``
is async, and only because it queries the memory server.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .holding import HoldingItem, HoldingType, RegisterNeeded

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default threshold
# ---------------------------------------------------------------------------

DEFAULT_DISCHARGE_THRESHOLD: float = 0.72
"""Minimum relevance_trigger similarity to pass the first gate.

Configurable in ``user.yaml`` under ``holding.discharge_threshold``.
"""

# ---------------------------------------------------------------------------
# Register gate
# ---------------------------------------------------------------------------

_URGENT = "urgent"


def _register_matches(register_needed: RegisterNeeded, inferred_register: str) -> bool:
    """True if *register_needed* is compatible with *inferred_register*.

    ``RegisterNeeded.ANY`` is compatible with any inferred register.
    All other values require an exact match.

    Args:
        register_needed: The item's ``register_needed`` field.
        inferred_register: The inferred register for the current turn.

    Returns:
        ``True`` if the gate is satisfied, ``False`` otherwise.

    Example::

        >>> from kai_daemon.state.holding import RegisterNeeded
        >>> _register_matches(RegisterNeeded.ANY, "urgent")
        True
        >>> _register_matches(RegisterNeeded.REFLECTIVE, "urgent")
        False
        >>> _register_matches(RegisterNeeded.CASUAL, "casual")
        True
    """
    if register_needed == RegisterNeeded.ANY:
        return True
    return register_needed.value == inferred_register


def _is_contradiction(item: HoldingItem) -> bool:
    return item.type == HoldingType.REASONED_DISAGREEMENT


# ---------------------------------------------------------------------------
# Selection (pure)
# ---------------------------------------------------------------------------


def select_discharge_candidate(
    items: list[HoldingItem],
    inferred_register: str,
    scores: dict[str, float],
    threshold: float = DEFAULT_DISCHARGE_THRESHOLD,
) -> HoldingItem | None:
    """Return the single best discharge candidate, or ``None``.

    Both gates must pass:

    1. ``scores[item.id] > threshold`` — relevance similarity gate.
    2. ``_register_matches(item.register_needed, inferred_register)`` —
       register gate.

    Additional exclusions:

    - Already-surfaced items (``item.surfaced is not None``).
    - ``type: reasoned_disagreement`` when ``inferred_register == "urgent"``.

    Among all passing items, the one with the highest similarity score is
    selected.  Ties are broken by iteration order (stable, no randomness).

    Args:
        items: All holding items, surfaced and unsurfaced.
        inferred_register: Register inferred for the current turn.
            One of ``"exploratory"``, ``"reflective"``, ``"casual"``,
            ``"urgent"``.
        scores: Mapping of ``item.id → similarity_score``.  Items whose
            ID is absent from this mapping are skipped (treated as score 0).
        threshold: Minimum similarity score to pass gate 1.
            Default: ``DEFAULT_DISCHARGE_THRESHOLD`` (0.72).

    Returns:
        The best-matching ``HoldingItem``, or ``None`` if no candidate passes
        both gates.

    """
    best: HoldingItem | None = None
    best_score: float = -math.inf

    for item in items:
        # Gate 0: already surfaced → skip
        if item.surfaced is not None:
            continue

        # Gate 0b: urgent register blocks contradiction discharge
        if inferred_register == _URGENT and _is_contradiction(item):
            continue

        # Gate 1: similarity score
        score = scores.get(item.id, 0.0)
        if score <= threshold:
            continue

        # Gate 2: register match
        if not _register_matches(item.register_needed, inferred_register):
            continue

        # Both gates passed — keep if best so far
        if score > best_score:
            best = item
            best_score = score

    return best


# ---------------------------------------------------------------------------
# Contradiction hydration
# ---------------------------------------------------------------------------


@dataclass
class ContradictionRecord:
    """A contradiction record fetched from daemon-memory-server.

    This is the full record associated with a ``HoldingItem`` whose
    ``contradiction_id`` is non-null.  Surfacing a ``reasoned_disagreement``
    item without hydrating this record first would expose only a bare UUID
    to the caller — never acceptable.

    Attributes:
        id: The contradiction record's UUID.
        item_a_id: UUID of the first semantic item in the conflict.
        item_b_id: UUID of the second semantic item in the conflict.
        conflict_summary: One-sentence summary of the contradiction.
    """

    id: str
    item_a_id: str
    item_b_id: str
    conflict_summary: str


@runtime_checkable
class ContradictionClientProtocol(Protocol):
    """Structural interface for contradiction record retrieval.

    Defined as a Protocol so ``hydrate_contradiction`` can be tested
    and type-checked without a concrete daemon-memory-client installation.
    Any client that exposes ``get_contradiction`` will satisfy this protocol.
    """

    async def get_contradiction(  # type: ignore[empty-body]
        self, contradiction_id: str
    ) -> ContradictionRecord | None: ...


async def hydrate_contradiction(
    item: HoldingItem,
    client: ContradictionClientProtocol,
) -> ContradictionRecord | None:
    """Fetch the full contradiction record for a ``reasoned_disagreement`` item.

    Must be called before surfacing any ``HoldingItem`` whose
    ``contradiction_id`` is non-null — surfacing a bare ID is not acceptable.

    If ``item.contradiction_id`` is ``None``, returns ``None`` immediately
    (no network call).

    If the memory server is unavailable or returns ``None``, logs a warning
    and returns ``None`` — the caller can decide whether to skip surfacing or
    surface the item with degraded content.

    Args:
        item: The discharge candidate.
        client: A client satisfying ``ContradictionClientProtocol``.

    Returns:
        ``ContradictionRecord`` on success, ``None`` otherwise.

    """
    if item.contradiction_id is None:
        return None

    try:
        record = await client.get_contradiction(item.contradiction_id)
    except Exception:
        logger.warning(
            "discharge: failed to hydrate contradiction %r; returning None",
            item.contradiction_id,
            exc_info=True,
        )
        return None

    if record is None:
        logger.warning(
            "discharge: contradiction %r not found in memory server",
            item.contradiction_id,
        )

    return record
