"""Conversational retrieval — §4D / §8b.

Presence before retrieval (philosophy §12):
    Retrieval serves the response; it does not precede or replace presence.
    ``conversational_retrieval`` is designed to be awaited *after* the
    response generation has begun — results are available for
    ``personal_assistant`` to weave in as they bear on the response, not to
    block generation.

Primary query:
    Searches ``user_pkm`` (weight 1.0), ``daemon`` (weight 0.3), ``shared``
    (weight 0.8).  ``always_include_shared=True``.  ``top_k=5`` per space
    before merge.

Secondary query (peripheral thread):
    When the thread stack has a peripheral entry, its ``central_question`` is
    used as a secondary query.  Secondary scores are multiplied by
    ``peripheral_weight=0.4`` at merge time.

Graceful degradation:
    Any exception from the memory client → empty ``RetrievalContext`` returned,
    caller is not notified.  The daemon proceeds from local state.

Pending artifacts:
    When a result has ``chunk_status: pending`` in its metadata, it is placed
    in ``RetrievalContext.pending_artifacts`` rather than ``semantic``.  The
    personal_assistant can acknowledge naturally ("still reading through it")
    rather than silently returning nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .thread_stack import ThreadStackEntry, ThreadStackState
from .threads import ThreadStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryClientProtocol(Protocol):
    """Structural interface satisfied by daemon-memory-client's semantic API.

    Defined here as a Protocol so retrieval can be tested and type-checked
    without requiring a concrete implementation of daemon-memory-client.
    When daemon-memory-client ships a ``semantic_query`` method, any client
    instance will satisfy this protocol automatically.
    """

    async def semantic_query(  # type: ignore[empty-body]
        self, query: SemanticQuery
    ) -> list[SemanticResult]: ...


# ---------------------------------------------------------------------------
# Query / result types
# ---------------------------------------------------------------------------


@dataclass
class SemanticQuery:
    """Parameters for a semantic search request.

    Attributes:
        query_text: The text to embed and search with.
        spaces: Knowledge spaces to search.
        space_weights: Per-space score multiplier applied at merge time.
        always_include_shared: If True, shared space results are always
            included even after top-k filtering.
        top_k: Number of results to request per space before merge.

    Example::

        >>> q = SemanticQuery(
        ...     query_text="example",
        ...     spaces=["user_pkm"],
        ...     space_weights={"user_pkm": 1.0},
        ... )
        >>> q.top_k
        5
    """

    query_text: str
    spaces: list[str]
    space_weights: dict[str, float]
    always_include_shared: bool = True
    top_k: int = 5


@dataclass
class SemanticResult:
    """A single result from a semantic query.

    Attributes:
        document_id: Unique identifier for the document/chunk.
        text: Text content of the result.
        score: Similarity score (0.0–1.0) after space-weight application.
        space: Knowledge space this result came from.
        metadata: Arbitrary metadata from the memory server (e.g.
            ``chunk_status``, ``artifact_id``, ``title``).

    Example::

        >>> r = SemanticResult(
        ...     document_id="doc-1",
        ...     text="hello",
        ...     score=0.9,
        ...     space="user_pkm",
        ... )
        >>> r.metadata
        {}
    """

    document_id: str
    text: str
    score: float
    space: str
    metadata: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())


@dataclass
class RetrievalContext:
    """Result of ``conversational_retrieval``.

    Attributes:
        semantic: Merged, ranked semantic results ready for use in generation.
        pending_artifacts: Results whose underlying artifact has
            ``chunk_status: pending`` — surfaced so personal_assistant can
            acknowledge naturally rather than silently returning nothing.

    Example::

        >>> ctx = RetrievalContext()
        >>> ctx.is_empty
        True
        >>> ctx.has_pending
        False
    """

    semantic: list[SemanticResult] = field(
        default_factory=lambda: list[SemanticResult]()
    )
    pending_artifacts: list[SemanticResult] = field(
        default_factory=lambda: list[SemanticResult]()
    )

    @property
    def has_pending(self) -> bool:
        """True if any artifacts are still being chunked."""
        return len(self.pending_artifacts) > 0

    @property
    def is_empty(self) -> bool:
        """True if neither semantic results nor pending artifacts are present."""
        return not self.semantic and not self.pending_artifacts


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def _merge(
    primary: list[SemanticResult],
    secondary: list[SemanticResult],
    peripheral_weight: float,
) -> list[SemanticResult]:
    """Merge primary and secondary results.

    Secondary result scores are multiplied by ``peripheral_weight`` before
    merging.  When the same document appears in both lists, the higher final
    score is kept.  Returns results sorted by score descending.

    Args:
        primary: Results from the primary query (scores already weighted).
        secondary: Results from the secondary query (scores will be scaled).
        peripheral_weight: Multiplier applied to secondary result scores.

    Returns:
        Deduplicated, merged list sorted by score descending.

    Example::

        >>> a = SemanticResult("d1", "text", 0.8, "user_pkm")
        >>> b = SemanticResult("d2", "text", 0.9, "shared")
        >>> c = SemanticResult("d1", "text", 0.6, "shared")
        >>> merged = _merge([a, b], [c], peripheral_weight=0.5)
        >>> [r.document_id for r in merged]
        ['d2', 'd1']
        >>> merged[1].score  # d1 keeps primary score 0.8 (> 0.6*0.5=0.3)
        0.8
    """
    merged: dict[str, SemanticResult] = {}

    for result in primary:
        merged[result.document_id] = result

    for result in secondary:
        scaled_score = result.score * peripheral_weight
        if result.document_id in merged:
            existing = merged[result.document_id]
            if scaled_score > existing.score:
                merged[result.document_id] = SemanticResult(
                    document_id=result.document_id,
                    text=result.text,
                    score=scaled_score,
                    space=result.space,
                    metadata=result.metadata,
                )
        else:
            merged[result.document_id] = SemanticResult(
                document_id=result.document_id,
                text=result.text,
                score=scaled_score,
                space=result.space,
                metadata=result.metadata,
            )

    return sorted(merged.values(), key=lambda r: r.score, reverse=True)


# ---------------------------------------------------------------------------
# Pending artifact partition
# ---------------------------------------------------------------------------


def _partition_pending(
    results: list[SemanticResult],
) -> tuple[list[SemanticResult], list[SemanticResult]]:
    """Split results into (ready, pending) on ``chunk_status``.

    A result is pending when its metadata contains
    ``{"chunk_status": "pending"}``.

    Args:
        results: Mixed list of results to partition.

    Returns:
        Tuple of ``(ready_results, pending_results)``.

    Example::

        >>> ready_r = SemanticResult("d1", "text", 0.9, "user_pkm")
        >>> pending_r = SemanticResult(
        ...     "d2", "text", 0.7, "shared",
        ...     metadata={"chunk_status": "pending"},
        ... )
        >>> ready, pending = _partition_pending([ready_r, pending_r])
        >>> ready[0].document_id
        'd1'
        >>> pending[0].document_id
        'd2'
    """
    ready: list[SemanticResult] = []
    pending: list[SemanticResult] = []
    for result in results:
        if result.metadata.get("chunk_status") == "pending":
            pending.append(result)
        else:
            ready.append(result)
    return ready, pending


# ---------------------------------------------------------------------------
# Primary retrieval function
# ---------------------------------------------------------------------------


async def conversational_retrieval(
    message: str,
    thread_stack: list[ThreadStackEntry],
    memory_client: MemoryClientProtocol,
    thread_store: ThreadStore | None = None,
) -> RetrievalContext:
    """Perform conversational retrieval for a single turn (§4D).

    Presence before retrieval (philosophy §12): this coroutine is designed to
    be awaited after the response has begun — results serve presence, not gate
    it.

    Args:
        message: The user's current message text (primary query).
        thread_stack: Active non-floating thread stack entries for this turn.
        memory_client: Async semantic query client satisfying
            ``MemoryClientProtocol``.
        thread_store: Thread store for loading peripheral thread data.  If
            ``None`` and a peripheral thread is present, secondary query is
            skipped gracefully.

    Returns:
        ``RetrievalContext`` with ``semantic`` results and any
        ``pending_artifacts``.  An empty context is returned (no exception
        raised) when the memory server is unavailable.

    Note:
        Any exception from the memory client is caught and suppressed — the
        daemon proceeds from local state rather than surfacing an error to the
        caller.
    """
    primary_query = SemanticQuery(
        query_text=message,
        spaces=["user_pkm", "daemon", "shared"],
        space_weights={"user_pkm": 1.0, "daemon": 0.3, "shared": 0.8},
        always_include_shared=True,
        top_k=5,
    )

    try:
        raw_results = await memory_client.semantic_query(primary_query)
    except Exception:
        # Memory server unavailable — proceed from local state, no error announced
        logger.debug(
            "Memory client unavailable during retrieval; returning empty context"
        )
        return RetrievalContext()

    # Secondary query on peripheral thread's central_question
    peripheral = next(
        (t for t in thread_stack if t.state == ThreadStackState.peripheral), None
    )

    if peripheral is not None and thread_store is not None:
        try:
            thread = thread_store.load(peripheral.thread_id)
            secondary_query = SemanticQuery(
                query_text=thread.central_question,
                spaces=["user_pkm", "daemon", "shared"],
                space_weights={"user_pkm": 0.6, "daemon": 0.2, "shared": 0.8},
                always_include_shared=True,
                top_k=3,
            )
            secondary_results = await memory_client.semantic_query(secondary_query)
            raw_results = _merge(raw_results, secondary_results, peripheral_weight=0.4)
        except Exception:
            # Secondary query failure is non-fatal — return primary results as-is
            logger.debug(
                "Secondary retrieval for peripheral thread %r failed; "
                "proceeding with primary results",
                peripheral.thread_id,
            )

    semantic, pending = _partition_pending(raw_results)
    return RetrievalContext(semantic=semantic, pending_artifacts=pending)
