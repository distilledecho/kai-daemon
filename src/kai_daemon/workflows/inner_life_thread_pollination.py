"""inner_life_thread_pollination workflow (§7e).

Cross-pollinates a fascination insight into semantically relevant threads.

For each active or dormant thread, the workflow:
1. Checks deduplication — skips if the same fascination wrote a perspective
   to this thread within the last ``dedup_hours`` hours (default 24h).
2. Runs an inference call to assess relevance.
3. If relevant, appends a ``DaemonPerspective`` to the thread.
4. If any perspective was written, emits a high-significance scratch
   ``SIGNAL`` with a 24h TTL targeting ``inner_life_push_evaluation``.

Only fires when the upstream ``daemon_integration`` result has a
fascination topic (routes ``new_fascination`` or ``develops_existing``).

Priority: 8 / Preemption: restart / Model: presentation
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ..state.scratch import (
    EpistemicOrigin,
    Lifecycle,
    ScratchNote,
    ScratchStore,
    ScratchType,
)
from ..state.threads import DaemonPerspective, Thread, ThreadStatus, ThreadStore
from .daemon_integration import IntegrationResult, IntegrationRoute

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLLINATION_DEDUP_HOURS: int = 24
POLLINATION_SIGNAL_TTL_HOURS: int = 24

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_RELEVANCE_PROMPT_TEMPLATE = """\
Does this fascination connect meaningfully to this conversation thread?

Fascination topic: {fascination_topic}

Thread title: {thread_title}
Thread central question: {central_question}
Thread current state: {current_state}

Reply with exactly one word: RELEVANT or NOT_RELEVANT.
"""

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PollinationResult:
    """Result of the inner_life_thread_pollination workflow."""

    threads_pollinated: list[str] = field(default_factory=lambda: list[str]())
    """Thread IDs that received a new ``DaemonPerspective``."""
    signal_written: bool = False
    """``True`` if a high-significance SIGNAL was written to scratch space."""
    skipped_no_fascination: bool = False
    """``True`` if there was no fascination topic to pollinate from."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_duplicate(
    thread: Thread,
    fascination_topic: str,
    now: datetime,
    dedup_hours: int,
) -> bool:
    """Return ``True`` if *fascination_topic* has a perspective within *dedup_hours*."""
    cutoff = now - timedelta(hours=dedup_hours)
    for p in thread.daemon_perspectives:
        if p.from_fascination.lower() == fascination_topic.lower():
            try:
                written = datetime.fromisoformat(p.written_at)
            except ValueError:
                continue
            if written.tzinfo is None:
                written = written.replace(tzinfo=UTC)
            if written >= cutoff:
                return True
    return False


def _is_relevant(
    fascination_topic: str,
    thread: Thread,
    inference_fn: Callable[[str], str],
) -> bool:
    """Return ``True`` if inference says the fascination is relevant to the thread."""
    prompt = _RELEVANCE_PROMPT_TEMPLATE.format(
        fascination_topic=fascination_topic,
        thread_title=thread.title,
        central_question=thread.central_question,
        current_state=thread.current_state,
    )
    response = inference_fn(prompt)
    token = response.strip().upper().split()[0] if response.strip() else ""
    if token == "RELEVANT":
        return True
    if token not in ("RELEVANT", "NOT_RELEVANT"):
        logger.warning(
            "Unrecognised relevance response %r for thread %r — not relevant",
            response.strip(),
            thread.id,
        )
    return False


def _build_perspective_content(
    fascination_topic: str,
    thought_content: str,
    inference_fn: Callable[[str], str],
) -> str:
    """Generate a perspective note linking the fascination to a thread context."""
    prompt = (
        f"Write a brief reflection (2–3 sentences) connecting this inner thought "
        f"to a conversation you're tracking.\n\n"
        f"Inner thought:\n{thought_content}\n\n"
        f"Fascination: {fascination_topic}\n\n"
        f"Write the reflection in first person."
    )
    return inference_fn(prompt).strip()


# ---------------------------------------------------------------------------
# Public workflow function
# ---------------------------------------------------------------------------


def inner_life_thread_pollination(
    result: IntegrationResult,
    *,
    thread_store: ThreadStore,
    scratch_store: ScratchStore,
    inference_fn: Callable[[str], str],
    session_id: str | None = None,
    now: datetime | None = None,
    dedup_hours: int = POLLINATION_DEDUP_HOURS,
    signal_ttl_hours: int = POLLINATION_SIGNAL_TTL_HOURS,
) -> PollinationResult:
    """Cross-pollinate a fascination insight into relevant threads.

    Parameters
    ----------
    result:
        ``IntegrationResult`` from the upstream ``daemon_integration``
        workflow run.  Only ``new_fascination`` and ``develops_existing``
        routes carry a fascination topic; all other routes return early.
    thread_store:
        Thread store to read and update threads.
    scratch_store:
        Scratch space store to write the high-significance SIGNAL.
    inference_fn:
        Callable that sends a prompt to the presentation model and returns
        the raw text response.  Injectable for testing.
    session_id:
        Session/run identifier for scratch notes.  A fresh UUID is used if
        not provided (appropriate for background workflows).
    now:
        Override current time (for testing).
    dedup_hours:
        Deduplication window in hours (default 24).
    signal_ttl_hours:
        TTL in hours for the emitted SIGNAL scratch note (default 24).

    Returns
    -------
    PollinationResult
        Pollinated thread IDs, signal-written flag, and skip reason.
    """
    _now = now if now is not None else datetime.now(UTC)
    _session_id = session_id or str(uuid.uuid4())

    # Only pollinate when there is a fascination topic
    if (
        result.route
        not in (
            IntegrationRoute.NEW_FASCINATION,
            IntegrationRoute.DEVELOPS_EXISTING,
        )
        or result.fascination_topic is None
    ):
        return PollinationResult(skipped_no_fascination=True)

    fascination_topic = result.fascination_topic
    pollinated: list[str] = []

    # Candidate threads: active and dormant
    candidate_statuses = {ThreadStatus.ACTIVE, ThreadStatus.DORMANT}
    threads = [t for t in thread_store.list_all() if t.status in candidate_statuses]

    for thread in threads:
        # 1. Deduplication check
        if _is_duplicate(thread, fascination_topic, _now, dedup_hours):
            logger.debug(
                "Skipping thread %r — duplicate perspective from %r within %dh",
                thread.id,
                fascination_topic,
                dedup_hours,
            )
            continue

        # 2. Relevance check
        if not _is_relevant(fascination_topic, thread, inference_fn):
            continue

        # 3. Build and write perspective
        content = _build_perspective_content(
            fascination_topic, result.thought_content, inference_fn
        )
        perspective = DaemonPerspective(
            content=content,
            from_fascination=fascination_topic,
            written_at=_now.isoformat(),
            thread_status_at_writing=thread.status,
        )
        thread_store.add_perspective(thread.id, perspective)
        pollinated.append(thread.id)

    # 4. Emit high-significance SIGNAL if any thread was pollinated
    signal_written = False
    if pollinated:
        ttl = (_now + timedelta(hours=signal_ttl_hours)).isoformat()
        note = ScratchNote(
            workflow_id="inner_life_thread_pollination",
            session_id=_session_id,
            content=(
                f"Pollinated {len(pollinated)} thread(s) from fascination "
                f"'{fascination_topic}': {', '.join(pollinated)}"
            ),
            type=ScratchType.SIGNAL,
            epistemic_origin=EpistemicOrigin.INNER_LIFE_PIPELINE,
            ttl=ttl,
            target_workflow="inner_life_push_evaluation",
            lifecycle=Lifecycle.ACTIVE,
            thread_ids=pollinated,
        )
        scratch_store.write(note)
        signal_written = True

    return PollinationResult(
        threads_pollinated=pollinated,
        signal_written=signal_written,
    )
