"""Session end sequence (§4I).

Implements the four-step session end contract:

    1. Snapshot working memory — immutable deepcopy; neither workflow
       receives a reference to the live working memory.
    2. Fire relational_update (Priority 4, restart) from the snapshot.
    3. Fire episodic_flush (Priority 4, suspend) from the snapshot.
    4. Working memory is cleared only after episodic_flush confirms
       success.  On failure, working memory is retained.

Both workflows fire concurrently from the same snapshot.

Working memory retention contract
    ``run_session_end`` returns a ``SessionEndResult``.  The caller MUST
    check ``result.flush_succeeded`` before clearing working memory.  If
    ``flush_succeeded`` is False, the caller must retain working memory
    and retry the flush on reconnection.

    This function never clears working memory itself — that is the
    caller's responsibility.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

from ..state.working_memory import WorkingMemory
from .episodic_flush import EpisodicFlushResult, episodic_flush
from .relational_update import RelationalUpdateResult, relational_update

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def take_snapshot(working_memory: WorkingMemory) -> WorkingMemory:
    """Return an immutable deepcopy of *working_memory*.

    Neither ``relational_update`` nor ``episodic_flush`` receives the live
    working memory object.  Mutations to the snapshot do not affect the
    original, and mutations to the original after this point do not affect
    the snapshot.
    """
    return copy.deepcopy(working_memory)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SessionEndResult:
    """Summary of the session end sequence.

    The caller must check ``flush_succeeded`` before clearing working
    memory.  If ``False``, working memory must be retained for retry.
    """

    session_id: str
    flush_succeeded: bool
    """True only when episodic_flush completed without raising."""

    relational_update_result: RelationalUpdateResult | None = None
    """None if relational_update raised."""

    flush_result: EpisodicFlushResult | None = None
    """None if episodic_flush raised."""

    relational_update_error: str | None = None
    """Exception message if relational_update failed."""

    flush_error: str | None = None
    """Exception message if episodic_flush failed."""

    flush_errors: list[str] = field(default_factory=lambda: list[str]())
    """All exception messages accumulated during the sequence."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_session_end(
    working_memory: WorkingMemory,
    ended_at: datetime,
    *,
    relational_update_fn: Callable[[WorkingMemory], RelationalUpdateResult],
    episodic_flush_fn: Callable[[WorkingMemory, datetime], EpisodicFlushResult],
) -> SessionEndResult:
    """Execute the session end sequence (§4I).

    Takes an immutable snapshot of *working_memory* and fires
    ``relational_update`` and ``episodic_flush`` concurrently.

    Returns a ``SessionEndResult``.  The caller must not clear working
    memory unless ``result.flush_succeeded`` is True.

    Parameters
    ----------
    working_memory:
        Live working memory for the session.  A deepcopy is taken
        immediately; the original is never passed to either workflow.
    ended_at:
        Session end timestamp passed to ``episodic_flush_fn``.
    relational_update_fn:
        Callable wrapping the ``relational_update`` workflow.
        Signature: ``(snapshot: WorkingMemory) → RelationalUpdateResult``.
    episodic_flush_fn:
        Callable wrapping the ``episodic_flush`` workflow.
        Signature: ``(snapshot: WorkingMemory, ended_at: datetime)
        → EpisodicFlushResult``.

    Returns
    -------
    SessionEndResult
        ``flush_succeeded=True`` only when episodic_flush completed
        without raising.  Working memory must be retained otherwise.
    """
    session_id = working_memory.session_id
    logger.info("session_end: starting session=%s", session_id)

    # Step 1: immutable snapshot — neither workflow sees the live object
    snapshot = take_snapshot(working_memory)
    logger.debug("session_end: snapshot taken session=%s", session_id)

    result = SessionEndResult(session_id=session_id, flush_succeeded=False)

    # Steps 2 + 3: submit both concurrently, then collect results
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="session-end") as pool:
        relational_future = pool.submit(relational_update_fn, snapshot)
        flush_future = pool.submit(episodic_flush_fn, snapshot, ended_at)

        # Collect relational_update result
        try:
            rel_result = relational_future.result()
            result.relational_update_result = rel_result
            logger.info(
                "session_end: relational_update complete session=%s version=%d",
                session_id,
                rel_result.relational_version,
            )
        except Exception as exc:
            msg = str(exc)
            result.relational_update_error = msg
            result.flush_errors.append(f"relational_update: {msg}")
            logger.warning(
                "session_end: relational_update failed session=%s: %s",
                session_id,
                msg,
                exc_info=True,
            )

        # Collect episodic_flush result
        try:
            fl_result = flush_future.result()
            result.flush_result = fl_result
            result.flush_succeeded = True
            logger.info(
                "session_end: episodic_flush complete session=%s record=%s",
                session_id,
                fl_result.session_record_id,
            )
        except Exception as exc:
            msg = str(exc)
            result.flush_error = msg
            result.flush_errors.append(f"episodic_flush: {msg}")
            logger.warning(
                "session_end: episodic_flush failed — "
                "working memory retained session=%s: %s",
                session_id,
                msg,
                exc_info=True,
            )

    # Step 4: working memory cleared only by the caller after flush_succeeded
    if result.flush_succeeded:
        logger.info(
            "session_end: complete session=%s — "
            "working memory may be cleared by caller",
            session_id,
        )
    else:
        logger.warning(
            "session_end: episodic_flush did not confirm — "
            "caller must retain working memory session=%s",
            session_id,
        )

    return result


# ---------------------------------------------------------------------------
# Convenience wrappers (for callers that pass full workflow callables)
# ---------------------------------------------------------------------------


def make_relational_update_fn(
    inference_fn: Callable[[str], str],
    **store_kwargs: object,
) -> Callable[[WorkingMemory], RelationalUpdateResult]:
    """Return a ``relational_update_fn`` bound to *inference_fn*.

    ``store_kwargs`` are forwarded to ``DaemonRelationalStore``.
    """
    from ..state.daemon_relational import DaemonRelationalStore

    store = DaemonRelationalStore(**store_kwargs)  # type: ignore[arg-type]

    def _fn(snapshot: WorkingMemory) -> RelationalUpdateResult:
        return relational_update(snapshot, inference_fn=inference_fn, store=store)

    return _fn


def make_episodic_flush_fn(
    inference_fn: Callable[[str], str],
    write_thread_episode_fn: object,
    update_cooccurrence_fn: object,
    write_handoff_note_fn: object,
    write_session_record_fn: object,
    write_session_thread_index_fn: object,
    generate_embedding_fn: object = None,
) -> Callable[[WorkingMemory, datetime], EpisodicFlushResult]:
    """Return an ``episodic_flush_fn`` bound to the provided callables."""

    def _fn(snapshot: WorkingMemory, ended_at: datetime) -> EpisodicFlushResult:
        return episodic_flush(
            snapshot,
            ended_at,
            inference_fn=inference_fn,  # type: ignore[arg-type]
            write_thread_episode_fn=write_thread_episode_fn,  # type: ignore[arg-type]
            update_cooccurrence_fn=update_cooccurrence_fn,  # type: ignore[arg-type]
            write_handoff_note_fn=write_handoff_note_fn,  # type: ignore[arg-type]
            write_session_record_fn=write_session_record_fn,  # type: ignore[arg-type]
            write_session_thread_index_fn=write_session_thread_index_fn,  # type: ignore[arg-type]
            generate_embedding_fn=generate_embedding_fn,  # type: ignore[arg-type]
        )

    return _fn
