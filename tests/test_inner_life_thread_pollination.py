"""Tests for inner_life_thread_pollination workflow (§7e)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai_daemon.state.scratch import ScratchStore, ScratchType
from kai_daemon.state.threads import (
    DaemonPerspective,
    EpistemicStatus,
    Stance,
    Thread,
    ThreadStatus,
    ThreadStore,
)
from kai_daemon.workflows.daemon_integration import IntegrationResult, IntegrationRoute
from kai_daemon.workflows.inner_life_thread_pollination import (
    POLLINATION_DEDUP_HOURS,
    POLLINATION_SIGNAL_TTL_HOURS,
    _is_duplicate,
    inner_life_thread_pollination,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 27, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def thread_store(tmp_path: Path) -> ThreadStore:
    return ThreadStore(
        threads_path=tmp_path / "threads",
        pickup_notes_path=tmp_path / "pickup_notes",
        chroma_client=None,
    )


@pytest.fixture
def scratch_store(tmp_path: Path) -> ScratchStore:
    return ScratchStore(path=tmp_path / "scratch.yaml")


def make_thread(
    title: str = "Test Thread",
    status: ThreadStatus = ThreadStatus.NASCENT,
    perspectives: list[DaemonPerspective] | None = None,
) -> Thread:
    return Thread(
        title=title,
        central_question=f"What is the nature of {title}?",
        status=status,
        current_state="ongoing exploration",
        unresolved="many things",
        stance=Stance(position="uncertain", epistemic_status=EpistemicStatus.LIVE),
        daemon_perspectives=perspectives or [],
    )


def _create_active_thread(store: ThreadStore, title: str = "Test Thread") -> Thread:
    """Create a thread in the store and transition it to ACTIVE."""
    t = store.create(make_thread(title))
    return store.transition(t.id, ThreadStatus.ACTIVE)


def make_result(
    route: IntegrationRoute = IntegrationRoute.DEVELOPS_EXISTING,
    fascination_topic: str | None = "recursion",
    thought: str = "a thought about recursion",
) -> IntegrationResult:
    return IntegrationResult(
        route=route,
        fascination_topic=fascination_topic,
        lifecycle_promoted=False,
        thought_content=thought,
    )


def _always_relevant(prompt: str) -> str:
    if "Reply with exactly one word" in prompt:
        return "RELEVANT"
    return "A reflection connecting the thought to the thread."


def _never_relevant(prompt: str) -> str:
    if "Reply with exactly one word" in prompt:
        return "NOT_RELEVANT"
    return "A reflection."


# ---------------------------------------------------------------------------
# _is_duplicate helper
# ---------------------------------------------------------------------------


def test_is_duplicate_false_when_no_perspectives() -> None:
    thread = make_thread(status=ThreadStatus.ACTIVE)
    assert _is_duplicate(thread, "recursion", _NOW, 24) is False


def test_is_duplicate_true_within_window() -> None:
    recent = _NOW - timedelta(hours=12)
    p = DaemonPerspective(
        content="a perspective",
        from_fascination="recursion",
        written_at=recent.isoformat(),
        thread_status_at_writing=ThreadStatus.ACTIVE,
    )
    thread = make_thread(status=ThreadStatus.ACTIVE, perspectives=[p])
    assert _is_duplicate(thread, "recursion", _NOW, 24) is True


def test_is_duplicate_false_outside_window() -> None:
    old = _NOW - timedelta(hours=30)
    p = DaemonPerspective(
        content="old perspective",
        from_fascination="recursion",
        written_at=old.isoformat(),
        thread_status_at_writing=ThreadStatus.ACTIVE,
    )
    thread = make_thread(status=ThreadStatus.ACTIVE, perspectives=[p])
    assert _is_duplicate(thread, "recursion", _NOW, 24) is False


def test_is_duplicate_case_insensitive() -> None:
    recent = _NOW - timedelta(hours=1)
    p = DaemonPerspective(
        content="cap perspective",
        from_fascination="Recursion",
        written_at=recent.isoformat(),
        thread_status_at_writing=ThreadStatus.ACTIVE,
    )
    thread = make_thread(status=ThreadStatus.ACTIVE, perspectives=[p])
    assert _is_duplicate(thread, "recursion", _NOW, 24) is True


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------


def test_skips_when_route_is_inert(
    thread_store: ThreadStore, scratch_store: ScratchStore
) -> None:
    result = make_result(route=IntegrationRoute.INERT, fascination_topic=None)
    out = inner_life_thread_pollination(
        result,
        thread_store=thread_store,
        scratch_store=scratch_store,
        inference_fn=_always_relevant,
        now=_NOW,
    )
    assert out.skipped_no_fascination is True
    assert out.threads_pollinated == []
    assert out.signal_written is False


def test_skips_when_route_is_aesthetic(
    thread_store: ThreadStore, scratch_store: ScratchStore
) -> None:
    result = make_result(
        route=IntegrationRoute.AESTHETIC_REACTION, fascination_topic=None
    )
    out = inner_life_thread_pollination(
        result,
        thread_store=thread_store,
        scratch_store=scratch_store,
        inference_fn=_always_relevant,
        now=_NOW,
    )
    assert out.skipped_no_fascination is True


# ---------------------------------------------------------------------------
# No threads
# ---------------------------------------------------------------------------


def test_no_threads_returns_empty_result(
    thread_store: ThreadStore, scratch_store: ScratchStore
) -> None:
    result = make_result()
    out = inner_life_thread_pollination(
        result,
        thread_store=thread_store,
        scratch_store=scratch_store,
        inference_fn=_always_relevant,
        now=_NOW,
    )
    assert out.threads_pollinated == []
    assert out.signal_written is False


# ---------------------------------------------------------------------------
# Relevant thread gets perspective
# ---------------------------------------------------------------------------


def test_relevant_thread_receives_perspective(
    thread_store: ThreadStore, scratch_store: ScratchStore
) -> None:
    t = _create_active_thread(thread_store, "Recursion Thread")
    result = make_result(fascination_topic="recursion")

    calls: list[str] = []

    def inference_fn(prompt: str) -> str:
        calls.append(prompt)
        if "Reply with exactly one word" in prompt:
            return "RELEVANT"
        return "This thought connects recursion to the thread's central question."

    out = inner_life_thread_pollination(
        result,
        thread_store=thread_store,
        scratch_store=scratch_store,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert t.id in out.threads_pollinated

    updated = thread_store.load(t.id)
    assert len(updated.daemon_perspectives) == 1
    assert updated.daemon_perspectives[0].from_fascination == "recursion"


def test_not_relevant_thread_not_pollinated(
    thread_store: ThreadStore, scratch_store: ScratchStore
) -> None:
    t = _create_active_thread(thread_store, "Unrelated Thread")
    result = make_result(fascination_topic="recursion")

    out = inner_life_thread_pollination(
        result,
        thread_store=thread_store,
        scratch_store=scratch_store,
        inference_fn=_never_relevant,
        now=_NOW,
    )
    assert t.id not in out.threads_pollinated


# ---------------------------------------------------------------------------
# Dormant threads are included
# ---------------------------------------------------------------------------


def test_dormant_thread_is_candidate(
    thread_store: ThreadStore, scratch_store: ScratchStore
) -> None:
    t = _create_active_thread(thread_store, "Dormant Thread")
    thread_store.transition(t.id, ThreadStatus.DORMANT)
    result = make_result(fascination_topic="recursion")

    out = inner_life_thread_pollination(
        result,
        thread_store=thread_store,
        scratch_store=scratch_store,
        inference_fn=_always_relevant,
        now=_NOW,
    )
    assert t.id in out.threads_pollinated


# ---------------------------------------------------------------------------
# Archived threads are not candidates
# ---------------------------------------------------------------------------


def test_archived_thread_is_not_candidate(
    thread_store: ThreadStore, scratch_store: ScratchStore
) -> None:
    t = _create_active_thread(thread_store, "Archived Thread")
    thread_store.transition(t.id, ThreadStatus.DORMANT)
    thread_store.transition(t.id, ThreadStatus.ARCHIVED)
    result = make_result(fascination_topic="recursion")

    out = inner_life_thread_pollination(
        result,
        thread_store=thread_store,
        scratch_store=scratch_store,
        inference_fn=_always_relevant,
        now=_NOW,
    )
    assert t.id not in out.threads_pollinated


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_deduplication_prevents_duplicate_perspective(
    thread_store: ThreadStore, scratch_store: ScratchStore
) -> None:
    t = _create_active_thread(thread_store, "Dup Thread")

    recent = _NOW - timedelta(hours=1)
    p = DaemonPerspective(
        content="existing perspective",
        from_fascination="recursion",
        written_at=recent.isoformat(),
        thread_status_at_writing=ThreadStatus.ACTIVE,
    )
    thread_store.add_perspective(t.id, p)

    result = make_result(fascination_topic="recursion")

    out = inner_life_thread_pollination(
        result,
        thread_store=thread_store,
        scratch_store=scratch_store,
        inference_fn=_always_relevant,
        now=_NOW,
    )
    assert t.id not in out.threads_pollinated

    updated = thread_store.load(t.id)
    assert len(updated.daemon_perspectives) == 1


# ---------------------------------------------------------------------------
# High-significance signal written to scratch
# ---------------------------------------------------------------------------


def test_signal_written_when_threads_pollinated(
    thread_store: ThreadStore, scratch_store: ScratchStore
) -> None:
    t = _create_active_thread(thread_store, "Signal Thread")
    result = make_result(fascination_topic="recursion")

    out = inner_life_thread_pollination(
        result,
        thread_store=thread_store,
        scratch_store=scratch_store,
        inference_fn=_always_relevant,
        now=_NOW,
    )
    assert out.signal_written is True

    notes = list(scratch_store._notes.values())
    assert len(notes) == 1
    note = notes[0]
    assert note.type == ScratchType.SIGNAL
    assert note.target_workflow == "inner_life_push_evaluation"
    assert note.ttl is not None
    assert t.id in note.thread_ids


def test_no_signal_when_no_threads_pollinated(
    thread_store: ThreadStore, scratch_store: ScratchStore
) -> None:
    _create_active_thread(thread_store, "Irrelevant Thread")
    result = make_result(fascination_topic="recursion")

    out = inner_life_thread_pollination(
        result,
        thread_store=thread_store,
        scratch_store=scratch_store,
        inference_fn=_never_relevant,
        now=_NOW,
    )
    assert out.signal_written is False
    assert list(scratch_store._notes.values()) == []


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_dedup_and_signal_ttl_constants() -> None:
    assert POLLINATION_DEDUP_HOURS == 24
    assert POLLINATION_SIGNAL_TTL_HOURS == 24
