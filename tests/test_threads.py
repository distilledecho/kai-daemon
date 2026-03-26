"""Tests for the thread store (§4f, §9)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from kai_daemon.state.threads import (
    DaemonPerspective,
    EpistemicStatus,
    HandoffNote,
    PickupNote,
    Stance,
    Thread,
    ThreadStatus,
    ThreadStore,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ThreadStore:
    threads = tmp_path / "threads"
    pickups = tmp_path / "pickup_notes"
    threads.mkdir()
    pickups.mkdir()
    return ThreadStore(threads_path=threads, pickup_notes_path=pickups)


def _stance(
    position: str = "still working it out",
    epistemic_status: EpistemicStatus = EpistemicStatus.LIVE,
) -> Stance:
    return Stance(position=position, epistemic_status=epistemic_status)


def _thread(
    title: str = "Test thread",
    central_question: str = "What does this mean?",
    current_state: str = "Early exploration",
    unresolved: str = "Much remains open",
    status: ThreadStatus = ThreadStatus.NASCENT,
    **kwargs: Any,
) -> Thread:
    return Thread(
        title=title,
        central_question=central_question,
        current_state=current_state,
        unresolved=unresolved,
        status=status,
        stance=_stance(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Thread model
# ---------------------------------------------------------------------------


def test_thread_defaults():
    t = _thread()
    assert t.status == ThreadStatus.NASCENT
    assert t.dormant_since is None
    assert t.daemon_perspectives == []
    assert t.handoff_notes == []
    assert t.key_tension is None
    assert t.daemon_is_watching is None


def test_thread_id_auto_assigned():
    t1 = _thread()
    t2 = _thread()
    assert t1.id != t2.id


def test_thread_frozen():
    t = _thread()
    with pytest.raises((TypeError, ValidationError)):
        t.title = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Stance epistemic_status values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        EpistemicStatus.LIVE,
        EpistemicStatus.ACCEPTED,
        EpistemicStatus.REJECTED,
        EpistemicStatus.SUSPENDED,
        EpistemicStatus.UNCERTAIN,
    ],
)
def test_stance_all_epistemic_statuses(status: EpistemicStatus):
    s = Stance(position="x", epistemic_status=status)
    assert s.epistemic_status == status


# ---------------------------------------------------------------------------
# DaemonPerspective validation
# ---------------------------------------------------------------------------


def test_perspective_valid_active():
    p = DaemonPerspective(
        content="interesting",
        from_fascination="topic-x",
        thread_status_at_writing=ThreadStatus.ACTIVE,
    )
    assert p.thread_status_at_writing == ThreadStatus.ACTIVE


def test_perspective_valid_dormant():
    p = DaemonPerspective(
        content="interesting",
        from_fascination="topic-x",
        thread_status_at_writing=ThreadStatus.DORMANT,
    )
    assert p.thread_status_at_writing == ThreadStatus.DORMANT


@pytest.mark.parametrize("bad_status", [ThreadStatus.NASCENT, ThreadStatus.ARCHIVED])
def test_perspective_invalid_status(bad_status: ThreadStatus):
    with pytest.raises(ValueError, match="thread_status_at_writing"):
        DaemonPerspective(
            content="x",
            from_fascination="f",
            thread_status_at_writing=bad_status,
        )


# ---------------------------------------------------------------------------
# Create / load
# ---------------------------------------------------------------------------


def test_create_and_load(store: ThreadStore):
    t = _thread()
    stored = store.create(t)
    loaded = store.load(stored.id)
    assert loaded.id == stored.id
    assert loaded.title == stored.title
    assert loaded.central_question == stored.central_question


def test_create_duplicate_raises(store: ThreadStore):
    t = _thread()
    store.create(t)
    with pytest.raises(ValueError, match="already exists"):
        store.create(t)


def test_load_missing_raises(store: ThreadStore):
    with pytest.raises(KeyError):
        store.load("nonexistent-id")


# ---------------------------------------------------------------------------
# list_all / list_by_status
# ---------------------------------------------------------------------------


def test_list_all_empty(store: ThreadStore):
    assert store.list_all() == []


def test_list_all_returns_all(store: ThreadStore):
    store.create(_thread(title="A"))
    store.create(_thread(title="B"))
    assert len(store.list_all()) == 2


def test_list_by_status_filters(store: ThreadStore):
    t = store.create(_thread())
    store.transition(t.id, ThreadStatus.ACTIVE)
    nascent = store.list_by_status(ThreadStatus.NASCENT)
    active = store.list_by_status(ThreadStatus.ACTIVE)
    assert len(nascent) == 0
    assert len(active) == 1


# ---------------------------------------------------------------------------
# State transitions — valid paths
# ---------------------------------------------------------------------------


def test_transition_nascent_to_active(store: ThreadStore):
    t = store.create(_thread())
    updated = store.transition(t.id, ThreadStatus.ACTIVE)
    assert updated.status == ThreadStatus.ACTIVE


def test_transition_active_to_dormant(store: ThreadStore):
    t = store.create(_thread())
    store.transition(t.id, ThreadStatus.ACTIVE)
    updated = store.transition(t.id, ThreadStatus.DORMANT)
    assert updated.status == ThreadStatus.DORMANT
    assert updated.dormant_since is not None


def test_transition_dormant_to_active(store: ThreadStore):
    t = store.create(_thread())
    store.transition(t.id, ThreadStatus.ACTIVE)
    store.transition(t.id, ThreadStatus.DORMANT)
    updated = store.transition(t.id, ThreadStatus.ACTIVE)
    assert updated.status == ThreadStatus.ACTIVE
    assert updated.dormant_since is None


def test_transition_dormant_to_archived(store: ThreadStore):
    t = store.create(_thread())
    store.transition(t.id, ThreadStatus.ACTIVE)
    store.transition(t.id, ThreadStatus.DORMANT)
    updated = store.transition(t.id, ThreadStatus.ARCHIVED)
    assert updated.status == ThreadStatus.ARCHIVED


# ---------------------------------------------------------------------------
# State transitions — invalid paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        (ThreadStatus.NASCENT, ThreadStatus.DORMANT),
        (ThreadStatus.NASCENT, ThreadStatus.ARCHIVED),
        (ThreadStatus.ACTIVE, ThreadStatus.NASCENT),
        (ThreadStatus.ACTIVE, ThreadStatus.ARCHIVED),
        (ThreadStatus.DORMANT, ThreadStatus.NASCENT),
    ],
)
def test_invalid_transition_raises(
    store: ThreadStore, path: tuple[ThreadStatus, ThreadStatus]
):
    start, target = path
    t = _thread(status=start)
    # For non-nascent starts, create with forced status via model_copy
    if start == ThreadStatus.NASCENT:
        stored = store.create(t)
    else:
        nascent = store.create(_thread())
        # Advance to the desired starting state
        if start == ThreadStatus.ACTIVE:
            store.transition(nascent.id, ThreadStatus.ACTIVE)
            stored = store.load(nascent.id)
        elif start == ThreadStatus.DORMANT:
            store.transition(nascent.id, ThreadStatus.ACTIVE)
            store.transition(nascent.id, ThreadStatus.DORMANT)
            stored = store.load(nascent.id)
        else:
            stored = nascent
    with pytest.raises(ValueError, match="Invalid transition"):
        store.transition(stored.id, target)


def test_archived_is_terminal(store: ThreadStore):
    t = store.create(_thread())
    store.transition(t.id, ThreadStatus.ACTIVE)
    store.transition(t.id, ThreadStatus.DORMANT)
    store.transition(t.id, ThreadStatus.ARCHIVED)
    for target in ThreadStatus:
        with pytest.raises(ValueError, match="Invalid transition"):
            store.transition(t.id, target)


# ---------------------------------------------------------------------------
# active → archived is invalid (must go dormant first)
# ---------------------------------------------------------------------------


def test_active_cannot_go_directly_to_archived(store: ThreadStore):
    t = store.create(_thread())
    store.transition(t.id, ThreadStatus.ACTIVE)
    with pytest.raises(ValueError, match="Invalid transition"):
        store.transition(t.id, ThreadStatus.ARCHIVED)


# ---------------------------------------------------------------------------
# dormant_since lifecycle
# ---------------------------------------------------------------------------


def test_dormant_since_set_on_dormant(store: ThreadStore):
    t = store.create(_thread())
    store.transition(t.id, ThreadStatus.ACTIVE)
    dormant = store.transition(t.id, ThreadStatus.DORMANT)
    assert dormant.dormant_since is not None


def test_dormant_since_cleared_on_resurface(store: ThreadStore):
    t = store.create(_thread())
    store.transition(t.id, ThreadStatus.ACTIVE)
    store.transition(t.id, ThreadStatus.DORMANT)
    resurfaced = store.transition(t.id, ThreadStatus.ACTIVE)
    assert resurfaced.dormant_since is None


# ---------------------------------------------------------------------------
# Perspectives
# ---------------------------------------------------------------------------


def test_add_perspective(store: ThreadStore):
    t = store.create(_thread())
    p = DaemonPerspective(
        content="a thought",
        from_fascination="fascination-1",
        thread_status_at_writing=ThreadStatus.ACTIVE,
    )
    updated = store.add_perspective(t.id, p)
    assert len(updated.daemon_perspectives) == 1
    assert updated.daemon_perspectives[0].content == "a thought"


def test_add_multiple_perspectives(store: ThreadStore):
    t = store.create(_thread())
    for i in range(3):
        p = DaemonPerspective(
            content=f"thought {i}",
            from_fascination="f",
            thread_status_at_writing=ThreadStatus.ACTIVE,
        )
        store.add_perspective(t.id, p)
    loaded = store.load(t.id)
    assert len(loaded.daemon_perspectives) == 3


# ---------------------------------------------------------------------------
# Handoff notes
# ---------------------------------------------------------------------------


def test_add_handoff_note(store: ThreadStore):
    t = store.create(_thread())
    note = HandoffNote(content="orientation for next session")
    updated = store.add_handoff_note(t.id, note)
    assert len(updated.handoff_notes) == 1
    assert updated.handoff_notes[0].content == "orientation for next session"


def test_add_multiple_handoff_notes(store: ThreadStore):
    t = store.create(_thread())
    for i in range(4):
        store.add_handoff_note(t.id, HandoffNote(content=f"note {i}"))
    loaded = store.load(t.id)
    assert len(loaded.handoff_notes) == 4


def test_handoff_notes_persist(store: ThreadStore):
    t = store.create(_thread())
    store.add_handoff_note(t.id, HandoffNote(content="session one"))
    store.add_handoff_note(t.id, HandoffNote(content="session two"))
    loaded = store.load(t.id)
    assert loaded.handoff_notes[0].content == "session one"
    assert loaded.handoff_notes[1].content == "session two"


# ---------------------------------------------------------------------------
# Handoff notes — ChromaDB embedding
# ---------------------------------------------------------------------------


def test_handoff_note_embedded_in_chroma():
    mock_chroma = MagicMock()
    mock_cq = MagicMock()
    mock_hn = MagicMock()
    mock_chroma.get_or_create_collection.side_effect = [mock_cq, mock_hn]
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        s = ThreadStore(
            threads_path=tmp / "threads",
            pickup_notes_path=tmp / "pickups",
            chroma_client=mock_chroma,
        )
        t = s.create(_thread())
        note = HandoffNote(content="handoff content")
        s.add_handoff_note(t.id, note)

    # upsert called on the handoff notes collection
    assert mock_hn.upsert.called
    call_kwargs = mock_hn.upsert.call_args
    assert call_kwargs.kwargs["documents"] == ["handoff content"]
    ids = call_kwargs.kwargs["ids"]
    assert ids[0].startswith(t.id + ":")
    meta = call_kwargs.kwargs["metadatas"][0]
    assert meta["thread_id"] == t.id


# ---------------------------------------------------------------------------
# Central question — ChromaDB embedding
# ---------------------------------------------------------------------------


def test_central_question_embedded_on_create():
    mock_chroma = MagicMock()
    mock_cq = MagicMock()
    mock_hn = MagicMock()
    mock_chroma.get_or_create_collection.side_effect = [mock_cq, mock_hn]
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        s = ThreadStore(
            threads_path=tmp / "threads",
            pickup_notes_path=tmp / "pickups",
            chroma_client=mock_chroma,
        )
        t = _thread(central_question="What is going on here?")
        s.create(t)

    assert mock_cq.upsert.called
    call_kwargs = mock_cq.upsert.call_args
    assert call_kwargs.kwargs["documents"] == ["What is going on here?"]
    assert call_kwargs.kwargs["ids"] == [t.id]
    assert call_kwargs.kwargs["metadatas"][0]["thread_id"] == t.id


# ---------------------------------------------------------------------------
# Pickup notes — time_gap_quality null at write time
# ---------------------------------------------------------------------------


def test_write_pickup_note_requires_null_time_gap_quality(store: ThreadStore):
    t = store.create(_thread())
    note = PickupNote(
        thread_id=t.id,
        content="orientation",
        dormant_since="2026-01-01T00:00:00+00:00",
        time_gap_quality="rich",  # must not be set at write time
    )
    with pytest.raises(ValueError, match="time_gap_quality must be null"):
        store.write_pickup_note(note)


def test_write_pickup_note_null_time_gap_quality_succeeds(store: ThreadStore):
    t = store.create(_thread())
    note = PickupNote(
        thread_id=t.id,
        content="orientation",
        dormant_since="2026-01-01T00:00:00+00:00",
    )
    stored = store.write_pickup_note(note)
    assert stored.time_gap_quality is None


def test_pickup_note_persists(store: ThreadStore):
    t = store.create(_thread())
    note = PickupNote(
        thread_id=t.id,
        content="pick up here next time",
        dormant_since="2026-01-01T00:00:00+00:00",
    )
    store.write_pickup_note(note)
    loaded = store.load_pickup_note(t.id)
    assert loaded.content == "pick up here next time"
    assert loaded.time_gap_quality is None


def test_write_pickup_note_requires_existing_thread(store: ThreadStore):
    note = PickupNote(
        thread_id="nonexistent",
        content="x",
        dormant_since="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(KeyError):
        store.write_pickup_note(note)


# ---------------------------------------------------------------------------
# fill_time_gap_quality — thread_pickup resumption path
# ---------------------------------------------------------------------------


def test_fill_time_gap_quality(store: ThreadStore):
    t = store.create(_thread())
    note = PickupNote(
        thread_id=t.id,
        content="orientation",
        dormant_since="2026-01-01T00:00:00+00:00",
    )
    store.write_pickup_note(note)
    filled = store.fill_time_gap_quality(t.id, "The gap gave useful distance")
    assert filled.time_gap_quality == "The gap gave useful distance"


def test_fill_time_gap_quality_persists(store: ThreadStore):
    t = store.create(_thread())
    note = PickupNote(
        thread_id=t.id,
        content="orientation",
        dormant_since="2026-01-01T00:00:00+00:00",
    )
    store.write_pickup_note(note)
    store.fill_time_gap_quality(t.id, "quality assessment")
    reloaded = store.load_pickup_note(t.id)
    assert reloaded.time_gap_quality == "quality assessment"


def test_fill_time_gap_quality_missing_note_raises(store: ThreadStore):
    t = store.create(_thread())
    with pytest.raises(KeyError):
        store.fill_time_gap_quality(t.id, "quality")


# ---------------------------------------------------------------------------
# load_pickup_note missing raises
# ---------------------------------------------------------------------------


def test_load_pickup_note_missing_raises(store: ThreadStore):
    with pytest.raises(KeyError):
        store.load_pickup_note("nonexistent")


# ---------------------------------------------------------------------------
# ChromaDB failure is non-fatal
# ---------------------------------------------------------------------------


def test_chroma_failure_does_not_block_create():
    mock_chroma = MagicMock()
    mock_cq = MagicMock()
    mock_hn = MagicMock()
    mock_cq.upsert.side_effect = RuntimeError("server down")
    mock_chroma.get_or_create_collection.side_effect = [mock_cq, mock_hn]
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        s = ThreadStore(
            threads_path=tmp / "threads",
            pickup_notes_path=tmp / "pickups",
            chroma_client=mock_chroma,
        )
        # Should not raise even though ChromaDB is broken
        t = s.create(_thread())
        loaded = s.load(t.id)
        assert loaded.id == t.id


def test_chroma_failure_does_not_block_handoff_note():
    mock_chroma = MagicMock()
    mock_cq = MagicMock()
    mock_hn = MagicMock()
    mock_hn.upsert.side_effect = RuntimeError("server down")
    mock_chroma.get_or_create_collection.side_effect = [mock_cq, mock_hn]
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        s = ThreadStore(
            threads_path=tmp / "threads",
            pickup_notes_path=tmp / "pickups",
            chroma_client=mock_chroma,
        )
        t = s.create(_thread())
        updated = s.add_handoff_note(t.id, HandoffNote(content="note"))
        assert len(updated.handoff_notes) == 1


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


def test_update_thread(store: ThreadStore):
    t = store.create(_thread())
    modified = t.model_copy(update={"current_state": "deeper understanding now"})
    store.update(modified)
    loaded = store.load(t.id)
    assert loaded.current_state == "deeper understanding now"


def test_update_missing_raises(store: ThreadStore):
    t = _thread()
    with pytest.raises(KeyError):
        store.update(t)


# ---------------------------------------------------------------------------
# Structural separation — threads.py has no reference to DAEMON_SELF or
# DAEMON_RELATIONAL model types (§4f isolation)
# ---------------------------------------------------------------------------


def test_thread_store_no_daemon_self_reference():
    for name, fn in inspect.getmembers(ThreadStore, predicate=inspect.isfunction):
        hints = fn.__annotations__
        for hint in hints.values():
            if isinstance(hint, str):
                assert "DaemonSelf" not in hint, (
                    f"ThreadStore.{name} annotation references DaemonSelf"
                )
            elif hasattr(hint, "__name__"):
                assert hint.__name__ != "DaemonSelf", (
                    f"ThreadStore.{name} annotation references DaemonSelf"
                )


def test_thread_store_no_daemon_relational_reference():
    for name, fn in inspect.getmembers(ThreadStore, predicate=inspect.isfunction):
        hints = fn.__annotations__
        for hint in hints.values():
            if isinstance(hint, str):
                assert "DaemonRelational" not in hint, (
                    f"ThreadStore.{name} annotation references DaemonRelational"
                )
            elif hasattr(hint, "__name__"):
                assert hint.__name__ != "DaemonRelational", (
                    f"ThreadStore.{name} annotation references DaemonRelational"
                )
