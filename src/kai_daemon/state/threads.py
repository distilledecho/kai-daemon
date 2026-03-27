"""Thread store — conversation thread lifecycle (§4f, §9).

State-transition invariants (§9a):

- ``nascent → active``
- ``active → dormant``
- ``dormant → active``  (resurfaced)
- ``dormant → archived``  (explicit decision only — never automatic)

All other transitions raise ``ValueError``.

Pickup-note invariant (§9b):

- ``time_gap_quality`` is null at write time.
- ``ThreadStore.write_pickup_note()`` raises ``ValueError`` if a caller
  passes a note with a non-null ``time_gap_quality``.
- Only ``ThreadStore.fill_time_gap_quality()`` may set it, modelling what
  ``thread_pickup`` does at actual resumption.

ChromaDB (§1E, §9c):

- ``central_question`` embedding stored per thread (direct retrieval path).
  Document ID: ``{thread_id}``.
- Each handoff note embedded with ``thread_id`` in metadata (indirect path).
  Document ID: ``{thread_id}:{note_index}``.
- ChromaDB failures warn but never block file writes.
"""

from __future__ import annotations

import logging
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._chroma import (
    THREAD_CENTRAL_QUESTIONS_COLLECTION,
    THREAD_HANDOFF_NOTES_COLLECTION,
)
from ._paths import pickup_notes_dir, threads_dir
from ._utils import _utcnow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ThreadStatus(StrEnum):
    NASCENT = "nascent"
    ACTIVE = "active"
    DORMANT = "dormant"
    ARCHIVED = "archived"


class EpistemicStatus(StrEnum):
    LIVE = "live"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    UNCERTAIN = "uncertain"


# Allowed target states for each source state (§9a).
_VALID_TRANSITIONS: dict[ThreadStatus, frozenset[ThreadStatus]] = {
    ThreadStatus.NASCENT: frozenset({ThreadStatus.ACTIVE}),
    ThreadStatus.ACTIVE: frozenset({ThreadStatus.DORMANT}),
    ThreadStatus.DORMANT: frozenset({ThreadStatus.ACTIVE, ThreadStatus.ARCHIVED}),
    ThreadStatus.ARCHIVED: frozenset(),
}


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Stance(BaseModel):
    """The daemon's epistemic position on a thread."""

    position: str
    epistemic_status: EpistemicStatus

    model_config = ConfigDict(frozen=True)


class DaemonPerspective(BaseModel):
    """Daemon perspective on a thread, sourced from a fascination."""

    content: str
    from_fascination: str
    written_at: str = Field(default_factory=_utcnow)
    thread_status_at_writing: ThreadStatus
    surfaced: bool = False

    @model_validator(mode="after")
    def _validate_writing_status(self) -> DaemonPerspective:
        allowed = {ThreadStatus.ACTIVE, ThreadStatus.DORMANT}
        if self.thread_status_at_writing not in allowed:
            raise ValueError(
                "thread_status_at_writing must be 'active' or 'dormant', "
                f"got {self.thread_status_at_writing!r}"
            )
        return self

    model_config = ConfigDict(frozen=True)


class HandoffNote(BaseModel):
    """A session-end orientation note written for the next resumption of this thread."""

    content: str
    written_at: str = Field(default_factory=_utcnow)

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Thread model
# ---------------------------------------------------------------------------


class Thread(BaseModel):
    """A single conversation thread.

    ``id`` is assigned automatically on creation.
    ``status`` advances only through ``ThreadStore.transition()``.
    ``handoff_notes`` grow only through ``ThreadStore.add_handoff_note()``.
    ``daemon_perspectives`` grow only through ``ThreadStore.add_perspective()``.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    central_question: str
    status: ThreadStatus = ThreadStatus.NASCENT
    created: str = Field(default_factory=_utcnow)
    last_touched: str = Field(default_factory=_utcnow)
    dormant_since: str | None = None
    current_state: str
    unresolved: str
    key_tension: str | None = None
    stance: Stance
    daemon_is_watching: str | None = None
    daemon_perspectives: list[DaemonPerspective] = []
    handoff_notes: list[HandoffNote] = []

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Pickup note
# ---------------------------------------------------------------------------


class PickupNote(BaseModel):
    """Orientation note written at dormancy by ``dormant_thread_writer``.

    ``time_gap_quality`` is null at write time; ``thread_pickup`` fills it
    at actual resumption via ``ThreadStore.fill_time_gap_quality()``.
    """

    thread_id: str
    content: str
    written_at: str = Field(default_factory=_utcnow)
    dormant_since: str
    time_gap_quality: str | None = None

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ThreadStore:
    """File-backed thread store.

    One YAML file per thread under ``threads_dir()``.
    One YAML file per pickup note under ``pickup_notes_dir()``.

    Inject ``threads_path`` / ``pickup_notes_path`` in tests to use a
    temporary directory.  Inject ``chroma_client=None`` to skip ChromaDB.
    """

    def __init__(
        self,
        threads_path: Path | None = None,
        pickup_notes_path: Path | None = None,
        chroma_client: Any | None = None,
    ) -> None:
        self._threads_path = threads_path or threads_dir()
        self._pickup_notes_path = pickup_notes_path or pickup_notes_dir()
        self._threads_path.mkdir(parents=True, exist_ok=True)
        self._pickup_notes_path.mkdir(parents=True, exist_ok=True)
        self._chroma = chroma_client
        self._cq_collection: Any | None = None
        self._hn_collection: Any | None = None
        if self._chroma is not None:
            self._cq_collection = self._chroma.get_or_create_collection(
                THREAD_CENTRAL_QUESTIONS_COLLECTION
            )
            self._hn_collection = self._chroma.get_or_create_collection(
                THREAD_HANDOFF_NOTES_COLLECTION
            )

    # ------------------------------------------------------------------
    # Internal persistence helpers
    # ------------------------------------------------------------------

    def _thread_path(self, thread_id: str) -> Path:
        return self._threads_path / f"{thread_id}.yaml"

    def _pickup_path(self, thread_id: str) -> Path:
        return self._pickup_notes_path / f"{thread_id}.yaml"

    def _write_thread(self, thread: Thread) -> None:
        self._thread_path(thread.id).write_text(
            yaml.dump(
                thread.model_dump(mode="json"), allow_unicode=True, sort_keys=False
            )
        )

    def _read_thread(self, thread_id: str) -> Thread:
        path = self._thread_path(thread_id)
        if not path.exists():
            raise KeyError(f"Thread {thread_id!r} not found")
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        return Thread.model_validate(raw)

    def _write_pickup(self, note: PickupNote) -> None:
        self._pickup_path(note.thread_id).write_text(
            yaml.dump(note.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
        )

    def _read_pickup(self, thread_id: str) -> PickupNote:
        path = self._pickup_path(thread_id)
        if not path.exists():
            raise KeyError(f"Pickup note for thread {thread_id!r} not found")
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        return PickupNote.model_validate(raw)

    # ------------------------------------------------------------------
    # ChromaDB helpers (best-effort)
    # ------------------------------------------------------------------

    def _embed_central_question(self, thread: Thread) -> None:
        if self._cq_collection is None:
            return
        try:
            self._cq_collection.upsert(
                documents=[thread.central_question],
                ids=[thread.id],
                metadatas=[{"thread_id": thread.id, "title": thread.title}],
            )
        except Exception:
            logger.warning(
                "Failed to embed central_question for thread %s",
                thread.id,
                exc_info=True,
            )

    def _embed_handoff_note(
        self, thread_id: str, note: HandoffNote, index: int
    ) -> None:
        if self._hn_collection is None:
            return
        try:
            self._hn_collection.upsert(
                documents=[note.content],
                ids=[f"{thread_id}:{index}"],
                metadatas=[{"thread_id": thread_id, "written_at": note.written_at}],
            )
        except Exception:
            logger.warning(
                "Failed to embed handoff note %d for thread %s",
                index,
                thread_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Create / read / update
    # ------------------------------------------------------------------

    def create(self, thread: Thread) -> Thread:
        """Persist a new thread.

        Raises ``ValueError`` if a thread with this ID already exists.
        Embeds ``central_question`` in ChromaDB (best-effort).
        """
        if self._thread_path(thread.id).exists():
            raise ValueError(f"Thread {thread.id!r} already exists")
        self._write_thread(thread)
        self._embed_central_question(thread)
        return thread

    def load(self, thread_id: str) -> Thread:
        """Return the thread with the given ID.

        Raises ``KeyError`` if not found.
        """
        return self._read_thread(thread_id)

    def update(self, thread: Thread) -> Thread:
        """Overwrite the stored thread document.

        The caller is responsible for building the updated ``Thread`` via
        ``thread.model_copy(update={...})``.  Re-embeds ``central_question``
        on every update (upsert is idempotent, best-effort).
        """
        if not self._thread_path(thread.id).exists():
            raise KeyError(f"Thread {thread.id!r} not found")
        self._write_thread(thread)
        self._embed_central_question(thread)
        return thread

    def list_all(self) -> list[Thread]:
        """Return all threads."""
        threads: list[Thread] = []
        for path in self._threads_path.glob("*.yaml"):
            raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
            threads.append(Thread.model_validate(raw))
        return threads

    def list_by_status(self, status: ThreadStatus) -> list[Thread]:
        """Return threads with the given status."""
        return [t for t in self.list_all() if t.status == status]

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def transition(self, thread_id: str, new_status: ThreadStatus) -> Thread:
        """Advance a thread's status along an allowed transition.

        Allowed transitions (§9a):

        - ``nascent → active``
        - ``active → dormant``
        - ``dormant → active``  (resurfaced)
        - ``dormant → archived``  (explicit only)

        Raises ``ValueError`` for any other transition.
        Sets ``dormant_since`` when entering dormant; clears it on exit.
        Updates ``last_touched``.
        """
        thread = self._read_thread(thread_id)
        allowed = _VALID_TRANSITIONS[thread.status]
        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition: {thread.status!r} → {new_status!r}. "
                f"Allowed from {thread.status!r}: "
                f"{sorted(s.value for s in allowed) or 'none (terminal)'}"
            )

        updates: dict[str, Any] = {
            "status": new_status,
            "last_touched": _utcnow(),
        }
        if new_status == ThreadStatus.DORMANT:
            updates["dormant_since"] = _utcnow()
        elif (
            thread.status == ThreadStatus.DORMANT and new_status == ThreadStatus.ACTIVE
        ):
            updates["dormant_since"] = None

        updated = thread.model_copy(update=updates)
        self._write_thread(updated)
        return updated

    # ------------------------------------------------------------------
    # Perspectives
    # ------------------------------------------------------------------

    def add_perspective(self, thread_id: str, perspective: DaemonPerspective) -> Thread:
        """Append a daemon perspective to a thread.

        Updates ``last_touched``.
        """
        thread = self._read_thread(thread_id)
        updated = thread.model_copy(
            update={
                "daemon_perspectives": [*thread.daemon_perspectives, perspective],
                "last_touched": _utcnow(),
            }
        )
        self._write_thread(updated)
        return updated

    # ------------------------------------------------------------------
    # Handoff notes
    # ------------------------------------------------------------------

    def add_handoff_note(self, thread_id: str, note: HandoffNote) -> Thread:
        """Append a session handoff note and embed it in ChromaDB.

        Embeds using ``{thread_id}:{note_index}`` as document ID so the
        indirect retrieval path (§9c) can find notes by thread ID.
        Updates ``last_touched``.
        """
        thread = self._read_thread(thread_id)
        new_notes = [*thread.handoff_notes, note]
        updated = thread.model_copy(
            update={
                "handoff_notes": new_notes,
                "last_touched": _utcnow(),
            }
        )
        self._write_thread(updated)
        self._embed_handoff_note(thread_id, note, index=len(new_notes) - 1)
        return updated

    # ------------------------------------------------------------------
    # Pickup notes
    # ------------------------------------------------------------------

    def write_pickup_note(self, note: PickupNote) -> PickupNote:
        """Persist a pickup note for a dormant thread.

        Raises ``ValueError`` if ``time_gap_quality`` is not null — this
        invariant (§9b) is enforced in code: quality is unknown at write
        time; only ``thread_pickup`` may fill it via
        ``fill_time_gap_quality()``.

        Raises ``KeyError`` if the thread does not exist.
        """
        if note.time_gap_quality is not None:
            raise ValueError(
                "time_gap_quality must be null when writing a pickup note; "
                "it is filled by thread_pickup at actual resumption"
            )
        # Verify thread exists
        self._read_thread(note.thread_id)
        self._write_pickup(note)
        return note

    def load_pickup_note(self, thread_id: str) -> PickupNote:
        """Return the pickup note for the given thread.

        Raises ``KeyError`` if not found.
        """
        return self._read_pickup(thread_id)

    def fill_time_gap_quality(self, thread_id: str, quality: str) -> PickupNote:
        """Set ``time_gap_quality`` on an existing pickup note.

        Models what ``thread_pickup`` does at actual resumption.
        Raises ``KeyError`` if the pickup note does not exist.
        """
        note = self._read_pickup(thread_id)
        updated = note.model_copy(update={"time_gap_quality": quality})
        self._write_pickup(updated)
        return updated
