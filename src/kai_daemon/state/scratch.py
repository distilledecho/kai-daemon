"""Typed scratch space — the shared workflow message bus (§4c).

All workflows read and write here. Key invariants enforced in code:

- ``epistemic_origin`` is set at write time and immutable thereafter (§5.5).
- ``lifecycle`` transitions only forward: active → archived (never reversed).
- Items with a past ``ttl`` are archived on next call to ``expire_ttl()``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ._paths import daemon_state_dir


class ScratchType(StrEnum):
    """Permitted types for scratch space items."""

    SIGNAL = "signal"
    OBSERVATION = "observation"
    INSIGHT = "insight"
    FLAG = "flag"
    CANDIDATE = "candidate"


class Lifecycle(StrEnum):
    """Lifecycle states for scratch space items. Transitions: active → archived only."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class EpistemicOrigin(StrEnum):
    """Where this item originated. Set at write time; immutable thereafter."""

    INTERNAL = "internal"
    EXTERNAL_SEARCH = "external_search"
    INNER_LIFE_PIPELINE = "inner_life_pipeline"


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class ScratchNote(BaseModel):
    """A single scratch space item."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    session_id: str
    timestamp: str = Field(default_factory=_utcnow)
    content: str
    type: ScratchType
    target_workflow: str | None = None
    ttl: str | None = None
    acknowledged_by: list[str] = Field(default_factory=list)
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    epistemic_origin: EpistemicOrigin
    thread_ids: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class ScratchStore:
    """File-backed scratch space store.

    ``epistemic_origin`` is fixed at write time and cannot be changed.
    ``lifecycle`` moves only forward: ``active`` → ``archived``.

    Thread-safe for single-process use (no external locking).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (daemon_state_dir() / "scratch.yaml")
        self._notes: dict[str, ScratchNote] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            raw: list[dict[str, Any]] = yaml.safe_load(self._path.read_text()) or []
            for item in raw:
                note = ScratchNote.model_validate(item)
                self._notes[note.id] = note

    def _save(self) -> None:
        # mode="json" serialises enums to their string values, not Python objects
        data = [note.model_dump(mode="json") for note in self._notes.values()]
        self._path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))

    # ------------------------------------------------------------------
    # Write / read
    # ------------------------------------------------------------------

    def write(self, note: ScratchNote) -> ScratchNote:
        """Persist a new scratch note. ``epistemic_origin`` is fixed here."""
        if note.id in self._notes:
            raise ValueError(
                f"Note {note.id!r} already exists. "
                "Use archive() or acknowledge() to mutate."
            )
        self._notes[note.id] = note
        self._save()
        return note

    def read(self, note_id: str) -> ScratchNote:
        """Return the note with the given ID."""
        if note_id not in self._notes:
            raise KeyError(f"Scratch note {note_id!r} not found")
        return self._notes[note_id]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def archive(self, note_id: str) -> ScratchNote:
        """Transition a note from active → archived.

        Raises ValueError if already archived (lifecycle is irreversible).
        """
        note = self.read(note_id)
        if note.lifecycle == Lifecycle.ARCHIVED:
            raise ValueError(f"Note {note_id!r} is already archived")
        updated = note.model_copy(update={"lifecycle": Lifecycle.ARCHIVED})
        self._notes[note_id] = updated
        self._save()
        return updated

    def unarchive(self, note_id: str) -> None:
        """Not permitted. Lifecycle is irreversible: active → archived only.

        Always raises ValueError.
        """
        raise ValueError(
            "Archived scratch notes cannot be returned to active lifecycle. "
            "Lifecycle transitions are irreversible."
        )

    # ------------------------------------------------------------------
    # Acknowledged_by
    # ------------------------------------------------------------------

    def acknowledge(self, note_id: str, workflow_id: str) -> ScratchNote:
        """Mark a note as acknowledged by a workflow. Idempotent."""
        note = self.read(note_id)
        if workflow_id in note.acknowledged_by:
            return note
        updated = note.model_copy(
            update={"acknowledged_by": [*note.acknowledged_by, workflow_id]}
        )
        self._notes[note_id] = updated
        self._save()
        return updated

    # ------------------------------------------------------------------
    # Mutable fields (epistemic_origin excluded)
    # ------------------------------------------------------------------

    _IMMUTABLE_FIELDS = frozenset(
        {"id", "workflow_id", "session_id", "timestamp", "epistemic_origin"}
    )

    def update_content(self, note_id: str, **kwargs: Any) -> ScratchNote:
        """Update allowed mutable fields.

        ``epistemic_origin`` and identity fields are always rejected.
        Use ``archive()`` / ``acknowledge()`` for those specific state changes.
        """
        for key in kwargs:
            if key in self._IMMUTABLE_FIELDS:
                raise ValueError(
                    f"Field {key!r} is immutable after write and cannot be modified. "
                    "epistemic_origin is set at write time and fixed forever."
                )
        note = self.read(note_id)
        updated = note.model_copy(update=kwargs)
        self._notes[note_id] = updated
        self._save()
        return updated

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_active(self) -> list[ScratchNote]:
        """Return all notes with lifecycle == active."""
        return [n for n in self._notes.values() if n.lifecycle == Lifecycle.ACTIVE]

    def list_by_session(self, session_id: str) -> list[ScratchNote]:
        """Return all notes for a given session_id."""
        return [n for n in self._notes.values() if n.session_id == session_id]

    def list_by_workflow(self, workflow_id: str) -> list[ScratchNote]:
        """Return all notes written by the given workflow."""
        return [n for n in self._notes.values() if n.workflow_id == workflow_id]

    # ------------------------------------------------------------------
    # TTL expiry
    # ------------------------------------------------------------------

    def expire_ttl(self) -> int:
        """Archive all active notes whose ``ttl`` has passed.

        Returns the count of notes archived.
        """
        now = _utcnow()
        count = 0
        for note in list(self._notes.values()):
            if (
                note.ttl is not None
                and note.lifecycle == Lifecycle.ACTIVE
                and note.ttl < now
            ):
                self.archive(note.id)
                count += 1
        return count
