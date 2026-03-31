"""Episodic memory records (§2b).

Written to daemon-memory-server at session end by ``episodic_flush``.
All records are append-only — written once, never modified.

Three record types:

``ThreadEpisode``
    One per thread per session in which that thread was meaningfully touched.
    Stored per-thread; answers "what happened with this thread in that session?"

``HandoffNote``
    One per session.  Orientation for a future self who will not remember this
    conversation.  Synthesized from thread episodes and notable turn notes.

``SessionRecord``
    One per session.  Structured and queryable.  Compiled deterministically from
    working memory; includes IDs referencing the handoff note and thread episodes.

``RegisterArcEntry``
    One row per turn; embedded inside ``SessionRecord.register_arc``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RegisterArcEntry:
    """One row per turn in the session's register arc."""

    turn: int
    register: str
    corrected: bool


@dataclass
class ThreadEpisode:
    """One per thread per session in which that thread was meaningfully touched.

    Stored as append-only JSONL per thread on daemon-memory-server
    (``thread_episodes/{thread_id}.jsonl``).
    """

    id: str
    """UUID."""
    thread_id: str
    session_id: str
    occurred_at: str
    """ISO8601 — session end timestamp."""
    status_at_start: str
    """nascent | active | dormant"""
    status_at_end: str
    """nascent | active | dormant | archived"""
    stance_movement: str | None
    """Prose description of epistemic status shift, or ``None``."""
    what_was_said: str
    """What was discussed and established about this thread."""
    what_moved: str | None
    """What progressed or shifted, or ``None`` if nothing moved."""
    what_didnt_move: str | None
    """What remained unresolved, or ``None``."""
    daemon_was_watching: str | None
    """What the daemon noticed but did not say, or ``None``."""
    embedding_id: str | None
    """Null when the embedding service was unavailable at write time."""


@dataclass
class HandoffNote:
    """One per session.  Synthesized from thread episodes and notable turn notes.

    Prompt framing from §3D: *"You are leaving a note for a future version of
    yourself who will not remember this session. Write an orientation, not a
    summary."*
    """

    id: str
    """UUID."""
    session_id: str
    written_at: str
    """ISO8601."""
    thread_ids: list[str]
    where_we_are: str
    """Full orientation prose — the verbatim §3D prompt response."""
    what_matters: str
    open_threads: str
    register_notes: str
    daemon_observations: str
    embedding_id: str | None
    """Null when the embedding service was unavailable at write time."""


@dataclass
class SessionRecord:
    """One per session.  Structured and queryable.

    Stored at ``sessions/{year}/{year-month}.jsonl`` on daemon-memory-server.
    Compiled deterministically from working memory; prose fields are empty
    strings when not yet implemented (Stage 4 populates them fully).
    """

    id: str
    """UUID."""
    started_at: str
    ended_at: str
    duration_seconds: int
    thread_ids: list[str]
    topics: list[str]
    """2–5 noun phrases per session, derived from turn note topics."""
    dominant_register: str
    register_arc: list[RegisterArcEntry] = field(
        default_factory=lambda: list[RegisterArcEntry]()
    )
    register_shifts: int = 0
    corrections_made: int = 0
    new_open_loops: list[str] = field(default_factory=lambda: list[str]())
    resolved_open_loops: list[str] = field(default_factory=lambda: list[str]())
    new_threads: list[str] = field(default_factory=lambda: list[str]())
    dormant_threads_touched: list[str] = field(default_factory=lambda: list[str]())
    artifacts_ingested: list[str] = field(default_factory=lambda: list[str]())
    shared_layer_additions: list[str] = field(default_factory=lambda: list[str]())
    contradictions_surfaced: list[str] = field(default_factory=lambda: list[str]())
    commissioned_inquiries: list[str] = field(default_factory=lambda: list[str]())
    embedding_id: str | None = None
    handoff_note_id: str | None = None
