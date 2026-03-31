"""Working memory — Layer 1 (§2a).

Session-scoped, in-process, ephemeral.  Accumulates turn notes and session
metadata during a conversation.  Compiled into episodic memory at session end
by ``episodic_flush``.

Working memory is **never** cleared until ``episodic_flush`` confirms success.
If the memory server is unavailable, the flush is queued and retried on
reconnection — working memory persists in-process until then.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnNote:
    """Written to working memory after each response is sent (§2a).

    ``note`` is non-null only when ``notable`` is ``True``.
    Most turns are not notable — the bar is meaningful, not generous.
    """

    turn_id: str
    """``{session_id}:{turn_number}``."""

    session_id: str
    turn_number: int
    timestamp: str
    """ISO8601."""

    thread_ids_active: list[str]
    """Thread IDs on the stack when this note was written."""

    register: str
    """Register classification for this turn."""

    register_corrected: bool
    """True if the register was corrected during or after this turn."""

    topics_touched: list[str]
    """2–5 noun phrases, daemon's judgment."""

    stance_movements: list[str]
    """Thread IDs whose epistemic status shifted in this turn."""

    artifacts_referenced: list[str]
    """Artifact IDs referenced in this turn."""

    notable: bool
    """True when the turn is worth preserving in episodic memory."""

    note: str | None
    """Prose note — non-null only when ``notable`` is ``True``."""


@dataclass
class WorkingMemory:
    """In-process session state (§2a).

    Never persisted directly.  Compiled into episodic memory at session end
    via ``episodic_flush``.  Cleared only after the flush confirms success.
    """

    session_id: str
    started_at: str
    """ISO8601 timestamp of session start."""

    turn_notes: list[TurnNote] = field(default_factory=lambda: list[TurnNote]())
    artifacts_this_session: list[str] = field(default_factory=lambda: list[str]())
    """Artifact IDs ingested or referenced during this session."""

    shared_layer_additions: list[str] = field(default_factory=lambda: list[str]())
    """Semantic item IDs added to the shared knowledge space this session."""

    contradictions_surfaced: list[str] = field(default_factory=lambda: list[str]())
    """Contradiction IDs surfaced during this session."""

    commissioned_inquiries: list[str] = field(default_factory=lambda: list[str]())
    """Inquiry IDs commissioned in this session."""
