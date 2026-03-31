"""episodic_flush workflow (§3D).

Compiles and writes the session's episodic record at conversation end.

Six write steps (§2b session end write order):

    0. Prepare embeddings (injectable; ``None`` → all ``embedding_id: null``)
    1. Compile thread episodes from notable turn notes (inference)
    2. Write thread episodes to daemon-memory-server
    3. Update co-occurrence index in SQLite
    ── checkpoint (preemption point, suspend mode) ──
    4. Synthesize and write handoff note (inference, verbatim §3D prompt)
    5. Compile and write session record
    6. Write ``session_thread_index`` rows

Atomicity contract
    All six steps complete or the session produces no episodic record.
    Partial writes are worse than no write.  The function raises on any
    failure.  The caller must **not** clear working memory unless this
    function returns successfully.  If the memory server is unavailable,
    the caller queues the flush and retries on reconnection.

Working memory
    ``episodic_flush`` reads working memory but never mutates it.
    Clearing working memory is the caller's responsibility, to be done only
    after a successful return.

Priority: 4 / Preemption: **suspend** / Model: reflection
Trigger: conversation_ended
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from ..state.episodic import (
    HandoffNote,
    RegisterArcEntry,
    SessionRecord,
    ThreadEpisode,
)
from ..state.working_memory import TurnNote, WorkingMemory
from .preemption import PreemptionContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callable type aliases (all memory-server writes are injectable)
# ---------------------------------------------------------------------------

WriteThreadEpisodeFn = Callable[[ThreadEpisode], None]
"""Write one ThreadEpisode to ``thread_episodes/{thread_id}.jsonl``."""

UpdateCooccurrenceFn = Callable[[str, list[str], list[str], list[str]], None]
"""Update the co-occurrence index for a session.

Signature: ``(session_id, thread_ids, artifact_ids, inquiry_ids) → None``.
"""

WriteHandoffNoteFn = Callable[[HandoffNote], None]
"""Write a HandoffNote to daemon-memory-server."""

WriteSessionRecordFn = Callable[[SessionRecord], None]
"""Write a SessionRecord to daemon-memory-server."""

WriteSessionThreadIndexFn = Callable[[str, list[str], str], None]
"""Write ``session_thread_index`` rows.

Signature: ``(session_id, thread_ids, occurred_at) → None``.
"""

GenerateEmbeddingFn = Callable[[str], str | None]
"""Generate an embedding for *content*; return the embedding ID or ``None``.

``None`` return means the embedding service is temporarily unavailable.
Callers write ``embedding_id: null`` and queue a backfill request.
"""

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Thread episode compilation — one call per unique thread.
# Returns five labeled sections for deterministic parsing.
_EPISODE_PROMPT_TEMPLATE = """\
Compile a thread episode from the following session notes.

Thread ID: {thread_id}
Notable turns involving this thread:
{turn_notes}

Write a brief account of what happened with this thread in this session.
Reply with exactly these five labeled sections (use "null" when nothing applies):

WHAT_WAS_SAID: <what was discussed and established>
WHAT_MOVED: <what progressed or shifted>
WHAT_DIDNT_MOVE: <what remained unresolved>
DAEMON_WAS_WATCHING: <what the daemon noticed but did not say>
STANCE_MOVEMENT: <description of epistemic status shift, or "null">
"""

# Handoff note synthesis — verbatim from §3D.
_HANDOFF_NOTE_PROMPT_TEMPLATE = """\
{context}

---

You are leaving a note for a future version of yourself who will not remember
this session. Write an orientation, not a summary.

Tell them: where we are right now, what matters most for the next session to
arrive ready for, what threads are live and unresolved, anything notable about
how this conversation went, and what you noticed that didn't surface.
"""

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class EpisodicFlushResult:
    """Returned on successful completion of all six steps."""

    session_id: str
    session_record_id: str
    handoff_note_id: str
    thread_episode_count: int
    embeddings_available: bool
    """False when the embedding service was unavailable; IDs are null."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _noop() -> None:
    pass


def _embed_or_null(
    text: str,
    embed: GenerateEmbeddingFn | None,
) -> str | None:
    """Return an embedding ID, or ``None`` if ``embed`` is not provided."""
    if embed is None:
        return None
    try:
        return embed(text)
    except Exception:
        logger.warning("Embedding generation failed; writing embedding_id: null")
        return None


def _format_turns_for_prompt(turns: list[TurnNote]) -> str:
    """Render notable turn notes as a human-readable block for prompts."""
    if not turns:
        return "(no notable turns)"
    parts: list[str] = []
    for tn in turns:
        note_text = tn.note or ""
        movements = (
            f" [stance shift: {', '.join(tn.stance_movements)}]"
            if tn.stance_movements
            else ""
        )
        parts.append(f"Turn {tn.turn_number} ({tn.timestamp}){movements}: {note_text}")
    return "\n".join(parts)


def _parse_episode_response(
    response: str,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    """Parse the labeled-section response from ``_EPISODE_PROMPT_TEMPLATE``.

    Returns ``(what_was_said, what_moved, what_didnt_move,
                daemon_was_watching, stance_movement)``.

    Falls back to the raw response in ``what_was_said`` for all other fields
    ``None`` on parse failure.
    """
    fields: dict[str, str | None] = {
        "WHAT_WAS_SAID": None,
        "WHAT_MOVED": None,
        "WHAT_DIDNT_MOVE": None,
        "DAEMON_WAS_WATCHING": None,
        "STANCE_MOVEMENT": None,
    }
    pattern = re.compile(
        r"^(WHAT_WAS_SAID|WHAT_MOVED|WHAT_DIDNT_MOVE|DAEMON_WAS_WATCHING|STANCE_MOVEMENT):\s*(.*)$",
        re.MULTILINE,
    )
    for m in pattern.finditer(response):
        key, value = m.group(1), m.group(2).strip()
        fields[key] = value if value.lower() != "null" else None

    what_was_said: str = fields["WHAT_WAS_SAID"] or response.strip()
    return (
        what_was_said,
        fields["WHAT_MOVED"],
        fields["WHAT_DIDNT_MOVE"],
        fields["DAEMON_WAS_WATCHING"],
        fields["STANCE_MOVEMENT"],
    )


def _compile_thread_episodes(
    session_id: str,
    turn_notes: list[TurnNote],
    occurred_at: str,
    inference_fn: Callable[[str], str],
    embed: GenerateEmbeddingFn | None,
) -> list[ThreadEpisode]:
    """Compile one ThreadEpisode per thread that has notable turns.

    Groups notable turn notes by ``thread_ids_active``.  A turn note that
    covers multiple threads contributes to each thread's episode.
    """
    # Collect notable turns per thread
    per_thread: dict[str, list[TurnNote]] = {}
    for tn in turn_notes:
        if not tn.notable:
            continue
        for tid in tn.thread_ids_active:
            per_thread.setdefault(tid, []).append(tn)

    episodes: list[ThreadEpisode] = []
    for thread_id, notable_turns in per_thread.items():
        turns_text = _format_turns_for_prompt(notable_turns)
        prompt = _EPISODE_PROMPT_TEMPLATE.format(
            thread_id=thread_id,
            turn_notes=turns_text,
        )
        response = inference_fn(prompt)
        (
            what_was_said,
            what_moved,
            what_didnt_move,
            daemon_was_watching,
            stance_movement,
        ) = _parse_episode_response(response)

        # Derive stance from turn notes if inference didn't produce one
        if stance_movement is None:
            all_movements = [
                tid
                for tn in notable_turns
                for tid in tn.stance_movements
                if tid == thread_id
            ]
            if all_movements:
                stance_movement = (
                    f"Epistemic status shifted in {len(all_movements)} turn(s)"
                )

        episode = ThreadEpisode(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            session_id=session_id,
            occurred_at=occurred_at,
            status_at_start="active",  # Stage 4 will supply actual thread state
            status_at_end="active",
            stance_movement=stance_movement,
            what_was_said=what_was_said,
            what_moved=what_moved,
            what_didnt_move=what_didnt_move,
            daemon_was_watching=daemon_was_watching,
            embedding_id=_embed_or_null(what_was_said, embed),
        )
        episodes.append(episode)

    return episodes


def _synthesize_handoff_note(
    session_id: str,
    thread_episodes: list[ThreadEpisode],
    notable_turns: list[TurnNote],
    written_at: str,
    inference_fn: Callable[[str], str],
    embed: GenerateEmbeddingFn | None,
) -> HandoffNote:
    """Synthesize a HandoffNote using the verbatim §3D prompt.

    Context block is prepended to the verbatim prompt so the model has
    the session material to orient from.
    """
    # Build context: thread episodes first, then notable turns
    episode_lines: list[str] = []
    for ep in thread_episodes:
        episode_lines.append(
            f"Thread {ep.thread_id}:\n"
            f"  {ep.what_was_said}\n"
            + (f"  Moved: {ep.what_moved}\n" if ep.what_moved else "")
            + (f"  Unresolved: {ep.what_didnt_move}\n" if ep.what_didnt_move else "")
        )
    episodes_text = (
        "\n".join(episode_lines) if episode_lines else "(no thread episodes)"
    )

    turns_text = _format_turns_for_prompt(notable_turns)

    context = (
        f"Thread episodes from this session:\n{episodes_text}\n\n"
        f"Notable turns:\n{turns_text}"
    )

    prompt = _HANDOFF_NOTE_PROMPT_TEMPLATE.format(context=context)
    response = inference_fn(prompt)
    prose = response.strip()

    thread_ids = [ep.thread_id for ep in thread_episodes]
    embedding_id = _embed_or_null(prose, embed)

    return HandoffNote(
        id=str(uuid.uuid4()),
        session_id=session_id,
        written_at=written_at,
        thread_ids=thread_ids,
        where_we_are=prose,
        what_matters="",
        open_threads="",
        register_notes="",
        daemon_observations="",
        embedding_id=embedding_id,
    )


def _compile_session_record(
    working_memory: WorkingMemory,
    ended_at: datetime,
    handoff_note_id: str,
    thread_episodes: list[ThreadEpisode],
    embed: GenerateEmbeddingFn | None,
) -> SessionRecord:
    """Compile a SessionRecord deterministically from working memory."""
    started = datetime.fromisoformat(working_memory.started_at)
    duration_seconds = max(0, int((ended_at - started).total_seconds()))
    ended_at_str = ended_at.isoformat()

    turn_notes = working_memory.turn_notes

    # All unique thread IDs seen across any turn (not just notable)
    thread_ids = list(
        dict.fromkeys(tid for tn in turn_notes for tid in tn.thread_ids_active)
    )

    # Topics: deduplicated across all turns, preserving first-seen order
    topics = list(
        dict.fromkeys(topic for tn in turn_notes for topic in tn.topics_touched)
    )

    # Register arc: one entry per turn
    register_arc = [
        RegisterArcEntry(
            turn=tn.turn_number,
            register=tn.register,
            corrected=tn.register_corrected,
        )
        for tn in turn_notes
    ]

    # Dominant register
    if turn_notes:
        counter: Counter[str] = Counter(tn.register for tn in turn_notes)
        dominant_register = counter.most_common(1)[0][0]
    else:
        dominant_register = ""

    # Register shifts: count consecutive changes
    registers = [tn.register for tn in turn_notes]
    register_shifts = sum(
        1 for a, b in zip(registers, registers[1:], strict=False) if a != b
    )

    # Corrections
    corrections_made = sum(1 for tn in turn_notes if tn.register_corrected)

    # Summary text for embedding (brief)
    summary = (
        f"Session {working_memory.session_id}: "
        f"{len(turn_notes)} turns, {len(thread_ids)} threads, "
        f"topics: {', '.join(topics[:5])}"
    )
    embedding_id = _embed_or_null(summary, embed)

    return SessionRecord(
        id=str(uuid.uuid4()),
        started_at=working_memory.started_at,
        ended_at=ended_at_str,
        duration_seconds=duration_seconds,
        thread_ids=thread_ids,
        topics=topics,
        dominant_register=dominant_register,
        register_arc=register_arc,
        register_shifts=register_shifts,
        corrections_made=corrections_made,
        artifacts_ingested=list(working_memory.artifacts_this_session),
        shared_layer_additions=list(working_memory.shared_layer_additions),
        contradictions_surfaced=list(working_memory.contradictions_surfaced),
        commissioned_inquiries=list(working_memory.commissioned_inquiries),
        embedding_id=embedding_id,
        handoff_note_id=handoff_note_id,
    )


# ---------------------------------------------------------------------------
# Public workflow function
# ---------------------------------------------------------------------------


def episodic_flush(
    working_memory: WorkingMemory,
    ended_at: datetime,
    *,
    inference_fn: Callable[[str], str],
    write_thread_episode_fn: WriteThreadEpisodeFn,
    update_cooccurrence_fn: UpdateCooccurrenceFn,
    write_handoff_note_fn: WriteHandoffNoteFn,
    write_session_record_fn: WriteSessionRecordFn,
    write_session_thread_index_fn: WriteSessionThreadIndexFn,
    preemption_ctx: PreemptionContext | None = None,
    checkpoint_fn: Callable[[], None] | None = None,
    rollback_fn: Callable[[], None] | None = None,
    generate_embedding_fn: GenerateEmbeddingFn | None = None,
) -> EpisodicFlushResult:
    """Compile and write the session's episodic record (§3D).

    Six write steps per §2b session end write order.  Checkpoints after
    step 3 when a ``preemption_ctx`` is provided (suspend mode).

    This function reads ``working_memory`` but **never mutates it**.  The
    caller must not clear working memory unless this function returns
    successfully.

    Parameters
    ----------
    working_memory:
        Accumulated working memory for the session.
    ended_at:
        Session end timestamp.
    inference_fn:
        Callable that sends a prompt to the reflection model and returns text.
    write_thread_episode_fn:
        Writes a ``ThreadEpisode`` to daemon-memory-server.
    update_cooccurrence_fn:
        Updates the co-occurrence SQLite index.
        ``(session_id, thread_ids, artifact_ids, inquiry_ids) → None``.
    write_handoff_note_fn:
        Writes a ``HandoffNote`` to daemon-memory-server.
    write_session_record_fn:
        Writes a ``SessionRecord`` to daemon-memory-server.
    write_session_thread_index_fn:
        Writes ``session_thread_index`` rows.
        ``(session_id, thread_ids, occurred_at) → None``.
    preemption_ctx:
        Preemption context from the workflow engine.  ``None`` means no
        preemption handling (e.g. in tests).
    checkpoint_fn:
        Called on checkpoint if ``preemption_ctx`` fires.  Defaults to noop.
    rollback_fn:
        Called on resume if ``preemption_ctx`` fires.  Defaults to noop.
    generate_embedding_fn:
        Optional.  Generates an embedding and returns its ID, or ``None``
        when the embedding service is unavailable.  If not provided, all
        ``embedding_id`` fields are ``null``.

    Returns
    -------
    EpisodicFlushResult
        IDs and counts for the completed episodic record.

    Raises
    ------
    Exception
        Any exception from an injectable callable propagates out.  The
        caller treats a raised exception as "no episodic record written"
        and queues a retry.
    """
    session_id = working_memory.session_id
    ended_at_str = ended_at.isoformat()
    embed = generate_embedding_fn

    logger.info("episodic_flush: starting session=%s", session_id)

    # ------------------------------------------------------------------
    # Step 0: Embedding availability determined by presence of embed fn
    # ------------------------------------------------------------------
    embeddings_available = embed is not None
    if not embeddings_available:
        logger.info(
            "episodic_flush: embedding service unavailable; "
            "all embedding_id fields will be null"
        )

    # ------------------------------------------------------------------
    # Step 1: Compile thread episodes from notable turn notes (inference)
    # ------------------------------------------------------------------
    logger.debug("episodic_flush: step 1 — compiling thread episodes")
    thread_episodes = _compile_thread_episodes(
        session_id=session_id,
        turn_notes=working_memory.turn_notes,
        occurred_at=ended_at_str,
        inference_fn=inference_fn,
        embed=embed,
    )
    logger.debug("episodic_flush: step 1 done — %d episode(s)", len(thread_episodes))

    # ------------------------------------------------------------------
    # Step 2: Write thread episodes
    # ------------------------------------------------------------------
    logger.debug("episodic_flush: step 2 — writing thread episodes")
    for episode in thread_episodes:
        write_thread_episode_fn(episode)

    # ------------------------------------------------------------------
    # Step 3: Update co-occurrence index
    # ------------------------------------------------------------------
    logger.debug("episodic_flush: step 3 — updating co-occurrence index")
    all_thread_ids = list(
        dict.fromkeys(
            tid for tn in working_memory.turn_notes for tid in tn.thread_ids_active
        )
    )
    update_cooccurrence_fn(
        session_id,
        all_thread_ids,
        list(working_memory.artifacts_this_session),
        list(working_memory.commissioned_inquiries),
    )

    # ------------------------------------------------------------------
    # Checkpoint — preemption point (after step 3, before step 4)
    # ------------------------------------------------------------------
    if preemption_ctx is not None:
        preemption_ctx.cooperate(
            checkpoint_fn=checkpoint_fn if checkpoint_fn is not None else _noop,
            rollback_fn=rollback_fn if rollback_fn is not None else _noop,
        )

    # ------------------------------------------------------------------
    # Step 4: Synthesize and write handoff note (inference, §3D prompt)
    # ------------------------------------------------------------------
    logger.debug("episodic_flush: step 4 — synthesizing handoff note")
    notable_turns = [tn for tn in working_memory.turn_notes if tn.notable]
    handoff_note = _synthesize_handoff_note(
        session_id=session_id,
        thread_episodes=thread_episodes,
        notable_turns=notable_turns,
        written_at=ended_at_str,
        inference_fn=inference_fn,
        embed=embed,
    )
    write_handoff_note_fn(handoff_note)

    # ------------------------------------------------------------------
    # Step 5: Compile and write session record
    # ------------------------------------------------------------------
    logger.debug("episodic_flush: step 5 — writing session record")
    session_record = _compile_session_record(
        working_memory=working_memory,
        ended_at=ended_at,
        handoff_note_id=handoff_note.id,
        thread_episodes=thread_episodes,
        embed=embed,
    )
    write_session_record_fn(session_record)

    # ------------------------------------------------------------------
    # Step 6: Write session_thread_index rows
    # ------------------------------------------------------------------
    logger.debug("episodic_flush: step 6 — writing session_thread_index")
    write_session_thread_index_fn(session_id, all_thread_ids, ended_at_str)

    logger.info(
        "episodic_flush: complete session=%s record=%s handoff=%s episodes=%d",
        session_id,
        session_record.id,
        handoff_note.id,
        len(thread_episodes),
    )

    return EpisodicFlushResult(
        session_id=session_id,
        session_record_id=session_record.id,
        handoff_note_id=handoff_note.id,
        thread_episode_count=len(thread_episodes),
        embeddings_available=embeddings_available,
    )
