"""Local file-based episodic store (no daemon-memory-server required).

Writes to data/episodic/ with:
  thread_episodes/{thread_id}.jsonl   one JSON line per ThreadEpisode
  handoff_notes.jsonl                 one JSON line per HandoffNote
  session_records.jsonl               one JSON line per SessionRecord
  session_thread_index.jsonl          one JSON line per index entry
  cooccurrence.db                     SQLite — session_cooccurrence table

All JSONL appends are serialised under a per-path threading.Lock so
concurrent writers never interleave partial lines.  The SQLite cooccurrence
DB uses the same per-path lock mechanism.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..workflows.episodic_flush import (
    UpdateCooccurrenceFn,
    WriteHandoffNoteFn,
    WriteSessionRecordFn,
    WriteSessionThreadIndexFn,
    WriteThreadEpisodeFn,
)
from .episodic import HandoffNote, SessionRecord, ThreadEpisode

# ---------------------------------------------------------------------------
# Per-path lock registry
# ---------------------------------------------------------------------------

_LOCKS_GUARD: threading.Lock = threading.Lock()
_LOCKS: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


def _lock_for(path: Path) -> threading.Lock:
    """Return (creating if absent) the threading.Lock for *path*."""
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS[key]


# ---------------------------------------------------------------------------
# JSONL append helper
# ---------------------------------------------------------------------------


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line to *path* under a per-path lock.

    Creates parent directories and the file if they do not exist.
    ``default=str`` serialises datetime/UUID values that are not natively
    JSON-serialisable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    lock = _lock_for(path)
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


# ---------------------------------------------------------------------------
# Default root (lazy import to avoid side effects at module load)
# ---------------------------------------------------------------------------


def _default_root() -> Path:
    from ._paths import episodic_dir  # noqa: PLC0415

    return episodic_dir()


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def make_write_thread_episode_fn(
    episodic_root: Path | None = None,
) -> WriteThreadEpisodeFn:
    """Return a WriteThreadEpisodeFn backed by local JSONL files.

    Appends to ``{episodic_root}/thread_episodes/{thread_id}.jsonl``,
    creating the file if absent.
    """

    def _fn(episode: ThreadEpisode) -> None:
        root = episodic_root if episodic_root is not None else _default_root()
        path = root / "thread_episodes" / f"{episode.thread_id}.jsonl"
        _append_jsonl(path, dataclasses.asdict(episode))

    return _fn


def make_update_cooccurrence_fn(
    episodic_root: Path | None = None,
) -> UpdateCooccurrenceFn:
    """Return an UpdateCooccurrenceFn backed by a local SQLite database.

    Writes to ``{episodic_root}/cooccurrence.db``, creating the
    ``session_cooccurrence`` table if absent.

    Inserts one row per thread_id, one per artifact_id, and one per
    inquiry_id — all tagged with the session_id.  The co-occurrence
    relationship is recoverable by joining on session_id.
    """

    def _fn(
        session_id: str,
        thread_ids: list[str],
        artifact_ids: list[str],
        inquiry_ids: list[str],
    ) -> None:
        root = episodic_root if episodic_root is not None else _default_root()
        root.mkdir(parents=True, exist_ok=True)
        db_path = root / "cooccurrence.db"
        lock = _lock_for(db_path)
        with lock:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_cooccurrence (
                        session_id   TEXT NOT NULL,
                        thread_id    TEXT,
                        artifact_id  TEXT,
                        inquiry_id   TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_id
                    ON session_cooccurrence (session_id)
                    """
                )
                rows = (
                    [(session_id, tid, None, None) for tid in thread_ids]
                    + [(session_id, None, aid, None) for aid in artifact_ids]
                    + [(session_id, None, None, iid) for iid in inquiry_ids]
                )
                conn.executemany(
                    "INSERT INTO session_cooccurrence"
                    " (session_id, thread_id, artifact_id, inquiry_id)"
                    " VALUES (?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
            finally:
                conn.close()

    return _fn


def make_write_handoff_note_fn(
    episodic_root: Path | None = None,
) -> WriteHandoffNoteFn:
    """Return a WriteHandoffNoteFn that appends to ``handoff_notes.jsonl``."""

    def _fn(note: HandoffNote) -> None:
        root = episodic_root if episodic_root is not None else _default_root()
        path = root / "handoff_notes.jsonl"
        _append_jsonl(path, dataclasses.asdict(note))

    return _fn


def make_write_session_record_fn(
    episodic_root: Path | None = None,
) -> WriteSessionRecordFn:
    """Return a WriteSessionRecordFn that appends to ``session_records.jsonl``."""

    def _fn(record: SessionRecord) -> None:
        root = episodic_root if episodic_root is not None else _default_root()
        path = root / "session_records.jsonl"
        _append_jsonl(path, dataclasses.asdict(record))

    return _fn


def make_write_session_thread_index_fn(
    episodic_root: Path | None = None,
) -> WriteSessionThreadIndexFn:
    """Return a WriteSessionThreadIndexFn that appends to session_thread_index.jsonl."""

    def _fn(session_id: str, thread_ids: list[str], occurred_at: str) -> None:
        root = episodic_root if episodic_root is not None else _default_root()
        path = root / "session_thread_index.jsonl"
        _append_jsonl(
            path,
            {
                "session_id": session_id,
                "thread_ids": thread_ids,
                "occurred_at": occurred_at,
            },
        )

    return _fn
