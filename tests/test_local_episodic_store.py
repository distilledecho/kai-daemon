"""Tests for local_episodic_store — local file-based episodic write functions."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
from pathlib import Path

from kai_daemon.state.episodic import HandoffNote, SessionRecord, ThreadEpisode
from kai_daemon.state.local_episodic_store import (
    make_update_cooccurrence_fn,
    make_write_handoff_note_fn,
    make_write_session_record_fn,
    make_write_session_thread_index_fn,
    make_write_thread_episode_fn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION_ID = "sess-local-001"


def _make_episode(
    thread_id: str = "thread-A",
    session_id: str = _SESSION_ID,
    ep_id: str = "ep-001",
) -> ThreadEpisode:
    return ThreadEpisode(
        id=ep_id,
        thread_id=thread_id,
        session_id=session_id,
        occurred_at="2026-01-01T00:00:00+00:00",
        status_at_start="active",
        status_at_end="active",
        stance_movement=None,
        what_was_said="We discussed the architecture.",
        what_moved=None,
        what_didnt_move=None,
        daemon_was_watching=None,
        embedding_id=None,
    )


def _make_handoff_note(session_id: str = _SESSION_ID) -> HandoffNote:
    return HandoffNote(
        id="hn-001",
        session_id=session_id,
        written_at="2026-01-01T00:00:00+00:00",
        thread_ids=["thread-A"],
        where_we_are="Orientation prose.",
        what_matters="",
        open_threads="",
        register_notes="",
        daemon_observations="",
        embedding_id=None,
    )


def _make_session_record(session_id: str = _SESSION_ID) -> SessionRecord:
    return SessionRecord(
        id="rec-001",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T01:00:00+00:00",
        duration_seconds=3600,
        thread_ids=["thread-A"],
        topics=["architecture"],
        dominant_register="exploratory",
    )


# ---------------------------------------------------------------------------
# write_thread_episode_fn
# ---------------------------------------------------------------------------


class TestWriteThreadEpisodeFn:
    def test_writes_jsonl_to_correct_path(self, tmp_path: Path) -> None:
        write_fn = make_write_thread_episode_fn(episodic_root=tmp_path)
        episode = _make_episode(thread_id="thread-X")
        write_fn(episode)

        path = tmp_path / "thread_episodes" / "thread-X.jsonl"
        assert path.exists()
        data = json.loads(path.read_text().strip())
        assert data["id"] == "ep-001"
        assert data["thread_id"] == "thread-X"
        assert data["session_id"] == _SESSION_ID

    def test_appends_multiple_episodes_to_same_thread_file(
        self, tmp_path: Path
    ) -> None:
        write_fn = make_write_thread_episode_fn(episodic_root=tmp_path)
        for i in range(3):
            write_fn(_make_episode(ep_id=f"ep-{i}"))

        path = tmp_path / "thread_episodes" / "thread-A.jsonl"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3
        ids = [json.loads(line)["id"] for line in lines]
        assert ids == ["ep-0", "ep-1", "ep-2"]

    def test_different_threads_write_separate_files(self, tmp_path: Path) -> None:
        write_fn = make_write_thread_episode_fn(episodic_root=tmp_path)
        write_fn(_make_episode(thread_id="thread-A", ep_id="ep-a"))
        write_fn(_make_episode(thread_id="thread-B", ep_id="ep-b"))

        assert (tmp_path / "thread_episodes" / "thread-A.jsonl").exists()
        assert (tmp_path / "thread_episodes" / "thread-B.jsonl").exists()

    def test_concurrent_writes_no_corruption(self, tmp_path: Path) -> None:
        """Twenty threads writing to the same file must each produce one valid line."""
        write_fn = make_write_thread_episode_fn(episodic_root=tmp_path)
        n = 20

        def _write(i: int) -> None:
            write_fn(_make_episode(ep_id=f"ep-{i:02d}"))

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        path = tmp_path / "thread_episodes" / "thread-A.jsonl"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == n
        # Every line must be valid JSON with the correct thread_id
        for line in lines:
            data = json.loads(line)
            assert data["thread_id"] == "thread-A"

    def test_serialises_all_dataclass_fields(self, tmp_path: Path) -> None:
        write_fn = make_write_thread_episode_fn(episodic_root=tmp_path)
        episode = _make_episode()
        write_fn(episode)

        path = tmp_path / "thread_episodes" / "thread-A.jsonl"
        data = json.loads(path.read_text().strip())
        expected = dataclasses.asdict(episode)
        assert data == expected


# ---------------------------------------------------------------------------
# update_cooccurrence_fn
# ---------------------------------------------------------------------------


class TestUpdateCooccurrenceFn:
    def test_creates_db_and_table(self, tmp_path: Path) -> None:
        update_fn = make_update_cooccurrence_fn(episodic_root=tmp_path)
        update_fn("sess-001", ["thread-A"], [], [])

        db_path = tmp_path / "cooccurrence.db"
        assert db_path.exists()
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        assert ("session_cooccurrence",) in tables

    def test_inserts_thread_rows(self, tmp_path: Path) -> None:
        update_fn = make_update_cooccurrence_fn(episodic_root=tmp_path)
        update_fn("sess-001", ["thread-A", "thread-B"], [], [])

        conn = sqlite3.connect(str(tmp_path / "cooccurrence.db"))
        rows = conn.execute(
            "SELECT session_id, thread_id FROM session_cooccurrence"
            " WHERE thread_id IS NOT NULL"
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        assert ("sess-001", "thread-A") in rows
        assert ("sess-001", "thread-B") in rows

    def test_inserts_artifact_and_inquiry_rows(self, tmp_path: Path) -> None:
        update_fn = make_update_cooccurrence_fn(episodic_root=tmp_path)
        update_fn("sess-001", ["thread-A"], ["art-1"], ["inq-1"])

        conn = sqlite3.connect(str(tmp_path / "cooccurrence.db"))
        rows = conn.execute(
            "SELECT session_id, thread_id, artifact_id, inquiry_id"
            " FROM session_cooccurrence"
        ).fetchall()
        conn.close()
        # thread-A + art-1 + inq-1 = 3 rows
        assert len(rows) == 3
        artifact_rows = [r for r in rows if r[2] is not None]
        inquiry_rows = [r for r in rows if r[3] is not None]
        assert len(artifact_rows) == 1
        assert artifact_rows[0][2] == "art-1"
        assert len(inquiry_rows) == 1
        assert inquiry_rows[0][3] == "inq-1"

    def test_empty_lists_write_no_rows(self, tmp_path: Path) -> None:
        update_fn = make_update_cooccurrence_fn(episodic_root=tmp_path)
        update_fn("sess-001", [], [], [])

        conn = sqlite3.connect(str(tmp_path / "cooccurrence.db"))
        rows = conn.execute("SELECT * FROM session_cooccurrence").fetchall()
        conn.close()
        assert rows == []

    def test_multiple_sessions_accumulate(self, tmp_path: Path) -> None:
        update_fn = make_update_cooccurrence_fn(episodic_root=tmp_path)
        update_fn("sess-001", ["thread-A"], [], [])
        update_fn("sess-002", ["thread-B"], [], [])

        conn = sqlite3.connect(str(tmp_path / "cooccurrence.db"))
        count = conn.execute("SELECT COUNT(*) FROM session_cooccurrence").fetchone()[0]
        conn.close()
        assert count == 2

    def test_concurrent_writes_no_corruption(self, tmp_path: Path) -> None:
        """Two threads writing simultaneously must each commit their row."""
        update_fn = make_update_cooccurrence_fn(episodic_root=tmp_path)
        n = 10

        def _write(i: int) -> None:
            update_fn(f"sess-{i:02d}", [f"thread-{i:02d}"], [], [])

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        conn = sqlite3.connect(str(tmp_path / "cooccurrence.db"))
        count = conn.execute("SELECT COUNT(*) FROM session_cooccurrence").fetchone()[0]
        conn.close()
        assert count == n


# ---------------------------------------------------------------------------
# write_handoff_note_fn
# ---------------------------------------------------------------------------


class TestWriteHandoffNoteFn:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        write_fn = make_write_handoff_note_fn(episodic_root=tmp_path)
        note = _make_handoff_note()
        write_fn(note)

        path = tmp_path / "handoff_notes.jsonl"
        assert path.exists()
        data = json.loads(path.read_text().strip())
        assert data["id"] == "hn-001"
        assert data["session_id"] == _SESSION_ID

    def test_appends_multiple_notes(self, tmp_path: Path) -> None:
        write_fn = make_write_handoff_note_fn(episodic_root=tmp_path)
        for i in range(3):
            note = _make_handoff_note(session_id=f"sess-{i}")
            note.id = f"hn-{i}"  # type: ignore[attr-defined]
            write_fn(note)

        lines = (tmp_path / "handoff_notes.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

    def test_serialises_all_fields(self, tmp_path: Path) -> None:
        write_fn = make_write_handoff_note_fn(episodic_root=tmp_path)
        note = _make_handoff_note()
        write_fn(note)

        data = json.loads((tmp_path / "handoff_notes.jsonl").read_text().strip())
        assert data == dataclasses.asdict(note)


# ---------------------------------------------------------------------------
# write_session_record_fn
# ---------------------------------------------------------------------------


class TestWriteSessionRecordFn:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        write_fn = make_write_session_record_fn(episodic_root=tmp_path)
        record = _make_session_record()
        write_fn(record)

        path = tmp_path / "session_records.jsonl"
        assert path.exists()
        data = json.loads(path.read_text().strip())
        assert data["id"] == "rec-001"
        assert "started_at" in data
        assert "ended_at" in data

    def test_appends_multiple_records(self, tmp_path: Path) -> None:
        write_fn = make_write_session_record_fn(episodic_root=tmp_path)
        for i in range(2):
            record = _make_session_record(session_id=f"sess-{i}")
            record.id = f"rec-{i}"  # type: ignore[attr-defined]
            write_fn(record)

        lines = (tmp_path / "session_records.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_register_arc_serialised(self, tmp_path: Path) -> None:
        from kai_daemon.state.episodic import RegisterArcEntry

        write_fn = make_write_session_record_fn(episodic_root=tmp_path)
        record = _make_session_record()
        record.register_arc = [
            RegisterArcEntry(turn=1, register="casual", corrected=False)
        ]
        write_fn(record)

        data = json.loads((tmp_path / "session_records.jsonl").read_text().strip())
        assert data["register_arc"] == [
            {"turn": 1, "register": "casual", "corrected": False}
        ]


# ---------------------------------------------------------------------------
# write_session_thread_index_fn
# ---------------------------------------------------------------------------


class TestWriteSessionThreadIndexFn:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        write_fn = make_write_session_thread_index_fn(episodic_root=tmp_path)
        write_fn("sess-001", ["thread-A", "thread-B"], "2026-01-01T01:00:00+00:00")

        path = tmp_path / "session_thread_index.jsonl"
        assert path.exists()
        data = json.loads(path.read_text().strip())
        assert data["session_id"] == "sess-001"
        assert data["thread_ids"] == ["thread-A", "thread-B"]
        assert data["occurred_at"] == "2026-01-01T01:00:00+00:00"

    def test_appends_multiple_entries(self, tmp_path: Path) -> None:
        write_fn = make_write_session_thread_index_fn(episodic_root=tmp_path)
        for i in range(3):
            write_fn(f"sess-{i}", [f"thread-{i}"], "2026-01-01T00:00:00+00:00")

        lines = (
            (tmp_path / "session_thread_index.jsonl").read_text().strip().splitlines()
        )
        assert len(lines) == 3
        session_ids = [json.loads(line)["session_id"] for line in lines]
        assert session_ids == ["sess-0", "sess-1", "sess-2"]
