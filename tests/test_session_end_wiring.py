"""Tests for wired session end: make_relational_update_fn + make_episodic_flush_fn."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from kai_daemon.state.daemon_relational import DaemonRelationalStore
from kai_daemon.state.episodic import HandoffNote, SessionRecord, ThreadEpisode
from kai_daemon.state.working_memory import WorkingMemory
from kai_daemon.workflows.session_end import (
    make_episodic_flush_fn,
    make_relational_update_fn,
    run_session_end,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENDED_AT = datetime(2026, 4, 20, 10, 0, 0, tzinfo=UTC)
_SESSION_ID = "sess-wiring-01"
_STARTED_AT = "2026-04-20T09:00:00+00:00"

# ---------------------------------------------------------------------------
# Inference stubs
# ---------------------------------------------------------------------------


def _relational_inference(prompt: str) -> str:
    return (
        "HOW_USER_THINKS: Thinks in analogies.\n"
        "WHAT_USER_IS_WORKING_ON: Building kai.\n"
        "USERS_CURRENT_REGISTER: Exploratory.\n"
        "WHERE_DAEMON_READS_USER_WRONG: unchanged"
    )


def _episode_inference(prompt: str) -> str:
    # Satisfy both the episode prompt and the handoff note prompt
    if "WHAT_WAS_SAID" in prompt or "Thread ID" in prompt:
        return (
            "WHAT_WAS_SAID: We discussed the system.\n"
            "WHAT_MOVED: null\n"
            "WHAT_DIDNT_MOVE: null\n"
            "DAEMON_WAS_WATCHING: null\n"
            "STANCE_MOVEMENT: null"
        )
    # Handoff note prompt
    return "Orientation text for next session."


def _dual_inference(prompt: str) -> str:
    if "HOW_USER_THINKS" in prompt or "WHAT_USER_IS_WORKING_ON" in prompt:
        return _relational_inference(prompt)
    return _episode_inference(prompt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wm(session_id: str = _SESSION_ID) -> WorkingMemory:
    return WorkingMemory(session_id=session_id, started_at=_STARTED_AT)


def _make_stores(tmp_path: Path) -> DaemonRelationalStore:
    state_dir = tmp_path / "state"
    history_dir = tmp_path / "hist"
    state_dir.mkdir(parents=True)
    history_dir.mkdir(parents=True)
    return DaemonRelationalStore(state_dir=state_dir, history_dir=history_dir)


# ---------------------------------------------------------------------------
# Full session end with mocked write functions
# ---------------------------------------------------------------------------


class TestFullSessionEndWiring:
    def test_flush_succeeded_true_with_mocked_writers(self, tmp_path: Path) -> None:
        store = _make_stores(tmp_path)
        written_episodes: list[ThreadEpisode] = []
        written_notes: list[HandoffNote] = []
        written_records: list[SessionRecord] = []
        written_index: list[tuple[str, list[str], str]] = []
        written_cooccurrence: list[tuple[str, list[str], list[str], list[str]]] = []

        flush_fn = make_episodic_flush_fn(
            inference_fn=_episode_inference,
            write_thread_episode_fn=written_episodes.append,
            update_cooccurrence_fn=lambda s, t, a, i: written_cooccurrence.append(
                (s, t, a, i)
            ),
            write_handoff_note_fn=written_notes.append,
            write_session_record_fn=written_records.append,
            write_session_thread_index_fn=lambda s, t, o: written_index.append(
                (s, t, o)
            ),
        )

        relational_fn = make_relational_update_fn(_relational_inference, store=store)

        wm = _make_wm()
        result = run_session_end(
            wm,
            _ENDED_AT,
            relational_update_fn=relational_fn,
            episodic_flush_fn=flush_fn,
        )

        assert result.flush_succeeded is True
        assert result.flush_result is not None
        assert result.flush_error is None

    def test_session_record_written_on_success(self, tmp_path: Path) -> None:
        store = _make_stores(tmp_path)
        written_records: list[SessionRecord] = []

        flush_fn = make_episodic_flush_fn(
            inference_fn=_episode_inference,
            write_thread_episode_fn=lambda ep: None,
            update_cooccurrence_fn=lambda s, t, a, i: None,
            write_handoff_note_fn=lambda n: None,
            write_session_record_fn=written_records.append,
            write_session_thread_index_fn=lambda s, t, o: None,
        )
        relational_fn = make_relational_update_fn(_relational_inference, store=store)

        wm = _make_wm()
        run_session_end(
            wm,
            _ENDED_AT,
            relational_update_fn=relational_fn,
            episodic_flush_fn=flush_fn,
        )

        assert len(written_records) == 1
        assert written_records[0].thread_ids == []  # no turns → no threads

    def test_cooccurrence_called_once(self, tmp_path: Path) -> None:
        store = _make_stores(tmp_path)
        cooccurrence_sessions: list[str] = []

        def _capture_cooccurrence(
            s: str, t: list[str], a: list[str], i: list[str]
        ) -> None:
            cooccurrence_sessions.append(s)

        flush_fn = make_episodic_flush_fn(
            inference_fn=_episode_inference,
            write_thread_episode_fn=lambda ep: None,
            update_cooccurrence_fn=_capture_cooccurrence,
            write_handoff_note_fn=lambda n: None,
            write_session_record_fn=lambda r: None,
            write_session_thread_index_fn=lambda s, t, o: None,
        )
        relational_fn = make_relational_update_fn(_relational_inference, store=store)

        run_session_end(
            _make_wm(),
            _ENDED_AT,
            relational_update_fn=relational_fn,
            episodic_flush_fn=flush_fn,
        )

        assert len(cooccurrence_sessions) == 1
        assert cooccurrence_sessions[0] == _SESSION_ID

    def test_flush_result_has_correct_session_id(self, tmp_path: Path) -> None:
        store = _make_stores(tmp_path)
        flush_fn = make_episodic_flush_fn(
            inference_fn=_episode_inference,
            write_thread_episode_fn=lambda ep: None,
            update_cooccurrence_fn=lambda s, t, a, i: None,
            write_handoff_note_fn=lambda n: None,
            write_session_record_fn=lambda r: None,
            write_session_thread_index_fn=lambda s, t, o: None,
        )
        relational_fn = make_relational_update_fn(_relational_inference, store=store)

        result = run_session_end(
            _make_wm(),
            _ENDED_AT,
            relational_update_fn=relational_fn,
            episodic_flush_fn=flush_fn,
        )

        assert result.session_id == _SESSION_ID
        assert result.flush_result is not None
        assert result.flush_result.session_id == _SESSION_ID


# ---------------------------------------------------------------------------
# relational_update updates daemon_relational.yaml
# ---------------------------------------------------------------------------


class TestRelationalUpdateWritesFile:
    def test_writes_daemon_relational_yaml(self, tmp_path: Path) -> None:
        store = _make_stores(tmp_path)
        state_dir = tmp_path / "state"

        relational_fn = make_relational_update_fn(_relational_inference, store=store)
        result = relational_fn(_make_wm())

        assert result.relational_version == 1
        assert (state_dir / "daemon_relational.yaml").exists()
        loaded = store.load()
        assert loaded is not None
        assert loaded.how_user_thinks == "Thinks in analogies."

    def test_second_call_increments_version(self, tmp_path: Path) -> None:
        store = _make_stores(tmp_path)
        relational_fn = make_relational_update_fn(_relational_inference, store=store)

        r1 = relational_fn(_make_wm())
        r2 = relational_fn(_make_wm())

        assert r2.relational_version == r1.relational_version + 1

    def test_relational_update_fields_preserved(self, tmp_path: Path) -> None:
        store = _make_stores(tmp_path)

        def _partial_inference(prompt: str) -> str:
            return (
                "HOW_USER_THINKS: unchanged\n"
                "WHAT_USER_IS_WORKING_ON: Building kai-daemon.\n"
                "USERS_CURRENT_REGISTER: unchanged\n"
                "WHERE_DAEMON_READS_USER_WRONG: unchanged"
            )

        relational_fn = make_relational_update_fn(_partial_inference, store=store)
        relational_fn(_make_wm())

        loaded = store.load()
        assert loaded is not None
        assert loaded.what_user_is_working_on == "Building kai-daemon."
        # unchanged fields stay as empty string (default)
        assert loaded.how_user_thinks == ""
