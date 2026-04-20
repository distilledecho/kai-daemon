"""Tests for the session end sequence (§4I) and relational_update workflow."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from kai_daemon.state.daemon_relational import DaemonRelational, DaemonRelationalStore
from kai_daemon.state.register_inference import SessionRelationalShadow
from kai_daemon.state.working_memory import WorkingMemory
from kai_daemon.workflows.episodic_flush import EpisodicFlushResult
from kai_daemon.workflows.relational_update import (
    RelationalUpdateResult,
    _parse_response,
    relational_update,
)
from kai_daemon.workflows.session_end import (
    run_session_end,
    take_snapshot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENDED_AT = datetime(2026, 4, 20, 10, 0, 0, tzinfo=UTC)
_SESSION_ID = "sess-test-01"
_STARTED_AT = "2026-04-20T09:00:00+00:00"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_working_memory(
    session_id: str = _SESSION_ID,
    corrections: list[tuple[str, str]] | None = None,
) -> WorkingMemory:
    wm = WorkingMemory(session_id=session_id, started_at=_STARTED_AT)
    if corrections:
        wm.relational_shadow = SessionRelationalShadow(
            corrections_this_session=corrections
        )
    return wm


def _relational_inference(prompt: str) -> str:
    return (
        "HOW_USER_THINKS: User thinks in analogies.\n"
        "WHAT_USER_IS_WORKING_ON: Building kai-daemon.\n"
        "USERS_CURRENT_REGISTER: Exploratory, sometimes reflective.\n"
        "WHERE_DAEMON_READS_USER_WRONG: unchanged"
    )


def _unchanged_inference(prompt: str) -> str:
    return (
        "HOW_USER_THINKS: unchanged\n"
        "WHAT_USER_IS_WORKING_ON: unchanged\n"
        "USERS_CURRENT_REGISTER: unchanged\n"
        "WHERE_DAEMON_READS_USER_WRONG: unchanged"
    )


def _flush_result() -> EpisodicFlushResult:
    return EpisodicFlushResult(
        session_id=_SESSION_ID,
        session_record_id="rec-001",
        handoff_note_id="hn-001",
        thread_episode_count=0,
        embeddings_available=False,
    )


def _relational_result(wm: WorkingMemory) -> RelationalUpdateResult:
    return RelationalUpdateResult(
        session_id=wm.session_id,
        relational_version=1,
        fields_updated=["how_user_thinks"],
    )


# ---------------------------------------------------------------------------
# take_snapshot
# ---------------------------------------------------------------------------


class TestTakeSnapshot:
    def test_returns_different_object(self) -> None:
        wm = make_working_memory()
        snap = take_snapshot(wm)
        assert snap is not wm

    def test_snapshot_is_deep_copy(self) -> None:
        wm = make_working_memory()
        snap = take_snapshot(wm)
        # Mutation to snapshot does not affect original
        snap.turn_count = 999
        assert wm.turn_count != 999

    def test_original_mutation_does_not_affect_snapshot(self) -> None:
        wm = make_working_memory()
        snap = take_snapshot(wm)
        wm.turn_count = 42
        assert snap.turn_count != 42

    def test_snapshot_deep_copies_lists(self) -> None:
        wm = make_working_memory()
        wm.artifacts_this_session.append("art-001")
        snap = take_snapshot(wm)
        snap.artifacts_this_session.append("art-002")
        assert "art-002" not in wm.artifacts_this_session

    def test_snapshot_deep_copies_relational_shadow(self) -> None:
        wm = make_working_memory(corrections=[("casual", "reflective")])
        snap = take_snapshot(wm)
        snap.relational_shadow.corrections_this_session.append(
            ("exploratory", "urgent")
        )
        assert len(wm.relational_shadow.corrections_this_session) == 1


# ---------------------------------------------------------------------------
# _parse_response (relational_update internal)
# ---------------------------------------------------------------------------


class TestParseRelationalResponse:
    def test_all_fields_updated(self) -> None:
        current = DaemonRelational()
        response = (
            "HOW_USER_THINKS: Thinks in analogies.\n"
            "WHAT_USER_IS_WORKING_ON: Building kai.\n"
            "USERS_CURRENT_REGISTER: Exploratory.\n"
            "WHERE_DAEMON_READS_USER_WRONG: Underestimates urgency."
        )
        updated, fields = _parse_response(response, current)
        assert updated.how_user_thinks == "Thinks in analogies."
        assert updated.what_user_is_working_on == "Building kai."
        assert updated.users_current_register == "Exploratory."
        assert updated.where_daemon_reads_user_wrong == "Underestimates urgency."
        assert len(fields) == 4

    def test_unchanged_fields_not_updated(self) -> None:
        current = DaemonRelational(how_user_thinks="Original prose.")
        response = (
            "HOW_USER_THINKS: unchanged\n"
            "WHAT_USER_IS_WORKING_ON: New work.\n"
            "USERS_CURRENT_REGISTER: unchanged\n"
            "WHERE_DAEMON_READS_USER_WRONG: unchanged"
        )
        updated, fields = _parse_response(response, current)
        assert updated.how_user_thinks == "Original prose."
        assert updated.what_user_is_working_on == "New work."
        assert fields == ["what_user_is_working_on"]

    def test_all_unchanged_returns_original_model(self) -> None:
        current = DaemonRelational(how_user_thinks="Original.")
        response = (
            "HOW_USER_THINKS: unchanged\n"
            "WHAT_USER_IS_WORKING_ON: unchanged\n"
            "USERS_CURRENT_REGISTER: unchanged\n"
            "WHERE_DAEMON_READS_USER_WRONG: unchanged"
        )
        updated, fields = _parse_response(response, current)
        assert updated is current
        assert fields == []

    def test_case_insensitive_unchanged(self) -> None:
        current = DaemonRelational(how_user_thinks="Kept.")
        response = "HOW_USER_THINKS: UNCHANGED\n"
        updated, fields = _parse_response(response, current)
        assert updated.how_user_thinks == "Kept."
        assert "how_user_thinks" not in fields


# ---------------------------------------------------------------------------
# relational_update workflow
# ---------------------------------------------------------------------------


class TestRelationalUpdate:
    def test_writes_new_version(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state"
        history_dir = tmp_path / "history"
        state_dir.mkdir()
        history_dir.mkdir()
        store = DaemonRelationalStore(state_dir=state_dir, history_dir=history_dir)

        wm = make_working_memory()
        result = relational_update(wm, inference_fn=_relational_inference, store=store)

        assert result.session_id == _SESSION_ID
        assert result.relational_version == 1
        assert len(result.fields_updated) > 0

    def test_never_mutates_working_memory(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state"
        history_dir = tmp_path / "history"
        state_dir.mkdir()
        history_dir.mkdir()
        store = DaemonRelationalStore(state_dir=state_dir, history_dir=history_dir)

        wm = make_working_memory(corrections=[("casual", "reflective")])
        original_corrections = list(wm.relational_shadow.corrections_this_session)
        relational_update(wm, inference_fn=_relational_inference, store=store)

        assert wm.relational_shadow.corrections_this_session == original_corrections

    def test_unchanged_fields_preserved(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state"
        history_dir = tmp_path / "history"
        state_dir.mkdir()
        history_dir.mkdir()
        store = DaemonRelationalStore(state_dir=state_dir, history_dir=history_dir)

        # Write initial version with existing content
        existing = DaemonRelational(
            how_user_thinks="Already known.",
            where_daemon_reads_user_wrong="Existing note.",
        )
        store.write(existing)

        wm = make_working_memory()
        relational_update(wm, inference_fn=_unchanged_inference, store=store)

        loaded = store.load()
        assert loaded is not None
        assert loaded.how_user_thinks == "Already known."
        assert loaded.where_daemon_reads_user_wrong == "Existing note."

    def test_increments_version_on_second_call(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state"
        history_dir = tmp_path / "history"
        state_dir.mkdir()
        history_dir.mkdir()
        store = DaemonRelationalStore(state_dir=state_dir, history_dir=history_dir)

        wm = make_working_memory()
        r1 = relational_update(wm, inference_fn=_relational_inference, store=store)
        r2 = relational_update(wm, inference_fn=_relational_inference, store=store)

        assert r2.relational_version == r1.relational_version + 1


# ---------------------------------------------------------------------------
# run_session_end
# ---------------------------------------------------------------------------


class TestRunSessionEnd:
    def test_flush_succeeded_true_on_success(self) -> None:
        wm = make_working_memory()
        result = run_session_end(
            wm,
            _ENDED_AT,
            relational_update_fn=_relational_result,
            episodic_flush_fn=lambda snap, ended_at: _flush_result(),
        )
        assert result.flush_succeeded is True
        assert result.flush_result is not None
        assert result.flush_error is None

    def test_flush_succeeded_false_when_flush_raises(self) -> None:
        def _failing_flush(
            snap: WorkingMemory, ended_at: datetime
        ) -> EpisodicFlushResult:
            raise RuntimeError("memory server unavailable")

        wm = make_working_memory()
        result = run_session_end(
            wm,
            _ENDED_AT,
            relational_update_fn=_relational_result,
            episodic_flush_fn=_failing_flush,
        )
        assert result.flush_succeeded is False
        assert result.flush_result is None
        assert result.flush_error is not None
        assert "memory server unavailable" in result.flush_error

    def test_relational_failure_does_not_block_flush(self) -> None:
        def _failing_relational(snap: WorkingMemory) -> RelationalUpdateResult:
            raise RuntimeError("relational store error")

        wm = make_working_memory()
        result = run_session_end(
            wm,
            _ENDED_AT,
            relational_update_fn=_failing_relational,
            episodic_flush_fn=lambda snap, ended_at: _flush_result(),
        )
        assert result.flush_succeeded is True
        assert result.relational_update_result is None
        assert result.relational_update_error is not None

    def test_both_workflows_receive_snapshot_not_live_wm(self) -> None:
        """Neither workflow receives the live working memory object."""
        received_objects: list[WorkingMemory] = []

        def _capture_relational(snap: WorkingMemory) -> RelationalUpdateResult:
            received_objects.append(snap)
            return _relational_result(snap)

        def _capture_flush(
            snap: WorkingMemory, ended_at: datetime
        ) -> EpisodicFlushResult:
            received_objects.append(snap)
            return _flush_result()

        wm = make_working_memory()
        run_session_end(
            wm,
            _ENDED_AT,
            relational_update_fn=_capture_relational,
            episodic_flush_fn=_capture_flush,
        )

        assert len(received_objects) == 2
        for obj in received_objects:
            assert obj is not wm

    def test_both_workflows_receive_same_snapshot(self) -> None:
        """Both workflows must receive the same snapshot object."""
        received_objects: list[WorkingMemory] = []

        def _capture_relational(snap: WorkingMemory) -> RelationalUpdateResult:
            received_objects.append(snap)
            return _relational_result(snap)

        def _capture_flush(
            snap: WorkingMemory, ended_at: datetime
        ) -> EpisodicFlushResult:
            received_objects.append(snap)
            return _flush_result()

        wm = make_working_memory()
        run_session_end(
            wm,
            _ENDED_AT,
            relational_update_fn=_capture_relational,
            episodic_flush_fn=_capture_flush,
        )

        assert len(received_objects) == 2
        # Both received the same deepcopy (same object ID)
        assert received_objects[0] is received_objects[1]

    def test_both_workflows_run_concurrently(self) -> None:
        """Both workflows must overlap in time (concurrent execution)."""
        events: list[str] = []
        barrier = threading.Barrier(2)

        def _slow_relational(snap: WorkingMemory) -> RelationalUpdateResult:
            events.append("relational_start")
            barrier.wait(timeout=2.0)
            events.append("relational_end")
            return _relational_result(snap)

        def _slow_flush(snap: WorkingMemory, ended_at: datetime) -> EpisodicFlushResult:
            events.append("flush_start")
            barrier.wait(timeout=2.0)
            events.append("flush_end")
            return _flush_result()

        wm = make_working_memory()
        run_session_end(
            wm,
            _ENDED_AT,
            relational_update_fn=_slow_relational,
            episodic_flush_fn=_slow_flush,
        )

        # Both start events must appear before both end events
        # (barrier ensures they were both running at the same time)
        assert "relational_start" in events
        assert "flush_start" in events
        assert "relational_end" in events
        assert "flush_end" in events

    def test_session_id_in_result(self) -> None:
        wm = make_working_memory(session_id="sess-xyz")
        result = run_session_end(
            wm,
            _ENDED_AT,
            relational_update_fn=_relational_result,
            episodic_flush_fn=lambda snap, ended_at: _flush_result(),
        )
        assert result.session_id == "sess-xyz"

    def test_both_fail_flush_succeeded_false(self) -> None:
        def _fail_relational(snap: WorkingMemory) -> RelationalUpdateResult:
            raise RuntimeError("relational error")

        def _fail_flush(snap: WorkingMemory, ended_at: datetime) -> EpisodicFlushResult:
            raise RuntimeError("flush error")

        wm = make_working_memory()
        result = run_session_end(
            wm,
            _ENDED_AT,
            relational_update_fn=_fail_relational,
            episodic_flush_fn=_fail_flush,
        )
        assert result.flush_succeeded is False
        assert len(result.sequence_errors) == 2

    def test_ended_at_forwarded_to_flush(self) -> None:
        received_ended_at: list[datetime] = []

        def _capture_flush(
            snap: WorkingMemory, ended_at: datetime
        ) -> EpisodicFlushResult:
            received_ended_at.append(ended_at)
            return _flush_result()

        custom_ended_at = datetime(2026, 4, 20, 11, 30, 0, tzinfo=UTC)
        wm = make_working_memory()
        run_session_end(
            wm,
            custom_ended_at,
            relational_update_fn=_relational_result,
            episodic_flush_fn=_capture_flush,
        )
        assert received_ended_at == [custom_ended_at]
