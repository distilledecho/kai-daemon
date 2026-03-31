"""Tests for the episodic_flush workflow (§3D)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

from kai_daemon.state.episodic import HandoffNote, SessionRecord, ThreadEpisode
from kai_daemon.state.working_memory import TurnNote, WorkingMemory
from kai_daemon.workflows.episodic_flush import (
    EpisodicFlushResult,
    _compile_session_record,
    _compile_thread_episodes,
    _parse_episode_response,
    _synthesize_handoff_note,
    episodic_flush,
)
from kai_daemon.workflows.preemption import PreemptionContext, PreemptionMode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENDED_AT = datetime(2026, 3, 28, 10, 0, 0, tzinfo=UTC)
_STARTED_AT = "2026-03-28T09:00:00+00:00"
_SESSION_ID = "sess-abc"
_THREAD_TIMEOUT = 2.0


# ---------------------------------------------------------------------------
# Helpers / builders
# ---------------------------------------------------------------------------


def make_turn_note(
    turn_number: int = 1,
    thread_ids: list[str] | None = None,
    register: str = "exploratory",
    register_corrected: bool = False,
    topics: list[str] | None = None,
    notable: bool = False,
    note: str | None = None,
    stance_movements: list[str] | None = None,
) -> TurnNote:
    return TurnNote(
        turn_id=f"{_SESSION_ID}:{turn_number}",
        session_id=_SESSION_ID,
        turn_number=turn_number,
        timestamp="2026-03-28T09:05:00+00:00",
        thread_ids_active=thread_ids or [],
        register=register,
        register_corrected=register_corrected,
        topics_touched=topics or ["topic-a"],
        stance_movements=stance_movements or [],
        artifacts_referenced=[],
        notable=notable,
        note=note,
    )


def make_working_memory(
    turn_notes: list[TurnNote] | None = None,
    artifacts: list[str] | None = None,
    shared_additions: list[str] | None = None,
    contradictions: list[str] | None = None,
    inquiries: list[str] | None = None,
) -> WorkingMemory:
    return WorkingMemory(
        session_id=_SESSION_ID,
        started_at=_STARTED_AT,
        turn_notes=turn_notes or [],
        artifacts_this_session=artifacts or [],
        shared_layer_additions=shared_additions or [],
        contradictions_surfaced=contradictions or [],
        commissioned_inquiries=inquiries or [],
    )


def _echo_inference(prompt: str) -> str:
    """Inference stub that returns labeled sections for episode prompts."""
    if "WHAT_WAS_SAID" in prompt or "Compile a thread episode" in prompt:
        return (
            "WHAT_WAS_SAID: We discussed this thread.\n"
            "WHAT_MOVED: A small step forward.\n"
            "WHAT_DIDNT_MOVE: null\n"
            "DAEMON_WAS_WATCHING: null\n"
            "STANCE_MOVEMENT: null"
        )
    return "Orientation prose for next session."


def _noop_write_episode(ep: ThreadEpisode) -> None:
    pass


def _noop_cooccurrence(
    sid: str, tids: list[str], aids: list[str], iids: list[str]
) -> None:
    pass


def _noop_write_handoff(hn: HandoffNote) -> None:
    pass


def _noop_write_record(sr: SessionRecord) -> None:
    pass


def _noop_write_index(sid: str, tids: list[str], occ: str) -> None:
    pass


def _run_flush(
    wm: WorkingMemory,
    *,
    generate_embedding_fn: None = None,
) -> EpisodicFlushResult:
    """Run episodic_flush with all-noop injectables (convenience for tests)."""
    return episodic_flush(
        wm,
        _ENDED_AT,
        inference_fn=_echo_inference,
        write_thread_episode_fn=_noop_write_episode,
        update_cooccurrence_fn=_noop_cooccurrence,
        write_handoff_note_fn=_noop_write_handoff,
        write_session_record_fn=_noop_write_record,
        write_session_thread_index_fn=_noop_write_index,
        generate_embedding_fn=generate_embedding_fn,
    )


# ---------------------------------------------------------------------------
# _parse_episode_response
# ---------------------------------------------------------------------------


class TestParseEpisodeResponse:
    def test_all_fields_parsed(self) -> None:
        raw = (
            "WHAT_WAS_SAID: Something was said.\n"
            "WHAT_MOVED: Progress made.\n"
            "WHAT_DIDNT_MOVE: Still unresolved.\n"
            "DAEMON_WAS_WATCHING: Daemon noticed this.\n"
            "STANCE_MOVEMENT: Status shifted."
        )
        ws, wm, wdm, dw, sm = _parse_episode_response(raw)
        assert ws == "Something was said."
        assert wm == "Progress made."
        assert wdm == "Still unresolved."
        assert dw == "Daemon noticed this."
        assert sm == "Status shifted."

    def test_null_values_become_none(self) -> None:
        raw = (
            "WHAT_WAS_SAID: Something.\n"
            "WHAT_MOVED: null\n"
            "WHAT_DIDNT_MOVE: null\n"
            "DAEMON_WAS_WATCHING: null\n"
            "STANCE_MOVEMENT: null"
        )
        _, wm, wdm, dw, sm = _parse_episode_response(raw)
        assert wm is None
        assert wdm is None
        assert dw is None
        assert sm is None

    def test_missing_labels_fallback(self) -> None:
        raw = "Some raw prose without labels."
        ws, wm, wdm, dw, sm = _parse_episode_response(raw)
        assert ws == raw.strip()
        assert wm is None
        assert wdm is None
        assert dw is None
        assert sm is None

    def test_null_case_insensitive(self) -> None:
        raw = "WHAT_WAS_SAID: X.\nWHAT_MOVED: NULL\n"
        _, wm, *_ = _parse_episode_response(raw)
        assert wm is None


# ---------------------------------------------------------------------------
# _compile_thread_episodes
# ---------------------------------------------------------------------------


class TestCompileThreadEpisodes:
    def test_no_notable_turns_returns_empty(self) -> None:
        turns = [make_turn_note(thread_ids=["t1"], notable=False)]
        episodes = _compile_thread_episodes(
            session_id=_SESSION_ID,
            turn_notes=turns,
            occurred_at=_ENDED_AT.isoformat(),
            inference_fn=_echo_inference,
            embed=None,
        )
        assert episodes == []

    def test_one_notable_turn_one_episode(self) -> None:
        turns = [make_turn_note(thread_ids=["t1"], notable=True, note="Notable.")]
        episodes = _compile_thread_episodes(
            session_id=_SESSION_ID,
            turn_notes=turns,
            occurred_at=_ENDED_AT.isoformat(),
            inference_fn=_echo_inference,
            embed=None,
        )
        assert len(episodes) == 1
        assert episodes[0].thread_id == "t1"
        assert episodes[0].session_id == _SESSION_ID

    def test_multiple_threads_produce_separate_episodes(self) -> None:
        turns = [
            make_turn_note(turn_number=1, thread_ids=["t1"], notable=True),
            make_turn_note(turn_number=2, thread_ids=["t2"], notable=True),
        ]
        episodes = _compile_thread_episodes(
            session_id=_SESSION_ID,
            turn_notes=turns,
            occurred_at=_ENDED_AT.isoformat(),
            inference_fn=_echo_inference,
            embed=None,
        )
        assert len(episodes) == 2
        thread_ids = {ep.thread_id for ep in episodes}
        assert thread_ids == {"t1", "t2"}

    def test_multi_thread_turn_contributes_to_all(self) -> None:
        turns = [make_turn_note(thread_ids=["t1", "t2"], notable=True)]
        episodes = _compile_thread_episodes(
            session_id=_SESSION_ID,
            turn_notes=turns,
            occurred_at=_ENDED_AT.isoformat(),
            inference_fn=_echo_inference,
            embed=None,
        )
        assert len(episodes) == 2

    def test_embedding_null_when_no_embed_fn(self) -> None:
        turns = [make_turn_note(thread_ids=["t1"], notable=True)]
        episodes = _compile_thread_episodes(
            session_id=_SESSION_ID,
            turn_notes=turns,
            occurred_at=_ENDED_AT.isoformat(),
            inference_fn=_echo_inference,
            embed=None,
        )
        assert episodes[0].embedding_id is None

    def test_embedding_set_when_embed_fn_provided(self) -> None:
        turns = [make_turn_note(thread_ids=["t1"], notable=True)]
        episodes = _compile_thread_episodes(
            session_id=_SESSION_ID,
            turn_notes=turns,
            occurred_at=_ENDED_AT.isoformat(),
            inference_fn=_echo_inference,
            embed=lambda _: "emb-id-001",
        )
        assert episodes[0].embedding_id == "emb-id-001"

    def test_inference_called_once_per_thread(self) -> None:
        calls: list[str] = []

        def counting_inference(prompt: str) -> str:
            calls.append(prompt)
            return _echo_inference(prompt)

        turns = [
            make_turn_note(turn_number=1, thread_ids=["t1"], notable=True),
            make_turn_note(turn_number=2, thread_ids=["t1"], notable=True),
            make_turn_note(turn_number=3, thread_ids=["t2"], notable=True),
        ]
        _compile_thread_episodes(
            session_id=_SESSION_ID,
            turn_notes=turns,
            occurred_at=_ENDED_AT.isoformat(),
            inference_fn=counting_inference,
            embed=None,
        )
        # Two threads → two inference calls
        assert len(calls) == 2

    def test_episode_has_correct_occurred_at(self) -> None:
        turns = [make_turn_note(thread_ids=["t1"], notable=True)]
        occurred = _ENDED_AT.isoformat()
        episodes = _compile_thread_episodes(
            session_id=_SESSION_ID,
            turn_notes=turns,
            occurred_at=occurred,
            inference_fn=_echo_inference,
            embed=None,
        )
        assert episodes[0].occurred_at == occurred

    def test_episode_ids_are_unique(self) -> None:
        turns = [
            make_turn_note(turn_number=i, thread_ids=[f"t{i}"], notable=True)
            for i in range(5)
        ]
        episodes = _compile_thread_episodes(
            session_id=_SESSION_ID,
            turn_notes=turns,
            occurred_at=_ENDED_AT.isoformat(),
            inference_fn=_echo_inference,
            embed=None,
        )
        ids = [ep.id for ep in episodes]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# _synthesize_handoff_note
# ---------------------------------------------------------------------------


class TestSynthesizeHandoffNote:
    def test_uses_verbatim_prompt(self) -> None:
        """The §3D verbatim prompt text must appear in every handoff note call."""
        prompts_seen: list[str] = []

        def capturing_inference(prompt: str) -> str:
            prompts_seen.append(prompt)
            return "Orientation."

        _synthesize_handoff_note(
            session_id=_SESSION_ID,
            thread_episodes=[],
            notable_turns=[],
            written_at=_ENDED_AT.isoformat(),
            inference_fn=capturing_inference,
            embed=None,
        )
        assert prompts_seen, "inference_fn was not called"
        prompt = prompts_seen[0]
        # Verbatim §3D text
        assert "You are leaving a note for a future version of yourself" in prompt
        assert "Write an orientation, not a summary." in prompt
        assert "what threads are live and unresolved" in prompt

    def test_returns_handoff_note(self) -> None:
        note = _synthesize_handoff_note(
            session_id=_SESSION_ID,
            thread_episodes=[],
            notable_turns=[],
            written_at=_ENDED_AT.isoformat(),
            inference_fn=lambda _: "Some prose.",
            embed=None,
        )
        assert isinstance(note, HandoffNote)
        assert note.session_id == _SESSION_ID
        assert note.where_we_are == "Some prose."
        assert note.embedding_id is None

    def test_thread_ids_from_episodes(self) -> None:
        ep = ThreadEpisode(
            id="ep-1",
            thread_id="t1",
            session_id=_SESSION_ID,
            occurred_at=_ENDED_AT.isoformat(),
            status_at_start="active",
            status_at_end="active",
            stance_movement=None,
            what_was_said="Discussion.",
            what_moved=None,
            what_didnt_move=None,
            daemon_was_watching=None,
            embedding_id=None,
        )
        note = _synthesize_handoff_note(
            session_id=_SESSION_ID,
            thread_episodes=[ep],
            notable_turns=[],
            written_at=_ENDED_AT.isoformat(),
            inference_fn=lambda _: "prose",
            embed=None,
        )
        assert note.thread_ids == ["t1"]

    def test_episode_context_included_in_prompt(self) -> None:
        prompts_seen: list[str] = []
        ep = ThreadEpisode(
            id="ep-1",
            thread_id="t-xyz",
            session_id=_SESSION_ID,
            occurred_at=_ENDED_AT.isoformat(),
            status_at_start="active",
            status_at_end="active",
            stance_movement=None,
            what_was_said="We talked about philosophy.",
            what_moved=None,
            what_didnt_move=None,
            daemon_was_watching=None,
            embedding_id=None,
        )

        def capturing(prompt: str) -> str:
            prompts_seen.append(prompt)
            return "result"

        _synthesize_handoff_note(
            session_id=_SESSION_ID,
            thread_episodes=[ep],
            notable_turns=[],
            written_at=_ENDED_AT.isoformat(),
            inference_fn=capturing,
            embed=None,
        )
        assert "t-xyz" in prompts_seen[0]
        assert "We talked about philosophy." in prompts_seen[0]

    def test_embedding_set_when_provided(self) -> None:
        note = _synthesize_handoff_note(
            session_id=_SESSION_ID,
            thread_episodes=[],
            notable_turns=[],
            written_at=_ENDED_AT.isoformat(),
            inference_fn=lambda _: "prose",
            embed=lambda _: "emb-hn-001",
        )
        assert note.embedding_id == "emb-hn-001"


# ---------------------------------------------------------------------------
# _compile_session_record
# ---------------------------------------------------------------------------


class TestCompileSessionRecord:
    def test_duration_calculated_correctly(self) -> None:
        wm = make_working_memory()
        # started_at is 1 hour before ended_at
        record = _compile_session_record(
            working_memory=wm,
            ended_at=_ENDED_AT,
            handoff_note_id="hn-1",
            thread_episodes=[],
            embed=None,
        )
        assert record.duration_seconds == 3600

    def test_dominant_register_most_common(self) -> None:
        turns = [
            make_turn_note(turn_number=1, register="exploratory"),
            make_turn_note(turn_number=2, register="exploratory"),
            make_turn_note(turn_number=3, register="casual"),
        ]
        wm = make_working_memory(turn_notes=turns)
        record = _compile_session_record(
            working_memory=wm,
            ended_at=_ENDED_AT,
            handoff_note_id="hn-1",
            thread_episodes=[],
            embed=None,
        )
        assert record.dominant_register == "exploratory"

    def test_register_shifts_counted(self) -> None:
        turns = [
            make_turn_note(turn_number=1, register="exploratory"),
            make_turn_note(turn_number=2, register="casual"),
            make_turn_note(turn_number=3, register="casual"),
            make_turn_note(turn_number=4, register="reflective"),
        ]
        wm = make_working_memory(turn_notes=turns)
        record = _compile_session_record(
            working_memory=wm,
            ended_at=_ENDED_AT,
            handoff_note_id="hn-1",
            thread_episodes=[],
            embed=None,
        )
        assert record.register_shifts == 2  # exploratory→casual, casual→reflective

    def test_corrections_made_counted(self) -> None:
        turns = [
            make_turn_note(turn_number=1, register_corrected=True),
            make_turn_note(turn_number=2, register_corrected=False),
            make_turn_note(turn_number=3, register_corrected=True),
        ]
        wm = make_working_memory(turn_notes=turns)
        record = _compile_session_record(
            working_memory=wm,
            ended_at=_ENDED_AT,
            handoff_note_id="hn-1",
            thread_episodes=[],
            embed=None,
        )
        assert record.corrections_made == 2

    def test_thread_ids_deduplicated(self) -> None:
        turns = [
            make_turn_note(turn_number=1, thread_ids=["t1", "t2"]),
            make_turn_note(turn_number=2, thread_ids=["t2", "t3"]),
        ]
        wm = make_working_memory(turn_notes=turns)
        record = _compile_session_record(
            working_memory=wm,
            ended_at=_ENDED_AT,
            handoff_note_id="hn-1",
            thread_episodes=[],
            embed=None,
        )
        assert set(record.thread_ids) == {"t1", "t2", "t3"}

    def test_topics_deduplicated(self) -> None:
        turns = [
            make_turn_note(turn_number=1, topics=["philosophy", "ethics"]),
            make_turn_note(turn_number=2, topics=["ethics", "science"]),
        ]
        wm = make_working_memory(turn_notes=turns)
        record = _compile_session_record(
            working_memory=wm,
            ended_at=_ENDED_AT,
            handoff_note_id="hn-1",
            thread_episodes=[],
            embed=None,
        )
        assert record.topics == ["philosophy", "ethics", "science"]

    def test_handoff_note_id_stored(self) -> None:
        wm = make_working_memory()
        record = _compile_session_record(
            working_memory=wm,
            ended_at=_ENDED_AT,
            handoff_note_id="hn-xyz",
            thread_episodes=[],
            embed=None,
        )
        assert record.handoff_note_id == "hn-xyz"

    def test_working_memory_lists_copied(self) -> None:
        """Session record lists must be independent copies of working memory lists."""
        arts = ["art-1"]
        wm = make_working_memory(artifacts=arts, contradictions=["c-1"])
        record = _compile_session_record(
            working_memory=wm,
            ended_at=_ENDED_AT,
            handoff_note_id="hn-1",
            thread_episodes=[],
            embed=None,
        )
        arts.append("art-2")  # mutate original
        assert "art-2" not in record.artifacts_ingested

    def test_register_arc_entry_per_turn(self) -> None:
        turns = [
            make_turn_note(turn_number=1, register="casual"),
            make_turn_note(turn_number=2, register="reflective"),
        ]
        wm = make_working_memory(turn_notes=turns)
        record = _compile_session_record(
            working_memory=wm,
            ended_at=_ENDED_AT,
            handoff_note_id="hn-1",
            thread_episodes=[],
            embed=None,
        )
        assert len(record.register_arc) == 2
        assert record.register_arc[0].register == "casual"
        assert record.register_arc[1].register == "reflective"

    def test_empty_turns_dominant_register(self) -> None:
        wm = make_working_memory()
        record = _compile_session_record(
            working_memory=wm,
            ended_at=_ENDED_AT,
            handoff_note_id="hn-1",
            thread_episodes=[],
            embed=None,
        )
        assert record.dominant_register == ""

    def test_embedding_id_null_when_no_embed(self) -> None:
        wm = make_working_memory()
        record = _compile_session_record(
            working_memory=wm,
            ended_at=_ENDED_AT,
            handoff_note_id="hn-1",
            thread_episodes=[],
            embed=None,
        )
        assert record.embedding_id is None


# ---------------------------------------------------------------------------
# episodic_flush — step execution order
# ---------------------------------------------------------------------------


class TestStepOrder:
    def _make_order_flush(
        self, order: list[str], wm: WorkingMemory
    ) -> EpisodicFlushResult:
        """Call episodic_flush with all steps recording to *order*."""

        def write_ep(ep: ThreadEpisode) -> None:
            order.append("episode")

        def cooccurrence(
            sid: str, tids: list[str], aids: list[str], iids: list[str]
        ) -> None:
            order.append("cooccurrence")

        def write_handoff(hn: HandoffNote) -> None:
            order.append("handoff")

        def write_record(sr: SessionRecord) -> None:
            order.append("record")

        def write_index(sid: str, tids: list[str], occ: str) -> None:
            order.append("index")

        return episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=write_ep,
            update_cooccurrence_fn=cooccurrence,
            write_handoff_note_fn=write_handoff,
            write_session_record_fn=write_record,
            write_session_thread_index_fn=write_index,
        )

    def test_all_six_steps_fire(self) -> None:
        """Every injectable callable must be called exactly once per flush."""
        order: list[str] = []
        wm = make_working_memory(
            turn_notes=[make_turn_note(thread_ids=["t1"], notable=True)]
        )
        self._make_order_flush(order, wm)
        assert "episode" in order
        assert "cooccurrence" in order
        assert "handoff" in order
        assert "record" in order
        assert "index" in order

    def test_write_order_episodes_before_cooccurrence(self) -> None:
        """Step 2 (write episodes) must precede step 3 (co-occurrence)."""
        order: list[str] = []
        wm = make_working_memory(
            turn_notes=[make_turn_note(thread_ids=["t1"], notable=True)]
        )
        self._make_order_flush(order, wm)
        assert order.index("episode") < order.index("cooccurrence")

    def test_cooccurrence_before_handoff(self) -> None:
        """Co-occurrence (step 3) must precede handoff note (step 4)."""
        order: list[str] = []
        wm = make_working_memory()
        self._make_order_flush(order, wm)
        assert order.index("cooccurrence") < order.index("handoff")

    def test_handoff_before_record(self) -> None:
        """Handoff note (step 4) must precede session record (step 5)."""
        order: list[str] = []
        wm = make_working_memory()
        self._make_order_flush(order, wm)
        assert order.index("handoff") < order.index("record")

    def test_record_before_index(self) -> None:
        """Session record (step 5) must precede thread index (step 6)."""
        order: list[str] = []
        wm = make_working_memory()
        self._make_order_flush(order, wm)
        assert order.index("record") < order.index("index")


# ---------------------------------------------------------------------------
# episodic_flush — result fields
# ---------------------------------------------------------------------------


class TestFlushResult:
    def test_result_is_episodic_flush_result(self) -> None:
        wm = make_working_memory()
        result = _run_flush(wm)
        assert isinstance(result, EpisodicFlushResult)

    def test_session_id_in_result(self) -> None:
        wm = make_working_memory()
        result = _run_flush(wm)
        assert result.session_id == _SESSION_ID

    def test_thread_episode_count(self) -> None:
        wm = make_working_memory(
            turn_notes=[
                make_turn_note(turn_number=1, thread_ids=["t1"], notable=True),
                make_turn_note(turn_number=2, thread_ids=["t2"], notable=True),
            ]
        )
        result = _run_flush(wm)
        assert result.thread_episode_count == 2

    def test_embeddings_available_false_by_default(self) -> None:
        wm = make_working_memory()
        result = _run_flush(wm)
        assert result.embeddings_available is False

    def test_embeddings_available_true_when_fn_provided(self) -> None:
        wm = make_working_memory()
        result = episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=lambda ep: None,
            update_cooccurrence_fn=lambda sid, tids, aids, iids: None,
            write_handoff_note_fn=lambda hn: None,
            write_session_record_fn=lambda sr: None,
            write_session_thread_index_fn=lambda sid, tids, occ: None,
            generate_embedding_fn=lambda _: "emb-001",
        )
        assert result.embeddings_available is True

    def test_result_ids_are_non_empty_strings(self) -> None:
        wm = make_working_memory()
        result = _run_flush(wm)
        assert result.session_record_id
        assert result.handoff_note_id

    def test_handoff_note_id_matches_written_note(self) -> None:
        """The handoff_note_id in the result must match what was written."""
        written: list[HandoffNote] = []
        wm = make_working_memory()
        result = episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=lambda ep: None,
            update_cooccurrence_fn=lambda sid, tids, aids, iids: None,
            write_handoff_note_fn=written.append,
            write_session_record_fn=lambda sr: None,
            write_session_thread_index_fn=lambda sid, tids, occ: None,
        )
        assert written, "write_handoff_note_fn was not called"
        assert result.handoff_note_id == written[0].id

    def test_session_record_id_matches_written_record(self) -> None:
        written: list[SessionRecord] = []
        wm = make_working_memory()
        result = episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=lambda ep: None,
            update_cooccurrence_fn=lambda sid, tids, aids, iids: None,
            write_handoff_note_fn=lambda hn: None,
            write_session_record_fn=written.append,
            write_session_thread_index_fn=lambda sid, tids, occ: None,
        )
        assert written, "write_session_record_fn was not called"
        assert result.session_record_id == written[0].id


# ---------------------------------------------------------------------------
# episodic_flush — working memory not mutated
# ---------------------------------------------------------------------------


class TestWorkingMemoryNotMutated:
    def test_turn_notes_unchanged(self) -> None:
        turns = [make_turn_note(thread_ids=["t1"], notable=True)]
        wm = make_working_memory(turn_notes=turns)
        original_count = len(wm.turn_notes)
        _run_flush(wm)
        assert len(wm.turn_notes) == original_count

    def test_artifacts_list_unchanged(self) -> None:
        wm = make_working_memory(artifacts=["art-1", "art-2"])
        _run_flush(wm)
        assert wm.artifacts_this_session == ["art-1", "art-2"]


# ---------------------------------------------------------------------------
# episodic_flush — empty working memory
# ---------------------------------------------------------------------------


class TestEmptyWorkingMemory:
    def test_no_thread_episodes_written(self) -> None:
        written: list[ThreadEpisode] = []
        wm = make_working_memory()
        episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=written.append,
            update_cooccurrence_fn=_noop_cooccurrence,
            write_handoff_note_fn=_noop_write_handoff,
            write_session_record_fn=_noop_write_record,
            write_session_thread_index_fn=_noop_write_index,
        )
        assert written == []

    def test_handoff_note_still_written(self) -> None:
        written: list[HandoffNote] = []
        wm = make_working_memory()
        episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=_noop_write_episode,
            update_cooccurrence_fn=_noop_cooccurrence,
            write_handoff_note_fn=written.append,
            write_session_record_fn=_noop_write_record,
            write_session_thread_index_fn=_noop_write_index,
        )
        assert len(written) == 1

    def test_session_record_still_written(self) -> None:
        written: list[SessionRecord] = []
        wm = make_working_memory()
        episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=_noop_write_episode,
            update_cooccurrence_fn=_noop_cooccurrence,
            write_handoff_note_fn=_noop_write_handoff,
            write_session_record_fn=written.append,
            write_session_thread_index_fn=_noop_write_index,
        )
        assert len(written) == 1
        assert written[0].thread_ids == []


# ---------------------------------------------------------------------------
# episodic_flush — co-occurrence index arguments
# ---------------------------------------------------------------------------


class TestCooccurrenceArguments:
    def test_all_thread_ids_passed(self) -> None:
        """All thread IDs from all turns (not just notable) go to co-occurrence."""
        cooccurrence_calls: list[tuple[str, list[str], list[str], list[str]]] = []
        turns = [
            make_turn_note(turn_number=1, thread_ids=["t1"], notable=False),
            make_turn_note(turn_number=2, thread_ids=["t2"], notable=True),
        ]
        wm = make_working_memory(turn_notes=turns, inquiries=["inq-1"])

        def capture_cooccurrence(
            sid: str,
            tids: list[str],
            aids: list[str],
            iids: list[str],
        ) -> None:
            cooccurrence_calls.append((sid, tids, aids, iids))

        episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=_noop_write_episode,
            update_cooccurrence_fn=capture_cooccurrence,
            write_handoff_note_fn=_noop_write_handoff,
            write_session_record_fn=_noop_write_record,
            write_session_thread_index_fn=_noop_write_index,
        )
        assert cooccurrence_calls
        _, tids, _, iids = cooccurrence_calls[0]
        assert set(tids) == {"t1", "t2"}
        assert iids == ["inq-1"]

    def test_session_id_passed_to_cooccurrence(self) -> None:
        cooccurrence_calls: list[str] = []
        wm = make_working_memory()

        def capture_sid(
            sid: str, tids: list[str], aids: list[str], iids: list[str]
        ) -> None:
            cooccurrence_calls.append(sid)

        episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=_noop_write_episode,
            update_cooccurrence_fn=capture_sid,
            write_handoff_note_fn=_noop_write_handoff,
            write_session_record_fn=_noop_write_record,
            write_session_thread_index_fn=_noop_write_index,
        )
        assert cooccurrence_calls == [_SESSION_ID]


# ---------------------------------------------------------------------------
# episodic_flush — preemption (suspend mode, checkpoint after step 3)
# ---------------------------------------------------------------------------


class TestPreemptionCheckpointAfterStep3:
    """Verify that checkpoint fires after step 3 (co-occurrence) and before
    step 4 (handoff note write), and that the workflow resumes correctly."""

    def test_no_preemption_ctx_runs_all_steps(self) -> None:
        """Without a PreemptionContext, all six steps complete normally."""
        written_records: list[SessionRecord] = []
        wm = make_working_memory()
        result = episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=_noop_write_episode,
            update_cooccurrence_fn=_noop_cooccurrence,
            write_handoff_note_fn=_noop_write_handoff,
            write_session_record_fn=written_records.append,
            write_session_thread_index_fn=_noop_write_index,
            preemption_ctx=None,
        )
        assert isinstance(result, EpisodicFlushResult)
        assert len(written_records) == 1

    def test_checkpoint_fires_after_cooccurrence_before_handoff(self) -> None:
        """The checkpoint must occur strictly after step 3 and before step 4."""
        event_order: list[str] = []
        ctx = PreemptionContext(PreemptionMode.SUSPEND)

        def track_cooccurrence(
            sid: str, tids: list[str], aids: list[str], iids: list[str]
        ) -> None:
            event_order.append("cooccurrence")

        def track_handoff(_: HandoffNote) -> None:
            event_order.append("handoff")

        checkpoint_event: threading.Event = threading.Event()

        def checkpoint_fn() -> None:
            event_order.append("checkpoint")
            checkpoint_event.set()

        rollback_done: threading.Event = threading.Event()

        def rollback_fn() -> None:
            event_order.append("rollback")
            rollback_done.set()

        wm = make_working_memory()
        flush_done: list[EpisodicFlushResult] = []

        def run_flush() -> None:
            result = episodic_flush(
                wm,
                _ENDED_AT,
                inference_fn=_echo_inference,
                write_thread_episode_fn=_noop_write_episode,
                update_cooccurrence_fn=track_cooccurrence,
                write_handoff_note_fn=track_handoff,
                write_session_record_fn=_noop_write_record,
                write_session_thread_index_fn=_noop_write_index,
                preemption_ctx=ctx,
                checkpoint_fn=checkpoint_fn,
                rollback_fn=rollback_fn,
            )
            flush_done.append(result)

        ctx.preempt()
        t = threading.Thread(target=run_flush, daemon=True)
        t.start()

        # Wait for checkpoint signal
        assert ctx.wait_for_checkpoint(timeout=_THREAD_TIMEOUT), "checkpoint timed out"
        assert "checkpoint" in event_order

        # Resume the suspended workflow
        ctx.resume()
        t.join(timeout=_THREAD_TIMEOUT)
        assert not t.is_alive(), "workflow did not complete after resume"

        # Order: cooccurrence → checkpoint → rollback → handoff
        assert event_order.index("cooccurrence") < event_order.index("checkpoint")
        assert event_order.index("checkpoint") < event_order.index("handoff")
        assert event_order.index("rollback") < event_order.index("handoff")

    def test_workflow_completes_after_resume(self) -> None:
        """All six steps must complete even after a suspend/resume cycle."""
        steps_completed: list[str] = []
        ctx = PreemptionContext(PreemptionMode.SUSPEND)
        wm = make_working_memory()
        flush_done: list[EpisodicFlushResult] = []

        def write_ep(ep: ThreadEpisode) -> None:
            steps_completed.append("episode")

        def cooccurrence(
            sid: str, tids: list[str], aids: list[str], iids: list[str]
        ) -> None:
            steps_completed.append("cooccurrence")

        def write_handoff(hn: HandoffNote) -> None:
            steps_completed.append("handoff")

        def write_record(sr: SessionRecord) -> None:
            steps_completed.append("record")

        def write_index(sid: str, tids: list[str], occ: str) -> None:
            steps_completed.append("index")

        def run_flush() -> None:
            result = episodic_flush(
                wm,
                _ENDED_AT,
                inference_fn=_echo_inference,
                write_thread_episode_fn=write_ep,
                update_cooccurrence_fn=cooccurrence,
                write_handoff_note_fn=write_handoff,
                write_session_record_fn=write_record,
                write_session_thread_index_fn=write_index,
                preemption_ctx=ctx,
                checkpoint_fn=lambda: None,
                rollback_fn=lambda: None,
            )
            flush_done.append(result)

        ctx.preempt()
        t = threading.Thread(target=run_flush, daemon=True)
        t.start()

        assert ctx.wait_for_checkpoint(timeout=_THREAD_TIMEOUT)
        ctx.resume()
        t.join(timeout=_THREAD_TIMEOUT)
        assert not t.is_alive()

        # All post-checkpoint steps must have fired
        assert "handoff" in steps_completed
        assert "record" in steps_completed
        assert "index" in steps_completed
        assert len(flush_done) == 1

    def test_no_preemption_if_not_preempted(self) -> None:
        """Providing a PreemptionContext but never calling preempt() → no pause."""
        ctx = PreemptionContext(PreemptionMode.SUSPEND)
        wm = make_working_memory()
        checkpoint_calls: list[bool] = []

        result = episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=_noop_write_episode,
            update_cooccurrence_fn=_noop_cooccurrence,
            write_handoff_note_fn=_noop_write_handoff,
            write_session_record_fn=_noop_write_record,
            write_session_thread_index_fn=_noop_write_index,
            preemption_ctx=ctx,
            checkpoint_fn=lambda: checkpoint_calls.append(True),
            rollback_fn=lambda: None,
        )
        assert isinstance(result, EpisodicFlushResult)
        assert checkpoint_calls == []  # checkpoint_fn was never called


# ---------------------------------------------------------------------------
# episodic_flush — session_thread_index arguments
# ---------------------------------------------------------------------------


class TestSessionThreadIndexArguments:
    def test_session_id_and_thread_ids_passed(self) -> None:
        index_calls: list[tuple[str, list[str], str]] = []
        turns = [
            make_turn_note(turn_number=1, thread_ids=["t1"]),
            make_turn_note(turn_number=2, thread_ids=["t2"]),
        ]
        wm = make_working_memory(turn_notes=turns)

        def capture_index(sid: str, tids: list[str], occ: str) -> None:
            index_calls.append((sid, tids, occ))

        episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=_noop_write_episode,
            update_cooccurrence_fn=_noop_cooccurrence,
            write_handoff_note_fn=_noop_write_handoff,
            write_session_record_fn=_noop_write_record,
            write_session_thread_index_fn=capture_index,
        )
        assert index_calls
        sid, tids, _ = index_calls[0]
        assert sid == _SESSION_ID
        assert set(tids) == {"t1", "t2"}

    def test_occurred_at_is_ended_at(self) -> None:
        occurred_ats: list[str] = []
        wm = make_working_memory()

        def capture_occ(sid: str, tids: list[str], occ: str) -> None:
            occurred_ats.append(occ)

        episodic_flush(
            wm,
            _ENDED_AT,
            inference_fn=_echo_inference,
            write_thread_episode_fn=_noop_write_episode,
            update_cooccurrence_fn=_noop_cooccurrence,
            write_handoff_note_fn=_noop_write_handoff,
            write_session_record_fn=_noop_write_record,
            write_session_thread_index_fn=capture_occ,
        )
        assert occurred_ats[0] == _ENDED_AT.isoformat()


# ---------------------------------------------------------------------------
# episodic_flush — exception propagation (atomicity)
# ---------------------------------------------------------------------------


class TestAtomicity:
    def test_exception_in_write_episode_propagates(self) -> None:
        def raising_write_episode(ep: ThreadEpisode) -> None:
            raise RuntimeError("episode write failed")

        wm = make_working_memory(
            turn_notes=[make_turn_note(thread_ids=["t1"], notable=True)]
        )
        with pytest.raises(RuntimeError, match="episode write failed"):
            episodic_flush(
                wm,
                _ENDED_AT,
                inference_fn=_echo_inference,
                write_thread_episode_fn=raising_write_episode,
                update_cooccurrence_fn=_noop_cooccurrence,
                write_handoff_note_fn=_noop_write_handoff,
                write_session_record_fn=_noop_write_record,
                write_session_thread_index_fn=_noop_write_index,
            )

    def test_exception_in_write_handoff_propagates(self) -> None:
        def raising_write_handoff(hn: HandoffNote) -> None:
            raise RuntimeError("handoff write failed")

        wm = make_working_memory()
        with pytest.raises(RuntimeError, match="handoff write failed"):
            episodic_flush(
                wm,
                _ENDED_AT,
                inference_fn=_echo_inference,
                write_thread_episode_fn=_noop_write_episode,
                update_cooccurrence_fn=_noop_cooccurrence,
                write_handoff_note_fn=raising_write_handoff,
                write_session_record_fn=_noop_write_record,
                write_session_thread_index_fn=_noop_write_index,
            )

    def test_exception_in_write_record_propagates(self) -> None:
        def raising_write_record(sr: SessionRecord) -> None:
            raise RuntimeError("record write failed")

        wm = make_working_memory()
        with pytest.raises(RuntimeError, match="record write failed"):
            episodic_flush(
                wm,
                _ENDED_AT,
                inference_fn=_echo_inference,
                write_thread_episode_fn=_noop_write_episode,
                update_cooccurrence_fn=_noop_cooccurrence,
                write_handoff_note_fn=_noop_write_handoff,
                write_session_record_fn=raising_write_record,
                write_session_thread_index_fn=_noop_write_index,
            )
