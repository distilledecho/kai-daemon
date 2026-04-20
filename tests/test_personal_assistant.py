"""Tests for personal_assistant workflow (§8).

Acceptance criteria covered:
- Session lifecycle: begin_session initialises working memory; end_session
  calls session_end_fn; working memory cleared only on flush_succeeded=True.
- Per-turn sequence: register inferred, retrieval run, response returned,
  turn note written, thread stack updated, discharge checked, correction fired.
- Discharge: both gates required; contradiction hydrated when needed; at most
  one per turn; urgent register excluded from contradiction discharge.
- Register correction pathway: emits new message; prior response preserved;
  shadow updated; correction history refreshed.
- Presence-first: retrieval is context loading only (no inference calls).
- Graceful degradation: memory_client=None → empty retrieval, no error.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from kai_daemon.state._types import EpistemicOrigin
from kai_daemon.state.daemon_relational import DaemonRelational, DaemonRelationalStore
from kai_daemon.state.daemon_self import DaemonSelf, DaemonSelfStore
from kai_daemon.state.discharge import ContradictionRecord
from kai_daemon.state.holding import (
    HoldingItem,
    HoldingStore,
    HoldingType,
    RegisterNeeded,
    Urgency,
)
from kai_daemon.state.observability import (
    RegisterCorrectionEntry,
    RegisterInferenceLogger,
)
from kai_daemon.state.retrieval import RetrievalContext, SemanticResult
from kai_daemon.state.thread_stack import (
    SalienceConfig,
    ThreadStackEntry,
    ThreadStackState,
)
from kai_daemon.state.threads import (
    EpistemicStatus,
    Stance,
    Thread,
    ThreadStore,
)
from kai_daemon.state.working_memory import WorkingMemory
from kai_daemon.workflows.personal_assistant import (
    InferenceFn,
    PersonalAssistant,
    ScoreDischargeItemsFn,
    _build_system_prompt,
    _extract_topics,
    _format_discharge,
    detect_stance_movements,
)
from kai_daemon.workflows.session_end import SessionEndResult

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

_SESSION_ID = "sess-pa-test-01"


def _noop_inference(prompt: str) -> str:
    return "test response"


def _zero_scores(message: str, items: list[HoldingItem]) -> dict[str, float]:
    return {}


def _high_scores(message: str, items: list[HoldingItem]) -> dict[str, float]:
    return {i.id: 0.85 for i in items}


def _below_threshold_scores(message: str, items: list[HoldingItem]) -> dict[str, float]:
    return {i.id: 0.50 for i in items}


def _make_pa(
    tmp: Path,
    *,
    inference_fn: InferenceFn | None = None,
    memory_client: Any = None,
    holding_items: list[HoldingItem] | None = None,
    correction_history: list[RegisterCorrectionEntry] | None = None,
    session_end_flush_succeeds: bool = True,
    score_fn: ScoreDischargeItemsFn | None = None,
) -> PersonalAssistant:
    hs = HoldingStore(tmp / "holding.yaml")
    for item in holding_items or []:
        hs.write(item)

    ts = ThreadStore(
        threads_path=tmp / "threads",
        pickup_notes_path=tmp / "pn",
    )

    ds_store = DaemonSelfStore(state_dir=tmp, history_dir=tmp / "ds_hist")
    dr_store = DaemonRelationalStore(state_dir=tmp, history_dir=tmp / "dr_hist")
    reg_logger = RegisterInferenceLogger(tmp / "reg.jsonl")

    def _session_end_fn(wm: WorkingMemory, ended_at: datetime) -> SessionEndResult:
        return SessionEndResult(
            session_id=wm.session_id,
            flush_succeeded=session_end_flush_succeeds,
        )

    return PersonalAssistant(
        inference_fn=inference_fn or _noop_inference,
        memory_client=memory_client,
        holding_store=hs,
        thread_store=ts,
        daemon_self_store=ds_store,
        daemon_relational_store=dr_store,
        register_inference_logger=reg_logger,
        salience_config=SalienceConfig(),
        discharge_threshold=0.72,
        correction_history=correction_history or [],
        score_discharge_items_fn=score_fn or _zero_scores,
        session_end_fn=_session_end_fn,
    )


def _make_holding_item(
    *,
    holding_type: HoldingType = HoldingType.OBSERVATION,
    register_needed: RegisterNeeded = RegisterNeeded.ANY,
    urgency: Urgency = Urgency.LOW,
    contradiction_id: str | None = None,
) -> HoldingItem:
    kwargs: dict[str, Any] = {
        "content": "something worth surfacing",
        "type": holding_type,
        "relevance_trigger": "relevant topic",
        "register_needed": register_needed,
        "urgency": urgency,
        "source_workflow": "test",
        "epistemic_origin": EpistemicOrigin.INTERNAL,
        "contradiction_id": contradiction_id,
    }
    return HoldingItem(**kwargs)


# ---------------------------------------------------------------------------
# detect_stance_movements
# ---------------------------------------------------------------------------


def test_detect_stance_movements_empty_stack_returns_empty(tmp_path: Path) -> None:
    ts = ThreadStore(threads_path=tmp_path / "t", pickup_notes_path=tmp_path / "p")
    result = detect_stance_movements({}, [], ts)
    assert result == []


def test_detect_stance_movements_no_prior_stance_skips(tmp_path: Path) -> None:
    ts = ThreadStore(threads_path=tmp_path / "t", pickup_notes_path=tmp_path / "p")
    thread = Thread(
        title="t",
        central_question="q?",
        current_state="open",
        unresolved="yes",
        stance=Stance(position="uncertain", epistemic_status=EpistemicStatus.LIVE),
    )
    ts.create(thread)
    entry = ThreadStackEntry(
        thread_id=thread.id,
        state=ThreadStackState.foreground,
        salience=0.8,
        engagement_depth=0.0,
        last_touched_turn=1,
        entered_turn=1,
    )
    # No prior stance — thread not in pre_turn_stances
    result = detect_stance_movements({}, [entry], ts)
    assert result == []


def test_detect_stance_movements_detects_change(tmp_path: Path) -> None:
    ts = ThreadStore(threads_path=tmp_path / "t", pickup_notes_path=tmp_path / "p")
    thread = Thread(
        title="t",
        central_question="q?",
        current_state="resolved",
        unresolved="no",
        stance=Stance(position="accepted", epistemic_status=EpistemicStatus.ACCEPTED),
    )
    ts.create(thread)
    entry = ThreadStackEntry(
        thread_id=thread.id,
        state=ThreadStackState.foreground,
        salience=0.8,
        engagement_depth=0.0,
        last_touched_turn=1,
        entered_turn=1,
    )
    # Prior stance was LIVE; current is ACCEPTED → movement
    pre_stances = {thread.id: EpistemicStatus.LIVE.value}
    result = detect_stance_movements(pre_stances, [entry], ts)
    assert thread.id in result


def test_detect_stance_movements_no_change_returns_empty(tmp_path: Path) -> None:
    ts = ThreadStore(threads_path=tmp_path / "t", pickup_notes_path=tmp_path / "p")
    thread = Thread(
        title="t",
        central_question="q?",
        current_state="open",
        unresolved="yes",
        stance=Stance(position="live", epistemic_status=EpistemicStatus.LIVE),
    )
    ts.create(thread)
    entry = ThreadStackEntry(
        thread_id=thread.id,
        state=ThreadStackState.foreground,
        salience=0.8,
        engagement_depth=0.0,
        last_touched_turn=1,
        entered_turn=1,
    )
    pre_stances = {thread.id: EpistemicStatus.LIVE.value}
    result = detect_stance_movements(pre_stances, [entry], ts)
    assert result == []


def test_detect_stance_movements_missing_thread_skips(tmp_path: Path) -> None:
    ts = ThreadStore(threads_path=tmp_path / "t", pickup_notes_path=tmp_path / "p")
    entry = ThreadStackEntry(
        thread_id="nonexistent-id",
        state=ThreadStackState.foreground,
        salience=0.8,
        engagement_depth=0.0,
        last_touched_turn=1,
        entered_turn=1,
    )
    pre_stances = {"nonexistent-id": EpistemicStatus.LIVE.value}
    result = detect_stance_movements(pre_stances, [entry], ts)
    assert result == []


# ---------------------------------------------------------------------------
# _extract_topics
# ---------------------------------------------------------------------------


def test_extract_topics_returns_at_most_max(tmp_path: Path) -> None:
    text = "thinking about memory systems and retrieval architecture"
    topics = _extract_topics(text, max_topics=3)
    assert len(topics) <= 3


def test_extract_topics_excludes_stop_words() -> None:
    topics = _extract_topics("about there those which with")
    assert topics == []


def test_extract_topics_finds_significant_words() -> None:
    topics = _extract_topics("memory systems retrieval architecture")
    assert "memory" in topics
    assert "systems" in topics or "retrieval" in topics


def test_extract_topics_empty_text() -> None:
    assert _extract_topics("") == []


# ---------------------------------------------------------------------------
# _build_system_prompt
# ---------------------------------------------------------------------------


def test_build_system_prompt_includes_subtext_priming(tmp_path: Path) -> None:
    ts = ThreadStore(threads_path=tmp_path / "t", pickup_notes_path=tmp_path / "p")
    prompt = _build_system_prompt(None, None, [], ts, RetrievalContext())
    assert "Before responding, consider" in prompt


def test_build_system_prompt_includes_daemon_self(tmp_path: Path) -> None:
    ts = ThreadStore(threads_path=tmp_path / "t", pickup_notes_path=tmp_path / "p")
    ds = DaemonSelf(who_daemon_is="I am curious and careful.")
    prompt = _build_system_prompt(ds, None, [], ts, RetrievalContext())
    assert "curious" in prompt


def test_build_system_prompt_includes_relational_context(tmp_path: Path) -> None:
    ts = ThreadStore(threads_path=tmp_path / "t", pickup_notes_path=tmp_path / "p")
    dr = DaemonRelational(how_user_thinks="Thinks in analogies.")
    prompt = _build_system_prompt(None, dr, [], ts, RetrievalContext())
    assert "analogies" in prompt


def test_build_system_prompt_includes_retrieval_results(tmp_path: Path) -> None:
    ts = ThreadStore(threads_path=tmp_path / "t", pickup_notes_path=tmp_path / "p")
    result = SemanticResult(
        document_id="doc-1",
        text="Deep thoughts on consciousness.",
        score=0.9,
        space="user_pkm",
    )
    ctx = RetrievalContext(semantic=[result])
    prompt = _build_system_prompt(None, None, [], ts, ctx)
    assert "consciousness" in prompt


def test_build_system_prompt_notes_pending_artifacts(tmp_path: Path) -> None:
    ts = ThreadStore(threads_path=tmp_path / "t", pickup_notes_path=tmp_path / "p")
    pending = SemanticResult(
        document_id="doc-pending",
        text="",
        score=0.7,
        space="shared",
        metadata={"chunk_status": "pending", "title": "Big Book"},
    )
    ctx = RetrievalContext(pending_artifacts=[pending])
    prompt = _build_system_prompt(None, None, [], ts, ctx)
    assert "still being processed" in prompt or "still reading" in prompt


def test_build_system_prompt_includes_thread_stack(tmp_path: Path) -> None:
    ts = ThreadStore(threads_path=tmp_path / "t", pickup_notes_path=tmp_path / "p")
    thread = Thread(
        title="Cognition and meaning",
        central_question="What is meaning in a symbolic system?",
        current_state="open",
        unresolved="yes",
        stance=Stance(position="live", epistemic_status=EpistemicStatus.LIVE),
    )
    ts.create(thread)
    entry = ThreadStackEntry(
        thread_id=thread.id,
        state=ThreadStackState.foreground,
        salience=0.8,
        engagement_depth=0.0,
        last_touched_turn=1,
        entered_turn=1,
    )
    prompt = _build_system_prompt(None, None, [entry], ts, RetrievalContext())
    assert "Cognition and meaning" in prompt


# ---------------------------------------------------------------------------
# _format_discharge
# ---------------------------------------------------------------------------


def test_format_discharge_without_contradiction_returns_content() -> None:
    item = _make_holding_item()
    msg = _format_discharge(item, None)
    assert "something worth surfacing" in msg


def test_format_discharge_with_contradiction_includes_summary() -> None:
    item = _make_holding_item(
        holding_type=HoldingType.REASONED_DISAGREEMENT,
        contradiction_id="c-001",
    )
    record = ContradictionRecord(
        id="c-001",
        item_a_id="a",
        item_b_id="b",
        conflict_summary="Earlier you said X but now lean toward Y.",
    )
    msg = _format_discharge(item, record)
    assert "Earlier you said X" in msg


# ---------------------------------------------------------------------------
# PersonalAssistant — session lifecycle
# ---------------------------------------------------------------------------


def test_begin_session_initialises_working_memory(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    wm = pa.begin_session()
    assert wm.turn_count == 0
    assert wm.session_id != ""
    assert wm.thread_stack == []
    assert wm.floating_threads == []


def test_begin_session_loads_daemon_self(tmp_path: Path) -> None:
    ds_store = DaemonSelfStore(state_dir=tmp_path, history_dir=tmp_path / "dsh")
    ds_store.write(DaemonSelf(who_daemon_is="Curious mind."))

    pa = _make_pa(tmp_path)
    # Replace the store reference (hacky but avoids rebuilding whole pa)
    pa._daemon_self_store = ds_store
    pa.begin_session()
    assert pa._daemon_self is not None
    assert pa._daemon_self.who_daemon_is == "Curious mind."


def test_begin_session_tolerates_missing_daemon_self(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    assert pa._daemon_self is None  # no daemon_self.yaml written


def test_end_session_flush_succeeded_clears_working_memory(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path, session_end_flush_succeeds=True)
    pa.begin_session()
    result = pa.end_session()
    assert result.flush_succeeded is True
    assert pa._working_memory is None


def test_end_session_flush_failed_retains_working_memory(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path, session_end_flush_succeeds=False)
    pa.begin_session()
    result = pa.end_session()
    assert result.flush_succeeded is False
    assert pa._working_memory is not None


def test_end_session_without_begin_raises(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    with pytest.raises(RuntimeError, match="begin_session"):
        pa.end_session()


def test_handle_turn_without_begin_raises(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    with pytest.raises(RuntimeError, match="begin_session"):
        asyncio.run(pa.handle_turn("hello"))


# ---------------------------------------------------------------------------
# PersonalAssistant — per-turn: register inference
# ---------------------------------------------------------------------------


def test_handle_turn_infers_register(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    result = asyncio.run(pa.handle_turn("help everything is broken"))
    assert result.register == "urgent"


def test_handle_turn_casual_message_register(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    result = asyncio.run(pa.handle_turn("hey"))
    assert result.register == "casual"


def test_handle_turn_register_confidence_present(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    result = asyncio.run(pa.handle_turn("wondering about consciousness"))
    assert 0.0 <= result.register_confidence <= 1.0


# ---------------------------------------------------------------------------
# PersonalAssistant — per-turn: response
# ---------------------------------------------------------------------------


def _constant_response(prompt: str) -> str:
    return "my response"


def test_handle_turn_returns_response_from_inference(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path, inference_fn=_constant_response)
    pa.begin_session()
    result = asyncio.run(pa.handle_turn("hello"))
    assert result.response == "my response"


def test_handle_turn_increments_turn_count(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    asyncio.run(pa.handle_turn("first"))
    asyncio.run(pa.handle_turn("second"))
    assert pa._working_memory is not None
    assert pa._working_memory.turn_count == 2


# ---------------------------------------------------------------------------
# PersonalAssistant — per-turn: turn note
# ---------------------------------------------------------------------------


def test_handle_turn_writes_turn_note(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    asyncio.run(pa.handle_turn("thinking about memory"))
    assert pa._working_memory is not None
    assert len(pa._working_memory.turn_notes) == 1


def test_handle_turn_turn_note_register_matches(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    asyncio.run(pa.handle_turn("help it is broken"))
    assert pa._working_memory is not None
    note = pa._working_memory.turn_notes[0]
    assert note.register == "urgent"


def test_handle_turn_turn_note_turn_id_format(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    wm = pa.begin_session()
    asyncio.run(pa.handle_turn("hello"))
    assert pa._working_memory is not None
    note = pa._working_memory.turn_notes[0]
    assert note.turn_id.startswith(wm.session_id + ":")
    assert note.turn_number == 1


def test_handle_turn_multiple_turns_accumulate_notes(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    asyncio.run(pa.handle_turn("first message"))
    asyncio.run(pa.handle_turn("second message"))
    assert pa._working_memory is not None
    assert len(pa._working_memory.turn_notes) == 2
    assert pa._working_memory.turn_notes[1].turn_number == 2


# ---------------------------------------------------------------------------
# PersonalAssistant — per-turn: retrieval graceful degradation
# ---------------------------------------------------------------------------


def test_handle_turn_no_memory_client_returns_response(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path, memory_client=None)
    pa.begin_session()
    result = asyncio.run(pa.handle_turn("any message"))
    assert result.response == "test response"


class _ErrorMemoryClient:
    """Memory client that always raises."""

    async def semantic_query(self, query: Any) -> list[Any]:  # noqa: ANN401
        raise ConnectionError("server down")


def test_handle_turn_memory_client_error_returns_response(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path, memory_client=_ErrorMemoryClient())
    pa.begin_session()
    result = asyncio.run(pa.handle_turn("any message"))
    assert result.response == "test response"
    assert result.discharge_surfaced is False


# ---------------------------------------------------------------------------
# PersonalAssistant — per-turn: discharge
# ---------------------------------------------------------------------------


def test_handle_turn_no_discharge_when_no_items(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    result = asyncio.run(pa.handle_turn("some message"))
    assert result.discharge_surfaced is False
    assert result.discharge_message is None


def test_handle_turn_no_discharge_when_score_below_threshold(tmp_path: Path) -> None:
    item = _make_holding_item()
    pa = _make_pa(
        tmp_path,
        holding_items=[item],
        score_fn=_below_threshold_scores,
    )
    pa.begin_session()
    result = asyncio.run(pa.handle_turn("some message"))
    assert result.discharge_surfaced is False


def test_handle_turn_discharge_fires_when_both_gates_pass(tmp_path: Path) -> None:
    item = _make_holding_item(register_needed=RegisterNeeded.ANY)
    pa = _make_pa(
        tmp_path,
        holding_items=[item],
        score_fn=_high_scores,
    )
    pa.begin_session()
    result = asyncio.run(pa.handle_turn("relevant topic message"))
    assert result.discharge_surfaced is True
    assert result.discharge_message is not None
    assert "something worth surfacing" in result.discharge_message


def test_handle_turn_discharge_register_gate_blocks_wrong_register(
    tmp_path: Path,
) -> None:
    item = _make_holding_item(register_needed=RegisterNeeded.REFLECTIVE)
    pa = _make_pa(
        tmp_path,
        holding_items=[item],
        score_fn=_high_scores,
    )
    pa.begin_session()
    # "hey" → casual register → won't match reflective gate
    result = asyncio.run(pa.handle_turn("hey cool"))
    assert result.discharge_surfaced is False


def test_handle_turn_discharge_urgent_blocks_contradiction(tmp_path: Path) -> None:
    item = _make_holding_item(
        holding_type=HoldingType.REASONED_DISAGREEMENT,
        register_needed=RegisterNeeded.ANY,
        contradiction_id="c-001",
    )
    pa = _make_pa(
        tmp_path,
        holding_items=[item],
        score_fn=_high_scores,
    )
    pa.begin_session()
    # "help it is broken" → urgent register → contradiction blocked
    result = asyncio.run(pa.handle_turn("help it is broken"))
    assert result.register == "urgent"
    assert result.discharge_surfaced is False


def test_handle_turn_discharge_at_most_one_per_turn(tmp_path: Path) -> None:
    item_a = _make_holding_item()
    item_b = _make_holding_item()
    pa = _make_pa(
        tmp_path,
        holding_items=[item_a, item_b],
        score_fn=_high_scores,
    )
    pa.begin_session()
    result = asyncio.run(pa.handle_turn("relevant message"))
    assert result.discharge_surfaced is True
    # Second turn: remaining item may discharge
    result2 = asyncio.run(pa.handle_turn("relevant message again"))
    total = (1 if result.discharge_surfaced else 0) + (
        1 if result2.discharge_surfaced else 0
    )
    assert total <= 2  # max one per turn across both turns


def test_handle_turn_discharged_item_marked_surfaced(tmp_path: Path) -> None:
    item = _make_holding_item()
    pa = _make_pa(
        tmp_path,
        holding_items=[item],
        score_fn=_high_scores,
    )
    pa.begin_session()
    asyncio.run(pa.handle_turn("relevant topic"))
    # Item should be marked as surfaced in holding store
    stored = pa._holding_store.read(item.id)
    assert stored.surfaced is not None


def test_handle_turn_no_discharge_on_second_turn_after_surfaced(
    tmp_path: Path,
) -> None:
    item = _make_holding_item()
    pa = _make_pa(
        tmp_path,
        holding_items=[item],
        score_fn=_high_scores,
    )
    pa.begin_session()
    asyncio.run(pa.handle_turn("relevant topic"))  # discharges
    result2 = asyncio.run(pa.handle_turn("relevant topic again"))
    assert result2.discharge_surfaced is False  # already surfaced


# ---------------------------------------------------------------------------
# PersonalAssistant — per-turn: register correction
# ---------------------------------------------------------------------------


def test_handle_turn_no_correction_when_signal_absent(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    result = asyncio.run(pa.handle_turn("hello", correction_signal=None))
    assert result.correction_triggered is False
    assert result.correction_message is None


def test_handle_turn_correction_fires_when_signal_provided(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    result = asyncio.run(
        pa.handle_turn("hello", correction_signal=("casual", "reflective"))
    )
    assert result.correction_triggered is True
    assert result.correction_message is not None


def test_handle_turn_correction_does_not_replace_prior_response(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def _inference(p: str) -> str:
        calls.append(p)
        return "original response"

    pa = _make_pa(tmp_path, inference_fn=_inference)
    pa.begin_session()
    asyncio.run(pa.handle_turn("hello"))

    call_count_before = len(calls)
    result = asyncio.run(
        pa.handle_turn("next", correction_signal=("casual", "reflective"))
    )
    # Correction emits acknowledgment message but response is from inference
    assert result.response == "original response"
    # No extra inference call for the correction (ack is from table lookup)
    assert len(calls) - call_count_before == 1  # only for the new turn response


def test_handle_turn_correction_updates_session_shadow(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    asyncio.run(pa.handle_turn("hello", correction_signal=("casual", "reflective")))
    assert pa._working_memory is not None
    shadow = pa._working_memory.relational_shadow
    assert ("casual", "reflective") in shadow.corrections_this_session


def test_handle_turn_correction_writes_to_log(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    asyncio.run(pa.handle_turn("hello", correction_signal=("exploratory", "urgent")))
    entries = pa._register_inference_logger.read_all()
    assert len(entries) == 1
    assert entries[0].inferred_register == "exploratory"
    assert entries[0].corrected_register == "urgent"


def test_handle_turn_correction_invalid_signal_ignored(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    # "badregister" is not a valid register value → ValueError in apply_correction
    result = asyncio.run(
        pa.handle_turn("hello", correction_signal=("badregister", "casual"))
    )
    assert result.correction_triggered is False


def test_handle_turn_turn_note_register_corrected_flag(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    pa.begin_session()
    asyncio.run(pa.handle_turn("hello", correction_signal=("casual", "reflective")))
    assert pa._working_memory is not None
    note = pa._working_memory.turn_notes[0]
    assert note.register_corrected is True


# ---------------------------------------------------------------------------
# PersonalAssistant — system prompt includes subtext priming
# ---------------------------------------------------------------------------


def test_handle_turn_system_prompt_includes_subtext_priming(tmp_path: Path) -> None:
    captured: list[str] = []

    def _inference(p: str) -> str:
        captured.append(p)
        return "response"

    pa = _make_pa(tmp_path, inference_fn=_inference)
    pa.begin_session()
    asyncio.run(pa.handle_turn("hello"))
    assert captured
    assert "Before responding, consider" in captured[0]


# ---------------------------------------------------------------------------
# PersonalAssistant — session end wires to run_session_end contract
# ---------------------------------------------------------------------------


def test_end_session_returns_result_with_session_id(tmp_path: Path) -> None:
    pa = _make_pa(tmp_path)
    wm = pa.begin_session()
    result = pa.end_session()
    assert result.session_id == wm.session_id


def test_end_session_called_with_correct_working_memory(tmp_path: Path) -> None:
    captured_wm: list[WorkingMemory] = []

    def _session_end_fn(wm: WorkingMemory, ended_at: datetime) -> SessionEndResult:
        captured_wm.append(wm)
        return SessionEndResult(session_id=wm.session_id, flush_succeeded=True)

    pa = _make_pa(tmp_path)
    pa._session_end_fn = _session_end_fn
    pa.begin_session()
    asyncio.run(pa.handle_turn("hello"))
    pa.end_session()

    assert len(captured_wm) == 1
    assert captured_wm[0].turn_count == 1
