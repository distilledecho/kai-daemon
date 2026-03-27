"""Tests for inner_life_push_evaluation workflow (§2F)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai_daemon.state.holding import HoldingStore, Urgency
from kai_daemon.state.push_history import PushHistoryStore, PushRecord
from kai_daemon.workflows.inner_life_push_evaluation import (
    PushOutcome,
    _parse_outcome,
    inner_life_push_evaluation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 27, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def push_history(tmp_path: Path) -> PushHistoryStore:
    return PushHistoryStore(path=tmp_path / "push_history.yaml")


@pytest.fixture
def holding_store(tmp_path: Path) -> HoldingStore:
    return HoldingStore(path=tmp_path / "holding.yaml")


def _silence_fn(prompt: str) -> str:
    return "SILENCE"


def _holding_fn(prompt: str) -> str:
    return "HOLDING_ITEM: this connection is worth keeping"


def _push_fn(prompt: str) -> str:
    return "PUSH: structural insight spanning multiple threads"


# ---------------------------------------------------------------------------
# _parse_outcome helper
# ---------------------------------------------------------------------------


def test_parse_outcome_silence() -> None:
    outcome, desc = _parse_outcome("SILENCE")
    assert outcome == PushOutcome.SILENCE
    assert desc == ""


def test_parse_outcome_holding_item_with_description() -> None:
    outcome, desc = _parse_outcome("HOLDING_ITEM: keep this insight")
    assert outcome == PushOutcome.HOLDING_ITEM
    assert desc == "keep this insight"


def test_parse_outcome_push_with_description() -> None:
    outcome, desc = _parse_outcome("PUSH: surface to user now")
    assert outcome == PushOutcome.PUSH
    assert desc == "surface to user now"


def test_parse_outcome_unknown_defaults_to_silence() -> None:
    outcome, _ = _parse_outcome("WHAT")
    assert outcome == PushOutcome.SILENCE


def test_parse_outcome_no_crash_on_lowercase() -> None:
    outcome, _ = _parse_outcome("silence")
    assert outcome in PushOutcome


# ---------------------------------------------------------------------------
# 7-day ceiling enforcement
# ---------------------------------------------------------------------------


def test_ceiling_enforced_before_inference(
    push_history: PushHistoryStore, holding_store: HoldingStore
) -> None:
    now = _NOW
    recent = now - timedelta(days=3)
    push_history._records = [
        PushRecord(timestamp=recent.isoformat(), content_summary="recent push")
    ]
    push_history._save()

    calls: list[str] = []

    def inference_fn(prompt: str) -> str:
        calls.append(prompt)
        return "PUSH: something"

    result = inner_life_push_evaluation(
        "a thought",
        None,
        [],
        push_history=push_history,
        holding_store=holding_store,
        inference_fn=inference_fn,
        now=now,
    )
    assert result.skipped_ceiling is True
    assert result.outcome == PushOutcome.SILENCE
    assert calls == []


def test_ceiling_cleared_runs_inference(
    push_history: PushHistoryStore, holding_store: HoldingStore
) -> None:
    old = _NOW - timedelta(days=10)
    push_history._records = [
        PushRecord(timestamp=old.isoformat(), content_summary="old push")
    ]
    push_history._save()

    calls: list[str] = []

    def inference_fn(prompt: str) -> str:
        calls.append(prompt)
        return "SILENCE"

    result = inner_life_push_evaluation(
        "a thought",
        "recursion",
        [],
        push_history=push_history,
        holding_store=holding_store,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert result.skipped_ceiling is False
    assert len(calls) == 1


def test_no_push_history_ceiling_not_triggered(
    push_history: PushHistoryStore, holding_store: HoldingStore
) -> None:
    result = inner_life_push_evaluation(
        "a thought",
        None,
        [],
        push_history=push_history,
        holding_store=holding_store,
        inference_fn=_silence_fn,
        now=_NOW,
    )
    assert result.skipped_ceiling is False


# ---------------------------------------------------------------------------
# SILENCE outcome
# ---------------------------------------------------------------------------


def test_silence_writes_nothing(
    push_history: PushHistoryStore, holding_store: HoldingStore
) -> None:
    result = inner_life_push_evaluation(
        "a thought",
        None,
        [],
        push_history=push_history,
        holding_store=holding_store,
        inference_fn=_silence_fn,
        now=_NOW,
    )
    assert result.outcome == PushOutcome.SILENCE
    assert holding_store.list_all() == []
    assert push_history.all_records() == []


# ---------------------------------------------------------------------------
# HOLDING_ITEM outcome
# ---------------------------------------------------------------------------


def test_holding_item_written_to_store(
    push_history: PushHistoryStore, holding_store: HoldingStore
) -> None:
    result = inner_life_push_evaluation(
        "a thought",
        "recursion",
        ["thread-1"],
        push_history=push_history,
        holding_store=holding_store,
        inference_fn=_holding_fn,
        now=_NOW,
    )
    assert result.outcome == PushOutcome.HOLDING_ITEM
    items = holding_store.list_all()
    assert len(items) == 1
    assert items[0].urgency == Urgency.MEDIUM
    assert items[0].epistemic_origin.value == "inner_life_pipeline"
    assert "thread-1" in items[0].thread_ids


def test_holding_item_does_not_record_push(
    push_history: PushHistoryStore, holding_store: HoldingStore
) -> None:
    inner_life_push_evaluation(
        "a thought",
        None,
        [],
        push_history=push_history,
        holding_store=holding_store,
        inference_fn=_holding_fn,
        now=_NOW,
    )
    assert push_history.all_records() == []


# ---------------------------------------------------------------------------
# PUSH outcome
# ---------------------------------------------------------------------------


def test_push_writes_high_urgency_holding_item(
    push_history: PushHistoryStore, holding_store: HoldingStore
) -> None:
    result = inner_life_push_evaluation(
        "a structural insight",
        "recursion",
        ["thread-1", "thread-2"],
        push_history=push_history,
        holding_store=holding_store,
        inference_fn=_push_fn,
        now=_NOW,
    )
    assert result.outcome == PushOutcome.PUSH
    items = holding_store.list_all()
    assert len(items) == 1
    assert items[0].urgency == Urgency.HIGH


def test_push_records_in_push_history(
    push_history: PushHistoryStore, holding_store: HoldingStore
) -> None:
    inner_life_push_evaluation(
        "a structural insight",
        "recursion",
        [],
        push_history=push_history,
        holding_store=holding_store,
        inference_fn=_push_fn,
        now=_NOW,
    )
    records = push_history.all_records()
    assert len(records) == 1
    assert "structural insight" in records[0].content_summary


# ---------------------------------------------------------------------------
# Prompt includes context
# ---------------------------------------------------------------------------


def test_prompt_includes_thought_fascination_thread_count(
    push_history: PushHistoryStore, holding_store: HoldingStore
) -> None:
    captured: list[str] = []

    def capture_fn(prompt: str) -> str:
        captured.append(prompt)
        return "SILENCE"

    inner_life_push_evaluation(
        "the original thought",
        "entropy",
        ["t1", "t2", "t3"],
        push_history=push_history,
        holding_store=holding_store,
        inference_fn=capture_fn,
        now=_NOW,
    )
    prompt = captured[0]
    assert "the original thought" in prompt
    assert "entropy" in prompt
    assert "3" in prompt
