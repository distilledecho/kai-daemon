"""Tests for the contradiction_detection workflow (§3F)."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from kai_daemon.state.holding import HoldingStore, HoldingType, RegisterNeeded
from kai_daemon.workflows.contradiction_detection import (
    CandidatePair,
    ContradictionDetectionResult,
    DetectionTrigger,
    SemanticItemSummary,
    _assess_conflict,
    _write_contradiction_holding_item,
    contradiction_detection,
)
from kai_daemon.workflows.preemption import PreemptionContext, PreemptionMode

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_THREAD_TIMEOUT = 2.0  # seconds


def _make_item(
    content: str = "Some finding.",
    content_type: str = "finding",
    inquiry_id: str | None = None,
) -> SemanticItemSummary:
    return SemanticItemSummary(
        id=str(uuid.uuid4()),
        content=content,
        content_type=content_type,
        inquiry_id=inquiry_id,
    )


def _make_pair(
    content_a: str = "The sky is blue.",
    content_b: str = "The sky is not blue.",
    similarity: float = 0.90,
) -> CandidatePair:
    return CandidatePair(
        item_a=_make_item(content=content_a),
        item_b=_make_item(content=content_b),
        similarity=similarity,
    )


def _inference_conflict(prompt: str) -> str:  # noqa: ARG001
    return "CONFLICT: These two statements directly contradict each other."


def _inference_no_conflict(prompt: str) -> str:  # noqa: ARG001
    return "NO_CONFLICT: These statements describe different aspects."


def _inference_ambiguous(prompt: str) -> str:  # noqa: ARG001
    return "Hmm, I'm not sure what to say here."


def _make_create_contradiction_fn() -> Callable[[str, str, str], str]:
    """Return a create_contradiction_fn that records calls and returns a UUID."""
    calls: list[tuple[str, str, str]] = []

    def fn(item_a_id: str, item_b_id: str, summary: str) -> str:
        calls.append((item_a_id, item_b_id, summary))
        return str(uuid.uuid4())

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def _no_pairs(items: list[SemanticItemSummary]) -> list[CandidatePair]:
    return []


# ---------------------------------------------------------------------------
# SemanticItemSummary
# ---------------------------------------------------------------------------


class TestSemanticItemSummary:
    def test_required_fields(self) -> None:
        item = SemanticItemSummary(id="abc", content="text", content_type="finding")
        assert item.id == "abc"
        assert item.content == "text"
        assert item.content_type == "finding"
        assert item.inquiry_id is None

    def test_optional_inquiry_id(self) -> None:
        item = SemanticItemSummary(
            id="x", content="y", content_type="z", inquiry_id="inq-1"
        )
        assert item.inquiry_id == "inq-1"


# ---------------------------------------------------------------------------
# CandidatePair
# ---------------------------------------------------------------------------


class TestCandidatePair:
    def test_fields_stored(self) -> None:
        pair = _make_pair(similarity=0.91)
        assert pair.similarity == 0.91
        assert pair.item_a.content == "The sky is blue."
        assert pair.item_b.content == "The sky is not blue."


# ---------------------------------------------------------------------------
# DetectionTrigger
# ---------------------------------------------------------------------------


class TestDetectionTrigger:
    def test_values(self) -> None:
        assert DetectionTrigger.CRON_NIGHTLY == "cron_nightly"
        assert DetectionTrigger.WRITE_THRESHOLD == "write_threshold"
        assert DetectionTrigger.INQUIRY_COMPLETION == "inquiry_completion"

    def test_is_str(self) -> None:
        assert isinstance(DetectionTrigger.CRON_NIGHTLY, str)


# ---------------------------------------------------------------------------
# _assess_conflict
# ---------------------------------------------------------------------------


class TestAssessConflict:
    def test_conflict_response_parsed(self) -> None:
        pair = _make_pair()
        confirmed, summary = _assess_conflict(pair, _inference_conflict)
        assert confirmed is True
        assert "contradict" in summary

    def test_no_conflict_response_parsed(self) -> None:
        pair = _make_pair()
        confirmed, summary = _assess_conflict(pair, _inference_no_conflict)
        assert confirmed is False
        assert "different" in summary

    def test_ambiguous_response_treated_as_no_conflict(self) -> None:
        pair = _make_pair()
        # Ambiguous responses fall back to NO_CONFLICT (conservative).
        # The warning goes through logging, not Python's warnings module.
        confirmed, _ = _assess_conflict(pair, _inference_ambiguous)
        assert confirmed is False

    def test_multiline_conflict_summary_captured(self) -> None:
        def inference(prompt: str) -> str:  # noqa: ARG001
            return "CONFLICT: Line one.\nLine two."

        pair = _make_pair()
        confirmed, summary = _assess_conflict(pair, inference)
        assert confirmed is True
        assert "Line one" in summary

    def test_case_exact_match(self) -> None:
        """Verdict must be uppercase CONFLICT or NO_CONFLICT."""

        def inference_lower(prompt: str) -> str:  # noqa: ARG001
            return "conflict: lowercase"

        pair = _make_pair()
        confirmed, _ = _assess_conflict(pair, inference_lower)
        assert confirmed is False  # lowercase not recognised → NO_CONFLICT fallback


# ---------------------------------------------------------------------------
# _write_contradiction_holding_item
# ---------------------------------------------------------------------------


class TestWriteContradictionHoldingItem:
    def test_holding_item_written(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair = _make_pair()
        item = _write_contradiction_holding_item(
            contradiction_id="ctr-001",
            pair=pair,
            conflict_summary="They conflict.",
            holding_store=store,
        )
        assert item.contradiction_id == "ctr-001"
        assert item.type == HoldingType.REASONED_DISAGREEMENT
        assert item.content == "They conflict."

    def test_register_needed_is_reflective(self, tmp_path: Path) -> None:
        """Register gate constraint: never urgent (§3F acceptance criteria)."""
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair = _make_pair()
        item = _write_contradiction_holding_item(
            contradiction_id="ctr-002",
            pair=pair,
            conflict_summary="Conflict.",
            holding_store=store,
        )
        assert item.register_needed == RegisterNeeded.REFLECTIVE
        assert item.register_needed != "urgent"

    def test_source_workflow_tagged(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair = _make_pair()
        item = _write_contradiction_holding_item(
            contradiction_id="ctr-003",
            pair=pair,
            conflict_summary="Conflict.",
            holding_store=store,
        )
        assert item.source_workflow == "contradiction_detection"

    def test_relevance_trigger_contains_item_ids(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair = _make_pair()
        item = _write_contradiction_holding_item(
            contradiction_id="ctr-004",
            pair=pair,
            conflict_summary="Conflict.",
            holding_store=store,
        )
        assert pair.item_a.id[:8] in item.relevance_trigger
        assert pair.item_b.id[:8] in item.relevance_trigger


# ---------------------------------------------------------------------------
# contradiction_detection — empty input
# ---------------------------------------------------------------------------


class TestContradictionDetectionEmpty:
    def test_no_items_returns_zero_counts(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        result = contradiction_detection(
            new_items=[],
            trigger=DetectionTrigger.CRON_NIGHTLY,
            inference_fn=_inference_conflict,
            find_candidate_pairs_fn=_no_pairs,
            create_contradiction_fn=_make_create_contradiction_fn(),
            holding_store=store,
        )
        assert result.items_assessed == 0
        assert result.candidate_pairs == 0
        assert result.contradictions_written == 0

    def test_no_items_inference_never_called(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        called: list[str] = []

        def inference(prompt: str) -> str:
            called.append(prompt)
            return "CONFLICT: x"

        contradiction_detection(
            new_items=[],
            trigger=DetectionTrigger.CRON_NIGHTLY,
            inference_fn=inference,
            find_candidate_pairs_fn=_no_pairs,
            create_contradiction_fn=_make_create_contradiction_fn(),
            holding_store=store,
        )
        assert called == []


# ---------------------------------------------------------------------------
# contradiction_detection — no candidate pairs
# ---------------------------------------------------------------------------


class TestNoCandidatePairs:
    def test_no_pairs_zero_contradictions(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        items = [_make_item(), _make_item()]
        result = contradiction_detection(
            new_items=items,
            trigger=DetectionTrigger.WRITE_THRESHOLD,
            inference_fn=_inference_conflict,
            find_candidate_pairs_fn=_no_pairs,
            create_contradiction_fn=_make_create_contradiction_fn(),
            holding_store=store,
        )
        assert result.items_assessed == 2
        assert result.candidate_pairs == 0
        assert result.contradictions_written == 0

    def test_no_pairs_holding_store_empty(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        contradiction_detection(
            new_items=[_make_item()],
            trigger=DetectionTrigger.CRON_NIGHTLY,
            inference_fn=_inference_conflict,
            find_candidate_pairs_fn=_no_pairs,
            create_contradiction_fn=_make_create_contradiction_fn(),
            holding_store=store,
        )
        assert store.list_all() == []


# ---------------------------------------------------------------------------
# contradiction_detection — confirmed conflict
# ---------------------------------------------------------------------------


class TestConfirmedConflict:
    def test_one_conflict_written(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair = _make_pair()

        def one_pair(items: list[SemanticItemSummary]) -> list[CandidatePair]:
            return [pair]

        create_fn = _make_create_contradiction_fn()
        result = contradiction_detection(
            new_items=[pair.item_a, pair.item_b],
            trigger=DetectionTrigger.CRON_NIGHTLY,
            inference_fn=_inference_conflict,
            find_candidate_pairs_fn=one_pair,
            create_contradiction_fn=create_fn,
            holding_store=store,
        )
        assert result.contradictions_written == 1
        assert len(result.contradiction_ids) == 1
        items = store.list_all()
        assert len(items) == 1
        assert items[0].type == HoldingType.REASONED_DISAGREEMENT
        assert items[0].contradiction_id is not None

    def test_create_contradiction_fn_called_with_correct_ids(
        self, tmp_path: Path
    ) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair = _make_pair()

        def one_pair(items: list[SemanticItemSummary]) -> list[CandidatePair]:
            return [pair]

        create_fn = _make_create_contradiction_fn()
        contradiction_detection(
            new_items=[pair.item_a, pair.item_b],
            trigger=DetectionTrigger.CRON_NIGHTLY,
            inference_fn=_inference_conflict,
            find_candidate_pairs_fn=one_pair,
            create_contradiction_fn=create_fn,
            holding_store=store,
        )
        assert len(create_fn.calls) == 1  # type: ignore[attr-defined]
        a_id, b_id, summary = create_fn.calls[0]  # type: ignore[attr-defined]
        assert a_id == pair.item_a.id
        assert b_id == pair.item_b.id
        assert "contradict" in summary

    def test_holding_item_contradiction_id_matches_record(self, tmp_path: Path) -> None:
        """The holding item's contradiction_id must match what create_contradiction_fn
        returned."""
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair = _make_pair()
        fixed_id = "fixed-contradiction-uuid"

        def one_pair(items: list[SemanticItemSummary]) -> list[CandidatePair]:
            return [pair]

        def create_fn(a: str, b: str, summary: str) -> str:
            return fixed_id

        result = contradiction_detection(
            new_items=[pair.item_a, pair.item_b],
            trigger=DetectionTrigger.CRON_NIGHTLY,
            inference_fn=_inference_conflict,
            find_candidate_pairs_fn=one_pair,
            create_contradiction_fn=create_fn,
            holding_store=store,
        )
        assert result.contradiction_ids == [fixed_id]
        holding_items = store.list_all()
        assert holding_items[0].contradiction_id == fixed_id


# ---------------------------------------------------------------------------
# contradiction_detection — no conflict
# ---------------------------------------------------------------------------


class TestNoConflict:
    def test_no_conflict_not_written(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair = _make_pair()

        def one_pair(items: list[SemanticItemSummary]) -> list[CandidatePair]:
            return [pair]

        result = contradiction_detection(
            new_items=[pair.item_a, pair.item_b],
            trigger=DetectionTrigger.CRON_NIGHTLY,
            inference_fn=_inference_no_conflict,
            find_candidate_pairs_fn=one_pair,
            create_contradiction_fn=_make_create_contradiction_fn(),
            holding_store=store,
        )
        assert result.candidate_pairs == 1
        assert result.contradictions_written == 0
        assert store.list_all() == []


# ---------------------------------------------------------------------------
# contradiction_detection — multiple pairs
# ---------------------------------------------------------------------------


class TestMultiplePairs:
    def test_partial_conflicts(self, tmp_path: Path) -> None:
        """2 candidate pairs, only first one confirmed — 1 contradiction written."""
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair_a = _make_pair("A says X.", "A says not-X.")
        pair_b = _make_pair("B is tall.", "B is short.")
        responses = iter(
            [
                "CONFLICT: Direct contradiction.",
                "NO_CONFLICT: Different subjects.",
            ]
        )

        def alternating_inference(prompt: str) -> str:  # noqa: ARG001
            return next(responses)

        def two_pairs(items: list[SemanticItemSummary]) -> list[CandidatePair]:
            return [pair_a, pair_b]

        result = contradiction_detection(
            new_items=[pair_a.item_a, pair_a.item_b, pair_b.item_a, pair_b.item_b],
            trigger=DetectionTrigger.WRITE_THRESHOLD,
            inference_fn=alternating_inference,
            find_candidate_pairs_fn=two_pairs,
            create_contradiction_fn=_make_create_contradiction_fn(),
            holding_store=store,
        )
        assert result.candidate_pairs == 2
        assert result.contradictions_written == 1
        assert len(store.list_all()) == 1

    def test_all_confirmed_all_written(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pairs = [_make_pair() for _ in range(3)]

        def all_pairs(items: list[SemanticItemSummary]) -> list[CandidatePair]:
            return pairs

        result = contradiction_detection(
            new_items=[p.item_a for p in pairs] + [p.item_b for p in pairs],
            trigger=DetectionTrigger.CRON_NIGHTLY,
            inference_fn=_inference_conflict,
            find_candidate_pairs_fn=all_pairs,
            create_contradiction_fn=_make_create_contradiction_fn(),
            holding_store=store,
        )
        assert result.contradictions_written == 3
        assert len(result.contradiction_ids) == 3
        assert len(store.list_all()) == 3


# ---------------------------------------------------------------------------
# contradiction_detection — trigger types and inquiry_id
# ---------------------------------------------------------------------------


class TestTriggerAndInquiryId:
    def test_trigger_preserved_in_result(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        for trigger in DetectionTrigger:
            result = contradiction_detection(
                new_items=[],
                trigger=trigger,
                inference_fn=_inference_conflict,
                find_candidate_pairs_fn=_no_pairs,
                create_contradiction_fn=_make_create_contradiction_fn(),
                holding_store=store,
            )
            assert result.trigger == trigger

    def test_inquiry_id_threaded_through(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        inq_id = "inq-uuid-abc"
        result = contradiction_detection(
            new_items=[],
            trigger=DetectionTrigger.INQUIRY_COMPLETION,
            inference_fn=_inference_conflict,
            find_candidate_pairs_fn=_no_pairs,
            create_contradiction_fn=_make_create_contradiction_fn(),
            holding_store=store,
            inquiry_id=inq_id,
        )
        assert result.inquiry_id == inq_id

    def test_inquiry_id_none_by_default(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        result = contradiction_detection(
            new_items=[],
            trigger=DetectionTrigger.CRON_NIGHTLY,
            inference_fn=_inference_conflict,
            find_candidate_pairs_fn=_no_pairs,
            create_contradiction_fn=_make_create_contradiction_fn(),
            holding_store=store,
        )
        assert result.inquiry_id is None

    def test_find_candidate_pairs_fn_receives_new_items(self, tmp_path: Path) -> None:
        """The finder callable is called with the exact items passed in."""
        store = HoldingStore(path=tmp_path / "holding.yaml")
        items = [_make_item(), _make_item(), _make_item()]
        received: list[list[SemanticItemSummary]] = []

        def capturing_finder(
            its: list[SemanticItemSummary],
        ) -> list[CandidatePair]:
            received.append(its)
            return []

        contradiction_detection(
            new_items=items,
            trigger=DetectionTrigger.WRITE_THRESHOLD,
            inference_fn=_inference_conflict,
            find_candidate_pairs_fn=capturing_finder,
            create_contradiction_fn=_make_create_contradiction_fn(),
            holding_store=store,
        )
        assert len(received) == 1
        assert received[0] == items


# ---------------------------------------------------------------------------
# contradiction_detection — preemption (suspend)
# ---------------------------------------------------------------------------


class TestSuspendPreemption:
    def test_checkpoint_called_and_workflow_resumes(self, tmp_path: Path) -> None:
        """Suspend preemption: workflow checkpoints, blocks, resumes after rollback."""
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair = _make_pair()
        ctx = PreemptionContext(PreemptionMode.SUSPEND)

        ckpt_calls: list[int] = []
        rb_calls: list[int] = []

        def checkpoint() -> None:
            ckpt_calls.append(1)

        def rollback() -> None:
            rb_calls.append(1)

        def one_pair(items: list[SemanticItemSummary]) -> list[CandidatePair]:
            return [pair]

        result_holder: list[ContradictionDetectionResult] = []

        def run_workflow() -> None:
            res = contradiction_detection(
                new_items=[pair.item_a, pair.item_b],
                trigger=DetectionTrigger.CRON_NIGHTLY,
                inference_fn=_inference_conflict,
                find_candidate_pairs_fn=one_pair,
                create_contradiction_fn=_make_create_contradiction_fn(),
                holding_store=store,
                preemption_ctx=ctx,
                checkpoint_fn=checkpoint,
                rollback_fn=rollback,
            )
            result_holder.append(res)

        ctx.preempt()  # signal before thread starts so cooperate() sees it immediately
        t = threading.Thread(target=run_workflow, daemon=True)
        t.start()
        assert ctx.wait_for_checkpoint(timeout=_THREAD_TIMEOUT)
        assert len(ckpt_calls) == 1

        ctx.resume()
        t.join(timeout=_THREAD_TIMEOUT)
        assert not t.is_alive()
        assert len(rb_calls) == 1
        assert result_holder[0].contradictions_written == 1

    def test_no_preemption_ctx_runs_normally(self, tmp_path: Path) -> None:
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair = _make_pair()

        def one_pair(items: list[SemanticItemSummary]) -> list[CandidatePair]:
            return [pair]

        result = contradiction_detection(
            new_items=[pair.item_a, pair.item_b],
            trigger=DetectionTrigger.CRON_NIGHTLY,
            inference_fn=_inference_conflict,
            find_candidate_pairs_fn=one_pair,
            create_contradiction_fn=_make_create_contradiction_fn(),
            holding_store=store,
        )
        assert result.contradictions_written == 1


# ---------------------------------------------------------------------------
# contradiction_detection — preemption (restart)
# ---------------------------------------------------------------------------


class TestRestartPreemption:
    def test_restart_raises_workflow_cancelled_error(self, tmp_path: Path) -> None:
        """Restart preemption: WorkflowCancelledError raised at cooperate()."""
        from kai_daemon.workflows.preemption import WorkflowCancelledError

        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair = _make_pair()
        ctx = PreemptionContext(PreemptionMode.RESTART)
        ctx.preempt()

        def one_pair(items: list[SemanticItemSummary]) -> list[CandidatePair]:
            return [pair]

        with pytest.raises(WorkflowCancelledError):
            contradiction_detection(
                new_items=[pair.item_a, pair.item_b],
                trigger=DetectionTrigger.CRON_NIGHTLY,
                inference_fn=_inference_conflict,
                find_candidate_pairs_fn=one_pair,
                create_contradiction_fn=_make_create_contradiction_fn(),
                holding_store=store,
                preemption_ctx=ctx,
            )

    def test_restart_before_write_no_holding_item(self, tmp_path: Path) -> None:
        """If preemption fires before assessment, no holding item is written."""
        from kai_daemon.workflows.preemption import WorkflowCancelledError

        store = HoldingStore(path=tmp_path / "holding.yaml")
        pair = _make_pair()
        ctx = PreemptionContext(PreemptionMode.RESTART)
        ctx.preempt()

        def one_pair(items: list[SemanticItemSummary]) -> list[CandidatePair]:
            return [pair]

        with pytest.raises(WorkflowCancelledError):
            contradiction_detection(
                new_items=[pair.item_a, pair.item_b],
                trigger=DetectionTrigger.CRON_NIGHTLY,
                inference_fn=_inference_conflict,
                find_candidate_pairs_fn=one_pair,
                create_contradiction_fn=_make_create_contradiction_fn(),
                holding_store=store,
                preemption_ctx=ctx,
            )
        assert store.list_all() == []


# ---------------------------------------------------------------------------
# Register gate invariant
# ---------------------------------------------------------------------------


class TestRegisterGateInvariant:
    def test_no_urgent_register_needed_written(self, tmp_path: Path) -> None:
        """Acceptance criteria: register gate excludes urgent (§3F)."""
        store = HoldingStore(path=tmp_path / "holding.yaml")
        pairs = [_make_pair() for _ in range(5)]

        def all_pairs(items: list[SemanticItemSummary]) -> list[CandidatePair]:
            return pairs

        contradiction_detection(
            new_items=[p.item_a for p in pairs] + [p.item_b for p in pairs],
            trigger=DetectionTrigger.CRON_NIGHTLY,
            inference_fn=_inference_conflict,
            find_candidate_pairs_fn=all_pairs,
            create_contradiction_fn=_make_create_contradiction_fn(),
            holding_store=store,
        )
        for item in store.list_all():
            assert item.register_needed != "urgent", (
                f"Holding item {item.id} has register_needed='urgent' "
                "which violates the register gate constraint (§3F)"
            )

    def test_reasoned_disagreement_requires_contradiction_id(
        self, tmp_path: Path
    ) -> None:
        """HoldingStore enforces: type=reasoned_disagreement requires non-null
        contradiction_id."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            HoldingItem = __import__(  # noqa: N806
                "kai_daemon.state.holding", fromlist=["HoldingItem"]
            ).HoldingItem
            HoldingItem(
                content="conflict",
                type=HoldingType.REASONED_DISAGREEMENT,
                relevance_trigger="trigger",
                register_needed=RegisterNeeded.REFLECTIVE,
                urgency="medium",
                source_workflow="contradiction_detection",
                epistemic_origin="internal",
                contradiction_id=None,  # should raise
            )
