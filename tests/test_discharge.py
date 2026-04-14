"""Tests for holding store discharge logic (§8d).

Acceptance criteria:
- Both gates required; either alone does not discharge
- Contradiction record hydrated via contradiction_id before surfacing
- At most one item per turn
- urgent register never produces a contradiction discharge
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from kai_daemon.state._types import EpistemicOrigin
from kai_daemon.state.discharge import (
    DEFAULT_DISCHARGE_THRESHOLD,
    ContradictionClientProtocol,
    ContradictionRecord,
    _register_matches,
    hydrate_contradiction,
    select_discharge_candidate,
)
from kai_daemon.state.holding import (
    HoldingItem,
    HoldingType,
    RegisterNeeded,
    Urgency,
)

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

_THRESHOLD = DEFAULT_DISCHARGE_THRESHOLD  # 0.72
_ABOVE = _THRESHOLD + 0.10  # 0.82 — passes gate 1
_BELOW = _THRESHOLD - 0.10  # 0.62 — fails gate 1
_EXACT = _THRESHOLD  # 0.72 — exactly at threshold, does NOT pass (strict >)


def _item(
    holding_type: HoldingType = HoldingType.OBSERVATION,
    register_needed: RegisterNeeded = RegisterNeeded.ANY,
    surfaced: str | None = None,
    contradiction_id: str | None = None,
    item_id: str | None = None,
) -> HoldingItem:
    kwargs: dict[str, Any] = {
        "content": "something noticed",
        "type": holding_type,
        "relevance_trigger": "when this topic comes up",
        "register_needed": register_needed,
        "urgency": Urgency.LOW,
        "source_workflow": "test",
        "epistemic_origin": EpistemicOrigin.INTERNAL,
        "contradiction_id": contradiction_id,
    }
    if surfaced is not None:
        kwargs["surfaced"] = surfaced
    if item_id is not None:
        kwargs["id"] = item_id
    return HoldingItem(**kwargs)


def _scores(*items: HoldingItem, score: float = _ABOVE) -> dict[str, float]:
    """Assign the same score to all items."""
    return {i.id: score for i in items}


# ---------------------------------------------------------------------------
# _register_matches unit tests
# ---------------------------------------------------------------------------


class TestRegisterMatches:
    def test_any_matches_casual(self) -> None:
        assert _register_matches(RegisterNeeded.ANY, "casual") is True

    def test_any_matches_urgent(self) -> None:
        assert _register_matches(RegisterNeeded.ANY, "urgent") is True

    def test_any_matches_exploratory(self) -> None:
        assert _register_matches(RegisterNeeded.ANY, "exploratory") is True

    def test_any_matches_reflective(self) -> None:
        assert _register_matches(RegisterNeeded.ANY, "reflective") is True

    def test_exact_match_exploratory(self) -> None:
        assert _register_matches(RegisterNeeded.EXPLORATORY, "exploratory") is True

    def test_exact_match_reflective(self) -> None:
        assert _register_matches(RegisterNeeded.REFLECTIVE, "reflective") is True

    def test_exact_match_casual(self) -> None:
        assert _register_matches(RegisterNeeded.CASUAL, "casual") is True

    def test_mismatch_reflective_vs_casual(self) -> None:
        assert _register_matches(RegisterNeeded.REFLECTIVE, "casual") is False

    def test_mismatch_exploratory_vs_reflective(self) -> None:
        assert _register_matches(RegisterNeeded.EXPLORATORY, "reflective") is False

    def test_mismatch_casual_vs_urgent(self) -> None:
        assert _register_matches(RegisterNeeded.CASUAL, "urgent") is False


# ---------------------------------------------------------------------------
# Gate 1: similarity threshold — strict >
# ---------------------------------------------------------------------------


class TestGate1Similarity:
    def test_above_threshold_passes(self) -> None:
        item = _item()
        result = select_discharge_candidate([item], "casual", {item.id: _ABOVE})
        assert result is item

    def test_exactly_at_threshold_fails(self) -> None:
        """Threshold is strictly greater-than, not >=."""
        item = _item()
        result = select_discharge_candidate([item], "casual", {item.id: _EXACT})
        assert result is None

    def test_below_threshold_fails(self) -> None:
        item = _item()
        result = select_discharge_candidate([item], "casual", {item.id: _BELOW})
        assert result is None

    def test_missing_score_treated_as_zero(self) -> None:
        """Items absent from the scores dict are treated as score 0."""
        item = _item()
        result = select_discharge_candidate(
            [item],
            "casual",
            {},  # item.id absent
        )
        assert result is None

    def test_custom_threshold_honoured(self) -> None:
        item = _item()
        # item at 0.50; custom threshold 0.40 → passes
        result = select_discharge_candidate(
            [item], "casual", {item.id: 0.50}, threshold=0.40
        )
        assert result is item

    def test_custom_threshold_blocks(self) -> None:
        item = _item()
        # item at 0.50; custom threshold 0.60 → fails
        result = select_discharge_candidate(
            [item], "casual", {item.id: 0.50}, threshold=0.60
        )
        assert result is None


# ---------------------------------------------------------------------------
# Gate 2: register match
# ---------------------------------------------------------------------------


class TestGate2Register:
    def test_any_passes_casual(self) -> None:
        item = _item(register_needed=RegisterNeeded.ANY)
        result = select_discharge_candidate([item], "casual", _scores(item))
        assert result is item

    def test_any_passes_exploratory(self) -> None:
        item = _item(register_needed=RegisterNeeded.ANY)
        result = select_discharge_candidate([item], "exploratory", _scores(item))
        assert result is item

    def test_reflective_passes_reflective(self) -> None:
        item = _item(register_needed=RegisterNeeded.REFLECTIVE)
        result = select_discharge_candidate([item], "reflective", _scores(item))
        assert result is item

    def test_reflective_fails_casual(self) -> None:
        item = _item(register_needed=RegisterNeeded.REFLECTIVE)
        result = select_discharge_candidate([item], "casual", _scores(item))
        assert result is None

    def test_exploratory_fails_reflective(self) -> None:
        item = _item(register_needed=RegisterNeeded.EXPLORATORY)
        result = select_discharge_candidate([item], "reflective", _scores(item))
        assert result is None

    def test_casual_fails_urgent(self) -> None:
        item = _item(register_needed=RegisterNeeded.CASUAL)
        result = select_discharge_candidate([item], "urgent", _scores(item))
        assert result is None


# ---------------------------------------------------------------------------
# Both gates required — either alone is not sufficient
# ---------------------------------------------------------------------------


class TestBothGatesRequired:
    def test_high_score_but_wrong_register_fails(self) -> None:
        """Gate 1 passes, gate 2 fails → no discharge."""
        item = _item(register_needed=RegisterNeeded.REFLECTIVE)
        result = select_discharge_candidate([item], "casual", {item.id: 0.99})
        assert result is None

    def test_right_register_but_low_score_fails(self) -> None:
        """Gate 2 passes, gate 1 fails → no discharge."""
        item = _item(register_needed=RegisterNeeded.CASUAL)
        result = select_discharge_candidate([item], "casual", {item.id: 0.10})
        assert result is None

    def test_both_pass_discharges(self) -> None:
        """Both gates pass → discharge."""
        item = _item(register_needed=RegisterNeeded.CASUAL)
        result = select_discharge_candidate([item], "casual", {item.id: _ABOVE})
        assert result is item


# ---------------------------------------------------------------------------
# Already-surfaced items excluded
# ---------------------------------------------------------------------------


class TestSurfacedExclusion:
    def test_surfaced_item_skipped(self) -> None:
        import datetime

        surfaced_ts = datetime.datetime.now(datetime.UTC).isoformat()
        item = _item(surfaced=surfaced_ts)
        result = select_discharge_candidate([item], "casual", _scores(item))
        assert result is None

    def test_surfaced_skipped_unsurfaced_returned(self) -> None:
        import datetime

        surfaced_ts = datetime.datetime.now(datetime.UTC).isoformat()
        done = _item(surfaced=surfaced_ts)
        fresh = _item()
        result = select_discharge_candidate(
            [done, fresh], "casual", _scores(done, fresh)
        )
        assert result is fresh


# ---------------------------------------------------------------------------
# At most one item per turn — highest score wins
# ---------------------------------------------------------------------------


class TestAtMostOnePerTurn:
    def test_empty_list_returns_none(self) -> None:
        result = select_discharge_candidate([], "casual", {})
        assert result is None

    def test_single_candidate_returned(self) -> None:
        item = _item()
        result = select_discharge_candidate([item], "casual", _scores(item))
        assert result is item

    def test_highest_score_wins(self) -> None:
        low = _item()
        high = _item()
        scores = {low.id: 0.80, high.id: 0.95}
        result = select_discharge_candidate([low, high], "casual", scores)
        assert result is high

    def test_only_one_returned_even_with_many_candidates(self) -> None:
        items = [_item() for _ in range(5)]
        scores = {i.id: _ABOVE for i in items}
        result = select_discharge_candidate(items, "casual", scores)
        assert result is not None
        assert result in items

    def test_second_best_not_returned(self) -> None:
        winner = _item()
        runner_up = _item()
        scores = {winner.id: 0.95, runner_up.id: 0.85}
        result = select_discharge_candidate([winner, runner_up], "casual", scores)
        assert result is winner


# ---------------------------------------------------------------------------
# urgent register + contradiction discharge exclusion
# ---------------------------------------------------------------------------


class TestUrgentContradictionExclusion:
    """Acceptance criterion: urgent register never discharges contradictions."""

    def _contradiction_item(self) -> HoldingItem:
        return _item(
            holding_type=HoldingType.REASONED_DISAGREEMENT,
            register_needed=RegisterNeeded.ANY,
            contradiction_id=str(uuid.uuid4()),
        )

    def test_urgent_excludes_reasoned_disagreement(self) -> None:
        item = self._contradiction_item()
        result = select_discharge_candidate([item], "urgent", _scores(item))
        assert result is None

    def test_urgent_allows_non_contradiction(self) -> None:
        """Non-contradiction items with register_needed=ANY can discharge on urgent."""
        item = _item(
            holding_type=HoldingType.OBSERVATION, register_needed=RegisterNeeded.ANY
        )
        result = select_discharge_candidate([item], "urgent", _scores(item))
        assert result is item

    def test_reflective_allows_reasoned_disagreement(self) -> None:
        """reflective + reasoned_disagreement → both gates pass, discharges."""
        item = _item(
            holding_type=HoldingType.REASONED_DISAGREEMENT,
            register_needed=RegisterNeeded.REFLECTIVE,
            contradiction_id=str(uuid.uuid4()),
        )
        result = select_discharge_candidate([item], "reflective", _scores(item))
        assert result is item

    def test_urgent_blocks_contradiction_even_when_another_item_present(self) -> None:
        """urgent blocks contradiction; non-contradiction item returned instead."""
        contradiction = self._contradiction_item()
        observation = _item(
            holding_type=HoldingType.OBSERVATION, register_needed=RegisterNeeded.ANY
        )
        # Give contradiction a higher score to prove it's blocked by the register rule,
        # not by score ordering.
        scores = {contradiction.id: 0.99, observation.id: 0.85}
        result = select_discharge_candidate(
            [contradiction, observation], "urgent", scores
        )
        assert result is observation

    def test_exploratory_allows_reasoned_disagreement_with_any(self) -> None:
        item = _item(
            holding_type=HoldingType.REASONED_DISAGREEMENT,
            register_needed=RegisterNeeded.ANY,
            contradiction_id=str(uuid.uuid4()),
        )
        result = select_discharge_candidate([item], "exploratory", _scores(item))
        assert result is item


# ---------------------------------------------------------------------------
# Contradiction hydration
# ---------------------------------------------------------------------------


class _OkClient:
    """Contradiction client that always returns a valid record."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get_contradiction(self, contradiction_id: str) -> ContradictionRecord:
        self.calls.append(contradiction_id)
        return ContradictionRecord(
            id=contradiction_id,
            item_a_id="aaa",
            item_b_id="bbb",
            conflict_summary="They contradict each other.",
        )


class _NoneClient:
    """Contradiction client that returns None (record not found)."""

    async def get_contradiction(self, contradiction_id: str) -> None:
        return None


class _ErrorClient:
    """Contradiction client that always raises."""

    async def get_contradiction(self, contradiction_id: str) -> ContradictionRecord:
        raise OSError("connection refused")


class TestHydrateContradiction:
    def test_none_contradiction_id_returns_none_immediately(self) -> None:
        item = _item(contradiction_id=None)
        result = asyncio.run(hydrate_contradiction(item, _OkClient()))
        assert result is None

    def test_client_not_called_when_no_contradiction_id(self) -> None:
        client = _OkClient()
        item = _item(contradiction_id=None)
        asyncio.run(hydrate_contradiction(item, client))
        assert client.calls == []

    def test_hydrates_record_when_contradiction_id_present(self) -> None:
        cid = str(uuid.uuid4())
        item = _item(
            holding_type=HoldingType.REASONED_DISAGREEMENT,
            contradiction_id=cid,
        )
        record = asyncio.run(hydrate_contradiction(item, _OkClient()))
        assert record is not None
        assert record.id == cid
        assert record.item_a_id == "aaa"
        assert record.item_b_id == "bbb"
        assert record.conflict_summary == "They contradict each other."

    def test_passes_contradiction_id_to_client(self) -> None:
        cid = "test-contradiction-id-123"
        item = _item(
            holding_type=HoldingType.REASONED_DISAGREEMENT,
            contradiction_id=cid,
        )
        client = _OkClient()
        asyncio.run(hydrate_contradiction(item, client))
        assert client.calls == [cid]

    def test_returns_none_when_client_returns_none(self) -> None:
        item = _item(
            holding_type=HoldingType.REASONED_DISAGREEMENT,
            contradiction_id=str(uuid.uuid4()),
        )
        result = asyncio.run(hydrate_contradiction(item, _NoneClient()))
        assert result is None

    def test_returns_none_on_client_error_no_exception_propagated(self) -> None:
        item = _item(
            holding_type=HoldingType.REASONED_DISAGREEMENT,
            contradiction_id=str(uuid.uuid4()),
        )
        result = asyncio.run(hydrate_contradiction(item, _ErrorClient()))
        assert result is None

    def test_contradiction_record_fields(self) -> None:
        """Verify ContradictionRecord is a proper dataclass."""
        record = ContradictionRecord(
            id="id-1",
            item_a_id="a-1",
            item_b_id="b-1",
            conflict_summary="Item A says X; Item B says not-X.",
        )
        assert record.id == "id-1"
        assert record.item_a_id == "a-1"
        assert record.item_b_id == "b-1"
        assert record.conflict_summary == "Item A says X; Item B says not-X."


# ---------------------------------------------------------------------------
# Protocol satisfaction check
# ---------------------------------------------------------------------------


class TestContradictionClientProtocol:
    def test_ok_client_satisfies_protocol(self) -> None:
        assert isinstance(_OkClient(), ContradictionClientProtocol)

    def test_none_client_satisfies_protocol(self) -> None:
        assert isinstance(_NoneClient(), ContradictionClientProtocol)

    def test_error_client_satisfies_protocol(self) -> None:
        assert isinstance(_ErrorClient(), ContradictionClientProtocol)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_items_above_threshold_returns_none(self) -> None:
        items = [_item() for _ in range(3)]
        scores = {i.id: 0.10 for i in items}
        result = select_discharge_candidate(items, "casual", scores)
        assert result is None

    def test_all_items_wrong_register_returns_none(self) -> None:
        items = [
            _item(register_needed=RegisterNeeded.REFLECTIVE),
            _item(register_needed=RegisterNeeded.EXPLORATORY),
        ]
        scores = {i.id: _ABOVE for i in items}
        result = select_discharge_candidate(items, "casual", scores)
        assert result is None

    def test_all_items_surfaced_returns_none(self) -> None:
        import datetime

        ts = datetime.datetime.now(datetime.UTC).isoformat()
        items = [_item(surfaced=ts) for _ in range(3)]
        scores = {i.id: _ABOVE for i in items}
        result = select_discharge_candidate(items, "casual", scores)
        assert result is None

    def test_mix_of_passing_and_failing_items(self) -> None:
        """Only the item that passes both gates is returned."""
        no_score = _item(register_needed=RegisterNeeded.ANY)
        wrong_reg = _item(register_needed=RegisterNeeded.REFLECTIVE)
        passes = _item(register_needed=RegisterNeeded.ANY)

        scores = {
            no_score.id: _BELOW,  # fails gate 1
            wrong_reg.id: _ABOVE,  # fails gate 2 (casual register)
            passes.id: _ABOVE,  # passes both
        }
        result = select_discharge_candidate(
            [no_score, wrong_reg, passes], "casual", scores
        )
        assert result is passes

    def test_urgent_with_no_contradiction_items_returns_observation(self) -> None:
        """urgent + only observations → observation discharges (register_needed=ANY)."""
        item = _item(
            holding_type=HoldingType.CONNECTION, register_needed=RegisterNeeded.ANY
        )
        result = select_discharge_candidate([item], "urgent", _scores(item))
        assert result is item

    def test_score_exactly_above_threshold(self) -> None:
        """Score of threshold + epsilon passes gate 1."""
        import sys

        item = _item()
        score = _THRESHOLD + sys.float_info.epsilon
        result = select_discharge_candidate([item], "casual", {item.id: score})
        assert result is item
