"""Tests for conversational retrieval (§4D).

Required test cases:
1. test_graceful_degradation_client_error — memory client raises → empty
   RetrievalContext, no exception propagated
2. test_graceful_degradation_returns_empty_context — the returned context is
   truly empty (both lists empty)
3. test_pending_artifact_surfaced — result with chunk_status:pending goes to
   pending_artifacts, not semantic
4. test_ready_result_in_semantic — result without chunk_status:pending goes to
   semantic
5. test_primary_query_spaces — primary query uses correct spaces and weights
6. test_primary_query_top_k — primary query uses top_k=5
7. test_no_secondary_query_without_peripheral — no peripheral thread → client
   called exactly once
8. test_secondary_query_on_peripheral_thread — peripheral thread triggers
   secondary query with central_question
9. test_merge_peripheral_weight_applied — secondary scores scaled by 0.4
10. test_merge_deduplicates_keeps_higher_score — same doc in both: higher wins
11. test_secondary_query_failure_graceful — secondary query raises → primary
    results still returned
12. test_no_secondary_query_without_thread_store — peripheral present but no
    thread_store → secondary query skipped gracefully
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from kai_daemon.state.retrieval import (
    RetrievalContext,
    SemanticQuery,
    SemanticResult,
    _merge,
    _partition_pending,
    conversational_retrieval,
)
from kai_daemon.state.thread_stack import (
    ThreadStackEntry,
    ThreadStackState,
)
from kai_daemon.state.threads import (
    EpistemicStatus,
    Stance,
    Thread,
    ThreadStatus,
    ThreadStore,
)

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_result(
    doc_id: str = "doc-1",
    score: float = 0.8,
    space: str = "user_pkm",
    metadata: dict[str, Any] | None = None,
) -> SemanticResult:
    return SemanticResult(
        document_id=doc_id,
        text="some text",
        score=score,
        space=space,
        metadata=metadata or {},
    )


def _make_stack_entry(
    thread_id: str,
    state: ThreadStackState = ThreadStackState.foreground,
    salience: float = 0.5,
) -> ThreadStackEntry:
    return ThreadStackEntry(
        thread_id=thread_id,
        state=state,
        salience=salience,
        engagement_depth=0.0,
        last_touched_turn=1,
        entered_turn=1,
    )


def _make_thread(thread_id: str, central_question: str = "What is truth?") -> Thread:
    return Thread(
        id=thread_id,
        title="Test Thread",
        central_question=central_question,
        current_state="ongoing",
        unresolved="nothing yet",
        stance=Stance(
            position="uncertain",
            epistemic_status=EpistemicStatus.UNCERTAIN,
        ),
        status=ThreadStatus.ACTIVE,
    )


class _OkClient:
    """Memory client that returns configurable results."""

    def __init__(self, results: list[SemanticResult]) -> None:
        self._results = results
        self.calls: list[SemanticQuery] = []

    async def semantic_query(self, query: SemanticQuery) -> list[SemanticResult]:
        self.calls.append(query)
        return list(self._results)


class _ErrorClient:
    """Memory client that always raises."""

    async def semantic_query(self, query: SemanticQuery) -> list[SemanticResult]:
        raise OSError("connection refused")


class _SecondaryErrorClient:
    """Memory client that succeeds on first call, raises on second."""

    def __init__(self, primary_results: list[SemanticResult]) -> None:
        self._primary = primary_results
        self._call_count = 0

    async def semantic_query(self, query: SemanticQuery) -> list[SemanticResult]:
        self._call_count += 1
        if self._call_count == 1:
            return list(self._primary)
        raise OSError("secondary failed")


# ---------------------------------------------------------------------------
# Test 1: graceful degradation — client error → empty context, no exception
# ---------------------------------------------------------------------------


def test_graceful_degradation_client_error() -> None:
    result = asyncio.run(
        conversational_retrieval(
            message="hello",
            thread_stack=[],
            memory_client=_ErrorClient(),
        )
    )
    assert isinstance(result, RetrievalContext)


# ---------------------------------------------------------------------------
# Test 2: graceful degradation returns genuinely empty context
# ---------------------------------------------------------------------------


def test_graceful_degradation_returns_empty_context() -> None:
    result = asyncio.run(
        conversational_retrieval(
            message="hello",
            thread_stack=[],
            memory_client=_ErrorClient(),
        )
    )
    assert result.is_empty
    assert result.semantic == []
    assert result.pending_artifacts == []


# ---------------------------------------------------------------------------
# Test 3: pending artifact is surfaced in pending_artifacts, not semantic
# ---------------------------------------------------------------------------


def test_pending_artifact_surfaced() -> None:
    pending_result = _make_result(
        doc_id="art-pending",
        metadata={"chunk_status": "pending", "title": "Draft Document"},
    )
    client = _OkClient([pending_result])

    result = asyncio.run(
        conversational_retrieval(
            message="hello",
            thread_stack=[],
            memory_client=client,
        )
    )

    assert result.semantic == []
    assert len(result.pending_artifacts) == 1
    assert result.pending_artifacts[0].document_id == "art-pending"
    assert result.has_pending
    assert not result.is_empty


# ---------------------------------------------------------------------------
# Test 4: ready result goes into semantic
# ---------------------------------------------------------------------------


def test_ready_result_in_semantic() -> None:
    ready_result = _make_result(doc_id="doc-ready", score=0.9)
    client = _OkClient([ready_result])

    result = asyncio.run(
        conversational_retrieval(
            message="hello",
            thread_stack=[],
            memory_client=client,
        )
    )

    assert len(result.semantic) == 1
    assert result.semantic[0].document_id == "doc-ready"
    assert result.pending_artifacts == []
    assert not result.has_pending


# ---------------------------------------------------------------------------
# Test 5: primary query uses correct spaces and space_weights
# ---------------------------------------------------------------------------


def test_primary_query_spaces() -> None:
    client = _OkClient([])

    asyncio.run(
        conversational_retrieval(
            message="test message",
            thread_stack=[],
            memory_client=client,
        )
    )

    assert len(client.calls) == 1
    query = client.calls[0]
    assert set(query.spaces) == {"user_pkm", "daemon", "shared"}
    assert query.space_weights["user_pkm"] == pytest.approx(1.0)  # type: ignore[reportUnknownMemberType]
    assert query.space_weights["daemon"] == pytest.approx(0.3)  # type: ignore[reportUnknownMemberType]
    assert query.space_weights["shared"] == pytest.approx(0.8)  # type: ignore[reportUnknownMemberType]
    assert query.always_include_shared is True


# ---------------------------------------------------------------------------
# Test 6: primary query uses top_k=5
# ---------------------------------------------------------------------------


def test_primary_query_top_k() -> None:
    client = _OkClient([])

    asyncio.run(
        conversational_retrieval(
            message="test",
            thread_stack=[],
            memory_client=client,
        )
    )

    assert client.calls[0].top_k == 5


# ---------------------------------------------------------------------------
# Test 7: no peripheral thread → memory client called exactly once
# ---------------------------------------------------------------------------


def test_no_secondary_query_without_peripheral() -> None:
    foreground_entry = _make_stack_entry("t1", ThreadStackState.foreground)
    client = _OkClient([_make_result()])

    asyncio.run(
        conversational_retrieval(
            message="hello",
            thread_stack=[foreground_entry],
            memory_client=client,
        )
    )

    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# Test 8: peripheral thread triggers secondary query on central_question
# ---------------------------------------------------------------------------


def test_secondary_query_on_peripheral_thread(tmp_path: Any) -> None:
    thread_id = str(uuid.uuid4())
    thread = _make_thread(thread_id, central_question="Is consciousness physical?")

    store = ThreadStore(
        threads_path=tmp_path / "threads",
        pickup_notes_path=tmp_path / "pickup",
        chroma_client=None,
    )
    store.create(thread)

    peripheral = _make_stack_entry(thread_id, ThreadStackState.peripheral)
    client = _OkClient([_make_result()])

    asyncio.run(
        conversational_retrieval(
            message="hello",
            thread_stack=[peripheral],
            memory_client=client,
            thread_store=store,
        )
    )

    assert len(client.calls) == 2
    secondary_call = client.calls[1]
    assert secondary_call.query_text == "Is consciousness physical?"
    assert secondary_call.top_k == 3


# ---------------------------------------------------------------------------
# Test 9: secondary scores scaled by peripheral_weight=0.4
# ---------------------------------------------------------------------------


def test_merge_peripheral_weight_applied() -> None:
    secondary = [_make_result("sec-doc", score=1.0)]
    merged = _merge([], secondary, peripheral_weight=0.4)
    assert len(merged) == 1
    assert merged[0].score == pytest.approx(0.4)  # type: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# Test 10: merge deduplicates — same doc in primary and secondary, higher wins
# ---------------------------------------------------------------------------


def test_merge_deduplicates_keeps_higher_score() -> None:
    primary = [_make_result("shared-doc", score=0.8)]
    secondary = [_make_result("shared-doc", score=0.6)]
    # scaled secondary = 0.6 * 0.4 = 0.24, which is less than primary 0.8
    merged = _merge(primary, secondary, peripheral_weight=0.4)
    assert len(merged) == 1
    assert merged[0].score == pytest.approx(0.8)  # type: ignore[reportUnknownMemberType]


def test_merge_secondary_wins_when_higher() -> None:
    primary = [_make_result("shared-doc", score=0.1)]
    secondary = [_make_result("shared-doc", score=0.9)]
    # scaled secondary = 0.9 * 0.4 = 0.36 > 0.1
    merged = _merge(primary, secondary, peripheral_weight=0.4)
    assert len(merged) == 1
    assert merged[0].score == pytest.approx(0.36)  # type: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# Test 11: secondary query failure → primary results still returned
# ---------------------------------------------------------------------------


def test_secondary_query_failure_graceful(tmp_path: Any) -> None:
    thread_id = str(uuid.uuid4())
    thread = _make_thread(thread_id)
    store = ThreadStore(
        threads_path=tmp_path / "threads",
        pickup_notes_path=tmp_path / "pickup",
        chroma_client=None,
    )
    store.create(thread)

    primary_result = _make_result("primary-doc", score=0.9)
    peripheral = _make_stack_entry(thread_id, ThreadStackState.peripheral)
    client = _SecondaryErrorClient([primary_result])

    result = asyncio.run(
        conversational_retrieval(
            message="hello",
            thread_stack=[peripheral],
            memory_client=client,
            thread_store=store,
        )
    )

    assert len(result.semantic) == 1
    assert result.semantic[0].document_id == "primary-doc"


# ---------------------------------------------------------------------------
# Test 12: peripheral present but no thread_store → secondary skipped, no error
# ---------------------------------------------------------------------------


def test_no_secondary_query_without_thread_store() -> None:
    peripheral = _make_stack_entry("t99", ThreadStackState.peripheral)
    primary_result = _make_result("doc-1", score=0.9)
    client = _OkClient([primary_result])

    result = asyncio.run(
        conversational_retrieval(
            message="hello",
            thread_stack=[peripheral],
            memory_client=client,
            # thread_store intentionally omitted
        )
    )

    # Only primary query fired
    assert len(client.calls) == 1
    assert len(result.semantic) == 1
    assert result.semantic[0].document_id == "doc-1"


# ---------------------------------------------------------------------------
# Unit tests for _partition_pending
# ---------------------------------------------------------------------------


def test_partition_pending_splits_correctly() -> None:
    ready = _make_result("r1", metadata={})
    pending = _make_result("p1", metadata={"chunk_status": "pending"})
    ready_out, pending_out = _partition_pending([ready, pending])
    assert [r.document_id for r in ready_out] == ["r1"]
    assert [r.document_id for r in pending_out] == ["p1"]


def test_partition_pending_all_ready() -> None:
    results = [_make_result("d1"), _make_result("d2")]
    ready, pending = _partition_pending(results)
    assert len(ready) == 2
    assert pending == []


def test_partition_pending_all_pending() -> None:
    results = [
        _make_result("d1", metadata={"chunk_status": "pending"}),
        _make_result("d2", metadata={"chunk_status": "pending"}),
    ]
    ready, pending = _partition_pending(results)
    assert ready == []
    assert len(pending) == 2
