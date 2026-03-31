"""Tests for the commissioned_inquiry workflow (§3E)."""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

from kai_daemon.state.inquiry import InquiryFinding, InquiryRecord, InquiryStatus
from kai_daemon.workflows.commissioned_inquiry import (
    CommissionedInquiryResult,
    WriteInquiryFindingFn,
    WriteInquiryRecordFn,
    _assert_query_sanitized,
    _format_prior_findings,
    _parse_finding_response,
    _parse_summary_response,
    commissioned_inquiry,
)
from kai_daemon.workflows.preemption import PreemptionContext, PreemptionMode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INQUIRY_ID = "inq-abc-123"
_QUESTION = "What are the key principles of epistemology?"
_SCOPE = "Western analytical tradition"
_THREAD_TIMEOUT = 2.0

# ---------------------------------------------------------------------------
# Inference stubs
# ---------------------------------------------------------------------------


def _query_inference(prompt: str) -> str:
    """Return a clean, self-contained query for query generation prompts."""
    if "Generate the next search query" in prompt:
        return "epistemology justified true belief"
    if "FINDING:" in prompt or "Synthesize the key finding" in prompt:
        return (
            "FINDING: Knowledge requires justification and truth.\n"
            "CONFIDENCE: medium\n"
            "OPEN_QUESTIONS: Is justification sufficient?\n"
        )
    if "SUMMARY:" in prompt or "Synthesize a final answer" in prompt:
        return (
            "SUMMARY: Epistemology concerns justified true belief.\n"
            "CONFIDENCE: high\n"
            "OPEN_QUESTIONS: - The Gettier problem remains open\n"
        )
    # push prompt
    return "I looked into this. Epistemology concerns how we know what we know."


# ---------------------------------------------------------------------------
# Noop callables
# ---------------------------------------------------------------------------


def _noop_write_record(r: InquiryRecord) -> None:
    pass


def _noop_complete(
    iid: str, summary: str, conf: float | None, count: int, oq: str | None
) -> None:
    pass


def _noop_retrieve(query: str) -> str:
    return "Context: classic debates on justified true belief."


def _noop_write_finding(f: InquiryFinding) -> None:
    pass


def _noop_contradiction(iid: str) -> None:
    pass


def _noop_push(msg: str, in_session: bool) -> None:
    pass


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def run_inquiry(
    inquiry_id: str = _INQUIRY_ID,
    question: str = _QUESTION,
    scope: str | None = _SCOPE,
    *,
    inference_fn: Callable[[str], str] = _query_inference,
    write_inquiry_record_fn: WriteInquiryRecordFn = _noop_write_record,
    mark_inquiry_complete_fn: Callable[
        [str, str, float | None, int, str | None], None
    ] = _noop_complete,
    retrieve_daemon_context_fn: Callable[[str], str] = _noop_retrieve,
    write_finding_fn: WriteInquiryFindingFn = _noop_write_finding,
    trigger_contradiction_detection_fn: Callable[[str], None] = _noop_contradiction,
    trigger_push_fn: Callable[[str, bool], None] = _noop_push,
    initiated_at: str | None = "2026-03-31T10:00:00+00:00",
    in_session: bool = False,
    max_iterations: int = 2,
    preemption_ctx: PreemptionContext | None = None,
    checkpoint_fn: Callable[[], None] | None = None,
    rollback_fn: Callable[[], None] | None = None,
) -> CommissionedInquiryResult:
    return commissioned_inquiry(
        inquiry_id=inquiry_id,
        question=question,
        scope=scope,
        inference_fn=inference_fn,
        write_inquiry_record_fn=write_inquiry_record_fn,
        mark_inquiry_complete_fn=mark_inquiry_complete_fn,
        retrieve_daemon_context_fn=retrieve_daemon_context_fn,
        write_finding_fn=write_finding_fn,
        trigger_contradiction_detection_fn=trigger_contradiction_detection_fn,
        trigger_push_fn=trigger_push_fn,
        initiated_at=initiated_at,
        in_session=in_session,
        max_iterations=max_iterations,
        preemption_ctx=preemption_ctx,
        checkpoint_fn=checkpoint_fn,
        rollback_fn=rollback_fn,
    )


# ---------------------------------------------------------------------------
# InquiryStatus
# ---------------------------------------------------------------------------


class TestInquiryStatus:
    def test_values(self) -> None:
        assert InquiryStatus.ACTIVE == "active"
        assert InquiryStatus.COMPLETED == "completed"
        assert InquiryStatus.ABANDONED == "abandoned"

    def test_is_str_enum(self) -> None:
        assert isinstance(InquiryStatus.ACTIVE, str)


# ---------------------------------------------------------------------------
# InquiryRecord dataclass
# ---------------------------------------------------------------------------


class TestInquiryRecord:
    def test_defaults(self) -> None:
        r = InquiryRecord(
            id="r1",
            initiated_at="2026-01-01T00:00:00+00:00",
            question="What is X?",
            scope=None,
        )
        assert r.status == InquiryStatus.ACTIVE
        assert r.completed_at is None
        assert r.summary is None
        assert r.findings_count == 0
        assert r.confidence_overall is None
        assert r.open_questions_remaining is None

    def test_scope_none(self) -> None:
        r = InquiryRecord(id="r2", initiated_at="t", question="Q?", scope=None)
        assert r.scope is None

    def test_scope_set(self) -> None:
        r = InquiryRecord(id="r3", initiated_at="t", question="Q?", scope="narrow")
        assert r.scope == "narrow"


# ---------------------------------------------------------------------------
# InquiryFinding dataclass
# ---------------------------------------------------------------------------


class TestInquiryFinding:
    def test_defaults(self) -> None:
        f = InquiryFinding(
            id="f1",
            inquiry_id="inq-1",
            iteration=0,
            content="Some finding.",
            epistemic_status="provisional",
            query_used="test query",
            written_at="2026-01-01T00:00:00+00:00",
        )
        assert f.sources_cited == []
        assert f.embedding_id is None
        assert f.open_questions is None

    def test_epistemic_status_provisional(self) -> None:
        f = InquiryFinding(
            id="f2",
            inquiry_id="inq-1",
            iteration=0,
            content="Finding.",
            epistemic_status="provisional",
            query_used="q",
            written_at="t",
        )
        assert f.epistemic_status == "provisional"


# ---------------------------------------------------------------------------
# _assert_query_sanitized
# ---------------------------------------------------------------------------


class TestAssertQuerySanitized:
    def test_clean_query_passes(self) -> None:
        _assert_query_sanitized("epistemology justified true belief")

    def test_user_pkm_raises(self) -> None:
        with pytest.raises(ValueError, match="user_pkm"):
            _assert_query_sanitized("Search user_pkm for notes on epistemology")

    def test_user_pkm_case_insensitive(self) -> None:
        with pytest.raises(ValueError, match="user_pkm"):
            _assert_query_sanitized("Search USER_PKM database")

    def test_private_note_raises(self) -> None:
        with pytest.raises(ValueError, match="private note"):
            _assert_query_sanitized("find my private note about philosophy")

    def test_personal_note_raises(self) -> None:
        with pytest.raises(ValueError, match="personal note"):
            _assert_query_sanitized("retrieve personal note on ethics")

    def test_journal_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="journal entry"):
            _assert_query_sanitized("journal entry about today's reading")

    def test_my_note_raises(self) -> None:
        with pytest.raises(ValueError, match="my note"):
            _assert_query_sanitized("find my note on Gettier problems")

    def test_user_pkm_phrase_raises(self) -> None:
        with pytest.raises(ValueError, match="user pkm"):
            _assert_query_sanitized("search user pkm collection")


# ---------------------------------------------------------------------------
# _format_prior_findings
# ---------------------------------------------------------------------------


class TestFormatPriorFindings:
    def _make_finding(
        self,
        iteration: int,
        content: str,
        open_questions: str | None = None,
    ) -> InquiryFinding:
        return InquiryFinding(
            id=str(iteration),
            inquiry_id="inq",
            iteration=iteration,
            content=content,
            epistemic_status="provisional",
            query_used="q",
            written_at="t",
            open_questions=open_questions,
        )

    def test_empty_returns_none_yet(self) -> None:
        assert _format_prior_findings([]) == "(none yet)"

    def test_single_finding(self) -> None:
        f = self._make_finding(0, "Knowledge is justified true belief.")
        result = _format_prior_findings([f])
        assert "[0]" in result
        assert "Knowledge is justified true belief." in result

    def test_open_questions_included(self) -> None:
        f = self._make_finding(1, "Content.", open_questions="Is it sufficient?")
        result = _format_prior_findings([f])
        assert "open: Is it sufficient?" in result

    def test_no_open_questions_no_pipe(self) -> None:
        f = self._make_finding(0, "Content.", open_questions=None)
        result = _format_prior_findings([f])
        assert "open:" not in result

    def test_multiple_findings_ordered(self) -> None:
        findings = [
            self._make_finding(0, "First."),
            self._make_finding(1, "Second."),
        ]
        result = _format_prior_findings(findings)
        assert result.index("[0]") < result.index("[1]")


# ---------------------------------------------------------------------------
# _parse_finding_response
# ---------------------------------------------------------------------------


class TestParseFindingResponse:
    def test_well_formed(self) -> None:
        response = (
            "FINDING: Knowledge requires truth.\n"
            "CONFIDENCE: high\n"
            "OPEN_QUESTIONS: What about Gettier cases?\n"
        )
        finding, confidence, oq = _parse_finding_response(response)
        assert finding == "Knowledge requires truth."
        assert confidence == "high"
        assert oq == "What about Gettier cases?"

    def test_confidence_medium(self) -> None:
        response = "FINDING: X\nCONFIDENCE: medium\nOPEN_QUESTIONS: none\n"
        _, confidence, oq = _parse_finding_response(response)
        assert confidence == "medium"
        assert oq is None

    def test_confidence_low(self) -> None:
        response = "FINDING: X\nCONFIDENCE: low\nOPEN_QUESTIONS: none\n"
        _, confidence, _ = _parse_finding_response(response)
        assert confidence == "low"

    def test_open_questions_none_string(self) -> None:
        response = "FINDING: X\nCONFIDENCE: medium\nOPEN_QUESTIONS: none\n"
        _, _, oq = _parse_finding_response(response)
        assert oq is None

    def test_open_questions_empty(self) -> None:
        response = "FINDING: X\nCONFIDENCE: medium\nOPEN_QUESTIONS: \n"
        _, _, oq = _parse_finding_response(response)
        assert oq is None

    def test_fallback_on_bad_format(self) -> None:
        response = "The model just returned this text without labels."
        finding, confidence, oq = _parse_finding_response(response)
        assert "model just returned" in finding
        assert confidence == "low"
        assert oq is None

    def test_unknown_confidence_falls_back_to_low(self) -> None:
        response = "FINDING: X\nCONFIDENCE: very_high\nOPEN_QUESTIONS: none\n"
        _, confidence, _ = _parse_finding_response(response)
        assert confidence == "low"


# ---------------------------------------------------------------------------
# _parse_summary_response
# ---------------------------------------------------------------------------


class TestParseSummaryResponse:
    def test_well_formed_high_confidence(self) -> None:
        response = (
            "SUMMARY: Epistemology concerns justification.\n"
            "CONFIDENCE: high\n"
            "OPEN_QUESTIONS: - Gettier cases\n"
        )
        summary, conf, oq = _parse_summary_response(response)
        assert summary == "Epistemology concerns justification."
        assert conf is not None and abs(conf - 0.9) < 1e-9
        assert oq is not None
        assert "Gettier" in oq

    def test_confidence_medium(self) -> None:
        response = "SUMMARY: S\nCONFIDENCE: medium\nOPEN_QUESTIONS: none\n"
        _, conf, oq = _parse_summary_response(response)
        assert conf is not None and abs(conf - 0.6) < 1e-9
        assert oq is None

    def test_confidence_low(self) -> None:
        response = "SUMMARY: S\nCONFIDENCE: low\nOPEN_QUESTIONS: none\n"
        _, conf, _ = _parse_summary_response(response)
        assert conf is not None and abs(conf - 0.3) < 1e-9

    def test_open_questions_none_string(self) -> None:
        response = "SUMMARY: S\nCONFIDENCE: high\nOPEN_QUESTIONS: none\n"
        _, _, oq = _parse_summary_response(response)
        assert oq is None

    def test_multi_line_open_questions(self) -> None:
        response = "SUMMARY: S\nCONFIDENCE: high\nOPEN_QUESTIONS: - First\n- Second\n"
        _, _, oq = _parse_summary_response(response)
        assert oq is not None
        assert "First" in oq
        assert "Second" in oq

    def test_unknown_confidence_returns_none(self) -> None:
        response = "SUMMARY: S\nCONFIDENCE: uncertain\nOPEN_QUESTIONS: none\n"
        _, conf, _ = _parse_summary_response(response)
        assert conf is None

    def test_fallback_on_bad_format(self) -> None:
        response = "No labels here, just raw text."
        summary, conf, _ = _parse_summary_response(response)
        assert "raw text" in summary
        assert conf is None


# ---------------------------------------------------------------------------
# Workflow — inquiry record written first
# ---------------------------------------------------------------------------


class TestInquiryRecordWrittenFirst:
    def test_record_written_before_any_finding(self) -> None:
        """Inquiry record must be written before any finding."""
        call_log: list[str] = []

        def write_record(r: InquiryRecord) -> None:
            call_log.append("record")

        def write_finding(f: InquiryFinding) -> None:
            call_log.append("finding")

        run_inquiry(
            write_inquiry_record_fn=write_record,
            write_finding_fn=write_finding,
            max_iterations=1,
        )

        assert call_log[0] == "record", "Record must be written before any finding"
        assert "finding" in call_log

    def test_record_id_matches_inquiry_id(self) -> None:
        records: list[InquiryRecord] = []

        def write_record(r: InquiryRecord) -> None:
            records.append(r)

        run_inquiry(write_inquiry_record_fn=write_record, max_iterations=1)
        assert len(records) == 1
        assert records[0].id == _INQUIRY_ID

    def test_record_question_preserved(self) -> None:
        records: list[InquiryRecord] = []
        run_inquiry(
            write_inquiry_record_fn=lambda r: records.append(r), max_iterations=1
        )
        assert records[0].question == _QUESTION

    def test_record_scope_preserved(self) -> None:
        records: list[InquiryRecord] = []
        run_inquiry(
            write_inquiry_record_fn=lambda r: records.append(r), max_iterations=1
        )
        assert records[0].scope == _SCOPE

    def test_record_status_active_at_write_time(self) -> None:
        records: list[InquiryRecord] = []
        run_inquiry(
            write_inquiry_record_fn=lambda r: records.append(r), max_iterations=1
        )
        assert records[0].status == InquiryStatus.ACTIVE

    def test_record_written_exactly_once(self) -> None:
        records: list[InquiryRecord] = []
        run_inquiry(
            write_inquiry_record_fn=lambda r: records.append(r), max_iterations=3
        )
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Workflow — query sanitization
# ---------------------------------------------------------------------------


class TestQuerySanitization:
    def test_clean_queries_accepted(self) -> None:
        """Workflow completes when inference returns clean queries."""
        result = run_inquiry(max_iterations=2)
        assert result.status == InquiryStatus.COMPLETED

    def test_pkm_query_raises(self) -> None:
        """Workflow raises ValueError if a generated query contains PKM markers."""

        def bad_inference(prompt: str) -> str:
            if "Generate the next search query" in prompt:
                return "search user_pkm for epistemology notes"
            return _query_inference(prompt)

        with pytest.raises(ValueError, match="user_pkm"):
            run_inquiry(inference_fn=bad_inference, max_iterations=1)

    def test_pkm_error_before_finding_written(self) -> None:
        """No finding is written if the query fails sanitization."""
        findings: list[InquiryFinding] = []

        def bad_inference(prompt: str) -> str:
            if "Generate the next search query" in prompt:
                return "my note on epistemology"
            return _query_inference(prompt)

        with pytest.raises(ValueError):
            run_inquiry(
                inference_fn=bad_inference,
                write_finding_fn=lambda f: findings.append(f),
                max_iterations=1,
            )

        assert findings == []


# ---------------------------------------------------------------------------
# Workflow — findings written progressively and provisionally
# ---------------------------------------------------------------------------


class TestFindingsProgressive:
    def test_findings_count_equals_max_iterations(self) -> None:
        findings: list[InquiryFinding] = []
        run_inquiry(
            write_finding_fn=lambda f: findings.append(f),
            max_iterations=3,
        )
        assert len(findings) == 3

    def test_all_findings_have_provisional_epistemic_status(self) -> None:
        findings: list[InquiryFinding] = []
        run_inquiry(
            write_finding_fn=lambda f: findings.append(f),
            max_iterations=3,
        )
        assert all(f.epistemic_status == "provisional" for f in findings)

    def test_findings_carry_inquiry_id(self) -> None:
        findings: list[InquiryFinding] = []
        run_inquiry(
            write_finding_fn=lambda f: findings.append(f),
            max_iterations=2,
        )
        assert all(f.inquiry_id == _INQUIRY_ID for f in findings)

    def test_finding_iterations_sequential(self) -> None:
        findings: list[InquiryFinding] = []
        run_inquiry(
            write_finding_fn=lambda f: findings.append(f),
            max_iterations=3,
        )
        assert [f.iteration for f in findings] == [0, 1, 2]

    def test_each_finding_has_query_used(self) -> None:
        findings: list[InquiryFinding] = []
        run_inquiry(
            write_finding_fn=lambda f: findings.append(f),
            max_iterations=2,
        )
        assert all(len(f.query_used) > 0 for f in findings)

    def test_each_finding_has_written_at(self) -> None:
        findings: list[InquiryFinding] = []
        run_inquiry(
            write_finding_fn=lambda f: findings.append(f),
            max_iterations=1,
        )
        assert findings[0].written_at != ""

    def test_each_finding_has_unique_id(self) -> None:
        findings: list[InquiryFinding] = []
        run_inquiry(
            write_finding_fn=lambda f: findings.append(f),
            max_iterations=4,
        )
        ids = [f.id for f in findings]
        assert len(ids) == len(set(ids))

    def test_finding_written_before_next_iteration_query(self) -> None:
        """Each finding is persisted before the next query is generated."""
        call_log: list[str] = []
        query_calls: list[int] = [0]

        def tracking_inference(prompt: str) -> str:
            if "Generate the next search query" in prompt:
                call_log.append(f"query-{query_calls[0]}")
                query_calls[0] += 1
                return "some clean query"
            if "Synthesize the key finding" in prompt:
                return (
                    "FINDING: Found something.\nCONFIDENCE: medium\n"
                    "OPEN_QUESTIONS: none\n"
                )
            return _query_inference(prompt)

        def tracking_write(f: InquiryFinding) -> None:
            call_log.append(f"finding-{f.iteration}")

        run_inquiry(
            inference_fn=tracking_inference,
            write_finding_fn=tracking_write,
            max_iterations=2,
        )

        # finding-0 must appear before query-1
        assert call_log.index("finding-0") < call_log.index("query-1")


# ---------------------------------------------------------------------------
# Workflow — completion
# ---------------------------------------------------------------------------


class TestCompletion:
    def test_result_status_completed(self) -> None:
        result = run_inquiry(max_iterations=2)
        assert result.status == InquiryStatus.COMPLETED

    def test_result_inquiry_id(self) -> None:
        result = run_inquiry(max_iterations=2)
        assert result.inquiry_id == _INQUIRY_ID

    def test_result_findings_count(self) -> None:
        result = run_inquiry(max_iterations=3)
        assert result.findings_count == 3

    def test_result_summary_not_none(self) -> None:
        result = run_inquiry(max_iterations=1)
        assert result.summary is not None
        assert len(result.summary) > 0

    def test_mark_complete_called_once(self) -> None:
        complete_calls: list[tuple[str, str, float | None, int, str | None]] = []

        def on_complete(
            iid: str, summary: str, conf: float | None, count: int, oq: str | None
        ) -> None:
            complete_calls.append((iid, summary, conf, count, oq))

        run_inquiry(mark_inquiry_complete_fn=on_complete, max_iterations=2)
        assert len(complete_calls) == 1

    def test_mark_complete_receives_correct_inquiry_id(self) -> None:
        calls: list[str] = []

        def on_complete_id(
            iid: str, summary: str, conf: float | None, count: int, oq: str | None
        ) -> None:
            calls.append(iid)

        run_inquiry(mark_inquiry_complete_fn=on_complete_id, max_iterations=1)
        assert calls[0] == _INQUIRY_ID

    def test_mark_complete_findings_count_matches(self) -> None:
        counts: list[int] = []

        def on_complete(
            iid: str, summary: str, conf: float | None, count: int, oq: str | None
        ) -> None:
            counts.append(count)

        run_inquiry(mark_inquiry_complete_fn=on_complete, max_iterations=3)
        assert counts[0] == 3


# ---------------------------------------------------------------------------
# Workflow — contradiction detection triggered
# ---------------------------------------------------------------------------


class TestContradictionDetection:
    def test_triggered_after_completion(self) -> None:
        triggered: list[str] = []
        run_inquiry(
            trigger_contradiction_detection_fn=lambda iid: triggered.append(iid),
            max_iterations=1,
        )
        assert triggered == [_INQUIRY_ID]

    def test_not_triggered_if_inference_raises(self) -> None:
        triggered: list[str] = []
        call_n: list[int] = [0]

        def failing_inference(prompt: str) -> str:
            if "Generate the next search query" in prompt:
                return "clean query"
            if "Synthesize the key finding" in prompt:
                call_n[0] += 1
                if call_n[0] == 1:
                    raise RuntimeError("inference failed")
            return _query_inference(prompt)

        with pytest.raises(RuntimeError):
            run_inquiry(
                inference_fn=failing_inference,
                trigger_contradiction_detection_fn=lambda iid: triggered.append(iid),
                max_iterations=1,
            )
        assert triggered == []

    def test_result_contradiction_detection_triggered_true(self) -> None:
        result = run_inquiry(max_iterations=1)
        assert result.contradiction_detection_triggered is True


# ---------------------------------------------------------------------------
# Workflow — push triggered
# ---------------------------------------------------------------------------


class TestPushTriggered:
    def test_push_triggered_after_completion(self) -> None:
        pushes: list[tuple[str, bool]] = []
        run_inquiry(
            trigger_push_fn=lambda msg, in_s: pushes.append((msg, in_s)),
            max_iterations=1,
        )
        assert len(pushes) == 1

    def test_push_in_session_false_by_default(self) -> None:
        pushes: list[tuple[str, bool]] = []
        run_inquiry(
            trigger_push_fn=lambda msg, in_s: pushes.append((msg, in_s)),
            max_iterations=1,
            in_session=False,
        )
        assert pushes[0][1] is False

    def test_push_in_session_true(self) -> None:
        pushes: list[tuple[str, bool]] = []
        run_inquiry(
            trigger_push_fn=lambda msg, in_s: pushes.append((msg, in_s)),
            max_iterations=1,
            in_session=True,
        )
        assert pushes[0][1] is True

    def test_push_message_is_non_empty(self) -> None:
        pushes: list[str] = []
        run_inquiry(
            trigger_push_fn=lambda msg, _: pushes.append(msg),
            max_iterations=1,
        )
        assert len(pushes[0]) > 0

    def test_result_push_triggered_true(self) -> None:
        result = run_inquiry(max_iterations=1)
        assert result.push_triggered is True

    def test_push_not_raw_dump_of_all_findings(self) -> None:
        """Push message is generated by inference, not a mechanical concatenation."""
        pushes: list[str] = []
        findings: list[InquiryFinding] = []

        def tracking_inference(prompt: str) -> str:
            if "brief message to surface" in prompt or "Write a brief" in prompt:
                return "I looked into epistemology. Here is the key finding."
            return _query_inference(prompt)

        run_inquiry(
            inference_fn=tracking_inference,
            write_finding_fn=lambda f: findings.append(f),
            trigger_push_fn=lambda msg, _: pushes.append(msg),
            max_iterations=2,
        )
        # The push message should be the inference result, not a raw concatenation
        assert len(pushes) == 1
        # The push came from inference, not a raw list of findings
        assert pushes[0] != "\n".join(f.content for f in findings)


# ---------------------------------------------------------------------------
# Workflow — order of operations
# ---------------------------------------------------------------------------


class TestOperationOrder:
    def test_contradiction_detection_before_push(self) -> None:
        """Contradiction detection must fire before the push (§3E)."""
        log: list[str] = []

        def on_push(msg: str, in_session: bool) -> None:
            log.append("push")

        run_inquiry(
            trigger_contradiction_detection_fn=lambda _: log.append("contradiction"),
            trigger_push_fn=on_push,
            max_iterations=1,
        )
        assert log.index("contradiction") < log.index("push")

    def test_mark_complete_before_contradiction_detection(self) -> None:
        log: list[str] = []

        def on_complete_log(
            iid: str, summary: str, conf: float | None, count: int, oq: str | None
        ) -> None:
            log.append("complete")

        run_inquiry(
            mark_inquiry_complete_fn=on_complete_log,
            trigger_contradiction_detection_fn=lambda _: log.append("contradiction"),
            max_iterations=1,
        )
        assert log.index("complete") < log.index("contradiction")

    def test_all_findings_written_before_summary(self) -> None:
        log: list[str] = []
        finding_n: list[int] = [0]

        def tracking_inference(prompt: str) -> str:
            if "Generate the next search query" in prompt:
                return "clean query"
            if "Synthesize the key finding" in prompt:
                log.append(f"finding-write-{finding_n[0]}")
                finding_n[0] += 1
                return "FINDING: Found.\nCONFIDENCE: medium\nOPEN_QUESTIONS: none\n"
            if "Synthesize a final answer" in prompt:
                log.append("summary")
                return "SUMMARY: S.\nCONFIDENCE: medium\nOPEN_QUESTIONS: none\n"
            return "Push message."

        run_inquiry(inference_fn=tracking_inference, max_iterations=2)
        # Both findings must appear before the summary
        assert log.index("finding-write-0") < log.index("summary")
        assert log.index("finding-write-1") < log.index("summary")


# ---------------------------------------------------------------------------
# Workflow — preemption (suspend mode)
# ---------------------------------------------------------------------------


class TestPreemptionSuspend:
    def test_cooperate_called_between_iterations(self) -> None:
        """cooperate() is called once per iteration."""
        cooperate_calls: list[int] = []
        ctx = PreemptionContext(PreemptionMode.SUSPEND)

        original_cooperate = ctx.cooperate

        def tracking_cooperate(
            checkpoint_fn: Callable[[], None],
            rollback_fn: Callable[[], None],
        ) -> None:
            cooperate_calls.append(1)
            original_cooperate(checkpoint_fn=checkpoint_fn, rollback_fn=rollback_fn)

        ctx.cooperate = tracking_cooperate  # type: ignore[method-assign]

        run_inquiry(preemption_ctx=ctx, max_iterations=3)
        assert len(cooperate_calls) == 3

    def test_no_preemption_ctx_completes_normally(self) -> None:
        """Without a preemption_ctx, the workflow completes without error."""
        result = run_inquiry(preemption_ctx=None, max_iterations=2)
        assert result.status == InquiryStatus.COMPLETED

    def test_suspend_pauses_and_resumes(self) -> None:
        """Workflow suspends at cooperate() and resumes after engine calls resume()."""
        ctx = PreemptionContext(PreemptionMode.SUSPEND)
        checkpoint_called: list[bool] = []
        rollback_called: list[bool] = []
        result_holder: list[CommissionedInquiryResult] = []

        def run() -> None:
            res = run_inquiry(
                preemption_ctx=ctx,
                checkpoint_fn=lambda: checkpoint_called.append(True),
                rollback_fn=lambda: rollback_called.append(True),
                max_iterations=1,
            )
            result_holder.append(res)

        ctx.preempt()
        t = threading.Thread(target=run, daemon=True)
        t.start()

        assert ctx.wait_for_checkpoint(timeout=_THREAD_TIMEOUT)
        assert checkpoint_called

        ctx.resume()
        t.join(timeout=_THREAD_TIMEOUT)
        assert not t.is_alive()
        assert rollback_called
        assert result_holder[0].status == InquiryStatus.COMPLETED

    def test_checkpoint_called_with_checkpoint_fn(self) -> None:
        ctx = PreemptionContext(PreemptionMode.SUSPEND)
        ckpt: list[bool] = []

        def run() -> None:
            run_inquiry(
                preemption_ctx=ctx,
                checkpoint_fn=lambda: ckpt.append(True),
                max_iterations=1,
            )

        ctx.preempt()
        t = threading.Thread(target=run, daemon=True)
        t.start()
        assert ctx.wait_for_checkpoint(timeout=_THREAD_TIMEOUT)
        ctx.resume()
        t.join(timeout=_THREAD_TIMEOUT)
        assert ckpt == [True]

    def test_rollback_called_on_resume(self) -> None:
        ctx = PreemptionContext(PreemptionMode.SUSPEND)
        rb: list[bool] = []

        def run() -> None:
            run_inquiry(
                preemption_ctx=ctx,
                rollback_fn=lambda: rb.append(True),
                max_iterations=1,
            )

        ctx.preempt()
        t = threading.Thread(target=run, daemon=True)
        t.start()
        ctx.wait_for_checkpoint(timeout=_THREAD_TIMEOUT)
        ctx.resume()
        t.join(timeout=_THREAD_TIMEOUT)
        assert rb == [True]


# ---------------------------------------------------------------------------
# Workflow — retrieve_daemon_context_fn called with sanitized query
# ---------------------------------------------------------------------------


class TestRetrieveDaemonContext:
    def test_retrieve_called_each_iteration(self) -> None:
        queries_retrieved: list[str] = []
        run_inquiry(
            retrieve_daemon_context_fn=lambda q: queries_retrieved.append(q) or "",
            max_iterations=3,
        )
        assert len(queries_retrieved) == 3

    def test_retrieve_receives_generated_query(self) -> None:
        queries: list[str] = []
        inference_queries: list[str] = []

        def tracking_inference(prompt: str) -> str:
            result = _query_inference(prompt)
            if "Generate the next search query" in prompt:
                inference_queries.append(result)
            return result

        run_inquiry(
            inference_fn=tracking_inference,
            retrieve_daemon_context_fn=lambda q: queries.append(q) or "",
            max_iterations=2,
        )
        # The query sent to retrieve must be the same one inference generated
        assert queries == inference_queries


# ---------------------------------------------------------------------------
# Workflow — scope=None handled gracefully
# ---------------------------------------------------------------------------


class TestNoneScope:
    def test_scope_none_completes(self) -> None:
        result = run_inquiry(scope=None, max_iterations=1)
        assert result.status == InquiryStatus.COMPLETED

    def test_scope_none_record_written(self) -> None:
        records: list[InquiryRecord] = []
        run_inquiry(
            scope=None,
            write_inquiry_record_fn=lambda r: records.append(r),
            max_iterations=1,
        )
        assert records[0].scope is None


# ---------------------------------------------------------------------------
# Workflow — max_iterations=1
# ---------------------------------------------------------------------------


class TestSingleIteration:
    def test_one_finding_written(self) -> None:
        findings: list[InquiryFinding] = []
        run_inquiry(
            write_finding_fn=lambda f: findings.append(f),
            max_iterations=1,
        )
        assert len(findings) == 1

    def test_result_findings_count_one(self) -> None:
        result = run_inquiry(max_iterations=1)
        assert result.findings_count == 1


# ---------------------------------------------------------------------------
# Workflow — exception propagation
# ---------------------------------------------------------------------------


class TestExceptionPropagation:
    def test_retrieve_raises_propagates(self) -> None:
        def bad_retrieve(q: str) -> str:
            raise RuntimeError("retrieval failed")

        with pytest.raises(RuntimeError, match="retrieval failed"):
            run_inquiry(retrieve_daemon_context_fn=bad_retrieve, max_iterations=1)

    def test_write_finding_raises_propagates(self) -> None:
        def bad_write(f: InquiryFinding) -> None:
            raise OSError("disk full")

        with pytest.raises(IOError, match="disk full"):
            run_inquiry(write_finding_fn=bad_write, max_iterations=1)

    def test_mark_complete_raises_propagates(self) -> None:
        def bad_complete(
            iid: str, s: str, c: float | None, n: int, oq: str | None
        ) -> None:
            raise RuntimeError("server unavailable")

        with pytest.raises(RuntimeError, match="server unavailable"):
            run_inquiry(mark_inquiry_complete_fn=bad_complete, max_iterations=1)

    def test_partial_findings_written_before_exception(self) -> None:
        """Findings written before the exception point are preserved."""
        findings: list[InquiryFinding] = []
        call_n: list[int] = [0]

        def partial_inference(prompt: str) -> str:
            if "Generate the next search query" in prompt:
                return "clean query"
            if "Synthesize the key finding" in prompt:
                call_n[0] += 1
                if call_n[0] >= 2:
                    raise RuntimeError("inference died")
                return "FINDING: Found.\nCONFIDENCE: medium\nOPEN_QUESTIONS: none\n"
            return _query_inference(prompt)

        with pytest.raises(RuntimeError):
            run_inquiry(
                inference_fn=partial_inference,
                write_finding_fn=lambda f: findings.append(f),
                max_iterations=3,
            )

        # The first finding was written before the error
        assert len(findings) == 1
