"""commissioned_inquiry workflow (§3E).

Researches a question explicitly handed to the daemon by the user.

Execution steps
---------------
    0. Write InquiryRecord to memory server (before research begins)
    1–N. Iterative research loop (up to ``max_iterations``):
          a. Generate a sanitized search query (no user PKM content)
          b. Retrieve daemon-space context via injectable callable
          c. Synthesize a finding from the retrieved context
          d. Write finding immediately (``epistemic_status: provisional``)
          e. ``cooperate()`` — preemption point (suspend mode)
    N+1. Synthesize final summary from all findings
    N+2. Mark inquiry complete; write summary to memory server
    N+3. Trigger ``contradiction_detection`` scoped to this inquiry_id
    N+4. Surface results via push (in-session or ``push_message`` workflow)

Abandonment contract
    Every finding is written as soon as it is synthesized — before the next
    iteration begins.  If the workflow is abandoned at any point, all findings
    written so far are preserved with ``epistemic_status: provisional``.
    The engine is responsible for calling ``mark_inquiry_abandoned_fn`` if it
    decides the inquiry will never be resumed.

Privacy invariant
    External queries are sanitized — no user PKM content is included.
    ``retrieve_daemon_context_fn`` must only return daemon-space items; it
    must never access ``user_pkm`` or ``shared`` collections (§5, §CLAUDE.md).
    The sanitization prompt instructs the model to produce a self-contained
    query, and an automated assertion confirms no known PKM markers appear in
    the generated query string.

Priority: 2 / Preemption: **suspend** / Model: reflection
Trigger: workflow_request
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from ..state.inquiry import InquiryFinding, InquiryRecord, InquiryStatus
from .preemption import PreemptionContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callable type aliases (all memory-server writes are injectable)
# ---------------------------------------------------------------------------

WriteInquiryRecordFn = Callable[[InquiryRecord], None]
"""Write the InquiryRecord to daemon-memory-server before research begins."""

MarkInquiryCompleteFn = Callable[[str, str, float | None, int, str | None], None]
"""Mark an inquiry completed.

Signature: ``(inquiry_id, summary, confidence, findings_count,
               open_questions_remaining) → None``.
"""

MarkInquiryAbandonedFn = Callable[[str, int], None]
"""Mark an inquiry abandoned with the number of findings written so far.

Signature: ``(inquiry_id, findings_count) → None``.
"""

RetrieveDaemonContextFn = Callable[[str], str]
"""Retrieve daemon-space context relevant to *query*.

MUST only access the ``daemon`` knowledge space — never ``user_pkm`` or
``shared`` (§5, §CLAUDE.md: "Three knowledge spaces never collapse").

Returns a prose block of retrieved text; empty string if nothing relevant.
"""

WriteInquiryFindingFn = Callable[[InquiryFinding], None]
"""Write one InquiryFinding to daemon-memory-server."""

TriggerContradictionDetectionFn = Callable[[str], None]
"""Trigger contradiction_detection scoped to *inquiry_id*.

Signature: ``(inquiry_id) → None``.
The engine implementation enqueues contradiction_detection with
``new_items`` scoped to ``inquiry_id == inquiry_id`` (§3F).
"""

TriggerPushFn = Callable[[str, bool], None]
"""Surface the inquiry results to the user.

Signature: ``(message_content, in_session) → None``.
- ``in_session=True``  → mid-session streaming push (§8f).
- ``in_session=False`` → enqueue ``push_message`` workflow (§2, Priority 2).
"""

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Query generation — instructs the model to produce a self-contained query
# with no personal or user-private content.
_QUERY_PROMPT_TEMPLATE = """\
You are researching the following question on behalf of a user.
Your search query will be sent to an external knowledge base.

Research question: {question}
Scope: {scope}

Findings so far ({iteration} of {max_iterations}):
{prior_findings_text}

Generate the next search query to advance this research.

Rules:
- The query must be entirely self-contained.
- Do not include any personal information, user preferences, private
  details, names, or context from private conversations.
- Do not reference any prior conversation context or private knowledge.
- The query must be usable without any additional context.

Reply with exactly one search query on a single line. No preamble.
"""

# Finding synthesis — receives retrieved daemon-space context and prior findings.
_FINDING_PROMPT_TEMPLATE = """\
Research question: {question}

Search query used: {query}

Retrieved context:
{context}

Prior findings from this inquiry:
{prior_findings_text}

Synthesize the key finding from the retrieved context above.
Focus on what is new or confirmatory relative to prior findings.
Be precise. Note what remains uncertain.

Reply with exactly these three labeled lines:
FINDING: <your finding — one to three sentences>
CONFIDENCE: <high|medium|low>
OPEN_QUESTIONS: <remaining questions, or "none">
"""

# Final summary synthesis — consolidates all iteration findings.
_SUMMARY_PROMPT_TEMPLATE = """\
Research question: {question}
Scope: {scope}

All findings from this inquiry ({finding_count} iteration(s)):
{all_findings_text}

Synthesize a final answer to the research question.
Be precise. Note what remains uncertain.
Identify the most important open questions that remain.

Reply with exactly these three labeled lines:
SUMMARY: <your synthesis — two to five sentences>
CONFIDENCE: <high|medium|low>
OPEN_QUESTIONS: <remaining questions, one per line starting with "- ", or "none">
"""

# Push message — natural surfacing of results, not a data dump.
_PUSH_PROMPT_TEMPLATE = """\
Research question: {question}
Summary: {summary}
Confidence: {confidence}
Open questions: {open_questions}

Write a brief message to surface these research findings to the user.
This should feel like a natural observation — not a report or a data dump.
Focus on what matters most, and name any important open questions.
Two to four sentences.
"""

# ---------------------------------------------------------------------------
# PKM content markers — used by the sanitization guard
# ---------------------------------------------------------------------------

# Patterns that should never appear in an externally-sent query.
# Canonical PKM collection identifiers live in daemon-memory-server.yaml
# and src/kai_daemon/state/_chroma.py (COLLECTION_USER_PKM et al.).
# Extend this list whenever a new user-private collection or content type
# is added to the schema.
_PKM_MARKERS: list[str] = [
    "user_pkm",
    "user pkm",
    "private note",
    "personal note",
    "journal entry",
    "my note",
]


def _assert_query_sanitized(query: str) -> None:
    """Raise ``ValueError`` if the query contains known PKM markers.

    This is the automated enforcement of the privacy invariant: external
    queries must never contain user PKM content (§3E, §CLAUDE.md).
    """
    lower = query.lower()
    for marker in _PKM_MARKERS:
        if marker in lower:
            raise ValueError(
                f"commissioned_inquiry: query contains PKM marker {marker!r}. "
                "External queries must be sanitized — no user PKM content."
            )


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def _parse_finding_response(response: str) -> tuple[str, str, str | None]:
    """Parse the labeled-line response from ``_FINDING_PROMPT_TEMPLATE``.

    Returns ``(finding, confidence, open_questions)``.
    Falls back to the raw response in ``finding`` on parse failure.
    """
    finding: str = response.strip()
    confidence: str = "low"
    open_questions: str | None = None

    for line in response.splitlines():
        if line.startswith("FINDING:"):
            finding = line[len("FINDING:") :].strip() or finding
        elif line.startswith("CONFIDENCE:"):
            raw = line[len("CONFIDENCE:") :].strip().lower()
            if raw in ("high", "medium", "low"):
                confidence = raw
        elif line.startswith("OPEN_QUESTIONS:"):
            raw = line[len("OPEN_QUESTIONS:") :].strip()
            open_questions = None if raw.lower() in ("none", "") else raw

    return finding, confidence, open_questions


def _parse_summary_response(response: str) -> tuple[str, float | None, str | None]:
    """Parse the labeled-line response from ``_SUMMARY_PROMPT_TEMPLATE``.

    Returns ``(summary, confidence_float, open_questions_text)``.
    ``confidence_float``: ``high`` → 0.9, ``medium`` → 0.6, ``low`` → 0.3.
    Falls back to raw response in ``summary`` on parse failure.
    """
    summary: str = response.strip()
    confidence_float: float | None = None
    open_questions: str | None = None

    _conf_map = {"high": 0.9, "medium": 0.6, "low": 0.3}

    # Multi-line OPEN_QUESTIONS: collect lines after the label until next label
    lines = response.splitlines()
    in_oq = False
    oq_lines: list[str] = []

    for line in lines:
        if line.startswith("SUMMARY:"):
            summary = line[len("SUMMARY:") :].strip() or summary
            in_oq = False
        elif line.startswith("CONFIDENCE:"):
            raw = line[len("CONFIDENCE:") :].strip().lower()
            confidence_float = _conf_map.get(raw)
            in_oq = False
        elif line.startswith("OPEN_QUESTIONS:"):
            raw = line[len("OPEN_QUESTIONS:") :].strip()
            if raw.lower() not in ("none", ""):
                oq_lines.append(raw)
            in_oq = True
        elif in_oq and line.strip():
            oq_lines.append(line.strip())

    if oq_lines:
        open_questions = "\n".join(oq_lines)

    return summary, confidence_float, open_questions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _noop() -> None:
    pass


def _format_prior_findings(findings: list[InquiryFinding]) -> str:
    if not findings:
        return "(none yet)"
    parts: list[str] = []
    for f in findings:
        oq = f" | open: {f.open_questions}" if f.open_questions else ""
        parts.append(f"[{f.iteration}] {f.content}{oq}")
    return "\n".join(parts)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class CommissionedInquiryResult:
    """Returned on completion (or abandonment) of the workflow."""

    inquiry_id: str
    status: InquiryStatus
    findings_count: int
    summary: str | None
    """``None`` when the inquiry was abandoned before the summary step."""

    push_triggered: bool
    contradiction_detection_triggered: bool


# ---------------------------------------------------------------------------
# Public workflow function
# ---------------------------------------------------------------------------


def commissioned_inquiry(
    inquiry_id: str,
    question: str,
    scope: str | None,
    *,
    inference_fn: Callable[[str], str],
    write_inquiry_record_fn: WriteInquiryRecordFn,
    mark_inquiry_complete_fn: MarkInquiryCompleteFn,
    retrieve_daemon_context_fn: RetrieveDaemonContextFn,
    write_finding_fn: WriteInquiryFindingFn,
    trigger_contradiction_detection_fn: TriggerContradictionDetectionFn,
    trigger_push_fn: TriggerPushFn,
    initiated_at: str | None = None,
    in_session: bool = False,
    max_iterations: int = 5,
    preemption_ctx: PreemptionContext | None = None,
    checkpoint_fn: Callable[[], None] | None = None,
    rollback_fn: Callable[[], None] | None = None,
) -> CommissionedInquiryResult:
    """Research a question explicitly commissioned by the user (§3E).

    Writes findings progressively so partial work is never lost on
    abandonment.  Checkpoints between iterations when ``preemption_ctx``
    is provided (suspend mode).

    Parameters
    ----------
    inquiry_id:
        Caller-supplied UUID for this inquiry.  All findings carry this ID.
    question:
        The research question as received from the user.
    scope:
        Optional scope restriction from the user.
    inference_fn:
        Sends a prompt to the reflection model; returns text.
    write_inquiry_record_fn:
        Writes the ``InquiryRecord`` to daemon-memory-server.
        Called **before** any research begins.
    mark_inquiry_complete_fn:
        Updates the inquiry record with status=completed, summary, confidence,
        findings_count, and open_questions_remaining.
        Signature: ``(inquiry_id, summary, confidence, findings_count,
                       open_questions_remaining) → None``.
    retrieve_daemon_context_fn:
        Returns a prose block of daemon-space context relevant to *query*.
        Must never access ``user_pkm`` or ``shared`` collections.
    write_finding_fn:
        Writes each ``InquiryFinding`` immediately after synthesis.
    trigger_contradiction_detection_fn:
        Enqueues ``contradiction_detection`` scoped to this ``inquiry_id``.
        Called after the inquiry is marked complete.
    trigger_push_fn:
        Surfaces results to the user.
        ``(message_content, in_session) → None``.
    initiated_at:
        ISO8601 timestamp; defaults to ``datetime.now(UTC).isoformat()``.
    in_session:
        ``True`` for mid-session push (§8f); ``False`` for ``push_message``
        workflow (Priority 2).
    max_iterations:
        Maximum research iterations before synthesizing the summary.
    preemption_ctx:
        Preemption context from the workflow engine.  ``None`` means no
        preemption handling (e.g. in tests).
    checkpoint_fn:
        Called on checkpoint if ``preemption_ctx`` fires.  Defaults to noop.
    rollback_fn:
        Called on resume if ``preemption_ctx`` fires.  Defaults to noop.

    Returns
    -------
    CommissionedInquiryResult
        Status and IDs for the completed (or abandoned) inquiry.

    Raises
    ------
    Exception
        Any exception from an injectable callable propagates out.  Partial
        findings written before the exception are preserved.
    ValueError
        If a generated query is found to contain PKM marker content.
    """
    started_at = initiated_at or _utcnow()

    logger.info(
        "commissioned_inquiry: starting inquiry=%s question=%r", inquiry_id, question
    )

    # ------------------------------------------------------------------
    # Step 0: Write inquiry record BEFORE research begins
    # ------------------------------------------------------------------
    record = InquiryRecord(
        id=inquiry_id,
        initiated_at=started_at,
        question=question,
        scope=scope,
        status=InquiryStatus.ACTIVE,
    )
    write_inquiry_record_fn(record)
    logger.debug("commissioned_inquiry: inquiry record written inquiry=%s", inquiry_id)

    # ------------------------------------------------------------------
    # Steps 1–N: Iterative research loop
    # ------------------------------------------------------------------
    findings: list[InquiryFinding] = []
    ckpt = checkpoint_fn if checkpoint_fn is not None else _noop
    rb = rollback_fn if rollback_fn is not None else _noop

    for iteration in range(max_iterations):
        prior_findings_text = _format_prior_findings(findings)

        # a. Generate sanitized search query
        query_prompt = _QUERY_PROMPT_TEMPLATE.format(
            question=question,
            scope=scope or "none",
            iteration=iteration,
            max_iterations=max_iterations,
            prior_findings_text=prior_findings_text,
        )
        query = inference_fn(query_prompt).strip()
        # Strip quotes the model may have added around the query
        query = re.sub(r'^["\']|["\']$', "", query).strip()

        # Privacy invariant: assert no PKM markers in externally-sent query
        _assert_query_sanitized(query)

        logger.debug(
            "commissioned_inquiry: iteration=%d query=%r inquiry=%s",
            iteration,
            query,
            inquiry_id,
        )

        # b. Retrieve daemon-space context (no user PKM)
        context = retrieve_daemon_context_fn(query)

        # c. Synthesize finding
        finding_prompt = _FINDING_PROMPT_TEMPLATE.format(
            question=question,
            query=query,
            context=context or "(no relevant context found)",
            prior_findings_text=prior_findings_text,
        )
        finding_response = inference_fn(finding_prompt)
        content, _iter_confidence, open_questions = _parse_finding_response(
            finding_response
        )

        # d. Write finding immediately (provisional)
        finding = InquiryFinding(
            id=str(uuid.uuid4()),
            inquiry_id=inquiry_id,
            iteration=iteration,
            content=content,
            epistemic_status="provisional",
            query_used=query,
            written_at=_utcnow(),
            open_questions=open_questions,
        )
        write_finding_fn(finding)
        findings.append(finding)
        logger.debug(
            "commissioned_inquiry: finding written iteration=%d inquiry=%s",
            iteration,
            inquiry_id,
        )

        # e. cooperate() — preemption point between iterations
        if preemption_ctx is not None:
            preemption_ctx.cooperate(checkpoint_fn=ckpt, rollback_fn=rb)

    # ------------------------------------------------------------------
    # Step N+1: Synthesize final summary
    # ------------------------------------------------------------------
    logger.debug(
        "commissioned_inquiry: synthesizing summary inquiry=%s findings=%d",
        inquiry_id,
        len(findings),
    )
    all_findings_text = _format_prior_findings(findings)
    summary_prompt = _SUMMARY_PROMPT_TEMPLATE.format(
        question=question,
        scope=scope or "none",
        finding_count=len(findings),
        all_findings_text=all_findings_text,
    )
    summary_response = inference_fn(summary_prompt)
    summary, confidence_float, open_questions_remaining = _parse_summary_response(
        summary_response
    )

    # ------------------------------------------------------------------
    # Step N+2: Mark inquiry complete
    # ------------------------------------------------------------------
    mark_inquiry_complete_fn(
        inquiry_id,
        summary,
        confidence_float,
        len(findings),
        open_questions_remaining,
    )
    logger.debug("commissioned_inquiry: marked complete inquiry=%s", inquiry_id)

    # ------------------------------------------------------------------
    # Step N+3: Trigger contradiction_detection scoped to this inquiry_id
    # ------------------------------------------------------------------
    trigger_contradiction_detection_fn(inquiry_id)
    logger.debug(
        "commissioned_inquiry: contradiction_detection triggered inquiry=%s", inquiry_id
    )

    # ------------------------------------------------------------------
    # Step N+4: Surface results via push (not a data dump)
    # ------------------------------------------------------------------
    push_prompt = _PUSH_PROMPT_TEMPLATE.format(
        question=question,
        summary=summary,
        confidence=confidence_float if confidence_float is not None else "unknown",
        open_questions=open_questions_remaining or "none",
    )
    push_message = inference_fn(push_prompt).strip()
    trigger_push_fn(push_message, in_session)
    logger.info(
        "commissioned_inquiry: complete inquiry=%s findings=%d in_session=%s",
        inquiry_id,
        len(findings),
        in_session,
    )

    return CommissionedInquiryResult(
        inquiry_id=inquiry_id,
        status=InquiryStatus.COMPLETED,
        findings_count=len(findings),
        summary=summary,
        push_triggered=True,
        contradiction_detection_triggered=True,
    )
