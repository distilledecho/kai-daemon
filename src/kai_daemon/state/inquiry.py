"""Inquiry record and finding types for commissioned_inquiry (§3E).

Written to daemon-memory-server; injectable callables in the workflow
keep this module free of any HTTP dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class InquiryStatus(StrEnum):
    """Lifecycle status of a commissioned inquiry."""

    ACTIVE = "active"
    """Research is in progress."""

    COMPLETED = "completed"
    """All iterations finished; summary written."""

    ABANDONED = "abandoned"
    """Inquiry was stopped before completion; partial findings preserved."""


@dataclass
class InquiryRecord:
    """Written to memory server *before* research begins (§3E).

    The ``id`` is a caller-supplied UUID; all subsequent findings carry
    ``inquiry_id`` equal to this value so contradiction_detection can scope
    its input to ``inquiry_id == completed_inquiry_id`` (§3F).
    """

    id: str
    initiated_at: str
    """ISO8601 UTC timestamp."""

    question: str
    """The research question as received from the user."""

    scope: str | None
    """Optional scope restriction supplied alongside the question."""

    status: InquiryStatus = InquiryStatus.ACTIVE
    completed_at: str | None = None
    summary: str | None = None
    findings_count: int = 0
    confidence_overall: float | None = None
    open_questions_remaining: str | None = None


@dataclass
class InquiryFinding:
    """A single finding produced during one research iteration (§3E).

    Written immediately as it is synthesized — before the next iteration —
    so partial work is never lost on abandonment.  All findings carry
    ``epistemic_status: provisional``; the engine may promote them after
    contradiction detection confirms consistency.
    """

    id: str
    inquiry_id: str
    iteration: int
    """Zero-based iteration index within the inquiry loop."""

    content: str
    """Prose finding synthesized from retrieved daemon-space context."""

    epistemic_status: str
    """Always ``'provisional'`` at write time."""

    query_used: str
    """The sanitized search query that produced the context for this finding."""

    written_at: str
    """ISO8601 UTC timestamp."""

    open_questions: str | None = None
    """Remaining questions after this iteration, or ``None``."""

    sources_cited: list[str] = field(default_factory=lambda: list[str]())
    """Optional list of source references returned by retrieve_daemon_context_fn."""

    embedding_id: str | None = None
