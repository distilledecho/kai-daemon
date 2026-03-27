"""Observability hooks — workflow run log and register inference log (§13).

These are writer functions that workflows call — not the workflow engine itself.

``WorkflowRunLogger.append()`` writes one JSON line to
``data/logs/workflow_runs.jsonl`` for every workflow execution.

``RegisterInferenceLogger.append()`` writes one JSON line to
``data/logs/register_inference.jsonl`` for every register correction.

Both logs are append-only. Failures warn but never raise — the same
resilience pattern used throughout the state layer.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ._paths import logs_dir
from ._utils import _utcnow

logger = logging.getLogger(__name__)

_M = TypeVar("_M", bound=BaseModel)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WorkflowStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    ABANDONED = "abandoned"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class WorkflowRunEntry(BaseModel):
    """Structured log entry written on every workflow execution."""

    model_config = ConfigDict(extra="forbid")

    workflow_name: str
    trigger: str
    started_at: str
    completed_at: str
    status: WorkflowStatus
    memory_server_available: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegisterCorrectionEntry(BaseModel):
    """Log entry written when the register inference correction pathway fires (§4G).

    Records the inferred register, the correction, and the thread context.
    Prior response is always preserved — this is checked by the caller, not
    enforced here.
    """

    model_config = ConfigDict(extra="forbid")

    corrected_at: str = Field(default_factory=_utcnow)
    thread_id: str | None = None
    inferred_register: str
    corrected_register: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loggers
# ---------------------------------------------------------------------------


def _append_jsonl(path: Path, entry: BaseModel) -> None:
    """Append one JSON line to *path*. Warns but never raises on I/O error."""
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
    except OSError:
        logger.warning("observability: failed to write to %s", path, exc_info=True)


def _read_jsonl(path: Path, model: type[_M]) -> list[_M]:
    """Read all entries from *path*. Skips malformed lines; returns [] if absent."""
    if not path.exists():
        return []
    entries: list[_M] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(model.model_validate_json(line))
                    except Exception:
                        logger.warning(
                            "observability: skipping malformed line in %s", path
                        )
    except OSError:
        logger.warning("observability: failed to read %s", path, exc_info=True)
    return entries


class WorkflowRunLogger:
    """Append-only writer for ``data/logs/workflow_runs.jsonl``.

    Args:
        log_path: Override the default log file path (for tests).
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._path = log_path or (logs_dir() / "workflow_runs.jsonl")

    def append(self, entry: WorkflowRunEntry) -> None:
        """Append *entry* to the workflow run log."""
        _append_jsonl(self._path, entry)

    def read_all(self) -> list[WorkflowRunEntry]:
        """Read all entries from the log file. Returns empty list if absent."""
        return _read_jsonl(self._path, WorkflowRunEntry)


class RegisterInferenceLogger:
    """Append-only writer for ``data/logs/register_inference.jsonl``.

    Args:
        log_path: Override the default log file path (for tests).
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._path = log_path or (logs_dir() / "register_inference.jsonl")

    def append(self, entry: RegisterCorrectionEntry) -> None:
        """Append *entry* to the register correction log."""
        _append_jsonl(self._path, entry)

    def read_all(self) -> list[RegisterCorrectionEntry]:
        """Read all entries from the log file. Returns empty list if absent."""
        return _read_jsonl(self._path, RegisterCorrectionEntry)
