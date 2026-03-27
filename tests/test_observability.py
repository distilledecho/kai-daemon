"""Tests for the observability hooks (§13)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kai_daemon.state.observability import (
    RegisterCorrectionEntry,
    RegisterInferenceLogger,
    WorkflowRunEntry,
    WorkflowRunLogger,
    WorkflowStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def run_log(tmp_path: Path) -> WorkflowRunLogger:
    return WorkflowRunLogger(log_path=tmp_path / "workflow_runs.jsonl")


@pytest.fixture
def reg_log(tmp_path: Path) -> RegisterInferenceLogger:
    return RegisterInferenceLogger(log_path=tmp_path / "register_inference.jsonl")


def _run_entry(
    workflow_name: str = "episodic_flush",
    trigger: str = "session_end",
    started_at: str = "2026-01-01T00:00:00+00:00",
    completed_at: str = "2026-01-01T00:00:05+00:00",
    status: WorkflowStatus = WorkflowStatus.SUCCESS,
    memory_server_available: bool = True,
    metadata: dict[str, Any] | None = None,
) -> WorkflowRunEntry:
    return WorkflowRunEntry(
        workflow_name=workflow_name,
        trigger=trigger,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        memory_server_available=memory_server_available,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# WorkflowRunEntry — model validation
# ---------------------------------------------------------------------------


class TestWorkflowRunEntry:
    def test_minimal_valid(self):
        e = _run_entry()
        assert e.workflow_name == "episodic_flush"
        assert e.trigger == "session_end"
        assert e.status == WorkflowStatus.SUCCESS
        assert e.memory_server_available is True
        assert e.metadata == {}

    def test_with_metadata(self):
        e = _run_entry(metadata={"items_flushed": 5, "threads_updated": 2})
        assert e.metadata["items_flushed"] == 5

    def test_status_failure(self):
        e = _run_entry(status=WorkflowStatus.FAILURE)
        assert e.status == WorkflowStatus.FAILURE

    def test_status_abandoned(self):
        e = _run_entry(status=WorkflowStatus.ABANDONED)
        assert e.status == WorkflowStatus.ABANDONED

    def test_memory_server_unavailable(self):
        e = _run_entry(memory_server_available=False)
        assert e.memory_server_available is False

    def test_extra_fields_forbidden(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WorkflowRunEntry(  # type: ignore[call-arg]
                workflow_name="x",
                trigger="y",
                started_at="t",
                completed_at="t",
                status=WorkflowStatus.SUCCESS,
                memory_server_available=True,
                unknown_field="oops",  # pyright: ignore[reportCallIssue]
            )

    def test_roundtrip_json(self):
        e = _run_entry(metadata={"k": "v"})
        serialised = e.model_dump_json()
        restored = WorkflowRunEntry.model_validate_json(serialised)
        assert restored == e


# ---------------------------------------------------------------------------
# RegisterCorrectionEntry — model validation
# ---------------------------------------------------------------------------


class TestRegisterCorrectionEntry:
    def test_minimal_valid(self):
        e = RegisterCorrectionEntry(
            inferred_register="reflective",
            corrected_register="casual",
        )
        assert e.inferred_register == "reflective"
        assert e.corrected_register == "casual"
        assert e.thread_id is None
        assert e.metadata == {}
        # corrected_at auto-populated
        assert e.corrected_at != ""

    def test_with_thread_id(self):
        e = RegisterCorrectionEntry(
            thread_id="thread-abc",
            inferred_register="exploratory",
            corrected_register="casual",
        )
        assert e.thread_id == "thread-abc"

    def test_with_metadata(self):
        e = RegisterCorrectionEntry(
            inferred_register="reflective",
            corrected_register="casual",
            metadata={"session_turn": 3},
        )
        assert e.metadata["session_turn"] == 3

    def test_roundtrip_json(self):
        e = RegisterCorrectionEntry(
            thread_id="t1",
            inferred_register="reflective",
            corrected_register="casual",
        )
        serialised = e.model_dump_json()
        restored = RegisterCorrectionEntry.model_validate_json(serialised)
        assert restored == e


# ---------------------------------------------------------------------------
# WorkflowRunLogger — file behaviour
# ---------------------------------------------------------------------------


class TestWorkflowRunLogger:
    def test_append_creates_file(self, run_log: WorkflowRunLogger, tmp_path: Path):
        path = tmp_path / "workflow_runs.jsonl"
        assert not path.exists()
        run_log.append(_run_entry())
        assert path.exists()

    def test_append_writes_valid_json_line(
        self, run_log: WorkflowRunLogger, tmp_path: Path
    ):
        run_log.append(_run_entry())
        line = (tmp_path / "workflow_runs.jsonl").read_text().strip()
        data = json.loads(line)
        assert data["workflow_name"] == "episodic_flush"
        assert data["status"] == "success"
        assert data["memory_server_available"] is True

    def test_multiple_appends_produce_multiple_lines(
        self, run_log: WorkflowRunLogger, tmp_path: Path
    ):
        run_log.append(_run_entry(workflow_name="a"))
        run_log.append(_run_entry(workflow_name="b"))
        run_log.append(_run_entry(workflow_name="c"))
        lines = (tmp_path / "workflow_runs.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

    def test_each_line_is_independent_json(
        self, run_log: WorkflowRunLogger, tmp_path: Path
    ):
        run_log.append(_run_entry(workflow_name="alpha"))
        run_log.append(_run_entry(workflow_name="beta"))
        lines = (tmp_path / "workflow_runs.jsonl").read_text().strip().splitlines()
        names = [json.loads(line)["workflow_name"] for line in lines]
        assert names == ["alpha", "beta"]

    def test_read_all_empty_when_no_file(self, run_log: WorkflowRunLogger):
        assert run_log.read_all() == []

    def test_read_all_returns_entries(self, run_log: WorkflowRunLogger):
        run_log.append(_run_entry(workflow_name="x"))
        run_log.append(_run_entry(workflow_name="y"))
        entries = run_log.read_all()
        assert len(entries) == 2
        assert entries[0].workflow_name == "x"
        assert entries[1].workflow_name == "y"

    def test_read_all_preserves_memory_server_available(
        self, run_log: WorkflowRunLogger
    ):
        run_log.append(_run_entry(memory_server_available=False))
        entries = run_log.read_all()
        assert entries[0].memory_server_available is False

    def test_append_is_idempotent_across_instances(self, tmp_path: Path):
        """Two logger instances pointing at the same file both append."""
        path = tmp_path / "runs.jsonl"
        WorkflowRunLogger(log_path=path).append(_run_entry(workflow_name="first"))
        WorkflowRunLogger(log_path=path).append(_run_entry(workflow_name="second"))
        entries = WorkflowRunLogger(log_path=path).read_all()
        assert len(entries) == 2
        assert entries[0].workflow_name == "first"
        assert entries[1].workflow_name == "second"

    def test_append_silent_on_unwritable_path(self, tmp_path: Path):
        """I/O errors warn but do not raise."""
        bad_path = tmp_path / "nonexistent_dir" / "runs.jsonl"
        log = WorkflowRunLogger(log_path=bad_path)
        # Should not raise
        log.append(_run_entry())

    def test_metadata_preserved_through_roundtrip(self, run_log: WorkflowRunLogger):
        run_log.append(_run_entry(metadata={"checkpoint": "step_3", "items": 7}))
        entry = run_log.read_all()[0]
        assert entry.metadata["checkpoint"] == "step_3"
        assert entry.metadata["items"] == 7

    def test_all_statuses_roundtrip(self, run_log: WorkflowRunLogger):
        for status in WorkflowStatus:
            run_log.append(_run_entry(status=status))
        entries = run_log.read_all()
        statuses = {e.status for e in entries}
        assert statuses == set(WorkflowStatus)

    def test_read_all_skips_malformed_lines(
        self, run_log: WorkflowRunLogger, tmp_path: Path
    ):
        """A corrupted line is skipped; valid entries before and after are returned."""
        path = tmp_path / "workflow_runs.jsonl"
        run_log.append(_run_entry(workflow_name="before"))
        with path.open("a") as fh:
            fh.write("not valid json\n")
        run_log.append(_run_entry(workflow_name="after"))
        entries = run_log.read_all()
        assert len(entries) == 2
        assert entries[0].workflow_name == "before"
        assert entries[1].workflow_name == "after"


# ---------------------------------------------------------------------------
# RegisterInferenceLogger — file behaviour
# ---------------------------------------------------------------------------


class TestRegisterInferenceLogger:
    def test_append_creates_file(
        self, reg_log: RegisterInferenceLogger, tmp_path: Path
    ):
        path = tmp_path / "register_inference.jsonl"
        assert not path.exists()
        reg_log.append(
            RegisterCorrectionEntry(
                inferred_register="reflective", corrected_register="casual"
            )
        )
        assert path.exists()

    def test_append_writes_valid_json_line(
        self, reg_log: RegisterInferenceLogger, tmp_path: Path
    ):
        reg_log.append(
            RegisterCorrectionEntry(
                inferred_register="reflective", corrected_register="casual"
            )
        )
        line = (tmp_path / "register_inference.jsonl").read_text().strip()
        data = json.loads(line)
        assert data["inferred_register"] == "reflective"
        assert data["corrected_register"] == "casual"

    def test_multiple_corrections_append(
        self, reg_log: RegisterInferenceLogger, tmp_path: Path
    ):
        for i in range(4):
            reg_log.append(
                RegisterCorrectionEntry(
                    inferred_register=f"r{i}", corrected_register="casual"
                )
            )
        lines = (tmp_path / "register_inference.jsonl").read_text().strip().splitlines()
        assert len(lines) == 4

    def test_read_all_empty_when_no_file(self, reg_log: RegisterInferenceLogger):
        assert reg_log.read_all() == []

    def test_read_all_returns_entries(self, reg_log: RegisterInferenceLogger):
        reg_log.append(
            RegisterCorrectionEntry(
                inferred_register="exploratory", corrected_register="casual"
            )
        )
        entries = reg_log.read_all()
        assert len(entries) == 1
        assert entries[0].inferred_register == "exploratory"

    def test_thread_id_preserved(self, reg_log: RegisterInferenceLogger):
        reg_log.append(
            RegisterCorrectionEntry(
                thread_id="thread-xyz",
                inferred_register="reflective",
                corrected_register="casual",
            )
        )
        assert reg_log.read_all()[0].thread_id == "thread-xyz"

    def test_null_thread_id_preserved(self, reg_log: RegisterInferenceLogger):
        reg_log.append(
            RegisterCorrectionEntry(
                inferred_register="reflective", corrected_register="casual"
            )
        )
        assert reg_log.read_all()[0].thread_id is None

    def test_corrected_at_auto_populated(self, reg_log: RegisterInferenceLogger):
        reg_log.append(
            RegisterCorrectionEntry(
                inferred_register="reflective", corrected_register="casual"
            )
        )
        entry = reg_log.read_all()[0]
        assert entry.corrected_at != ""

    def test_append_silent_on_unwritable_path(self, tmp_path: Path):
        bad_path = tmp_path / "nonexistent_dir" / "register_inference.jsonl"
        log = RegisterInferenceLogger(log_path=bad_path)
        log.append(
            RegisterCorrectionEntry(
                inferred_register="reflective", corrected_register="casual"
            )
        )

    def test_metadata_preserved(self, reg_log: RegisterInferenceLogger):
        reg_log.append(
            RegisterCorrectionEntry(
                inferred_register="reflective",
                corrected_register="casual",
                metadata={"session_turn": 5},
            )
        )
        assert reg_log.read_all()[0].metadata["session_turn"] == 5

    def test_read_all_skips_malformed_lines(
        self, reg_log: RegisterInferenceLogger, tmp_path: Path
    ):
        """A corrupted line is skipped; valid entries before and after are returned."""
        path = tmp_path / "register_inference.jsonl"
        reg_log.append(
            RegisterCorrectionEntry(
                inferred_register="reflective", corrected_register="casual"
            )
        )
        with path.open("a") as fh:
            fh.write("not valid json\n")
        reg_log.append(
            RegisterCorrectionEntry(
                inferred_register="exploratory", corrected_register="casual"
            )
        )
        entries = reg_log.read_all()
        assert len(entries) == 2
        assert entries[0].inferred_register == "reflective"
        assert entries[1].inferred_register == "exploratory"
