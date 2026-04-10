"""Tests for the Kai SDK — permission enforcement and tool call logging (Stage 3.5).

Verifies that:
- sdk_tool-decorated functions pass through normally when no context is set.
- ToolPermissionError is raised (and logged) when a tool is called outside its
  permitted set for the current workflow.
- Successful tool calls are logged with outcome="success".
- Erroring tool calls are logged with outcome="error".
- workflow_id is captured from the ContextVar in all log entries.
- ToolCallLogger round-trips entries through JSONL correctly.
- get_workflow_id() returns the active workflow or None.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kai_daemon.sdk import (
    ToolPermissionError,
    WorkflowContext,
    _set_tool_call_logger,
    get_workflow_id,
    reset_workflow_context,
    sdk_tool,
    set_workflow_context,
)
from kai_daemon.state.observability import ToolCallEntry, ToolCallLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "tool_calls.jsonl"


@pytest.fixture
def tool_logger(log_path: Path) -> ToolCallLogger:
    return ToolCallLogger(log_path=log_path)


@pytest.fixture(autouse=True)
def inject_test_logger(tool_logger: ToolCallLogger, log_path: Path) -> None:
    """Point the SDK's module-level logger at a temp file for each test."""
    _set_tool_call_logger(tool_logger)


# ---------------------------------------------------------------------------
# Simple decorated tools for testing
# ---------------------------------------------------------------------------


@sdk_tool("test_tool_a")
def _tool_a(x: int) -> int:
    """Return x * 2."""
    return x * 2


@sdk_tool("test_tool_b")
def _tool_b(msg: str) -> str:
    """Return msg uppercased."""
    return msg.upper()


@sdk_tool("failing_tool")
def _failing_tool() -> None:
    """Always raises ValueError."""
    raise ValueError("deliberate error")


# ---------------------------------------------------------------------------
# Behaviour outside workflow context
# ---------------------------------------------------------------------------


class TestNoContext:
    def test_tool_called_without_context_succeeds(self) -> None:
        assert _tool_a(5) == 10

    def test_tool_called_without_context_produces_no_log(self, log_path: Path) -> None:
        _tool_a(5)
        assert not log_path.exists(), "Tools outside workflow context must not log"

    def test_get_workflow_id_returns_none_without_context(self) -> None:
        assert get_workflow_id() is None


# ---------------------------------------------------------------------------
# Permission enforcement
# ---------------------------------------------------------------------------


class TestPermissionEnforcement:
    def test_permitted_tool_succeeds(self, log_path: Path) -> None:
        ctx = WorkflowContext(
            workflow_id="wf_a", allowed_tools=frozenset(["test_tool_a"])
        )
        token = set_workflow_context(ctx)
        try:
            result = _tool_a(3)
        finally:
            reset_workflow_context(token)
        assert result == 6

    def test_forbidden_tool_raises_tool_permission_error(self) -> None:
        ctx = WorkflowContext(
            workflow_id="wf_a",
            allowed_tools=frozenset(["test_tool_b"]),  # not a
        )
        token = set_workflow_context(ctx)
        try:
            with pytest.raises(ToolPermissionError, match="test_tool_a"):
                _tool_a(3)
        finally:
            reset_workflow_context(token)

    def test_forbidden_tool_error_message_contains_workflow_id(self) -> None:
        ctx = WorkflowContext(workflow_id="my_workflow", allowed_tools=frozenset())
        token = set_workflow_context(ctx)
        try:
            with pytest.raises(ToolPermissionError, match="my_workflow"):
                _tool_a(1)
        finally:
            reset_workflow_context(token)

    def test_empty_allowed_tools_blocks_all(self) -> None:
        ctx = WorkflowContext(workflow_id="wf_empty", allowed_tools=frozenset())
        token = set_workflow_context(ctx)
        try:
            with pytest.raises(ToolPermissionError):
                _tool_a(1)
        finally:
            reset_workflow_context(token)

    def test_context_reset_restores_no_permission_check(self) -> None:
        ctx = WorkflowContext(workflow_id="wf", allowed_tools=frozenset())
        token = set_workflow_context(ctx)
        reset_workflow_context(token)
        # After reset — no context, so no permission check
        assert _tool_a(2) == 4


# ---------------------------------------------------------------------------
# Tool call logging — success
# ---------------------------------------------------------------------------


class TestSuccessLogging:
    def test_permitted_call_logged_with_success(self, log_path: Path) -> None:
        ctx = WorkflowContext(
            workflow_id="wf_log", allowed_tools=frozenset(["test_tool_a"])
        )
        token = set_workflow_context(ctx)
        try:
            _tool_a(7)
        finally:
            reset_workflow_context(token)
        entries = ToolCallLogger(log_path).read_all()
        assert len(entries) == 1
        e = entries[0]
        assert e.tool == "test_tool_a"
        assert e.outcome == "success"
        assert e.workflow_id == "wf_log"
        assert e.error is None

    def test_inputs_logged(self, log_path: Path) -> None:
        ctx = WorkflowContext(
            workflow_id="wf_log", allowed_tools=frozenset(["test_tool_a"])
        )
        token = set_workflow_context(ctx)
        try:
            _tool_a(42)
        finally:
            reset_workflow_context(token)
        e = ToolCallLogger(log_path).read_all()[0]
        # The "x" argument should appear in inputs
        assert "x" in e.inputs
        assert e.inputs["x"] == 42

    def test_timestamp_populated(self, log_path: Path) -> None:
        from datetime import datetime

        ctx = WorkflowContext(
            workflow_id="wf_ts", allowed_tools=frozenset(["test_tool_a"])
        )
        token = set_workflow_context(ctx)
        try:
            _tool_a(1)
        finally:
            reset_workflow_context(token)
        e = ToolCallLogger(log_path).read_all()[0]
        datetime.fromisoformat(e.timestamp)  # must parse without error


# ---------------------------------------------------------------------------
# Tool call logging — permission denied
# ---------------------------------------------------------------------------


class TestPermissionDeniedLogging:
    def test_permission_denied_logged(self, log_path: Path) -> None:
        ctx = WorkflowContext(workflow_id="wf_denied", allowed_tools=frozenset())
        token = set_workflow_context(ctx)
        try:
            with pytest.raises(ToolPermissionError):
                _tool_a(1)
        finally:
            reset_workflow_context(token)
        entries = ToolCallLogger(log_path).read_all()
        assert len(entries) == 1
        e = entries[0]
        assert e.outcome == "permission_denied"
        assert e.tool == "test_tool_a"
        assert e.workflow_id == "wf_denied"
        assert e.error is not None

    def test_permission_denied_error_field_non_empty(self, log_path: Path) -> None:
        ctx = WorkflowContext(workflow_id="wf_x", allowed_tools=frozenset())
        token = set_workflow_context(ctx)
        try:
            with pytest.raises(ToolPermissionError):
                _tool_a(1)
        finally:
            reset_workflow_context(token)
        e = ToolCallLogger(log_path).read_all()[0]
        assert len(e.error or "") > 0


# ---------------------------------------------------------------------------
# Tool call logging — error outcome
# ---------------------------------------------------------------------------


class TestErrorLogging:
    def test_tool_error_logged_with_error_outcome(self, log_path: Path) -> None:
        ctx = WorkflowContext(
            workflow_id="wf_err", allowed_tools=frozenset(["failing_tool"])
        )
        token = set_workflow_context(ctx)
        try:
            with pytest.raises(ValueError, match="deliberate error"):
                _failing_tool()
        finally:
            reset_workflow_context(token)
        entries = ToolCallLogger(log_path).read_all()
        assert len(entries) == 1
        e = entries[0]
        assert e.outcome == "error"
        assert e.tool == "failing_tool"
        assert e.workflow_id == "wf_err"
        assert "deliberate error" in (e.error or "")

    def test_exception_still_propagates(self, log_path: Path) -> None:
        ctx = WorkflowContext(
            workflow_id="wf_err", allowed_tools=frozenset(["failing_tool"])
        )
        token = set_workflow_context(ctx)
        try:
            with pytest.raises(ValueError):
                _failing_tool()
        finally:
            reset_workflow_context(token)


# ---------------------------------------------------------------------------
# get_workflow_id
# ---------------------------------------------------------------------------


class TestGetWorkflowId:
    def test_returns_none_without_context(self) -> None:
        assert get_workflow_id() is None

    def test_returns_workflow_id_in_context(self) -> None:
        ctx = WorkflowContext(workflow_id="my_wf", allowed_tools=frozenset())
        token = set_workflow_context(ctx)
        try:
            assert get_workflow_id() == "my_wf"
        finally:
            reset_workflow_context(token)

    def test_returns_none_after_context_reset(self) -> None:
        ctx = WorkflowContext(workflow_id="wf", allowed_tools=frozenset())
        token = set_workflow_context(ctx)
        reset_workflow_context(token)
        assert get_workflow_id() is None


# ---------------------------------------------------------------------------
# ToolCallLogger — JSONL file behaviour
# ---------------------------------------------------------------------------


class TestToolCallLogger:
    def test_append_creates_file(
        self, tool_logger: ToolCallLogger, log_path: Path
    ) -> None:
        assert not log_path.exists()
        ctx = WorkflowContext(
            workflow_id="wf", allowed_tools=frozenset(["test_tool_a"])
        )
        token = set_workflow_context(ctx)
        try:
            _tool_a(1)
        finally:
            reset_workflow_context(token)
        assert log_path.exists()

    def test_each_line_is_valid_json(
        self, tool_logger: ToolCallLogger, log_path: Path
    ) -> None:
        ctx = WorkflowContext(
            workflow_id="wf", allowed_tools=frozenset(["test_tool_a", "test_tool_b"])
        )
        token = set_workflow_context(ctx)
        try:
            _tool_a(1)
            _tool_b("hello")
        finally:
            reset_workflow_context(token)
        for line in log_path.read_text().strip().splitlines():
            data = json.loads(line)
            assert "tool" in data
            assert "outcome" in data

    def test_read_all_empty_when_no_file(self, tmp_path: Path) -> None:
        log = ToolCallLogger(log_path=tmp_path / "nonexistent.jsonl")
        assert log.read_all() == []

    def test_roundtrip_success_entry(self, tmp_path: Path) -> None:
        log_path = tmp_path / "tc.jsonl"
        log = ToolCallLogger(log_path=log_path)
        entry = ToolCallEntry(
            workflow_id="wf_rt",
            tool="my_tool",
            inputs={"a": 1},
            outcome="success",
        )
        log.append(entry)
        restored = log.read_all()[0]
        assert restored.workflow_id == "wf_rt"
        assert restored.tool == "my_tool"
        assert restored.outcome == "success"
        assert restored.inputs == {"a": 1}

    def test_roundtrip_permission_denied_entry(self, tmp_path: Path) -> None:
        log_path = tmp_path / "tc.jsonl"
        log = ToolCallLogger(log_path=log_path)
        entry = ToolCallEntry(
            workflow_id="wf",
            tool="blocked_tool",
            inputs={},
            outcome="permission_denied",
            error="not permitted",
        )
        log.append(entry)
        restored = log.read_all()[0]
        assert restored.outcome == "permission_denied"
        assert restored.error == "not permitted"

    def test_append_silent_on_unwritable_path(self, tmp_path: Path) -> None:
        bad = ToolCallLogger(log_path=tmp_path / "missing_dir" / "tc.jsonl")
        entry = ToolCallEntry(workflow_id="wf", tool="t", inputs={}, outcome="success")
        bad.append(entry)  # must not raise

    def test_read_all_skips_malformed_lines(self, tmp_path: Path) -> None:
        log_path = tmp_path / "tc.jsonl"
        log = ToolCallLogger(log_path=log_path)
        entry = ToolCallEntry(workflow_id="wf", tool="t", inputs={}, outcome="success")
        log.append(entry)
        with log_path.open("a") as fh:
            fh.write("not valid json\n")
        log.append(entry)
        entries = log.read_all()
        assert len(entries) == 2
