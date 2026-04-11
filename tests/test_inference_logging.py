"""Tests for inference call logging — InstrumentedMlxKvClient (Stage 3.5).

Verifies that:
- Every primitive call is logged to inference_calls.jsonl with all fields.
- Failed calls are logged with success=False and the exception still propagates.
- workflow_id is captured from the SDK ContextVar at call time.
- status() is proxied without logging.
- A failed status() call substitutes 0 for token counts rather than blocking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kai_daemon.inference import InstrumentedMlxKvClient
from kai_daemon.sdk import (
    WorkflowContext,
    reset_workflow_context,
    set_workflow_context,
)
from kai_daemon.state.observability import InferenceCallEntry, InferenceCallLogger

# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------


class _MockStatus:
    """Minimal status object with a cache_used_tokens attribute."""

    def __init__(self, tokens: int) -> None:
        self.cache_used_tokens = tokens


class _MockKvClient:
    """Duck-typed mlx-kv-server client for testing."""

    def __init__(self, tokens_before: int = 100, tokens_after: int = 150) -> None:
        self._tokens_before = tokens_before
        self._tokens_after = tokens_after
        self._call_count = 0
        self.calls: list[str] = []

    def status(self) -> _MockStatus:
        # Alternate between before and after token counts so calls alternate.
        count = self._call_count
        self._call_count += 1
        if count % 2 == 0:
            return _MockStatus(self._tokens_before)
        return _MockStatus(self._tokens_after)

    def checkpoint(self) -> str:
        self.calls.append("checkpoint")
        return "ok"

    def rollback(self) -> None:
        self.calls.append("rollback")

    def evict(self) -> None:
        self.calls.append("evict")

    def prefill(self, prompt: str) -> str:
        self.calls.append("prefill")
        return f"filled:{prompt}"

    def generate(self, max_tokens: int = 256) -> str:
        self.calls.append("generate")
        return "generated text"


class _FailingKvClient(_MockKvClient):
    """Client whose checkpoint() raises RuntimeError."""

    def checkpoint(self) -> str:  # type: ignore[override]
        raise RuntimeError("server error")


class _BadStatusClient(_MockKvClient):
    """Client whose status() raises to test fallback to 0 token count."""

    def status(self) -> _MockStatus:  # type: ignore[override]
        raise ConnectionError("status unavailable")

    def checkpoint(self) -> str:  # type: ignore[override]
        return "ok"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "inference_calls.jsonl"


@pytest.fixture
def mock_client() -> _MockKvClient:
    return _MockKvClient(tokens_before=100, tokens_after=200)


def _make_client(mock: Any, log_path: Path) -> InstrumentedMlxKvClient:
    return InstrumentedMlxKvClient(mock, log_path=log_path)


# ---------------------------------------------------------------------------
# Logging — field completeness
# ---------------------------------------------------------------------------


class TestInferenceCallLogging:
    def test_checkpoint_logged_with_all_fields(
        self, mock_client: _MockKvClient, log_path: Path
    ) -> None:
        client = _make_client(mock_client, log_path)
        client.checkpoint()
        entries = InferenceCallLogger(log_path).read_all()
        assert len(entries) == 1
        e = entries[0]
        assert e.primitive == "checkpoint"
        assert e.tokens_before == 100
        assert e.tokens_after == 200
        assert e.duration_ms >= 0
        assert e.success is True
        assert e.workflow_id is None  # no workflow context set

    def test_rollback_logged(self, mock_client: _MockKvClient, log_path: Path) -> None:
        _make_client(mock_client, log_path).rollback()
        entries = InferenceCallLogger(log_path).read_all()
        assert len(entries) == 1
        assert entries[0].primitive == "rollback"
        assert entries[0].success is True

    def test_evict_logged(self, mock_client: _MockKvClient, log_path: Path) -> None:
        _make_client(mock_client, log_path).evict()
        entries = InferenceCallLogger(log_path).read_all()
        assert entries[0].primitive == "evict"

    def test_prefill_logged(self, mock_client: _MockKvClient, log_path: Path) -> None:
        _make_client(mock_client, log_path).prefill("hello")
        entries = InferenceCallLogger(log_path).read_all()
        assert entries[0].primitive == "prefill"

    def test_generate_logged(self, mock_client: _MockKvClient, log_path: Path) -> None:
        _make_client(mock_client, log_path).generate(max_tokens=64)
        entries = InferenceCallLogger(log_path).read_all()
        assert entries[0].primitive == "generate"

    def test_multiple_calls_append(
        self, mock_client: _MockKvClient, log_path: Path
    ) -> None:
        client = _make_client(mock_client, log_path)
        client.checkpoint()
        client.rollback()
        client.evict()
        entries = InferenceCallLogger(log_path).read_all()
        assert len(entries) == 3
        primitives = [e.primitive for e in entries]
        assert primitives == ["checkpoint", "rollback", "evict"]

    def test_timestamp_is_iso8601(
        self, mock_client: _MockKvClient, log_path: Path
    ) -> None:
        from datetime import datetime

        _make_client(mock_client, log_path).checkpoint()
        e = InferenceCallLogger(log_path).read_all()[0]
        # Should parse without error
        datetime.fromisoformat(e.timestamp)

    def test_log_is_valid_jsonl(
        self, mock_client: _MockKvClient, log_path: Path
    ) -> None:
        client = _make_client(mock_client, log_path)
        client.checkpoint()
        client.rollback()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "primitive" in parsed
            assert "success" in parsed

    def test_roundtrip_via_model(
        self, mock_client: _MockKvClient, log_path: Path
    ) -> None:
        _make_client(mock_client, log_path).checkpoint()
        raw_line = log_path.read_text().strip()
        entry = InferenceCallEntry.model_validate_json(raw_line)
        assert entry.primitive == "checkpoint"
        assert entry.success is True


# ---------------------------------------------------------------------------
# Failed call — success=False, exception re-raised
# ---------------------------------------------------------------------------


class TestFailedCall:
    def test_failed_call_logged_with_success_false(self, log_path: Path) -> None:
        client = _make_client(_FailingKvClient(), log_path)
        with pytest.raises(RuntimeError, match="server error"):
            client.checkpoint()
        entries = InferenceCallLogger(log_path).read_all()
        assert len(entries) == 1
        assert entries[0].success is False
        assert entries[0].primitive == "checkpoint"

    def test_exception_propagates_after_logging(self, log_path: Path) -> None:
        client = _make_client(_FailingKvClient(), log_path)
        with pytest.raises(RuntimeError):
            client.checkpoint()
        # Log was still written
        assert len(InferenceCallLogger(log_path).read_all()) == 1

    def test_failed_call_token_fields_present(self, log_path: Path) -> None:
        client = _make_client(_FailingKvClient(), log_path)
        with pytest.raises(RuntimeError):
            client.checkpoint()
        e = InferenceCallLogger(log_path).read_all()[0]
        assert isinstance(e.tokens_before, int)
        assert isinstance(e.tokens_after, int)


# ---------------------------------------------------------------------------
# Bad status() — token count falls back to 0
# ---------------------------------------------------------------------------


class TestBadStatus:
    def test_status_failure_substitutes_zero(self, log_path: Path) -> None:
        client = _make_client(_BadStatusClient(), log_path)
        client.checkpoint()
        e = InferenceCallLogger(log_path).read_all()[0]
        assert e.tokens_before == 0
        assert e.tokens_after == 0
        assert e.success is True  # primitive itself succeeded


# ---------------------------------------------------------------------------
# workflow_id threading via ContextVar
# ---------------------------------------------------------------------------


class TestWorkflowIdThreading:
    def test_workflow_id_captured_from_context(
        self, mock_client: _MockKvClient, log_path: Path
    ) -> None:
        ctx = WorkflowContext(
            workflow_id="daemon_integration", allowed_tools=frozenset()
        )
        token = set_workflow_context(ctx)
        try:
            _make_client(mock_client, log_path).checkpoint()
        finally:
            reset_workflow_context(token)
        e = InferenceCallLogger(log_path).read_all()[0]
        assert e.workflow_id == "daemon_integration"

    def test_workflow_id_null_outside_context(
        self, mock_client: _MockKvClient, log_path: Path
    ) -> None:
        _make_client(mock_client, log_path).checkpoint()
        e = InferenceCallLogger(log_path).read_all()[0]
        assert e.workflow_id is None

    def test_workflow_id_reset_after_context_exits(
        self, mock_client: _MockKvClient, log_path: Path
    ) -> None:
        ctx = WorkflowContext(workflow_id="test_wf", allowed_tools=frozenset())
        token = set_workflow_context(ctx)
        reset_workflow_context(token)
        # After reset, workflow_id should be None again
        _make_client(mock_client, log_path).checkpoint()
        e = InferenceCallLogger(log_path).read_all()[0]
        assert e.workflow_id is None


# ---------------------------------------------------------------------------
# status() proxy — not logged
# ---------------------------------------------------------------------------


class TestStatusProxy:
    def test_status_not_logged(
        self, mock_client: _MockKvClient, log_path: Path
    ) -> None:
        client = _make_client(mock_client, log_path)
        client.status()
        assert not log_path.exists(), "status() must not produce a log entry"

    def test_status_returns_underlying_result(
        self, mock_client: _MockKvClient, log_path: Path
    ) -> None:
        client = _make_client(mock_client, log_path)
        result = client.status()
        assert hasattr(result, "cache_used_tokens")


# ---------------------------------------------------------------------------
# Primitive return value passthrough
# ---------------------------------------------------------------------------


class TestReturnValue:
    def test_prefill_return_value_preserved(
        self, mock_client: _MockKvClient, log_path: Path
    ) -> None:
        result = _make_client(mock_client, log_path).prefill("hello")
        assert result == "filled:hello"

    def test_checkpoint_return_value_preserved(
        self, mock_client: _MockKvClient, log_path: Path
    ) -> None:
        result = _make_client(mock_client, log_path).checkpoint()
        assert result == "ok"
