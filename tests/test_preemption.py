"""Tests for the preemption model (§6e, Stage 3A)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from kai_daemon.workflows.preemption import (
    PreemptionContext,
    PreemptionMode,
    WorkflowCancelledError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THREAD_TIMEOUT = 2.0  # seconds — tests fail rather than hang if exceeded


def _run_in_thread(fn: Callable[[], None]) -> threading.Thread:
    """Start *fn* in a daemon thread and return it."""
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


def _noop() -> None:
    pass


# ---------------------------------------------------------------------------
# PreemptionMode
# ---------------------------------------------------------------------------


class TestPreemptionMode:
    def test_values(self) -> None:
        assert PreemptionMode.SUSPEND == "suspend"
        assert PreemptionMode.RESTART == "restart"

    def test_is_str_enum(self) -> None:
        assert isinstance(PreemptionMode.SUSPEND, str)
        assert isinstance(PreemptionMode.RESTART, str)


# ---------------------------------------------------------------------------
# WorkflowCancelledError
# ---------------------------------------------------------------------------


class TestWorkflowCancelledError:
    def test_is_exception(self) -> None:
        assert issubclass(WorkflowCancelledError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(WorkflowCancelledError):
            raise WorkflowCancelledError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def suspend_ctx() -> PreemptionContext:
    return PreemptionContext(PreemptionMode.SUSPEND)


@pytest.fixture
def restart_ctx() -> PreemptionContext:
    return PreemptionContext(PreemptionMode.RESTART)


# ---------------------------------------------------------------------------
# is_preempted
# ---------------------------------------------------------------------------


class TestIsPreempted:
    def test_false_initially_suspend(self, suspend_ctx: PreemptionContext) -> None:
        assert not suspend_ctx.is_preempted

    def test_false_initially_restart(self, restart_ctx: PreemptionContext) -> None:
        assert not restart_ctx.is_preempted

    def test_true_after_preempt_suspend(self, suspend_ctx: PreemptionContext) -> None:
        suspend_ctx.preempt()
        assert suspend_ctx.is_preempted

    def test_true_after_preempt_restart(self, restart_ctx: PreemptionContext) -> None:
        restart_ctx.preempt()
        assert restart_ctx.is_preempted


# ---------------------------------------------------------------------------
# No preemption signal — cooperate() is a no-op
# ---------------------------------------------------------------------------


class TestCooperateNoSignal:
    def test_noop_suspend(self, suspend_ctx: PreemptionContext) -> None:
        calls: list[str] = []
        suspend_ctx.cooperate(
            checkpoint_fn=lambda: calls.append("ckpt"),
            rollback_fn=lambda: calls.append("rb"),
        )
        assert calls == []

    def test_noop_restart(self, restart_ctx: PreemptionContext) -> None:
        calls: list[str] = []
        restart_ctx.cooperate(
            checkpoint_fn=lambda: calls.append("ckpt"),
            rollback_fn=lambda: calls.append("rb"),
        )
        assert calls == []


# ---------------------------------------------------------------------------
# preempt() is non-blocking
# ---------------------------------------------------------------------------


class TestPreemptNonBlocking:
    def test_preempt_returns_immediately_suspend(
        self, suspend_ctx: PreemptionContext
    ) -> None:
        start = time.monotonic()
        suspend_ctx.preempt()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_preempt_returns_immediately_restart(
        self, restart_ctx: PreemptionContext
    ) -> None:
        start = time.monotonic()
        restart_ctx.preempt()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_preempt_twice_is_safe(self, suspend_ctx: PreemptionContext) -> None:
        suspend_ctx.preempt()
        suspend_ctx.preempt()  # idempotent — must not raise
        assert suspend_ctx.is_preempted

    def test_resume_twice_is_safe(self, suspend_ctx: PreemptionContext) -> None:
        suspend_ctx.resume()
        suspend_ctx.resume()  # idempotent — must not raise


# ---------------------------------------------------------------------------
# Restart mode
# ---------------------------------------------------------------------------


class TestRestartCooperate:
    def test_raises_workflow_cancelled(self, restart_ctx: PreemptionContext) -> None:
        restart_ctx.preempt()
        with pytest.raises(WorkflowCancelledError):
            restart_ctx.cooperate(checkpoint_fn=_noop, rollback_fn=_noop)

    def test_checkpoint_fn_not_called(self, restart_ctx: PreemptionContext) -> None:
        calls: list[str] = []
        restart_ctx.preempt()
        with pytest.raises(WorkflowCancelledError):
            restart_ctx.cooperate(
                checkpoint_fn=lambda: calls.append("ckpt"),
                rollback_fn=_noop,
            )
        assert calls == []

    def test_rollback_fn_not_called(self, restart_ctx: PreemptionContext) -> None:
        calls: list[str] = []
        restart_ctx.preempt()
        with pytest.raises(WorkflowCancelledError):
            restart_ctx.cooperate(
                checkpoint_fn=_noop,
                rollback_fn=lambda: calls.append("rb"),
            )
        assert calls == []

    def test_wait_for_checkpoint_times_out(
        self, restart_ctx: PreemptionContext
    ) -> None:
        # restart mode never sets the checkpoint event
        restart_ctx.preempt()
        result = restart_ctx.wait_for_checkpoint(timeout=0.05)
        assert result is False


# ---------------------------------------------------------------------------
# Suspend mode — cooperative checkpoint/resume cycle
# ---------------------------------------------------------------------------


class TestSuspendCooperate:
    def test_checkpoint_fn_called(self, suspend_ctx: PreemptionContext) -> None:
        """cooperate() calls checkpoint_fn after preemption signal."""
        calls: list[str] = []

        def workflow() -> None:
            suspend_ctx.cooperate(
                checkpoint_fn=lambda: calls.append("ckpt"),
                rollback_fn=lambda: calls.append("rb"),
            )

        suspend_ctx.preempt()
        t = _run_in_thread(workflow)
        # Wait for checkpoint to be signalled
        assert suspend_ctx.wait_for_checkpoint(timeout=_THREAD_TIMEOUT)
        assert "ckpt" in calls
        # Resume so the thread can complete
        suspend_ctx.resume()
        t.join(timeout=_THREAD_TIMEOUT)
        assert not t.is_alive()

    def test_blocks_until_resume(self, suspend_ctx: PreemptionContext) -> None:
        """cooperate() blocks the workflow thread until resume() is called."""
        done: list[bool] = []

        def workflow() -> None:
            suspend_ctx.cooperate(checkpoint_fn=_noop, rollback_fn=_noop)
            done.append(True)

        suspend_ctx.preempt()
        t = _run_in_thread(workflow)
        assert suspend_ctx.wait_for_checkpoint(timeout=_THREAD_TIMEOUT)

        # Workflow should still be blocked
        t.join(timeout=0.1)
        assert t.is_alive(), "Workflow should be blocked before resume()"
        assert done == []

        # Resume unblocks it
        suspend_ctx.resume()
        t.join(timeout=_THREAD_TIMEOUT)
        assert not t.is_alive()
        assert done == [True]

    def test_rollback_fn_called_after_resume(
        self, suspend_ctx: PreemptionContext
    ) -> None:
        """rollback_fn is called after resume(), before cooperate() returns."""
        calls: list[str] = []

        def workflow() -> None:
            suspend_ctx.cooperate(
                checkpoint_fn=lambda: calls.append("ckpt"),
                rollback_fn=lambda: calls.append("rb"),
            )
            calls.append("continued")

        suspend_ctx.preempt()
        t = _run_in_thread(workflow)
        assert suspend_ctx.wait_for_checkpoint(timeout=_THREAD_TIMEOUT)

        # rollback not called yet
        assert "rb" not in calls
        suspend_ctx.resume()
        t.join(timeout=_THREAD_TIMEOUT)
        assert not t.is_alive()

        # Order: checkpoint → rollback → continued
        assert calls == ["ckpt", "rb", "continued"]

    def test_cooperate_returns_normally(self, suspend_ctx: PreemptionContext) -> None:
        """cooperate() does not raise in suspend mode — workflow continues."""
        raised: list[BaseException] = []

        def workflow() -> None:
            try:
                suspend_ctx.cooperate(checkpoint_fn=_noop, rollback_fn=_noop)
            except BaseException as exc:
                raised.append(exc)

        suspend_ctx.preempt()
        t = _run_in_thread(workflow)
        assert suspend_ctx.wait_for_checkpoint(timeout=_THREAD_TIMEOUT)
        suspend_ctx.resume()
        t.join(timeout=_THREAD_TIMEOUT)
        assert not t.is_alive()
        assert raised == []

    def test_wait_for_checkpoint_returns_true(
        self, suspend_ctx: PreemptionContext
    ) -> None:
        suspend_ctx.preempt()
        t = _run_in_thread(
            lambda: suspend_ctx.cooperate(checkpoint_fn=_noop, rollback_fn=_noop)
        )
        result = suspend_ctx.wait_for_checkpoint(timeout=_THREAD_TIMEOUT)
        assert result is True
        suspend_ctx.resume()
        t.join(timeout=_THREAD_TIMEOUT)

    def test_wait_for_checkpoint_timeout_returns_false(
        self, suspend_ctx: PreemptionContext
    ) -> None:
        # No cooperate() call — checkpoint event is never set
        suspend_ctx.preempt()
        result = suspend_ctx.wait_for_checkpoint(timeout=0.05)
        assert result is False

    def test_resume_before_cooperate_does_not_hang(
        self, suspend_ctx: PreemptionContext
    ) -> None:
        """resume() called before cooperate() — cooperate() must not block."""
        suspend_ctx.preempt()
        suspend_ctx.resume()  # set resume event before the workflow arrives

        done: list[bool] = []

        def workflow() -> None:
            suspend_ctx.cooperate(checkpoint_fn=_noop, rollback_fn=_noop)
            done.append(True)

        t = _run_in_thread(workflow)
        t.join(timeout=_THREAD_TIMEOUT)
        assert not t.is_alive()
        assert done == [True]


# ---------------------------------------------------------------------------
# Engine non-blocking: simulate priority preemption
# ---------------------------------------------------------------------------


class TestEngineNonBlocking:
    def test_engine_starts_high_priority_work_immediately(
        self, suspend_ctx: PreemptionContext
    ) -> None:
        """After preempt(), the engine can start high-priority work without
        waiting for the low-priority workflow to cooperate."""
        low_priority_at_safe_point: threading.Event = threading.Event()
        high_priority_done: list[bool] = []

        def low_priority_workflow() -> None:
            # Simulate some work before the safe point
            low_priority_at_safe_point.wait()
            suspend_ctx.cooperate(checkpoint_fn=_noop, rollback_fn=_noop)

        t = _run_in_thread(low_priority_workflow)

        # Engine signals preemption — non-blocking
        start = time.monotonic()
        suspend_ctx.preempt()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1, "preempt() must return immediately"

        # Engine does high-priority work right away (does not wait for checkpoint)
        high_priority_done.append(True)

        # Now let the low-priority workflow reach its safe point
        low_priority_at_safe_point.set()
        assert suspend_ctx.wait_for_checkpoint(timeout=_THREAD_TIMEOUT)

        # Resume and clean up
        suspend_ctx.resume()
        t.join(timeout=_THREAD_TIMEOUT)
        assert not t.is_alive()
        assert high_priority_done == [True]

    def test_restart_engine_non_blocking(self, restart_ctx: PreemptionContext) -> None:
        """restart: WorkflowCancelledError is raised and engine is unblocked."""
        cancelled: list[bool] = []

        def workflow() -> None:
            try:
                restart_ctx.cooperate(checkpoint_fn=_noop, rollback_fn=_noop)
            except WorkflowCancelledError:
                cancelled.append(True)

        restart_ctx.preempt()
        t = _run_in_thread(workflow)
        t.join(timeout=_THREAD_TIMEOUT)
        assert not t.is_alive()
        assert cancelled == [True]
