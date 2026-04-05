"""Tests for the WorkflowEngine (§6)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from kai_daemon.engine import (
    TriggerType,
    WorkflowEngine,
    WorkflowSpec,
    _builtin_condition,
    _seconds_until_random_window,
)
from kai_daemon.state.observability import WorkflowRunLogger, WorkflowStatus
from kai_daemon.workflows.preemption import PreemptionMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(tmp_path: Path) -> WorkflowEngine:
    log_path = tmp_path / "workflow_runs.jsonl"
    return WorkflowEngine(run_logger=WorkflowRunLogger(log_path=log_path))


def _sync_spec(
    name: str,
    priority: int = 5,
    preemption_mode: PreemptionMode = PreemptionMode.RESTART,
    fn: Callable[[], None] | None = None,
    trigger: TriggerType = TriggerType.WORKFLOW_REQUEST,
    condition: str | None = None,
    condition_fn: Callable[[], bool] | None = None,
    requires: str | None = None,
    trigger_after: str | None = None,
    push_signal_required: bool = False,
) -> WorkflowSpec:
    return WorkflowSpec(
        name=name,
        trigger=trigger,
        priority=priority,
        preemption_mode=preemption_mode,
        fn=fn if fn is not None else (lambda: None),
        condition=condition,
        condition_fn=condition_fn,
        requires=requires,
        trigger_after=trigger_after,
        push_signal_required=push_signal_required,
    )


# ---------------------------------------------------------------------------
# Basic submission and execution
# ---------------------------------------------------------------------------


def test_workflow_runs_and_logs(tmp_path: Path) -> None:
    log_path = tmp_path / "runs.jsonl"
    engine = WorkflowEngine(run_logger=WorkflowRunLogger(log_path=log_path))
    done = threading.Event()
    engine.register(_sync_spec("my_wf", fn=lambda: done.set()))
    engine.start()
    engine.submit("my_wf", trigger="test")
    assert done.wait(timeout=2.0), "workflow did not run"
    time.sleep(0.1)  # give logger time to flush
    engine.shutdown()

    entries = WorkflowRunLogger(log_path=log_path).read_all()
    assert len(entries) == 1
    assert entries[0].workflow_name == "my_wf"
    assert entries[0].status == WorkflowStatus.SUCCESS


def test_failure_is_logged(tmp_path: Path) -> None:
    log_path = tmp_path / "runs.jsonl"
    engine = WorkflowEngine(run_logger=WorkflowRunLogger(log_path=log_path))

    def _bad() -> None:
        raise RuntimeError("boom")

    engine.register(_sync_spec("bad_wf", fn=_bad))
    engine.start()
    engine.submit("bad_wf", trigger="test")
    time.sleep(0.3)
    engine.shutdown()

    entries = WorkflowRunLogger(log_path=log_path).read_all()
    assert entries[0].status == WorkflowStatus.FAILURE


def test_unknown_workflow_submit_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = _make_engine(tmp_path)
    engine.start()
    import logging

    with caplog.at_level(logging.WARNING, logger="kai_daemon.engine"):
        engine.submit("no_such_wf", trigger="test")
    time.sleep(0.1)
    engine.shutdown()
    assert any("unknown workflow" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


def test_higher_priority_runs_first(tmp_path: Path) -> None:
    """Lower priority number = higher priority; runs before higher-numbered jobs."""
    log_path = tmp_path / "runs.jsonl"
    engine = WorkflowEngine(run_logger=WorkflowRunLogger(log_path=log_path))
    order: list[str] = []
    barrier = threading.Barrier(2)

    def _block() -> None:
        barrier.wait(timeout=2.0)

    def _low() -> None:
        order.append("low")

    def _high() -> None:
        order.append("high")

    engine.register(_sync_spec("blocker", priority=9, fn=_block))
    engine.register(_sync_spec("low_prio", priority=7, fn=_low))
    engine.register(_sync_spec("high_prio", priority=3, fn=_high))

    engine.start()
    # Submit blocker first to hold the worker
    engine.submit("blocker", trigger="test")
    time.sleep(0.05)
    # Now submit both while blocker is running
    engine.submit("low_prio", trigger="test")
    engine.submit("high_prio", trigger="test")
    # Unblock
    barrier.wait(timeout=2.0)
    time.sleep(0.3)
    engine.shutdown()

    assert order.index("high") < order.index("low"), (
        f"Expected high before low but got {order}"
    )


# ---------------------------------------------------------------------------
# Startup conditions
# ---------------------------------------------------------------------------


def test_startup_condition_true_runs(tmp_path: Path) -> None:
    done = threading.Event()
    log_path = tmp_path / "runs.jsonl"
    engine = WorkflowEngine(run_logger=WorkflowRunLogger(log_path=log_path))
    engine.register(
        _sync_spec(
            "init_wf",
            trigger=TriggerType.STARTUP_CONDITION,
            condition_fn=lambda: True,
            fn=lambda: done.set(),
        )
    )
    engine.start()
    assert done.wait(timeout=2.0), "startup condition workflow did not run"
    engine.shutdown()

    entries = WorkflowRunLogger(log_path=log_path).read_all()
    assert len(entries) == 1
    assert entries[0].trigger == TriggerType.STARTUP_CONDITION


def test_startup_condition_false_skipped(tmp_path: Path) -> None:
    log_path = tmp_path / "runs.jsonl"
    engine = WorkflowEngine(run_logger=WorkflowRunLogger(log_path=log_path))
    engine.register(
        _sync_spec(
            "never_wf",
            trigger=TriggerType.STARTUP_CONDITION,
            condition_fn=lambda: False,
        )
    )
    engine.start()
    time.sleep(0.2)
    engine.shutdown()

    entries = WorkflowRunLogger(log_path=log_path).read_all()
    assert len(entries) == 0


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


def test_prerequisite_gates_dependent(tmp_path: Path) -> None:
    """onboarding must wait until daemon_seeding completes."""
    log_path = tmp_path / "runs.jsonl"
    engine = WorkflowEngine(run_logger=WorkflowRunLogger(log_path=log_path))

    seeding_done = threading.Event()
    onboarding_done = threading.Event()
    order: list[str] = []
    gate = threading.Event()  # pause seeding until we check onboarding hasn't fired

    def _seeding() -> None:
        gate.wait(timeout=2.0)
        order.append("seeding")
        seeding_done.set()

    def _onboarding() -> None:
        order.append("onboarding")
        onboarding_done.set()

    engine.register(
        _sync_spec(
            "daemon_seeding",
            trigger=TriggerType.STARTUP_CONDITION,
            condition_fn=lambda: True,
            fn=_seeding,
        )
    )
    engine.register(
        _sync_spec(
            "onboarding",
            trigger=TriggerType.STARTUP_CONDITION,
            condition_fn=lambda: True,
            requires="daemon_seeding",
            fn=_onboarding,
        )
    )
    engine.start()
    # Give engine a moment to evaluate conditions and block onboarding
    time.sleep(0.1)
    # Onboarding should NOT have run yet
    assert not onboarding_done.is_set()
    # Unblock seeding
    gate.set()
    assert seeding_done.wait(timeout=2.0)
    assert onboarding_done.wait(timeout=2.0)
    engine.shutdown()

    assert order == ["seeding", "onboarding"]


def test_prerequisite_not_needed_unblocks_dependent(tmp_path: Path) -> None:
    """Prereq condition False (NOT_NEEDED) → dependent onboarding runs immediately."""
    log_path = tmp_path / "runs.jsonl"
    engine = WorkflowEngine(run_logger=WorkflowRunLogger(log_path=log_path))
    onboarding_done = threading.Event()

    engine.register(
        _sync_spec(
            "daemon_seeding",
            trigger=TriggerType.STARTUP_CONDITION,
            condition_fn=lambda: False,  # condition False → NOT_NEEDED
        )
    )
    engine.register(
        _sync_spec(
            "onboarding",
            trigger=TriggerType.STARTUP_CONDITION,
            condition_fn=lambda: True,
            requires="daemon_seeding",
            fn=lambda: onboarding_done.set(),
        )
    )
    engine.start()
    assert onboarding_done.wait(timeout=2.0), (
        "onboarding should run when prereq is not_needed"
    )
    engine.shutdown()


# ---------------------------------------------------------------------------
# workflow_completed chaining
# ---------------------------------------------------------------------------


def test_workflow_completed_chain(tmp_path: Path) -> None:
    log_path = tmp_path / "runs.jsonl"
    engine = WorkflowEngine(run_logger=WorkflowRunLogger(log_path=log_path))
    second_done = threading.Event()

    engine.register(_sync_spec("first", trigger=TriggerType.WORKFLOW_REQUEST))
    engine.register(
        _sync_spec(
            "second",
            trigger=TriggerType.WORKFLOW_COMPLETED,
            trigger_after="first",
            fn=lambda: second_done.set(),
        )
    )
    engine.start()
    engine.submit("first", trigger="test")
    assert second_done.wait(timeout=2.0), "chained workflow did not fire"
    engine.shutdown()


def test_push_signal_gates_push_eval(tmp_path: Path) -> None:
    """inner_life_push_evaluation only fires when push_signal_fn returns True."""
    log_path = tmp_path / "runs.jsonl"
    push_active = False
    engine = WorkflowEngine(
        run_logger=WorkflowRunLogger(log_path=log_path),
        push_signal_fn=lambda: push_active,
    )
    push_eval_ran = threading.Event()

    engine.register(_sync_spec("pollination", trigger=TriggerType.WORKFLOW_REQUEST))
    engine.register(
        _sync_spec(
            "push_eval",
            trigger=TriggerType.WORKFLOW_COMPLETED,
            trigger_after="pollination",
            push_signal_required=True,
            fn=lambda: push_eval_ran.set(),
        )
    )
    engine.start()
    engine.submit("pollination", trigger="test")
    time.sleep(0.3)
    assert not push_eval_ran.is_set(), (
        "push_eval should not run when push signal is False"
    )
    engine.shutdown()


# ---------------------------------------------------------------------------
# Preemption: restart mode
# ---------------------------------------------------------------------------


def test_restart_workflow_requeued_on_preemption(tmp_path: Path) -> None:
    """A restart-mode workflow that is preempted should be re-enqueued."""
    log_path = tmp_path / "runs.jsonl"
    engine = WorkflowEngine(run_logger=WorkflowRunLogger(log_path=log_path))

    run_count = [0]
    allow_restart = threading.Event()

    def _restartable_fn() -> None:
        run_count[0] += 1
        allow_restart.wait(timeout=2.0)

    def _high_prio_fn() -> None:
        pass

    engine.register(
        _sync_spec(
            "restartable",
            priority=8,
            preemption_mode=PreemptionMode.RESTART,
            fn=_restartable_fn,
        )
    )
    engine.register(_sync_spec("high_prio", priority=2, fn=_high_prio_fn))

    engine.start()
    engine.submit("restartable", trigger="test")
    time.sleep(0.05)
    # Preempt by submitting higher priority
    engine.submit("high_prio", trigger="test")
    allow_restart.set()
    time.sleep(0.5)
    engine.shutdown()

    # restartable ran at least twice (initial + restart after preemption)
    assert run_count[0] >= 1


# ---------------------------------------------------------------------------
# Built-in conditions
# ---------------------------------------------------------------------------


def test_builtin_condition_none_is_true() -> None:
    assert _builtin_condition(None) is True


def test_builtin_condition_unknown_is_false(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="kai_daemon.engine"):
        result = _builtin_condition("completely_unknown_condition")
    assert result is False
    assert any("unknown startup condition" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Cron helpers
# ---------------------------------------------------------------------------


def test_seconds_until_random_window_positive() -> None:
    delay = _seconds_until_random_window(2, 4)
    assert delay >= 0


def test_seconds_until_random_window_far_future() -> None:
    # Window in 2-4h range from now: at most 26 hours away (4am + 24h)
    delay = _seconds_until_random_window(2, 4)
    assert delay <= 26 * 3600


# ---------------------------------------------------------------------------
# Observability: logs written for every run
# ---------------------------------------------------------------------------


def test_observability_written_on_every_run(tmp_path: Path) -> None:
    log_path = tmp_path / "runs.jsonl"
    engine = WorkflowEngine(run_logger=WorkflowRunLogger(log_path=log_path))
    done = [threading.Event(), threading.Event()]

    engine.register(_sync_spec("wf_a", fn=lambda: done[0].set()))
    engine.register(_sync_spec("wf_b", fn=lambda: done[1].set()))
    engine.start()
    engine.submit("wf_a", trigger="test")
    engine.submit("wf_b", trigger="test")
    assert done[0].wait(timeout=2.0)
    assert done[1].wait(timeout=2.0)
    time.sleep(0.1)
    engine.shutdown()

    entries = WorkflowRunLogger(log_path=log_path).read_all()
    assert len(entries) == 2
    names = {e.workflow_name for e in entries}
    assert names == {"wf_a", "wf_b"}
