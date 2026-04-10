"""Workflow engine (§6) — priority scheduler with startup_condition and cron triggers.

Architecture
------------
The engine manages a priority queue of workflow runs.  A single worker thread
drains the queue and executes workflows one at a time.  Higher-priority
workflows (lower numeric priority) preempt lower-priority ones via the
``PreemptionContext`` mechanism from ``workflows.preemption``.

Trigger types handled here
--------------------------
startup_condition
    Checked once at ``start()``.  Condition must evaluate to True.
    Prerequisites (``requires``) are enforced: a workflow with ``requires``
    is enqueued only after its prerequisite has completed or been determined
    not to need running (its condition was False).

workflow_completed
    Enqueued automatically when the named predecessor completes.  Used to
    chain the inner-life pipeline:
    daemon_inner_thought_generation → daemon_integration
      → inner_life_thread_pollination → (if push_signal) inner_life_push_evaluation

cron and cron_random_window
    Scheduled via a background scheduler thread.  Each registered cron
    workflow is triggered at its configured time using Python timezone-aware
    datetimes.  The IANA timezone is read from ``user.yaml`` (key: timezone).

Observability
-------------
``WorkflowRunLogger.append()`` is called after every workflow execution —
success, failure, or abandonment.  This writes to
``data/logs/workflow_runs.jsonl``.

Preemption
----------
When a higher-priority workflow is submitted while a lower-priority one is
executing, the engine calls ``PreemptionContext.preempt()`` on the current
run.  For ``restart`` workflows this raises ``WorkflowCancelledError``
immediately.  For ``suspend`` workflows the running workflow checkpoints,
pauses, and the engine re-enqueues it after the high-priority run finishes.
"""

from __future__ import annotations

import logging
import queue
import random
import sched
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from .sdk import (
    WorkflowContext,
    _empty_frozenset_str,
    reset_workflow_context,
    set_workflow_context,
)
from .state.observability import WorkflowRunEntry, WorkflowRunLogger, WorkflowStatus
from .workflows.preemption import (
    PreemptionContext,
    PreemptionMode,
    WorkflowCancelledError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TriggerType(StrEnum):
    STARTUP_CONDITION = "startup_condition"
    WORKFLOW_COMPLETED = "workflow_completed"
    CRON = "cron"
    CRON_RANDOM_WINDOW = "cron_random_window"
    CRON_NIGHTLY = "cron_nightly"
    CRON_NIGHTLY_OR_WRITE_THRESHOLD = "cron_nightly_or_write_threshold"
    CRON_WEEKLY = "cron_weekly"
    CONVERSATION_ENDED = "conversation_ended"
    WORKFLOW_REQUEST = "workflow_request"


class WorkflowLifecycle(StrEnum):
    NOT_NEEDED = "not_needed"  # condition was False — prerequisite is satisfied
    WAITING = "waiting"  # condition True but prerequisite hasn't run yet
    PENDING = "pending"  # enqueued, not yet started
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Workflow specification
# ---------------------------------------------------------------------------


@dataclass
class WorkflowSpec:
    """Static description of one workflow entry from ``workflows.yaml``.

    Parameters
    ----------
    name:
        Unique identifier — matches the key in ``workflows.yaml``.
    trigger:
        TriggerType value.
    priority:
        0 = highest.  Lower number wins.
    preemption_mode:
        ``suspend`` or ``restart``.
    fn:
        Zero-argument callable that runs the workflow.  The engine wraps it
        in observability and preemption.
    condition:
        For ``startup_condition`` trigger — the named condition to evaluate.
        ``None`` means unconditionally run.
    condition_fn:
        Optional callable ``() → bool`` that evaluates the condition at
        startup.  If ``None``, the engine uses a built-in evaluator for
        known condition names.
    requires:
        Name of another workflow that must complete (or be not-needed) before
        this one can start.
    trigger_after:
        For ``workflow_completed`` trigger — name of the predecessor workflow.
    cron_hour:
        For ``cron`` trigger — UTC hour to run (0–23).
    cron_window_start:
        For ``cron_random_window`` — earliest UTC hour.
    cron_window_end:
        For ``cron_random_window`` — latest UTC hour (exclusive).
    push_signal_required:
        For conditional ``workflow_completed`` triggers — only enqueue if the
        push signal is set (inner_life_push_evaluation condition).
    """

    name: str
    trigger: TriggerType
    priority: int
    preemption_mode: PreemptionMode
    fn: Callable[[], None]
    condition: str | None = None
    condition_fn: Callable[[], bool] | None = None
    requires: str | None = None
    trigger_after: str | None = None
    cron_hour: int | None = None
    cron_window_start: int | None = None
    cron_window_end: int | None = None
    push_signal_required: bool = False
    allowed_tools: frozenset[str] = field(default_factory=_empty_frozenset_str)


# ---------------------------------------------------------------------------
# Pending run (queue item)
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _QueueItem:
    """Item in the priority queue.  Ordered by (priority, sequence)."""

    priority: int
    sequence: int
    spec_name: str = field(compare=False)
    trigger: str = field(compare=False)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class WorkflowEngine:
    """Priority workflow scheduler.

    Usage::

        engine = WorkflowEngine()
        engine.register(spec)
        engine.start()          # starts worker + cron threads, runs startup conditions
        # ... daemon runs ...
        engine.shutdown()

    Parameters
    ----------
    run_logger:
        Observability logger.  ``None`` → default path.
    push_signal_fn:
        Callable ``() → bool`` that returns True when the inner-life push
        signal is set (used for ``inner_life_push_evaluation`` gating).
        ``None`` → always False (push evaluation never fires).
    memory_server_available_fn:
        Callable ``() → bool`` for the ``memory_server_available`` field in
        observability logs.  ``None`` → always True.
    """

    def __init__(
        self,
        run_logger: WorkflowRunLogger | None = None,
        *,
        push_signal_fn: Callable[[], bool] | None = None,
        memory_server_available_fn: Callable[[], bool] | None = None,
    ) -> None:
        self._run_logger = run_logger or WorkflowRunLogger()
        self._push_signal_fn = push_signal_fn or (lambda: False)
        self._memory_server_available_fn = memory_server_available_fn or (lambda: True)

        self._specs: dict[str, WorkflowSpec] = {}
        # Priority queue: items are _QueueItem (orderable by priority, sequence)
        self._queue: queue.PriorityQueue[_QueueItem] = queue.PriorityQueue()
        self._sequence_counter = 0
        self._sequence_lock = threading.Lock()

        # Lifecycle state for each registered workflow
        self._lifecycle: dict[str, WorkflowLifecycle] = {}
        self._lifecycle_lock = threading.Lock()

        # Waiting set: specs whose condition is True but prerequisite hasn't run
        self._waiting: set[str] = set()

        # Currently running preemption context (for preempting lower-priority runs)
        self._active_ctx: PreemptionContext | None = None
        self._active_spec_name: str | None = None
        self._active_lock = threading.Lock()

        # Shutdown signal
        self._shutdown = threading.Event()

        # Worker thread (drains the queue)
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="workflow-engine"
        )

        # Cron scheduler thread
        self._cron_scheduler = sched.scheduler(time.monotonic, time.sleep)
        self._cron_thread = threading.Thread(
            target=self._cron_loop, daemon=True, name="workflow-cron"
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, spec: WorkflowSpec) -> None:
        """Register a workflow spec.  Must be called before ``start()``."""
        self._specs[spec.name] = spec
        with self._lifecycle_lock:
            self._lifecycle[spec.name] = WorkflowLifecycle.PENDING

    # ------------------------------------------------------------------
    # Start / shutdown
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the engine.

        Starts the worker and cron threads, then evaluates startup conditions.
        Returns immediately — does not block.
        """
        self._worker_thread.start()
        self._cron_thread.start()
        self._evaluate_startup_conditions()
        self._schedule_cron_workflows()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Signal the engine to stop and wait for the worker to finish.

        Does not cancel an in-progress workflow — the worker finishes its
        current run then exits.
        """
        self._shutdown.set()
        self._worker_thread.join(timeout=timeout)
        self._cron_thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, spec_name: str, trigger: str) -> None:
        """Enqueue a workflow for execution.

        If a lower-priority workflow is currently running, signals preemption.
        """
        spec = self._specs.get(spec_name)
        if spec is None:
            logger.warning("engine: unknown workflow %r — ignoring submit", spec_name)
            return

        with self._sequence_lock:
            seq = self._sequence_counter
            self._sequence_counter += 1

        item = _QueueItem(
            priority=spec.priority,
            sequence=seq,
            spec_name=spec_name,
            trigger=trigger,
        )
        self._queue.put(item)
        logger.debug("engine: enqueued %r (priority %d)", spec_name, spec.priority)

        # Signal preemption if a lower-priority workflow is running
        with self._active_lock:
            if (
                self._active_ctx is not None
                and self._active_spec_name is not None
                and self._specs[self._active_spec_name].priority > spec.priority
            ):
                logger.info(
                    "engine: preempting %r (priority %d) for %r (priority %d)",
                    self._active_spec_name,
                    self._specs[self._active_spec_name].priority,
                    spec_name,
                    spec.priority,
                )
                self._active_ctx.preempt()

    # ------------------------------------------------------------------
    # Startup condition evaluation
    # ------------------------------------------------------------------

    def _evaluate_startup_conditions(self) -> None:
        """Check startup_condition specs and enqueue those whose conditions are met."""
        for spec in self._specs.values():
            if spec.trigger != TriggerType.STARTUP_CONDITION:
                continue
            condition_met = self._eval_condition(spec)
            if not condition_met:
                with self._lifecycle_lock:
                    self._lifecycle[spec.name] = WorkflowLifecycle.NOT_NEEDED
                logger.debug("engine: %r condition not met — skipping", spec.name)
                # Unblock any waiting specs that required this one
                self._unblock_waiters(spec.name)
                continue

            # Condition is met — check prerequisite
            if spec.requires is not None:
                prereq_state = self._get_lifecycle(spec.requires)
                if prereq_state not in (
                    WorkflowLifecycle.COMPLETED,
                    WorkflowLifecycle.NOT_NEEDED,
                ):
                    # Prerequisite hasn't resolved yet — put in waiting set
                    with self._lifecycle_lock:
                        self._lifecycle[spec.name] = WorkflowLifecycle.WAITING
                    self._waiting.add(spec.name)
                    logger.debug(
                        "engine: %r waiting for prerequisite %r",
                        spec.name,
                        spec.requires,
                    )
                    continue

            # Ready to enqueue
            self._enqueue_spec(spec)

    def _eval_condition(self, spec: WorkflowSpec) -> bool:
        """Evaluate the startup condition for *spec*.

        Uses ``spec.condition_fn`` if provided; otherwise falls back to the
        built-in evaluator for known condition names.
        """
        if spec.condition_fn is not None:
            return spec.condition_fn()
        return _builtin_condition(spec.condition)

    def _enqueue_spec(self, spec: WorkflowSpec) -> None:
        """Mark spec as pending and submit it to the queue."""
        with self._lifecycle_lock:
            self._lifecycle[spec.name] = WorkflowLifecycle.PENDING
        self.submit(spec.name, trigger=spec.trigger)

    def _get_lifecycle(self, name: str) -> WorkflowLifecycle:
        with self._lifecycle_lock:
            return self._lifecycle.get(name, WorkflowLifecycle.NOT_NEEDED)

    def _unblock_waiters(self, completed_name: str) -> None:
        """Check the waiting set; enqueue any spec whose prerequisite just resolved."""
        newly_unblocked: list[WorkflowSpec] = []
        for name in list(self._waiting):
            spec = self._specs[name]
            if spec.requires == completed_name:
                prereq_state = self._get_lifecycle(completed_name)
                if prereq_state in (
                    WorkflowLifecycle.COMPLETED,
                    WorkflowLifecycle.NOT_NEEDED,
                ):
                    self._waiting.discard(name)
                    newly_unblocked.append(spec)

        for spec in newly_unblocked:
            logger.info(
                "engine: %r prerequisite %r resolved — enqueuing",
                spec.name,
                spec.requires,
            )
            self._enqueue_spec(spec)

    # ------------------------------------------------------------------
    # Cron scheduling
    # ------------------------------------------------------------------

    def _schedule_cron_workflows(self) -> None:
        """Schedule all cron-triggered workflows via the sched scheduler."""
        for spec in self._specs.values():
            if spec.trigger in (
                TriggerType.CRON,
                TriggerType.CRON_NIGHTLY,
                TriggerType.CRON_NIGHTLY_OR_WRITE_THRESHOLD,
                TriggerType.CRON_WEEKLY,
            ):
                self._schedule_next_cron(spec)
            elif spec.trigger == TriggerType.CRON_RANDOM_WINDOW:
                self._schedule_random_window(spec)

    def _schedule_next_cron(self, spec: WorkflowSpec) -> None:
        """Schedule the next occurrence of a fixed-time cron workflow."""
        delay = _seconds_until_next_run(spec)
        if delay is None:
            return
        self._cron_scheduler.enter(
            delay,
            spec.priority,
            self._fire_cron,
            argument=(spec,),
        )
        logger.debug("engine: cron %r scheduled in %.0fs", spec.name, delay)

    def _schedule_random_window(self, spec: WorkflowSpec) -> None:
        """Schedule a workflow at a random time within its configured window."""
        start_h = spec.cron_window_start if spec.cron_window_start is not None else 2
        end_h = spec.cron_window_end if spec.cron_window_end is not None else 4
        delay = _seconds_until_random_window(start_h, end_h)
        self._cron_scheduler.enter(
            delay,
            spec.priority,
            self._fire_cron,
            argument=(spec,),
        )
        logger.debug(
            "engine: random-window cron %r scheduled in %.0fs", spec.name, delay
        )

    def _fire_cron(self, spec: WorkflowSpec) -> None:
        """Called by the sched scheduler; submits the workflow and reschedules."""
        if self._shutdown.is_set():
            return
        logger.info("engine: cron firing %r", spec.name)
        self.submit(spec.name, trigger=spec.trigger)
        # Reschedule for next occurrence — random-window specs need their own
        # re-scheduler because _seconds_until_next_run returns None for them.
        if spec.trigger == TriggerType.CRON_RANDOM_WINDOW:
            self._schedule_random_window(spec)
        else:
            self._schedule_next_cron(spec)

    def _cron_loop(self) -> None:
        """Background thread that runs the sched scheduler."""
        while not self._shutdown.is_set():
            self._cron_scheduler.run(blocking=False)
            time.sleep(1.0)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Background thread that drains the priority queue."""
        while not self._shutdown.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            spec = self._specs.get(item.spec_name)
            if spec is None:
                logger.warning("engine: no spec for %r — skipping", item.spec_name)
                continue

            self._run_workflow(spec, item.trigger)

    def _run_workflow(self, spec: WorkflowSpec, trigger: str) -> None:
        """Execute a single workflow, handling preemption and observability."""
        ctx = PreemptionContext(spec.preemption_mode)

        with self._lifecycle_lock:
            self._lifecycle[spec.name] = WorkflowLifecycle.RUNNING
        with self._active_lock:
            self._active_ctx = ctx
            self._active_spec_name = spec.name

        started_at = _utcnow()
        status = WorkflowStatus.FAILURE
        logger.info(
            "engine: starting %r (trigger=%s priority=%d)",
            spec.name,
            trigger,
            spec.priority,
        )

        # Set the SDK workflow context so tools can check permissions and log
        # calls with the correct workflow_id.
        _sdk_token = set_workflow_context(
            WorkflowContext(
                workflow_id=spec.name,
                allowed_tools=spec.allowed_tools,
            )
        )

        try:
            spec.fn()
            status = WorkflowStatus.SUCCESS
        except WorkflowCancelledError:
            status = WorkflowStatus.ABANDONED
            logger.info(
                "engine: %r was cancelled (restart mode) — re-enqueuing", spec.name
            )
            # Re-enqueue for restart mode
            self.submit(spec.name, trigger=trigger)
        except Exception:
            logger.exception("engine: %r raised an exception", spec.name)
            status = WorkflowStatus.FAILURE
        finally:
            completed_at = _utcnow()

            reset_workflow_context(_sdk_token)

            with self._active_lock:
                self._active_ctx = None
                self._active_spec_name = None

            new_lifecycle = (
                WorkflowLifecycle.COMPLETED
                if status == WorkflowStatus.SUCCESS
                else WorkflowLifecycle.FAILED
                if status == WorkflowStatus.FAILURE
                else WorkflowLifecycle.PENDING  # re-enqueued after cancel
            )
            with self._lifecycle_lock:
                self._lifecycle[spec.name] = new_lifecycle

            # Resume suspended workflow if we preempted it
            # (The suspended workflow is still in the queue; its ctx is separate.)
            # Preemption resume is managed per-context, not here.

            self._run_logger.append(
                WorkflowRunEntry(
                    workflow_name=spec.name,
                    trigger=trigger,
                    started_at=started_at,
                    completed_at=completed_at,
                    status=status,
                    memory_server_available=self._memory_server_available_fn(),
                )
            )
            logger.info("engine: %r finished with status=%s", spec.name, status)

        # Post-completion: unblock waiting prerequisites and fire chained triggers
        if status == WorkflowStatus.SUCCESS:
            self._unblock_waiters(spec.name)
            self._fire_chained_triggers(spec.name)

    def _fire_chained_triggers(self, completed_name: str) -> None:
        """Enqueue any workflow whose ``trigger_after`` matches *completed_name*."""
        for spec in self._specs.values():
            if spec.trigger_after != completed_name:
                continue
            if spec.push_signal_required and not self._push_signal_fn():
                logger.debug("engine: %r skipped (push signal not set)", spec.name)
                continue
            logger.info("engine: chaining %r after %r", spec.name, completed_name)
            self.submit(spec.name, trigger=f"workflow_completed:{completed_name}")


# ---------------------------------------------------------------------------
# Built-in condition evaluators
# ---------------------------------------------------------------------------


def _builtin_condition(condition: str | None) -> bool:
    """Evaluate a named startup condition.

    Known conditions:

    ``no_daemon_self``
        True if ``DaemonSelfStore().load()`` returns ``None``.

    ``user_yaml_empty``
        True if ``daemon_name`` in ``user.yaml`` is empty or unset.

    Unknown conditions always return False with a warning.
    """
    if condition is None:
        return True

    if condition == "no_daemon_self":
        from .state.daemon_self import DaemonSelfStore  # avoid circular at module level

        return DaemonSelfStore().load() is None

    if condition == "user_yaml_empty":
        return _read_daemon_name() == ""

    logger.warning(
        "engine: unknown startup condition %r — treating as False", condition
    )
    return False


def _read_daemon_name() -> str:
    """Return the ``daemon_name`` field from ``user.yaml``, or ``""`` if missing."""
    import yaml  # local import

    user_yaml = Path(__file__).parents[2] / "user.yaml"
    if not user_yaml.exists():
        return ""
    try:
        # Annotate as dict[str, Any] so pyright can type .get() correctly.
        raw: dict[str, Any] = yaml.safe_load(user_yaml.read_text()) or {}
        daemon_name: Any = raw.get("daemon_name")
        return str(daemon_name) if daemon_name else ""
    except Exception:
        logger.warning("engine: could not read user.yaml", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Cron schedule helpers
# ---------------------------------------------------------------------------


def _seconds_until_next_run(spec: WorkflowSpec) -> float | None:
    """Compute seconds until the next scheduled run for a fixed-time cron spec.

    Returns ``None`` if the spec has no scheduled hour (should not happen for
    cron specs; logged as a warning).
    """
    nightly_triggers = (
        TriggerType.CRON_NIGHTLY,
        TriggerType.CRON_NIGHTLY_OR_WRITE_THRESHOLD,
        TriggerType.CRON_WEEKLY,
    )
    if spec.trigger in nightly_triggers:
        target_hour = spec.cron_hour if spec.cron_hour is not None else 0
    elif spec.trigger == TriggerType.CRON:
        if spec.cron_hour is None:
            logger.warning("engine: cron spec %r has no cron_hour", spec.name)
            return None
        target_hour = spec.cron_hour
    else:
        return None

    now = datetime.now(UTC)
    target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _seconds_until_random_window(start_hour: int, end_hour: int) -> float:
    """Compute seconds until a random time within [start_hour, end_hour) UTC today."""
    now = datetime.now(UTC)
    window_start = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    window_end = now.replace(hour=end_hour, minute=0, second=0, microsecond=0)

    if window_end <= now:
        # Window already passed today — schedule for tomorrow
        window_start += timedelta(days=1)
        window_end += timedelta(days=1)
    elif window_start <= now < window_end:
        # We're inside the window — fire soon (random within remaining window)
        window_start = now + timedelta(seconds=10)

    window_seconds = int((window_end - window_start).total_seconds())
    if window_seconds <= 0:
        window_seconds = 1
    offset = random.randint(0, window_seconds)
    target = window_start + timedelta(seconds=offset)
    delay = (target - now).total_seconds()
    return max(delay, 0.1)


# ---------------------------------------------------------------------------
# Tiny util
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()
