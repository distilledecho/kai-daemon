"""Preemption model for the workflow engine (§6e, Stage 3A).

Two preemption modes per ``workflows.yaml``:

suspend
    The workflow calls ``cooperate()`` at a safe point between inference
    calls.  ``cooperate()`` invokes ``checkpoint_fn()`` (which calls
    ``checkpoint`` on mlx-kv-server to save the KV cache), signals the
    engine that it is safe to start higher-priority work, then blocks
    until the engine calls ``resume()``.  On resume, ``rollback_fn()``
    is called to restore the KV cache and the workflow continues from
    where it paused.

restart
    The workflow calls ``cooperate()`` at a safe point.  ``cooperate()``
    raises ``WorkflowCancelledError`` immediately — no checkpoint, no state
    saved.  The engine catches ``WorkflowCancelledError`` and restarts the
    workflow from the beginning once higher-priority work is done.
    All ``restart`` workflows must have idempotent writes.

Non-blocking contract
    ``preempt()`` sets an event and returns immediately.  The engine can
    start higher-priority work without waiting.  For ``suspend`` mode
    the engine MAY call ``wait_for_checkpoint()`` to defer new inference
    until the running workflow has actually checkpointed; this avoids
    contention on the mlx-kv-server KV cache.

Checkpoint and rollback callables are injected — this module has no
dependency on mlx-kv-client (consistent with the project's injectable
callable pattern throughout the workflow layer).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import StrEnum


class PreemptionMode(StrEnum):
    """Preemption mode for a workflow, as declared in ``workflows.yaml``."""

    SUSPEND = "suspend"
    RESTART = "restart"


class WorkflowCancelledError(Exception):
    """Raised inside a ``restart`` workflow when a preemption signal is received.

    The workflow runner catches this, discards the partial run, and restarts
    the workflow from the beginning once higher-priority work has finished.
    All ``restart`` workflows must ensure their writes are idempotent.
    """


class PreemptionContext:
    """Coordinates preemption between the workflow engine and a running workflow.

    Engine-side usage::

        ctx = PreemptionContext(PreemptionMode.SUSPEND)
        # start workflow thread, passing ctx
        ctx.preempt()                        # non-blocking signal
        ctx.wait_for_checkpoint(timeout=5.0) # optional: wait for safe boundary
        # run higher-priority work
        ctx.resume()                         # unblock the suspended workflow

    Workflow-side usage (called at every safe point between inference calls)::

        ctx.cooperate(checkpoint_fn, rollback_fn)

    Each ``PreemptionContext`` instance is intended for a single
    preemption/resume cycle.  Create a new instance for each workflow run.
    """

    def __init__(self, mode: PreemptionMode) -> None:
        self._mode: PreemptionMode = mode
        self._preempt_event: threading.Event = threading.Event()
        self._checkpoint_done_event: threading.Event = threading.Event()
        self._resume_event: threading.Event = threading.Event()

    @property
    def is_preempted(self) -> bool:
        """``True`` after ``preempt()`` has been called."""
        return self._preempt_event.is_set()

    def preempt(self) -> None:
        """Signal the workflow to cooperate with preemption.

        Returns immediately (non-blocking).  The workflow will notice the
        signal at its next ``cooperate()`` call.
        """
        self._preempt_event.set()

    def wait_for_checkpoint(self, timeout: float) -> bool:
        """Wait until the workflow has called ``checkpoint_fn`` and paused.

        Returns ``True`` if the checkpoint was confirmed within *timeout*
        seconds, ``False`` on timeout.  Only meaningful for ``suspend`` mode;
        for ``restart`` mode the workflow terminates without checkpointing so
        this will always time out.

        The engine should handle a ``False`` return gracefully (proceed
        anyway) — it must not block indefinitely waiting for a checkpoint.
        """
        return self._checkpoint_done_event.wait(timeout=timeout)

    def resume(self) -> None:
        """Unblock a suspended workflow so it can call ``rollback_fn`` and continue.

        Idempotent.  Safe to call before ``cooperate()`` reaches its internal
        wait — the workflow will not hang when it arrives.
        """
        self._resume_event.set()

    def cooperate(
        self,
        checkpoint_fn: Callable[[], None],
        rollback_fn: Callable[[], None],
    ) -> None:
        """Called by the workflow at a safe preemption point.

        If no preemption signal has been received, returns immediately (the
        common hot path — no locking, one boolean check).

        If preempted:

        * ``restart`` mode: raises ``WorkflowCancelledError`` immediately.
          ``checkpoint_fn`` and ``rollback_fn`` are never called.
        * ``suspend`` mode: calls ``checkpoint_fn()``, signals that the
          checkpoint is done (unblocking ``wait_for_checkpoint()``), then
          blocks until ``resume()`` is called, then calls ``rollback_fn()``
          and returns so the workflow can continue.

        If ``checkpoint_fn`` raises, ``_checkpoint_done_event`` is never set
        so ``wait_for_checkpoint()`` will time out and return ``False``.  The
        engine must handle ``False`` gracefully (proceed anyway).  The
        exception propagates out of ``cooperate()`` and the workflow thread
        exits with that error — no resume is needed.

        Parameters
        ----------
        checkpoint_fn:
            Zero-argument callable that saves the current inference state on
            mlx-kv-server.  Injected by the workflow runner.
        rollback_fn:
            Zero-argument callable that restores the saved inference state on
            mlx-kv-server.  Injected by the workflow runner.
        """
        if not self._preempt_event.is_set():
            return

        if self._mode == PreemptionMode.RESTART:
            raise WorkflowCancelledError

        # --- suspend path ---
        # If checkpoint_fn raises, the exception propagates here. The
        # _checkpoint_done_event is left unset so wait_for_checkpoint() times
        # out and returns False — the engine proceeds without waiting.
        checkpoint_fn()
        self._checkpoint_done_event.set()  # engine's wait_for_checkpoint unblocks
        self._resume_event.wait()  # block until engine calls resume()
        rollback_fn()
        # return normally — workflow continues from this point
