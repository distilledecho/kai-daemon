"""Interface for ``python -m kai_daemon``."""

from __future__ import annotations

import logging
import signal
import threading
from argparse import ArgumentParser
from collections.abc import Callable, Sequence

from . import __version__
from .api import DEFAULT_PORT, ActionServer
from .engine import TriggerType, WorkflowEngine, WorkflowSpec
from .state.observability import WorkflowRunLogger
from .workflows.onboarding import run_onboarding
from .workflows.preemption import PreemptionMode

__all__ = ["main"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference placeholder
# ---------------------------------------------------------------------------


def _noop_inference(_prompt: str) -> str:
    """Placeholder used before mlx-kv-client is wired up (Stage 4)."""
    logger.warning(
        "inference: mlx-kv-client not yet configured — "
        "writing minimal DAEMON_SELF v1 (Stage 4 will provide real inference)"
    )
    return ""


def _make_seeding_fn() -> Callable[[], None]:
    """Return the daemon_seeding workflow function."""
    from .workflows.daemon_seeding import run_daemon_seeding

    def _fn() -> None:
        run_daemon_seeding(inference_fn=_noop_inference)

    return _fn


# ---------------------------------------------------------------------------
# Workflow registry
# ---------------------------------------------------------------------------


def _noop(name: str) -> None:
    logger.debug("%s: not yet implemented — skipping", name)


def _build_engine(run_logger: WorkflowRunLogger) -> WorkflowEngine:
    """Construct and register all known workflows with the engine."""

    engine = WorkflowEngine(run_logger=run_logger)

    # ------------------------------------------------------------------
    # Priority 0 — Initialization
    # ------------------------------------------------------------------

    engine.register(
        WorkflowSpec(
            name="daemon_seeding",
            trigger=TriggerType.STARTUP_CONDITION,
            priority=0,
            preemption_mode=PreemptionMode.SUSPEND,
            fn=_make_seeding_fn(),
            condition="no_daemon_self",
        )
    )

    engine.register(
        WorkflowSpec(
            name="onboarding",
            trigger=TriggerType.STARTUP_CONDITION,
            priority=0,
            preemption_mode=PreemptionMode.SUSPEND,
            fn=run_onboarding,
            condition="user_yaml_empty",
            requires="daemon_seeding",
        )
    )

    # ------------------------------------------------------------------
    # Priority 9 — Deep background (inner life generation)
    # ------------------------------------------------------------------

    engine.register(
        WorkflowSpec(
            name="daemon_inner_thought_generation",
            trigger=TriggerType.CRON_RANDOM_WINDOW,
            priority=9,
            preemption_mode=PreemptionMode.SUSPEND,
            fn=lambda: _noop("daemon_inner_thought_generation"),
            cron_window_start=2,
            cron_window_end=4,
        )
    )

    # ------------------------------------------------------------------
    # Priority 8 — Background chained (inner life pipeline)
    # ------------------------------------------------------------------

    engine.register(
        WorkflowSpec(
            name="daemon_integration",
            trigger=TriggerType.WORKFLOW_COMPLETED,
            priority=8,
            preemption_mode=PreemptionMode.RESTART,
            fn=lambda: _noop("daemon_integration"),
            trigger_after="daemon_inner_thought_generation",
        )
    )

    engine.register(
        WorkflowSpec(
            name="inner_life_thread_pollination",
            trigger=TriggerType.WORKFLOW_COMPLETED,
            priority=8,
            preemption_mode=PreemptionMode.RESTART,
            fn=lambda: _noop("inner_life_thread_pollination"),
            trigger_after="daemon_integration",
        )
    )

    engine.register(
        WorkflowSpec(
            name="inner_life_push_evaluation",
            trigger=TriggerType.WORKFLOW_COMPLETED,
            priority=8,
            preemption_mode=PreemptionMode.RESTART,
            fn=lambda: _noop("inner_life_push_evaluation"),
            trigger_after="inner_life_thread_pollination",
            push_signal_required=True,
        )
    )

    # ------------------------------------------------------------------
    # Priority 5 — Nightly knowledge
    # ------------------------------------------------------------------

    engine.register(
        WorkflowSpec(
            name="contradiction_detection",
            trigger=TriggerType.CRON_NIGHTLY,
            priority=5,
            preemption_mode=PreemptionMode.SUSPEND,
            fn=lambda: _noop("contradiction_detection"),
            cron_hour=0,
        )
    )

    # ------------------------------------------------------------------
    # Priority 6 — Nightly maintenance
    # ------------------------------------------------------------------

    for _wf_name, _wf_hour in [
        ("open_loop_review", 22),
        ("embedding_backfill", 1),
        ("transcript_pruning", 1),
        ("holding_review", 2),
    ]:
        engine.register(
            WorkflowSpec(
                name=_wf_name,
                trigger=TriggerType.CRON_NIGHTLY,
                priority=6,
                preemption_mode=PreemptionMode.SUSPEND,
                fn=lambda n=_wf_name: _noop(n),
                cron_hour=_wf_hour,
            )
        )

    # ------------------------------------------------------------------
    # Priority 7 — Late night
    # ------------------------------------------------------------------

    engine.register(
        WorkflowSpec(
            name="associative_retrieval",
            trigger=TriggerType.CRON,
            priority=7,
            preemption_mode=PreemptionMode.SUSPEND,
            fn=lambda: _noop("associative_retrieval"),
            cron_hour=1,
        )
    )

    return engine


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------


def _init_state() -> None:
    """Ensure all daemon-local state directories exist (idempotent)."""
    from .state._paths import (
        daemon_relational_history_dir,
        daemon_self_history_dir,
        daemon_state_dir,
        logs_dir,
        memory_queue_dir,
        pickup_notes_dir,
        threads_dir,
    )

    daemon_state_dir()
    daemon_self_history_dir()
    daemon_relational_history_dir()
    logs_dir()
    threads_dir()
    pickup_notes_dir()
    memory_queue_dir()
    logger.debug("state: all directories initialised")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(args: Sequence[str] | None = None) -> None:
    """CLI entry point for the kai-daemon."""
    parser = ArgumentParser(description="kai-daemon — persistent AI daemon")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Action API port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parsed = parser.parse_args(args)

    logging.basicConfig(
        level=getattr(logging, parsed.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    logger.info("kai-daemon %s starting", __version__)

    # 1. Initialise daemon-local state directories
    _init_state()

    # 2. Start the localhost action API in a daemon thread
    run_logger = WorkflowRunLogger()
    action_server = ActionServer(port=parsed.port)
    api_thread = threading.Thread(
        target=action_server.serve_forever,
        daemon=True,
        name="action-api",
    )
    api_thread.start()
    host, port = action_server.address
    logger.info("action-api: ready at http://%s:%d", host, port)

    # 3. Build and start the workflow engine
    #    start() evaluates startup_conditions (daemon_seeding → onboarding)
    #    and schedules background cron workflows.
    engine = _build_engine(run_logger)
    engine.start()

    # 4. Block until SIGINT / SIGTERM
    stop = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("daemon: received signal %d — shutting down", signum)
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("kai-daemon: running (Ctrl-C to stop)")
    stop.wait()

    logger.info("kai-daemon: shutting down")
    engine.shutdown()
    action_server.shutdown()
    logger.info("kai-daemon: stopped")


if __name__ == "__main__":
    main()
