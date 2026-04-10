"""Interface for ``python -m kai_daemon``."""

from __future__ import annotations

import logging
import signal
import threading
from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

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


def _load_allowed_tools(workflows_yaml: Path) -> dict[str, frozenset[str]]:
    """Parse ``workflows.yaml`` and return a workflow_name → allowed_tools map.

    Returns an empty dict (and warns) if the file is missing or malformed.
    """
    if not workflows_yaml.exists():
        logger.warning(
            "workflows.yaml not found at %s — all workflows have empty tool lists",
            workflows_yaml,
        )
        return {}
    try:
        loaded: Any = yaml.safe_load(workflows_yaml.read_text()) or {}
        if not isinstance(loaded, dict):
            logger.warning("workflows.yaml root is not a mapping — ignoring")
            return {}
        raw: dict[str, Any] = cast(dict[str, Any], loaded)
        result: dict[str, frozenset[str]] = {}
        for wf_name, wf_config in raw.items():
            if not isinstance(wf_config, dict):
                continue
            wf_dict: dict[str, Any] = cast(dict[str, Any], wf_config)
            tools_raw: Any = wf_dict.get("permitted_tools", [])
            if isinstance(tools_raw, list):
                tools_list: list[Any] = cast(list[Any], tools_raw)
                result[str(wf_name)] = frozenset(str(t) for t in tools_list)
        return result
    except Exception:
        logger.warning("Failed to parse workflows.yaml", exc_info=True)
        return {}


def _build_engine(
    run_logger: WorkflowRunLogger,
    *,
    allowed_tools: dict[str, frozenset[str]] | None = None,
) -> WorkflowEngine:
    """Construct and register all known workflows with the engine."""

    _tools = allowed_tools or {}

    def _t(name: str) -> frozenset[str]:
        """Return the allowed tools frozenset for *name*, defaulting to empty."""
        return _tools.get(name, frozenset())

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
            allowed_tools=_t("daemon_seeding"),
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
            allowed_tools=_t("onboarding"),
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
            allowed_tools=_t("daemon_inner_thought_generation"),
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
            allowed_tools=_t("daemon_integration"),
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
            allowed_tools=_t("inner_life_thread_pollination"),
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
            allowed_tools=_t("inner_life_push_evaluation"),
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
            allowed_tools=_t("contradiction_detection"),
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
                allowed_tools=_t(_wf_name),
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
            allowed_tools=_t("associative_retrieval"),
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

    # 3. Build and start the workflow engine.
    #    Load workflows.yaml for the SDK permission matrix, then
    #    start() evaluates startup_conditions (daemon_seeding → onboarding)
    #    and schedules background cron workflows.
    _workflows_yaml = Path(__file__).parents[2] / "workflows.yaml"
    _allowed_tools = _load_allowed_tools(_workflows_yaml)
    engine = _build_engine(run_logger, allowed_tools=_allowed_tools)
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
