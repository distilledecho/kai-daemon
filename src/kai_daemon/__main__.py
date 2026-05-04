"""Interface for ``python -m kai_daemon``."""

from __future__ import annotations

import logging
import re
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
from .workflows.personal_assistant import (
    _PROMPT_RESPONSE_MARKER,
    _PROMPT_USER_MARKER,
)
from .workflows.preemption import PreemptionMode

__all__ = ["main"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference — mlx-kv-client wiring
# ---------------------------------------------------------------------------

_KV_SOCKET_PATH: str = "/tmp/mlx-kv-server.sock"
_INFERENCE_CACHE_ID: str = "kai-daemon-main"

_inference_fn: Callable[[str], str] | None = None
_inference_lock: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Output normalizers — one per model family, selected at construction time
# ---------------------------------------------------------------------------


def _strip_model_artifacts(text: str) -> str:
    """Strip common model output artifacts present in all chat model families.

    Removes <think>...</think> blocks and bare role-label lines ("assistant",
    "user", "system") that some models emit before their actual output.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s*(assistant|user|system)\s*\n?", "", text)
    return text.strip()


def _strip_generic_artifacts(text: str) -> str:
    """Fallback normalizer — strips leading/trailing whitespace only."""
    return text.strip()


_NORMALIZERS: list[tuple[str, Callable[[str], str]]] = [
    ("qwen3", _strip_model_artifacts),
    ("qwen2", _strip_model_artifacts),  # same artifact family
]


def _get_normalizer(model_name: str) -> Callable[[str], str]:
    """Return the normalizer for *model_name* via case-insensitive substring match."""
    lower = model_name.lower()
    for prefix, fn in _NORMALIZERS:
        if prefix in lower:
            return fn
    return _strip_generic_artifacts


def _make_inference_fn() -> Callable[[str], str]:
    """Build the real inference closure backed by mlx-kv-client.

    Imports are deferred to the function body so that missing mlx packages
    do not cause import errors in the test environment (mlx only runs on M1).
    """
    from mlx_kv_client import MlxKvClient  # type: ignore[import-untyped]
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    kv_client: Any = cast(Any, MlxKvClient(_KV_SOCKET_PATH))
    status: Any = kv_client.status()
    logger.info("inference: connected — model=%s", status.model)

    # Reads from ~/.cache/huggingface/hub — no network access.
    # Raises OSError immediately if the cache is missing.
    # cast(Any, ...) keeps pyright strict-mode clean despite untyped import.
    tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(
        status.model,
        local_files_only=True,
    )

    normalizer = _get_normalizer(status.model)
    logger.info("inference: output normalizer=%s", normalizer.__name__)

    def _inference(prompt: str) -> str:
        if _PROMPT_USER_MARKER in prompt and _PROMPT_RESPONSE_MARKER in prompt:
            # Shape 1: personal_assistant format — split into system + user turns.
            parts = prompt.split(_PROMPT_USER_MARKER, 1)
            system_text = parts[0]
            user_text = parts[1].split(_PROMPT_RESPONSE_MARKER, 1)[0]
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
        else:
            # Shape 2: plain instructional prompt (seeding, etc.).
            messages = [{"role": "user", "content": prompt}]
        formatted: str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # apply_chat_template already inserts special tokens as text;
        # re-encoding with add_special_tokens=True would double-add them.
        tokens: Any = tokenizer.encode(formatted, add_special_tokens=False)
        kv_client.prefill(tokens, _INFERENCE_CACHE_ID)
        output_tokens: list[Any] = []
        for token in kv_client.generate([tokenizer.eos_token_id], _INFERENCE_CACHE_ID):
            output_tokens.append(token)
        return normalizer(tokenizer.decode(output_tokens, skip_special_tokens=True))

    _inference._kv_client = kv_client  # type: ignore[attr-defined]  # noqa: SLF001
    return _inference


def _get_inference_fn() -> Callable[[str], str]:
    """Return the singleton inference callable, initialising it on first call."""
    global _inference_fn
    with _inference_lock:
        if _inference_fn is None:
            _inference_fn = _make_inference_fn()
        return _inference_fn


def _shutdown_inference() -> None:
    """Evict the KV cache entry and reset the inference singleton."""
    global _inference_fn
    fn = _inference_fn
    if fn is not None:
        kv_client = getattr(fn, "_kv_client", None)
        if kv_client is not None:
            try:
                result = kv_client.evict(_INFERENCE_CACHE_ID)
                logger.info("inference: evict result=%r", result)
            except Exception:
                logger.warning("inference: evict failed during shutdown", exc_info=True)
    _inference_fn = None


def _make_seeding_fn() -> Callable[[], None]:
    """Return the daemon_seeding workflow function."""
    from .workflows.daemon_seeding import run_daemon_seeding

    def _fn() -> None:
        run_daemon_seeding(inference_fn=_get_inference_fn())
        # Evict KV cache after seeding so conversation starts clean.
        # Without this, conversation prefill extends the seeding context
        # and the model echoes the user message as a continuation.
        fn = _inference_fn
        if fn is not None:
            kv_client = getattr(fn, "_kv_client", None)
            if kv_client is not None:
                try:
                    kv_client.evict(_INFERENCE_CACHE_ID)
                    logger.info(
                        "inference: seeding cache evicted — ready for conversation"
                    )
                except Exception:
                    logger.warning(
                        "inference: failed to evict seeding cache", exc_info=True
                    )

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
        "--conv-port",
        type=int,
        default=9272,
        help="Conversation server port (default: 9272)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--force-reseed",
        action="store_true",
        default=False,
        help="Delete daemon_self.yaml before starting so daemon_seeding re-runs",
    )
    parsed = parser.parse_args(args)

    logging.basicConfig(
        level=getattr(logging, parsed.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    logger.info("kai-daemon %s starting", __version__)

    _init_state()

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

    from .conversation_server import run_conversation_server

    def _conv_inference(prompt: str) -> str:
        return _get_inference_fn()(prompt)

    conv_thread = threading.Thread(
        target=run_conversation_server,
        kwargs={
            "inference_fn": _conv_inference,
            "host": "0.0.0.0",
            "port": parsed.conv_port,
        },
        daemon=True,
        name="conv-server",
    )
    conv_thread.start()
    logger.info("conv-server: starting at http://0.0.0.0:%d", parsed.conv_port)

    if parsed.force_reseed:
        from .state._paths import daemon_state_dir

        _daemon_self_path = daemon_state_dir() / "daemon_self.yaml"
        logger.warning("--force-reseed: daemon_seeding will be forced to re-run")
        if _daemon_self_path.exists():
            _daemon_self_path.unlink()
            logger.warning("--force-reseed: deleted %s", _daemon_self_path)

    _workflows_yaml = Path(__file__).parents[2] / "workflows.yaml"
    _allowed_tools = _load_allowed_tools(_workflows_yaml)
    engine = _build_engine(run_logger, allowed_tools=_allowed_tools)
    engine.start()

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
    _shutdown_inference()
    action_server.shutdown()
    logger.info("kai-daemon: stopped")


if __name__ == "__main__":
    main()
