"""Kai SDK — permission enforcement and tool call logging (Stage 3.5).

The SDK is the enforcement layer for the tool permission matrix declared in
``workflows.yaml``.  It provides:

1. **WorkflowContext** — a frozen dataclass carried in a ``ContextVar`` for the
   duration of each workflow run.  Set by the engine; never by workflow code.

2. **sdk_tool decorator** — wraps a tool function to add permission checking
   and call logging.  When no workflow context is active (e.g., in tests) the
   tool is called directly with no side effects.

3. **ToolPermissionError** — raised immediately when a tool is invoked outside
   its permitted set for the current workflow.  The error is logged to
   ``tool_calls.jsonl`` before being raised.

Permission enforcement
----------------------
The engine calls ``set_workflow_context()`` before invoking ``spec.fn()`` and
resets it in the ``finally`` block.  Tools decorated with ``@sdk_tool`` check
the context at call time; there is no way to bypass this check by passing
additional arguments.

Usage (workflow engine — not workflow code)::

    ctx = WorkflowContext(workflow_id="daemon_integration",
                          allowed_tools=frozenset(["daemon_inner_thought_filter"]))
    token = set_workflow_context(ctx)
    try:
        spec.fn()
    finally:
        reset_workflow_context(token)

Usage (tool declaration)::

    @sdk_tool("daemon_inner_thought")
    def daemon_inner_thought(fascinations, *, inference_fn, ...):
        ...
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from .state.observability import ToolCallEntry, ToolCallLogger

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Frozenset helper (resolves pyright frozenset[Unknown] inference)
# ---------------------------------------------------------------------------


def _empty_frozenset_str() -> frozenset[str]:
    return frozenset()


# ---------------------------------------------------------------------------
# Workflow context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowContext:
    """Permission context for the currently executing workflow.

    Carried in ``_ctx_var`` for the duration of a single workflow run.
    Set by the engine before calling ``spec.fn()``; never by workflow code.

    Attributes
    ----------
    workflow_id:
        Unique identifier matching the key in ``workflows.yaml``.
    allowed_tools:
        Frozenset of tool names the workflow is permitted to call.
    """

    workflow_id: str
    allowed_tools: frozenset[str] = field(default_factory=_empty_frozenset_str)


_ctx_var: contextvars.ContextVar[WorkflowContext | None] = contextvars.ContextVar(
    "_kai_workflow_ctx", default=None
)


def get_workflow_context() -> WorkflowContext | None:
    """Return the active ``WorkflowContext``, or ``None`` outside a workflow."""
    return _ctx_var.get()


def get_workflow_id() -> str | None:
    """Return the active workflow's ID, or ``None`` outside a workflow.

    Exported for use by the inference call logger so both logs share the
    same workflow identifier without a separate ContextVar.
    """
    ctx = _ctx_var.get()
    return ctx.workflow_id if ctx is not None else None


def set_workflow_context(
    ctx: WorkflowContext,
) -> contextvars.Token[WorkflowContext | None]:
    """Set the active workflow context.

    Returns a token for ``reset_workflow_context``.
    """
    return _ctx_var.set(ctx)


def reset_workflow_context(
    token: contextvars.Token[WorkflowContext | None],
) -> None:
    """Reset the workflow context to its previous value using *token*."""
    _ctx_var.reset(token)


# ---------------------------------------------------------------------------
# ToolPermissionError
# ---------------------------------------------------------------------------


class ToolPermissionError(Exception):
    """Raised when a tool is invoked outside its permitted workflow context.

    Logged to ``tool_calls.jsonl`` with ``outcome="permission_denied"`` before
    being raised.  Execution halts; the workflow runner propagates this as a
    failure.
    """


# ---------------------------------------------------------------------------
# Module-level tool call logger (injectable for tests)
# ---------------------------------------------------------------------------

# Lazily initialised to avoid creating data/logs/ at import time in contexts
# where KAI_DATA_DIR is not yet configured.
_tool_call_logger: ToolCallLogger | None = None


def _get_tool_call_logger() -> ToolCallLogger:
    """Return the module-level ToolCallLogger, initialising lazily on first call."""
    global _tool_call_logger
    if _tool_call_logger is None:
        _tool_call_logger = ToolCallLogger()
    return _tool_call_logger


def _set_tool_call_logger(  # pyright: ignore[reportUnusedFunction]
    log: ToolCallLogger,
) -> None:
    """Replace the module-level ToolCallLogger (for tests).

    Not intended for production use — the default logger writes to the
    configured data/logs/ path.
    """
    global _tool_call_logger
    _tool_call_logger = log


# ---------------------------------------------------------------------------
# Input serialisation helpers
# ---------------------------------------------------------------------------


def _to_loggable(value: Any) -> Any:
    """Convert *value* to a JSON-serialisable form for the inputs dict.

    Tries: direct JSON, Pydantic model_dump(), truncated repr.
    """
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass
    try:
        return value.model_dump()  # type: ignore[union-attr]
    except AttributeError:
        pass
    r = repr(value)
    return r[:500] if len(r) > 500 else r


def _serialise_inputs(
    sig: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Bind *args*/*kwargs* to *sig* and return a JSON-safe inputs dict.

    Callables are omitted (they are implementation details, not data).
    Returns ``{}`` on any error so logging never breaks a tool call.
    """
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return {
            k: _to_loggable(v) for k, v in bound.arguments.items() if not callable(v)
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# sdk_tool decorator
# ---------------------------------------------------------------------------


def sdk_tool(
    name: str, log_path: Path | None = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that registers a tool and enforces permission + logging.

    Uses ``ParamSpec`` so the decorated function's full signature is preserved
    for type checkers — callers see exactly the original parameter types.

    When no workflow context is active (``_ctx_var`` is ``None``), the tool
    is called directly — no permission check, no logging.  This keeps existing
    tests transparent.

    Parameters
    ----------
    name:
        Canonical tool identifier, as declared in ``workflows.yaml``.
    log_path:
        Override the default ``tool_calls.jsonl`` path (for tests).
    """
    _log: ToolCallLogger | None = (
        ToolCallLogger(log_path) if log_path is not None else None
    )

    def _logger() -> ToolCallLogger:
        return _log if _log is not None else _get_tool_call_logger()

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        _sig = inspect.signature(fn)  # computed once at decoration time

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            ctx = _ctx_var.get()
            if ctx is None:
                # Outside workflow context — no check, no logging.
                return fn(*args, **kwargs)

            # --- permission check ---
            if name not in ctx.allowed_tools:
                error_msg = (
                    f"Tool '{name}' is not permitted for workflow '{ctx.workflow_id}'"
                )
                try:
                    _logger().append(
                        ToolCallEntry(
                            workflow_id=ctx.workflow_id,
                            tool=name,
                            inputs={},
                            outcome="permission_denied",
                            error=error_msg,
                        )
                    )
                except Exception:
                    logger.warning(
                        "sdk: failed to log permission_denied for tool %r",
                        name,
                        exc_info=True,
                    )
                raise ToolPermissionError(error_msg)

            # --- call and log ---
            inputs = _serialise_inputs(_sig, args, kwargs)  # type: ignore[arg-type]
            try:
                result: T = fn(*args, **kwargs)
                try:
                    _logger().append(
                        ToolCallEntry(
                            workflow_id=ctx.workflow_id,
                            tool=name,
                            inputs=inputs,
                            outcome="success",
                        )
                    )
                except Exception:
                    logger.warning(
                        "sdk: failed to log success for tool %r",
                        name,
                        exc_info=True,
                    )
                return result
            except ToolPermissionError:
                raise  # already logged above; do not double-log
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                try:
                    _logger().append(
                        ToolCallEntry(
                            workflow_id=ctx.workflow_id,
                            tool=name,
                            inputs=inputs,
                            outcome="error",
                            error=error_msg,
                        )
                    )
                except Exception:
                    logger.warning(
                        "sdk: failed to log error for tool %r",
                        name,
                        exc_info=True,
                    )
                raise

        return wrapper

    return decorator
