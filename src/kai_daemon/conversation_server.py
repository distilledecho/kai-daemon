"""Conversation HTTP server — OpenAI-compatible chat completions endpoint.

Exposes ``POST /v1/chat/completions`` matching the OpenAI schema so Open WebUI
works without configuration, and ``GET /health`` for endpoint verification.

Binds to ``0.0.0.0`` so Tailscale can reach it (unlike the action API which
is localhost-only).
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from .state.daemon_relational import DaemonRelationalStore
from .state.daemon_self import DaemonSelfStore
from .state.holding import HoldingItem, HoldingStore
from .state.observability import RegisterCorrectionEntry, RegisterInferenceLogger
from .state.retrieval import MemoryClientProtocol
from .state.thread_stack import SalienceConfig
from .state.threads import ThreadStore
from .state.working_memory import WorkingMemory
from .workflows.personal_assistant import (
    InferenceFn,
    PersonalAssistant,
    ScoreDischargeItemsFn,
    SessionEndFn,
)
from .workflows.session_end import SessionEndResult

__all__ = ["make_app", "run_conversation_server"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI-compatible Pydantic models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Incoming chat completion request."""

    model: str = "kai"
    messages: list[ChatMessage]
    stream: bool = False


class ChatChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str


class ChatResponse(BaseModel):
    """Chat completion response (OpenAI-compatible)."""

    id: str
    object: str = "chat.completion"
    model: str = "kai"
    choices: list[ChatChoice]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def make_app(
    inference_fn: InferenceFn,
    *,
    memory_client: MemoryClientProtocol | None,
    holding_store: HoldingStore,
    thread_store: ThreadStore,
    daemon_self_store: DaemonSelfStore,
    daemon_relational_store: DaemonRelationalStore,
    register_inference_logger: RegisterInferenceLogger,
    salience_config: SalienceConfig,
    discharge_threshold: float,
    correction_history: list[RegisterCorrectionEntry],
    score_discharge_items_fn: ScoreDischargeItemsFn,
    session_end_fn: SessionEndFn,
) -> FastAPI:
    """Create and return the conversation FastAPI application.

    All :class:`PersonalAssistant` dependencies are accepted explicitly for
    testability. In production, use :func:`run_conversation_server` which
    constructs these from default state paths.

    The ``PersonalAssistant`` is initialised lazily on the first request and
    reused across subsequent requests (one persistent session per app instance).

    Args:
        inference_fn: Inference callable ``(prompt: str) → str``.
        memory_client: Memory retrieval client, or ``None`` to disable.
        holding_store: Holding store for discharge candidates.
        thread_store: Thread store for stack context.
        daemon_self_store: DAEMON_SELF store.
        daemon_relational_store: DAEMON_RELATIONAL store.
        register_inference_logger: Register correction log.
        salience_config: Salience computation constants.
        discharge_threshold: Discharge similarity gate (0–1).
        correction_history: Prior register correction entries.
        score_discharge_items_fn: Discharge item similarity scorer.
        session_end_fn: Session end handler.

    Returns:
        Configured :class:`fastapi.FastAPI` application.

    Example::

        >>> import tempfile
        >>> from pathlib import Path
        >>> from kai_daemon.state.daemon_self import DaemonSelfStore
        >>> from kai_daemon.state.daemon_relational import DaemonRelationalStore
        >>> from kai_daemon.state.holding import HoldingStore
        >>> from kai_daemon.state.threads import ThreadStore
        >>> from kai_daemon.state.observability import RegisterInferenceLogger
        >>> from kai_daemon.state.thread_stack import SalienceConfig
        >>> from kai_daemon.workflows.session_end import SessionEndResult
        >>> from kai_daemon.conversation_server import make_app
        >>> with tempfile.TemporaryDirectory() as d:
        ...     dp = Path(d)
        ...     app = make_app(
        ...         inference_fn=lambda p: "ok",
        ...         memory_client=None,
        ...         holding_store=HoldingStore(dp / "h.yaml"),
        ...         thread_store=ThreadStore(
        ...             threads_path=dp / "t", pickup_notes_path=dp / "pn"
        ...         ),
        ...         daemon_self_store=DaemonSelfStore(dp),
        ...         daemon_relational_store=DaemonRelationalStore(dp),
        ...         register_inference_logger=RegisterInferenceLogger(dp / "r.jsonl"),
        ...         salience_config=SalienceConfig(),
        ...         discharge_threshold=0.72,
        ...         correction_history=[],
        ...         score_discharge_items_fn=lambda m, i: {},
        ...         session_end_fn=lambda wm, dt: SessionEndResult(
        ...             session_id=wm.session_id, flush_succeeded=True
        ...         ),
        ...     )
        ...     app.title
        'kai'
    """
    app = FastAPI(title="kai")

    # Per-app state stored in a mutable container for closure capture
    _app_state: list[PersonalAssistant | None] = [None]
    _app_lock = threading.Lock()

    def _get_assistant() -> PersonalAssistant:
        with _app_lock:
            if _app_state[0] is None:
                pa = PersonalAssistant(
                    inference_fn=inference_fn,
                    memory_client=memory_client,
                    holding_store=holding_store,
                    thread_store=thread_store,
                    daemon_self_store=daemon_self_store,
                    daemon_relational_store=daemon_relational_store,
                    register_inference_logger=register_inference_logger,
                    salience_config=salience_config,
                    discharge_threshold=discharge_threshold,
                    correction_history=correction_history,
                    score_discharge_items_fn=score_discharge_items_fn,
                    session_end_fn=session_end_fn,
                )
                pa.begin_session()
                _app_state[0] = pa
                return pa
            return _app_state[0]  # type: ignore[return-value]

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest) -> ChatResponse:
        assistant = _get_assistant()
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "",
        )
        result = await assistant.handle_turn(last_user)
        content = result.response
        if result.discharge_message is not None:
            content = f"{content}\n\n{result.discharge_message}"
        if result.correction_message is not None:
            content = f"{content}\n\n{result.correction_message}"
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
        )

    return app


# ---------------------------------------------------------------------------
# Production entry point
# ---------------------------------------------------------------------------


def run_conversation_server(
    inference_fn: InferenceFn,
    *,
    host: str = "0.0.0.0",
    port: int = 9272,
) -> None:
    """Start the conversation HTTP server, blocking until stopped.

    Constructs all :class:`PersonalAssistant` dependencies from default
    daemon state paths.  Binds to *host* (default ``0.0.0.0``) so
    Tailscale can reach it.

    Args:
        inference_fn: Inference callable ``(prompt: str) → str``.
        host: Bind address (default ``0.0.0.0``).
        port: HTTP port (default 9272).
    """
    from .state._paths import (
        daemon_relational_history_dir,
        daemon_self_history_dir,
        daemon_state_dir,
        logs_dir,
        pickup_notes_dir,
        threads_dir,
    )

    state_dir = daemon_state_dir()

    # TODO: wire real vector similarity scorer once daemon-memory-client is
    # available; _zero_scores disables discharge surfacing entirely.
    def _zero_scores(_message: str, _items: list[HoldingItem]) -> dict[str, float]:
        return {}

    # TODO: wire real episodic_flush here; flush_succeeded=True bypasses the
    # working-memory gate (memory is cleared even when the memory server is down).
    def _simple_session_end(wm: WorkingMemory, _dt: Any) -> SessionEndResult:
        return SessionEndResult(session_id=wm.session_id, flush_succeeded=True)

    app = make_app(
        inference_fn=inference_fn,
        # TODO: wire daemon-memory-client here; retrieval is disabled until then.
        memory_client=None,
        holding_store=HoldingStore(state_dir / "holding.yaml"),
        thread_store=ThreadStore(
            threads_path=threads_dir(),
            pickup_notes_path=pickup_notes_dir(),
        ),
        daemon_self_store=DaemonSelfStore(
            state_dir=state_dir,
            history_dir=daemon_self_history_dir(),
        ),
        daemon_relational_store=DaemonRelationalStore(
            state_dir=state_dir,
            history_dir=daemon_relational_history_dir(),
        ),
        register_inference_logger=RegisterInferenceLogger(
            logs_dir() / "register_inference.jsonl"
        ),
        salience_config=SalienceConfig(),
        discharge_threshold=0.72,
        correction_history=[],
        score_discharge_items_fn=_zero_scores,
        session_end_fn=_simple_session_end,
    )
    uvicorn.run(app, host=host, port=port)
