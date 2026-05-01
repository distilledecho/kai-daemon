"""Tests for conversation_server — OpenAI-compatible HTTP chat endpoint."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from kai_daemon.conversation_server import make_app
from kai_daemon.state.daemon_relational import DaemonRelationalStore
from kai_daemon.state.daemon_self import DaemonSelfStore
from kai_daemon.state.holding import HoldingStore
from kai_daemon.state.observability import RegisterInferenceLogger
from kai_daemon.state.thread_stack import SalienceConfig
from kai_daemon.state.threads import ThreadStore
from kai_daemon.workflows.personal_assistant import PersonalAssistant, TurnResult
from kai_daemon.workflows.session_end import SessionEndResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MESSAGES_HELLO = [{"role": "user", "content": "hello"}]


def _make_app(tmp: Path) -> object:
    """Create a fresh make_app instance backed by tmp_path stores."""

    def _se(wm: object, dt: object) -> SessionEndResult:
        from kai_daemon.state.working_memory import WorkingMemory

        assert isinstance(wm, WorkingMemory)
        return SessionEndResult(session_id=wm.session_id, flush_succeeded=True)

    return make_app(
        inference_fn=lambda p: "test response",
        memory_client=None,
        holding_store=HoldingStore(tmp / "holding.yaml"),
        thread_store=ThreadStore(
            threads_path=tmp / "threads",
            pickup_notes_path=tmp / "pn",
        ),
        daemon_self_store=DaemonSelfStore(tmp),
        daemon_relational_store=DaemonRelationalStore(tmp),
        register_inference_logger=RegisterInferenceLogger(tmp / "reg.jsonl"),
        salience_config=SalienceConfig(),
        discharge_threshold=0.72,
        correction_history=[],
        score_discharge_items_fn=lambda m, i: {},
        session_end_fn=_se,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_returns_200(tmp_path: Path) -> None:
    """GET /health returns 200 and {ok: true}."""
    app = _make_app(tmp_path)

    async def _run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            return await client.get("/health")

    resp = asyncio.run(_run())
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_chat_completions_returns_nonempty_response(tmp_path: Path) -> None:
    """POST /v1/chat/completions returns 200 with non-empty content."""
    app = _make_app(tmp_path)

    async def _run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            return await client.post(
                "/v1/chat/completions",
                json={"messages": _MESSAGES_HELLO},
            )

    resp = asyncio.run(_run())
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] != ""


def test_chat_completions_appends_discharge_message(tmp_path: Path) -> None:
    """discharge_message from TurnResult is appended to response content."""
    app = _make_app(tmp_path)

    mock_result = TurnResult(
        response="main response",
        register="casual",
        register_confidence=0.8,
        discharge_surfaced=True,
        discharge_message="something surfaced",
        correction_triggered=False,
        correction_message=None,
    )

    async def _run() -> httpx.Response:
        with patch.object(
            PersonalAssistant,
            "handle_turn",
            new=AsyncMock(return_value=mock_result),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
                base_url="http://test",
            ) as client:
                return await client.post(
                    "/v1/chat/completions",
                    json={"messages": _MESSAGES_HELLO},
                )

    resp = asyncio.run(_run())
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "main response" in content
    assert "something surfaced" in content


def test_chat_completions_reuses_same_assistant(tmp_path: Path) -> None:
    """Two requests to the same app use the same PersonalAssistant instance."""
    app = _make_app(tmp_path)
    begin_session_calls: list[int] = []
    original_begin_session = PersonalAssistant.begin_session

    def _tracking_begin_session(self: PersonalAssistant) -> object:
        begin_session_calls.append(1)
        return original_begin_session(self)

    async def _run() -> None:
        with patch.object(PersonalAssistant, "begin_session", _tracking_begin_session):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
                base_url="http://test",
            ) as client:
                await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "first"}]},
                )
                await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "second"}]},
                )

    asyncio.run(_run())
    assert len(begin_session_calls) == 1, (
        f"begin_session called {len(begin_session_calls)} times; expected 1"
    )
