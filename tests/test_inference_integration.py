"""Integration tests for the _inference closure against live mlx-kv-server.

All tests are skipped automatically when mlx-kv-server is not reachable,
so this file is safe on CI and on machines without mlx-kv-server running.
"""

from __future__ import annotations

import socket

import pytest

from kai_daemon.__main__ import _make_inference_fn
from kai_daemon.workflows.personal_assistant import (
    _PROMPT_RESPONSE_MARKER,
    _PROMPT_USER_MARKER,
)


def _kv_server_reachable() -> bool:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect("/tmp/mlx-kv-server.sock")
        return True
    except OSError:
        return False
    finally:
        s.close()


requires_kv_server = pytest.mark.skipif(
    not _kv_server_reachable(),
    reason="mlx-kv-server not reachable at /tmp/mlx-kv-server.sock",
)


@requires_kv_server
def test_inference_returns_nonempty_response() -> None:
    """Shape 2 prompt produces a non-empty response that is not the prompt."""
    inference_fn = _make_inference_fn()
    prompt = "Say hello in one sentence."
    response = inference_fn(prompt)
    assert isinstance(response, str)
    assert len(response) > 0
    assert response != prompt


@requires_kv_server
def test_inference_shape1_no_echo() -> None:
    """Shape 1 prompt does not echo the user message verbatim."""
    inference_fn = _make_inference_fn()
    system_prompt = "You are a helpful assistant."
    user_message = "Hello"
    prompt = (
        f"{system_prompt}{_PROMPT_USER_MARKER}{user_message}{_PROMPT_RESPONSE_MARKER}"
    )
    response = inference_fn(prompt)
    assert isinstance(response, str)
    assert len(response) > 0
    # The exact echo regression: response is the user message verbatim.
    assert response.strip() != user_message


@requires_kv_server
@pytest.mark.parametrize("message", ["Hello", "How are you?", "What is 2+2?"])
def test_inference_does_not_echo_user_message(message: str) -> None:
    """Each short message gets a genuine response, not a verbatim echo."""
    inference_fn = _make_inference_fn()
    system_prompt = "You are a helpful assistant."
    prompt = f"{system_prompt}{_PROMPT_USER_MARKER}{message}{_PROMPT_RESPONSE_MARKER}"
    response = inference_fn(prompt)
    assert response.strip() != message
