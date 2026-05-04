"""Tests for _make_inference_fn, _get_inference_fn, and _shutdown_inference.

All external packages (mlx_kv_client, mlx_lm) are mocked via sys.modules so that
tests run without Apple Silicon hardware or network access.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_inference_singleton() -> Generator[None, None, None]:
    import kai_daemon.__main__ as m

    m._inference_fn = None
    yield
    m._inference_fn = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patched_modules(
    mock_kv_client: MagicMock,
    mock_load: MagicMock,
) -> dict[str, Any]:
    """Build a sys.modules patch dict with mock mlx_kv_client and mlx_lm."""
    mlx_kv_mod = MagicMock()
    mlx_kv_mod.MlxKvClient = MagicMock(return_value=mock_kv_client)
    mlx_lm_mod = MagicMock()
    mlx_lm_mod.load = mock_load
    return {"mlx_kv_client": mlx_kv_mod, "mlx_lm": mlx_lm_mod}


# ---------------------------------------------------------------------------
# Test 1 — tokenizer construction contract
# ---------------------------------------------------------------------------


def test_tokenizer_construction_contract() -> None:
    import kai_daemon.__main__ as m

    mock_kv_client = MagicMock()
    mock_load = MagicMock(return_value=(MagicMock(), MagicMock()))

    mods = _patched_modules(mock_kv_client, mock_load)

    with patch.dict(sys.modules, mods):
        result = m._make_inference_fn()

    mods["mlx_kv_client"].MlxKvClient.assert_called_once_with(m._KV_SOCKET_PATH)
    mock_kv_client.status.assert_called_once_with()
    mock_load.assert_called_once_with(mock_kv_client.status.return_value.model)
    assert callable(result)


# ---------------------------------------------------------------------------
# Test 2 — inference closure call sequence
# ---------------------------------------------------------------------------


def test_inference_closure_call_sequence() -> None:
    import kai_daemon.__main__ as m

    mock_kv_client = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "hello"
    mock_kv_client.generate.return_value = iter([4, 5])

    mock_load = MagicMock(return_value=(MagicMock(), mock_tokenizer))
    mods = _patched_modules(mock_kv_client, mock_load)

    with patch.dict(sys.modules, mods):
        inference_fn = m._make_inference_fn()

    result = inference_fn("test prompt")

    mock_tokenizer.encode.assert_called_once_with("test prompt")
    mock_kv_client.prefill.assert_called_once_with([1, 2, 3], m._INFERENCE_CACHE_ID)
    mock_kv_client.generate.assert_called_once_with([0], m._INFERENCE_CACHE_ID)
    mock_tokenizer.decode.assert_called_once_with([4, 5], skip_special_tokens=True)
    assert result == "hello"


# ---------------------------------------------------------------------------
# Test 3 — singleton
# ---------------------------------------------------------------------------


def test_singleton() -> None:
    import kai_daemon.__main__ as m

    first_mock = MagicMock()
    second_mock = MagicMock()
    side_effects = [first_mock, second_mock]

    with patch.object(m, "_make_inference_fn", side_effect=side_effects) as patched:
        r1 = m._get_inference_fn()
        r2 = m._get_inference_fn()

    patched.assert_called_once()
    assert r1 is r2
    assert r1 is first_mock


# ---------------------------------------------------------------------------
# Test 4 — shutdown evicts and resets
# ---------------------------------------------------------------------------


def test_shutdown_evicts_and_resets() -> None:
    import kai_daemon.__main__ as m

    mock_kv_client = MagicMock()
    mock_fn = MagicMock()
    mock_fn._kv_client = mock_kv_client
    m._inference_fn = mock_fn

    m._shutdown_inference()

    mock_kv_client.evict.assert_called_once_with(m._INFERENCE_CACHE_ID)
    assert m._inference_fn is None


# ---------------------------------------------------------------------------
# Test 5 — shutdown swallows evict exceptions
# ---------------------------------------------------------------------------


def test_shutdown_swallows_evict_exception() -> None:
    import kai_daemon.__main__ as m

    mock_kv_client = MagicMock()
    mock_kv_client.evict.side_effect = RuntimeError("boom")
    mock_fn = MagicMock()
    mock_fn._kv_client = mock_kv_client
    m._inference_fn = mock_fn

    m._shutdown_inference()  # must not raise

    assert m._inference_fn is None


# ---------------------------------------------------------------------------
# Test 6 — shutdown no-ops when not initialised
# ---------------------------------------------------------------------------


def test_shutdown_noop_when_not_initialised() -> None:
    import kai_daemon.__main__ as m

    assert m._inference_fn is None
    m._shutdown_inference()  # must not raise
