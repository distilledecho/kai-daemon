"""Tests for _make_inference_fn, _get_inference_fn, and _shutdown_inference.

Both mlx_kv_client and transformers are mocked via sys.modules so that tests
run without Apple Silicon hardware, model weights, or network access.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
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


def _make_inference_mocks(
    mock_tokenizer: MagicMock | None = None,
) -> tuple[MagicMock, MagicMock, dict[str, Any]]:
    """Return (mock_kv_client, mock_tokenizer, sys_modules_patch).

    Injects mock mlx_kv_client and transformers modules into sys.modules so
    that _make_inference_fn() can be called without real packages installed.
    """
    mock_kv_client = MagicMock()
    if mock_tokenizer is None:
        mock_tokenizer = MagicMock()

    mlx_kv_mod = MagicMock()
    mlx_kv_mod.MlxKvClient = MagicMock(return_value=mock_kv_client)

    transformers_mod = MagicMock()
    transformers_mod.AutoTokenizer.from_pretrained = MagicMock(
        return_value=mock_tokenizer
    )

    mods: dict[str, Any] = {
        "mlx_kv_client": mlx_kv_mod,
        "transformers": transformers_mod,
    }
    return mock_kv_client, mock_tokenizer, mods


# ---------------------------------------------------------------------------
# Normalizer registry tests
# ---------------------------------------------------------------------------


def test_get_normalizer_qwen3_family() -> None:
    import kai_daemon.__main__ as m

    fn = m._get_normalizer("mlx-community/Qwen3.5-35B-A3B-4bit")
    assert fn is m._strip_model_artifacts


def test_get_normalizer_qwen2_family() -> None:
    import kai_daemon.__main__ as m

    fn = m._get_normalizer("Qwen2-7B-Instruct")
    assert fn is m._strip_model_artifacts


def test_get_normalizer_unknown_returns_generic() -> None:
    import kai_daemon.__main__ as m

    fn = m._get_normalizer("some-unknown-model-v1")
    assert fn is m._strip_generic_artifacts


def test_strip_model_artifacts_role_label_then_think_then_response() -> None:
    import kai_daemon.__main__ as m

    raw = "\nassistant\n<think>\n\n</think>\n\nActual response"
    assert m._strip_model_artifacts(raw) == "Actual response"


def test_strip_model_artifacts_multiline_think_block() -> None:
    import kai_daemon.__main__ as m

    raw = "<think>multi\nline\nthinking</think>\n\nActual response"
    assert m._strip_model_artifacts(raw) == "Actual response"


def test_strip_model_artifacts_clean_input_is_noop() -> None:
    import kai_daemon.__main__ as m

    raw = "Actual response"
    assert m._strip_model_artifacts(raw) == "Actual response"


def test_strip_generic_artifacts_strips_whitespace() -> None:
    import kai_daemon.__main__ as m

    assert m._strip_generic_artifacts("  hello\n") == "hello"


# ---------------------------------------------------------------------------
# Test 1 — tokenizer construction contract
# ---------------------------------------------------------------------------


def test_tokenizer_construction_contract() -> None:
    import kai_daemon.__main__ as m

    mock_kv_client, _, mods = _make_inference_mocks()

    with patch.dict(sys.modules, mods):
        result = m._make_inference_fn()

    mods["mlx_kv_client"].MlxKvClient.assert_called_once_with(m._KV_SOCKET_PATH)
    mock_kv_client.status.assert_called_once_with()
    status = mock_kv_client.status.return_value
    mods["transformers"].AutoTokenizer.from_pretrained.assert_called_once_with(
        status.model,
        local_files_only=True,
    )
    assert callable(result)


# ---------------------------------------------------------------------------
# Test 2 — inference closure call sequence (Shape 2 / plain prompt)
# ---------------------------------------------------------------------------


def test_inference_closure_call_sequence() -> None:
    import kai_daemon.__main__ as m

    mock_kv_client = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = (
        "<|im_start|>user\ntest prompt<|im_end|>"
    )
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "hello"
    mock_kv_client.generate.return_value = iter([4, 5])

    _, _, mods = _make_inference_mocks(mock_tokenizer)
    mods["mlx_kv_client"].MlxKvClient = MagicMock(return_value=mock_kv_client)

    with patch.dict(sys.modules, mods):
        inference_fn = m._make_inference_fn()

    result = inference_fn("test prompt")

    mock_tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "test prompt"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    mock_tokenizer.encode.assert_called_once_with(
        mock_tokenizer.apply_chat_template.return_value,
        add_special_tokens=False,
    )
    mock_kv_client.prefill.assert_called_once_with([1, 2, 3], m._INFERENCE_CACHE_ID)
    mock_kv_client.generate.assert_called_once_with([0], m._INFERENCE_CACHE_ID)
    mock_tokenizer.decode.assert_called_once_with([4, 5], skip_special_tokens=True)
    assert result == "hello"


# ---------------------------------------------------------------------------
# Test 2a — Shape 1 prompt splits into system + user messages
# ---------------------------------------------------------------------------


def test_inference_shape1_extracts_system_and_user() -> None:
    """personal_assistant prompt is split into system + user chat messages."""
    import kai_daemon.__main__ as m

    mock_kv_client = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<formatted>"
    mock_tokenizer.encode.return_value = [10, 20]
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "ok"
    mock_kv_client.generate.return_value = iter([30])

    _, _, mods = _make_inference_mocks(mock_tokenizer)
    mods["mlx_kv_client"].MlxKvClient = MagicMock(return_value=mock_kv_client)

    with patch.dict(sys.modules, mods):
        inference_fn = m._make_inference_fn()

    shape1 = "You are a helpful assistant.\n\nUser: Hello there\n\nResponse:"
    inference_fn(shape1)

    mock_tokenizer.apply_chat_template.assert_called_once_with(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello there"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    mock_tokenizer.encode.assert_called_once_with(
        "<formatted>", add_special_tokens=False
    )


# ---------------------------------------------------------------------------
# Test 2b — Shape 2 plain prompt wraps as single user message
# ---------------------------------------------------------------------------


def test_inference_shape2_wraps_as_user_message() -> None:
    """A plain instructional prompt (seeding etc.) is wrapped as a user message."""
    import kai_daemon.__main__ as m

    mock_kv_client = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<formatted>"
    mock_tokenizer.encode.return_value = [7, 8]
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "ok"
    mock_kv_client.generate.return_value = iter([9])

    _, _, mods = _make_inference_mocks(mock_tokenizer)
    mods["mlx_kv_client"].MlxKvClient = MagicMock(return_value=mock_kv_client)

    with patch.dict(sys.modules, mods):
        inference_fn = m._make_inference_fn()

    plain = "You are a mind coming into being. Write YAML."
    inference_fn(plain)

    mock_tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": plain}],
        tokenize=False,
        add_generation_prompt=True,
    )
    mock_tokenizer.encode.assert_called_once_with(
        "<formatted>", add_special_tokens=False
    )


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


# ---------------------------------------------------------------------------
# Test 7 — --force-reseed deletes daemon_self.yaml
# ---------------------------------------------------------------------------


def test_force_reseed_deletes_daemon_self(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--force-reseed deletes daemon_self.yaml before the engine starts."""
    monkeypatch.setenv("KAI_DATA_DIR", str(tmp_path))

    import kai_daemon.__main__ as m

    # Pre-create daemon_self.yaml to simulate an existing seeding run.
    state_dir = tmp_path / "daemon_state"
    state_dir.mkdir(parents=True)
    daemon_self = state_dir / "daemon_self.yaml"
    daemon_self.write_text("version: 1\n")

    mock_server = MagicMock()
    mock_server.address = ("127.0.0.1", 9999)

    with (
        patch("kai_daemon.__main__.WorkflowRunLogger"),
        patch("kai_daemon.__main__.ActionServer", return_value=mock_server),
        patch("kai_daemon.__main__._build_engine"),
        patch("kai_daemon.conversation_server.run_conversation_server"),
        patch("threading.Event.wait"),
    ):
        m.main(["--force-reseed", "--log-level", "WARNING"])

    assert not daemon_self.exists()
