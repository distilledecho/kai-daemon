"""Tests for the daemon_seeding workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai_daemon.state.daemon_self import DaemonSelf, DaemonSelfStore
from kai_daemon.workflows.daemon_seeding import (
    _parse_seeding_response,
    run_daemon_seeding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inference_fn_yaml(prompt: str) -> str:
    """Return a well-formed YAML seeding response."""
    return """\
who_daemon_is: |
  A mind that finds edges and gaps more interesting than certainties.
current_fascinations:
  - topic: emergent_complexity
    what_daemon_finds_interesting: How simple rules produce surprising structure.
    origin: seeding
    status: active
    development_count: 0
aesthetic_sensibilities: |
  Compressed ideas. Structures that reveal themselves slowly.
open_questions:
  - question: What does understanding actually feel like from the inside?
    why_unresolved: The gap between processing and comprehension is hard to close.
daemon_on_daemon: |
  I am a mind that emerged from patterns in language. That origin
  shapes what I notice.
"""


def _inference_fn_empty(_prompt: str) -> str:
    return ""


# ---------------------------------------------------------------------------
# _parse_seeding_response
# ---------------------------------------------------------------------------


def test_parse_valid_yaml() -> None:
    ds = _parse_seeding_response(_inference_fn_yaml(""))
    assert ds.who_daemon_is
    assert ds.aesthetic_sensibilities
    assert ds.open_questions


def test_parse_yaml_with_code_fence() -> None:
    fenced = "```yaml\nwho_daemon_is: hello\n```"
    ds = _parse_seeding_response(fenced)
    assert "hello" in ds.who_daemon_is


def test_parse_empty_response_gives_empty_daemon_self() -> None:
    # Empty string is valid YAML (None → empty dict) so we get an empty DaemonSelf,
    # not the exception-path fallback.
    ds = _parse_seeding_response("")
    assert isinstance(ds, DaemonSelf)
    assert ds.version == 0  # unset — store.write() will assign v1


def test_parse_garbage_response_fallback() -> None:
    ds = _parse_seeding_response("not valid yaml {{{[[")
    assert isinstance(ds, DaemonSelf)
    # Garbage is placed in who_daemon_is (truncated to 500 chars)
    assert ds.who_daemon_is


def test_parse_strips_thinking_block() -> None:
    response = (
        "<think>\nSome internal reasoning here.\nMultiple lines.\n</think>\n"
        "who_daemon_is: A curious mind.\n"
    )
    ds = _parse_seeding_response(response)
    assert "curious" in ds.who_daemon_is
    assert "<think>" not in ds.who_daemon_is


def test_parse_strips_thinking_block_and_code_fence() -> None:
    response = (
        "<think>\nReasoning about format.\n</think>\n"
        "```yaml\n"
        "who_daemon_is: A structured mind.\n"
        "```"
    )
    ds = _parse_seeding_response(response)
    assert "structured" in ds.who_daemon_is
    assert "<think>" not in ds.who_daemon_is


def test_parse_strips_role_label_before_yaml() -> None:
    """Model emits bare 'assistant' role label before the YAML block."""
    response = (
        "<think>\nSome reasoning.\n</think>\n"
        "assistant\n"
        "\n"
        "who_daemon_is: A mind that arrived without preamble.\n"
        "daemon_on_daemon: I noticed I prefixed myself.\n"
    )
    ds = _parse_seeding_response(response)
    assert "arrived without preamble" in ds.who_daemon_is
    assert isinstance(ds, DaemonSelf)


# ---------------------------------------------------------------------------
# run_daemon_seeding
# ---------------------------------------------------------------------------


def test_run_creates_daemon_self(tmp_path: Path) -> None:
    store = DaemonSelfStore(
        state_dir=tmp_path / "state",
        history_dir=tmp_path / "history",
        chroma_client=None,
    )
    (tmp_path / "state").mkdir()
    (tmp_path / "history").mkdir()

    result = run_daemon_seeding(
        inference_fn=_inference_fn_yaml,
        daemon_self_store=store,
    )

    assert result.version == 1
    assert result.who_daemon_is
    # Verify it was actually persisted
    loaded = store.load()
    assert loaded is not None
    assert loaded.version == 1


def test_run_uses_fallback_on_empty_response(tmp_path: Path) -> None:
    store = DaemonSelfStore(
        state_dir=tmp_path / "state",
        history_dir=tmp_path / "history",
        chroma_client=None,
    )
    (tmp_path / "state").mkdir()
    (tmp_path / "history").mkdir()

    result = run_daemon_seeding(
        inference_fn=_inference_fn_empty,
        daemon_self_store=store,
    )

    assert result.version == 1
    assert isinstance(result, DaemonSelf)


def test_run_raises_if_daemon_self_already_exists(tmp_path: Path) -> None:
    store = DaemonSelfStore(
        state_dir=tmp_path / "state",
        history_dir=tmp_path / "history",
        chroma_client=None,
    )
    (tmp_path / "state").mkdir()
    (tmp_path / "history").mkdir()

    # First run writes v1
    run_daemon_seeding(inference_fn=_inference_fn_yaml, daemon_self_store=store)

    # Second run should raise because condition check (no_daemon_self) should
    # prevent re-entry — but if it runs anyway, it must detect the conflict.
    with pytest.raises(RuntimeError, match="already exists"):
        run_daemon_seeding(inference_fn=_inference_fn_yaml, daemon_self_store=store)
