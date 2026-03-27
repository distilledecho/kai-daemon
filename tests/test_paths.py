"""Tests for _paths path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai_daemon.state._paths import (
    daemon_relational_history_dir,
    daemon_self_history_dir,
    daemon_state_dir,
    data_dir,
    logs_dir,
    memory_queue_dir,
    pickup_notes_dir,
    threads_dir,
)


def test_data_dir_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAI_DATA_DIR", str(tmp_path / "custom_data"))
    result = data_dir()
    assert result == tmp_path / "custom_data"


def test_data_dir_default_without_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAI_DATA_DIR", raising=False)
    result = data_dir()
    assert result.name == "data"


def test_daemon_state_dir_creates_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAI_DATA_DIR", str(tmp_path / "d"))
    result = daemon_state_dir()
    assert result.exists()
    assert result.is_dir()
    assert result.name == "daemon_state"


def test_logs_dir_creates_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAI_DATA_DIR", str(tmp_path / "d"))
    result = logs_dir()
    assert result.exists()
    assert result.name == "logs"


def test_threads_dir_creates_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAI_DATA_DIR", str(tmp_path / "d"))
    result = threads_dir()
    assert result.exists()
    assert result.name == "threads"


def test_pickup_notes_dir_creates_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAI_DATA_DIR", str(tmp_path / "d"))
    result = pickup_notes_dir()
    assert result.exists()
    assert result.name == "pickup_notes"


def test_memory_queue_dir_creates_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAI_DATA_DIR", str(tmp_path / "d"))
    result = memory_queue_dir()
    assert result.exists()
    assert result.name == "memory_queue"


def test_daemon_self_history_dir_creates_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAI_DATA_DIR", str(tmp_path / "d"))
    result = daemon_self_history_dir()
    assert result.exists()
    assert result.name == "daemon_self_history"


def test_daemon_relational_history_dir_creates_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAI_DATA_DIR", str(tmp_path / "d"))
    result = daemon_relational_history_dir()
    assert result.exists()
    assert result.name == "daemon_relational_history"
