"""Tests for _chroma factory helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import kai_daemon.state._chroma as chroma_mod
from kai_daemon.state._chroma import _load_config, make_chroma_client


def test_load_config_returns_empty_dict_when_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chroma_mod, "_CONFIG_PATH", tmp_path / "nonexistent.yaml")
    assert _load_config() == {}


def test_load_config_reads_yaml_when_file_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "server.yaml"
    cfg.write_text("connection:\n  host: myhost\n  port: 1234\n")
    monkeypatch.setattr(chroma_mod, "_CONFIG_PATH", cfg)
    result = _load_config()
    assert result["connection"]["host"] == "myhost"


def test_make_chroma_client_uses_config_host_and_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "server.yaml"
    cfg.write_text("connection:\n  host: testhost\n  port: 9876\n")
    monkeypatch.setattr(chroma_mod, "_CONFIG_PATH", cfg)

    mock_client = MagicMock()
    with patch("chromadb.HttpClient", return_value=mock_client) as mock_http:
        result = make_chroma_client()
        mock_http.assert_called_once_with(host="testhost", port=9876)
        assert result is mock_client


def test_make_chroma_client_defaults_when_no_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chroma_mod, "_CONFIG_PATH", tmp_path / "missing.yaml")

    mock_client = MagicMock()
    with patch("chromadb.HttpClient", return_value=mock_client) as mock_http:
        make_chroma_client()
        mock_http.assert_called_once_with(host="localhost", port=8765)
