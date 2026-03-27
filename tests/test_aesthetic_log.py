"""Tests for AestheticLog (§4e)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai_daemon.state.aesthetic_log import AestheticLog

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def log(tmp_path: Path) -> AestheticLog:
    return AestheticLog(path=tmp_path / "aesthetic_log.yaml")


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_empty_log_has_no_entries(log: AestheticLog) -> None:
    assert log.all_entries() == []


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------


def test_append_returns_entry(log: AestheticLog) -> None:
    entry = log.append(thought="a thought", reaction="a reaction")
    assert entry.thought == "a thought"
    assert entry.reaction == "a reaction"
    assert entry.id
    assert entry.timestamp


def test_append_multiple_entries(log: AestheticLog) -> None:
    log.append(thought="t1", reaction="r1")
    log.append(thought="t2", reaction="r2")
    entries = log.all_entries()
    assert len(entries) == 2
    assert entries[0].thought == "t1"
    assert entries[1].thought == "t2"


def test_entries_are_append_only(log: AestheticLog) -> None:
    log.append(thought="original", reaction="reaction")
    first_id = log.all_entries()[0].id
    log.append(thought="second", reaction="second reaction")
    assert log.all_entries()[0].id == first_id


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_append_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "aesthetic_log.yaml"
    log1 = AestheticLog(path=path)
    log1.append(thought="persisted", reaction="persisted reaction")

    log2 = AestheticLog(path=path)
    assert len(log2.all_entries()) == 1
    assert log2.all_entries()[0].thought == "persisted"


# ---------------------------------------------------------------------------
# Corrupt file
# ---------------------------------------------------------------------------


def test_corrupt_file_warns_and_starts_empty(tmp_path: Path) -> None:
    path = tmp_path / "aesthetic_log.yaml"
    path.write_text("not: valid: yaml: [\n")
    with pytest.warns(UserWarning, match="could not be parsed"):
        bad_log = AestheticLog(path=path)
    assert bad_log.all_entries() == []
