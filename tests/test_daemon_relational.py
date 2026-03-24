"""Tests for DAEMON_RELATIONAL versioned store (§4b)."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from unittest.mock import MagicMock

from kai_daemon.state.daemon_relational import (
    TOKEN_BUDGET,
    DaemonRelational,
    DaemonRelationalStore,
    FollowUpStyle,
    OpenLoop,
    OpenLoopType,
    _count_tokens,
    _truncate_overflow,
    _yaml_text,
)
from kai_daemon.state.daemon_self import DaemonSelf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path: Path, chroma: object | None = None) -> DaemonRelationalStore:
    state = tmp_path / "state"
    history = tmp_path / "history"
    state.mkdir()
    history.mkdir()
    return DaemonRelationalStore(
        state_dir=state, history_dir=history, chroma_client=chroma
    )


def _minimal() -> DaemonRelational:
    return DaemonRelational(how_user_thinks="test")


def _over_budget() -> DaemonRelational:
    """Return a DaemonRelational whose serialized token count exceeds TOKEN_BUDGET."""
    return DaemonRelational(
        how_user_thinks="short",
        overflow="word " * (TOKEN_BUDGET + 50),
    )


# ---------------------------------------------------------------------------
# Basic load / write
# ---------------------------------------------------------------------------


def test_load_returns_none_when_no_file(tmp_path: Path) -> None:
    assert _store(tmp_path).load() is None


def test_write_creates_current_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(_minimal())
    current = tmp_path / "state" / "daemon_relational.yaml"
    assert current.exists()


def test_write_returns_stamped_document(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.write(_minimal())
    assert result.version == 1
    assert result.timestamp  # non-empty


def test_write_increments_version(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(_minimal())
    result = store.write(_minimal())
    assert result.version == 2


def test_load_round_trips_content(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(
        DaemonRelational(
            how_user_thinks="bottom-up thinker",
            overflow="extra context",
        )
    )
    loaded = store.load()
    assert loaded is not None
    assert loaded.how_user_thinks == "bottom-up thinker"
    assert loaded.overflow == "extra context"


# ---------------------------------------------------------------------------
# Versioning — never overwrite in place
# ---------------------------------------------------------------------------


def test_current_replaced_on_second_write(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(DaemonRelational(how_user_thinks="v1"))
    store.write(DaemonRelational(how_user_thinks="v2"))
    loaded = store.load()
    assert loaded is not None
    assert loaded.how_user_thinks == "v2"
    assert loaded.version == 2


def test_history_accumulates_on_writes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(3):
        store.write(_minimal())
    history = store.history()
    assert len(history) == 2
    assert [d.version for d in history] == [1, 2]


def test_history_file_never_overwritten(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(DaemonRelational(how_user_thinks="original v1"))
    store.write(_minimal())
    store.write(_minimal())
    v1_path = tmp_path / "history" / "v1.yaml"
    import yaml

    content = yaml.safe_load(v1_path.read_text())
    assert content["how_user_thinks"] == "original v1"


def test_history_returns_sorted_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(4):
        store.write(_minimal())
    versions = [d.version for d in store.history()]
    assert versions == sorted(versions)


def test_history_empty_before_second_write(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(_minimal())
    assert store.history() == []


# ---------------------------------------------------------------------------
# Token budget — warnings
# ---------------------------------------------------------------------------


def test_load_warns_when_over_budget(tmp_path: Path) -> None:
    over = _over_budget()
    import yaml

    stamped = over.model_copy(update={"version": 1})
    current = tmp_path / "state" / "daemon_relational.yaml"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text(yaml.dump(stamped.model_dump(mode="json"), allow_unicode=True))
    store = DaemonRelationalStore(
        state_dir=tmp_path / "state",
        history_dir=tmp_path / "history",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.load()
    messages = [str(w.message) for w in caught]
    assert any("DAEMON_RELATIONAL" in m and "budget" in m for m in messages)


def test_write_truncates_overflow_when_over_budget(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.write(_over_budget())
    assert _count_tokens(_yaml_text(result)) <= TOKEN_BUDGET


def test_write_warns_if_still_over_after_truncation(tmp_path: Path) -> None:
    big_base = DaemonRelational(
        how_user_thinks="word " * (TOKEN_BUDGET + 100),
        overflow="small",
    )
    store = _store(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.write(big_base)
    messages = [str(w.message) for w in caught]
    assert any("after overflow truncation" in m for m in messages)


# ---------------------------------------------------------------------------
# Token budget — truncation helper
# ---------------------------------------------------------------------------


def test_truncate_overflow_removes_excess() -> None:
    over = _over_budget()
    truncated = _truncate_overflow(over)
    assert _count_tokens(_yaml_text(truncated)) <= TOKEN_BUDGET


def test_truncate_overflow_clears_field_when_insufficient() -> None:
    big = DaemonRelational(
        how_user_thinks="word " * (TOKEN_BUDGET + 50),
        overflow="tiny",
    )
    result = _truncate_overflow(big)
    assert result.overflow == ""


def test_truncate_overflow_is_noop_under_budget() -> None:
    under = DaemonRelational(how_user_thinks="short", overflow="also short")
    assert _truncate_overflow(under) == under


# ---------------------------------------------------------------------------
# ChromaDB integration
# ---------------------------------------------------------------------------


def test_chroma_add_called_on_write(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    store = _store(tmp_path, chroma=mock_client)
    store.write(_minimal())

    mock_collection.add.assert_called_once()
    assert mock_collection.add.call_args.kwargs["ids"] == ["v1"]


def test_chroma_add_called_with_correct_collection_name(tmp_path: Path) -> None:
    from kai_daemon.state._chroma import DAEMON_RELATIONAL_COLLECTION

    mock_client = MagicMock()
    _store(tmp_path, chroma=mock_client)
    mock_client.get_or_create_collection.assert_called_once_with(
        DAEMON_RELATIONAL_COLLECTION
    )


def test_chroma_failure_does_not_prevent_write(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.add.side_effect = RuntimeError("server down")
    mock_client.get_or_create_collection.return_value = mock_collection

    store = _store(tmp_path, chroma=mock_client)
    store.write(DaemonRelational(how_user_thinks="persisted despite chroma failure"))

    # Version file must exist and contain the correct content.
    loaded = store.load()
    assert loaded is not None
    assert loaded.version == 1
    assert loaded.how_user_thinks == "persisted despite chroma failure"


def test_no_chroma_write_when_client_is_none(tmp_path: Path) -> None:
    store = _store(tmp_path, chroma=None)
    store.write(_minimal())


# ---------------------------------------------------------------------------
# Structural separation from DAEMON_SELF
# ---------------------------------------------------------------------------


def test_daemon_relational_store_has_no_self_return_type() -> None:
    """DaemonRelationalStore must not expose any method returning DaemonSelf."""
    for _name, method in inspect.getmembers(
        DaemonRelationalStore, predicate=inspect.isfunction
    ):
        hints = method.__annotations__
        for hint_value in hints.values():
            assert hint_value is not DaemonSelf, (
                f"DaemonRelationalStore.{_name} has a DaemonSelf annotation — "
                "this violates the structural separation constraint"
            )


def test_daemon_relational_store_has_no_self_parameter() -> None:
    """DaemonRelationalStore must not accept DaemonSelf as a parameter."""
    for _name, method in inspect.getmembers(
        DaemonRelationalStore, predicate=inspect.isfunction
    ):
        sig = inspect.signature(method)
        for param in sig.parameters.values():
            assert param.annotation is not DaemonSelf, (
                f"DaemonRelationalStore.{_name} accepts DaemonSelf — "
                "this violates the structural separation constraint"
            )


# ---------------------------------------------------------------------------
# Model fields
# ---------------------------------------------------------------------------


def test_open_loop_model_defaults() -> None:
    loop = OpenLoop(
        content="finish the report",
        type=OpenLoopType.INTENTION,
        follow_up_style=FollowUpStyle.CHECK_IN,
        follow_up_after="3d",
    )
    assert loop.resolved is False
    assert loop.last_surfaced is None
    assert loop.id  # auto-generated UUID


def test_daemon_relational_with_open_loops_round_trips(tmp_path: Path) -> None:
    dr = DaemonRelational(
        how_user_thinks="systematic",
        open_loops=[
            OpenLoop(
                content="write the spec",
                type=OpenLoopType.PLAN,
                follow_up_style=FollowUpStyle.CHALLENGE,
                follow_up_after="1w",
            )
        ],
    )
    store = _store(tmp_path)
    store.write(dr)
    loaded = store.load()
    assert loaded is not None
    assert len(loaded.open_loops) == 1
    assert loaded.open_loops[0].content == "write the spec"
    assert loaded.open_loops[0].type == OpenLoopType.PLAN
