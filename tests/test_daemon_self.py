"""Tests for DAEMON_SELF versioned store (§4a)."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from unittest.mock import MagicMock

from kai_daemon.state.daemon_self import (
    TOKEN_BUDGET,
    DaemonSelf,
    DaemonSelfStore,
    Fascination,
    FascinationOrigin,
    FascinationStatus,
    OpenQuestion,
    _count_tokens,
    _truncate_overflow,
    _yaml_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path: Path, chroma: object | None = None) -> DaemonSelfStore:
    state = tmp_path / "state"
    history = tmp_path / "history"
    state.mkdir()
    history.mkdir()
    return DaemonSelfStore(state_dir=state, history_dir=history, chroma_client=chroma)


def _minimal() -> DaemonSelf:
    return DaemonSelf(who_daemon_is="test")


def _over_budget() -> DaemonSelf:
    """Return a DaemonSelf whose serialized token count exceeds TOKEN_BUDGET."""
    # Overflow field is the truncation target so we overfill it.
    return DaemonSelf(
        who_daemon_is="short",
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
    current = tmp_path / "state" / "daemon_self.yaml"
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
    store.write(DaemonSelf(who_daemon_is="persistent identity", overflow="extra"))
    loaded = store.load()
    assert loaded is not None
    assert loaded.who_daemon_is == "persistent identity"
    assert loaded.overflow == "extra"


# ---------------------------------------------------------------------------
# Versioning — never overwrite in place
# ---------------------------------------------------------------------------


def test_current_replaced_on_second_write(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(DaemonSelf(who_daemon_is="v1"))
    store.write(DaemonSelf(who_daemon_is="v2"))
    loaded = store.load()
    assert loaded is not None
    assert loaded.who_daemon_is == "v2"
    assert loaded.version == 2


def test_history_accumulates_on_writes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(3):
        store.write(_minimal())
    history = store.history()
    assert len(history) == 2  # v1 and v2 are in history; v3 is current
    assert [d.version for d in history] == [1, 2]


def test_history_file_never_overwritten(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(DaemonSelf(who_daemon_is="original v1"))
    store.write(_minimal())
    # Write a third version; v1 history file must not be overwritten
    store.write(_minimal())
    v1_path = tmp_path / "history" / "v1.yaml"
    import yaml

    content = yaml.safe_load(v1_path.read_text())
    assert content["who_daemon_is"] == "original v1"


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
    store = _store(tmp_path)
    # Write directly so no truncation occurs during write for this assertion
    over = _over_budget()
    # Bypass store.write truncation by writing the raw file
    import yaml

    stamped = over.model_copy(update={"version": 1})
    current = tmp_path / "state" / "daemon_self.yaml"
    current.write_text(yaml.dump(stamped.model_dump(mode="json"), allow_unicode=True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.load()
    messages = [str(w.message) for w in caught]
    assert any("DAEMON_SELF" in m and "budget" in m for m in messages)


def test_write_truncates_overflow_when_over_budget(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.write(_over_budget())
    token_count = _count_tokens(_yaml_text(result))
    assert token_count <= TOKEN_BUDGET


def test_write_warns_if_still_over_after_truncation(tmp_path: Path) -> None:
    """If overflow alone can't absorb the excess, a warning must be emitted."""
    # Make the non-overflow fields alone exceed the budget.
    big_base = DaemonSelf(
        who_daemon_is="word " * (TOKEN_BUDGET + 100),
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
    """When overflow can't hold the excess, it is cleared entirely."""
    big = DaemonSelf(
        who_daemon_is="word " * (TOKEN_BUDGET + 50),
        overflow="tiny",
    )
    result = _truncate_overflow(big)
    assert result.overflow == ""


def test_truncate_overflow_is_noop_under_budget() -> None:
    under = DaemonSelf(who_daemon_is="short", overflow="also short")
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
    call_kwargs = mock_collection.add.call_args
    assert call_kwargs.kwargs["ids"] == ["v1"]


def test_chroma_add_called_with_correct_collection_name(tmp_path: Path) -> None:
    from kai_daemon.state._chroma import DAEMON_SELF_COLLECTION

    mock_client = MagicMock()
    _store(tmp_path, chroma=mock_client)
    mock_client.get_or_create_collection.assert_called_once_with(DAEMON_SELF_COLLECTION)


def test_chroma_failure_does_not_prevent_write(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.add.side_effect = RuntimeError("server down")
    mock_client.get_or_create_collection.return_value = mock_collection

    store = _store(tmp_path, chroma=mock_client)
    store.write(DaemonSelf(who_daemon_is="persisted despite chroma failure"))

    # Version file must exist and contain the correct content.
    loaded = store.load()
    assert loaded is not None
    assert loaded.version == 1
    assert loaded.who_daemon_is == "persisted despite chroma failure"


def test_no_chroma_write_when_client_is_none(tmp_path: Path) -> None:
    store = _store(tmp_path, chroma=None)
    # Should not raise anything
    store.write(_minimal())


# ---------------------------------------------------------------------------
# Structural separation from DAEMON_RELATIONAL
# ---------------------------------------------------------------------------


def test_daemon_self_store_has_no_relational_type_references() -> None:
    """No annotation in DaemonSelfStore may reference DaemonRelational.

    Under ``from __future__ import annotations`` (PEP 563), annotations are
    stored as strings, not as live type objects.  Identity checks against the
    class (``hint is not DaemonRelational``) always pass regardless of what is
    written in source.  We check the string representation directly instead,
    which correctly catches return types, parameter types, and nested forms
    such as ``Optional[DaemonRelational]``.
    """
    for name, method in inspect.getmembers(
        DaemonSelfStore, predicate=inspect.isfunction
    ):
        for annotation_str in method.__annotations__.values():
            assert "DaemonRelational" not in str(annotation_str), (
                f"DaemonSelfStore.{name} references DaemonRelational — "
                "this violates the structural separation constraint"
            )


# ---------------------------------------------------------------------------
# Model fields
# ---------------------------------------------------------------------------


def test_fascination_model() -> None:
    f = Fascination(
        topic="emergence",
        what_daemon_finds_interesting="patterns in noise",
        origin=FascinationOrigin.SEEDING,
    )
    assert f.status == FascinationStatus.ACTIVE
    assert f.development_count == 0
    assert f.connection_to_user is None


def test_open_question_model() -> None:
    q = OpenQuestion(question="why?", why_unresolved="unclear")
    assert q.user_has_touched_this is False


def test_daemon_self_with_fascinations_round_trips(tmp_path: Path) -> None:
    ds = DaemonSelf(
        who_daemon_is="curious entity",
        current_fascinations=[
            Fascination(
                topic="recursion",
                what_daemon_finds_interesting="self-reference",
                origin=FascinationOrigin.CONVERSATION,
            )
        ],
    )
    store = _store(tmp_path)
    store.write(ds)
    loaded = store.load()
    assert loaded is not None
    assert len(loaded.current_fascinations) == 1
    assert loaded.current_fascinations[0].topic == "recursion"
