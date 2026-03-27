"""Tests for daemon_integration workflow (§7c, §7d)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai_daemon.state.aesthetic_log import AestheticLog
from kai_daemon.state.daemon_self import (
    DaemonSelf,
    DaemonSelfStore,
    Fascination,
    FascinationOrigin,
    FascinationStatus,
)
from kai_daemon.workflows.daemon_integration import (
    LIFECYCLE_CHECK_THRESHOLD,
    IntegrationRoute,
    _parse_response,
    _parse_route,
    daemon_integration,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 27, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def ds_store(tmp_path: Path) -> DaemonSelfStore:
    state = tmp_path / "state"
    history = tmp_path / "history"
    state.mkdir()
    history.mkdir()
    return DaemonSelfStore(
        state_dir=state,
        history_dir=history,
        chroma_client=None,
    )


@pytest.fixture
def aesthetic(tmp_path: Path) -> AestheticLog:
    return AestheticLog(path=tmp_path / "aesthetic_log.yaml")


def make_fascination(
    topic: str,
    count: int = 0,
    status: FascinationStatus = FascinationStatus.ACTIVE,
) -> Fascination:
    return Fascination(
        topic=topic,
        what_daemon_finds_interesting=f"Interesting things about {topic}",
        origin=FascinationOrigin.SEEDING,
        development_count=count,
        status=status,
    )


def make_ds(*fascination_topics: str) -> DaemonSelf:
    return DaemonSelf(
        who_daemon_is="Test daemon",
        current_fascinations=[make_fascination(t) for t in fascination_topics],
    )


# ---------------------------------------------------------------------------
# _parse_response / _parse_route helpers
# ---------------------------------------------------------------------------


def test_parse_response_extracts_fields() -> None:
    response = "ROUTE: new_fascination\nTOPIC: entropy\nINTERESTING: It's everywhere"
    fields = _parse_response(response)
    assert fields["ROUTE"] == "new_fascination"
    assert fields["TOPIC"] == "entropy"
    assert fields["INTERESTING"] == "It's everywhere"


def test_parse_route_unknown_defaults_to_inert() -> None:
    fields = {"ROUTE": "something_unknown"}
    assert _parse_route(fields) == IntegrationRoute.INERT


def test_parse_route_missing_defaults_to_inert() -> None:
    assert _parse_route({}) == IntegrationRoute.INERT


# ---------------------------------------------------------------------------
# new_fascination route
# ---------------------------------------------------------------------------


def test_new_fascination_creates_entry(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    ds_store.write(make_ds())

    def inference_fn(prompt: str) -> str:
        return (
            "ROUTE: new_fascination\nTOPIC: recursion\n"
            "INTERESTING: Self-referential systems"
        )

    result = daemon_integration(
        "thinking about recursion",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert result.route == IntegrationRoute.NEW_FASCINATION
    assert result.fascination_topic == "recursion"
    assert result.lifecycle_promoted is False

    loaded = ds_store.load()
    assert loaded is not None
    topics = [f.topic for f in loaded.current_fascinations]
    assert "recursion" in topics


def test_new_fascination_missing_topic_routes_as_inert(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    ds_store.write(make_ds())

    def inference_fn(prompt: str) -> str:
        return "ROUTE: new_fascination\nINTERESTING: something"

    result = daemon_integration(
        "vague thought",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert result.route == IntegrationRoute.INERT


def test_new_fascination_with_no_existing_store(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    """Works even when no DAEMON_SELF exists yet."""

    def inference_fn(prompt: str) -> str:
        return (
            "ROUTE: new_fascination\n"
            "TOPIC: emergence\n"
            "INTERESTING: Complexity from simplicity"
        )

    result = daemon_integration(
        "patterns emerging",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert result.route == IntegrationRoute.NEW_FASCINATION
    assert result.fascination_topic == "emergence"


# ---------------------------------------------------------------------------
# develops_existing route
# ---------------------------------------------------------------------------


def test_develops_existing_increments_count(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    ds_store.write(make_ds("recursion", "emergence"))

    def inference_fn(prompt: str) -> str:
        return "ROUTE: develops_existing\nFASCINATION: recursion"

    result = daemon_integration(
        "more about recursion",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert result.route == IntegrationRoute.DEVELOPS_EXISTING
    assert result.fascination_topic == "recursion"

    loaded = ds_store.load()
    assert loaded is not None
    rec_f = next(f for f in loaded.current_fascinations if f.topic == "recursion")
    assert rec_f.development_count == 1
    assert rec_f.last_developed is not None


def test_develops_existing_case_insensitive_match(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    ds_store.write(make_ds("Recursion"))

    def inference_fn(prompt: str) -> str:
        return "ROUTE: develops_existing\nFASCINATION: recursion"

    result = daemon_integration(
        "recursive thought",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert result.fascination_topic == "Recursion"


def test_develops_existing_no_active_fascinations_routes_inert(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    ds = make_ds()
    ds = ds.model_copy(
        update={
            "current_fascinations": [
                make_fascination("old", status=FascinationStatus.SUSPENDED)
            ]
        }
    )
    ds_store.write(ds)

    def inference_fn(prompt: str) -> str:
        return "ROUTE: develops_existing\nFASCINATION: old"

    result = daemon_integration(
        "developing old",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert result.route == IntegrationRoute.INERT


# ---------------------------------------------------------------------------
# Lifecycle check at development_count >= 3
# ---------------------------------------------------------------------------


def test_lifecycle_check_triggers_at_threshold(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    ds = DaemonSelf(
        who_daemon_is="test",
        current_fascinations=[make_fascination("recursion", count=2)],
    )
    ds_store.write(ds)

    call_log: list[str] = []

    def inference_fn(prompt: str) -> str:
        call_log.append(prompt)
        if "crystallised" in prompt:
            return "KEEP"
        return "ROUTE: develops_existing\nFASCINATION: recursion"

    result = daemon_integration(
        "deeper recursion",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert len(call_log) == 2
    assert result.route == IntegrationRoute.DEVELOPS_EXISTING


def test_lifecycle_check_not_triggered_below_threshold(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    ds = DaemonSelf(
        who_daemon_is="test",
        current_fascinations=[make_fascination("recursion", count=1)],
    )
    ds_store.write(ds)

    call_log: list[str] = []

    def inference_fn(prompt: str) -> str:
        call_log.append(prompt)
        return "ROUTE: develops_existing\nFASCINATION: recursion"

    daemon_integration(
        "more recursion",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert len(call_log) == 1


def test_lifecycle_check_threshold_constant() -> None:
    assert LIFECYCLE_CHECK_THRESHOLD == 3


def test_lifecycle_promote_updates_status_and_open_questions(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    ds = DaemonSelf(
        who_daemon_is="test",
        current_fascinations=[make_fascination("recursion", count=2)],
    )
    ds_store.write(ds)

    def inference_fn(prompt: str) -> str:
        if "crystallised" in prompt:
            return "PROMOTE"
        return "ROUTE: develops_existing\nFASCINATION: recursion"

    result = daemon_integration(
        "deep recursion",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert result.lifecycle_promoted is True

    loaded = ds_store.load()
    assert loaded is not None
    rec_f = next(f for f in loaded.current_fascinations if f.topic == "recursion")
    assert rec_f.status == FascinationStatus.PROMOTED_TO_OPEN_QUESTION
    assert any(oq.question == "recursion" for oq in loaded.open_questions)


# ---------------------------------------------------------------------------
# aesthetic_reaction route
# ---------------------------------------------------------------------------


def test_aesthetic_reaction_writes_to_log(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    ds_store.write(make_ds())

    def inference_fn(prompt: str) -> str:
        return "ROUTE: aesthetic_reaction\nREACTION: The pattern is beautiful"

    result = daemon_integration(
        "noticing beauty",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert result.route == IntegrationRoute.AESTHETIC_REACTION
    assert result.fascination_topic is None

    entries = aesthetic.all_entries()
    assert len(entries) == 1
    assert entries[0].reaction == "The pattern is beautiful"


def test_aesthetic_reaction_does_not_modify_daemon_self(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    ds = make_ds("recursion")
    ds_store.write(ds)
    before = ds_store.load()
    assert before is not None
    v_before = before.version

    def inference_fn(prompt: str) -> str:
        return "ROUTE: aesthetic_reaction\nREACTION: Lovely"

    daemon_integration(
        "aesthetic",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    after = ds_store.load()
    assert after is not None
    assert after.version == v_before


# ---------------------------------------------------------------------------
# inert route
# ---------------------------------------------------------------------------


def test_inert_no_state_changes(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    ds = make_ds("recursion")
    ds_store.write(ds)

    def inference_fn(prompt: str) -> str:
        return "ROUTE: inert"

    result = daemon_integration(
        "rambling",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert result.route == IntegrationRoute.INERT
    assert result.fascination_topic is None
    assert aesthetic.all_entries() == []


# ---------------------------------------------------------------------------
# Result carries thought_content
# ---------------------------------------------------------------------------


def test_result_carries_thought_content(
    ds_store: DaemonSelfStore, aesthetic: AestheticLog
) -> None:
    ds_store.write(make_ds())

    def inference_fn(prompt: str) -> str:
        return "ROUTE: inert"

    result = daemon_integration(
        "the original thought",
        daemon_self_store=ds_store,
        aesthetic_log=aesthetic,
        inference_fn=inference_fn,
        now=_NOW,
    )
    assert result.thought_content == "the original thought"
