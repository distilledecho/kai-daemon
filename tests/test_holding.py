"""Tests for the holding store (§4d)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from kai_daemon.state._types import EpistemicOrigin
from kai_daemon.state.holding import (
    HoldingItem,
    HoldingStore,
    HoldingType,
    RegisterNeeded,
    Urgency,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> HoldingStore:
    return HoldingStore(path=tmp_path / "holding.yaml")


def _item(
    content: str = "noticed something",
    type: HoldingType = HoldingType.OBSERVATION,
    relevance_trigger: str = "when discussing the topic",
    register_needed: RegisterNeeded = RegisterNeeded.ANY,
    urgency: Urgency = Urgency.LOW,
    source_workflow: str = "test_workflow",
    epistemic_origin: EpistemicOrigin = EpistemicOrigin.INTERNAL,
    contradiction_id: str | None = None,
    created: str | None = None,
) -> HoldingItem:
    kwargs: dict[str, Any] = {
        "content": content,
        "type": type,
        "relevance_trigger": relevance_trigger,
        "register_needed": register_needed,
        "urgency": urgency,
        "source_workflow": source_workflow,
        "epistemic_origin": epistemic_origin,
        "contradiction_id": contradiction_id,
    }
    if created is not None:
        kwargs["created"] = created
    return HoldingItem(**kwargs)


# ---------------------------------------------------------------------------
# Validation rule — reasoned_disagreement requires contradiction_id
# ---------------------------------------------------------------------------


class TestReasonedDisagreementValidation:
    """Three spec-required test cases for the validation rule."""

    def test_a_reasoned_disagreement_with_valid_contradiction_id_succeeds(self) -> None:
        """(a) type: reasoned_disagreement + valid contradiction_id → succeeds."""
        item = _item(
            type=HoldingType.REASONED_DISAGREEMENT,
            contradiction_id="550e8400-e29b-41d4-a716-446655440000",
        )
        assert item.type == HoldingType.REASONED_DISAGREEMENT
        assert item.contradiction_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_b_reasoned_disagreement_without_contradiction_id_raises(self) -> None:
        """(b) reasoned_disagreement + contradiction_id: null → ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="contradiction_id"):
            _item(
                type=HoldingType.REASONED_DISAGREEMENT,
                contradiction_id=None,
            )

    def test_c_observation_without_contradiction_id_succeeds(self) -> None:
        """(c) type: observation + contradiction_id: null → succeeds."""
        item = _item(type=HoldingType.OBSERVATION, contradiction_id=None)
        assert item.type == HoldingType.OBSERVATION
        assert item.contradiction_id is None


# ---------------------------------------------------------------------------
# HoldingStore — write / read / list
# ---------------------------------------------------------------------------


class TestHoldingStoreWriteRead:
    def test_write_and_read_roundtrip(self, store: HoldingStore) -> None:
        item = _item(content="a thing I noticed")
        stored = store.write(item)
        assert stored.id == item.id
        assert store.read(item.id).content == "a thing I noticed"

    def test_write_duplicate_raises(self, store: HoldingStore) -> None:
        item = _item()
        store.write(item)
        with pytest.raises(ValueError, match="already exists"):
            store.write(item)

    def test_read_missing_raises(self, store: HoldingStore) -> None:
        with pytest.raises(KeyError):
            store.read("nonexistent-id")

    def test_list_all_empty(self, store: HoldingStore) -> None:
        assert store.list_all() == []

    def test_list_all_returns_all(self, store: HoldingStore) -> None:
        store.write(_item())
        store.write(_item())
        assert len(store.list_all()) == 2

    def test_list_unsurfaced_excludes_discharged(self, store: HoldingStore) -> None:
        a = _item()
        b = _item()
        store.write(a)
        store.write(b)
        store.discharge(a.id)
        unsurfaced = store.list_unsurfaced()
        assert len(unsurfaced) == 1
        assert unsurfaced[0].id == b.id


# ---------------------------------------------------------------------------
# Persistence — reload from disk
# ---------------------------------------------------------------------------


class TestHoldingStorePersistence:
    def test_persists_across_reload(self, tmp_path: Path) -> None:
        p = tmp_path / "holding.yaml"
        store = HoldingStore(path=p)
        item = _item(content="persisted content")
        store.write(item)

        store2 = HoldingStore(path=p)
        loaded = store2.read(item.id)
        assert loaded.content == "persisted content"
        assert loaded.type == HoldingType.OBSERVATION

    def test_persists_reasoned_disagreement_with_contradiction_id(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path / "holding.yaml"
        store = HoldingStore(path=p)
        item = _item(
            type=HoldingType.REASONED_DISAGREEMENT,
            contradiction_id="abc-123",
        )
        store.write(item)

        store2 = HoldingStore(path=p)
        loaded = store2.read(item.id)
        assert loaded.type == HoldingType.REASONED_DISAGREEMENT
        assert loaded.contradiction_id == "abc-123"


# ---------------------------------------------------------------------------
# Discharge
# ---------------------------------------------------------------------------


class TestDischarge:
    def test_discharge_sets_surfaced(self, store: HoldingStore) -> None:
        item = _item()
        store.write(item)
        discharged = store.discharge(item.id, discharge_notes="discussed in session")
        assert discharged.surfaced is not None
        assert discharged.discharge_notes == "discussed in session"

    def test_discharge_without_notes(self, store: HoldingStore) -> None:
        item = _item()
        store.write(item)
        discharged = store.discharge(item.id)
        assert discharged.surfaced is not None
        assert discharged.discharge_notes is None

    def test_discharge_twice_raises(self, store: HoldingStore) -> None:
        item = _item()
        store.write(item)
        store.discharge(item.id)
        with pytest.raises(ValueError, match="already discharged"):
            store.discharge(item.id)

    def test_discharge_persists(self, tmp_path: Path) -> None:
        p = tmp_path / "holding.yaml"
        store = HoldingStore(path=p)
        item = _item()
        store.write(item)
        store.discharge(item.id, discharge_notes="done")

        store2 = HoldingStore(path=p)
        loaded = store2.read(item.id)
        assert loaded.surfaced is not None
        assert loaded.discharge_notes == "done"


# ---------------------------------------------------------------------------
# Forced surface — urgency thresholds
# ---------------------------------------------------------------------------


class TestForcedSurface:
    def _old(self, urgency: Urgency, days_ago: int) -> HoldingItem:
        created = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
        return _item(urgency=urgency, created=created)

    def test_high_urgency_forced_after_7_days(self, store: HoldingStore) -> None:
        item = self._old(Urgency.HIGH, days_ago=8)
        store.write(item)
        assert any(i.id == item.id for i in store.forced_surface())

    def test_high_urgency_not_forced_before_7_days(self, store: HoldingStore) -> None:
        item = self._old(Urgency.HIGH, days_ago=6)
        store.write(item)
        assert not any(i.id == item.id for i in store.forced_surface())

    def test_medium_urgency_forced_after_21_days(self, store: HoldingStore) -> None:
        item = self._old(Urgency.MEDIUM, days_ago=22)
        store.write(item)
        assert any(i.id == item.id for i in store.forced_surface())

    def test_medium_urgency_not_forced_before_21_days(
        self, store: HoldingStore
    ) -> None:
        item = self._old(Urgency.MEDIUM, days_ago=20)
        store.write(item)
        assert not any(i.id == item.id for i in store.forced_surface())

    def test_low_urgency_never_forced(self, store: HoldingStore) -> None:
        item = self._old(Urgency.LOW, days_ago=365)
        store.write(item)
        assert not any(i.id == item.id for i in store.forced_surface())

    def test_discharged_items_excluded_from_forced_surface(
        self, store: HoldingStore
    ) -> None:
        item = self._old(Urgency.HIGH, days_ago=8)
        store.write(item)
        store.discharge(item.id)
        assert not any(i.id == item.id for i in store.forced_surface())

    def test_forced_surface_at_exact_threshold(self, store: HoldingStore) -> None:
        """Boundary: exactly at threshold (>= timedelta) is forced."""
        now = datetime.now(UTC)
        created = (now - timedelta(days=7)).isoformat()
        item = _item(urgency=Urgency.HIGH, created=created)
        store.write(item)
        assert any(i.id == item.id for i in store.forced_surface(now=now))

    def test_forced_surface_accepts_custom_now(self, store: HoldingStore) -> None:
        item = _item(urgency=Urgency.HIGH)
        store.write(item)
        future = datetime.now(UTC) + timedelta(days=10)
        assert any(i.id == item.id for i in store.forced_surface(now=future))


# ---------------------------------------------------------------------------
# All holding types round-trip
# ---------------------------------------------------------------------------


class TestAllTypes:
    @pytest.mark.parametrize(
        "holding_type,contradiction_id",
        [
            (HoldingType.OBSERVATION, None),
            (HoldingType.CONNECTION, None),
            (HoldingType.CHALLENGE, None),
            (HoldingType.DAEMON_CURIOSITY, None),
            (HoldingType.OPEN_LOOP_FOLLOW_UP, None),
            (HoldingType.REASONED_DISAGREEMENT, "some-uuid"),
        ],
    )
    def test_all_types_persist(
        self,
        tmp_path: Path,
        holding_type: HoldingType,
        contradiction_id: str | None,
    ) -> None:
        p = tmp_path / "holding.yaml"
        store = HoldingStore(path=p)
        item = _item(type=holding_type, contradiction_id=contradiction_id)
        store.write(item)
        store2 = HoldingStore(path=p)
        assert store2.read(item.id).type == holding_type
