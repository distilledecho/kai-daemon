"""Stage 1A acceptance tests — typed scratch space.

Covers:
- epistemic_origin immutability (set at write, never modifiable)
- Lifecycle rules (active → archived only; archived is terminal)
- All ScratchType values accepted
- Persistence through reload
- TTL expiry
- Acknowledgement (idempotent)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kai_daemon.state.scratch import (
    EpistemicOrigin,
    Lifecycle,
    ScratchNote,
    ScratchStore,
    ScratchType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_note(**kwargs: object) -> ScratchNote:
    defaults: dict[str, object] = {
        "workflow_id": "test_workflow",
        "session_id": "session-abc",
        "content": "test content",
        "type": ScratchType.OBSERVATION,
        "epistemic_origin": EpistemicOrigin.INTERNAL,
    }
    defaults.update(kwargs)
    return ScratchNote(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def store(tmp_path: Path) -> ScratchStore:
    return ScratchStore(path=tmp_path / "scratch.yaml")


# ---------------------------------------------------------------------------
# epistemic_origin immutability
# ---------------------------------------------------------------------------


class TestEpistemicOriginImmutability:
    def test_origin_preserved_after_write(self, store: ScratchStore) -> None:
        note = store.write(make_note(epistemic_origin=EpistemicOrigin.INTERNAL))
        assert note.epistemic_origin == EpistemicOrigin.INTERNAL

    def test_origin_cannot_be_changed_via_update_content(
        self, store: ScratchStore
    ) -> None:
        note = store.write(make_note())
        with pytest.raises(ValueError, match="immutable"):
            store.update_content(
                note.id, epistemic_origin=EpistemicOrigin.EXTERNAL_SEARCH
            )

    def test_origin_immutable_at_model_level_direct_assignment(
        self, store: ScratchStore
    ) -> None:
        """frozen=True blocks direct assignment, not just store.update_content."""
        note = store.write(make_note(epistemic_origin=EpistemicOrigin.INTERNAL))
        with pytest.raises(ValidationError):
            note.epistemic_origin = EpistemicOrigin.EXTERNAL_SEARCH  # type: ignore[misc]

    def test_all_fields_immutable_at_model_level(self, store: ScratchStore) -> None:
        """frozen=True covers all fields, not just epistemic_origin."""
        note = store.write(make_note())
        with pytest.raises(ValidationError):
            note.content = "mutated"  # type: ignore[misc]

    def test_all_origin_values_accepted_at_write(self, store: ScratchStore) -> None:
        for origin in EpistemicOrigin:
            n = store.write(make_note(epistemic_origin=origin))
            assert n.epistemic_origin == origin

    def test_origin_preserved_across_acknowledge(self, store: ScratchStore) -> None:
        note = store.write(
            make_note(epistemic_origin=EpistemicOrigin.INNER_LIFE_PIPELINE)
        )
        updated = store.acknowledge(note.id, "workflow_a")
        assert updated.epistemic_origin == EpistemicOrigin.INNER_LIFE_PIPELINE

    def test_origin_preserved_across_archive(self, store: ScratchStore) -> None:
        note = store.write(make_note(epistemic_origin=EpistemicOrigin.EXTERNAL_SEARCH))
        archived = store.archive(note.id)
        assert archived.epistemic_origin == EpistemicOrigin.EXTERNAL_SEARCH


# ---------------------------------------------------------------------------
# Lifecycle rules
# ---------------------------------------------------------------------------


class TestLifecycleRules:
    def test_new_note_starts_active(self, store: ScratchStore) -> None:
        note = store.write(make_note())
        assert note.lifecycle == Lifecycle.ACTIVE

    def test_archive_transitions_to_archived(self, store: ScratchStore) -> None:
        note = store.write(make_note())
        archived = store.archive(note.id)
        assert archived.lifecycle == Lifecycle.ARCHIVED

    def test_archived_note_cannot_be_unarchived(self, store: ScratchStore) -> None:
        note = store.write(make_note())
        store.archive(note.id)
        with pytest.raises(ValueError):
            store.unarchive(note.id)

    def test_archiving_already_archived_note_raises(self, store: ScratchStore) -> None:
        note = store.write(make_note())
        store.archive(note.id)
        with pytest.raises(ValueError):
            store.archive(note.id)

    def test_archived_excluded_from_list_active(self, store: ScratchStore) -> None:
        n1 = store.write(make_note())
        n2 = store.write(make_note())
        store.archive(n1.id)
        active_ids = {n.id for n in store.list_active()}
        assert n1.id not in active_ids
        assert n2.id in active_ids

    def test_lifecycle_state_persisted(self, tmp_path: Path) -> None:
        s1 = ScratchStore(path=tmp_path / "scratch.yaml")
        note = s1.write(make_note())
        s1.archive(note.id)

        s2 = ScratchStore(path=tmp_path / "scratch.yaml")
        reloaded = s2.read(note.id)
        assert reloaded.lifecycle == Lifecycle.ARCHIVED


# ---------------------------------------------------------------------------
# Scratch types
# ---------------------------------------------------------------------------


class TestScratchTypes:
    def test_all_type_values_accepted(self, store: ScratchStore) -> None:
        for t in ScratchType:
            n = store.write(make_note(type=t))
            assert n.type == t


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_note_survives_store_reload(self, tmp_path: Path) -> None:
        s1 = ScratchStore(path=tmp_path / "scratch.yaml")
        note = s1.write(make_note(content="persisted content"))

        s2 = ScratchStore(path=tmp_path / "scratch.yaml")
        reloaded = s2.read(note.id)
        assert reloaded.content == "persisted content"

    def test_epistemic_origin_survives_reload(self, tmp_path: Path) -> None:
        s1 = ScratchStore(path=tmp_path / "scratch.yaml")
        note = s1.write(make_note(epistemic_origin=EpistemicOrigin.INNER_LIFE_PIPELINE))

        s2 = ScratchStore(path=tmp_path / "scratch.yaml")
        reloaded = s2.read(note.id)
        assert reloaded.epistemic_origin == EpistemicOrigin.INNER_LIFE_PIPELINE

    def test_multiple_notes_survive_reload(self, tmp_path: Path) -> None:
        s1 = ScratchStore(path=tmp_path / "scratch.yaml")
        ids = [s1.write(make_note(content=str(i))).id for i in range(5)]

        s2 = ScratchStore(path=tmp_path / "scratch.yaml")
        for note_id in ids:
            assert note_id in {n.id for n in s2.list_active()}

    def test_duplicate_write_raises(self, store: ScratchStore) -> None:
        note = store.write(make_note())
        with pytest.raises(ValueError, match="already exists"):
            store.write(note)


# ---------------------------------------------------------------------------
# Acknowledgement
# ---------------------------------------------------------------------------


class TestAcknowledgement:
    def test_acknowledge_adds_workflow_id(self, store: ScratchStore) -> None:
        note = store.write(make_note())
        updated = store.acknowledge(note.id, "daemon_integration")
        assert "daemon_integration" in updated.acknowledged_by

    def test_acknowledge_is_idempotent(self, store: ScratchStore) -> None:
        note = store.write(make_note())
        store.acknowledge(note.id, "workflow_a")
        updated = store.acknowledge(note.id, "workflow_a")
        assert updated.acknowledged_by.count("workflow_a") == 1

    def test_multiple_workflows_can_acknowledge(self, store: ScratchStore) -> None:
        note = store.write(make_note())
        store.acknowledge(note.id, "workflow_a")
        updated = store.acknowledge(note.id, "workflow_b")
        assert "workflow_a" in updated.acknowledged_by
        assert "workflow_b" in updated.acknowledged_by


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


class TestTTLExpiry:
    def test_past_ttl_archived_on_expire(self, store: ScratchStore) -> None:
        note = store.write(make_note(ttl="2000-01-01T00:00:00+00:00"))
        count = store.expire_ttl()
        assert count == 1
        assert store.read(note.id).lifecycle == Lifecycle.ARCHIVED

    def test_future_ttl_not_archived(self, store: ScratchStore) -> None:
        note = store.write(make_note(ttl="2099-01-01T00:00:00+00:00"))
        count = store.expire_ttl()
        assert count == 0
        assert store.read(note.id).lifecycle == Lifecycle.ACTIVE

    def test_no_ttl_not_archived(self, store: ScratchStore) -> None:
        note = store.write(make_note(ttl=None))
        store.expire_ttl()
        assert store.read(note.id).lifecycle == Lifecycle.ACTIVE

    def test_already_archived_not_double_counted(self, store: ScratchStore) -> None:
        note = store.write(make_note(ttl="2000-01-01T00:00:00+00:00"))
        store.archive(note.id)
        count = store.expire_ttl()
        assert count == 0


# ---------------------------------------------------------------------------
# Query methods
# ---------------------------------------------------------------------------


class TestQueries:
    def test_list_by_session(self, store: ScratchStore) -> None:
        store.write(make_note(session_id="session-1"))
        store.write(make_note(session_id="session-1"))
        store.write(make_note(session_id="session-2"))
        assert len(store.list_by_session("session-1")) == 2
        assert len(store.list_by_session("session-2")) == 1

    def test_list_by_workflow(self, store: ScratchStore) -> None:
        store.write(make_note(workflow_id="workflow_a"))
        store.write(make_note(workflow_id="workflow_a"))
        store.write(make_note(workflow_id="workflow_b"))
        assert len(store.list_by_workflow("workflow_a")) == 2

    def test_read_missing_note_raises_key_error(self, store: ScratchStore) -> None:
        with pytest.raises(KeyError):
            store.read("nonexistent-id")

    def test_list_active_empty_store(self, store: ScratchStore) -> None:
        assert store.list_active() == []


# ---------------------------------------------------------------------------
# update_content
# ---------------------------------------------------------------------------


class TestUpdateContent:
    def test_update_mutable_fields(self, store: ScratchStore) -> None:
        note = store.write(make_note(content="original", target_workflow=None))
        updated = store.update_content(
            note.id,
            content="updated",
            ttl="2099-01-01T00:00:00+00:00",
            target_workflow="some_workflow",
        )
        assert updated.content == "updated"
        assert updated.ttl == "2099-01-01T00:00:00+00:00"
        assert updated.target_workflow == "some_workflow"

    def test_update_lifecycle_raises(self, store: ScratchStore) -> None:
        note = store.write(make_note())
        with pytest.raises(ValueError, match="immutable"):
            store.update_content(note.id, lifecycle="archived")

    def test_update_acknowledged_by_raises(self, store: ScratchStore) -> None:
        note = store.write(make_note())
        with pytest.raises(ValueError, match="immutable"):
            store.update_content(note.id, acknowledged_by=["workflow_a"])


# ---------------------------------------------------------------------------
# TTL validation
# ---------------------------------------------------------------------------


class TestTTLValidation:
    def test_malformed_ttl_raises_at_construction(self) -> None:
        with pytest.raises(ValidationError):
            make_note(ttl="not-a-date")

    def test_non_utc_offset_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_note(ttl="2000-01-01T00:00:00+02:00")

    def test_past_utc_offset_string_expires(self, store: ScratchStore) -> None:
        note = store.write(make_note(ttl="2000-01-01T00:00:00+00:00"))
        count = store.expire_ttl()
        assert count == 1
        assert store.read(note.id).lifecycle == Lifecycle.ARCHIVED
