"""Tests for the localhost action API (§13)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from kai_daemon.api import DEFAULT_PORT, ActionServer
from kai_daemon.state._types import EpistemicOrigin
from kai_daemon.state.borderline import BorderlinePool, BorderlineStatus
from kai_daemon.state.holding import (
    HoldingItem,
    HoldingStore,
    HoldingType,
    RegisterNeeded,
    Urgency,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_holding_item(contradiction_id: str) -> HoldingItem:
    return HoldingItem(
        content="item A and item B conflict",
        type=HoldingType.REASONED_DISAGREEMENT,
        relevance_trigger="test trigger",
        register_needed=RegisterNeeded.REFLECTIVE,
        urgency=Urgency.MEDIUM,
        source_workflow="test",
        epistemic_origin=EpistemicOrigin.INTERNAL,
        contradiction_id=contradiction_id,
    )


def _post(server: ActionServer, path: str) -> tuple[int, dict[str, object]]:
    host, port = server.address
    url = f"http://{host}:{port}{path}"
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            status: int = resp.status
            body: dict[str, object] = json.loads(resp.read())
            return status, body
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stores(tmp_path: Path) -> tuple[HoldingStore, BorderlinePool]:
    holding = HoldingStore(path=tmp_path / "holding.yaml")
    pool = BorderlinePool(path=tmp_path / "borderline_pool.yaml")
    return holding, pool


@pytest.fixture
def server(stores: tuple[HoldingStore, BorderlinePool]) -> Iterator[ActionServer]:
    holding, pool = stores
    s = ActionServer(port=0, holding_store=holding, borderline_pool=pool)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    yield s
    s.shutdown()
    t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Address and defaults
# ---------------------------------------------------------------------------


def test_default_port_value() -> None:
    assert DEFAULT_PORT == 9271


def test_address_is_localhost(server: ActionServer) -> None:
    host, _ = server.address
    assert host == "127.0.0.1"


def test_address_port_nonzero_after_os_assign(server: ActionServer) -> None:
    _, port = server.address
    assert port > 0


def test_context_manager_without_serve_forever(tmp_path: Path) -> None:
    # shutdown() must not deadlock when serve_forever() was never called.
    holding = HoldingStore(path=tmp_path / "h.yaml")
    pool = BorderlinePool(path=tmp_path / "b.yaml")
    with ActionServer(port=0, holding_store=holding, borderline_pool=pool) as s:
        host, port = s.address
        assert host == "127.0.0.1"
        assert port > 0


def test_context_manager_with_serve_forever(tmp_path: Path) -> None:
    holding = HoldingStore(path=tmp_path / "h.yaml")
    pool = BorderlinePool(path=tmp_path / "b.yaml")
    with ActionServer(port=0, holding_store=holding, borderline_pool=pool) as s:
        t = threading.Thread(target=s.serve_forever, daemon=True)
        t.start()
        host, port = s.address
        assert host == "127.0.0.1"
        assert port > 0
    t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Unknown route
# ---------------------------------------------------------------------------


def test_unknown_route_returns_404(server: ActionServer) -> None:
    status, body = _post(server, "/not/a/real/endpoint")
    assert status == 404
    assert body["ok"] is False


def test_unknown_action_on_valid_prefix_returns_404(server: ActionServer) -> None:
    # "delete" is not a known action for contradiction
    status, body = _post(server, "/actions/contradiction/some-id/delete")
    assert status == 404
    assert body["ok"] is False


# ---------------------------------------------------------------------------
# POST /actions/contradiction/{id}/resolve
# ---------------------------------------------------------------------------


def test_contradiction_resolve_returns_200(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    holding, _ = stores
    cid = "cid-resolve-001"
    holding.write(_make_holding_item(cid))

    status, body = _post(server, f"/actions/contradiction/{cid}/resolve")

    assert status == 200
    assert body["ok"] is True
    assert body["id"] == cid


def test_contradiction_resolve_discharges_holding_item(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    holding, _ = stores
    cid = "cid-resolve-002"
    item = holding.write(_make_holding_item(cid))

    _post(server, f"/actions/contradiction/{cid}/resolve")

    refreshed = holding.read(item.id)
    assert refreshed.surfaced is not None
    assert refreshed.discharge_notes == "resolve"


def test_contradiction_resolve_unknown_id_returns_404(
    server: ActionServer,
) -> None:
    status, body = _post(server, "/actions/contradiction/nonexistent-cid/resolve")
    assert status == 404
    assert body["ok"] is False
    assert "not found" in str(body["error"]).lower()


def test_contradiction_resolve_already_discharged_returns_409(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    holding, _ = stores
    cid = "cid-resolve-003"
    item = holding.write(_make_holding_item(cid))
    holding.discharge(item.id)

    status, body = _post(server, f"/actions/contradiction/{cid}/resolve")

    assert status == 409
    assert body["ok"] is False


# ---------------------------------------------------------------------------
# POST /actions/contradiction/{id}/dismiss
# ---------------------------------------------------------------------------


def test_contradiction_dismiss_returns_200(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    holding, _ = stores
    cid = "cid-dismiss-001"
    holding.write(_make_holding_item(cid))

    status, body = _post(server, f"/actions/contradiction/{cid}/dismiss")

    assert status == 200
    assert body["ok"] is True
    assert body["id"] == cid


def test_contradiction_dismiss_discharges_with_dismiss_notes(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    holding, _ = stores
    cid = "cid-dismiss-002"
    item = holding.write(_make_holding_item(cid))

    _post(server, f"/actions/contradiction/{cid}/dismiss")

    refreshed = holding.read(item.id)
    assert refreshed.surfaced is not None
    assert refreshed.discharge_notes == "dismiss"


def test_contradiction_dismiss_unknown_id_returns_404(
    server: ActionServer,
) -> None:
    status, body = _post(server, "/actions/contradiction/ghost-id/dismiss")
    assert status == 404
    assert body["ok"] is False


def test_contradiction_dismiss_already_discharged_returns_409(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    holding, _ = stores
    cid = "cid-dismiss-003"
    item = holding.write(_make_holding_item(cid))
    holding.discharge(item.id)

    status, body = _post(server, f"/actions/contradiction/{cid}/dismiss")

    assert status == 409
    assert body["ok"] is False


# ---------------------------------------------------------------------------
# POST /actions/borderline/{id}/promote
# ---------------------------------------------------------------------------


def test_borderline_promote_returns_200(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    _, pool = stores
    item = pool.append("raw inner thought output")

    status, body = _post(server, f"/actions/borderline/{item.id}/promote")

    assert status == 200
    assert body["ok"] is True
    assert body["id"] == item.id


def test_borderline_promote_marks_item_promoted(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    _, pool = stores
    item = pool.append("raw inner thought output 2")

    _post(server, f"/actions/borderline/{item.id}/promote")

    refreshed = pool.get(item.id)
    assert refreshed.status == BorderlineStatus.PROMOTED
    assert refreshed.promoted_at is not None


def test_borderline_promote_unknown_id_returns_404(
    server: ActionServer,
) -> None:
    status, body = _post(server, "/actions/borderline/no-such-item/promote")
    assert status == 404
    assert body["ok"] is False
    assert "not found" in str(body["error"]).lower()


def test_borderline_promote_already_promoted_returns_409(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    _, pool = stores
    item = pool.append("thought")
    pool.promote(item.id)

    status, body = _post(server, f"/actions/borderline/{item.id}/promote")

    assert status == 409
    assert body["ok"] is False


def test_borderline_promote_already_discarded_returns_409(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    _, pool = stores
    item = pool.append("thought")
    pool.discard(item.id)

    status, body = _post(server, f"/actions/borderline/{item.id}/promote")

    assert status == 409
    assert body["ok"] is False


# ---------------------------------------------------------------------------
# POST /actions/borderline/{id}/discard
# ---------------------------------------------------------------------------


def test_borderline_discard_returns_200(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    _, pool = stores
    item = pool.append("raw output to discard")

    status, body = _post(server, f"/actions/borderline/{item.id}/discard")

    assert status == 200
    assert body["ok"] is True
    assert body["id"] == item.id


def test_borderline_discard_marks_item_discarded(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    _, pool = stores
    item = pool.append("raw output to discard 2")

    _post(server, f"/actions/borderline/{item.id}/discard")

    refreshed = pool.get(item.id)
    assert refreshed.status == BorderlineStatus.DISCARDED
    assert refreshed.discarded_at is not None


def test_borderline_discard_unknown_id_returns_404(
    server: ActionServer,
) -> None:
    status, body = _post(server, "/actions/borderline/missing/discard")
    assert status == 404
    assert body["ok"] is False


def test_borderline_discard_already_discarded_returns_409(
    server: ActionServer, stores: tuple[HoldingStore, BorderlinePool]
) -> None:
    _, pool = stores
    item = pool.append("thought")
    pool.discard(item.id)

    status, body = _post(server, f"/actions/borderline/{item.id}/discard")

    assert status == 409
    assert body["ok"] is False
