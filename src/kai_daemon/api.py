"""Localhost-only action API for kai-devtools (§13).

Exposes four write actions that kai-devtools uses for the contradiction-resolution
and BORDERLINE pool review surfaces.  The server binds exclusively to ``127.0.0.1``
and is never exposed on the network.

Endpoints
---------
POST /actions/contradiction/{id}/resolve
    Discharge the holding item whose ``contradiction_id`` matches *id*,
    recording ``discharge_notes="resolve"``.

POST /actions/contradiction/{id}/dismiss
    Discharge the holding item whose ``contradiction_id`` matches *id*,
    recording ``discharge_notes="dismiss"`` (false positive — items remain
    compatible).

POST /actions/borderline/{id}/promote
    Promote a BORDERLINE pool item to the integration routing queue.

POST /actions/borderline/{id}/discard
    Discard a BORDERLINE pool item (suppress, do not integrate).

All endpoints return JSON: ``{"ok": true, "id": "<id>"}`` on success or
``{"ok": false, "error": "<message>"}`` on error.

HTTP status codes
-----------------
200  OK
404  Not found (unknown contradiction_id or BORDERLINE item id)
409  Conflict (holding item already discharged; BORDERLINE item not pending)
500  Unexpected internal error
"""

from __future__ import annotations

import json
import logging
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .state.borderline import BorderlinePool
from .state.holding import HoldingStore

logger = logging.getLogger(__name__)

DEFAULT_PORT: int = 9271
"""Default port the action API listens on."""

_LOCALHOST = "127.0.0.1"

_CONTRADICTION_PATTERN = re.compile(
    r"^/actions/contradiction/(?P<cid>[^/]+)/(?P<action>resolve|dismiss)$"
)
_BORDERLINE_PATTERN = re.compile(
    r"^/actions/borderline/(?P<bid>[^/]+)/(?P<action>promote|discard)$"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_holding_by_contradiction_id(
    holding_store: HoldingStore, contradiction_id: str
) -> str | None:
    """Return the id of the ``HoldingItem`` with the given ``contradiction_id``.

    Returns ``None`` when no matching item exists.
    """
    for item in holding_store.list_all():
        if item.contradiction_id == contradiction_id:
            return item.id
    return None


def _make_handler(
    holding_store: HoldingStore,
    borderline_pool: BorderlinePool,
) -> type[BaseHTTPRequestHandler]:
    """Return a request handler class closed over the given stores."""

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            path: str = self.path

            m = _CONTRADICTION_PATTERN.match(path)
            if m is not None:
                self._handle_contradiction(m.group("cid"), m.group("action"))
                return

            m = _BORDERLINE_PATTERN.match(path)
            if m is not None:
                self._handle_borderline(m.group("bid"), m.group("action"))
                return

            self._respond(404, {"ok": False, "error": "not found"})

        def _handle_contradiction(self, contradiction_id: str, action: str) -> None:
            item_id = _find_holding_by_contradiction_id(holding_store, contradiction_id)
            if item_id is None:
                self._respond(
                    404,
                    {
                        "ok": False,
                        "error": f"contradiction {contradiction_id!r} not found",
                    },
                )
                return
            try:
                holding_store.discharge(item_id, discharge_notes=action)
            except ValueError as exc:
                self._respond(409, {"ok": False, "error": str(exc)})
                return
            self._respond(200, {"ok": True, "id": contradiction_id})

        def _handle_borderline(self, item_id: str, action: str) -> None:
            try:
                if action == "promote":
                    borderline_pool.promote(item_id)
                else:
                    borderline_pool.discard(item_id)
            except KeyError:
                self._respond(
                    404,
                    {
                        "ok": False,
                        "error": f"BORDERLINE item {item_id!r} not found",
                    },
                )
                return
            except ValueError as exc:
                self._respond(409, {"ok": False, "error": str(exc)})
                return
            self._respond(200, {"ok": True, "id": item_id})

        def _respond(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.debug("action-api: " + format, *args)

    return _Handler


# ---------------------------------------------------------------------------
# Public server class
# ---------------------------------------------------------------------------


class ActionServer:
    """Localhost-only HTTP action server for kai-devtools (§13).

    Binds exclusively to ``127.0.0.1``.  Use as a context manager or call
    ``serve_forever()`` and ``shutdown()`` directly.

    Parameters
    ----------
    port:
        Port to listen on.  Pass ``0`` to let the OS assign a free port
        (useful in tests).  Default: ``DEFAULT_PORT`` (9271).
    holding_store:
        Holding store instance.  ``None`` → default path.
    borderline_pool:
        BORDERLINE pool instance.  ``None`` → default path.
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        *,
        holding_store: HoldingStore | None = None,
        borderline_pool: BorderlinePool | None = None,
    ) -> None:
        _store = holding_store if holding_store is not None else HoldingStore()
        _pool = borderline_pool if borderline_pool is not None else BorderlinePool()
        handler = _make_handler(_store, _pool)
        self._server = ThreadingHTTPServer((_LOCALHOST, port), handler)
        self._serving = False

    @property
    def address(self) -> tuple[str, int]:
        """``(host, port)`` the server is bound to.

        When *port* was ``0`` at construction, the OS-assigned port is
        returned here.
        """
        addr = self._server.server_address
        return str(addr[0]), int(addr[1])

    def serve_forever(self) -> None:
        """Start serving requests (blocks until ``shutdown()`` is called)."""
        host, port = self.address
        logger.info("action-api: listening on %s:%d", host, port)
        self._serving = True
        try:
            self._server.serve_forever()
        finally:
            self._serving = False

    def shutdown(self) -> None:
        """Stop serving and release the socket.

        Safe to call even if ``serve_forever()`` was never started.
        ``BaseServer.shutdown()`` deadlocks when called without a running
        ``serve_forever()`` loop, so we only call it when actually serving.
        """
        if self._serving:
            self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> ActionServer:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
