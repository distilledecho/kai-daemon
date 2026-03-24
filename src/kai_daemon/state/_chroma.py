"""ChromaDB client factory.

Reads connection config from daemon-memory-server.yaml at repo root.
Provides collection-name constants for DAEMON_SELF and DAEMON_RELATIONAL.

This module handles connection setup only — it has no read path and does not
combine data from DAEMON_SELF and DAEMON_RELATIONAL in any way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Collection names — kept here so both store modules reference the same constants
# without importing each other.
DAEMON_SELF_COLLECTION = "daemon_self_versions"
DAEMON_RELATIONAL_COLLECTION = "daemon_relational_versions"

_CONFIG_PATH = Path(__file__).parents[3] / "daemon-memory-server.yaml"


def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(_CONFIG_PATH.read_text()) or {}


def make_chroma_client() -> Any:
    """Return a ChromaDB HTTP client configured from daemon-memory-server.yaml.

    Raises ``ImportError`` if chromadb is not installed.
    Raises ``chromadb.errors.ChromaError`` (or similar) if the server is unreachable.
    """
    import chromadb  # type: ignore[import-untyped]

    config = _load_config()
    conn = config.get("connection", {})
    host: str = conn.get("host", "localhost")
    port: int = int(conn.get("port", 8765))
    return chromadb.HttpClient(host=host, port=port)
