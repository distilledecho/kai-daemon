"""Path helpers for daemon-local state directories.

All paths are configurable via the KAI_DATA_DIR environment variable.
Directories are created on first access.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Root data directory. Override with KAI_DATA_DIR env var."""
    env = os.environ.get("KAI_DATA_DIR")
    if env:
        return Path(env)
    # Default: data/ at repo root (three levels up from src/kai_daemon/state/)
    return Path(__file__).parents[3] / "data"


def daemon_state_dir() -> Path:
    """data/daemon_state/ — all versioned daemon-local state."""
    p = data_dir() / "daemon_state"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    """data/logs/ — observability logs."""
    p = data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def threads_dir() -> Path:
    """data/daemon_state/threads/ — one YAML file per thread."""
    p = daemon_state_dir() / "threads"
    p.mkdir(parents=True, exist_ok=True)
    return p


def pickup_notes_dir() -> Path:
    """data/daemon_state/pickup_notes/ — dormancy pickup notes."""
    p = daemon_state_dir() / "pickup_notes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def memory_queue_dir() -> Path:
    """data/daemon_state/memory_queue/ — queued memory writes (server unavailable)."""
    p = daemon_state_dir() / "memory_queue"
    p.mkdir(parents=True, exist_ok=True)
    return p


def daemon_self_history_dir() -> Path:
    """data/daemon_state/daemon_self_history/ — all prior DAEMON_SELF versions."""
    p = daemon_state_dir() / "daemon_self_history"
    p.mkdir(parents=True, exist_ok=True)
    return p


def daemon_relational_history_dir() -> Path:
    """data/daemon_state/daemon_relational_history/ — prior DAEMON_RELATIONAL ver."""
    p = daemon_state_dir() / "daemon_relational_history"
    p.mkdir(parents=True, exist_ok=True)
    return p
