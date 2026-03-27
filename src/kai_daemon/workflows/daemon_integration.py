"""daemon_integration workflow (§7c, §7d).

Receives a KEEP-filtered inner thought and routes it to one of four
integration categories:

- ``new_fascination``    — creates a new fascination in DAEMON_SELF
- ``develops_existing`` — increments ``development_count`` on an
  existing fascination; triggers a lifecycle check when
  ``development_count >= 3``
- ``aesthetic_reaction`` — appends an entry to the aesthetic log
- ``inert``              — no state change

Fascination lifecycle check (§7d):
When a development pass brings a fascination's ``development_count`` to
3 or more, a second inference call asks whether to promote the fascination
to an explicit ``open_question``.  If promoted, the fascination status is
set to ``promoted_to_open_question`` and a new ``OpenQuestion`` entry is
appended to DAEMON_SELF.

Priority: 8 / Preemption: restart / Model: presentation
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from ..state.aesthetic_log import AestheticLog
from ..state.daemon_self import (
    DaemonSelf,
    DaemonSelfStore,
    Fascination,
    FascinationOrigin,
    FascinationStatus,
    OpenQuestion,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIFECYCLE_CHECK_THRESHOLD: int = 3

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_INTEGRATION_PROMPT_TEMPLATE = """\
Route this inner thought to the appropriate integration category.

Current fascinations:
{fascinations_list}

Inner thought:
{thought}

Choose exactly one category and respond in the exact format shown.

If the thought begins a new line of interest not covered by any fascination:
ROUTE: new_fascination
TOPIC: <brief topic name>
INTERESTING: <one sentence — what makes it interesting>

If the thought develops, extends, or deepens an existing fascination:
ROUTE: develops_existing
FASCINATION: <exact topic name from the list above>

If the thought is an aesthetic or sensory reaction
(beauty, discomfort, pattern recognition):
ROUTE: aesthetic_reaction
REACTION: <brief description of the reaction>

If the thought is unfocused or doesn't fit any category:
ROUTE: inert
"""

_INTEGRATION_NO_FASCINATIONS_PROMPT_TEMPLATE = """\
Route this inner thought to the appropriate integration category.

There are no existing fascinations yet.

Inner thought:
{thought}

Choose exactly one category and respond in the exact format shown.

If the thought begins a new line of interest:
ROUTE: new_fascination
TOPIC: <brief topic name>
INTERESTING: <one sentence — what makes it interesting>

If the thought is an aesthetic or sensory reaction
(beauty, discomfort, pattern recognition):
ROUTE: aesthetic_reaction
REACTION: <brief description of the reaction>

If the thought is unfocused or doesn't fit any category:
ROUTE: inert
"""

_LIFECYCLE_CHECK_PROMPT_TEMPLATE = """\
This fascination has been developed {development_count} times:

Topic: {topic}
What is interesting: {what_daemon_finds_interesting}

Has this matured into a concrete question worth holding explicitly as an open question?

Reply with exactly one word:
PROMOTE — the fascination has crystallised into a specific, well-defined open question
KEEP — still exploring; not ready to crystallise
"""

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class IntegrationRoute(StrEnum):
    """Four-way routing classification for an inner thought."""

    NEW_FASCINATION = "new_fascination"
    DEVELOPS_EXISTING = "develops_existing"
    AESTHETIC_REACTION = "aesthetic_reaction"
    INERT = "inert"


@dataclass
class IntegrationResult:
    """Result of the daemon_integration workflow."""

    route: IntegrationRoute
    fascination_topic: str | None
    """The created or developed fascination topic.  ``None`` for aesthetic/inert."""
    lifecycle_promoted: bool
    """``True`` if a fascination was promoted to ``open_question``."""
    thought_content: str
    """The original inner thought, carried forward for downstream workflows."""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_response(response: str) -> dict[str, str]:
    """Parse a ``KEY: value`` response into a dict (keys uppercased)."""
    fields: dict[str, str] = {}
    for line in response.strip().splitlines():
        line = line.strip()
        if ": " in line:
            key, _, val = line.partition(": ")
            fields[key.strip().upper()] = val.strip()
    return fields


def _parse_route(fields: dict[str, str]) -> IntegrationRoute:
    route_str = fields.get("ROUTE", "").lower().strip()
    try:
        return IntegrationRoute(route_str)
    except ValueError:
        logger.warning(
            "Unrecognised integration route %r — defaulting to inert", route_str
        )
        return IntegrationRoute.INERT


def _parse_lifecycle(response: str) -> bool:
    """Return ``True`` if the lifecycle check response is PROMOTE."""
    token = response.strip().upper().split()[0] if response.strip() else ""
    if token == "PROMOTE":
        return True
    if token not in ("KEEP", "PROMOTE"):
        logger.warning(
            "Unrecognised lifecycle response %r — defaulting to KEEP", response.strip()
        )
    return False


# ---------------------------------------------------------------------------
# Internal route handlers
# ---------------------------------------------------------------------------


def _handle_new_fascination(
    fields: dict[str, str],
    ds: DaemonSelf,
    now: datetime,
) -> tuple[DaemonSelf, str | None]:
    """Create a new fascination from parsed fields.  Returns (updated_ds, topic)."""
    topic = fields.get("TOPIC", "").strip()
    interesting = fields.get("INTERESTING", "").strip()
    if not topic:
        logger.warning("new_fascination route missing TOPIC — routing as inert")
        return ds, None
    fascination = Fascination(
        topic=topic,
        what_daemon_finds_interesting=interesting or topic,
        created=now.isoformat(),
        last_updated=now.isoformat(),
        origin=FascinationOrigin.INTEGRATION,
    )
    updated = ds.model_copy(
        update={
            "current_fascinations": [*ds.current_fascinations, fascination],
        }
    )
    return updated, topic


def _handle_develops_existing(
    fields: dict[str, str],
    ds: DaemonSelf,
    now: datetime,
    inference_fn: Callable[[str], str],
) -> tuple[DaemonSelf, str | None, bool]:
    """Develop an existing fascination.

    Returns (updated_ds, topic, lifecycle_promoted).
    """
    target_topic = fields.get("FASCINATION", "").strip()
    active = [
        f for f in ds.current_fascinations if f.status == FascinationStatus.ACTIVE
    ]

    # Find the matching fascination (case-insensitive)
    match: Fascination | None = None
    for f in active:
        if f.topic.lower() == target_topic.lower():
            match = f
            break

    if match is None:
        # Fallback: pick the first active fascination if there is one
        if active:
            logger.warning(
                "develops_existing: fascination %r not found among active — "
                "falling back to first active fascination",
                target_topic,
            )
            match = active[0]
        else:
            logger.warning(
                "develops_existing: no active fascinations — routing as inert"
            )
            return ds, None, False

    new_count = match.development_count + 1
    updated_f = match.model_copy(
        update={
            "development_count": new_count,
            "last_developed": now.isoformat(),
            "last_updated": now.isoformat(),
        }
    )

    # Lifecycle check at threshold
    promoted = False
    if new_count >= LIFECYCLE_CHECK_THRESHOLD:
        prompt = _LIFECYCLE_CHECK_PROMPT_TEMPLATE.format(
            development_count=new_count,
            topic=match.topic,
            what_daemon_finds_interesting=match.what_daemon_finds_interesting,
        )
        response = inference_fn(prompt)
        if _parse_lifecycle(response):
            updated_f = updated_f.model_copy(
                update={"status": FascinationStatus.PROMOTED_TO_OPEN_QUESTION}
            )
            promoted = True

    # Rebuild fascinations list with updated entry
    new_fascinations = [
        updated_f if f.topic == match.topic else f for f in ds.current_fascinations
    ]
    new_open_questions = list(ds.open_questions)
    if promoted:
        new_open_questions.append(
            OpenQuestion(
                question=match.topic,
                why_unresolved=match.what_daemon_finds_interesting,
                created=now.isoformat(),
            )
        )

    updated_ds = ds.model_copy(
        update={
            "current_fascinations": new_fascinations,
            "open_questions": new_open_questions,
        }
    )
    return updated_ds, match.topic, promoted


def _handle_aesthetic_reaction(
    fields: dict[str, str],
    thought: str,
    aesthetic_log: AestheticLog,
) -> None:
    """Append an aesthetic reaction entry."""
    reaction = fields.get("REACTION", "").strip() or thought
    aesthetic_log.append(thought=thought, reaction=reaction)


# ---------------------------------------------------------------------------
# Public workflow function
# ---------------------------------------------------------------------------


def daemon_integration(
    thought: str,
    *,
    daemon_self_store: DaemonSelfStore,
    aesthetic_log: AestheticLog,
    inference_fn: Callable[[str], str],
    now: datetime | None = None,
) -> IntegrationResult:
    """Route a KEEP-filtered inner thought to the appropriate integration category.

    Parameters
    ----------
    thought:
        The KEEP-filtered inner thought text.
    daemon_self_store:
        Loaded/saved DAEMON_SELF store.
    aesthetic_log:
        Aesthetic log for aesthetic_reaction route.
    inference_fn:
        Callable that sends a prompt to the presentation model and returns
        the raw text response.  Injectable for testing.
    now:
        Override current time (for testing).

    Returns
    -------
    IntegrationResult
        Route classification, fascination topic (if applicable), lifecycle
        promotion flag, and the original thought content.
    """
    _now = now if now is not None else datetime.now(UTC)

    ds = daemon_self_store.load() or DaemonSelf()
    active_fascinations = [
        f for f in ds.current_fascinations if f.status == FascinationStatus.ACTIVE
    ]

    # Build routing prompt
    if active_fascinations:
        fascinations_list = "\n".join(f"- {f.topic}" for f in active_fascinations)
        prompt = _INTEGRATION_PROMPT_TEMPLATE.format(
            fascinations_list=fascinations_list,
            thought=thought,
        )
    else:
        prompt = _INTEGRATION_NO_FASCINATIONS_PROMPT_TEMPLATE.format(thought=thought)

    response = inference_fn(prompt)
    fields = _parse_response(response)
    route = _parse_route(fields)

    # If develops_existing but no active fascinations, fall back to inert
    if route == IntegrationRoute.DEVELOPS_EXISTING and not active_fascinations:
        logger.warning(
            "develops_existing route with no active fascinations — routing as inert"
        )
        route = IntegrationRoute.INERT

    fascination_topic: str | None = None
    lifecycle_promoted = False

    if route == IntegrationRoute.NEW_FASCINATION:
        updated_ds, fascination_topic = _handle_new_fascination(fields, ds, _now)
        if fascination_topic is not None:
            daemon_self_store.write(updated_ds)
        else:
            route = IntegrationRoute.INERT

    elif route == IntegrationRoute.DEVELOPS_EXISTING:
        updated_ds, fascination_topic, lifecycle_promoted = _handle_develops_existing(
            fields, ds, _now, inference_fn
        )
        if fascination_topic is not None:
            daemon_self_store.write(updated_ds)
        else:
            route = IntegrationRoute.INERT

    elif route == IntegrationRoute.AESTHETIC_REACTION:
        _handle_aesthetic_reaction(fields, thought, aesthetic_log)

    # IntegrationRoute.INERT — no state change

    return IntegrationResult(
        route=route,
        fascination_topic=fascination_topic,
        lifecycle_promoted=lifecycle_promoted,
        thought_content=thought,
    )
