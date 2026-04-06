"""onboarding workflow (Priority 0, startup_condition: user_yaml_empty).

Introduces the daemon to the user on first run.

Requires: daemon_seeding must complete first.
The daemon must have a self (DAEMON_SELF v1) before meeting the user.

Trigger: startup_condition (condition: user_yaml_empty)
Priority: 0 / Preemption: suspend / Model: presentation
Requires: daemon_seeding

Stage 4 note
------------
Full onboarding (interactive conversation with the user) is wired up in
Stage 4 via ``personal_assistant``.  This module provides the structural
scaffolding — it logs the intent and is a no-op until the conversation
layer exists.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_onboarding() -> None:
    """Onboarding entry point.

    Called by the workflow engine when ``daemon_name`` in ``user.yaml``
    is empty and ``daemon_seeding`` has completed.

    Stage 4 will replace this with the interactive onboarding session.
    For now this logs the pending state so it is visible in the
    observability log.
    """
    logger.info(
        "onboarding: daemon_name not yet set in user.yaml — "
        "onboarding will run via personal_assistant in Stage 4"
    )
