"""The Anthropic client the repair tier uses. Retries transient failures."""

from __future__ import annotations

import anthropic

from . import config


def resilient_client() -> anthropic.Anthropic:
    """The client every repair tier should use: retries transient failures.

    The SDK retries connection errors, timeouts, 408/409/429 and 5xx on its own
    once configured — what it will not do is guess that we wanted it.

    Both knobs come from `config`, deliberately. They were module-local here,
    which meant the two numbers governing how long a paid call can hang and how
    many times it is repeated did not appear in the file everyone reads to
    answer "what can this cost". Cost and latency parameters belong in one
    place or they cannot be measured.
    """
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY,
                               max_retries=config.MAX_RETRIES,
                               timeout=config.API_TIMEOUT_S)
