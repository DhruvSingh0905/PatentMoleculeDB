"""Claude API client with retry logic, logging, and cost tracking."""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import config
from .cost_tracker import cost_tracker
from .api_cache import get_cached, store_cached

logger = logging.getLogger(__name__)

# Failure log
FAILURES_LOG = config.LOGS_DIR / "failures.jsonl"
API_LOG = config.LOGS_DIR / "api_log.jsonl"


def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _log_api_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    success: bool,
    patent_id: str = "",
    compound_id: str = "",
    error: str = "",
):
    """Log every API call to jsonl for auditability."""
    cost = cost_tracker.compute_cost(input_tokens, output_tokens, model)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "latency_ms": round(latency_ms, 1),
        "success": success,
        "patent_id": patent_id,
        "compound_id": compound_id,
        "error": error,
    }
    with open(API_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _log_failure(prompt: str, error: str, patent_id: str = "", compound_id: str = ""):
    """Log permanent failures for manual review."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "patent_id": patent_id,
        "compound_id": compound_id,
        "prompt_preview": prompt[:500],
        "error": error,
    }
    with open(FAILURES_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


@retry(
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(
        multiplier=2,
        min=config.RETRY_MIN_WAIT,
        max=config.RETRY_MAX_WAIT,
    ),
    retry=retry_if_exception_type((
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
        anthropic.InternalServerError,
    )),
)
def _call_api(client, model, messages, max_tokens=2048, system=""):
    """Raw API call with retry on transient errors."""
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    return client.messages.create(**kwargs)


def call_claude_text(
    prompt: str,
    model: str = config.DEFAULT_MODEL,
    system: str = "",
    patent_id: str = "",
    compound_id: str = "",
    max_tokens: int = 2048,
) -> str | None:
    """Send a text prompt to Claude and return the response text.

    Args:
        prompt: The user prompt.
        model: Model to use.
        system: Optional system prompt.
        patent_id: For logging.
        compound_id: For logging.
        max_tokens: Max response tokens.

    Returns:
        Response text, or None on permanent failure.
    """
    # Check cache first
    cached = get_cached(model, prompt)
    if cached is not None:
        return cached

    client = _get_client()
    messages = [{"role": "user", "content": prompt}]

    start = time.monotonic()
    try:
        response = _call_api(client, model, messages, max_tokens, system=system)

        elapsed_ms = (time.monotonic() - start) * 1000
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        cost_tracker.record(input_tokens, output_tokens, model)
        _log_api_call(model, input_tokens, output_tokens, elapsed_ms, True,
                      patent_id, compound_id)

        result_text = response.content[0].text.strip()
        store_cached(model, prompt, result_text,
                     input_tokens=input_tokens, output_tokens=output_tokens)
        return result_text

    except (anthropic.RateLimitError, anthropic.APITimeoutError,
            anthropic.InternalServerError):
        # tenacity will retry these — re-raise
        raise
    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        _log_api_call(model, 0, 0, elapsed_ms, False, patent_id, compound_id, str(e))
        _log_failure(prompt, str(e), patent_id, compound_id)
        logger.error(f"Permanent API failure: {e}")
        return None


def call_claude_vision(
    prompt: str,
    image_path: str | Path,
    model: str = config.DEFAULT_MODEL,
    patent_id: str = "",
    compound_id: str = "",
    max_tokens: int = 2048,
) -> str | None:
    """Send an image + text prompt to Claude Vision.

    Args:
        prompt: Text prompt.
        image_path: Path to the image file.
        model: Model to use.
        patent_id: For logging.
        compound_id: For logging.
        max_tokens: Max response tokens.

    Returns:
        Response text, or None on permanent failure.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return None

    # Read and encode image
    image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

    # Check cache (use image hash + prompt as key)
    cached = get_cached(model, prompt, image_data)
    if cached is not None:
        return cached

    # Determine media type
    suffix = image_path.suffix.lower()
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    media_type = media_types.get(suffix, "image/png")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    client = _get_client()
    start = time.monotonic()
    try:
        response = _call_api(client, model, messages, max_tokens)

        elapsed_ms = (time.monotonic() - start) * 1000
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        cost_tracker.record(input_tokens, output_tokens, model)
        _log_api_call(model, input_tokens, output_tokens, elapsed_ms, True,
                      patent_id, compound_id)

        result_text = response.content[0].text.strip()
        store_cached(model, prompt, result_text, image_data=image_data,
                     input_tokens=input_tokens, output_tokens=output_tokens)
        return result_text

    except (anthropic.RateLimitError, anthropic.APITimeoutError,
            anthropic.InternalServerError):
        raise
    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        _log_api_call(model, 0, 0, elapsed_ms, False, patent_id, compound_id, str(e))
        _log_failure(prompt, str(e), patent_id, compound_id)
        logger.error(f"Permanent Vision API failure: {e}")
        return None
