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


# A capability patch is a long generation over a large prompt, and a dropped
# connection cost two whole patents on the first real run — US10227341 and
# US10266548 both died on APITimeoutError with no retry, losing the gap rather
# than the request. `uspto_xml._get` has retried transient HTTP for months; the
# model calls retried nothing.
_API_TIMEOUT_S = 600.0
_API_RETRIES = 3


def resilient_client() -> anthropic.Anthropic:
    """The client every repair tier should use: retries transient failures.

    The SDK retries connection errors, timeouts, 408/409/429 and 5xx on its own
    once configured — what it will not do is guess that we wanted it.
    """
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY,
                               max_retries=_API_RETRIES, timeout=_API_TIMEOUT_S)


def _get_client() -> anthropic.Anthropic:
    return resilient_client()


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
    """Raw API call with retry on transient errors.

    `temperature=0` is deliberate and load-bearing. This pipeline caches
    every response permanently keyed on (model, prompt), so whatever the
    first call returns becomes the answer forever. Left at the API
    default of 1.0, that froze a single high-temperature sample into the
    cache and made reruns unreproducible. Extraction is a
    transcription task, not a creative one — it wants the mode, not a draw.
    """
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": 0,
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
    cost_category: str = "lm",
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
    # Guard: reject empty prompts
    if not prompt or not prompt.strip():
        logger.warning(f"Empty prompt for {patent_id}/{compound_id} — skipping API call")
        return None

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

        cost_tracker.record(input_tokens, output_tokens, model, patent_id=patent_id,
                            cost_category=cost_category)
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

        cost_tracker.record(input_tokens, output_tokens, model, patent_id=patent_id)
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


# ── Batched API calls (Anthropic Message Batches API) ─────────────


def call_claude_text_batch(
    requests: list[dict],
    *,
    poll_interval_s: float = 15.0,
    poll_timeout_s: float = 1800.0,
    patent_id: str = "",
) -> list[str | None]:
    """Submit many text prompts as one Message Batch and return
    results in input order.

    The batches API gives **50 % off** the per-token price and runs
    requests concurrently server-side — for 30+ Agent 1/2/2b calls
    on a typical HARVEST burst, this is both cheaper and faster
    (wall-clock 1-5 min vs. sequential 15-30 min).

    Each ``requests[i]`` is a dict with keys:
      - ``prompt``       (required)  the user-side text
      - ``model``        (optional)  default config.DEFAULT_MODEL
      - ``system``       (optional)  system prompt
      - ``max_tokens``   (optional)  default 2048
      - ``cache_key``    (optional)  used for ``get_cached`` / ``store_cached``.
                                     Defaults to ``prompt`` (same shape as
                                     ``call_claude_text``).

    Returns: ``list[str | None]`` parallel to ``requests``. Cached
    entries are returned without consuming batch quota. ``None`` means
    a permanent error (logged via ``_log_failure``).

    Note: Anthropic's batches API guarantees completion within 24 h
    but typically returns in seconds-to-minutes for ≤ 100 requests.
    We poll every 15 s and time out at 30 min by default.
    """
    if not requests:
        return []

    # 1) Cache-check pass — separate hits from misses
    resolved: list[str | None] = [None] * len(requests)
    pending: list[tuple[int, dict]] = []  # (index_in_requests, normalized_req)
    for i, req in enumerate(requests):
        prompt = req.get("prompt") or ""
        if not prompt.strip():
            continue
        model = req.get("model") or config.DEFAULT_MODEL
        cache_key = req.get("cache_key") or prompt
        cached = get_cached(model, cache_key)
        if cached is not None:
            resolved[i] = cached
        else:
            pending.append((i, req))

    n_cached = sum(1 for r in resolved if r is not None)
    if not pending:
        logger.info(
            "batch %s: 100%% cache hit (%d/%d requests)",
            patent_id or "-", n_cached, len(requests),
        )
        return resolved

    # 2) Build batch request payload
    from anthropic.types.messages.batch_create_params import Request as BatchRequest
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    batch_items = []
    for idx, (orig_i, req) in enumerate(pending):
        params: dict = {
            "model": req.get("model") or config.DEFAULT_MODEL,
            "max_tokens": int(req.get("max_tokens") or 2048),
            "messages": [{"role": "user", "content": req["prompt"]}],
            "temperature": 0,   # same reason as _call_api — results are cached forever
        }
        if req.get("system"):
            params["system"] = req["system"]
        batch_items.append(BatchRequest(
            custom_id=f"req-{idx:04d}",
            params=MessageCreateParamsNonStreaming(**params),  # type: ignore[arg-type]
        ))

    client = _get_client()
    start = time.monotonic()
    try:
        batch = client.messages.batches.create(requests=batch_items)
    except Exception as e:
        logger.error("batch %s: submit failed: %r", patent_id or "-", e)
        # Fall back to sequential — preserves correctness, loses discount
        for orig_i, req in pending:
            resolved[orig_i] = call_claude_text(
                prompt=req["prompt"],
                model=req.get("model") or config.DEFAULT_MODEL,
                system=req.get("system") or "",
                max_tokens=int(req.get("max_tokens") or 2048),
                patent_id=patent_id,
            )
        return resolved

    logger.info(
        "batch %s: submitted batch %s (%d cached, %d queued)",
        patent_id or "-", batch.id, n_cached, len(pending),
    )

    # 3) Poll until terminal status.
    #
    # This loop blocks for up to `poll_timeout_s`. It used to report progress
    # only through `logger`, whose output is buffered when stdout is not a tty —
    # so a normal cache-miss run looked like a hard freeze with zero output for
    # up to half an hour. It emits an unbuffered heartbeat now: a blocked
    # pipeline must always be able to say what it is blocked on.
    deadline = time.monotonic() + poll_timeout_s
    waited = 0.0
    while True:
        try:
            batch = client.messages.batches.retrieve(batch.id)
        except Exception as e:
            logger.warning("batch %s: poll failed (%r), retrying", batch.id, e)
            time.sleep(poll_interval_s)
            waited += poll_interval_s
            continue
        status = batch.processing_status
        if status == "ended":
            break
        if status in ("canceling", "errored"):
            logger.error("batch %s: terminal status=%s", batch.id, status)
            return resolved
        if time.monotonic() > deadline:
            logger.error(
                "batch %s: poll timeout after %.0fs (status=%s); aborting",
                batch.id, poll_timeout_s, status,
            )
            return resolved
        print(
            f"  [batch {batch.id[:12]}] {status} — waited {waited:.0f}s / "
            f"{poll_timeout_s:.0f}s cap, {len(pending)} requests",
            flush=True,
        )
        time.sleep(poll_interval_s)
        waited += poll_interval_s

    elapsed_s = time.monotonic() - start
    logger.info(
        "batch %s: completed in %.0fs (succeeded=%d, errored=%d, canceled=%d, expired=%d)",
        batch.id,
        elapsed_s,
        batch.request_counts.succeeded,
        batch.request_counts.errored,
        batch.request_counts.canceled,
        batch.request_counts.expired,
    )

    # 4) Pull results — stream JSONL, route by custom_id
    pending_by_custom_id = {f"req-{idx:04d}": (orig_i, req) for idx, (orig_i, req) in enumerate(pending)}
    try:
        for result in client.messages.batches.results(batch.id):
            cid = result.custom_id
            if cid not in pending_by_custom_id:
                continue
            orig_i, req = pending_by_custom_id[cid]
            r_type = result.result.type
            if r_type == "succeeded":
                msg = result.result.message  # type: ignore[union-attr]
                input_tokens = msg.usage.input_tokens
                output_tokens = msg.usage.output_tokens
                # Batches API gets a 50 % discount — record at half cost
                cost_tracker.record(
                    input_tokens, output_tokens,
                    req.get("model") or config.DEFAULT_MODEL,
                    patent_id=patent_id,
                    discount=0.5,
                )
                # First text block (skip thinking / tool-use blocks)
                text = ""
                for block in msg.content:
                    if getattr(block, "type", None) == "text":
                        text = getattr(block, "text", "").strip()
                        break
                resolved[orig_i] = text
                cache_key = req.get("cache_key") or req["prompt"]
                store_cached(
                    req.get("model") or config.DEFAULT_MODEL,
                    cache_key, text,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                )
            elif r_type == "errored":
                err_obj = getattr(result.result, "error", None)  # type: ignore[union-attr]
                err = getattr(err_obj, "message", None) or str(err_obj or result.result)
                _log_failure(req["prompt"], err, patent_id, "")
                logger.error("batch result %s: errored: %s", cid, err[:160])
            elif r_type in ("canceled", "expired"):
                logger.warning("batch result %s: %s", cid, r_type)
    except Exception as e:
        logger.error("batch %s: result-fetch failed: %r", batch.id, e)
    return resolved
