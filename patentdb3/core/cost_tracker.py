"""Cost tracking with threshold alerts, ceiling enforcement and per-route
attribution.

Every paid call in the package ends at `CostTracker.record()` — the three
`core/api_client.py` entry points and the four `repair/` sites that drive
`client.messages.create` themselves. That makes `record()` the one place where
attribution can be taken WITHOUT asking the caller for it, which is the only
kind of attribution that survives contact with a growing codebase: see
`derive_route` below.
"""

from __future__ import annotations

import logging
import sys
import threading

from . import config

logger = logging.getLogger(__name__)


class CostCeilingExceeded(Exception):
    """Raised when cumulative API cost exceeds the hard ceiling."""
    pass


# ── Route attribution ──────────────────────────────────────────────
#
# TRANSPORT, not routes. A frame in one of these modules describes HOW a call
# was made, never WHY, and attributing to it would put the corpus's entire LM
# bill under `core.api_client:call_claude_text` — a decomposition that answers
# nothing. The walk skips them and keeps going outward until it reaches the
# code that wanted the answer.
_TRANSPORT_MODULES = frozenset({
    "patentdb3.core.cost_tracker",
    "patentdb3.core.api_client",
})

_PKG_PREFIX = "patentdb3."


def derive_route() -> str:
    """Name the route responsible for the spend being recorded right now.

    Read off the call stack, not off an argument. The alternatives were both
    tried and both are opt-in at the call site:

      * the `attribute(route)` context manager this replaces — it shipped with
        ZERO callers, so `route_audit.json`'s `by_route` was `{}` on every
        patent this project has ever processed;
      * a `contextvars.ContextVar` set once per stage — invisible inside a
        `ThreadPoolExecutor` worker, and `iupac_to_smiles` makes its paid calls
        from a 10-worker pool (`core/iupac_to_smiles.py:1079`). The largest LM
        consumer in the pipeline would have been the one path it could not see.

    Anything derived from the stack, by contrast, cannot be forgotten: a paid
    path added tomorrow is attributed the first time it spends, under its own
    `module:function`, with no edit here and nothing to register. The cost of
    that is a label that changes when a function is renamed — cheap, and
    visible, next to a bucket that silently stays empty.

    Returns `"<module below patentdb.>:<function>"`, e.g.
    `"repair.synthesize:propose"` or `"core.iupac_to_smiles:_llm_direct_smiles"`.

    Known collision, stated so nobody reads more into a label than it holds:
    `iupac_burst` and `iupac_burst_targeted` are separately gated entry points
    that both spend through `harvest/agent_iupac_extract.extract_iupac_pairs`,
    so they share one bucket. The label names the function that made the call,
    which is the contract; separating the two callers means walking further out
    of the stack, and no cost question has needed it yet.
    """
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__") or "?"
        if module not in _TRANSPORT_MODULES:
            if module.startswith(_PKG_PREFIX):
                module = module[len(_PKG_PREFIX):]
            return f"{module}:{frame.f_code.co_name}"
        frame = frame.f_back
    # Only reachable if the whole stack is transport, which cannot happen from
    # `record()`. Named rather than dropped: an unattributable dollar is still
    # a dollar, and a silent zero is the failure this module exists to end.
    return "unattributed:?"


class CostTracker:
    """Tracks cumulative API costs and alerts at thresholds."""

    def __init__(self):
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.call_count: int = 0
        self._alerted: set[int] = set()
        # Patents that have already tripped the per-patent hard alarm (so we
        # warn ONCE, not on every subsequent call).
        self._patent_alerted: set[str] = set()
        # Per-patent LM spend tracker (for $0.20/patent cap)
        self.per_patent: dict[str, float] = {}
        # Per-patent LLM-realigner spend (the always-fire table extractor). Kept
        # SEPARATE from per_patent so a big assay table (US11254686: 17 chunks,
        # ~$1) doesn't instantly trip the $0.20 IUPAC cap and starve name
        # recovery. Has its own, more generous cap (PER_PATENT_REALIGN_CAP).
        self.per_patent_realign: dict[str, float] = {}
        # Per-patent per-route LM spend. Every call `record()` sees lands here
        # under the route `derive_route()` reads off the stack, regardless of
        # which per-patent BUDGET the call belongs to — the buckets above are
        # three separate caps, not three separate ledgers, and a decomposition
        # that omitted the realigner would omit the largest cap of the three.
        self.per_patent_route: dict[str, dict[str, float]] = {}
        # Guards every mutation of the counters above. `iupac_to_smiles` runs a
        # 10-worker ThreadPoolExecutor that makes paid calls, and `record()`
        # does read-modify-write (`self.total_cost += cost`, `d[k] = d.get(k,0)
        # + cost`) — neither is atomic, so concurrent calls lost updates and the
        # per-patent caps under-enforced. Money accounting is the one place that
        # must not silently undercount.
        self._lock = threading.Lock()
        self._unpriced_warned: set[str] = set()

    def compute_cost(self, input_tokens: int, output_tokens: int, model: str,
                     cache_read_tokens: int = 0,
                     cache_write_tokens: int = 0) -> float:
        """Cost in USD for one API call, computed from the token counts.

        THERE IS NO COST FIELD IN THE API RESPONSE. `usage` carries token
        counts and nothing else, so a dollar figure is always derived — the
        only alternative is Anthropic's organization-level Usage & Cost Report,
        which needs an admin key, is scoped to the whole org rather than a run,
        and lags. Neither property works for a per-patent gate, so this is
        computed locally and its accuracy rests entirely on `config.PRICING`
        matching current rates.

        CACHED TOKENS ARE BILLED DIFFERENTLY AND WERE NOT COUNTED AT ALL.
        `usage.input_tokens` EXCLUDES anything served from or written to the
        prompt cache; those arrive as `cache_read_input_tokens` and
        `cache_creation_input_tokens`. `synthesize` marks the SYSTEM prompt
        `cache_control: ephemeral`, so on every call after the first the bulk of
        the prompt lands in the read bucket — and was priced at zero. The tracker
        UNDER-reported, which is the worse direction for a spend gate: a cap
        cannot trip on tokens it never sees.

        Multipliers are Anthropic's standard 5-minute-TTL ones — writes at 1.25x
        the input rate, reads at 0.1x. A 1-hour TTL writes at 2x; nothing here
        requests one.
        """
        pricing = config.PRICING.get(model)
        if pricing is None:
            # Falling back silently to the most expensive tier misreports spend
            # and hides the fact that a model was never priced. Warn once.
            if model not in self._unpriced_warned:
                self._unpriced_warned.add(model)
                logger.warning(
                    "cost: %r is not in config.PRICING — billing it at Opus rates, "
                    "which will overstate spend. Add it to PRICING.", model)
            pricing = config.PRICING[config.MODEL_OPUS]
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
            + cache_write_tokens * pricing["input"] * 1.25 / 1_000_000
            + cache_read_tokens * pricing["input"] * 0.10 / 1_000_000
        )
        return cost

    def record(self, input_tokens: int, output_tokens: int, model: str,
               patent_id: str = "", discount: float = 0.0,
               cost_category: str = "lm", cache_read_tokens: int = 0,
               cache_write_tokens: int = 0) -> float:
        """Record an API call and check thresholds.

        Args:
            input_tokens, output_tokens, model: API call details.
            cache_read_tokens, cache_write_tokens: prompt-cache buckets, which
                `usage.input_tokens` does NOT include. Pass them or the call is
                priced as if the cached portion were free.
            patent_id: If provided, cost is also tracked per-patent for the
                       per-patent cap guards.
            discount: Fractional discount (0.0 = full price, 0.5 = Message
                      Batches 50% off). Applied to the computed cost.
            cost_category: Which per-patent budget this call belongs to.
                "lm" (default) → per_patent (PER_PATENT_LM_CAP); "realign" →
                per_patent_realign (PER_PATENT_REALIGN_CAP). Always counts
                toward the global total_cost ceiling regardless.

        Returns:
            Cost of this call in USD.

        Raises:
            CostCeilingExceeded: If cumulative cost exceeds the hard ceiling.
        """
        cost = self.compute_cost(input_tokens, output_tokens, model,
                                 cache_read_tokens, cache_write_tokens)
        if discount > 0:
            cost = cost * (1.0 - discount)

        # Taken OUTSIDE the lock: it only reads this thread's own frames, and
        # it must be taken here rather than deeper because the stack below this
        # point is `record`'s own.
        route = derive_route()

        # Everything below mutates shared counters. Held under one lock so a
        # concurrent worker can't interleave a read-modify-write and lose spend.
        with self._lock:
            self.total_cost += cost
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.call_count += 1
            # Realigner spend → its own per-patent bucket (don't pollute the LM cap).
            if patent_id and cost_category == "realign":
                self.per_patent_realign[patent_id] = (
                    self.per_patent_realign.get(patent_id, 0.0) + cost)
            if patent_id and cost_category != "realign":
                self.per_patent[patent_id] = self.per_patent.get(patent_id, 0.0) + cost
                # Per-patent HARD ALARM (visibility backstop). The soft cap is
                # enforced by per-path `patent_lm_exceeded` guards; if some
                # unguarded path slips past it, this surfaces the overspend LOUDLY
                # and once, instead of letting it silently run (the failure mode
                # that burned $5.6 on US10214537 via the force=True iupac_burst).
                if (self.per_patent[patent_id] >= config.PER_PATENT_LM_HARD_CAP
                        and patent_id not in self._patent_alerted):
                    self._patent_alerted.add(patent_id)
                    logger.warning(
                        "PER-PATENT COST ALARM: %s has spent $%.2f (hard alarm "
                        "$%.2f, soft cap $%.2f). An LLM path is firing past the "
                        "per-patent cap — grep COST-GATED / audit the path.",
                        patent_id, self.per_patent[patent_id],
                        config.PER_PATENT_LM_HARD_CAP, config.PER_PATENT_LM_CAP,
                    )
            # Attribution, for EVERY category — deliberately outside the two
            # budget branches above. `by_route` answers "where did this
            # patent's money go", and the realigner is money.
            #
            # A call with no `patent_id` cannot be attributed to a patent and
            # is dropped here rather than filed under a blank key; the
            # per-patent budgets already ignore it for exactly the same reason,
            # so this loses nothing `per_patent` was not already losing. Three
            # live sites spend without one and are therefore invisible to BOTH
            # the cap and the breakdown: `harvest/agent1_targets.extract_targets`
            # and `harvest/agent2_activities.extract_activities` /
            # `._extract_row_patterns`, the non-batch fallbacks. Fixing that is
            # one keyword argument at each site, not a change here.
            if patent_id:
                bucket = self.per_patent_route.setdefault(patent_id, {})
                bucket[route] = bucket.get(route, 0.0) + cost

            # Check thresholds
            for threshold in config.COST_THRESHOLDS:
                if self.total_cost >= threshold and threshold not in self._alerted:
                    self._alerted.add(threshold)
                    logger.warning(
                        f"COST ALERT: ${self.total_cost:.2f} spent "
                        f"(threshold ${threshold} reached). "
                        f"Calls: {self.call_count}"
                    )

            # Hard ceiling
            if self.total_cost >= config.COST_CEILING:
                raise CostCeilingExceeded(
                    f"Hard ceiling ${config.COST_CEILING} reached. "
                    f"Total spent: ${self.total_cost:.2f} over {self.call_count} calls."
                )

        return cost


# Global singleton
cost_tracker = CostTracker()
