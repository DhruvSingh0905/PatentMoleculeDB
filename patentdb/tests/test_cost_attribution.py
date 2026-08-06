"""`route_audit.json`'s `by_route` must decompose real spend, automatically.

The block was added in `1ec0dad` and shipped `{}` on every patent ever
processed. The reason was not a bug in the writer: `all_route_spend()` reads
`per_patent_route`, which `record()` populated only inside a
`cost_tracker.attribute(route)` context manager — and `attribute()` had **zero
callers**. An opt-in attribution API that every future spend site must remember
to wrap is the same shape as the four other controls this codebase computed,
recorded and never consulted (`record_image`, `patent_image_exceeded`,
`reset_patent`, and `attribute` itself).

So attribution is derived from the **call stack at the spend site**. `record()`
is the one funnel every paid call in the package passes through — the three
`api_client` entry points and the four `repair/` sites that drive
`client.messages.create` themselves all end in it — and at that moment the stack
already names who is spending. Nothing has to be wrapped, decorated or
registered, which is what makes the coverage impossible to drift out of:

  * a NEW paid path is attributed the first time it spends, under its own
    `module:function`, with no edit to this file or to the tracker;
  * a path that is gated OFF contributes nothing and needs no bookkeeping;
  * a caller that forgets something cannot silently un-attribute its spend,
    because there is nothing for it to forget.

These tests pin that contract. None makes a paid call: every one either drives
`CostTracker` directly or hands `api_client` a fake client.
"""
from __future__ import annotations

import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from patentdb.core import config
from patentdb.core.cost_tracker import CostTracker

MODEL = config.DEFAULT_MODEL
PID = "USTEST0001"


def _spend(tracker: CostTracker, *, patent_id: str = PID, **kw) -> float:
    """A stand-in for any paid path: it calls `record` and asks for nothing."""
    return tracker.record(1000, 100, MODEL, patent_id=patent_id, **kw)


def _call_from_synthetic_module(name: str, tracker: CostTracker) -> None:
    """Spend from a module that does not exist on disk.

    The only way to prove attribution needs no registration is to attribute a
    module nothing could have registered.
    """
    mod = types.ModuleType(name)
    exec(  # noqa: S102 — synthesising a caller is the point
        "def spend(tracker, model, patent_id):\n"
        "    tracker.record(1000, 100, model, patent_id=patent_id)\n",
        mod.__dict__,
    )
    mod.spend(tracker, MODEL, PID)  # type: ignore[attr-defined]


# ── the core contract ──────────────────────────────────────────────


def test_spend_is_attributed_without_the_caller_asking():
    """No context manager, no decorator, no argument — just a `record()` call.

    This is the whole design. `_spend` above is written exactly the way the
    seven existing spend sites are written, and its cost still lands in
    `by_route` under the function that made it.
    """
    t = CostTracker()
    cost = _spend(t)
    routes = t.all_route_spend(PID)
    assert routes, "by_route is empty — attribution did not fire"
    assert routes == {"tests.test_cost_attribution:_spend": pytest.approx(cost)}


def test_the_route_names_the_function_not_just_the_module():
    """CLAUDE.md: a number is not a result until you can name the function that
    emitted it. `core.iupac_to_smiles` is a file; `_llm_direct_smiles` is an
    answer."""
    t = CostTracker()

    def inner_helper():
        _spend(t)

    inner_helper()
    key = next(iter(t.all_route_spend(PID)))
    module, _, func = key.partition(":")
    assert module == "tests.test_cost_attribution"
    assert func == "_spend"


def test_realign_spend_is_attributed_too():
    """`cost_category="realign"` is a separate BUDGET, not a separate ledger.

    Attribution used to sit inside `if patent_id and cost_category != "realign"`,
    so `llm_realigner` — the always-fire table extractor with the largest
    per-patent cap of the three ($1.50) — could never have appeared in
    `by_route` even once `attribute()` had a caller.
    """
    t = CostTracker()
    cost = _spend(t, cost_category="realign")
    assert t.all_route_spend(PID) == {
        "tests.test_cost_attribution:_spend": pytest.approx(cost)
    }
    # …and it still goes to its own budget, not the LM cap.
    assert t.patent_realign_spend(PID) == pytest.approx(cost)
    assert t.patent_spend(PID) == 0.0


def test_a_brand_new_paid_path_is_attributed_with_no_registration():
    """The anti-drift property, stated as a test.

    A lookup table mapping known modules to route names would answer "unknown"
    for the next path someone adds — which is how `by_route` would go back to
    being useless one commit at a time. The label is DERIVED, so a module that
    did not exist when this test was written attributes itself.
    """
    t = CostTracker()
    _call_from_synthetic_module("patentdb.core.a_route_nobody_has_written_yet", t)

    routes = t.all_route_spend(PID)
    assert list(routes) == ["core.a_route_nobody_has_written_yet:spend"], routes
    assert routes["core.a_route_nobody_has_written_yet:spend"] > 0


# ── the transport layer is not a route ─────────────────────────────


class _FakeUsage:
    input_tokens = 1000
    output_tokens = 100


class _FakeBlock:
    type = "text"
    text = "ok"


class _FakeResponse:
    usage = _FakeUsage()
    content = [_FakeBlock()]
    stop_reason = "end_turn"


class _FakeMessages:
    def create(self, **kw):
        return _FakeResponse()


class _FakeClient:
    """Never opens a socket. Returns token counts so `record()` has something
    to attribute."""

    messages = _FakeMessages()


@pytest.fixture
def offline_api(monkeypatch):
    """`api_client` with the network, the response cache and the jsonl log
    removed. A test that writes to `output_v2/cache` would poison paid output.
    """
    import patentdb.core.api_client as api

    monkeypatch.setattr(api, "_get_client", lambda: _FakeClient())
    monkeypatch.setattr(api, "get_cached", lambda *a, **k: None)
    monkeypatch.setattr(api, "store_cached", lambda *a, **k: None)
    monkeypatch.setattr(api, "_log_api_call", lambda *a, **k: None)
    return api


def test_api_client_frames_are_transport_not_routes(offline_api):
    """`call_claude_text` is HOW a call was made, never WHY.

    Attributing to the innermost frame would put 100 % of the corpus's spend
    under `core.api_client:call_claude_text` and answer nothing. The walk skips
    `api_client` and `cost_tracker` and keeps going outward.
    """
    from patentdb.core.cost_tracker import cost_tracker

    pid = "USTEST_TRANSPORT"
    before = cost_tracker.all_route_spend(pid)
    assert not before

    offline_api.call_claude_text("hello", model=MODEL, patent_id=pid)

    routes = cost_tracker.all_route_spend(pid)
    assert list(routes) == [
        "tests.test_cost_attribution:test_api_client_frames_are_transport_not_routes"
    ], routes


def test_batch_spend_is_attributed_to_the_caller_of_the_batch(offline_api):
    """The batch path records inside `api_client`'s own result loop, several
    frames below the route that asked for the burst. It must still name the
    route.

    Driven here through the submit-failure fallback (the fake client has no
    `batches` surface), which is itself a live path: a batch that fails to
    submit re-issues sequentially and must not lose its attribution.
    """
    from patentdb.core.cost_tracker import cost_tracker

    pid = "USTEST_BATCH"
    offline_api.call_claude_text_batch(
        [{"prompt": "one", "model": MODEL}], patent_id=pid,
    )
    routes = cost_tracker.all_route_spend(pid)
    assert list(routes) == [
        "tests.test_cost_attribution:test_batch_spend_is_attributed_to_the_caller_of_the_batch"
    ], routes


# ── threads ────────────────────────────────────────────────────────


def test_attribution_survives_a_thread_pool():
    """`iupac_to_smiles` makes paid calls from a 10-worker ThreadPoolExecutor
    (`core/iupac_to_smiles.py:1079`), and `call_claude_text_batch` fans out too.

    This is why the route is read off the stack rather than out of a
    `contextvars.ContextVar` set once per stage: a ContextVar set on the main
    thread is NOT visible inside a pool worker, so the whole IUPAC cascade —
    the single largest LM consumer — would have recorded its spend under
    whatever the parent last set, or under nothing. The assertion below pins
    that stdlib behaviour, because it is the premise of the design.
    """
    import contextvars

    var: contextvars.ContextVar[str] = contextvars.ContextVar("route", default="")
    var.set("set_on_the_main_thread")

    t = CostTracker()
    seen: list[str] = []

    def worker(i: int):
        seen.append(var.get())
        _spend(t)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(worker, range(4)))

    assert seen == [""] * 4, (
        "a ContextVar set on the main thread reached a pool worker; if this "
        "ever becomes true the contextvar design is viable, but it is not today"
    )
    routes = t.all_route_spend(PID)
    assert list(routes) == ["tests.test_cost_attribution:_spend"], routes
    assert routes["tests.test_cost_attribution:_spend"] > 0


# ── per-run deltas ─────────────────────────────────────────────────


def test_route_spend_since_is_a_per_run_delta():
    """`lm_usd` in the audit is `patent_spend(pid) - initial_spend`, i.e. what
    THIS run spent. `by_route` read the cumulative dict, so a second
    `process_patent` call on the same patent in one process (which
    `calltrace_run --overhead` does three times) would have written a
    `by_route` that did not sum to `lm_usd`.
    """
    t = CostTracker()
    _spend(t)
    baseline = t.all_route_spend(PID)
    second = _spend(t)

    delta = t.route_spend_since(PID, baseline)
    assert delta == {"tests.test_cost_attribution:_spend": pytest.approx(second)}
    assert t.route_spend_since(PID, t.all_route_spend(PID)) == {}


def test_route_spend_since_keeps_a_route_that_is_new_this_run():
    t = CostTracker()
    _spend(t)
    baseline = t.all_route_spend(PID)

    # A different frame → a key that is absent from the baseline entirely.
    _call_from_synthetic_module("patentdb.core.second_route", t)

    delta = t.route_spend_since(PID, baseline)
    assert list(delta) == ["core.second_route:spend"], delta


# ── the controls that are gone ─────────────────────────────────────


def test_the_opt_in_attribution_api_is_gone():
    """`attribute()` is what kept `by_route` empty. Keeping it alongside the
    automatic derivation would reintroduce exactly one way to get attribution
    wrong."""
    t = CostTracker()
    assert not hasattr(t, "attribute")
    assert not hasattr(t, "route_spend")


def test_the_image_budget_is_gone():
    """The $0.50 image cap gated nothing and `image_usd` was always 0.0.

    `per_patent_image` had exactly one writer, `record_image()`, and
    `record_image()` had zero callers anywhere in the tree. The only vision
    entry point, `api_client.call_claude_vision`, has zero callers of its own
    AND records with the default `cost_category="lm"` — so even wiring the cap
    would have gated a bucket that no code can fill.

    The cap CONSTANT is deliberately not named in this file. `import_audit`'s
    unused-constant check is a plain text search over every `.py` in the
    package, so a docstring mentioning it would keep it looking live for as
    long as this test exists — a metric that cannot fail is not a check.
    """
    t = CostTracker()
    for gone in ("record_image", "patent_image_exceeded", "patent_image_spend"):
        assert not hasattr(t, gone), f"{gone} is a control nothing consults"
    assert "per_patent_image" not in t.summary()


def test_summary_reports_the_route_breakdown():
    t = CostTracker()
    _spend(t)
    s = t.summary()
    assert s["per_patent_route"][PID] == {
        "tests.test_cost_attribution:_spend": pytest.approx(
            t.patent_spend(PID), abs=1e-6
        )
    }
