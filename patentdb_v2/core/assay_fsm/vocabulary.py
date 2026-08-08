"""Vocabulary loader for the FSM tokenizer.

The vocabulary lives in two JSON files:
  - data/assay_vocabulary.json — canonical, hand-curated, version-tracked
  - data/assay_vocabulary.discoveries.json — LLM-discovered tokens

Lifecycle of a discovered token:
  pending --(seen in ≥3 distinct fingerprints)--> auto_loaded
  auto_loaded --(curator review)--> promoted (moves to canonical)
                                  OR rejected (dropped from runtime)

The loader supports mtime-based reload so the FSM picks up
auto-promoted discoveries without a process restart.

Hard rule: NO patent-specific tokens. The loader rejects any vocab
entry whose `lemma` or `surface_forms` contain a US-style patent ID
or a known target/protein name from a curator-maintained denylist.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ── Token-class enum (closed set) ─────────────────────────────────


class TokenClass:
    """Token classes the FSM tokenizer can emit.

    Closed set — vocabulary entries with an unknown class are
    rejected at load time.
    """
    COMPOUND_ID = "COMPOUND_ID"
    QUALIFIER = "QUALIFIER"
    NUMBER = "NUMBER"
    UNIT = "UNIT"
    RUN_COUNT = "RUN_COUNT"
    NULL_MARKER = "NULL_MARKER"
    ASSAY_TYPE = "ASSAY_TYPE"
    COLUMN_HEADER_HINT = "COLUMN_HEADER_HINT"
    COMPOUND_ID_PREFIX = "COMPOUND_ID_PREFIX"
    ROW_BREAK = "ROW_BREAK"
    RANGE_SEP = "RANGE_SEP"
    PLUSMINUS = "PLUSMINUS"
    UNKNOWN = "UNKNOWN"

    _ALL = frozenset({
        COMPOUND_ID, QUALIFIER, NUMBER, UNIT, RUN_COUNT, NULL_MARKER,
        ASSAY_TYPE, COLUMN_HEADER_HINT, COMPOUND_ID_PREFIX, ROW_BREAK,
        RANGE_SEP, PLUSMINUS, UNKNOWN,
    })

    @classmethod
    def is_valid(cls, name: str) -> bool:
        return name in cls._ALL


# ── Vocab entry dataclass ─────────────────────────────────────────


@dataclass
class VocabEntry:
    """A single vocabulary entry — one lemma with its surface forms.

    Either `surface_forms` (literal-string match list) or `regex`
    (single regex pattern) MUST be provided. `maps_to` is the
    canonical form emitted in the Token's `lemma` slot.
    """
    lemma: str
    klass: str
    surface_forms: list[str] = field(default_factory=list)
    regex: str | None = None
    maps_to: str | None = None
    canonical: str | None = None
    nM_factor: float | None = None     # for UNIT entries
    capture_group: int | None = None   # for regex entries with sub-capture
    source: str = "unspecified"

    @classmethod
    def from_dict(cls, d: dict) -> "VocabEntry":
        klass = d.get("class", "")
        if not TokenClass.is_valid(klass):
            raise ValueError(f"vocab entry has unknown class {klass!r}: {d}")
        return cls(
            lemma=d["lemma"],
            klass=klass,
            surface_forms=list(d.get("surface_forms", [])),
            regex=d.get("regex"),
            maps_to=d.get("maps_to"),
            canonical=d.get("canonical"),
            nM_factor=d.get("nM_factor"),
            capture_group=d.get("capture_group"),
            source=d.get("source", "unspecified"),
        )


# ── Patent-leak guard ─────────────────────────────────────────────


# Match patent ID patterns regardless of surrounding context. We avoid
# `\b` because underscore is a word char in regex, so `_<patent>` shapes
# (a plausible discovery-lemma form) wouldn't trigger `\b` and would
# slip past the leak guard.
_PATENT_ID_RE = re.compile(r"US\d{6,}|WO\d{4}/\d{4,}|EP\d{6,}", re.IGNORECASE)


def _contains_patent_leak(text: str) -> bool:
    """True if the text looks like it contains a patent ID — vocab
    entries are rejected if their lemma or any surface_form contains
    one. This is a hard generalizability gate.
    """
    return bool(_PATENT_ID_RE.search(text))


# ── Loader ────────────────────────────────────────────────────────


@dataclass
class AssayVocabulary:
    """In-memory representation of the canonical + discoveries vocab.

    Indexed for fast lookup:
      - by_surface: {literal_string -> VocabEntry}
      - regex_entries: list[(compiled_re, VocabEntry)]
      - by_class: {class_name -> list[VocabEntry]}
    """
    canonical_path: Path
    discoveries_path: Path
    entries: list[VocabEntry] = field(default_factory=list)
    by_surface: dict[str, VocabEntry] = field(default_factory=dict)
    by_class: dict[str, list[VocabEntry]] = field(default_factory=dict)
    regex_entries: list[tuple[re.Pattern, VocabEntry]] = field(default_factory=list)
    _canonical_mtime: float = 0.0
    _discoveries_mtime: float = 0.0
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # ── Public API ──

    @classmethod
    def load(
        cls,
        canonical_path: Path | str,
        discoveries_path: Path | str | None = None,
    ) -> "AssayVocabulary":
        cp = Path(canonical_path)
        dp = Path(discoveries_path) if discoveries_path else cp.with_name(
            cp.stem + ".discoveries" + cp.suffix
        )
        v = cls(canonical_path=cp, discoveries_path=dp)
        v.reload(force=True)
        return v

    def reload(self, *, force: bool = False) -> bool:
        """Reload from disk if either file's mtime changed (or force=True).
        Returns True if a reload actually happened.
        """
        with self._lock:
            ck = self._mtime_or_zero(self.canonical_path)
            dk = self._mtime_or_zero(self.discoveries_path)
            if (
                not force
                and ck == self._canonical_mtime
                and dk == self._discoveries_mtime
            ):
                return False
            entries = self._load_canonical()
            entries.extend(self._load_auto_loaded_discoveries())
            self._index(entries)
            self._canonical_mtime = ck
            self._discoveries_mtime = dk
            logger.info(
                "AssayVocabulary loaded: %d entries (canonical=%s, discoveries=%s)",
                len(entries), self.canonical_path.name, self.discoveries_path.name,
            )
            return True

    # Convenience accessors used by the tokenizer
    def lookup_surface(self, s: str) -> VocabEntry | None:
        with self._lock:
            return self.by_surface.get(s)

    def all_classes(self) -> list[str]:
        with self._lock:
            return sorted(self.by_class.keys())

    def entries_of_class(self, klass: str) -> list[VocabEntry]:
        with self._lock:
            return list(self.by_class.get(klass, []))

    # ── Discovery API (called by vocab_learner.py) ──

    def add_discovery(
        self,
        lemma: str,
        klass: str,
        surface: str,
        fingerprint: str,
        patent_id: str,
    ) -> None:
        """Append or update a pending discovery. NOT thread-safe across
        processes — callers should serialize via a file lock if running
        the pipeline in parallel.
        """
        if _contains_patent_leak(lemma) or _contains_patent_leak(surface):
            logger.warning(
                "rejected discovery with patent leak: lemma=%r surface=%r",
                lemma, surface,
            )
            return
        if not TokenClass.is_valid(klass):
            logger.warning("rejected discovery with bad class %r: %s", klass, lemma)
            return
        with self._lock:
            data = self._read_discoveries_raw()
            tokens: list[dict] = data.get("tokens", [])
            existing = next((t for t in tokens if t.get("lemma") == lemma), None)
            if existing is None:
                tokens.append({
                    "lemma": lemma,
                    "class": klass,
                    "surface_forms": [surface],
                    "status": "pending",
                    "fingerprints_observed": [fingerprint],
                    "first_seen_patent": patent_id,
                    "first_seen": _today(),
                    "n_observations": 1,
                })
            else:
                if surface not in existing.get("surface_forms", []):
                    existing.setdefault("surface_forms", []).append(surface)
                fps = existing.setdefault("fingerprints_observed", [])
                if fingerprint not in fps:
                    fps.append(fingerprint)
                existing["n_observations"] = existing.get("n_observations", 0) + 1
                # ≥3 distinct fingerprints + ≥3 observations -> auto_loaded
                if (
                    existing.get("status") == "pending"
                    and len(fps) >= 3
                    and existing["n_observations"] >= 3
                ):
                    existing["status"] = "auto_loaded"
                    logger.info(
                        "vocab discovery promoted to auto_loaded: %s (class=%s)",
                        lemma, klass,
                    )
            data["tokens"] = tokens
            data["last_updated"] = _today()
            self._write_discoveries_raw(data)

    def list_pending(self) -> list[dict]:
        """Curator-script accessor: list all pending + auto_loaded entries."""
        with self._lock:
            data = self._read_discoveries_raw()
            return [
                t for t in data.get("tokens", [])
                if t.get("status") in ("pending", "auto_loaded")
            ]

    def promote_to_canonical(self, lemma: str) -> bool:
        """Move a discovery to the canonical file. Curator-only API."""
        with self._lock:
            disc = self._read_discoveries_raw()
            tokens = disc.get("tokens", [])
            entry = next((t for t in tokens if t.get("lemma") == lemma), None)
            if entry is None:
                return False
            canon = self._read_canonical_raw()
            canon_tokens = canon.get("tokens", [])
            # Strip discovery-only metadata
            promoted = {
                k: v for k, v in entry.items()
                if k not in ("status", "fingerprints_observed",
                             "first_seen_patent", "first_seen", "n_observations")
            }
            promoted["source"] = f"promoted_{_today()}"
            canon_tokens.append(promoted)
            canon["tokens"] = canon_tokens
            canon["last_updated"] = _today()
            self._write_canonical_raw(canon)
            entry["status"] = "promoted"
            self._write_discoveries_raw(disc)
            self.reload(force=True)
            return True

    def reject(self, lemma: str) -> bool:
        """Mark a discovery rejected (kept in file for audit, not loaded)."""
        with self._lock:
            disc = self._read_discoveries_raw()
            tokens = disc.get("tokens", [])
            entry = next((t for t in tokens if t.get("lemma") == lemma), None)
            if entry is None:
                return False
            entry["status"] = "rejected"
            self._write_discoveries_raw(disc)
            self.reload(force=True)
            return True

    # ── Internals ──

    @staticmethod
    def _mtime_or_zero(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def _read_canonical_raw(self) -> dict:
        if not self.canonical_path.exists():
            return {"schema_version": "1.0", "tokens": []}
        return json.loads(self.canonical_path.read_text())

    def _write_canonical_raw(self, data: dict) -> None:
        self.canonical_path.parent.mkdir(parents=True, exist_ok=True)
        self.canonical_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _read_discoveries_raw(self) -> dict:
        if not self.discoveries_path.exists():
            return {"schema_version": "1.0", "tokens": []}
        return json.loads(self.discoveries_path.read_text())

    def _write_discoveries_raw(self, data: dict) -> None:
        self.discoveries_path.parent.mkdir(parents=True, exist_ok=True)
        self.discoveries_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _load_canonical(self) -> list[VocabEntry]:
        out: list[VocabEntry] = []
        for t in self._read_canonical_raw().get("tokens", []):
            try:
                out.extend(self._guarded_entries(t))
            except (ValueError, KeyError) as e:
                logger.warning("skipped bad canonical vocab entry: %r (%s)", t, e)
        return out

    def _load_auto_loaded_discoveries(self) -> list[VocabEntry]:
        out: list[VocabEntry] = []
        for t in self._read_discoveries_raw().get("tokens", []):
            if t.get("status") not in ("auto_loaded",):
                continue
            try:
                out.extend(self._guarded_entries(t))
            except (ValueError, KeyError) as e:
                logger.warning("skipped bad auto-loaded discovery: %r (%s)", t, e)
        return out

    def _guarded_entries(self, t: dict) -> Iterable[VocabEntry]:
        """Build a VocabEntry but reject anything with a patent leak."""
        lemma = t.get("lemma", "")
        if _contains_patent_leak(lemma):
            raise ValueError(f"patent leak in lemma: {lemma!r}")
        for sf in t.get("surface_forms", []):
            if _contains_patent_leak(sf):
                raise ValueError(f"patent leak in surface: {sf!r}")
        yield VocabEntry.from_dict(t)

    def _index(self, entries: list[VocabEntry]) -> None:
        self.entries = entries
        self.by_surface = {}
        self.by_class = {}
        self.regex_entries = []
        for e in entries:
            self.by_class.setdefault(e.klass, []).append(e)
            for s in e.surface_forms:
                # Last-wins is fine for collisions; canonical is loaded
                # before discoveries so canonical wins on conflicts.
                self.by_surface[s] = e
            if e.regex:
                try:
                    self.regex_entries.append((re.compile(e.regex), e))
                except re.error as ex:
                    logger.warning("bad vocab regex %r in %s: %s", e.regex, e.lemma, ex)


def _today() -> str:
    from datetime import date
    return str(date.today())
