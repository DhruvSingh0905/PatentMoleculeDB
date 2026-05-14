"""IUPAC extraction-pattern library — sibling of `vocabulary.AssayVocabulary`.

Stores extraction *patterns* (e.g. "TABLE_COMMA_ORDERED_LIST",
"REVERSE_PAREN_FORMAT") that the IUPAC HARVEST fallback (Strategy 5) discovers
when regex Strategies 1–4 undershoot. Patterns auto-promote to runtime-active
status once they hit ≥3 distinct fingerprints AND ≥3 observations, with the
same patent-leak guard the assay vocabulary uses.

Why a separate library: assay vocabulary is bound to a fixed `TokenClass` enum
(COMPOUND_ID / QUALIFIER / ASSAY_TYPE / UNIT / etc.) that doesn't model
"this is a parser strategy". IUPAC patterns are a different taxonomy — they
describe how a patent's IUPAC names are formatted, not what kind of token
they are.

Files:
    data/iupac_patterns.json              — hand-curated, version-tracked
    data/iupac_patterns.discoveries.json  — auto-grown by the fallback agent

Both default to an empty token list if missing.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Patent-leak guard ─────────────────────────────────────────────


_PATENT_ID_RE = re.compile(r"US\d{6,}|WO\d{4}/\d{4,}|EP\d{6,}", re.IGNORECASE)


def _contains_patent_leak(text: str) -> bool:
    """Reject any pattern whose name/description contains a patent ID — same
    generalizability gate as `vocabulary._contains_patent_leak`."""
    return bool(_PATENT_ID_RE.search(text))


def _today() -> str:
    return date.today().isoformat()


# ── Pattern record ────────────────────────────────────────────────


@dataclass
class IupacPattern:
    """One discovered/curated IUPAC extraction pattern."""

    pattern_name: str            # e.g. "TABLE_COMMA_ORDERED_LIST"
    description: str             # human-readable, what this pattern recognizes
    regex_or_heuristic: str      # the rule (informal — narrative is fine)
    example_patent: str = ""     # one example patent id where it applied
    status: str = "pending"      # pending → auto_loaded → promoted/rejected
    n_observations: int = 0
    fingerprints: list[str] = field(default_factory=list)
    first_seen: str = ""
    last_updated: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "IupacPattern":
        return cls(
            pattern_name=d["pattern_name"],
            description=d.get("description", ""),
            regex_or_heuristic=d.get("regex_or_heuristic", ""),
            example_patent=d.get("example_patent", ""),
            status=d.get("status", "pending"),
            n_observations=d.get("n_observations", 0),
            fingerprints=list(d.get("fingerprints", [])),
            first_seen=d.get("first_seen", ""),
            last_updated=d.get("last_updated", ""),
        )

    def to_dict(self) -> dict:
        return {
            "pattern_name": self.pattern_name,
            "description": self.description,
            "regex_or_heuristic": self.regex_or_heuristic,
            "example_patent": self.example_patent,
            "status": self.status,
            "n_observations": self.n_observations,
            "fingerprints": self.fingerprints,
            "first_seen": self.first_seen,
            "last_updated": self.last_updated,
        }


# ── Library ───────────────────────────────────────────────────────


@dataclass
class IupacPatternLibrary:
    """In-memory + on-disk store of IUPAC extraction patterns.

    Mirrors the shape of `AssayVocabulary` for consistency. Patterns become
    runtime-active (returned by `runtime_active()`) once they reach
    `auto_loaded` status — the same ≥3-fingerprint / ≥3-observation gate the
    assay vocabulary uses.
    """

    canonical_path: Path
    discoveries_path: Path
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # ── Constructors ──

    @classmethod
    def default(cls, repo_root: Path | str | None = None) -> "IupacPatternLibrary":
        """Construct with the standard `data/iupac_patterns*.json` paths.

        `repo_root` defaults to the patent_extraction_v2 package root's parent
        (i.e. the project root). Callers can override for tests.
        """
        if repo_root is None:
            # core/assay_fsm/iupac_pattern_library.py → up 3 → repo root
            repo_root = Path(__file__).resolve().parents[3]
        repo_root = Path(repo_root)
        return cls(
            canonical_path=repo_root / "data" / "iupac_patterns.json",
            discoveries_path=repo_root / "data" / "iupac_patterns.discoveries.json",
        )

    # ── Public API ──

    def runtime_active(self) -> list[IupacPattern]:
        """All patterns that should be applied at runtime: hand-curated
        canonical entries plus discoveries that auto-promoted to
        `auto_loaded` status."""
        out: list[IupacPattern] = []
        out.extend(self._load_canonical())
        out.extend(p for p in self._load_discoveries() if p.status == "auto_loaded")
        return out

    def list_pending(self) -> list[IupacPattern]:
        """Curator accessor: every discovery (pending + auto_loaded)."""
        return [
            p for p in self._load_discoveries()
            if p.status in ("pending", "auto_loaded")
        ]

    def add_discovery(
        self,
        pattern: IupacPattern,
        fingerprint: str,
        patent_id: str,
    ) -> None:
        """Append-or-merge a discovery. Re-evaluates the auto-promotion gate
        (≥3 distinct fingerprints AND ≥3 observations).

        Patent-leak guard: discoveries whose `pattern_name`, `description`, or
        `regex_or_heuristic` contain a patent ID are rejected. Same rule the
        assay-vocab leak guard applies.
        """
        for field_text in (pattern.pattern_name, pattern.description, pattern.regex_or_heuristic):
            if _contains_patent_leak(field_text):
                logger.warning(
                    "rejected IUPAC pattern with patent leak: pattern_name=%r",
                    pattern.pattern_name,
                )
                return

        with self._lock:
            data = self._read_raw(self.discoveries_path)
            patterns: list[dict] = data.get("patterns", [])
            existing = next(
                (p for p in patterns if p.get("pattern_name") == pattern.pattern_name),
                None,
            )
            if existing is None:
                patterns.append({
                    "pattern_name": pattern.pattern_name,
                    "description": pattern.description,
                    "regex_or_heuristic": pattern.regex_or_heuristic,
                    "example_patent": patent_id,
                    "status": "pending",
                    "n_observations": 1,
                    "fingerprints": [fingerprint],
                    "first_seen": _today(),
                    "last_updated": _today(),
                })
            else:
                fps = existing.setdefault("fingerprints", [])
                if fingerprint not in fps:
                    fps.append(fingerprint)
                existing["n_observations"] = existing.get("n_observations", 0) + 1
                existing["last_updated"] = _today()
                # ≥3 distinct fingerprints + ≥3 observations -> auto_loaded
                if (
                    existing.get("status") == "pending"
                    and len(fps) >= 3
                    and existing["n_observations"] >= 3
                ):
                    existing["status"] = "auto_loaded"
                    logger.info(
                        "IUPAC pattern promoted to auto_loaded: %s",
                        pattern.pattern_name,
                    )
            data["patterns"] = patterns
            data["last_updated"] = _today()
            self._write_raw(self.discoveries_path, data)

    def promote_to_canonical(self, pattern_name: str) -> bool:
        """Move a discovery to the canonical file. Curator-only."""
        with self._lock:
            disc = self._read_raw(self.discoveries_path)
            patterns = disc.get("patterns", [])
            entry = next(
                (p for p in patterns if p.get("pattern_name") == pattern_name),
                None,
            )
            if entry is None:
                return False
            canon = self._read_raw(self.canonical_path)
            canon_patterns = canon.get("patterns", [])
            # Strip discovery-only metadata before promoting
            promoted = {
                k: v for k, v in entry.items()
                if k not in (
                    "status", "fingerprints", "example_patent",
                    "n_observations", "first_seen", "last_updated",
                )
            }
            promoted["source"] = f"promoted_{_today()}"
            canon_patterns.append(promoted)
            canon["patterns"] = canon_patterns
            canon["last_updated"] = _today()
            self._write_raw(self.canonical_path, canon)
            entry["status"] = "promoted"
            self._write_raw(self.discoveries_path, disc)
            return True

    def reject(self, pattern_name: str) -> bool:
        """Mark a discovery rejected (kept for audit, not loaded)."""
        with self._lock:
            disc = self._read_raw(self.discoveries_path)
            patterns = disc.get("patterns", [])
            entry = next(
                (p for p in patterns if p.get("pattern_name") == pattern_name),
                None,
            )
            if entry is None:
                return False
            entry["status"] = "rejected"
            self._write_raw(self.discoveries_path, disc)
            return True

    # ── Internals ──

    def _load_canonical(self) -> list[IupacPattern]:
        out: list[IupacPattern] = []
        for p in self._read_raw(self.canonical_path).get("patterns", []):
            try:
                if _contains_patent_leak(p.get("pattern_name", "")):
                    raise ValueError(f"patent leak in pattern_name: {p.get('pattern_name')!r}")
                out.append(IupacPattern.from_dict(p))
            except (ValueError, KeyError) as e:
                logger.warning("skipped bad canonical IUPAC pattern: %r (%s)", p, e)
        return out

    def _load_discoveries(self) -> list[IupacPattern]:
        out: list[IupacPattern] = []
        for p in self._read_raw(self.discoveries_path).get("patterns", []):
            try:
                if _contains_patent_leak(p.get("pattern_name", "")):
                    raise ValueError(f"patent leak in pattern_name: {p.get('pattern_name')!r}")
                out.append(IupacPattern.from_dict(p))
            except (ValueError, KeyError) as e:
                logger.warning("skipped bad discovery IUPAC pattern: %r (%s)", p, e)
        return out

    @staticmethod
    def _read_raw(p: Path) -> dict:
        if not p.exists():
            return {"schema_version": "1.0", "patterns": []}
        return json.loads(p.read_text())

    @staticmethod
    def _write_raw(p: Path, data: dict) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
