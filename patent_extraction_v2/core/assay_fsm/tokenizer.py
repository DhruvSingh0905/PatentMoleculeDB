"""Stage 2 — Vocabulary-driven FSM tokenizer.

Converts a region's text into a typed Token stream. The tokenizer is
**purely data-driven**: it has zero hardcoded knowledge of patents,
targets, or compound IDs. It walks the input character-by-character,
attempting to match (in priority order):

  1. Vocabulary regex entries (RUN_COUNT patterns: `(N)`, `[N]`, `n=N`)
  2. Vocabulary literal surface forms (longest match wins per position)
  3. Generic NUMBER pattern (qualifier? + decimal value)
  4. Generic COMPOUND_ID pattern (letter? + digits + letters?)
  5. Whitespace / line breaks → ROW_BREAK or skip
  6. Anything else → UNKNOWN token (preserved with its surface)

Token stream is then handed to `row_aligner.py` to be partitioned
into rows.

Adding a new qualifier or null-marker is a vocab JSON edit, not a
code change. That's the architectural commitment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .vocabulary import AssayVocabulary, TokenClass


@dataclass(frozen=True)
class Token:
    """One emitted token from the FSM tokenizer."""
    klass: str                          # one of TokenClass.*
    surface: str                        # raw substring as it appears in source
    lemma: str | None = None            # canonical form when known
    value: float | None = None          # set for NUMBER tokens
    qualifier: str | None = None        # set for NUMBER tokens (the leading <,>,~,etc.)
    n_runs: int | None = None           # set for RUN_COUNT tokens
    span: tuple[int, int] = (0, 0)      # (start, end) in the region's source text


@dataclass
class TokenStream:
    """A stream of tokens for one region."""
    tokens: list[Token] = field(default_factory=list)
    source_text: str = ""

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


# ── Generic patterns (used after vocab lookup misses) ─────────────


# A bare number, optionally preceded by a known qualifier character.
# We DO NOT bake the qualifier list here — that comes from the vocab.
# This pattern matches ONLY the number portion; the qualifier is
# resolved by checking the previous token.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# Compound-id shape — letter? + digits + letter-suffix? (no dots, no
# decimals — those would be numbers).
_COMPOUND_ID_RE = re.compile(r"[A-Z]?\d{1,5}[A-Za-z]{0,3}\b")

# Whitespace and row-break detection.
_WHITESPACE_RE = re.compile(r"[ \t\xa0]+")
_NEWLINE_RE = re.compile(r"(?:\r\n|\r|\n)+")


# ── Tokenizer ────────────────────────────────────────────────────


def tokenize(region_text: str, vocab: AssayVocabulary) -> TokenStream:
    """Tokenize a region's text using the loaded vocabulary.

    Args:
        region_text: Already-normalized region text (from
            `region_detector.detect_regions(...).text`).
        vocab: Loaded vocabulary (`AssayVocabulary.load(...)`).

    Returns:
        A TokenStream. Empty token list is a valid result for
        non-assay text. Never raises.

    Algorithm:
        Scan left-to-right with a position cursor. At each position:
          1. Try to match every vocab regex; longest match wins.
          2. Try to match the longest known surface form starting here
             (longest-match-wins via the vocab's surface index).
          3. Try the generic NUMBER pattern — but only if the next
             chars look like a number (avoid eating compound IDs).
          4. Try the generic COMPOUND_ID pattern.
          5. Match a whitespace run (skip; emit ROW_BREAK if newline).
          6. Anything else: emit a single-char UNKNOWN token and advance.

        After the bare token sequence is emitted, post-pass merges
        adjacent QUALIFIER + NUMBER tokens into a single NUMBER token
        with the qualifier attached, and similarly merges trailing
        RUN_COUNT into the preceding NUMBER. This produces the
        higher-level "value cell" semantic that the row aligner expects.
    """
    if not region_text:
        return TokenStream(tokens=[], source_text=region_text)

    tokens: list[Token] = []
    text = region_text
    i = 0
    n = len(text)

    # Pre-compile vocab surface forms sorted by length (descending) so
    # longest match wins at any position. This matches "Cmp No." before "Cmp".
    surface_index = sorted(
        vocab.by_surface.keys(),
        key=lambda s: (-len(s), s),
    )

    while i < n:
        # Skip leading-of-token whitespace runs (but emit ROW_BREAK on \n)
        m = _NEWLINE_RE.match(text, i)
        if m:
            tokens.append(Token(
                klass=TokenClass.ROW_BREAK,
                surface=m.group(0),
                lemma="newline",
                span=(m.start(), m.end()),
            ))
            i = m.end()
            continue

        m = _WHITESPACE_RE.match(text, i)
        if m:
            i = m.end()
            continue

        # 1. Vocab regex entries (e.g., n_runs `(N)`)
        regex_hit = _try_vocab_regex(text, i, vocab)
        if regex_hit is not None:
            tok, new_i = regex_hit
            tokens.append(tok)
            i = new_i
            continue

        # 2. Vocab literal surface forms (longest first)
        surf_hit = _try_vocab_surface(text, i, vocab, surface_index)
        if surf_hit is not None:
            tok, new_i = surf_hit
            tokens.append(tok)
            i = new_i
            continue

        # 3. Generic number
        m = _NUMBER_RE.match(text, i)
        if m:
            try:
                value = float(m.group(0))
            except ValueError:
                value = None
            tokens.append(Token(
                klass=TokenClass.NUMBER,
                surface=m.group(0),
                value=value,
                span=(m.start(), m.end()),
            ))
            i = m.end()
            continue

        # 4. Generic compound id (letter? + digits + letters?)
        m = _COMPOUND_ID_RE.match(text, i)
        if m:
            tokens.append(Token(
                klass=TokenClass.COMPOUND_ID,
                surface=m.group(0),
                span=(m.start(), m.end()),
            ))
            i = m.end()
            continue

        # 5. UNKNOWN single char
        tokens.append(Token(
            klass=TokenClass.UNKNOWN,
            surface=text[i],
            span=(i, i + 1),
        ))
        i += 1

    # Post-pass: merge QUALIFIER + NUMBER and NUMBER + RUN_COUNT
    tokens = _merge_qualifier_number(tokens)
    tokens = _merge_number_runs(tokens)
    tokens = _classify_compound_ids(tokens)

    return TokenStream(tokens=tokens, source_text=region_text)


# ── Helpers ──────────────────────────────────────────────────────


def _try_vocab_regex(
    text: str, pos: int, vocab: AssayVocabulary,
) -> tuple[Token, int] | None:
    """Try every regex vocab entry at position `pos`. Returns the
    longest match (Token, new_position) or None.
    """
    best: tuple[Token, int] | None = None
    for pattern, entry in vocab.regex_entries:
        m = pattern.match(text, pos)
        if not m:
            continue
        # Build the token from the matched span
        n_runs_val = None
        if entry.klass == TokenClass.RUN_COUNT and entry.capture_group:
            try:
                n_runs_val = int(m.group(entry.capture_group))
            except (IndexError, ValueError):
                n_runs_val = None
        tok = Token(
            klass=entry.klass,
            surface=m.group(0),
            lemma=entry.lemma,
            n_runs=n_runs_val,
            span=(m.start(), m.end()),
        )
        if best is None or (m.end() - m.start()) > (best[0].span[1] - best[0].span[0]):
            best = (tok, m.end())
    return best


def _try_vocab_surface(
    text: str, pos: int, vocab: AssayVocabulary, sorted_surfaces: list[str],
) -> tuple[Token, int] | None:
    """Try literal-surface lookup, longest-match-first."""
    for surf in sorted_surfaces:
        if not surf:
            continue
        end = pos + len(surf)
        if end > len(text):
            continue
        # Case-insensitive match for ASSAY_TYPE / NULL_MARKER /
        # COMPOUND_ID_PREFIX (real-world tables are inconsistent about
        # case); case-sensitive for QUALIFIER and UNIT (μ vs M matters).
        entry = vocab.by_surface[surf]
        if entry.klass in (
            TokenClass.ASSAY_TYPE,
            TokenClass.NULL_MARKER,
            TokenClass.COMPOUND_ID_PREFIX,
        ):
            if text[pos:end].lower() != surf.lower():
                continue
        else:
            if text[pos:end] != surf:
                continue
        # Reject mid-word matches for word-character surface forms
        # (e.g., "nM" should not match inside "nMolar"). Anchor by
        # requiring non-word boundary after if the surface ends in a
        # word-char.
        if surf and surf[-1].isalnum():
            if end < len(text) and text[end].isalnum() and text[end] != "M":
                # Special case: "nM" + digit is fine, but "nM" + letter is suspicious
                continue
        # Reject mid-word matches before (e.g., catching "ki" inside "Aki")
        if surf and surf[0].isalnum():
            if pos > 0 and text[pos - 1].isalnum():
                continue
        tok = Token(
            klass=entry.klass,
            surface=text[pos:end],
            lemma=entry.maps_to or entry.canonical or entry.lemma,
            span=(pos, end),
        )
        return (tok, end)
    return None


def _merge_qualifier_number(tokens: list[Token]) -> list[Token]:
    """If a QUALIFIER token is immediately followed by a NUMBER, merge
    them into a single NUMBER with the qualifier in `qualifier`.
    """
    out: list[Token] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.klass == TokenClass.QUALIFIER and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            # Only merge if no text gap (same-cell adjacency)
            if nxt.klass == TokenClass.NUMBER and t.span[1] == nxt.span[0] or _adjacent(t, nxt):
                merged = Token(
                    klass=TokenClass.NUMBER,
                    surface=t.surface + nxt.surface,
                    lemma=nxt.lemma,
                    value=nxt.value,
                    qualifier=t.lemma,
                    span=(t.span[0], nxt.span[1]),
                )
                out.append(merged)
                i += 2
                continue
        out.append(t)
        i += 1
    return out


def _merge_number_runs(tokens: list[Token]) -> list[Token]:
    """If a RUN_COUNT token immediately follows a NUMBER, attach it
    to the NUMBER (set n_runs).
    """
    out: list[Token] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.klass == TokenClass.NUMBER and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt.klass == TokenClass.RUN_COUNT and _adjacent(t, nxt, max_gap=2):
                merged = Token(
                    klass=TokenClass.NUMBER,
                    surface=t.surface + " " + nxt.surface,
                    lemma=t.lemma,
                    value=t.value,
                    qualifier=t.qualifier,
                    n_runs=nxt.n_runs,
                    span=(t.span[0], nxt.span[1]),
                )
                out.append(merged)
                i += 2
                continue
        out.append(t)
        i += 1
    return out


def _classify_compound_ids(tokens: list[Token]) -> list[Token]:
    """Tag bare digits or alphanum tokens that look like compound IDs.

    Bare-digit tokens are emitted as NUMBER by the generic pattern.
    But if a bare-digit token is preceded by a COMPOUND_ID_PREFIX (like
    `Cpd. No.`) OR appears at the start of a row (after ROW_BREAK or
    at position 0) followed by a value cell, it's a compound id.

    Note: this pass is a soft hint. The row aligner makes the final
    determination based on column position.
    """
    out: list[Token] = []
    for i, t in enumerate(tokens):
        if t.klass == TokenClass.COMPOUND_ID_PREFIX and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt.klass == TokenClass.NUMBER and nxt.value is not None and nxt.value.is_integer():
                # Promote next token to compound id
                promoted = Token(
                    klass=TokenClass.COMPOUND_ID,
                    surface=nxt.surface,
                    lemma=str(int(nxt.value)),
                    span=nxt.span,
                )
                out.append(t)
                out.append(promoted)
                # Mark next i to skip
                tokens[i + 1] = promoted   # type: ignore  (we replace and continue iterating)
                continue
        out.append(t)
    return out


def _adjacent(a: Token, b: Token, *, max_gap: int = 0) -> bool:
    """Two tokens are adjacent if there are at most `max_gap` chars
    of whitespace between them.
    """
    return 0 <= b.span[0] - a.span[1] <= max_gap
