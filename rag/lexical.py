"""Shared lexical helpers used by both generation (sentence selection) and
verification (claim grounding), so the two stages agree on what counts as
"the same word". Deliberately simple and dependency-free (no NLTK/spaCy
model download needed in this sandbox): lowercase tokenization, a small
stopword list, and a crude suffix-stripping stemmer so that e.g. "power"
and "powers", or "exercise" and "exercised", count as a match.

This is a heuristic, not a linguistically correct stemmer (Porter/Snowball
would do better) -- documented here and in README as a known limitation:
it can over-stem (rare) or under-stem (common, e.g. irregular plurals).
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "as", "by", "with", "from", "shall",
    "may", "must", "not", "no", "which", "who", "whom", "their", "they",
    "has", "have", "had", "will", "would", "can", "could", "if", "than",
    "such", "also", "any", "other", "into", "up", "out", "so", "do",
    "does", "did", "about", "above", "under", "between", "each", "what",
    "how", "when", "where", "why",
}

MIN_TOKEN_LEN = 3


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def stem(token: str) -> str:
    """Crude suffix stripping -- enough to fold plurals/verb inflections
    ("powers"->"power", "exercised"->"exercis") without a model."""
    for suffix in ("ies",):
        if token.endswith(suffix) and len(token) > 4:
            return token[: -len(suffix)] + "y"
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) > 5:
            return token[: -len(suffix)]
    for suffix in ("es",):
        if token.endswith(suffix) and len(token) > 4:
            return token[: -len(suffix)]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def significant_tokens(text: str) -> set[str]:
    """Lowercased, stopword-filtered, stemmed content-word set."""
    return {
        stem(t)
        for t in tokenize(text)
        if t not in STOPWORDS and len(t) >= MIN_TOKEN_LEN
    }
