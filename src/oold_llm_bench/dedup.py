"""Deciding whether two records are one entity, cheaply before expensively.

Three tiers, and the point of the first two is that they are free. A judge
call is the only thing here that costs anything, so what a resolver is worth
is not only how often it is right but how seldom it had to ask.

The shape is taken from Graphiti's `dedup_helpers`, which is the best public
version of this: exact match first, then approximate string similarity, then
the model. What is taken is the **entropy gate**, which is the part most
implementations miss. A short or repetitive name carries little information,
so two records agreeing on one is weak evidence and a similarity score over it
is confident noise. Those are handed to the judge rather than trusted.

What is not taken is Graphiti's two-way answer. Its prompt returns an id or
``-1``, and ``-1`` means "no match **or you are unsure**", so a deferral and a
decision collapse into one bucket and nothing can be routed to a person. The
three-way split is the reason this exists, and
:class:`~oold_llm_bench.grading.identity.IdentityScore` reports coverage so
that the deferral is not free.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from oold_llm_bench.corpus.wikidata_identity import IdentityClass, Pair, agreeing_properties, shared_strings
from oold_llm_bench.grading.identity import AGREEMENT, JUDGE, PREFILTER, Judgement

__all__ = [
    "MIN_NAME_LENGTH",
    "NAME_ENTROPY_THRESHOLD",
    "SHINGLE",
    "Resolver",
    "jaccard",
    "name_entropy",
    "shingles",
]

NAME_ENTROPY_THRESHOLD = 1.5
"""Below this, a name is too thin to decide on and the pair goes to the judge.

Shannon entropy over the characters, in bits. "aaaa" scores 0.0 and tells a
resolver nothing; a real name scores above 3. Graphiti's threshold, kept
rather than retuned, because tuning it against this corpus would make the
gate a fact about these 270 pairs.
"""

MIN_NAME_LENGTH = 6
"""Shorter than this and the name is not evidence either, whatever its entropy."""

SHINGLE = 3
"""Character n-gram width for the approximate tier."""

_WORD = re.compile(r"[^\w\s]+", re.UNICODE)


def normalise(name: str) -> str:
    """Case-folded, punctuation-stripped, whitespace-collapsed."""
    return " ".join(_WORD.sub(" ", name).casefold().split())


def shingles(name: str, width: int = SHINGLE) -> frozenset[str]:
    text = normalise(name)
    if len(text) <= width:
        return frozenset({text}) if text else frozenset()
    return frozenset(text[i : i + width] for i in range(len(text) - width + 1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def name_entropy(name: str) -> float:
    """Shannon entropy over the characters, in bits."""
    text = normalise(name)
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    size = len(text)
    return -sum((n / size) * math.log2(n / size) for n in counts.values())


def informative(name: str) -> bool:
    """Whether agreeing on this name says anything."""
    return len(normalise(name)) >= MIN_NAME_LENGTH and name_entropy(name) >= NAME_ENTROPY_THRESHOLD


@dataclass
class Resolver:
    """Exact agreement, then similarity, then a judge.

    ``judge`` is the system under test and is never consulted by the grader.
    Left as ``None`` the resolver abstains wherever the cheap tiers cannot
    decide, which is the honest floor to measure a judge against: it shows
    what the deterministic part is worth on its own, at the coverage it
    actually achieves.

    ``similarity`` is deliberately high. The tier exists to catch a typo and a
    spelling variant, not to decide identity, and everything it does not catch
    is a question rather than a negative.
    """

    judge: Callable[[Pair], tuple[IdentityClass, str]] | None = None
    similarity: float = 0.9
    require_informative: bool = True

    def __call__(self, pair: Pair) -> Judgement:
        left, right = pair.left, pair.right
        names = shared_strings(left, right)
        usable = [name for name in names if informative(name)] if self.require_informative else list(names)

        if usable:
            shared, conflicting = agreeing_properties(left, right)
            if shared and not conflicting:
                return Judgement(
                    pair_id=pair.id,
                    outcome=IdentityClass.SAME,
                    route=AGREEMENT,
                    evidence=f"share the name {sorted(usable)[0]!r} and {shared} properties, none conflicting",
                )

        best = self._closest(left.strings, right.strings)
        if best is not None and best[0] >= self.similarity and informative(best[1]):
            return Judgement(
                pair_id=pair.id,
                outcome=IdentityClass.SAME,
                route=PREFILTER,
                evidence=f"names agree at jaccard {best[0]:.2f}: {best[1]!r}",
            )

        if self.judge is None:
            return Judgement(
                pair_id=pair.id,
                outcome=IdentityClass.UNCLEAR,
                route=PREFILTER,
                evidence="no cheap tier settled it and no judge was configured",
            )
        outcome, evidence = self.judge(pair)
        return Judgement(pair_id=pair.id, outcome=outcome, route=JUDGE, evidence=evidence)

    def _closest(self, left: Iterable[str], right: Iterable[str]) -> tuple[float, str] | None:
        """The most similar pair of names, and the better-formed of the two."""
        best: tuple[float, str] | None = None
        rights = [(name, shingles(name)) for name in right]
        for name in left:
            grams = shingles(name)
            for other, other_grams in rights:
                score = jaccard(grams, other_grams)
                if best is None or score > best[0]:
                    best = (score, name if len(name) >= len(other) else other)
        return best
