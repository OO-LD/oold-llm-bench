"""Scoring a system that decides whether two records are one entity.

Separate from :mod:`oold_llm_bench.corpus.wikidata_identity`, which scores
*rules*. A rule has no judge in it and exists to say whether the corpus is hard
before any model is called. This scores a system that may call one, and the
difference it has to report is where each decision came from.

**The judge is in the system under test and never in the grader.** Identity
properties are not definable: schema.org designates none, and which properties
identify an entity is class- and context-dependent, so a system needs
judgement and the thing worth measuring is that judgement. What this refuses is
a judge deciding whether the system was right. The ground truth here is a
person's recorded merge or ``P1889``, so nothing in this module calls a model.

**Accuracy and coverage are reported together or not at all.** A resolver that
answers ``closeMatch`` to everything is never wrong, and one that guesses on
everything is never silent. Either number alone is gameable, which is the same
discipline :class:`~oold_llm_bench.grading.score.Score` already applies to
entity precision against recall.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from oold_llm_bench.corpus.wikidata_identity import IdentityClass, Pair

__all__ = [
    "AGREEMENT",
    "JUDGE",
    "PREFILTER",
    "IdentityScore",
    "Judgement",
    "score_identity",
]

AGREEMENT = "agreement"
"""Decided by the records agreeing, with no call."""

PREFILTER = "prefilter"
"""Decided or deferred by a deterministic tier before any call."""

JUDGE = "judge"
"""Decided by a call. The only route that costs anything."""


@dataclass(frozen=True)
class Judgement:
    """One system decision about one pair, and how it was reached."""

    pair_id: str
    outcome: IdentityClass
    route: str
    evidence: str = ""

    @property
    def decided(self) -> bool:
        """Whether the system committed, rather than deferring to a person.

        ``UNCLEAR`` is a real answer and the corpus has a class for it, so a
        pair the corpus records as unclear and the system calls unclear is
        both correct and undecided. Coverage counts commitment, accuracy
        counts agreement, and the two questions are not the same one.
        """
        return self.outcome is not IdentityClass.UNCLEAR


@dataclass
class IdentityScore:
    """What a resolver scored, with the cost of getting there beside it."""

    name: str
    total: int = 0
    correct: int = 0
    decided: int = 0
    decided_correct: int = 0
    per_class: dict[str, tuple[int, int]] = field(default_factory=dict)
    routes: dict[str, int] = field(default_factory=dict)
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    """Truth to prediction, so a wrong merge and a missed one are apart.

    They are different failures with different fixes, and on this corpus they
    are not symmetric: a wrong merge cannot be undone by a later answer, and a
    missed one can.
    """

    @property
    def accuracy(self) -> float:
        """Agreement with the recorded class over every pair."""
        return self.correct / self.total if self.total else 0.0

    @property
    def coverage(self) -> float:
        """What share the system committed on rather than deferring."""
        return self.decided / self.total if self.total else 0.0

    @property
    def decided_accuracy(self) -> float:
        """Agreement over the pairs it committed on.

        The number a resolver raises by abstaining, which is why it is never
        reported without :attr:`coverage`.
        """
        return self.decided_correct / self.decided if self.decided else 0.0

    @property
    def calls(self) -> int:
        return self.routes.get(JUDGE, 0)

    @property
    def wrong_merges(self) -> int:
        """Pairs a person called different and the system merged.

        Reported on its own because it is the unrecoverable error. A graph
        that merged two entities has lost the evidence that they were two.
        """
        return sum(
            count
            for (truth, predicted), count in self.confusion.items()
            if truth == IdentityClass.DIFFERENT.value and predicted == IdentityClass.SAME.value
        )

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "coverage": round(self.coverage, 4),
            "decided_accuracy": round(self.decided_accuracy, 4),
            "calls": self.calls,
            "wrong_merges": self.wrong_merges,
            "routes": dict(sorted(self.routes.items())),
            "per_class": dict(sorted(self.per_class.items())),
        }


def score_identity(
    pairs: Iterable[Pair],
    resolve: Callable[[Pair], Judgement],
    name: str,
) -> IdentityScore:
    """Run a resolver over the corpus and count where it agreed.

    Every pair is offered, including the unscored ones, so that a resolver
    cannot be credited for a pair the corpus itself declines to score. Those
    are skipped here rather than at the caller, because skipping them
    somewhere else is how a total stops matching the corpus.
    """
    score = IdentityScore(name=name)
    per_class: dict[str, list[int]] = {}
    for pair in pairs:
        if not pair.scored:
            continue
        judgement = resolve(pair)
        hit = int(judgement.outcome is pair.identity)
        score.total += 1
        score.correct += hit
        score.routes[judgement.route] = score.routes.get(judgement.route, 0) + 1
        key = (pair.identity.value, judgement.outcome.value)
        score.confusion[key] = score.confusion.get(key, 0) + 1
        bucket = per_class.setdefault(pair.identity.value, [0, 0])
        bucket[0] += hit
        bucket[1] += 1
        if judgement.decided:
            score.decided += 1
            score.decided_correct += hit
    score.per_class = {k: (v[0], v[1]) for k, v in sorted(per_class.items())}
    return score
