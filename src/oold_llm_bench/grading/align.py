"""Deciding which produced entity answers which expected one.

Alignment has to happen before anything can be scored, and it decides the
score. The predecessor walked the produced entities in dictionary order and
took the first that matched, so a later entity that matched better never got
the chance, and the result depended on insertion order.

This maximises total agreement over the whole assignment instead, and breaks
ties on a stable key, so the same inputs give the same alignment on every run
and on every Python version.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from oold_llm_bench.grading.compare import MatchMode, same_value
from oold_llm_bench.grading.triples import Scalar, Triple, make_triples
from oold_llm_bench.tasks.models import ExpectedInstance

__all__ = ["Alignment", "Pairing", "align", "duplicates", "overlap"]

_EXHAUSTIVE_LIMIT = 8
"""Above this many entities on either side, fall back to a greedy pass over
globally sorted pairs. Exhaustive assignment is factorial, and a task with
more than eight expected entities is rare enough that the exact optimum is not
worth an unbounded runtime."""

CLASS_AGREEMENT = 1e-3
"""Weight given to the class agreeing, as a tie-break only.

Small enough that value evidence always wins, large enough that an entity of
the right class whose values are all wrong still aligns. Without it that
entity cannot match anything, and an arm that picks classes correctly but
extracts nothing scores the same as an arm that produced no output at all,
which throws away the distinction the class dimension exists to report.
"""


@dataclass(frozen=True)
class Pairing:
    expected_key: str
    produced_key: str | None
    score: float


@dataclass(frozen=True)
class Alignment:
    pairings: tuple[Pairing, ...]
    unmatched_expected: tuple[str, ...]
    unmatched_produced: tuple[str, ...]
    exhaustive: bool
    """Whether the optimum was found or approximated, recorded so a result
    says which."""

    def matched(self) -> dict[str, str]:
        return {p.expected_key: p.produced_key for p in self.pairings if p.produced_key is not None}


def overlap(
    expected: ExpectedInstance,
    produced: frozenset[Triple],
    modes: dict[str, MatchMode] | None = None,
    produced_class: str | None = None,
) -> float:
    """How many of an expected instance's values the produced one carries.

    Counted over required fields only. Optional fields are scored when present
    but never make one candidate look like a better answer than another. A
    matching class adds :data:`CLASS_AGREEMENT`, which only ever decides a tie.
    """
    modes = modes or {}
    bonus = CLASS_AGREEMENT if produced_class == expected.class_path else 0.0
    wanted = make_triples(expected.key, expected.fields)
    if not wanted:
        return bonus
    by_property: dict[str, set[Scalar]] = {}
    for triple in produced:
        by_property.setdefault(triple.prop, set()).add(triple.value)

    hits = 0
    for triple in wanted:
        candidates = by_property.get(triple.prop, set())
        mode = modes.get(triple.prop, MatchMode.EXACT)
        if any(same_value(triple.value, candidate, mode) for candidate in candidates):
            hits += 1
    return hits / len(wanted) + bonus


def align(
    expected: list[ExpectedInstance],
    produced: dict[str, frozenset[Triple]],
    modes: dict[str, MatchMode] | None = None,
    classes: dict[str, str] | None = None,
) -> Alignment:
    """Match expected instances to produced ones, maximising total agreement."""
    classes = classes or {}
    expected_keys = [instance.key for instance in expected]
    produced_keys = sorted(produced)
    scores = {
        (instance.key, produced_key): overlap(instance, produced[produced_key], modes, classes.get(produced_key))
        for instance in expected
        for produced_key in produced_keys
    }

    exhaustive = len(expected_keys) <= _EXHAUSTIVE_LIMIT and len(produced_keys) <= _EXHAUSTIVE_LIMIT
    if exhaustive:
        chosen = _best_assignment(expected_keys, produced_keys, scores)
    else:
        chosen = _greedy_assignment(expected_keys, produced_keys, scores)

    pairings = tuple(
        Pairing(
            expected_key=key,
            produced_key=chosen.get(key),
            score=scores.get((key, chosen[key]), 0.0) if key in chosen else 0.0,
        )
        for key in expected_keys
    )
    taken = set(chosen.values())
    return Alignment(
        pairings=pairings,
        unmatched_expected=tuple(k for k in expected_keys if k not in chosen),
        unmatched_produced=tuple(k for k in produced_keys if k not in taken),
        exhaustive=exhaustive,
    )


def duplicates(
    expected: list[ExpectedInstance],
    produced: dict[str, frozenset[Triple]],
    alignment: Alignment,
    modes: dict[str, MatchMode] | None = None,
) -> dict[str, str]:
    """Produced entities that restate one the alignment already accounted for.

    Assignment is one to one, so a second emission of an entity can never win
    a pairing and arrives here among the leftovers. A leftover is a duplicate
    when it shares at least one required value with an expected instance the
    alignment matched to another produced entity. The result maps each such
    leftover to the expected instance it restates, so a count is never the
    only thing a duplicate leaves behind.

    The first rule is :func:`overlap` with the class bonus withheld. Withheld
    because sharing a class says two entities are the same kind of thing, and
    a document naming two people is the ordinary case rather than a
    repetition; a duplicate has to agree on a value. Everything else about
    the comparison is the matcher the alignment itself ran, so a pair the
    aligner would have taken had the slot been free is the pair counted here,
    and the duplicate count can be read against the entity count.

    The second rule is :func:`_restated_values`, and it exists because the
    first one states "a duplicate has to agree on a value" and then asks
    something narrower: whether a *property* holds that value.  A model
    restating an entity rarely restates the property it was filed under. It
    writes a stub that names the thing, and the name lands in ``name`` while
    the corpus had it in ``availableOnDevice``. Agreement on the value alone
    therefore counts, where the value is distinctive enough for the agreement
    to mean something.

    Leftovers that answer no expected instance are not duplicates, however
    alike they are to one another. Two copies of an invention are two
    inventions. The grader compares against the corpus and not against the
    arm's own consistency, and folding self-agreement in would put "said the
    right thing twice" and "said the same wrong thing twice" in one number
    when the fix for each is a different one.
    """
    modes = modes or {}
    matched_keys = set(alignment.matched())
    candidates = [instance for instance in expected if instance.key in matched_keys]
    found: dict[str, str] = {}
    for produced_key in sorted(alignment.unmatched_produced):
        values = produced[produced_key]
        ranked = sorted(
            ((overlap(instance, values, modes), instance.key) for instance in candidates),
            key=lambda pair: (-pair[0], pair[1]),
        )
        if ranked and ranked[0][0] > 0:
            found[produced_key] = ranked[0][1]
            continue
        restated = sorted((len(_restated_values(instance, values)), instance.key) for instance in candidates)
        if restated and restated[-1][0] > 0:
            found[produced_key] = restated[-1][1]
    return found


def _restated_values(expected: ExpectedInstance, produced: frozenset[Triple]) -> set[Scalar]:
    """Distinctive values an entity carries that an expected instance also has.

    The same value under a different property name. :func:`overlap` cannot see
    it, because it asks whether a property holds a value and this asks only
    whether the value is there at all.

    That gap was the twentieth instrument fault, and it hid the commonest way
    a model duplicates an entity. Asked for a link, a model writes the target
    again as a stub naming it: ``{"name": "Brinvin Torvale"}`` beside an
    expected record carrying that string under ``availableOnDevice``. Property
    agreement is zero, so the restatement was scored as an invention. Over the
    1,920 cells of ``e4`` the dimension counted 200 restatements and missed
    **405**.

    Distinctive, because the agreement has to mean something. A boolean, a
    small enumeration or a short code recurs across unrelated entities by
    chance, and counting those would turn coincidence into duplication. Six
    characters and not all digits, which on ``e4`` moves the count by two.
    """
    carried = {triple.value for triple in produced if _distinctive(triple.value)}
    return carried & {value for value in expected.fields.values() if _distinctive(value)}


def _distinctive(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 6 and not value.isdigit()


def _best_assignment(
    expected_keys: list[str],
    produced_keys: list[str],
    scores: dict[tuple[str, str], float],
) -> dict[str, str]:
    """The assignment with the highest total score.

    The longer side is permuted and the shorter is held still, so every
    injective pairing is reachable in both directions. Permuting the produced
    side alone only reaches the first ``width`` expected keys, and an answer
    that left out an entity could then never match the ones after the gap:
    four of five entities found were reported as three, so one missed entity
    cost recall twice.

    Ties break on the produced keys in sorted order, which is why the
    candidates are generated from a sorted list.
    """
    best: dict[str, str] = {}
    best_total = -1.0
    width = min(len(expected_keys), len(produced_keys))
    if not width:
        return best
    if len(produced_keys) >= len(expected_keys):
        sides = ((expected_keys, candidate) for candidate in permutations(produced_keys, width))
    else:
        sides = ((candidate, produced_keys) for candidate in permutations(expected_keys, width))
    for wanted, offered in sides:
        assignment = {
            key: produced_key
            for key, produced_key in zip(wanted, offered, strict=False)
            if scores[(key, produced_key)] > 0
        }
        total = sum(scores[(k, v)] for k, v in assignment.items())
        if total > best_total:
            best_total = total
            best = assignment
    return best


def _greedy_assignment(
    expected_keys: list[str],
    produced_keys: list[str],
    scores: dict[tuple[str, str], float],
) -> dict[str, str]:
    """Take the globally best remaining pair until none is left.

    Not first-fit: every pair is sorted before any is taken, so the result
    does not depend on the order entities arrived in.
    """
    ranked = sorted(
        (pair for pair, score in scores.items() if score > 0),
        key=lambda pair: (-scores[pair], pair[0], pair[1]),
    )
    assignment: dict[str, str] = {}
    used_produced: set[str] = set()
    for expected_key, produced_key in ranked:
        if expected_key in assignment or produced_key in used_produced:
            continue
        assignment[expected_key] = produced_key
        used_produced.add(produced_key)
    return assignment
