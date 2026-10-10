"""Reading a property name against the vocabulary that defines it.

A catalogue that offers `author` and `creator` together asks a step to make a
distinction the document does not draw. "Written by Jane Doe" licenses both,
and Wikidata says so outright: P50 is a subproperty of P170. A step that
answers the parent has read the text and chosen a wider slot, which is not
the same failure as reading the wrong thing, and scoring it as a false
positive plus a false negative charges two errors for one answer.

So the relation is reported, never silently forgiven. :data:`Relation` says
which way the answer missed, and the strict count stays exactly as it was
beside the lenient one. A lenient number quoted alone would say the
vocabulary gap does not exist; the pair says how large it is.

The edges come from :mod:`scripts.build_property_hierarchy`, which takes them
from Wikidata and schema.org rather than from anyone's judgement here.

The same relation holds between classes: a catalogue drawn from a corpus with
a lineage asks a step to tell `Actor` from `Person` the same way it asks it to
tell `author` from `creator`, and answering the ancestor is the same kind of
under-specification. :func:`hierarchy_of` builds the structure from a task's
own ``class_parents`` rather than from a vocabulary file, for exactly that
comparison.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

__all__ = ["HIERARCHY_PATH", "PropertyHierarchy", "Relation", "hierarchy_of", "read_hierarchy"]

HIERARCHY_PATH = Path(__file__).resolve().parent.parent / "data" / "property_hierarchy.json"


class Relation(str, Enum):
    """How a produced property name stands to the expected one."""

    EXACT = "exact"
    BROADER = "broader"
    """The answer names an ancestor: right reading, wider slot."""
    NARROWER = "narrower"
    """The answer names a descendant.

    Reported apart from :attr:`BROADER` because it is the opposite mistake. A
    wider slot asserts less than the document says; a narrower one asserts
    more, and a step that reliably narrows is inventing specificity rather
    than hedging."""
    NONE = "none"


@dataclass(frozen=True)
class PropertyHierarchy:
    """Which offered properties sit under which others.

    ``parents`` holds the closure, so a lookup is a set membership and not a
    walk. The closure is taken over the union of both vocabularies here
    rather than in the build, because an edge one vocabulary asserts can
    extend a chain the other started and neither file knows about the join.
    """

    parents: dict[str, frozenset[str]]
    contested: frozenset[frozenset[str]] = frozenset()
    """Pairs the two vocabularies order in opposite directions.

    Kept so a report can say that a forgiveness rests on a disagreement. The
    closure already makes such a pair mutual, which is the right outcome for
    two names each vocabulary calls the other's parent, but it is not a fact
    either vocabulary asserts on its own."""

    def ancestors(self, name: str) -> frozenset[str]:
        return self.parents.get(name, frozenset())

    def relation(self, produced: str, expected: str) -> Relation:
        if produced == expected:
            return Relation.EXACT
        if produced in self.ancestors(expected):
            return Relation.BROADER
        if expected in self.ancestors(produced):
            return Relation.NARROWER
        return Relation.NONE

    def near(self, produced: str, expected: str) -> bool:
        """Whether the two name the same reading of the document."""
        return self.relation(produced, expected) is not Relation.NONE

    def forgive(self, produced: set[str], expected: set[str]) -> tuple[int, frozenset[str], frozenset[str]]:
        """How many of what a strict set comparison counts wrong name the same
        reading, paired one-to-one.

        Greedy and in sorted order, so the result does not depend on set
        iteration order. One-to-one because a catalogue can offer several
        names under one parent: `illustrator` and `author` both sit under
        `creator`, and answering `creator` once must not forgive both.

        Every caller scoring a vocabulary-aware dimension goes through this
        one pairing, rather than each writing its own: :mod:`steps.score` at
        the fillable step and :mod:`grading.score` at the triple grader forgive
        the identical relation, and two copies of the same rule is how they
        drift apart.

        Returns the count, and which names on each side were spent doing it,
        so a caller can build its own leftover false-positive and
        false-negative counts without re-deriving the pairing.
        """
        spare = sorted(produced - expected)
        missing = sorted(expected - produced)
        taken_produced: set[str] = set()
        taken_expected: set[str] = set()
        for name in spare:
            match = next((other for other in missing if other not in taken_expected and self.near(name, other)), None)
            if match is not None:
                taken_expected.add(match)
                taken_produced.add(name)
        return len(taken_expected), frozenset(taken_produced), frozenset(taken_expected)


def _close(direct: dict[str, set[str]]) -> dict[str, frozenset[str]]:
    """Every ancestor of every name, guarded against a cycle.

    A cycle is not hypothetical: the two vocabularies disagree on whether
    `affiliation` sits under `memberOf` or the other way round, and both
    edges are kept.
    """
    out: dict[str, frozenset[str]] = {}
    for start in direct:
        seen: set[str] = set()
        queue = list(direct[start])
        while queue:
            node = queue.pop()
            if node in seen or node == start:
                continue
            seen.add(node)
            queue.extend(direct.get(node, ()))
        out[start] = frozenset(seen)
    return out


@lru_cache(maxsize=4)
def read_hierarchy(path: Path | None = None) -> PropertyHierarchy:
    """The built hierarchy, or an empty one where none was built.

    Empty rather than raising: a corpus whose vocabulary declares no
    hierarchy is a corpus where every near miss is a real miss, and that is a
    correct answer rather than a missing file.
    """
    source = path or HIERARCHY_PATH
    if not source.is_file():
        return PropertyHierarchy(parents={})
    body = json.loads(source.read_text(encoding="utf-8"))
    direct: dict[str, set[str]] = {}
    for edge in body.get("edges", ()):
        direct.setdefault(edge["child"], set()).add(edge["parent"])
        direct.setdefault(edge["parent"], set())
    return PropertyHierarchy(
        parents=_close(direct),
        contested=frozenset(frozenset(pair) for pair in body.get("contested", ())),
    )


def hierarchy_of(parents: Mapping[str, Iterable[str]]) -> PropertyHierarchy:
    """A hierarchy built from a mapping already in hand, not from the built file.

    For a class lineage: a task's ``class_parents`` is the catalogue's own
    declared structure, not a vocabulary shared across tasks, so it is built
    fresh here rather than cached under a path the way :func:`read_hierarchy`
    is. No ``contested`` edges, since there is only the one source.
    """
    direct: dict[str, set[str]] = {name: set(values) for name, values in parents.items()}
    for values in list(direct.values()):
        for value in values:
            direct.setdefault(value, set())
    return PropertyHierarchy(parents=_close(direct))
