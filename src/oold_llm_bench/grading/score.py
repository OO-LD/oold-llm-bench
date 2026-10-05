"""Scoring one task, on several dimensions instead of one.

The primary number is micro F1 over ``(entity, property, value)`` triples. A
binary pass fails on one error exactly as hard as on five, which hides the
difference between an arm that misses a field and an arm that invents an
entity. It also has four possible values at three runs per cell, which is not
enough resolution to see an interaction.

Class assignment is a dimension and not a precondition. That lets an
arm which was given no schema, and therefore assigns no class, be scored by
the same grader as one which was.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

from oold_llm_bench.grading.align import Alignment, align, duplicates
from oold_llm_bench.grading.compare import (
    MatchMode,
    UnitMatch,
    same_number,
    same_quantity,
    same_unit,
    same_value,
)
from oold_llm_bench.grading.triples import (
    Dimension,
    Quantity,
    Reference,
    Scalar,
    Triple,
    TripleSet,
    make_triples,
)
from oold_llm_bench.tasks.models import ExpectedInstance, TaskRecord

__all__ = ["Score", "TaskScore", "score_task"]


@dataclass(frozen=True)
class Score:
    """Precision, recall and F1 over one dimension."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        total = 2 * self.true_positives + self.false_positives + self.false_negatives
        return 2 * self.true_positives / total if total else 0.0

    @property
    def support(self) -> int:
        return self.true_positives + self.false_negatives

    def __add__(self, other: Score) -> Score:
        return Score(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )

    def describe(self) -> dict[str, float | int]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "support": self.support,
        }


@dataclass
class TaskScore:
    """Every dimension for one task, plus the alignment it rests on."""

    task_id: str
    dimensions: dict[Dimension, Score] = field(default_factory=dict)
    alignment: Alignment | None = None
    parse_errors: int = 0
    class_errors: list[tuple[str, str | None]] = field(default_factory=list)
    """Expected and produced class for every mismatch.

    Counts say how often class assignment failed. Naming the pair says how,
    and that is the difference between a number and a finding. Class names are
    corpus vocabulary, so nothing here identifies an account."""

    @property
    def primary(self) -> float:
        """Micro F1 over value triples, the headline number."""
        return self.dimensions.get(Dimension.VALUE, Score()).f1

    def describe(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "primary_f1": round(self.primary, 4),
            "parse_errors": self.parse_errors,
            "class_errors": [list(pair) for pair in self.class_errors],
            "dimensions": {
                dimension.value: score.describe()
                for dimension, score in sorted(self.dimensions.items(), key=lambda item: item[0].value)
            },
            "alignment": {
                "matched": self.alignment.matched() if self.alignment else {},
                "unmatched_expected": list(self.alignment.unmatched_expected if self.alignment else ()),
                "unmatched_produced": list(self.alignment.unmatched_produced if self.alignment else ()),
                "exhaustive": self.alignment.exhaustive if self.alignment else None,
            },
        }


def _class_matches(
    expected: ExpectedInstance,
    produced_class: str | None,
    subclasses: dict[str, set[str]] | None,
) -> bool:
    if produced_class is None:
        return False
    if produced_class == expected.class_path:
        return True
    if not expected.allow_subclass or not subclasses:
        return False
    return expected.class_path in subclasses.get(produced_class, set())


def _magnitude_partner(expected: Scalar, candidates: list[Scalar]) -> Scalar | None:
    """The candidate that answers this value, ignoring how the unit is spelled.

    Used only to decide which produced quantity the unit dimension should
    judge. A wrong unit then scores as a wrong unit instead of disappearing
    because the value it belonged to did not match.
    """
    if not isinstance(expected, Quantity):
        return None
    for candidate in candidates:
        if isinstance(candidate, Quantity) and same_quantity(expected, candidate, unit_match=UnitMatch.PHYSICAL):
            return candidate
    for candidate in candidates:
        if isinstance(candidate, Quantity) and same_number(expected.magnitude, candidate.magnitude):
            return candidate
    return None


def _value_score(
    expected: ExpectedInstance,
    produced: frozenset[Triple],
    modes: dict[str, MatchMode],
    unit_match: UnitMatch = UnitMatch.EXACT,
) -> tuple[Score, Score, Score]:
    """Value, property and unit scores for one aligned pair."""
    wanted = make_triples(expected.key, expected.fields) | make_triples(expected.key, expected.optional_fields)
    optional_properties = {t.prop for t in make_triples(expected.key, expected.optional_fields)}
    by_property: dict[str, list[Scalar]] = {}
    for triple in produced:
        by_property.setdefault(triple.prop, []).append(triple.value)

    value = Score()
    unit = Score()
    matched_values: set[tuple[str, int]] = set()

    for triple in sorted(wanted, key=lambda t: (t.prop, str(t.value))):
        candidates = by_property.get(triple.prop, [])
        mode = modes.get(triple.prop, MatchMode.EXACT)
        hit = next(
            (
                candidate
                for candidate in candidates
                if (triple.prop, id(candidate)) not in matched_values
                and same_value(triple.value, candidate, mode, unit_match=unit_match)
            ),
            None,
        )
        if hit is not None:
            matched_values.add((triple.prop, id(hit)))
            value = value + Score(true_positives=1)
        elif triple.prop not in optional_properties:
            value = value + Score(false_negatives=1)

        # The unit is judged on its own candidate, found by physical equality.
        # Pinning it to the value match would erase unit failures under exact
        # matching, because a wrong unit already makes the value miss.
        partner = hit if hit is not None else _magnitude_partner(triple.value, candidates)
        verdict = same_unit(triple.value, partner, unit_match=unit_match) if partner is not None else None
        if verdict is True:
            unit = unit + Score(true_positives=1)
        elif verdict is False:
            unit = unit + Score(false_negatives=1)

    # Anything produced that answered nothing is a false positive. An arm that
    # empties the schema into every entity should not score well for it.
    value = value + Score(false_positives=max(len(produced) - len(matched_values), 0))

    wanted_properties = {t.prop for t in wanted}
    produced_properties = set(by_property)
    prop = Score(
        true_positives=len(wanted_properties & produced_properties),
        false_positives=len(produced_properties - wanted_properties),
        false_negatives=len(wanted_properties - produced_properties),
    )
    return value, prop, unit


def score_task(
    task: TaskRecord,
    produced: TripleSet,
    *,
    modes: dict[str, MatchMode] | None = None,
    subclasses: dict[str, set[str]] | None = None,
    unit_match: UnitMatch | None = None,
    shortlist: Collection[str] | None = None,
) -> TaskScore:
    """Score one arm's output for one task.

    ``modes`` loosens matching for named properties, and defaults to exact
    everywhere. ``subclasses`` maps a class path to its ancestors, so an
    instance of a subclass can satisfy an expectation when the task allows it.

    ``unit_match`` overrides how strictly the primary metric spells a unit.
    Left unset, the task decides: a task that asks for normalisation accepts a
    rescaled answer, and every other task wants the magnitude and the unit the
    document states. Physical equality is reported either way as its own
    dimension, so the gap is visible instead of load-bearing. It has to be
    visible, because an arm with no enum can emit ``MeV`` and an arm with one
    cannot, so a lenient primary metric hands the unconstrained arm marks the
    constrained arm has no way to earn.

    :attr:`Dimension.DUPLICATE` is counted over the entities that were found:
    each matched entity is one emission that answered something, and each
    leftover restating one of them is a surplus beside it. Nothing can be a
    false negative there, so the dimension asks only whether an entity was
    emitted once, and an answer that found nothing carries no duplicate score
    at all.
    """
    modes = modes or {}
    # Pooled across entities, matching what the fill step is actually sent.
    # Scoring it per entity would claim a precision the constraint does not
    # have, because one union covers every entity in the call.
    offered = None if shortlist is None else set(shortlist)
    if unit_match is None:
        unit_match = UnitMatch.PHYSICAL if task.accepts_conversion else UnitMatch.EXACT
    by_entity = {key: produced.for_entity(key) for key in produced.entities()}
    alignment = align(list(task.expected), by_entity, modes, produced.classes)
    matched = alignment.matched()
    by_entity = _resolve_references(by_entity, matched)

    result = TaskScore(task_id=task.id, alignment=alignment)
    entity = Score()
    class_score = Score()
    value = Score()
    prop = Score()
    unit = Score()
    unit_physical = Score()
    shortlisted = Score()
    provenance = Score()

    for instance in task.expected:
        if offered is not None:
            hit = instance.class_path in offered
            shortlisted = shortlisted + Score(true_positives=int(hit), false_negatives=int(not hit))
        produced_key = matched.get(instance.key)
        if produced_key is None:
            entity = entity + Score(false_negatives=1)
            class_score = class_score + Score(false_negatives=1)
            result.class_errors.append((instance.class_path, None))
            wanted = make_triples(instance.key, instance.fields)
            value = value + Score(false_negatives=len(wanted))
            prop = prop + Score(false_negatives=len({t.prop for t in wanted}))
            provenance = provenance + Score(false_negatives=1)
            continue

        entity = entity + Score(true_positives=1)
        if _class_matches(instance, produced.classes.get(produced_key), subclasses):
            class_score = class_score + Score(true_positives=1)
        else:
            class_score = class_score + Score(false_negatives=1)
            result.class_errors.append((instance.class_path, produced.classes.get(produced_key)))
        if produced_key in produced.provenance:
            provenance = provenance + Score(true_positives=1)
        else:
            provenance = provenance + Score(false_negatives=1)

        pair_value, pair_prop, pair_unit = _value_score(instance, by_entity[produced_key], modes, unit_match)
        value = value + pair_value
        prop = prop + pair_prop
        unit = unit + pair_unit
        _, _, pair_physical = _value_score(instance, by_entity[produced_key], modes, UnitMatch.PHYSICAL)
        unit_physical = unit_physical + pair_physical

    entity = entity + Score(false_positives=len(alignment.unmatched_produced))
    for key in alignment.unmatched_produced:
        value = value + Score(false_positives=len(by_entity[key]))

    result.dimensions = {
        Dimension.ENTITY: entity,
        Dimension.GROUNDED: _grounded_score(task, by_entity),
        Dimension.CLASS: class_score,
        Dimension.VALUE: value,
        Dimension.PROPERTY: prop,
        Dimension.UNIT: unit,
        Dimension.UNIT_PHYSICAL: unit_physical,
        Dimension.PROVENANCE: provenance,
    }
    if offered is not None:
        result.dimensions[Dimension.SHORTLIST] = shortlisted
    if matched:
        restated = duplicates(list(task.expected), by_entity, alignment, modes)
        result.dimensions[Dimension.DUPLICATE] = Score(
            true_positives=len(matched),
            false_positives=len(restated),
        )
    result.parse_errors = produced.parse_errors
    return result


def _resolve_references(
    by_entity: dict[str, frozenset[Triple]],
    matched: dict[str, str],
) -> dict[str, frozenset[Triple]]:
    """Rewrite a produced link to the expected entity it points at.

    A link is correct when it reaches the entity the expected link reaches,
    whatever either side called it. Alignment already rewrites the entity a
    triple is about; this does the same for an entity a triple points to,
    which is the only reason a graph can be scored at all.

    A reference to a produced entity that matched nothing is left alone. It
    cannot equal an expected key and should not, because pointing at an
    invented entity is a wrong link and not a missing one.
    """
    back = {produced: expected for expected, produced in matched.items()}
    if not back:
        return by_entity
    return {
        key: frozenset(
            triple.model_copy(update={"value": Reference(key=back[triple.value.key])})
            if isinstance(triple.value, Reference) and triple.value.key in back
            else triple
            for triple in triples
        )
        for key, triples in by_entity.items()
    }


def _grounded_score(task: TaskRecord, by_entity: dict[str, frozenset[Triple]]) -> Score:
    """How much of what was produced can be found in the document.

    Case-folded substring, over text values only. A number, a date or a unit
    is expected in a canonical form the document need not spell, so counting
    those would report the corpus's own normalisation as invention. See
    :attr:`~oold_llm_bench.grading.triples.Dimension.GROUNDED`.
    """
    document = (task.document or "").casefold()
    if not document:
        return Score()
    score = Score()
    for triples in by_entity.values():
        for triple in triples:
            value = triple.value
            if not isinstance(value, str) or not value.strip():
                continue
            if value.casefold().strip() in document:
                score = score + Score(true_positives=1)
            else:
                score = score + Score(false_positives=1)
    return score
