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
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from oold_llm_bench.grading.vocabulary import PropertyHierarchy

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


def _class_hit(
    expected: ExpectedInstance,
    produced_class: str | None,
    subclasses: dict[str, set[str]] | None,
    lineage: PropertyHierarchy,
) -> tuple[bool, bool]:
    """Strict and lenient class credit for one aligned pair.

    The lenient half is never false where the strict half is true, and adds
    an ancestor or a descendant of the expected class: `Person` for an
    `Actor` is under-specified, not invented, unless the instance itself
    declared discrimination as the question with `allow_subclass=False`.
    """
    hit = _class_matches(expected, produced_class, subclasses)
    if hit:
        return True, True
    near_hit = (
        expected.allow_subclass and produced_class is not None and lineage.near(produced_class, expected.class_path)
    )
    return False, near_hit


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


def _near_candidate(
    triple: Triple,
    by_property: dict[str, list[Scalar]],
    vocabulary: PropertyHierarchy,
    matched_values: set[tuple[str, int]],
    matched_near: set[tuple[str, int]],
    mode: MatchMode,
    unit_match: UnitMatch,
) -> tuple[str, Scalar] | None:
    """A produced value under a name the vocabulary calls broader or narrower
    than ``triple.prop``, not already spent on another match.

    Factored out of :func:`_value_score` because the strict loop it extends is
    already busy: this is the one extra idea in it, searched for once a wanted
    triple's own name turned up nothing.
    """
    return next(
        (
            (name, candidate)
            for name, items in by_property.items()
            if name != triple.prop and vocabulary.near(name, triple.prop)
            for candidate in items
            if (name, id(candidate)) not in matched_values
            and (name, id(candidate)) not in matched_near
            and same_value(triple.value, candidate, mode, unit_match=unit_match)
        ),
        None,
    )


def _value_score(
    expected: ExpectedInstance,
    produced: frozenset[Triple],
    modes: dict[str, MatchMode],
    unit_match: UnitMatch = UnitMatch.EXACT,
    vocabulary: PropertyHierarchy | None = None,
) -> tuple[Score, Score, Score, Score, Score]:
    """Value, property, unit, and their vocabulary-aware counterparts, for one
    aligned pair.

    The lenient two are never smaller than the strict ones: every match strict
    scoring makes, lenient scoring keeps, and a name the vocabulary calls
    broader or narrower than the expected one recovers what strict scoring
    missed. Without the recovery, a value correctly read and filed under a
    defensible synonym has no candidate to be found under at all, since the
    lookup is keyed on the expected name: it is charged a miss on the value it
    got right together with an invention on the name it used, for one answer.
    ``vocabulary`` absent scores exactly as before and the lenient two equal
    the strict two, which is why a caller with no hierarchy can ignore them.
    """
    wanted = make_triples(expected.key, expected.fields) | make_triples(expected.key, expected.optional_fields)
    optional_properties = {t.prop for t in make_triples(expected.key, expected.optional_fields)}
    by_property: dict[str, list[Scalar]] = {}
    for triple in produced:
        by_property.setdefault(triple.prop, []).append(triple.value)

    value = Score()
    value_near = Score()
    unit = Score()
    matched_values: set[tuple[str, int]] = set()
    matched_near: set[tuple[str, int]] = set()

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

        near_hit = (triple.prop, hit) if hit is not None else None
        if near_hit is None and vocabulary is not None:
            near_hit = _near_candidate(triple, by_property, vocabulary, matched_values, matched_near, mode, unit_match)
            if near_hit is not None:
                matched_near.add((near_hit[0], id(near_hit[1])))
        if near_hit is not None:
            value_near = value_near + Score(true_positives=1)
        elif triple.prop not in optional_properties:
            value_near = value_near + Score(false_negatives=1)

        # The unit is judged on its own candidate, found by physical equality.
        # Pinning it to the value match would erase unit failures under exact
        # matching, because a wrong unit already makes the value miss.
        unit = unit + _unit_verdict(triple, hit, candidates, unit_match)

    # Anything produced that answered nothing is a false positive. An arm that
    # empties the schema into every entity should not score well for it.
    value = value + Score(false_positives=max(len(produced) - len(matched_values), 0))
    value_near = value_near + Score(false_positives=max(len(produced) - len(matched_values) - len(matched_near), 0))

    prop, prop_near = _property_scores({t.prop for t in wanted}, set(by_property), vocabulary)
    return value, prop, unit, value_near, prop_near


def _unit_verdict(triple: Triple, hit: Scalar | None, candidates: list[Scalar], unit_match: UnitMatch) -> Score:
    """Unit conformance for one wanted triple, judged on its own candidate.

    Found by physical equality rather than pinned to the value match: a wrong
    unit already makes the value miss, and pinning the unit to it would erase
    unit failures under exact matching instead of counting them."""
    partner = hit if hit is not None else _magnitude_partner(triple.value, candidates)
    verdict = same_unit(triple.value, partner, unit_match=unit_match) if partner is not None else None
    if verdict is True:
        return Score(true_positives=1)
    if verdict is False:
        return Score(false_negatives=1)
    return Score()


def _property_scores(
    wanted_properties: set[str], produced_properties: set[str], vocabulary: PropertyHierarchy | None
) -> tuple[Score, Score]:
    """Strict and vocabulary-aware :attr:`Dimension.PROPERTY`, over the names
    one aligned pair used, with no vocabulary the lenient half equals the
    strict one."""
    prop = Score(
        true_positives=len(wanted_properties & produced_properties),
        false_positives=len(produced_properties - wanted_properties),
        false_negatives=len(wanted_properties - produced_properties),
    )
    if vocabulary is None:
        return prop, prop
    forgiven, taken_produced, taken_wanted = vocabulary.forgive(produced_properties, wanted_properties)
    prop_near = Score(
        true_positives=prop.true_positives + forgiven,
        false_positives=len(produced_properties - wanted_properties) - len(taken_produced),
        false_negatives=len(wanted_properties - produced_properties) - len(taken_wanted),
    )
    return prop, prop_near


@dataclass(frozen=True)
class _InstanceScores:
    """Every dimension's contribution from one expected instance."""

    entity: Score
    class_score: Score
    class_near: Score
    value: Score
    value_near: Score
    prop: Score
    prop_near: Score
    unit: Score
    unit_physical: Score
    provenance: Score
    class_error: tuple[str, str | None] | None
    """What to record in :attr:`TaskScore.class_errors`, or ``None`` where the
    class was right and there is nothing to explain."""


def _score_instance(
    instance: ExpectedInstance,
    produced_key: str | None,
    produced: TripleSet,
    by_entity: dict[str, frozenset[Triple]],
    modes: dict[str, MatchMode],
    unit_match: UnitMatch,
    subclasses: dict[str, set[str]] | None,
    lineage: PropertyHierarchy,
    vocabulary: PropertyHierarchy,
) -> _InstanceScores:
    """Every dimension's contribution from one expected instance, matched or
    not. One function rather than a branch in :func:`score_task`'s own loop:
    a miss and a hit score every dimension differently, and keeping both
    here is what lets that loop stay a sum over instances.
    """
    if produced_key is None:
        wanted = make_triples(instance.key, instance.fields)
        properties = len({t.prop for t in wanted})
        return _InstanceScores(
            entity=Score(false_negatives=1),
            class_score=Score(false_negatives=1),
            class_near=Score(false_negatives=1),
            value=Score(false_negatives=len(wanted)),
            value_near=Score(false_negatives=len(wanted)),
            prop=Score(false_negatives=properties),
            prop_near=Score(false_negatives=properties),
            unit=Score(),
            unit_physical=Score(),
            provenance=Score(false_negatives=1),
            class_error=(instance.class_path, None),
        )

    produced_class = produced.classes.get(produced_key)
    hit, near_hit = _class_hit(instance, produced_class, subclasses, lineage)
    pair_value, pair_prop, pair_unit, pair_value_near, pair_prop_near = _value_score(
        instance, by_entity[produced_key], modes, unit_match, vocabulary
    )
    _, _, pair_physical, _, _ = _value_score(instance, by_entity[produced_key], modes, UnitMatch.PHYSICAL)
    return _InstanceScores(
        entity=Score(true_positives=1),
        class_score=Score(true_positives=1) if hit else Score(false_negatives=1),
        class_near=Score(true_positives=1) if near_hit else Score(false_negatives=1),
        value=pair_value,
        value_near=pair_value_near,
        prop=pair_prop,
        prop_near=pair_prop_near,
        unit=pair_unit,
        unit_physical=pair_physical,
        provenance=Score(true_positives=1) if produced_key in produced.provenance else Score(false_negatives=1),
        class_error=None if hit else (instance.class_path, produced_class),
    )


def score_task(
    task: TaskRecord,
    produced: TripleSet,
    *,
    modes: dict[str, MatchMode] | None = None,
    subclasses: dict[str, set[str]] | None = None,
    unit_match: UnitMatch | None = None,
    shortlist: Collection[str] | None = None,
    vocabulary: PropertyHierarchy | None = None,
) -> TaskScore:
    """Score one arm's output for one task.

    ``vocabulary`` defaults to the built property hierarchy and is accepted
    explicitly for tests. Where it holds edges, :attr:`Dimension.VALUE_NEAR`
    and :attr:`Dimension.PROPERTY_NEAR` are reported beside the strict two: a
    value correctly read but filed under a name the vocabulary calls broader
    or narrower than the one expected is recovered there rather than charged
    as a miss on the value and an invention on the name, which is what the
    strict dimensions alone would do to it.

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
    from oold_llm_bench.grading.vocabulary import hierarchy_of, read_hierarchy

    modes = modes or {}
    hierarchy = vocabulary if vocabulary is not None else read_hierarchy()
    # A task's own lineage, not the property vocabulary: the catalogue a
    # corpus draws from is the hierarchy an answer is judged against.
    lineage = hierarchy_of(task.class_parents or {})
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
    class_near = Score()
    value = Score()
    value_near = Score()
    prop = Score()
    prop_near = Score()
    unit = Score()
    unit_physical = Score()
    shortlisted = Score()
    provenance = Score()

    for instance in task.expected:
        if offered is not None:
            hit = instance.class_path in offered
            shortlisted = shortlisted + Score(true_positives=int(hit), false_negatives=int(not hit))
        scored = _score_instance(
            instance, matched.get(instance.key), produced, by_entity, modes, unit_match, subclasses, lineage, hierarchy
        )
        entity = entity + scored.entity
        class_score = class_score + scored.class_score
        class_near = class_near + scored.class_near
        value = value + scored.value
        value_near = value_near + scored.value_near
        prop = prop + scored.prop
        prop_near = prop_near + scored.prop_near
        unit = unit + scored.unit
        unit_physical = unit_physical + scored.unit_physical
        provenance = provenance + scored.provenance
        if scored.class_error is not None:
            result.class_errors.append(scored.class_error)

    entity = entity + Score(false_positives=len(alignment.unmatched_produced))
    for key in alignment.unmatched_produced:
        value = value + Score(false_positives=len(by_entity[key]))
        value_near = value_near + Score(false_positives=len(by_entity[key]))

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
    if hierarchy.parents:
        result.dimensions[Dimension.VALUE_NEAR] = value_near
        result.dimensions[Dimension.PROPERTY_NEAR] = prop_near
    if lineage.parents:
        result.dimensions[Dimension.CLASS_NEAR] = class_near
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
