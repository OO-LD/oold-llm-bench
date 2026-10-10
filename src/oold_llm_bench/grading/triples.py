"""Triple schema.

Every arm's output, from free prose to a grounded OO-LD instance, is reduced
to the same set of ``(entity, property, value)`` triples before anything is
scored. One deterministic grader can then compare an arm that was
given a schema against one that was not, with no judge model in the loop.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

TRIPLE_SCHEMA_VERSION = "1"


class Dimension(str, Enum):
    """What a score is about.

    Reported separately instead of collapsed, because an arm can get every
    value right while assigning the wrong class, and that is a different
    finding from getting the values wrong.
    """

    ENTITY = "entity"
    """Whether each expected entity was found, and nothing else was emitted.

    Precision and recall answer different questions here and are reported
    apart from each other. Recall falls when an entity was missed, precision
    when one was invented or restated, and an F1 that moved could be any of
    the three. A document holding several entities cannot be diagnosed from
    one number."""
    CLASS = "class"
    CLASS_NEAR = "class_near"
    """:attr:`CLASS` with an ancestor or a descendant of the expected class
    accepted. `Person` for an `Actor` is under-specified, not mistaken, the
    same way a wide property name is under-specified rather than wrong, and
    answering one should not cost as much as naming an unrelated class.

    Read from the task's own `class_parents`, not from a vocabulary file:
    the lineage a corpus declares is the one its catalogue draws from, and
    that is the hierarchy an answer is being judged against.

    Absent when the task carries no lineage, because then it is :attr:`CLASS`
    under another name. Respects :attr:`ExpectedInstance.allow_subclass`: an
    instance that declared class discrimination itself as the question is not
    forgiven here either."""
    PROPERTY = "property"
    PROPERTY_NEAR = "property_near"
    """:attr:`PROPERTY` with a name the vocabulary calls broader or narrower
    than the expected one accepted, paired one-to-one against the names that
    would otherwise be misses so one wide name does not answer for two narrow
    ones. See :attr:`FILLABLE_NEAR`, which forgives the same relation at the
    step that names a slot before anything fills it; this is the same
    forgiveness where the slot was filled.

    Absent when no hierarchy is available, because then it is :attr:`PROPERTY`
    under another name."""
    VALUE = "value"
    VALUE_NEAR = "value_near"
    """:attr:`VALUE` where a value the document states was found, but filed
    under a name the vocabulary calls broader or narrower than the one the
    task expects.

    Needed beside :attr:`PROPERTY_NEAR` and not covered by it: the value
    lookup is keyed on the expected property name, so a value correctly read
    and filed under a defensible synonym had no candidate to be found under
    at all, charging a miss on the value together with the invention on
    :attr:`PROPERTY`. One answer, two dimensions, two counted errors.

    Absent when no hierarchy is available."""
    UNIT = "unit"
    """Conformance: the unit is spelled as the corpus spells it."""
    UNIT_PHYSICAL = "unit_physical"
    """Physical equality through pint, reported beside conformance so the gap
    between them is visible instead of being folded into one number."""
    SHORTLIST = "shortlist"
    """Whether a two-step orchestration ever offered the true class.

    Absent for a single-shot run. Without it, the select step never offering
    the class and the fill step choosing wrongly from a shortlist that held it
    are the same number, and the orchestration cannot be attributed."""
    DUPLICATE = "duplicate"
    """Whether every entity that was found was emitted once.

    Reads in the same direction as every other dimension: 1.00 is one
    emission per entity, and it falls as an entity is restated. Two predicted
    entities that answer the same expected one is the case it isolates.
    Without it, saying the right thing twice and inventing something both
    come off entity precision and cannot be told apart, and deduplicating
    output is not the same fix as not hallucinating.

    Absent when the alignment matched nothing, because a duplicate is defined
    against an entity that was found. Absent rather than zero: a dimension
    reporting zero where it does not apply would pull down every cell it
    appeared in."""
    MENTION = "mention"
    """Whether an entity was reported under words the document uses for it.

    The identify step answers a class and the text it read that class from,
    and the steps after it are told the entity by those words. A plan that
    finds the right number of entities and names them wrongly hands the next
    step a question about something else.

    Soft by construction: an entity may be referred to several ways and any
    of them counts, matched after the same folding
    :func:`~oold_llm_bench.grading.compare.normalise_text` applies elsewhere.
    A variance in the mention does not fail the entity. It is reported on its
    own, because the thing worth knowing is whether the words point at the
    right entity, not whether they are the words we happened to record."""
    FILLABLE = "fillable"
    """Whether the properties a document fills were named, before filling them.

    Precision and recall over the set, against the slots the corpus actually
    stated. Separate from :attr:`PROPERTY`, which scores what was filled:
    a step can name the right slots and the next one still get the values
    wrong, and naming slots the document never fills makes the extract schema
    demand values that are not there."""
    FILLABLE_NEAR = "fillable_near"
    """:attr:`FILLABLE` with a name the vocabulary calls broader or narrower
    than the expected one accepted.

    Reported beside the strict count, never instead of it. A catalogue
    offering `author` and `creator` together asks for a distinction the
    sentence "written by Jane Doe" does not draw, and Wikidata records P50 as
    a subproperty of P170, so answering the parent is a reading the vocabulary
    licenses. The gap between the two numbers is the share of the precision
    loss that belongs to the vocabulary rather than to the step.

    Absent when no hierarchy is available, because then it is the strict
    count under another name."""
    PATCH = "patch"
    """Whether a merge kept what was known, added what was new, and refused
    what contradicted.

    Three counts rather than one. A merge that preserves everything and adds
    nothing scores the same as one that overwrites, under any single number,
    and those are opposite failures."""
    PROVENANCE = "provenance"
    GROUNDED = "grounded"
    """Whether a produced text value can be found in the document it came from.

    Taken from Text2KGBench's ``subject_hallucination`` / ``object_hallucination``,
    which require the extracted string to appear in the source sentence. It is
    judge-free and it separates an error the triple F1 cannot: a value that is
    wrong because the model read the wrong thing, and a value that is wrong
    because the model invented it.

    **A rate, not a score.** A value the corpus states in one form and expects
    in another fails this by construction, which is most of the unit slot on
    Wiki-Measurements and every canonical date. So it is reported beside the
    other dimensions and never folded into them, and a cell's number is only
    comparable to another cell on the same corpus.
    """


class Quantity(BaseModel):
    """A magnitude with a unit, compared by physical equality.

    1.0 m and 100 cm are the same answer. Holding the unit separately is also
    what makes unit correctness scorable on its own.
    """

    model_config = ConfigDict(frozen=True)

    magnitude: float
    unit: str

    def __str__(self) -> str:
        return f"{self.magnitude} {self.unit}"


class Reference(BaseModel):
    """A value that is another entity rather than a literal.

    Without it a graph cannot be scored. A nested object read as its own
    entity drops the edge that pointed at it, and a reference written as an id
    becomes an ordinary string compared literally, which can never match
    because alignment rewrites produced keys to expected ones. Either failure
    leaves a perfect graph scoring the same as a correct bag of unlinked
    entities.

    The key is whichever side wrote it, and alignment rewrites a produced key
    to the expected key it was matched to, exactly as it does for the entity
    field of a triple. A link is then right when it points at the entity the
    expected link points at, whatever either side called it.
    """

    model_config = ConfigDict(frozen=True)

    key: str

    def __str__(self) -> str:
        return f"-> {self.key}"


Scalar = str | float | int | bool | Quantity | Reference | None


class Triple(BaseModel):
    """One asserted fact about one entity."""

    model_config = ConfigDict(frozen=True)

    entity: str
    """Entity key. Predictions carry the model's own key until alignment
    rewrites it to the expected key it was matched to."""
    prop: str
    value: Scalar

    @field_validator("prop")
    @classmethod
    def _normalised(cls, value: str) -> str:
        if value != normalise_property(value):
            raise ValueError(
                f"property {value!r} is not normalised; use normalise_property() before constructing a Triple"
            )
        return value

    def __str__(self) -> str:
        return f"({self.entity}, {self.prop}, {self.value})"


class TripleSet(BaseModel):
    """Triples extracted from one arm's output for one task."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = TRIPLE_SCHEMA_VERSION
    triples: frozenset[Triple]
    classes: dict[str, str]
    """Entity key to the class path claimed for it. Empty for arms that assign
    no class, which is the point of the unenforced arms and not a failure."""
    provenance: dict[str, str]
    """Entity key to the span of the document it was drawn from, where an arm
    reports one."""
    nested: frozenset[str] = frozenset()
    """Entities the answer wrote inside another entity's field.

    Embedded rather than named and pointed at. The schema decides which of
    the two a property takes, so the shape is itself something an arm gets
    right or wrong, and the triples no longer say which was used once they
    are flattened.

    Scored as any other entity. An embedded entity carries a class and
    properties, may point at others and may be pointed at, so nothing else
    about it differs.

    Empty for an answer that nested nothing, and for every record written
    before this was kept."""
    parse_errors: int = 0
    """How many fragments the extractor could not turn into a triple. For
    a prose arm this is the instrument's own error, reported next to the score."""

    def entities(self) -> frozenset[str]:
        """Every entity this answer asserts anything about.

        Union of the three sources, not triples alone: an entity answered by
        class with every property null produces no triple at all, and would
        otherwise never become a candidate for alignment to match against.
        `align.CLASS_AGREEMENT` exists precisely to let a right-class,
        wrong-or-missing-values answer align rather than score as though
        nothing were produced, and it cannot fire for an entity this method
        never surfaces. A class pinned at decode time and then filled with
        nothing is exactly the shape that exposed the gap."""
        return frozenset(t.entity for t in self.triples) | frozenset(self.classes) | frozenset(self.provenance)

    def for_entity(self, key: str) -> frozenset[Triple]:
        return frozenset(t for t in self.triples if t.entity == key)


def normalise_property(name: str) -> str:
    """Fold a property name to the form triples are keyed on.

    camelCase, PascalCase, kebab-case, spaced and snake_case spellings of the
    same property have to collide, or an arm is penalised for its output
    convention and not for its content.
    """
    out: list[str] = []
    for index, char in enumerate(name.strip()):
        if char in " -.":
            out.append("_")
        elif char.isupper():
            previous = name[index - 1] if index else ""
            if index and (previous.islower() or previous.isdigit()):
                out.append("_")
            out.append(char.lower())
        else:
            out.append(char)
    collapsed = "".join(out)
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed.strip("_")


def make_triple(entity: str, prop: str, value: Any) -> Triple:
    """Build a triple, normalising the property name."""
    return Triple(entity=entity, prop=normalise_property(prop), value=value)


def make_triples(entity: str, fields: dict[str, Any]) -> frozenset[Triple]:
    """Flatten one entity's fields into triples.

    Lists become one triple per element, so order never affects the score.
    Nested objects are not flattened here, because the extractor decides
    whether a nested object is a value or a separate entity.
    """
    out: set[Triple] = set()
    for prop, value in fields.items():
        for item in value if isinstance(value, (list, tuple, set)) else [value]:
            out.add(make_triple(entity, prop, item))
    return frozenset(out)


def merge(sets: Iterable[TripleSet]) -> TripleSet:
    """Combine triple sets, for arms that emit output in several steps."""
    triples: set[Triple] = set()
    classes: dict[str, str] = {}
    provenance: dict[str, str] = {}
    errors = 0
    for one in sets:
        triples |= set(one.triples)
        classes.update(one.classes)
        provenance.update(one.provenance)
        errors += one.parse_errors
    return TripleSet(
        triples=frozenset(triples),
        classes=classes,
        provenance=provenance,
        parse_errors=errors,
    )
