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
    PROPERTY = "property"
    VALUE = "value"
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
    parse_errors: int = 0
    """How many fragments the extractor could not turn into a triple. For
    a prose arm this is the instrument's own error, reported next to the score."""

    def entities(self) -> frozenset[str]:
        return frozenset(t.entity for t in self.triples)

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
