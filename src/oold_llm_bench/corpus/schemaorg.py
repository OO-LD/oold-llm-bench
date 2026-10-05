"""Tasks generated from the schema.org corpus by round trip.

The quantity corpus measures one reading at a time: a magnitude, a unit, a
class. Nothing is attached to a subject, so order carries no information and
no arm can get the wrong value onto the right entity. That is a deliberate
simplification and it leaves the most common extraction failure unmeasured.

schema.org supplies the missing shape. A class here is a subject with several
properties of different types at once, so a document states four or five facts
about one entity and an arm has to keep them together. The property slot is
also where the closed enumeration lives: ``bookFormat`` admits six members and
nothing else, which makes a wrong value detectable rather than merely odd, the
same property that made the unit enumeration worth leading with.

The vocabulary is the reason for the second variant. Every model has read
schema.org, so a native document can be answered partly from memory, and an
arm given no schema scores higher by whatever it remembers. Each
document is therefore emitted twice: once with the real class and property
names, once with the same structure and the same values under opaque
identifiers. The gap between the two measures what pretraining exposure is worth.

Labelling is the second axis, and it exists because the first one was not
enough. Every frame below interpolates the property it carries and the opening
interpolates the class, so a schema shown in the prompt states nothing the
document has not already stated. Measured over the 120 documents of ``g3b``:
the class is readable in 120 and all 480 property names appear. Under
:attr:`Labelling.IMPLIED` a document states its values and names nothing, and
the three decisions that took are these.

*What tells a reader which property a value fills.* The value's own form,
against the properties the catalogue lists for the class: a duration reads as a
duration, a URL as a URL, an enumeration member as a member of the one
enumeration that holds it. The forms are read as a partition rather than as the
subtype lattice schema.org declares, so a member of a closed list is not also a
candidate for a free-text slot. Integer and number are the one pair where the
specific form does not rule the general slot out, and they are folded into one
form for that reason.

*What happens to the slots that form cannot tell apart.* They are not drawn.
Drawing them and leaving them out of the answer would make a correct reading a
false positive; drawing them and keeping them would put a ceiling on the corpus
that moves per document and cannot be stated. The price is paid in the pool
instead, where it can be counted. Text is 1392 of the 3047 slots the
describable classes carry and a URL is another 562, so most classes declare
several of each and neither form is alone often. Of the 122 describable classes
only 24 keep three slots a value can point at, 19 of those also have a slot set
no other class in the catalogue could take, and 19 is the pool this mode draws
from; at four slots both numbers fall to 5. What survives is mostly closed
enumerations, 33 of the 67 slots, which is the property that made them worth
leading with.

*The class.* The opening frame does not name it. Otherwise class selection is
still read off the page, which is the differentiator the benchmark is built on.
A perfect answer then has to pick the class out of the catalogue by the forms
alone, so the corpus draws only slot sets that fit one class in the pool and no
other, and the ceiling stays at 1.00 rather than at the 181 of 240 an
unfiltered draw reaches. Measured over 120 documents against the same
catalogue as the labelled set: the class name is readable in none of them and
none of the 360 property names appears.

Notation is the third axis, and it moves the values rather than the names.
:func:`spell` wrote every value as the schema stores it, so a date reached the
page as ``2008-11-05``, a duration as ``PT3H18M`` and a boolean as ``true``.
Nothing in a document like that has to be normalised, and normalisation from a
surface form to a canonical value is most of what extraction is. Under
:attr:`~oold_llm_bench.corpus.quantities.Notation.WRITTEN` the same draw writes
``5 November 2008``, ``3 h 18 min`` and ``yes`` while the answer stays what it
was, so reaching it takes a rule instead of a copy.

What that is worth is measurable, and the measurement is the one the quantity
corpus made when it found 115 of its 120 documents answerable by transcription.
An arm that copies each value exactly as the page states it and normalises
nothing is perfect on 88 of 120 labelled documents under the canonical
notation and on 10 under the written one, 86 of 120 against 0 under the implied
one. The built-in spelling control cannot say this: it looks for a number and
the words beside it and submits them as a magnitude and a unit, which is not
the shape any schema.org answer takes, so it scores 0.00 under both notations
and separates neither.

*Which slots move.* Seven kinds of the eleven: date, datetime, time, duration,
boolean, number and integer. Text, URL and email are already what a document
writes, and rewriting them would invent a convention rather than follow one.
Enumeration members stay canonical too, and that is a decision and not an
oversight: ``EventScheduled`` has no published surface form, the corpus's
``x-oold-ui-enum-titles`` repeats the identifier and schema.org's
``rdfs:comment`` is a definition and not a phrase, so nothing deterministic
supplies one. Measured over the 122 describable classes, 272 of their 706
declared slots are of a kind this mode respells; over the 19 classes the
implied mode draws from, 21 of the 67 legible ones.

*Injectivity, one form at a time.* A date is ``5 November 2008``: day, month
name, four-digit year, and no leading zero on the day, so one string spells one
date. A datetime is ``13 May 2000 at 10:55:18 pm UTC``, which is that date form
and :func:`written_time` with the zone spelled out. A time is ``10:55:18 pm``,
a bijection onto the twenty-four-hour clock once midnight is ``12 am`` and noon
is ``12 pm``, which :func:`written_time` states rather than assumes. A duration
is ``3 h 18 min``, every component carrying its own label, which is what keeps
``1 mo`` and ``1 min`` apart where ISO 8601 spells both ``M``. A boolean is
``yes`` or ``no``. A number is ``76,037.1179``, digits grouped in threes, which
is typography and returns the value exactly. :data:`DECLINED` is the list of
forms a document does write and this module does not, each with the collision
that rules it out.

*The pool.* Unchanged, which is the difference from the quantity corpus. Every
kind the corpus draws has a written form, so no class leaves the pool and no
slot leaves a draw, and the cost of the mode is zero against the 79 quantity
kinds that lose theirs. The one thing measured rather than assumed is the
boolean: none of the 530 distinct enumeration members the describable classes
carry is ``yes`` or ``no``, so a boolean value still points at the boolean slot.

*What a written prompt has to carry.* The answer is the canonical value and the
document no longer shows it, so the kind of a slot has to reach the model. It
reaches it through the catalogue entry and nowhere else, which is the gap
:func:`_refuse_a_silent_catalogue` already closes for the implied mode, so a
written task with a catalogue is refused unless the catalogue is described.

*The implied mode.* Composes, and the reason is that :func:`legible_slots` and
:func:`identifying_sets` read the schema and never the page, so the pool of 19
classes is the same pool under either notation. What could still break is
whether a reader names the form of a value correctly, and the written forms
stay pairwise apart: a datetime carries a clock where a date does not, a time
carries a meridiem where a duration carries labelled components, and a grouped
number carries no word. This is where ``3:18`` would have cost something beyond
its own ambiguity, since it is also how a time reads.

The one pair that touches is a date inside a datetime, ``13 May 2000`` inside
``13 May 2000 at 10:55:18 pm UTC``. Each value still stands on the page in full
and the longer one carries a clock, so the two are told apart by what they
state and not by where they start, which is the relation ``2000-05-13`` and
``2000-05-13T22:55:18Z`` already had. It is also unreachable: no identifying
set in the pool of 19 carries a date and a datetime at once.

*The grader.* Unchanged, as it was for the quantity corpus. The expected value
is still canonical and :func:`~oold_llm_bench.grading.compare.same_value` still
compares it, so a correct answer is accepted without a variant list. One
leniency is worth naming: that function already strips digit separators before
comparing numbers, so ``"76,037.1179"`` passes for ``76037.1179`` and the two
numeric kinds add surface variation without adding a normalisation the score
depends on. Measured over 120 written documents, the kinds a transcriber still
gets right are text, email, an enumeration member, an integer and a number, and
the five that carry the mode are the date, the datetime, the time, the duration
and the boolean.

Linking is the fourth axis, and it is the first thing here that is not flat.
Every task under the three above expects one entity per record and no record
says anything about another, so multi-entity extraction was never exercised and
no ground truth asserted an edge. Under ``linked`` a draw is laid out as pairs,
a source and the target it reaches, and the expected answer carries a
:class:`~oold_llm_bench.grading.triples.Reference` where the source's link is.
An answer is then right on the edge when it reaches the entity the expected
edge reaches, whatever either side called that entity.

*How a reader knows which entity a property points at.* By a value. The
document names the target with one of the values the target's own record
shows, and the frame it sits in says it is a cross-reference rather than a
value of its own: "the {prop} column refers to the record of Calder Wynn",
with "Calder Wynn" standing on the Person record two sentences later. That is
the whole mechanism, and everything a reader cannot resolve by it is left
undrawn rather than drawn and forgiven, which is the decision
:attr:`Labelling.IMPLIED` and :func:`identifying_sets` already took.

*What that excludes, and why.* Seven things.

A value that is not text cannot name an entry. "The entry filed as 8852" is a
record number, "as 2008-11-05" a filing date, "as Hardcover" a category and "as
true" is not a phrase; each sends a reader somewhere other than the record it
meant. Code-like text goes with them, by the rule :data:`_CODE_HINTS` already
states: a designator that reads as an identifier invites an answer that copies
it as the link's value instead of resolving it, which is the one failure this
mode exists to tell apart from a correct link. Inherited text goes too, for the
reason :func:`legible_slots` gives: ``name`` is on every class, so a corpus
naming its entries by it would be naming Things. :func:`designating_slots` is
where those three are counted, and 87 of the 122 describable classes keep a
slot that can name them.

A link whose range is not in the draw reaches nothing on the page, and a link
whose range carries no designator reaches a record that cannot be named.
:func:`resolvable_links` drops both, and :func:`linked_classes` is the pool
that survives: 62 of the 122, their 524 declared links yielding 397 drawable
pairs over 38 distinct targets and 226 distinct relations. The declared-link
preference is :attr:`SchemaClass.own_links`, and it costs most of the corpus:
the same 122 classes carry 3,180 links once inheritance is counted, and drawing
those would describe most classes by ``CreativeWork``'s relations and make the
choice of class stop mattering, the argument :attr:`SchemaClass.own_slots`
makes for slots.

An enumeration member is a node in schema.org and a closed list here, and
reading it as a link would take the enumerations out of the slots that make
them worth asking about. A property whose range mixes a literal with a node
target stays a slot, because a document stating the literal satisfies it; one
property is one or the other, never both, or its ground truth would be
arguable.

:attr:`Labelling.IMPLIED` is refused outright. There a value points at its slot
by its form alone, and a designator has the form of the text values the target
already carries, so nothing on an implied page separates a value an entry holds
from a value that names another entry. :attr:`Notation.WRITTEN` composes
instead, and for a reason rather than by luck: text is one of the kinds that
mode leaves alone, so the designator reaches the page unchanged while the rest
of the record is respelled.

Two entries answering to one name is the last case, and it is a fact about the
values a seed drew rather than about the classes, so it cannot be ruled out in
the pool. Text values come from 65,536 proper nouns and
:func:`_refuse_a_shared_designator` checks every draw, as :func:`rename_map`
checks its identifiers.

*What it is worth, as a score.* A correct graph is 1.00. The same entities
with the links left out is 12 of 13 at three slots and two entities, because
two of the fourteen expected triples are edges and the rest are values, and
that gap is the finding the reference work exists for. An edge pointing at
another record on the page scores below a missing edge, 0.857 against 0.923 on
a four-entity draw, and an edge pointing at a record that is not on the page
scores below that again at 0.667, because an invented target is an invented
entity as well as a wrong link.

*The floor two linked documents share.* Stated as the labelled corpus states
its own. Over 200 pairs of documents the primary score of one document's answer
against another is zero in 195 and never above 0.143, and the edge itself
reaches the right entity in 5 of the 200, which happens when both draws took
the same class pair. A grid drawn from one pair alone therefore has an edge
that is answerable from the class structure and nothing else, and the
wrong-document control says so rather than the corpus hiding it.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any, TypeVar

from oold_llm_bench.corpus.quantities import Notation, canonical_number, readable_kind, written_number
from oold_llm_bench.grading.triples import Reference, Scalar, normalise_property
from oold_llm_bench.tasks.models import (
    CorpusRef,
    Difficulty,
    ExpectedInstance,
    Source,
    Split,
    TaskRecord,
    Variant,
)

__all__ = [
    "BOOLEAN_WORDS",
    "CATALOGUE_SLOTS",
    "DECLINED",
    "EXCLUDED_PROPERTIES",
    "FRAME_COUNT",
    "IMPLIED_FRAME_COUNT",
    "MONTHS",
    "Edge",
    "Entity",
    "Form",
    "Kind",
    "Labelling",
    "Link",
    "Notation",
    "SchemaClass",
    "Slot",
    "answer_schema",
    "branches_for",
    "canonical_value",
    "class_identifier",
    "class_phrase",
    "describable_classes",
    "designating_slots",
    "draw",
    "draw_linked",
    "expected_for",
    "form_of",
    "generate_pair",
    "generate_task",
    "identifying_sets",
    "implied_classes",
    "implied_values",
    "legible_slots",
    "link_schema",
    "linked_classes",
    "load_classes",
    "opaque_name",
    "property_identifier",
    "property_phrase",
    "rename_map",
    "render",
    "render_class",
    "render_entry",
    "resolvable_links",
    "seed_values",
    "spell",
    "written_value",
]

_SUFFIX = ".schema.json"
_MEMBER_POINTER = "#/$defs/member"
_LITERAL_TYPES = frozenset({"string", "number", "integer", "boolean"})

CATALOGUE_SLOTS = 12
"""How many of a class's properties the prompt material carries.

One number for the catalogue entry, the union branches and the answer schema,
because a property an implied document was built from has to be a property the
model was shown.
"""

EXCLUDED_PROPERTIES = frozenset({
    "type",
    "id",
    "identifier",
    "sameAs",
    "additionalType",
    "potentialAction",
    "subjectOf",
    "mainEntityOfPage",
    "image",
    "disambiguatingDescription",
})
"""Properties no document is built from.

``type`` and ``id`` are the serialisation, not the content. The rest are
Thing-level bookkeeping that every class inherits, so leaving them in would
make most tasks a test of the same handful of slots regardless of which class
was drawn.
"""


class Kind(str, Enum):
    """What a property slot holds, after the datatype leaf is resolved.

    Coarser than the schema: ``Float`` and ``Number`` are one kind here,
    because the distinction changes nothing about how a value is written down
    or how it is graded, while ``Date`` and ``Text`` change both.
    """

    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    DURATION = "duration"
    URL = "url"
    EMAIL = "email"
    ENUM = "enum"


class Labelling(str, Enum):
    """Whether a document names the fields it fills.

    ``NAMED`` is what every measured result rests on, so it is the default and
    the frames behind it are never touched. ``IMPLIED`` states the values and
    names neither the class nor the property, which leaves the schema in the
    prompt as the only thing that can attach a value to a slot.
    """

    NAMED = "named"
    IMPLIED = "implied"


class Form(str, Enum):
    """The shape a value can be recognised by, coarser than :class:`Kind`.

    A reader of an implied document names the form of a value and then looks
    for the slot the catalogue gives that form to. ``NUMBER`` and ``INTEGER``
    are one form here because an integer satisfies a number slot, so an integer
    value never rules the number slot out and the two could only be told apart
    by elimination over the whole entity.
    """

    TEXT = "text"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    DURATION = "duration"
    URL = "url"
    EMAIL = "email"
    ENUM = "enum"


_FORM_BY_KIND = {
    Kind.TEXT: Form.TEXT,
    Kind.NUMBER: Form.NUMERIC,
    Kind.INTEGER: Form.NUMERIC,
    Kind.BOOLEAN: Form.BOOLEAN,
    Kind.DATE: Form.DATE,
    Kind.DATETIME: Form.DATETIME,
    Kind.TIME: Form.TIME,
    Kind.DURATION: Form.DURATION,
    Kind.URL: Form.URL,
    Kind.EMAIL: Form.EMAIL,
    Kind.ENUM: Form.ENUM,
}


_KIND_BY_FORMAT = {
    "date": Kind.DATE,
    "date-time": Kind.DATETIME,
    "time": Kind.TIME,
    "duration": Kind.DURATION,
    "email": Kind.EMAIL,
    "uri": Kind.URL,
}


@dataclass(frozen=True)
class Slot:
    """One literal-valued property a class can carry."""

    name: str
    kind: Kind
    choices: tuple[str, ...] = ()
    """The members an enumerated slot admits. Closed, so a wrong value is
    detectable and not merely implausible, which is the property the unit
    enumeration gave the quantity corpus."""
    description: str = ""
    inherited: bool = False
    """Whether the class declares this slot or takes it from an ancestor."""


@dataclass(frozen=True)
class Link:
    """One property whose value is another entity rather than a literal.

    Held apart from :class:`Slot` because the two are graded differently. A
    slot is compared against what the document spells; a link is compared
    against the entity it reaches, whatever either side called that entity.
    """

    name: str
    ranges: tuple[str, ...]
    """The classes the property may point at, as the projection names them.

    A tuple because schema.org ranges are unions: ``author`` is a ``Person``
    or an ``Organization``, and keeping only the first would drop half the
    pairs a document could be built from."""
    description: str = ""
    inherited: bool = False


@dataclass(frozen=True)
class Edge:
    """One drawn link: where it points, and what the document calls that."""

    link: Link
    target: str
    """The key of the entity in the same draw that the link reaches."""
    designator: str
    """The value the document names the target by.

    Carried here as well as on the target, so one entity can be written down
    without the rest of the draw in hand."""


_Property = TypeVar("_Property", Slot, Link)
"""A slot or a link. The two are inherited by one rule and nothing else."""


@dataclass(frozen=True)
class SchemaClass:
    """One schema.org class, with the literal slots it can be described by."""

    name: str
    slots: tuple[Slot, ...]
    links: tuple[Link, ...] = ()
    parents: tuple[str, ...] = ()
    """Every class named in ``allOf``, in the order the schema names them.

    A tuple and not one name: 48 of the 901 classes declare two parents or
    three, and ``LocalBusiness`` is both an ``Organization`` and a ``Place``.
    Keeping only the last would lose every property of the first, which for
    ``LocalBusiness`` is the whole organisation half of the class."""
    label: str = ""
    description: str = ""
    iri: str = ""

    @property
    def parent(self) -> str | None:
        return self.parents[0] if self.parents else None

    @property
    def own_slots(self) -> tuple[Slot, ...]:
        """The slots this class declares itself.

        A task prefers these. Every class inherits Thing's, so drawing without
        a preference would describe most classes with the same four
        properties and the choice of class would stop mattering.
        """
        return tuple(slot for slot in self.slots if not slot.inherited)

    @property
    def own_links(self) -> tuple[Link, ...]:
        """The links this class declares itself.

        The preference :attr:`own_slots` states, for the same reason and with
        one more behind it: an inherited relation belongs to the ancestor that
        declared it, so a document built on one says which ancestor the subject
        is and not which class.
        """
        return tuple(link for link in self.links if not link.inherited)

    @property
    def has_slots(self) -> bool:
        return bool(self.slots)


@dataclass(frozen=True)
class DataType:
    """A datatype leaf reduced to the JSON type and format it ends at."""

    name: str
    json_type: str
    format: str = ""


def _resolve_datatypes(raw: dict[str, dict]) -> dict[str, DataType]:
    """Follow the datatype leaves down to the JSON type each one bottoms out at.

    schema.org models refinement by reference: ``Email`` is ``Text`` with a
    format, ``Integer`` is ``Number`` with one, ``URL`` and ``XPathType`` are
    both ``Text``. Reading only the declared ``type`` would classify all four
    as typeless and drop every property that uses them.
    """

    def walk(name: str, seen: frozenset[str]) -> DataType | None:
        schema = raw.get(name)
        if schema is None or name in seen:
            return None
        declared = schema.get("type")
        fmt = schema.get("format") or ""
        if isinstance(declared, str):
            return DataType(name=name, json_type=declared, format=fmt)
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.endswith(_SUFFIX):
            parent = walk(ref[: -len(_SUFFIX)], seen | {name})
            if parent is None:
                return None
            return DataType(name=name, json_type=parent.json_type, format=fmt or parent.format)
        return None

    resolved: dict[str, DataType] = {}
    for name in raw:
        datatype = walk(name, frozenset())
        if datatype is not None and datatype.json_type in _LITERAL_TYPES:
            resolved[name] = datatype
    return resolved


def _kind_of(datatype: DataType) -> Kind:
    if datatype.format in _KIND_BY_FORMAT:
        return _KIND_BY_FORMAT[datatype.format]
    if datatype.json_type == "boolean":
        return Kind.BOOLEAN
    if datatype.json_type == "integer":
        return Kind.INTEGER
    if datatype.json_type == "number":
        return Kind.NUMBER
    return Kind.TEXT


def _choices_in(node: dict) -> tuple[str, ...]:
    """The members of an enumeration, spelled as a document would spell them.

    The schema carries both ``schema:Hardcover`` and the title ``Hardcover``.
    The title is used, because ground truth a document does not
    contain is unanswerable, and no prose writes the prefixed form.
    """
    titles = node.get("x-oold-ui-enum-titles")
    if isinstance(titles, list) and titles:
        return tuple(str(title) for title in titles)
    return tuple(str(value).split(":", 1)[-1] for value in node.get("enum") or [])


def _enum_choices(raw: dict[str, dict], ref: str) -> tuple[str, ...]:
    source = ref.split("#", 1)[0]
    schema = raw.get(source[: -len(_SUFFIX)] if source.endswith(_SUFFIX) else source)
    return _choices_in(((schema or {}).get("$defs") or {}).get("member") or {})


def _slot_for(
    name: str,
    definition: dict,
    raw: dict[str, dict],
    datatypes: dict[str, DataType],
) -> Slot | None:
    """One property as a slot, or ``None`` when it holds no literal.

    A property whose range is another class is a node reference, and one whose
    range mixes literals with node targets is projected as a companion key
    ending in ``_text``. Neither is drawn from. A reference has nothing to
    write down, and the companion key is an artifact of the projection rather
    than schema.org vocabulary, so a document built on it would be asking the
    model about a name schema.org does not have.
    """
    if name in EXCLUDED_PROPERTIES or name.endswith("_text"):
        return None
    items = definition.get("items") if definition.get("type") == "array" else definition
    if not isinstance(items, dict):
        return None
    description = (definition.get("description") or "").strip()

    for branch in _branches(items):
        # A property whose range spans several enumerations carries the union
        # inline instead of referencing one member definition, so the inline
        # form is checked first or every such property would be lost.
        inline = _choices_in(branch)
        if inline:
            return Slot(name=name, kind=Kind.ENUM, choices=inline, description=description)
        ref = branch.get("$ref")
        if not isinstance(ref, str):
            continue
        if _MEMBER_POINTER in ref:
            choices = _enum_choices(raw, ref)
            if choices:
                return Slot(name=name, kind=Kind.ENUM, choices=choices, description=description)
            continue
        if not ref.endswith(_SUFFIX):
            continue
        datatype = datatypes.get(ref[: -len(_SUFFIX)])
        if datatype is not None:
            return Slot(name=name, kind=_kind_of(datatype), description=description)
    return None


def _branches(items: dict) -> list[dict]:
    """The alternatives a range offers, in the order the schema lists them.

    A range that mixes a literal with a node target, ``Number`` or
    ``MonetaryAmount`` for ``amount``, is a real schema.org range and a
    document stating the number satisfies it. Dropping the whole property
    because one branch is an object would lose the slot for the sake of a
    branch no prose would take anyway, so the first branch that resolves to a
    literal is the one the slot is built from.
    """
    alternatives = items.get("anyOf") or items.get("oneOf")
    if isinstance(alternatives, list):
        return [branch for branch in alternatives if isinstance(branch, dict)]
    return [items]


def _link_for(name: str, definition: dict) -> Link | None:
    """One property as a link, or ``None`` when nothing in it points at a node.

    A branch that is a bare IRI reference carrying ``x-oold-range`` and
    nothing else is a node target. A branch that also carries ``$ref`` or
    ``enum`` is an enumeration member, which schema.org models as a node and
    this corpus reads as a closed list. Reading those as links too would take
    the enumerations out of the slots that make them worth asking about, so
    the narrow shape is the one matched.
    """
    if name in EXCLUDED_PROPERTIES or name.endswith("_text"):
        return None
    items = definition.get("items") if definition.get("type") == "array" else definition
    if not isinstance(items, dict):
        return None
    ranges: list[str] = []
    for branch in _branches(items):
        declared = branch.get("x-oold-range")
        if declared is None or branch.get("$ref") or branch.get("enum"):
            continue
        for target in declared if isinstance(declared, list) else [declared]:
            if isinstance(target, str) and target.endswith(_SUFFIX):
                ranges.append(target[: -len(_SUFFIX)])
    if not ranges:
        return None
    return Link(
        name=name,
        ranges=tuple(dict.fromkeys(ranges)),
        description=(definition.get("description") or "").strip(),
    )


def _parents_of(schema: dict) -> tuple[str, ...]:
    above: list[str] = []
    for entry in schema.get("allOf") or []:
        ref = entry.get("$ref", "") if isinstance(entry, dict) else ""
        if isinstance(ref, str) and ref.endswith(_SUFFIX):
            above.append(ref[: -len(_SUFFIX)])
    return tuple(above)


def _declared_slots(schema: dict, raw: dict[str, dict], datatypes: dict[str, DataType]) -> tuple[Slot, ...]:
    found: list[Slot] = []
    for name, definition in (schema.get("properties") or {}).items():
        if not isinstance(definition, dict):
            continue
        slot = _slot_for(name, definition, raw, datatypes)
        if slot is not None:
            found.append(slot)
    return tuple(found)


def _declared_links(schema: dict, raw: dict[str, dict], datatypes: dict[str, DataType]) -> tuple[Link, ...]:
    """The node-valued properties a class declares.

    A property that resolves to a literal is never one of these, however many
    node targets its range also offers. ``amount`` is a ``Number`` or a
    ``MonetaryAmount`` and a document stating the number satisfies it, so it
    stays a slot and the corpus never asks for it both ways.
    """
    found: list[Link] = []
    for name, definition in (schema.get("properties") or {}).items():
        if not isinstance(definition, dict) or _slot_for(name, definition, raw, datatypes) is not None:
            continue
        link = _link_for(name, definition)
        if link is not None:
            found.append(link)
    return tuple(found)


def load_classes(directory: Path) -> list[SchemaClass]:
    """Read the corpus, resolving each class's effective set of slots.

    The parent chain is followed for the same reason ``load_kinds`` follows it
    for units. A subclass declares only what it adds, so reading the declared
    properties alone would leave ``Book`` without a name and ``Restaurant``
    without an address, and both of those are what a document about one
    actually says.

    Whether a slot was declared or inherited is kept, because it decides what
    a task should be built from: every class inherits the same Thing-level
    properties, so those identify nothing.

    Links are resolved the same way and kept beside the slots rather than
    among them, because the two are graded differently and a draw has to ask
    for one or the other.
    """
    raw = {}
    for path in sorted(directory.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        raw[schema.get("title") or path.stem] = schema

    parents = {name: _parents_of(schema) for name, schema in raw.items()}
    datatypes = _resolve_datatypes(raw)
    slots = {name: _declared_slots(schema, raw, datatypes) for name, schema in raw.items()}
    links = {name: _declared_links(schema, raw, datatypes) for name, schema in raw.items()}

    return [
        SchemaClass(
            name=name,
            slots=_effective(name, parents, slots),
            links=_effective(name, parents, links),
            parents=parents.get(name, ()),
            label=(schema.get("title") or name).strip(),
            description=(schema.get("description") or "").strip(),
            iri=str(schema.get("x-oold-iri") or ""),
        )
        for name, schema in raw.items()
        if name not in datatypes
    ]


def _effective(
    name: str,
    parents: dict[str, tuple[str, ...]],
    declared: dict[str, tuple[_Property, ...]],
    seen: frozenset[str] = frozenset(),
) -> tuple[_Property, ...]:
    """A class's own properties, followed by the ones it takes from above.

    Slots and links are inherited by the same rule, so one walk serves both.
    A name a class declares itself always wins over the ancestor's, and
    everything that arrives from above is marked so a draw can prefer what
    the class says about itself.
    """
    merged = list(declared.get(name, ()))
    taken = {item.name for item in merged}
    for parent in parents.get(name, ()):
        if parent in seen or parent not in declared:
            continue
        for item in _effective(parent, parents, declared, seen | {name}):
            if item.name in taken:
                continue
            taken.add(item.name)
            merged.append(replace(item, inherited=True))
    return tuple(merged)


def describable_classes(classes: list[SchemaClass], min_own_slots: int = 3) -> list[SchemaClass]:
    """The classes a document can be built from without falling back on Thing.

    A class with fewer declared slots than a task needs would be padded out
    with inherited ones, and a document padded that way describes Thing rather
    than the class it claims to, so nothing in it points at the answer.
    """
    return [cls for cls in classes if len(cls.own_slots) >= min_own_slots]


def form_of(slot: Slot) -> Form:
    return _FORM_BY_KIND[slot.kind]


def legible_slots(cls: SchemaClass) -> tuple[Slot, ...]:
    """The slots a value can point at without a label on it.

    Declared slots only, and judged against the declared slots alone, because
    the catalogue entry lists exactly those and they are the alternatives the
    prompt offers. Two slots of one form are both dropped rather than one kept:
    which of the pair a value belongs to is then a coin toss, and a task whose
    gold answer is arguable measures the annotator.

    Enumerations are compared by their members and not by the fact that both
    are enumerations. Two closed lists that never meet leave a member pointing
    at one slot, which is the whole reason the enumeration is worth having.
    """
    declared = cls.own_slots[:CATALOGUE_SLOTS]
    plain = [slot for slot in declared if slot.kind is not Kind.ENUM]
    enums = [slot for slot in declared if slot.kind is Kind.ENUM]
    alone = {form_of(slot) for slot in plain if sum(form_of(s) is form_of(slot) for s in plain) == 1}
    kept: list[Slot] = []
    for slot in declared:
        if slot.kind is Kind.ENUM:
            members = set(slot.choices)
            if members and not any(o.name != slot.name and members & set(o.choices) for o in enums):
                kept.append(slot)
        elif form_of(slot) in alone:
            kept.append(slot)
    return tuple(kept)


def _holds(offered: Slot, wanted: Slot) -> bool:
    """Whether a value written for ``wanted`` could be read as ``offered``."""
    if wanted.kind is Kind.ENUM:
        return offered.kind is Kind.ENUM and bool(set(offered.choices) & set(wanted.choices))
    return offered.kind is not Kind.ENUM and form_of(offered) is form_of(wanted)


def _assignable(wanted: tuple[Slot, ...], offered: tuple[Slot, ...]) -> bool:
    if not wanted:
        return True
    head, rest = wanted[0], wanted[1:]
    return any(
        _holds(slot, head) and _assignable(rest, offered[:index] + offered[index + 1 :])
        for index, slot in enumerate(offered)
    )


def identifying_sets(
    cls: SchemaClass,
    classes: list[SchemaClass],
    n_slots: int = 3,
) -> tuple[tuple[Slot, ...], ...]:
    """The slot sets whose forms fit this class and no other on offer.

    The analogue of :func:`~oold_llm_bench.corpus.quantities.unit_identifiable`
    for a corpus whose classes differ in their properties rather than in one
    enumeration. A document that names no class is answerable only where the
    forms it states rule every other class out, so the sets that do not are
    never drawn.

    Compared on the enumerations' members meeting rather than on the member a
    draw happened to take, so whether a set identifies its class is a fact
    about the corpus and not about the seed.
    """
    legible = legible_slots(cls)
    if len(legible) < n_slots:
        return ()
    others = [other for other in classes if other.name != cls.name]
    return tuple(
        wanted
        for wanted in combinations(legible, n_slots)
        if not any(_assignable(wanted, other.own_slots[:CATALOGUE_SLOTS]) for other in others)
    )


def designating_slots(cls: SchemaClass) -> tuple[Slot, ...]:
    """The slots whose value can stand in a sentence as the name of an entry.

    What a linked document needs of its target: a reader is sent to another
    record by a value, and finds it by seeing the same value on that record.
    The value therefore has to read as an appellation, and of the eleven kinds
    only text does. "The entry filed as 8852" is a record number, "as
    2008-11-05" is a filing date, "as Hardcover" is a category and "as true"
    is not a phrase; each of those sends a reader somewhere other than the
    entry it meant.

    Code-like text goes with them, by the rule :data:`_CODE_HINTS` already
    states. "The entry filed as AB-12345" names a catalogue number rather than
    a thing, and a designator that reads as an identifier invites an answer
    that copies it as the link's value instead of resolving it, which is the
    one failure this mode exists to separate from a correct link.

    Declared slots only, as :func:`legible_slots` takes them and for the same
    reason: ``name`` is on every class, so a document that named its entries
    by it would be naming Things.
    """
    return tuple(slot for slot in cls.own_slots[:CATALOGUE_SLOTS] if slot.kind is Kind.TEXT and not _is_code(slot))


def resolvable_links(cls: SchemaClass, classes: list[SchemaClass]) -> tuple[tuple[Link, SchemaClass], ...]:
    """Each of this class's links paired with a target a document can name.

    One entry per range, because a union range is several drawable pairs and
    collapsing it to the first would lose the rest. A range that names no
    class in ``classes`` is dropped: the target has to be an entity the
    document actually writes down, or the link reaches nothing on the page.
    """
    by_name = {other.name: other for other in classes}
    return tuple(
        (link, target)
        for link in cls.own_links
        for name in link.ranges
        if (target := by_name.get(name)) is not None and designating_slots(target)
    )


def linked_classes(classes: list[SchemaClass]) -> list[SchemaClass]:
    """The classes a linked document can be built from.

    The analogue of :func:`implied_classes` and of
    :func:`~oold_llm_bench.corpus.quantities.writable`: the price of the mode
    is paid in the pool, where it can be counted, rather than in a ceiling
    that moves per document.

    Measured over the 122 describable classes of the schema.org corpus: 87
    carry a slot that can name them, 62 declare a link whose range is one of
    those 87, and those 62 are the pool. The 524 declared links between them
    yield 397 drawable pairs over 38 distinct targets.
    """
    return [cls for cls in classes if resolvable_links(cls, classes)]


def implied_classes(
    classes: list[SchemaClass],
    n_slots: int = 3,
    against: list[SchemaClass] | None = None,
) -> list[SchemaClass]:
    """The classes an implied document can be built from.

    ``against`` is the catalogue the answer has to be told apart from, which is
    wider than the pool whenever part of the corpus is held out of the draw. It
    defaults to ``classes``, the case where every offered class is also a
    possible answer.

    Measured over the 122 describable classes of the schema.org corpus: 24
    keep three slots a value can point at, 19 of those also name themselves,
    and at four slots both numbers fall to 5. That is the price of the mode and
    it is a pool statistic rather than a ceiling, which is why the slots it
    drops are dropped here instead of being drawn and forgiven later.
    """
    catalogue = classes if against is None else against
    return [cls for cls in classes if identifying_sets(cls, catalogue, n_slots)]


_RENAME_SALT = "oold-llm-bench/schemaorg"


def opaque_name(name: str, prefix: str, salt: str = _RENAME_SALT) -> str:
    """A stable opaque identifier for one schema.org name.

    Derived from the name and not from the task seed. A per-task renaming
    would give one class a different identifier in every document, so a
    catalogue could not be shared across a task set and each arm would be
    measured against a moving target.

    The shape follows Wikidata's, ``Q`` for a class and ``p`` for a property,
    because that is an opaque vocabulary a community actually uses. Reusing
    its shape keeps the renamed arm looking at something real instead of at an
    obviously artificial scramble.
    """
    digest = hashlib.blake2s(f"{salt}:{name}".encode(), digest_size=5).hexdigest()
    return f"{prefix}{digest}"


def class_identifier(name: str, variant: Variant) -> str:
    return name if variant is Variant.NATIVE else opaque_name(name, "Q")


def property_identifier(name: str, variant: Variant) -> str:
    return name if variant is Variant.NATIVE else opaque_name(name, "p")


def rename_map(names: list[str], prefix: str, salt: str = _RENAME_SALT) -> dict[str, str]:
    """Opaque identifiers for a set of names, refusing a collision.

    Two names folded onto one identifier would silently merge two classes into
    one answer, and the score would look like a class error that never
    happened. Forty bits over a corpus of a few thousand names makes this
    unlikely, which is exactly why it has to be checked rather than assumed.
    """
    mapping: dict[str, str] = {}
    taken: dict[str, str] = {}
    for name in names:
        identifier = opaque_name(name, prefix, salt)
        clash = taken.get(identifier)
        if clash is not None and clash != name:
            raise ValueError(f"opaque identifier {identifier} is claimed by both {clash} and {name}")
        taken[identifier] = name
        mapping[name] = identifier
    return mapping


_GIVEN = (
    "Cal",
    "Ner",
    "Vol",
    "Hes",
    "Mar",
    "Tor",
    "Brin",
    "Sel",
    "Qar",
    "Fen",
    "Oss",
    "Lin",
    "Dav",
    "Rho",
    "Mick",
    "Tav",
)
_ENDING = (
    "der",
    "vin",
    "ast",
    "ric",
    "mon",
    "quel",
    "ton",
    "rith",
    "vale",
    "dros",
    "mer",
    "lin",
    "bec",
    "nor",
    "sen",
    "tay",
)
_LETTERS = "BCDFGHJKLMNPQRSTVWXZ"

_CODE_HINTS = ("isbn", "issn", "ismn", "iswc", "sku", "gtin", "duns", "code", "number", "serial")
"""Property names whose text value is an identifier rather than a name.

Matched on the name because the schema does not say it: ``isbn`` and
``alternateName`` are both ``Text``. Getting one wrong costs nothing that
matters, since the corpus defines its own ground truth, but "the isbn is
Calder Wynn" reads as a generation artefact and invites a model to treat the
whole document as noise.
"""


def _word(rng: random.Random) -> str:
    return rng.choice(_GIVEN) + rng.choice(_ENDING)


def _proper_noun(rng: random.Random) -> str:
    return f"{_word(rng)} {_word(rng)}"


def _code(rng: random.Random) -> str:
    letters = "".join(rng.choice(_LETTERS) for _ in range(2))
    return f"{letters}-{rng.randrange(10000, 99999)}"


def _slug(rng: random.Random) -> str:
    return _word(rng).lower()


def _is_code(slot: Slot) -> bool:
    lowered = slot.name.lower()
    return any(hint in lowered for hint in _CODE_HINTS)


_NAME_TOKENS = frozenset({
    "name",
    "surname",
    "nickname",
    "alias",
    "pseudonym",
    "epithet",
    "honorific",
    "title",
    "headline",
    "label",
    "brand",
    "byline",
    "author",
    "creator",
    "founder",
    "publisher",
    "provider",
    "manufacturer",
    "owner",
    "seller",
    "vendor",
    "sponsor",
})
"""Name tokens, for the text slots whose value really is an appellation.

Matched on a camel-case token and not a substring, because ``width`` contains
"id" and ``surname`` contains "name" and only one of those is intended.
"""


def _tokens(name: str) -> list[str]:
    """``paymentMethodId`` as ``payment method id``, so a hint can be exact."""
    return [token.lower() for token in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", name)]


def _is_name(slot: Slot) -> bool:
    return bool(set(_tokens(slot.name)) & _NAME_TOKENS)


def _phrase(rng: random.Random) -> str:
    """A text value that cannot be read as the name of a thing.

    Lower case and hyphenated, so it has the form of a category token. That is
    the whole purpose: a reader can tell it from an appellation at a glance,
    and a model asked which entities a document describes will not offer it as
    one.

    Drawn from the same 65,536 combinations the appellation form uses, so the
    collision arithmetic :func:`_refuse_a_shared_designator` relies on is
    unchanged.
    """
    return f"{_word(rng).lower()}-{_word(rng).lower()}"


def _text_value(slot: Slot, rng: random.Random, *, appellation: bool = False) -> str:
    """Values are invented, not borrowed from the world.

    A value a model can guess from what it already knows does not measure
    reading, which is the same reason the quantity corpus draws its magnitudes
    at random instead of quoting real measurements.

    Two words at most, and never a clause. A value that runs on has no
    boundary a reader can point at inside a sentence, and a task whose gold
    value is arguable measures the annotator.

    Three forms, and which one a slot takes is decided by its name, because the
    schema does not say: ``isbn`` and ``alternateName`` are both ``Text``. A
    code where the name says identifier, an appellation where it says name,
    and a category token everywhere else.

    **The default is the category token and not the appellation.** With the
    appellation as default, ``award`` reads "Rhosen Linlin", ``asin`` reads
    "Davquel Ossder" and ``accessibilityAPI`` reads as a person's name: over
    120 linked tasks, 45 per cent of all literal values come out as two
    capitalised words across 94 distinct slots. A document whose values all
    read as names does not let a reader tell a value from the designator of a
    linked entity, and a detect step duly returns one entity per name-shaped
    value: 46 of 49 extra entities over 30 documents have a literal for their
    mention.

    ``appellation`` overrides the name rule for the one slot a draw designates
    its target by. That value has to read as an appellation whatever the slot
    is called, because the document names the target by it, and it is the only
    text on the page that should.
    """
    if _is_code(slot):
        return _code(rng)
    if appellation or _is_name(slot):
        return _proper_noun(rng)
    return _phrase(rng)


_VALUE_BY_KIND: dict[Kind, Callable[[random.Random], Scalar]] = {
    Kind.BOOLEAN: lambda rng: rng.choice((True, False)),
    Kind.INTEGER: lambda rng: rng.randrange(2, 9999),
    Kind.NUMBER: lambda rng: round(rng.uniform(0.5, 5000.0), 2),
    Kind.DATE: lambda rng: _date(rng),
    Kind.DATETIME: lambda rng: f"{_date(rng)}T{_clock(rng)}Z",
    Kind.TIME: lambda rng: _clock(rng),
    Kind.DURATION: lambda rng: f"PT{rng.randrange(1, 23)}H{rng.randrange(1, 59)}M",
    Kind.URL: lambda rng: f"https://{_slug(rng)}.example.org/{_slug(rng)}",
    Kind.EMAIL: lambda rng: f"{_slug(rng)}@{_slug(rng)}.example.org",
}


def _value_for(slot: Slot, rng: random.Random, *, appellation: bool = False) -> Scalar:
    """One value for one slot.

    A boolean has two values and a small enumeration has a handful, so two
    unrelated documents agree on such a slot now and then and an answer copied
    from the wrong document does not always score exactly zero. Measured over
    900 pairs the floor is zero in 894 and never above 0.25. That is a
    property of the value space and not of the generator, so it is stated
    rather than engineered away: suppressing the closed-vocabulary slots would
    remove the ones the enumeration makes worth asking about.

    :attr:`Labelling.IMPLIED` sits higher, because its pool is 19 classes and
    two thirds of what it draws is an enumeration. Measured over 120 pairs the
    floor is zero in 117, 0.33 in two and 0.67 in one. Same cause, and the
    same reason for leaving it: the mode exists because the enumerations are
    what a schema can still say.
    """
    if slot.kind is Kind.ENUM:
        return rng.choice(slot.choices)
    maker = _VALUE_BY_KIND.get(slot.kind)
    return maker(rng) if maker is not None else _text_value(slot, rng, appellation=appellation)


def _date(rng: random.Random) -> str:
    return f"{rng.randrange(1998, 2025)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"


def _clock(rng: random.Random) -> str:
    return f"{rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}:{rng.randrange(0, 60):02d}"


@dataclass(frozen=True)
class Entity:
    """One instance to be written down, before anything decides how."""

    key: str
    cls: SchemaClass
    values: tuple[tuple[Slot, Scalar], ...]
    edges: tuple[Edge, ...] = ()
    """The links this entity asserts, empty for every unlinked draw."""


def seed_values(
    cls: SchemaClass,
    rng: random.Random,
    n_slots: int,
    must_include: Slot | None = None,
) -> tuple[tuple[Slot, Scalar], ...]:
    """Fill a class's slots with values drawn from one seeded generator.

    Declared slots are taken before inherited ones, and the slots are then
    ordered as the schema declares them so the document reads as a record
    rather than as a shuffle. Two slots whose names fold onto the same graded
    property would be one answer with two values, so the second is dropped.

    ``must_include`` pins one slot into the draw. A linked document names its
    target by a value the target's own record has to show, so which slot that
    value comes from cannot be left to the sample.

    The pool is capped at :data:`CATALOGUE_SLOTS`, the same cap the catalogue
    entry, the union branches and the answer schema already carry. Drawing
    past it put 36 of 480 expected fields outside the schema the arm was
    given, so 26 of 120 tasks were unanswerable under a union before a model
    saw them and every named union-against-flat figure was biased against the union by
    0.060. The implied mode never had it, because `legible_slots` caps first.
    """
    pool = list(cls.own_slots[:CATALOGUE_SLOTS])
    if len(pool) < n_slots:
        pool += [slot for slot in cls.slots if slot.inherited][: max(0, CATALOGUE_SLOTS - len(pool))]
    if len(pool) < n_slots:
        raise ValueError(f"{cls.name} offers {len(pool)} slots, asked for {n_slots}")

    chosen: list[Slot] = []
    taken: set[str] = set()
    if must_include is not None:
        chosen.append(must_include)
        taken.add(normalise_property(must_include.name))
    for slot in rng.sample(pool, len(pool)):
        graded = normalise_property(slot.name)
        if graded in taken:
            continue
        taken.add(graded)
        chosen.append(slot)
        if len(chosen) == n_slots:
            break
    if len(chosen) < n_slots:
        raise ValueError(f"{cls.name} offers {len(chosen)} distinct slots, asked for {n_slots}")

    order = {slot.name: index for index, slot in enumerate(cls.slots)}
    chosen.sort(key=lambda slot: order.get(slot.name, len(order)))
    naming = must_include.name if must_include is not None else None
    return tuple((slot, _value_for(slot, rng, appellation=slot.name == naming)) for slot in chosen)


def implied_values(
    cls: SchemaClass,
    rng: random.Random,
    n_slots: int,
    against: list[SchemaClass],
) -> tuple[tuple[Slot, Scalar], ...]:
    """Fill a slot set that names this class and no other in ``against``.

    The slots stay in the order they were drawn rather than being sorted into
    the class's declaration order. With no label on a value its position is the
    only other channel there is, and a document that states its values in
    schema order hands back a key this mode exists to withhold.
    """
    usable = identifying_sets(cls, against, n_slots)
    if not usable:
        raise ValueError(f"{cls.name} offers no {n_slots} slots that tell it apart from the rest of the catalogue")
    return tuple((slot, _value_for(slot, rng)) for slot in rng.choice(usable))


def draw(
    classes: list[SchemaClass],
    rng: random.Random,
    n_entities: int,
    n_slots: int,
    labelling: Labelling = Labelling.NAMED,
    against: list[SchemaClass] | None = None,
    linked: bool = False,
) -> list[Entity]:
    """Choose the classes and fill their slots, before any prose exists.

    Split from rendering so one draw can be written down twice. The native and
    the renamed document have to carry identical values, or the contrast
    between them measures the values as well as the vocabulary.

    Under :attr:`Labelling.IMPLIED` the class has to be recoverable from the
    forms alone, so only classes with an identifying slot set are usable and
    ``against`` names the catalogue they have to be told apart from.

    ``linked`` lays the draw out as pairs instead of as independents, and what
    that costs is :func:`draw_linked`.
    """
    if linked:
        return draw_linked(classes, rng, n_entities, n_slots, labelling)

    if labelling is Labelling.IMPLIED:
        catalogue = classes if against is None else against
        usable = implied_classes(classes, n_slots, catalogue)
        if not usable:
            raise ValueError(f"no schema.org class offers {n_slots} slots its values alone identify")
        if n_entities > len(usable):
            raise ValueError(f"asked for {n_entities} entities, corpus offers {len(usable)}")
        return [
            Entity(key=f"e{index}", cls=cls, values=implied_values(cls, rng, n_slots, catalogue))
            for index, cls in enumerate(rng.sample(usable, n_entities), start=1)
        ]

    usable = [cls for cls in classes if len(cls.slots) >= n_slots]
    if not usable:
        raise ValueError(f"no schema.org class offers {n_slots} literal slots")
    if n_entities > len(usable):
        raise ValueError(f"asked for {n_entities} entities, corpus offers {len(usable)}")
    return [
        Entity(key=f"e{index}", cls=cls, values=seed_values(cls, rng, n_slots))
        for index, cls in enumerate(rng.sample(usable, n_entities), start=1)
    ]


def draw_linked(
    classes: list[SchemaClass],
    rng: random.Random,
    n_entities: int,
    n_slots: int,
    labelling: Labelling = Labelling.NAMED,
) -> list[Entity]:
    """Lay the draw out as pairs, each one a source and the target it reaches.

    Pairs and not a free graph, so the edge count follows from the entity
    count: one edge at two entities or three, two at four or five. An odd
    entity is drawn unlinked and is there to be told apart from the target.

    :attr:`Labelling.IMPLIED` is refused rather than supported, and the reason
    is the mode's own rule. There a value points at its slot by its form, and
    a designator has the form of the text values the target already carries,
    so nothing on the page separates a value an entry holds from a value that
    names another entry. The link would not be recoverable, and drawing it
    anyway is the ceiling :func:`legible_slots` declines to accept.
    """
    if labelling is Labelling.IMPLIED:
        raise ValueError("a linked document names the property it links on, which the implied mode withholds")
    if n_entities < 2:
        raise ValueError(f"a link needs two entities, asked for {n_entities}")
    sources = linked_classes(classes)
    if not sources:
        raise ValueError("no schema.org class links to a class a document can name")

    entities: list[Entity] = []
    for index in range(1, n_entities - n_entities % 2, 2):
        cls = rng.choice(sources)
        link, target_cls = rng.choice(resolvable_links(cls, classes))
        naming = rng.choice(designating_slots(target_cls))
        target = Entity(
            key=f"e{index + 1}",
            cls=target_cls,
            values=seed_values(target_cls, rng, n_slots, must_include=naming),
        )
        designator = next(spell(value) for slot, value in target.values if slot.name == naming.name)
        entities.append(
            Entity(
                key=f"e{index}",
                cls=cls,
                values=seed_values(cls, rng, n_slots),
                edges=(Edge(link=link, target=target.key, designator=designator),),
            )
        )
        entities.append(target)
    if n_entities % 2:
        spare = [cls for cls in classes if len(cls.slots) >= n_slots]
        if not spare:
            raise ValueError(f"no schema.org class offers {n_slots} literal slots")
        cls = rng.choice(spare)
        entities.append(Entity(key=f"e{n_entities}", cls=cls, values=seed_values(cls, rng, n_slots)))
    _refuse_a_shared_designator(entities)
    return entities


def _refuse_a_shared_designator(entities: list[Entity]) -> None:
    """A name two entries answer to names neither of them.

    The one ambiguity the pool rules cannot rule out, because it is a fact
    about the values a seed happened to draw rather than about the classes.
    Text values come from 65,536 proper nouns, so a clash is unlikely, which
    is exactly why it is checked here rather than assumed, as
    :func:`rename_map` checks its identifiers.
    """
    holders: dict[str, set[str]] = {}
    for entity in entities:
        for _, value in entity.values:
            holders.setdefault(spell(value), set()).add(entity.key)
    for edge in (edge for entity in entities for edge in entity.edges):
        named = holders.get(edge.designator, set())
        if named != {edge.target}:
            raise ValueError(f"the name {edge.designator!r} is carried by {sorted(named)}, not by {edge.target} alone")


_OPENING = (
    "This record is of type {cls}.",
    "The entry below was filed as {cls}.",
    "Filed under {cls}.",
    "What follows was catalogued as {cls}.",
    "One entry of type {cls} was added to the register.",
    "The register lists this under {cls}.",
    "The clerk recorded it as {cls}.",
    "Next in the file, type {cls}.",
    "The sheet covers a single entry of type {cls}.",
    "The record type is {cls}.",
    "Logged as {cls}.",
    "This is the {cls} entry.",
)

_GENERIC = (
    "Its {prop} is {value}.",
    "The {prop} is given as {value}.",
    "{prop}: {value}.",
    "It carries the {prop} {value}.",
    "The {prop} reads {value}.",
    "For {prop}, the entry says {value}.",
    "Under {prop} the sheet has {value}.",
    "The {prop} was noted as {value}.",
    "Recorded {prop}: {value}.",
    "The entry gives {value} for its {prop}.",
    "{value} is what the {prop} field holds.",
    "The clerk entered {value} as the {prop}.",
    "Its {prop} came out as {value}.",
    "The {prop} on file is {value}.",
    "As for the {prop}, it is {value}.",
    "The {prop} column shows {value}.",
)

_TEXTUAL = (
    "It goes by the {prop} {value}.",
    "The {prop} is spelled {value}.",
    "Written on the form under {prop} is {value}.",
    "The {prop} given is {value}.",
    "Someone wrote {value} in the {prop} field.",
    "Its {prop} appears as {value}.",
)

_NUMERIC = (
    "The {prop} counts {value}.",
    "It has {value} for {prop}.",
    "The {prop} came to {value}.",
    "{prop} totals {value}.",
    "The figure under {prop} is {value}.",
    "Counted {value} for {prop}.",
)

_TEMPORAL = (
    "The {prop} falls on {value}.",
    "Its {prop} is dated {value}.",
    "The {prop} is recorded as {value}.",
    "{prop} was {value}.",
    "The date under {prop} is {value}.",
    "It was stamped {value} for {prop}.",
)

_FLAG = (
    "The {prop} flag is {value}.",
    "For {prop} the answer is {value}.",
    "{prop} is marked {value}.",
    "The box for {prop} reads {value}.",
    "Its {prop} is set to {value}.",
    "Under {prop} the form says {value}.",
    "The {prop} entry is {value}.",
    "{prop}: {value}.",
)

_CHOSEN = (
    "The {prop} chosen was {value}.",
    "From the {prop} list, {value}.",
    "It was classified under {prop} as {value}.",
    "The {prop} selected is {value}.",
    "{prop} was set to {value}.",
)

_SEPARATED = (
    "The entry shows {value}. That is its {prop}.",
    "{value} appears near the top. It belongs to the {prop} field.",
    "One line reads {value}. The column it sits in is {prop}.",
    "The form has {value} written on it, and the heading above it is {prop}.",
    "Someone noted {value}. Which field that was is given below as {prop}.",
    "There is {value} on the sheet. The {prop} field is where it goes.",
    "The value {value} was entered. Its label, further down, is {prop}.",
    "Copied across: {value}. The heading for it reads {prop}.",
)
"""Frames that put the value and its label in different sentences.

Difficulty moves distance and nothing else. Harder prose separates the two and
surrounds them with text that carries no answer. It never makes the answer
ambiguous, because a task whose gold value is arguable measures the annotator.
"""

_LINKED = (
    "Its {prop} is the entry filed as {value}.",
    "The {prop} refers to the entry for {value}.",
    "Under {prop} the sheet points at the entry for {value}.",
    "The {prop} names another entry on the same file, {value}.",
    "For {prop} the clerk cross-referenced the entry for {value}.",
    "The {prop} column refers to the record of {value}.",
    "{prop}: see the entry for {value}.",
    "The {prop} was entered as a cross-reference to {value}.",
)
"""Frames that send a reader to another entry rather than stating a value.

Every one of them says so in as many words. A frame reading "its {prop} is
{value}" would be the generic one, and a reader could not tell the link from a
text slot that happens to hold a proper noun, which is the whole of what a
linked document asks.
"""

_FILLER = (
    "Nothing else on the sheet was legible.",
    "The rest of the page was left blank.",
    "A second clerk countersigned the entry without comment.",
    "The file was closed the same afternoon.",
    "No correction was made afterwards.",
    "The entry was copied into the ledger unchanged.",
    "Two earlier drafts of the same record were discarded.",
    "The remaining columns were struck through.",
)

FRAME_COUNT = (
    len(_OPENING)
    + len(_GENERIC)
    + len(_TEXTUAL)
    + len(_NUMERIC)
    + len(_TEMPORAL)
    + len(_FLAG)
    + len(_CHOSEN)
    + len(_SEPARATED)
    + len(_LINKED)
    + len(_FILLER)
)
"""How many sentence frames the corpus writes from.

Recorded because the quantity corpus has sixteen, and sixteen frames over
thousands of documents means an arm can learn the phrasing instead of reading
it. This is the number that has to stay well above that.
"""

_IMPLIED_OPENING = (
    "One entry was added to the register.",
    "The sheet below holds a single record.",
    "What follows was copied from the register unchanged.",
    "The clerk opened a new entry that morning.",
    "Next in the file.",
    "A single record, transcribed as found.",
    "The entry reads as follows.",
    "This page carries one entry and nothing else.",
    "Taken from the register.",
    "The record below was filed complete.",
    "One line in the register was filled in.",
    "The file was opened and the entry begun.",
    "Below is what the register holds.",
    "The page was started the same afternoon.",
    "An entry, copied out in full.",
    "The register was opened at the next free line.",
)
"""Openings that carry no class.

Sixteen and not twelve, because the labelled opening interpolates the class
name and so reaches 79 distinct first sentences over 120 documents and 238 over
a thousand, while this one has nothing to vary on and stops at 16. The count
matters less here than it does there. A labelled frame carries the tie between
a value and a property, and an implied frame carries no part of the answer at
all, so an arm that learns all sixteen has learnt nothing it can answer with.
"""

_IMPLIED_PLAIN = (
    "The entry gives {value}.",
    "One line reads {value}.",
    "The sheet has {value} on it.",
    "Written on it: {value}.",
    "The clerk entered {value}.",
    "Copied across: {value}.",
    "The next line is {value}.",
    "Below that stands {value}.",
    "There is {value} as well.",
    "It carries {value}.",
    "Also entered: {value}.",
    "The following line holds {value}.",
    "Further down, {value}.",
    "The form shows {value}.",
    "Noted on the page: {value}.",
    "A line further on reads {value}.",
    "The column beside it carries {value}.",
    "Set down next was {value}.",
)
"""Frames that say where a value sits and never what it means.

Every one of them is about the page. A frame that reached for the sense of the
value, "it goes by {value}", would point at ``name`` and hand back the label
the mode withholds.
"""

_IMPLIED_FLAG = (
    "One box on the form was marked {value}.",
    "A box further down reads {value}.",
    "The answer given there was {value}.",
    "Against the next line the clerk put {value}.",
    "The box below it says {value}.",
    "On the form, one box stands at {value}.",
)
"""Grammar for a bare ``true`` or ``false``. "The sheet has true on it" reads
as a generation artefact and invites a model to treat the document as noise."""

_IMPLIED_EMBEDDED = (
    "Some way down the page, after two lines that had been struck through, the entry gives {value}.",
    "The clerk paused, checked the previous page, and then wrote {value}.",
    "Between the heading and the countersignature the sheet carries {value}, with nothing beside it.",
    "What was copied in from the earlier draft, once the crossings-out are ignored, is {value}.",
    "Near the foot of the page, in the same hand as the rest, the sheet reads {value}.",
    "The line was begun, abandoned, and begun again, and what it finally reads is {value}.",
    "Once the margin notes are set aside, the entry on that line is {value}.",
    "Further down, past the fold in the paper, the sheet has {value}.",
    "After the countersignature and before the date stamp the clerk added {value}.",
    "The entry was continued on the next line, where it reads {value}.",
)
"""Where difficulty goes when there is no label to move away from.

The labelled corpus makes a document harder by putting the value and its
property in different sentences. Nothing here has a property to separate, so
distance is bought by surrounding the value with text that carries no answer,
the way the quantity corpus does it for a reading with no kind named.
"""

IMPLIED_FRAME_COUNT = (
    len(_IMPLIED_OPENING) + len(_IMPLIED_PLAIN) + len(_IMPLIED_FLAG) + len(_IMPLIED_EMBEDDED) + len(_FILLER)
)
"""How many sentence frames the implied mode writes from, against 83 labelled."""

_POOL_BY_KIND = {
    Kind.TEXT: _TEXTUAL + _GENERIC,
    Kind.URL: _TEXTUAL + _GENERIC,
    Kind.EMAIL: _TEXTUAL + _GENERIC,
    Kind.NUMBER: _NUMERIC + _GENERIC,
    Kind.INTEGER: _NUMERIC + _GENERIC,
    Kind.DATE: _TEMPORAL + _GENERIC,
    Kind.DATETIME: _TEMPORAL + _GENERIC,
    Kind.TIME: _TEMPORAL + _GENERIC,
    Kind.DURATION: _TEMPORAL + _GENERIC,
    Kind.BOOLEAN: _FLAG + _GENERIC,
    Kind.ENUM: _CHOSEN + _GENERIC,
}
"""Each kind draws from its own frames and from the generic ones.

Most schema.org slots are text, so a kind-only pool would put six frames in
front of a model for the great majority of documents and leave the phrasing
easier to learn than the content.
"""


def _pick(pool: tuple[str, ...], rng: random.Random, used: set[str]) -> str:
    """One frame, avoiding a frame this document has already used.

    Bounded retries rather than sampling without replacement, because the
    pools differ per slot and a shared exclusion set would shrink the choice
    for later slots the most, which is the opposite of what is wanted.
    """
    frame = rng.choice(pool)
    for _ in range(3):
        if frame not in used:
            break
        frame = rng.choice(pool)
    used.add(frame)
    return frame


MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
"""The month names a written date is built from.

Authored here and not taken from a vocabulary, unlike the quantity corpus's
unit symbols. Nothing publishes a surface form for a date the way QUDT
publishes one for a unit: CLDR carries the names but the choice of pattern is
still ours, and a committed file claiming CLDR as its source would be
provenance for a decision this module made. What is separable is the judgement,
and that is :data:`DECLINED`.
"""

BOOLEAN_WORDS = {True: "yes", False: "no"}
"""Two words for two values, so the mapping back is a lookup and not a guess.

No enumeration in the corpus admits either word, measured over the 530 distinct
members the describable classes carry, so a value of this form is still a
boolean and not a member of a closed list. That matters only for
:attr:`Labelling.IMPLIED`, where the form of a value is all a reader has.
"""

DECLINED: tuple[tuple[str, str], ...] = (
    ("05/11/2008", "day-first and month-first are both current, so one string is two dates"),
    ("2008-W45-3", "a week date needs a week-numbering rule no prompt states"),
    ("5 Nov 08", "a two-digit year is short by a century"),
    ("13 May 2000, 22:55", "the seconds the answer carries cannot be put back"),
    ("3:18", "hours with minutes and minutes with seconds are the same string"),
    ("76.037,1179", "a decimal comma is a different number under a locale no prompt states"),
    ("76,037.12", "rounding cannot be undone"),
)
"""Forms a document really writes, and the collision that rules each one out.

The counterpart of :data:`~oold_llm_bench.corpus.quantities.UNWRITABLE`, which
lists the published unit symbols no document writes. Nothing rejects these at
run time, because nothing produces them; they are here so that what was
declined stays readable beside what was kept.
"""

_ISO_DURATION = re.compile(
    r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?"
)

_DURATION_LABELS = ("y", "mo", "wk", "d", "h", "min", "s")
"""One label per ISO 8601 duration component, in the order the standard writes
them. ``mo`` and ``min`` are the reason the labels exist: the standard spells
both ``M`` and tells them apart by position alone."""

_DURATION_LETTERS = ("Y", "M", "W", "D", "H", "M", "S")


def written_date(value: str) -> str:
    """``2008-11-05`` as ``5 November 2008``."""
    year, month, day = value.split("-")
    return f"{int(day)} {MONTHS[int(month) - 1]} {year}"


def canonical_date(text: str) -> str:
    """The date behind a written one. The inverse of the above."""
    day, month, year = text.split(" ")
    return f"{year}-{MONTHS.index(month) + 1:02d}-{int(day):02d}"


def written_time(value: str) -> str:
    """``22:55:18`` as ``10:55:18 pm``.

    Midnight is ``12:00:00 am`` and noon is ``12:00:00 pm``, which is the one
    place a twelve-hour clock stops being obvious, so it is stated. With that
    fixed the two clocks are in bijection and the seconds are kept, because
    dropping them would leave a third of the answer off the page.
    """
    hour, minute, second = value.split(":")
    count = int(hour)
    return f"{count % 12 or 12}:{minute}:{second} {'am' if count < 12 else 'pm'}"


def canonical_time(text: str) -> str:
    """The time behind a written one. The inverse of the above."""
    clock, meridiem = text.rsplit(" ", 1)
    hour, minute, second = clock.split(":")
    count = int(hour) % 12 + (12 if meridiem == "pm" else 0)
    return f"{count:02d}:{minute}:{second}"


def written_datetime(value: str) -> str:
    """``2000-05-13T22:55:18Z`` as ``13 May 2000 at 10:55:18 pm UTC``.

    The zone is spelled out instead of left to a convention. Every datetime
    this corpus draws is UTC, so a rule could restore the ``Z``, but a rule the
    document does not state is one more thing a reader has to have been told.
    """
    date, clock = value.removesuffix("Z").split("T")
    return f"{written_date(date)} at {written_time(clock)} UTC"


def canonical_datetime(text: str) -> str:
    """The datetime behind a written one. The inverse of the above."""
    date, clock = text.removesuffix(" UTC").split(" at ")
    return f"{canonical_date(date)}T{canonical_time(clock)}Z"


def written_duration(value: str) -> str:
    """``PT3H18M`` as ``3 h 18 min``.

    Only the components the duration states are written, so ``PT45M`` is
    ``45 min`` and not ``0 h 45 min``, and the inverse writes back the
    components it finds. ``3:18`` would be shorter and is declined: it is both
    three hours and eighteen minutes and three minutes and eighteen seconds,
    and it is also how this module writes a time.
    """
    found = _ISO_DURATION.fullmatch(value)
    if found is None or not any(found.groups()):
        raise ValueError(f"no document could write the duration {value}")
    return " ".join(
        f"{count} {label}" for count, label in zip(found.groups(), _DURATION_LABELS, strict=True) if count is not None
    )


def canonical_duration(text: str) -> str:
    """The duration behind a written one. The inverse of the above."""
    parts = text.split(" ")
    counts = dict(zip(parts[1::2], parts[::2], strict=True))
    spelled = [
        (f"{counts[label]}{letter}" if label in counts else "")
        for label, letter in zip(_DURATION_LABELS, _DURATION_LETTERS, strict=True)
    ]
    clock = "".join(spelled[4:])
    return "P" + "".join(spelled[:4]) + (f"T{clock}" if clock else "")


_WRITTEN_BY_KIND: dict[Kind, Callable[[Any], str]] = {
    Kind.BOOLEAN: BOOLEAN_WORDS.__getitem__,
    Kind.NUMBER: written_number,
    Kind.INTEGER: written_number,
    Kind.DATE: written_date,
    Kind.DATETIME: written_datetime,
    Kind.TIME: written_time,
    Kind.DURATION: written_duration,
}
"""How each kind reaches the page when a document writes it out.

A kind that is absent is one a document already writes canonically: text, a
URL and an email address are spelled the same either way, and an enumeration
member has no second spelling to take.
"""

_CANONICAL_BY_KIND: dict[Kind, Callable[[str], Scalar]] = {
    Kind.BOOLEAN: {word: value for value, word in BOOLEAN_WORDS.items()}.__getitem__,
    Kind.NUMBER: canonical_number,
    Kind.INTEGER: lambda text: int(canonical_number(text)),
    Kind.DATE: canonical_date,
    Kind.DATETIME: canonical_datetime,
    Kind.TIME: canonical_time,
    Kind.DURATION: canonical_duration,
}


def written_value(kind: Kind, value: Scalar) -> str:
    """One value as a document writes it rather than as the schema stores it."""
    writer = _WRITTEN_BY_KIND.get(kind)
    return writer(value) if writer is not None else spell(value)


def canonical_value(kind: Kind, text: str) -> Scalar:
    """The value a written form means. The inverse of :func:`written_value`.

    The test the whole mode rests on, because an expected value no rule reaches
    from the page is a task with no correct answer.
    """
    reader = _CANONICAL_BY_KIND.get(kind)
    return reader(text) if reader is not None else text


def spell(value: Scalar, kind: Kind | None = None, notation: Notation = Notation.CANONICAL) -> str:
    """A value as the document writes it.

    Booleans are written ``true`` and ``false`` instead of ``True`` and
    ``False``, so an arm answering in JSON writes back what it read.

    Under :attr:`~oold_llm_bench.corpus.quantities.Notation.WRITTEN` the slot's
    kind decides the form, and a caller that has not supplied one is refused
    rather than given the canonical spelling. A mode that fell back quietly
    would report the fallback.
    """
    if notation is Notation.WRITTEN:
        if kind is None:
            raise ValueError("a written value needs the kind of the slot it fills")
        return written_value(kind, value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def class_phrase(cls: SchemaClass, variant: Variant) -> str:
    """How the document names the class.

    ``PostalAddress`` reads "postal address". Lowercasing it whole gives
    "postaladdress", a token no document contains and one a model could only
    match by guessing the corpus convention.
    """
    return readable_kind(cls.name) if variant is Variant.NATIVE else class_identifier(cls.name, variant)


def property_phrase(slot: Slot | Link, variant: Variant) -> str:
    """How the document names a property, whether it holds a value or a node."""
    return readable_kind(slot.name) if variant is Variant.NATIVE else property_identifier(slot.name, variant)


def render(
    entity: Entity,
    difficulty: Difficulty,
    rng: random.Random,
    variant: Variant = Variant.NATIVE,
    labelling: Labelling = Labelling.NAMED,
    notation: Notation = Notation.CANONICAL,
) -> str:
    """Write one entity down as the prose a model will read.

    The frame is chosen from the slot's kind and the difficulty, never from
    the variant, so the native and the renamed document are the same sentences
    with the vocabulary swapped. Anything else would confound the two things
    the pair exists to separate.

    ``notation`` is chosen the same way and for the same reason: it moves how a
    value is spelled and leaves which frame holds it alone, so a pair of
    documents at one seed differs in the values and in nothing else.

    An implied document ignores the variant altogether, because it spells no
    name of either vocabulary. The pair then differs in the catalogue and in
    the answer and in nothing else, which is a cleaner reading of what
    pretraining exposure is worth than a pair whose documents differ too.

    An entity's links are written after its values, in frames that say they
    are cross-references. The target is named and not described, so this stays
    a one-entity renderer and a linked document is its paragraphs in order.
    """
    if labelling is Labelling.IMPLIED:
        return _render_implied(entity, difficulty, rng, notation)
    used: set[str] = set()
    parts = [_pick(_OPENING, rng, used).format(cls=class_phrase(entity.cls, variant))]
    for index, (slot, value) in enumerate(entity.values):
        separated = difficulty is not Difficulty.EASY and index % 2 == 0
        pool = _SEPARATED if separated else _POOL_BY_KIND.get(slot.kind, _GENERIC)
        written = spell(value, slot.kind, notation)
        parts.append(_pick(pool, rng, used).format(prop=property_phrase(slot, variant), value=written))
        if difficulty is Difficulty.HARD and index % 2 == 1:
            parts.append(_pick(_FILLER, rng, used))
    for edge in entity.edges:
        parts.append(_pick(_LINKED, rng, used).format(prop=property_phrase(edge.link, variant), value=edge.designator))
    return " ".join(parts)


def _render_implied(
    entity: Entity,
    difficulty: Difficulty,
    rng: random.Random,
    notation: Notation = Notation.CANONICAL,
) -> str:
    """The same ladder of difficulty, with nothing on the page named.

    The structure follows the labelled renderer sentence for sentence, so a
    document of one difficulty stays longer than the one below it and the two
    modes can be read against each other.
    """
    used: set[str] = set()
    parts = [_pick(_IMPLIED_OPENING, rng, used)]
    for index, (slot, value) in enumerate(entity.values):
        embedded = difficulty is not Difficulty.EASY and index % 2 == 0
        plain = _IMPLIED_FLAG if slot.kind is Kind.BOOLEAN else _IMPLIED_PLAIN
        pool = _IMPLIED_EMBEDDED if embedded else plain
        parts.append(_pick(pool, rng, used).format(value=spell(value, slot.kind, notation)))
        if difficulty is Difficulty.HARD and index % 2 == 1:
            parts.append(_pick(_FILLER, rng, used))
    return " ".join(parts)


def render_entry(
    entity: Entity,
    variant: Variant = Variant.NATIVE,
    labelling: Labelling = Labelling.NAMED,
    notation: Notation = Notation.CANONICAL,
) -> str:
    """One entity as an entry in a list of several.

    Several records are a list, not a run of paragraphs. The entry names its
    own class, because with more than one subject on the page a reading that
    is not attached to one is unattributable, the failure the quantity
    corpus sidesteps by having no subjects at all.

    An implied entry names nothing and leads with its first value, because a
    bullet reading "an entry" once per record is chrome a generator would write
    and a clerk would not. The grouping is still given, so what it asks is
    which class each bag of values belongs to and not which bag a value is in.
    """
    written = [spell(value, slot.kind, notation) for slot, value in entity.values]
    if labelling is Labelling.IMPLIED:
        first, *rest = written
        return "\n".join([f"- {first}", *(f"  {value}" for value in rest)])
    lines = [f"- {class_phrase(entity.cls, variant)}"]
    lines += [
        f"  {property_phrase(slot, variant)}: {value}" for (slot, _), value in zip(entity.values, written, strict=True)
    ]
    return "\n".join(lines)


def render_class(
    cls: SchemaClass,
    variant: Variant,
    max_slots: int = CATALOGUE_SLOTS,
    with_links: bool = False,
) -> str:
    """One catalogue entry, as the prompt can show it.

    The renamed entry carries the slot structure and nothing else. Keeping the
    label and the description would hand back in the catalogue exactly the
    vocabulary the rename removed, and the arm would be answering from the
    English gloss rather than from the structure.

    ``with_links`` adds the node-valued properties and the classes they point
    at. Off by default, because every result measured so far was taken against
    an entry that lists the literal slots alone, and a catalogue that moves
    underneath a run makes the runs incomparable.
    """
    head = class_identifier(cls.name, variant)
    lines = [f"- {head}"]
    if variant is Variant.NATIVE:
        if cls.description:
            lines.append(f"  {cls.description}")
        if cls.parent:
            lines.append(f"  parent: {cls.parent}")
    shown = cls.own_slots[:max_slots] or cls.slots[:max_slots]
    for slot in shown:
        text = f"  {property_identifier(slot.name, variant)} ({slot.kind.value})"
        if slot.choices:
            text += ": " + ", ".join(slot.choices[:8])
        lines.append(text)
    if with_links:
        for link in cls.own_links:
            targets = ", ".join(class_identifier(name, variant) for name in link.ranges)
            lines.append(f"  {property_identifier(link.name, variant)} (link: {targets})")
    return "\n".join(lines)


def expected_for(entity: Entity, variant: Variant) -> ExpectedInstance:
    """What one entity's record should yield, values and edges together.

    A link becomes a :class:`~oold_llm_bench.grading.triples.Reference` to the
    key of the entity it reaches, so it is graded by where it points and not
    by what either side called the target.
    """
    fields: dict[str, Any] = {property_identifier(slot.name, variant): value for slot, value in entity.values}
    for edge in entity.edges:
        fields[property_identifier(edge.link.name, variant)] = Reference(key=edge.target)
    return ExpectedInstance(
        key=entity.key,
        class_path=class_identifier(entity.cls.name, variant),
        fields=fields,
    )


def generate_task(
    classes: list[SchemaClass],
    *,
    task_id: str,
    seed: int,
    variant: Variant = Variant.NATIVE,
    difficulty: Difficulty = Difficulty.EASY,
    n_entities: int = 1,
    n_slots: int = 4,
    split: Split = Split.DEV,
    catalogue: tuple[str, ...] | None = None,
    describe_catalogue: bool = False,
    draw_from: list[SchemaClass] | None = None,
    labelling: Labelling = Labelling.NAMED,
    notation: Notation = Notation.CANONICAL,
    linked: bool = False,
) -> TaskRecord:
    """One task: an entity written down as prose, and what it was.

    The seed is the whole of the randomness, so a task id and a seed reproduce
    the document byte for byte, and the same seed under the other variant
    reproduces the same facts under a different vocabulary.

    ``draw_from`` narrows which classes the answer may be, while ``classes``
    still supplies the catalogue and the hierarchy. The two differ whenever
    part of the corpus is held out of something: the offered list has to keep
    both halves, or a held-out class has no way to be chosen and the contrast
    cannot be measured.

    ``notation`` defaults to
    :attr:`~oold_llm_bench.corpus.quantities.Notation.CANONICAL`, because every
    measured result was taken under it and a corpus that moves underneath a run
    makes the runs incomparable.

    ``linked`` asks for a document whose entities point at one another, and
    defaults off for the same reason.
    """
    _refuse_a_silent_catalogue(labelling, catalogue, describe_catalogue, notation)
    entities = draw(draw_from or classes, random.Random(seed), n_entities, n_slots, labelling, classes, linked)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
    return _task_from(
        entities,
        classes,
        task_id=task_id,
        seed=seed,
        variant=variant,
        difficulty=difficulty,
        split=split,
        catalogue=catalogue,
        describe_catalogue=describe_catalogue,
        labelling=labelling,
        notation=notation,
    )


def _refuse_a_silent_catalogue(
    labelling: Labelling,
    catalogue: tuple[str, ...] | None,
    describe_catalogue: bool,
    notation: Notation = Notation.CANONICAL,
) -> None:
    """An implied or a written task needs a catalogue that states its kinds.

    :func:`slot_schema` writes every string-valued kind as ``string``, so a URL
    slot and a duration slot and a text slot are one shape in the branches and
    in the answer schema. The kind reaches the model through the catalogue
    entry and nowhere else, and without it a document that names nothing is
    unanswerable rather than hard.

    A written document has the same need for a different reason. It states
    ``5 November 2008`` where the answer is ``2008-11-05``, so the form the
    answer takes is no longer on the page and the slot's kind is what is left
    to infer it from.
    """
    if not catalogue or describe_catalogue:
        return
    if labelling is Labelling.IMPLIED:
        raise ValueError("an implied task needs describe_catalogue, or no arm ever sees the kind of a slot")
    if notation is Notation.WRITTEN:
        raise ValueError("a written task needs describe_catalogue, or no arm ever sees what form an answer takes")


def generate_pair(
    classes: list[SchemaClass],
    *,
    task_id: str,
    seed: int,
    difficulty: Difficulty = Difficulty.EASY,
    n_entities: int = 1,
    n_slots: int = 4,
    split: Split = Split.DEV,
    catalogue: tuple[str, ...] | None = None,
    describe_catalogue: bool = False,
    draw_from: list[SchemaClass] | None = None,
    labelling: Labelling = Labelling.NAMED,
    notation: Notation = Notation.CANONICAL,
    linked: bool = False,
) -> tuple[TaskRecord, TaskRecord]:
    """The same facts as a native task and as a renamed one.

    Emitted together rather than generated twice from the same seed, because a
    pair that drifted would look like a vocabulary effect and be one only by
    accident.

    ``draw_from`` narrows which classes the answer may be, as in
    :func:`generate_task`.
    """
    _refuse_a_silent_catalogue(labelling, catalogue, describe_catalogue, notation)
    entities = draw(draw_from or classes, random.Random(seed), n_entities, n_slots, labelling, classes, linked)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one

    def under(variant: Variant) -> TaskRecord:
        return _task_from(
            entities,
            classes,
            task_id=f"{task_id}-{variant.value}",
            seed=seed,
            variant=variant,
            difficulty=difficulty,
            split=split,
            catalogue=_catalogue_in(catalogue, classes, variant),
            describe_catalogue=describe_catalogue,
            labelling=labelling,
            notation=notation,
        )

    return under(Variant.NATIVE), under(Variant.RENAMED)


_JSON_TYPE: dict[str, str] = {
    "TEXT": "string",
    "URL": "string",
    "EMAIL": "string",
    "DATE": "string",
    "DATETIME": "string",
    "TIME": "string",
    "DURATION": "string",
    "ENUM": "string",
    "NUMBER": "number",
    "INTEGER": "integer",
    "BOOLEAN": "boolean",
}


def slot_schema(slot: Slot, variant: Variant) -> dict[str, Any]:
    """One property as JSON Schema, with its enumeration when it has one.

    A schema.org enumeration is a closed list the same way a unit list is, so
    it is the part a constrained arm can actually enforce. Members keep their
    UI spelling, since that is what the document says and what the answer is
    graded against.
    """
    built: dict[str, Any] = {"type": _JSON_TYPE.get(slot.kind.name, "string")}
    if slot.kind is Kind.ENUM and slot.choices:
        built["enum"] = list(slot.choices)
    return built


def link_schema() -> dict[str, Any]:
    """One node-valued property as JSON Schema.

    An object and nothing more. The extractor reads a nested object as an
    edge to the entity inside it, which is the shape that has to be reachable;
    restating the target's own properties here would repeat the branch the
    catalogue already carries for that class.
    """
    return {"type": "object"}


def branches_for(
    classes: list[SchemaClass],
    variant: Variant,
    max_slots: int = CATALOGUE_SLOTS,
    with_links: bool = False,
) -> dict[str, dict[str, Any]]:
    """What each class narrows, keyed by the identifier the catalogue uses.

    schema.org differs from a quantity corpus here. There a subclass
    narrows one enumeration and every class carries the same five properties.
    Here the class decides which properties exist at all, so a branch is the
    whole difference between one class and another.

    ``with_links`` is off for the same reason it is off in
    :func:`render_class`.
    """
    return {
        class_identifier(cls.name, variant): {
            property_identifier(slot.name, variant): slot_schema(slot, variant) for slot in cls.slots[:max_slots]
        }
        | ({property_identifier(link.name, variant): link_schema() for link in cls.own_links} if with_links else {})
        for cls in classes
    }


def answer_schema(
    classes: list[SchemaClass],
    variant: Variant,
    max_slots: int = CATALOGUE_SLOTS,
    with_links: bool = False,
) -> dict[str, Any]:
    """The shape an answer takes for this corpus.

    Every property any offered class defines, so an unconstrained arm has
    somewhere to put its answer. That is exactly the flat union the
    orchestration exists to avoid: it admits `alumniOf` on an `Organization`,
    and nothing in it says which class allows which property.
    """
    properties: dict[str, Any] = {
        "type": {
            "type": "string",
            "description": "The class this entity is an instance of.",
        }
    }
    for cls in classes:
        for slot in cls.slots[:max_slots]:
            properties.setdefault(property_identifier(slot.name, variant), slot_schema(slot, variant))
        if with_links:
            for link in cls.own_links:
                properties.setdefault(property_identifier(link.name, variant), link_schema())
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Entities",
        "description": "Every entity the document describes.",
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {"type": "object", "properties": properties, "required": ["type"]},
            }
        },
        "required": ["entities"],
    }


def _implied_schema(
    offered: list[SchemaClass],
    variant: Variant,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]] | None, None]:
    """The answer surface of an implied task: what each class declares, no more.

    ``Thing`` gives every class ``name``, ``alternateName`` and ``url``, and a
    slot that takes any text or any URL makes every text and URL value
    ambiguous on every class at once. A mode whose only signal is the form of
    the value cannot also offer them, so an implied class is exactly the
    properties it declares, which is the set :func:`render_class` already puts
    in the catalogue entry.

    No lineage follows, because a branch that states everything the class
    admits has nothing left for an ancestor to contribute.
    """
    if not offered:
        return None, None, None
    declared = [replace(cls, slots=cls.own_slots) for cls in offered]
    return answer_schema(declared, variant), branches_for(declared, variant), None


def _ancestors_of(name: str, classes: list[SchemaClass]) -> set[str]:
    """Every class this one inherits from, however many parents it has."""
    by_name = {cls.name: cls for cls in classes}
    found: set[str] = set()
    queue = list(by_name[name].parents) if name in by_name else []
    while queue:
        current = queue.pop()
        if current in found or current not in by_name:
            continue
        found.add(current)
        queue.extend(by_name[current].parents)
    return found


def _catalogue_in(
    catalogue: tuple[str, ...] | None,
    classes: list[SchemaClass],
    variant: Variant,
) -> tuple[str, ...] | None:
    """The same classes, named as this variant names them.

    A caller builds one catalogue and gets both variants back, so the renamed
    task would otherwise be offered native names while its answer is an opaque
    one, and the answer would not be in its own catalogue. Order is kept, so
    the pair differs in the names and in nothing else.
    """
    if not catalogue:
        return catalogue
    by_native = {cls.name: cls for cls in classes}
    by_any = {class_identifier(cls.name, v): cls for cls in classes for v in Variant}
    renamed = []
    for identifier in catalogue:
        found = by_native.get(identifier) or by_any.get(identifier)
        renamed.append(class_identifier(found.name, variant) if found else identifier)
    return tuple(renamed)


def _task_from(
    entities: list[Entity],
    classes: list[SchemaClass],
    *,
    task_id: str,
    seed: int,
    variant: Variant,
    difficulty: Difficulty,
    split: Split,
    catalogue: tuple[str, ...] | None,
    describe_catalogue: bool,
    labelling: Labelling = Labelling.NAMED,
    notation: Notation = Notation.CANONICAL,
) -> TaskRecord:
    """Write one draw down under one vocabulary.

    The rendering generator is seeded afresh here rather than carried over
    from the draw, so both variants pick the same frames and the pair differs
    only in the names.
    """
    rng = random.Random(seed)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
    linked = any(entity.edges for entity in entities)
    if linked:
        # One paragraph per record. A link sentence says "its" of the subject
        # the paragraph opened with, and a list entry has no subject to say it
        # of, so the prose form is the one that can carry an edge.
        document = "\n\n".join(render(entity, difficulty, rng, variant, labelling, notation) for entity in entities)
    elif len(entities) == 1:
        document = render(entities[0], difficulty, rng, variant, labelling, notation)
    else:
        document = "Records:\n" + "\n".join(render_entry(entity, variant, labelling, notation) for entity in entities)

    chosen = {class_identifier(cls.name, variant) for cls in classes} & set(catalogue or ())
    by_identifier = {class_identifier(cls.name, variant): cls for cls in classes}
    offered = [by_identifier[name] for name in chosen]
    if labelling is Labelling.IMPLIED:
        shape, narrowed, lineage = _implied_schema(offered, variant)
    else:
        shape = answer_schema(offered, variant, with_links=linked) if offered else None

        # Ancestors go in too. A union states a property where it is declared,
        # so a class that is offered needs every class it inherits from
        # present, even when that ancestor is not itself an answer.
        needed: set[str] = set()
        for cls in offered:
            needed.add(cls.name)
            needed |= _ancestors_of(cls.name, classes)
        family = [cls for cls in classes if cls.name in needed]
        narrowed = branches_for(family, variant, with_links=linked) if family else None
        lineage = (
            {
                class_identifier(cls.name, variant): [class_identifier(p, variant) for p in cls.parents if p in needed]
                for cls in family
            }
            if family
            else None
        )

    described: dict[str, str] | None = None
    if catalogue and describe_catalogue:
        by_identifier = {class_identifier(cls.name, variant): cls for cls in classes}
        described = {
            identifier: render_class(by_identifier[identifier], variant, with_links=linked)
            for identifier in catalogue
            if identifier in by_identifier
        }

    return TaskRecord(
        id=task_id,
        document=document,
        expected=[expected_for(entity, variant) for entity in entities],
        corpus=CorpusRef(
            name="schemaorg",
            source=Source.SYNTHETIC,
            document_id=task_id,
            content_hash=hashlib.sha256(document.encode("utf-8")).hexdigest(),
        ),
        split=split,
        variant=variant,
        difficulty=difficulty,
        notes=f"variant={variant.value},classes="
        + "|".join(entity.cls.name for entity in entities)
        + (f",notation={notation.value}" if notation is not Notation.CANONICAL else ""),
        catalogue=list(catalogue) if catalogue else None,
        catalogue_text=described,
        answer_schema=shape,
        branches=narrowed,
        class_parents=lineage,
        property_ranges=_ranges_of(offered) or None,
    )


def _ranges_of(offered: list[SchemaClass]) -> dict[str, list[str]]:
    """Which classes each link property may point at, over the offered set.

    Ranges outside the offered classes are kept. A plan cannot contain an
    entity of a class nobody offered, so such a range simply pins nothing, and
    dropping it here would make an absent target and an unoffered one the same
    silence.
    """
    ranges: dict[str, set[str]] = {}
    for cls in offered:
        for link in cls.own_links:
            if link.ranges:
                ranges.setdefault(link.name, set()).update(link.ranges)
    return {name: sorted(values) for name, values in sorted(ranges.items())}


def top_level_sets(classes: list[SchemaClass], pool: list[SchemaClass] | None = None) -> dict[str, tuple[str, ...]]:
    """The describable classes grouped under the branch of Thing they descend from.

    A catalogue is a choice about what the model is asked to choose between,
    and a count is the wrong way to make it. Twenty-five of 122 leaves the
    answer in or out by accident: a pasted "Andrea works at Siemens" was once
    offered 25 classes holding neither Person nor Organization, so the only
    fitting class was Thing, which declares no links, and the edge had nowhere
    to go.

    The grouping is the ontology's own. Every class reaches Thing, and the
    child of Thing it passes through names the set. Measured over the 122
    describable classes: Intangible 60, CreativeWork 31, MedicalEntity 8,
    Organization 5, Product 4, Place 3, and nine smaller branches.

    Multiple parents take the first, as ``class_parents`` records them. A class
    reachable through two branches is rare here and naming it twice would make
    the sets overlap, which is not what a chooser wants.
    """
    by_name = {cls.name: cls for cls in classes}

    def branch(name: str, seen: frozenset[str] = frozenset()) -> str:
        cls = by_name.get(name)
        if cls is None or not cls.parents or name in seen:
            return name
        if "Thing" in cls.parents:
            return name
        return branch(cls.parents[0], seen | {name})

    grouped: dict[str, list[str]] = {}
    for cls in pool if pool is not None else describable_classes(classes):
        grouped.setdefault(branch(cls.name), []).append(cls.name)
    return {name: tuple(sorted(members)) for name, members in sorted(grouped.items())}
