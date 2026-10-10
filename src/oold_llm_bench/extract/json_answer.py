"""Turning an arm's JSON answer into triples.

One extractor, applied identically to every arm. That is the whole point. An
arm given a schema and an arm given nothing must be read by the same rules,
or a measured difference between them is partly a difference between
extractors.

So this is deliberately lenient about shape and strict about nothing. A
grounded arm answers with ``{"type": ["Length"], "value": 1.75, "unit":
"meter"}`` and an unconstrained one might answer with ``{"length": {"amount":
1.75, "units": "m"}}``. Both reduce to the same triple. What it will not do is
guess. A fragment it cannot read becomes a parse error and is counted, so the
cost of leniency is visible in the result instead of hidden in it.
"""

from __future__ import annotations

from typing import Any

from oold_llm_bench.grading.triples import (
    Quantity,
    Reference,
    Triple,
    TripleSet,
    make_triple,
    normalise_property,
)

__all__ = [
    "CLASS_KEYS",
    "UNIT_KEYS",
    "VALUE_KEYS",
    "extract_json",
]

CLASS_KEYS = ("type", "@type", "class", "class_path", "kind", "quantity_kind", "category")
"""Keys an answer might use to say what something is. Order is preference."""

VALUE_KEYS = ("value", "magnitude", "amount", "measurement", "reading", "number")
UNIT_KEYS = ("unit", "units", "unit_code", "unitcode", "uom", "unit_of_measure")

PROVENANCE_KEYS = ("provenance", "source", "span", "evidence", "quote")

ID_KEYS = ("id", "@id", "key", "entity_id")
"""Keys an answer uses to name an entity it will point at.

A segmented orchestration plans the document, pins each entity to a planned
id and writes an edge as that id. Without reading the id the link is an
ordinary string compared literally, which is the case ``Reference``'s own
docstring calls unscoreable."""

_ENTITY_LIST_KEYS = ("entities", "instances", "items", "results", "measurements", "values")
"""Wrappers a model puts a list of answers behind."""


def _first(payload: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, Any] | None:
    lowered = {normalise_property(k): k for k in payload}
    for key in keys:
        actual = lowered.get(normalise_property(key))
        if actual is not None and payload[actual] not in (None, "", [], {}):
            return actual, payload[actual]
    return None


def _as_class(value: Any) -> str | None:
    """A class name from whatever shape the answer put it in."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple)) and value:
        return _as_class(value[0])
    return None


def _as_quantity(value: Any, unit: Any) -> Quantity | None:
    if not isinstance(unit, str) or not unit.strip():
        return None
    try:
        magnitude = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    return Quantity(magnitude=magnitude, unit=unit.strip())


def _scalarise(value: Any) -> Any:
    """Collapse a nested value to something a triple can hold.

    A dict carrying a value and a unit is a quantity. Anything else nested is
    left alone here and handled by the caller, which knows whether it is
    looking at a field or at another entity.
    """
    if isinstance(value, dict):
        found_value = _first(value, VALUE_KEYS)
        found_unit = _first(value, UNIT_KEYS)
        if found_value and found_unit:
            return _as_quantity(found_value[1], found_unit[1])
    return value


class _Reader:
    def __init__(self) -> None:
        self.triples: set[Triple] = set()
        self.classes: dict[str, str] = {}
        self.provenance: dict[str, str] = {}
        self.parse_errors = 0
        self._next = 0
        self._taken: set[str] = set()
        """Every key handed out, stated or invented. See :meth:`key`."""
        self.nested: set[str] = set()
        """Keys of entities that arrived inside another entity's field.

        The same thing a top-level entity is, written differently: the answer
        put the object where the value goes rather than naming it and
        pointing at the name. Embedding is a serialisation choice the schema
        asks for and not a claim about identity, so an embedded entity may
        still carry an id, point at others and be pointed at.

        Recorded because nothing in the triples says which shape the answer
        used once they are flattened, and which shape it used is a thing the
        arm got right or wrong."""
        self.stated: set[str] = set()
        """Keys an entity named for itself.

        Only these attract a reference. A key this reader invented is a label
        no answer could have pointed at, so resolving against one would turn a
        value that happens to read ``e2`` into an edge nobody asserted."""

    def key(self, suggested: str | None = None) -> str:
        """The key this entity is filed under.

        A key the reader invents must not collide with one the answer stated,
        and the two are drawn from one namespace: the prompt asks for "a short
        identifier for this entity, e.g. e1", so a model naming its entities
        ``e1, e2`` is doing what it was told. An answer that ids some entities
        and not others is the ordinary shape of a segmented or multi-step
        fill, and before this it merged them: three entities in, two out, the
        first one's class overwritten by the second, and ``parse_errors`` left
        at zero. Every dimension then read wrong and plausibly.

        So an invented key skips anything already taken, and a stated key that
        collides with one already invented is given a suffix rather than
        silently folded into it. Taken keys accumulate across the whole answer
        because a later entity can state the name an earlier one was given.
        """
        if suggested:
            key = suggested
            while key in self._taken:
                key = f"{suggested}#{len(self._taken)}"
            self._taken.add(key)
            return key
        self._next += 1
        while f"e{self._next}" in self._taken:
            self._next += 1
        key = f"e{self._next}"
        self._taken.add(key)
        return key

    def entity(self, payload: Any, suggested: str | None = None, named: str | None = None) -> str | None:
        """Read one entity and return the key it was filed under.

        The key is returned so a parent can point at it. A nested object
        read as its own entity, with nothing recording where it came from,
        loses the edge, which is the part a graph is made of.

        ``named`` is a class the wrapper asserted about this entity, used only
        when the entity states none of its own.
        """
        if not isinstance(payload, dict):
            self.parse_errors += 1
            return None

        found_id = _first(payload, ID_KEYS)
        errors_before = self.parse_errors
        consumed: set[str] = set()
        if found_id is not None and isinstance(found_id[1], str) and found_id[1]:
            key = self.key(found_id[1])
            self.stated.add(key)
            consumed.add(found_id[0])
        else:
            key = self.key(suggested)

        consumed |= self._read_class(payload, key, named)

        found_provenance = _first(payload, PROVENANCE_KEYS)
        if found_provenance and isinstance(found_provenance[1], str):
            self.provenance[key] = found_provenance[1]
            consumed.add(found_provenance[0])

        # A bare quantity: the entity itself carries a value and a unit.
        found_value = _first(payload, VALUE_KEYS)
        found_unit = _first(payload, UNIT_KEYS)
        if found_value and found_unit:
            quantity = _as_quantity(found_value[1], found_unit[1])
            if quantity is None:
                self.parse_errors += 1
            else:
                self.triples.add(make_triple(key, "value", quantity))
            consumed.update({found_value[0], found_unit[0]})

        for field, value in payload.items():
            if field in consumed:
                continue
            self.field(key, field, value)

        # One failure, one error. Counting an unreadable field and then the
        # empty entity it left behind would log the same mistake twice, which
        # made the predecessor's error counts unusable.
        already_counted = self.parse_errors > errors_before
        yielded_nothing = not any(t.entity == key for t in self.triples) and key not in self.classes
        if yielded_nothing and not already_counted:
            self.parse_errors += 1
        return key

    def _read_class(self, payload: dict[str, Any], key: str, named: str | None) -> set[str]:
        """Record the class this entity states, and say which field held it.

        An answer keyed by class name states the class in the key and nowhere
        else. Reading the key as an id and dropping it discarded the one field
        the answer carried, and every arm without a schema answers in that
        shape, because nothing fixes the envelope. A class the entity states
        itself always wins over the key.
        """
        found = _first(payload, CLASS_KEYS)
        if found:
            name = _as_class(found[1])
            if name:
                self.classes[key] = name
                return {found[0]}
            return set()
        if named:
            self.classes[key] = named
        return set()

    def field(self, key: str, field: str, value: Any) -> None:
        if value is None or value == "":
            # A null field is an absence, not an extracted value. Scoring it
            # as one would let an answer of nothing earn triples.
            return
        scalar = _scalarise(value)
        if isinstance(scalar, dict):
            # A nested object that is not a quantity is a separate entity,
            # and the field that held it is an edge to it.
            child = self.entity(scalar)
            if child is not None:
                self.nested.add(child)
                self.triples.add(make_triple(key, field, Reference(key=child)))
            return
        if isinstance(scalar, list):
            for item in scalar:
                self.field(key, field, item)
            return
        if scalar is None:
            # A value and a unit that would not parse as a quantity.
            self.parse_errors += 1
            return
        self.triples.add(make_triple(key, field, scalar))

    def read(self, payload: Any) -> None:
        if isinstance(payload, list):
            for item in payload:
                self.entity(item)
            return
        if not isinstance(payload, dict):
            self.parse_errors += 1
            return

        wrapper = _first(payload, _ENTITY_LIST_KEYS)
        if wrapper and isinstance(wrapper[1], list):
            for item in wrapper[1]:
                self.entity(item)
            return

        # A mapping of names to entities, the shape a keyed answer arrives in.
        # The key is both the id and the asserted class, so it is offered as
        # both and the entity's own class wins where it states one.
        if payload and all(isinstance(v, dict) for v in payload.values()):
            for name, item in payload.items():
                self.entity(item, suggested=name, named=name)
            return

        self.entity(payload)


def extract_json(payload: Any) -> TripleSet:
    """Read one arm's JSON answer.

    ``parse_errors`` counts the fragments that yielded nothing, so a lenient
    read never inflates a score without saying so.
    """
    reader = _Reader()
    reader.read(payload)
    return TripleSet(
        triples=frozenset(_resolve_stated_ids(reader.triples, reader.stated)),
        classes=reader.classes,
        provenance=reader.provenance,
        nested=frozenset(reader.nested),
        parse_errors=reader.parse_errors,
    )


def _resolve_stated_ids(triples: set[Triple], stated: set[str]) -> set[Triple]:
    """Read a link written as a planned id, not as a nested object.

    A segmented answer names its entities and writes an edge as one of those
    names. Only ids an entity claimed for itself resolve: a key this reader
    invented was never visible to the answer, so a value that happens to read
    ``e2`` is a value and not an edge.

    An entity pointing at itself is left alone. That is a defect in the answer
    and reading it as a self-loop would score it as a link.
    """
    if not stated:
        return triples
    return {
        triple.model_copy(update={"value": Reference(key=triple.value)})
        if isinstance(triple.value, str) and triple.value in stated and triple.value != triple.entity
        else triple
        for triple in triples
    }
