"""Generate an OO-LD schema module from Wikidata, with nothing but Wikidata in it.

No model is asked anything here. A class becomes a schema by a rule, a
property becomes a slot by a rule, and a rerun against the same cache produces
the same bytes, so the module is an artefact of the Wikidata API and the
arguments and of nothing else. That is the whole claim: a generated vocabulary
a benchmark measures against has to be rebuildable by someone who does not
trust the person who built it.

*Which classes.* A set of roots walked down ``P279`` to a declared depth, or
an explicit list. The walk is breadth first and each level is one query over
the frontier, so a class reached twice is read once.

*Which properties, and by which rule.* ``P1963`` ("properties for this type")
is the class's own statement about what describes it, and where a class
carries it, it decides. Most classes do not: over a 60-class walk from
``film``, ``human``, ``organization`` and ``city``, 13 carry ``P1963`` and 47
do not, and for those the sample decides, keeping a property at least
:data:`DEFAULT_USAGE` of the sampled instances state. Both rules are recorded
per class in ``_meta.json`` rather than left to be inferred from the output,
because "the class declares this" and "instances happen to do this" are not
the same claim and a reader comparing two classes has to know which they are
looking at. A class declaring more properties than the cap keeps the ones its
instances use most: ``film`` declares 117, and a schema with 117 slots is not
a question anyone asks a model.

*Which parents.* All of them. ``film`` is a subclass of three things at once
and the schema.org module keeps the first, which is a recorded defect of that
module rather than a property of Wikidata. ``allOf`` lists every ``P279``
parent the module also emits, in QID order, and ``@context`` reflects them in
that same order because the validator requires it. A parent outside the walk
cannot be reached by a ``$ref`` at all; those are counted and listed in
``_meta.json`` rather than dropped in silence.

*Which JSON Schema type.* From the property's Wikidata datatype and nothing
else, through :data:`JSON_TYPE`. ``quantity`` is a number, ``time`` is a
string with ``format: date``, ``url`` and the entity-valued datatypes are
strings with an IRI format, and everything else Wikidata stores as text is a
string. An item-valued property is a ``$ref`` when its value-type constraint
(``P2302`` / ``Q21510865``) names a class the module emits, and a bare IRI
reference when it does not.

Two things in Wikidata's own data do not survive this map, and both are
counted rather than papered over. There is no boolean datatype at all, so no
property here reaches ``{"type": "boolean"}``; a yes/no fact is an item-valued
statement whose constraint lists two items, which is a constraint and not a
type. And ``globe-coordinate`` has no scalar form, so a coordinate property is
dropped from the class that would carry it.

*What the validator decides, rather than taste.* Every emitted file is put in
front of ``oold.validation`` offline and strict, and three of the shapes above
are what they are because of what came back. A ``$ref`` that would close a
cycle in the scoped-context graph is written as a bare IRI reference instead,
because no JSON-LD processor resolves a cyclic scoped context and a class
embedding its own type cannot be framed apart from its value;
:func:`_acyclic` is that rule and ``_meta.json`` names every embed it demoted.
``x-oold-instance-rdf-type`` is emitted so framing has a type to select on.
And an embedded value carries the ``type`` its schema defaults to, because a
node with nothing but an ``id`` returns from RDF as a bare string.

*What an instance is for.* Every emitted class whose sample found a real item
also gets one ``<Class>.instance.json`` built from that item's statements,
because a schema that validates and admits nothing real is half a check. A
statement whose date is coarser than a day is skipped rather than written as a
date the format does not admit, and a class with no instances at all gets no
file rather than an invented one.

*Determinism.* Every list is sorted before it is written, every file is dumped
with sorted keys, and the only clock that reaches the output is the date the
cache was fetched on. Two runs over one cache are byte-identical, which is
what makes :func:`~oold_llm_bench.corpus.provenance.corpus_digest` a pin and
not a timestamp.

    uv run python scripts/generate_wikidata_oold.py --roots Q11424 --depth 1
    uv run python scripts/generate_wikidata_oold.py --roots Q11424 --depth 1 --apply

The rate limit, the backoff and the resumable cache are
``harvest_wikidata_schemaorg.py``'s, unchanged: the query service refuses a
busy client for a minute at a time, and a harvest that cannot be resumed is a
harvest nobody finishes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oold_llm_bench.corpus.provenance import corpus_digest

SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "oold-llm-bench/1.0 (+https://github.com/OO-LD/oold-llm-bench)"

META_SCHEMA = "https://oo-ld.org/latest/meta/oold-meta-schema.json"
SCHEMA_VERSION = "0.1.0"
SUFFIX = ".schema.json"
INSTANCE_SUFFIX = ".instance.json"

ENTITY = "http://www.wikidata.org/entity/"
DIRECT = "http://www.wikidata.org/prop/direct/"

SUBCLASS_OF = "P279"
PROPERTIES_FOR_TYPE = "P1963"
PROPERTY_CONSTRAINT = "P2302"
VALUE_TYPE_CONSTRAINT = "Q21510865"
CONSTRAINT_CLASS = "P2308"
"""The five identifiers the whole generator rests on.

``P1963`` and ``P279`` are what a class says about itself. The constraint
triple is what a property says about its range, and it is a community
convention expressed in data rather than a schema Wikidata enforces, which is
why a property carrying none falls back to a bare IRI instead of being
dropped.
"""

SUBCLASS_QUERY = "SELECT ?child ?parent WHERE { VALUES ?parent { %s } ?child wdt:P279 ?parent . }"
"""One level of the walk, for a whole frontier at once.

``VALUES`` rather than a property path: ``wdt:P279*`` asks the service to
enumerate a closure that runs to 176,085 items under ``Q215627`` and times
out, while one level of a frontier answers in a second and is the same walk
done by the client, where the depth is visible and can be stopped.
"""

INSTANCE_QUERY = "SELECT ?item WHERE { ?item wdt:P31 wd:%(qid)s } LIMIT %(sample)d"
"""The sample a class's observed properties and its example instance come from.

``wdt:P31`` alone and not ``wdt:P31/wdt:P279*``. A class's own instances are
what the class is about; everything under it has its own schema in this module
and its own sample, and pooling them would describe ``film`` with the
properties of ``animated film``.

No ``ORDER BY``. The service has to enumerate a class before it can order it,
which is a timeout on every class with more than a few thousand instances, and
the sample is reproducible from the QIDs the cache records rather than from
the order they arrived in.
"""

JSON_TYPE: dict[str, dict[str, Any]] = {
    "commonsMedia": {"type": "string"},
    "entity-schema": {"type": "string", "format": "iri-reference"},
    "external-id": {"type": "string"},
    "geo-shape": {"type": "string", "format": "iri-reference"},
    "math": {"type": "string"},
    "monolingualtext": {"type": "string"},
    "musical-notation": {"type": "string"},
    "quantity": {"type": "number"},
    "string": {"type": "string"},
    "tabular-data": {"type": "string", "format": "iri-reference"},
    "time": {"type": "string", "format": "date"},
    "url": {"type": "string", "format": "uri"},
    "wikibase-form": {"type": "string", "format": "iri-reference"},
    "wikibase-lexeme": {"type": "string", "format": "iri-reference"},
    "wikibase-property": {"type": "string", "format": "iri-reference"},
    "wikibase-sense": {"type": "string", "format": "iri-reference"},
}
"""Every Wikidata datatype that has a scalar JSON form, and what it is.

``wikibase-item`` is absent because its form depends on whether the module
emits its range, which is a question about this run and not about the
datatype. ``globe-coordinate`` is absent because it has none: a point is two
numbers, and flattening it to a string would invent a notation no Wikidata
consumer reads back.

A format is only declared where the string really is a reference, because a
term is coerced to ``@id`` exactly when it carries one, and coercing an
external identifier would turn an ISNI into an IRI.
"""

DEFAULT_SAMPLE = 20
DEFAULT_USAGE = 0.25
DEFAULT_CAP = 20
DEFAULT_CLASSES = 400
"""What the four knobs default to.

The sample is 20 because it decides two things and both are cheap at that
size: which properties instances actually use, and which item the example
instance is built from. The usage floor is a quarter, which is the point where
a property stops being one entity's peculiarity. The cap is 20 slots, and the
ceiling of 400 classes is there because the closure under ``Q215627`` runs to
176,085 items, a class costs a query the service may take a minute over, and a
depth typed by mistake should cost a message rather than a day.
"""

_NON_WORD = re.compile(r"[^0-9A-Za-z]+")
_RESERVED = ("id", "type", "label", "description")
"""Term names the module itself uses, which a property may not take.

``id`` and ``type`` are the JSON-LD keyword aliases every schema here carries.
``label`` and ``description`` are reserved rather than used: a Wikidata label
is not a statement, and a property called "label" taking the slot would make
the two indistinguishable in an instance.
"""

_last = [0.0]


def _pause(gap: float) -> None:
    waited = time.time() - _last[0]
    if waited < gap:
        time.sleep(gap - waited)
    _last[0] = time.time()


def sparql(query: str, *, attempts: int = 6, timeout: int = 300) -> list[dict[str, Any]]:
    """One query, retried with a growing backoff.

    The service answers a cheap query in a second and refuses everything for a
    minute once it decides a client is busy, so a fixed pause is either too
    slow or too fast.
    """
    data = urllib.parse.urlencode({"query": query}).encode()
    for attempt in range(attempts):
        _pause(2.0)
        request = urllib.request.Request(  # noqa: S310 - one constant endpoint
            SPARQL,
            data=data,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/sparql-results+json",
                "Accept-Encoding": "gzip",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - same constant
                raw = response.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            # A label may hold a raw control character and the service passes
            # it through unescaped. Strict parsing refuses a whole page of
            # results over one string nothing here reads.
            return json.loads(raw.decode("utf-8"), strict=False)["results"]["bindings"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            if attempt == attempts - 1:
                raise
            print(f"    retry {attempt + 1}: {type(error).__name__} {str(error)[:60]}", flush=True)
            time.sleep(20 * (attempt + 1))
    return []


def api(params: dict[str, Any], *, gap: float = 1.0, attempts: int = 5) -> dict[str, Any]:
    """One ``wbgetentities`` read, rate limited, retried and gzipped.

    ``Accept-Encoding: gzip`` is not a micro-optimisation: a batch of 50
    entities with their full claims is several megabytes uncompressed.
    """
    url = WIKIDATA_API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    for attempt in range(attempts):
        _pause(gap)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})  # noqa: S310 - one constant endpoint
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 - same constant
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            if attempt == attempts - 1:
                raise
            print(f"    retry {attempt + 1}: {type(error).__name__} {str(error)[:60]}", flush=True)
            time.sleep(5 * (attempt + 1))
    return {}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_json(path: Path, payload: Any) -> None:
    """A cache file, written whole after every batch so a kill costs one batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")


def write_document(path: Path, payload: Any) -> None:
    """An emitted file, in the one form two runs agree on.

    Sorted keys and a trailing newline. Key order in JSON carries no meaning
    and a dump that follows insertion order makes the digest depend on the
    order the generator happened to build a dict in, which is not something a
    reader can check.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def qid(binding: dict[str, Any]) -> str:
    return str(binding["value"]).rsplit("/", 1)[-1]


def _snak(snak: dict[str, Any]) -> dict[str, Any] | None:
    """One claim's main value, flattened to what the generator can read.

    An unknown or a missing value is dropped rather than carried as a marker:
    JSON Schema has no way to write "some value", so there is nothing an
    instance could hold. A globe coordinate is dropped for the reason
    :data:`JSON_TYPE` gives.
    """
    if snak.get("snaktype") != "value":
        return None
    data = snak.get("datavalue") or {}
    value, kind = data.get("value"), data.get("type")
    if kind == "string":
        return {"kind": "string", "value": value}
    if kind == "monolingualtext" and isinstance(value, dict):
        return {"kind": "string", "value": value.get("text")} if value.get("language") == "en" else None
    if kind == "wikibase-entityid" and isinstance(value, dict):
        return {"kind": "item", "value": value.get("id")}
    if kind == "time" and isinstance(value, dict):
        return {"kind": "time", "value": value.get("time"), "precision": value.get("precision")}
    if kind == "quantity" and isinstance(value, dict):
        return {"kind": "quantity", "value": value.get("amount")}
    return None


def _values(body: dict[str, Any], prop: str) -> list[str]:
    """The item ids one property states, at truthy rank, in statement order."""
    found = []
    for claim in (body.get("claims") or {}).get(prop) or []:
        if claim.get("rank") == "deprecated":
            continue
        snak = _snak(claim.get("mainsnak") or {})
        if snak and snak["kind"] == "item" and snak["value"]:
            found.append(str(snak["value"]))
    return found


@dataclass(frozen=True)
class WikidataExample:
    """One real item of a class, and what it states.

    Carried so the module can ship an instance per class. A schema that
    validates and admits nothing real is half a check, and the only source of
    something real here is an item that already exists.
    """

    qid: str
    label: str
    claims: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass(frozen=True)
class WikidataClass:
    """One class, as the API reports it and the sample found it."""

    qid: str
    label: str
    description: str = ""
    parents: tuple[str, ...] = ()
    """Every ``P279`` target, not the first."""
    declared: tuple[str, ...] = ()
    """``P1963``, which is the class's own answer to what describes it."""
    observed: tuple[tuple[str, int], ...] = ()
    """Property id to the number of sampled instances stating it."""
    sampled: int = 0
    revision: int = 0
    example: WikidataExample | None = None


@dataclass(frozen=True)
class WikidataProperty:
    """One property, its datatype, and the classes its values are declared to be."""

    pid: str
    label: str
    datatype: str
    description: str = ""
    ranges: tuple[str, ...] = ()
    """The value-type constraint's classes, which is the only published
    statement about an item-valued property's range."""
    revision: int = 0


@dataclass(frozen=True)
class Module:
    """A generated module: the files, and what has to be said about them."""

    schemas: dict[str, dict[str, Any]]
    instances: dict[str, dict[str, Any]]
    names: dict[str, str]
    """QID to the name its schema is filed under."""
    rules: dict[str, str]
    """QID to the rule that chose its properties."""
    outside: dict[str, list[str]]
    """QID to the parents the walk did not reach, which no ``$ref`` can name."""
    dropped: dict[str, list[str]]
    """Why a property reached no schema, to the properties it cost."""
    demoted: dict[str, list[str]]
    """Schema to the embed targets written as a bare IRI because embedding
    them would have closed a cycle.

    Kept because it is the one place the module departs from the shape a
    reader would expect, and a reader asking why ``human`` embeds ``country``
    and ``country`` only references ``human`` has to be able to find out
    without rerunning the generator."""


def _words(label: str) -> list[str]:
    """A label cut into the ASCII words a term is built from.

    Decomposed first and the combining marks dropped, so ``AlloCiné`` becomes
    ``AlloCine`` rather than ``AlloCin``. Dropping the accent from a decomposed
    letter keeps the letter; dropping the character does not, and a term that
    has lost a letter is unreadable rather than merely anglicised.
    """
    folded = unicodedata.normalize("NFKD", label)
    stripped = "".join(mark for mark in folded if not unicodedata.combining(mark))
    return [word for word in _NON_WORD.split(stripped) if word]


def schema_name(label: str, item: str) -> str:
    """A class's file stem, from its English label.

    Pascal case, which is what the QUDT and schema.org modules look like and
    what a JSON-LD term can be. A class with no English label is filed under
    its QID: the item is still a class, and refusing it would drop a branch of
    the walk over a missing translation.
    """
    name = "".join(word[0].upper() + word[1:] for word in _words(label))
    return name or item


def property_name(label: str, prop: str) -> str:
    """A property's term, from its English label.

    Snake case, because that is what the slot is read and written as, and a
    label is a phrase: ``date of birth`` is ``date_of_birth``. A property
    whose label collides with a term the module needs for itself keeps its
    identifier instead, which is why :data:`_RESERVED` exists.
    """
    name = "_".join(word.lower() for word in _words(label))
    if not name or name in _RESERVED or name[0].isdigit():
        return prop.lower()
    return name


def _disambiguate(names: Mapping[str, str], suffix: Mapping[str, str]) -> dict[str, str]:
    """Give every member of a colliding group its identifier back.

    All of them, not the second one onwards. Which member keeps the bare name
    would otherwise depend on the order the group was built in, and a rename
    changes the digest, so the same module built from the same cache twice
    would have to be built in the same order to be the same module.
    """
    taken = Counter(names.values())
    return {key: f"{name}_{suffix[key]}" if taken[name] > 1 else name for key, name in sorted(names.items())}


def class_names(classes: Sequence[WikidataClass]) -> dict[str, str]:
    """Every class's file stem, with collisions resolved by QID."""
    stems = {cls.qid: schema_name(cls.label, cls.qid) for cls in classes}
    return _disambiguate(stems, {cls.qid: cls.qid for cls in classes})


def property_names(properties: Iterable[WikidataProperty]) -> dict[str, str]:
    """Every property's term, with collisions resolved by identifier.

    Resolved across the whole module rather than per class. Two classes
    declaring the same term for different properties would give one instance
    two meanings for one key, and the module is read as one vocabulary.
    """
    listed = sorted(properties, key=lambda prop: prop.pid)
    stems = {prop.pid: property_name(prop.label, prop.pid) for prop in listed}
    return _disambiguate(stems, {prop.pid: prop.pid.lower() for prop in listed})


def properties_for(
    cls: WikidataClass, known: Mapping[str, WikidataProperty], *, cap: int, usage: float
) -> tuple[tuple[str, ...], str]:
    """Which properties a class declares, and which rule said so.

    ``P1963`` wins wherever the class carries it, because it is the class's
    own statement and the sample is an inference about it. Where there is
    none, a property kept by the sample has to be stated by at least ``usage``
    of it, which is what separates what the class is from what one entity
    happened to have.

    The cap is spent on the properties the sample uses most either way. A
    declared property no sampled instance states still ranks, last, because
    the class said it belongs and an empty sample is not a contradiction.
    """
    counts = dict(cls.observed)
    declared = [prop for prop in cls.declared if prop in known]
    if declared:
        chosen, rule = declared, PROPERTIES_FOR_TYPE
    else:
        floor = max(1, math.ceil(usage * cls.sampled)) if cls.sampled else 1
        chosen, rule = [prop for prop, seen in cls.observed if prop in known and seen >= floor], "instances"
    ranked = sorted(chosen, key=lambda prop: (-counts.get(prop, 0), int(prop[1:])))
    return tuple(sorted(ranked[:cap], key=lambda prop: int(prop[1:]))), rule


def embed_targets(prop: WikidataProperty, names: Mapping[str, str]) -> tuple[str, ...]:
    """The schemas an item-valued property could embed, before cycles are cut.

    Its value-type constraint names classes; the ones the module emits can be
    pointed at, and a class outside the module can only be named by its IRI.
    All of them, not the first, for the reason every parent is kept.
    """
    if prop.datatype != "wikibase-item":
        return ()
    return tuple(sorted({names[item] + SUFFIX for item in prop.ranges if item in names}))


def value_schema(
    prop: WikidataProperty, names: Mapping[str, str], embeddable: Collection[str] = ()
) -> dict[str, Any] | None:
    """One property's value, as JSON Schema, or ``None`` when it has no form.

    An item-valued property is the only one whose answer depends on the run.
    ``embeddable`` is the subset of its targets this class may embed, which
    :func:`_acyclic` decides; a target left out of it is written as the IRI of
    the item the statement names, which is what RDF does with a reference
    anyway.
    """
    if prop.datatype != "wikibase-item":
        declared = JSON_TYPE.get(prop.datatype)
        return dict(declared) if declared else None
    targets = [target for target in embed_targets(prop, names) if target in embeddable]
    if len(targets) == 1:
        return {"$ref": targets[0]}
    if targets:
        return {"anyOf": [{"$ref": target} for target in targets]}
    return {"type": "string", "format": "iri-reference"}


def term_for(prop: WikidataProperty, value: dict[str, Any]) -> dict[str, Any] | str:
    """One property's ``@context`` term, matched to the shape of its value.

    An embed carries a scoped ``@context`` naming the schemas it may hold, so
    their terms resolve under this property and not against the root, where
    two classes sharing a key would describe each other. Without one the
    embedded object's keys reach no predicate and fall out of RDF entirely.

    A string that is a reference is coerced to ``@id`` and nothing else is. An
    external identifier is text that looks like a reference and is not one,
    and coercing it would publish an ISNI as an IRI.
    """
    iri = f"wdt:{prop.pid}"
    if isinstance(value.get("$ref"), str):
        return {"@id": iri, "@context": value["$ref"]}
    branches = value.get("anyOf")
    if isinstance(branches, list):
        return {"@id": iri, "@context": [branch["$ref"] for branch in branches]}
    if value.get("format") in ("uri", "iri-reference"):
        return {"@id": iri, "@type": "@id"}
    return iri


def _acyclic(edges: Mapping[str, tuple[str, ...]]) -> dict[str, frozenset[str]]:
    """Which embeds can stay embeds, and which become a bare reference.

    A scoped ``@context`` that closes a cycle cannot be resolved by PyLD or by
    jsonld.js: both recurse until they run out of stack or heap, so the
    validator declines to round-trip a schema that reaches one and the module
    would ship with its JSON-LD half unchecked. A class that embeds its own
    type is worse than unresolvable: the embedded node carries the same
    ``@type`` as its container, and framing cannot tell the two apart.

    So an embed that would close a cycle is written as the IRI of the item the
    statement names instead, which is what RDF makes of a reference in any
    case. The edges are considered in sorted order and one is kept only if its
    target cannot already reach its source, so which edge of a cycle loses is
    decided by the file names and not by the order the module was built in.
    """
    kept: dict[str, set[str]] = {source: set() for source in edges}

    def reaches(source: str, target: str, seen: frozenset[str] = frozenset()) -> bool:
        if source == target:
            return True
        return any(reaches(step, target, seen | {source}) for step in sorted(kept.get(source, ())) if step not in seen)

    for source in sorted(edges):
        for target in sorted(edges[source]):
            if not reaches(target, source):
                kept[source].add(target)
    return {source: frozenset(targets) for source, targets in kept.items()}


def _base_context() -> dict[str, Any]:
    """The context a class with no parent in the module has to carry itself.

    ``@version`` and the two keyword aliases are what the OO-LD rules ask of
    every document, and a class whose parents are all outside the walk
    inherits them from nothing.
    """
    return {
        "@version": 1.1,
        "wd": ENTITY,
        "wdt": DIRECT,
        "id": "@id",
        "type": {"@id": "@type", "@container": "@set"},
    }


def _inherited(
    item: str,
    parents: Mapping[str, tuple[str, ...]],
    chosen: Mapping[str, tuple[str, ...]],
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    """Every property an ancestor of this class already declares.

    A subclass declares what it adds and nothing else, the way the QUDT and
    schema.org modules do. Redeclaring an inherited slot would also put the
    derived schema in front of the validator's narrowing rule for no gain,
    since the two declarations would be the same bytes. ``seen`` is the guard
    a real cycle needs: ``P279`` is asserted by editors and does run in
    circles.
    """
    found: set[str] = set()
    for parent in parents.get(item, ()):
        if parent in seen:
            continue
        found |= set(chosen.get(parent, ())) | _inherited(parent, parents, chosen, seen | {item})
    return found


def build_schema(
    cls: WikidataClass,
    chosen: Sequence[str],
    *,
    names: Mapping[str, str],
    terms: Mapping[str, str],
    properties: Mapping[str, WikidataProperty],
    parents: Sequence[str],
    embeddable: Collection[str] = (),
) -> dict[str, Any]:
    """One class as an OO-LD schema: JSON Schema and JSON-LD in one document.

    ``allOf`` and ``@context`` list the same parents in the same order,
    because a schema is read as a remote context as it stands and the
    validator refuses one where the two disagree.
    """
    name = names[cls.qid]
    own = name + SUFFIX
    refs = [names[parent] + SUFFIX for parent in parents]
    term_map: dict[str, Any] = {} if refs else _base_context()
    term_map["wd"], term_map["wdt"] = ENTITY, DIRECT
    term_map[name] = f"wd:{cls.qid}"

    slots: dict[str, Any] = {"type": {"default": [name]}}
    if not refs:
        slots["type"] = {"type": "array", "items": {"type": "string"}, "default": [name]}
    for prop in chosen:
        value = value_schema(properties[prop], names, embeddable)
        if value is None:
            continue
        described = properties[prop].description
        slots[terms[prop]] = {
            **value,
            "title": properties[prop].label,
            **({"description": described} if described else {}),
        }
        term_map[terms[prop]] = term_for(properties[prop], value)

    schema: dict[str, Any] = {
        "$schema": META_SCHEMA,
        "$id": own,
        "x-oold-uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, ENTITY + cls.qid)),
        "x-oold-version": SCHEMA_VERSION,
        "x-oold-iri": f"wd:{cls.qid}",
        # The term and not the IRI, because the pinned `type` default has to
        # agree with it and that is what an instance writes. The term map
        # resolves it, so a processor still reaches wd:Q515.
        "x-oold-instance-rdf-type": [name],
        "title": name,
        "x-oold-multilang-title": {"en": cls.label},
    }
    if cls.description:
        schema["description"] = cls.description
    if refs:
        schema["allOf"] = [{"$ref": ref} for ref in refs]
        schema["@context"] = [*refs, term_map]
    else:
        schema["type"] = "object"
        schema["@context"] = term_map
    schema["properties"] = slots
    return schema


DAY = 11
"""Wikidata's precision code for a date stated to the day."""


def _written(snak: dict[str, Any], value: dict[str, Any]) -> Any:
    """One statement as the value its slot admits, or ``None``.

    Three forms are chosen by what survives a round trip through RDF rather
    than by taste, because the validator checks that an instance comes back as
    it went in. An item is written compact, ``wd:Q30``, which is what
    compaction against this module's own context produces. A quantity with no
    fraction is written as an integer, because ``238.0`` returns as ``238``.
    And an embedded item carries the type its schema defaults to, because a
    node with nothing but an ``id`` comes back as a bare string: the type is
    the embedding schema's own claim, already made when the value-type
    constraint became a ``$ref``, and not a new one about the item.

    A union of embeds states nothing in the instance. The module cannot say
    which of the branches a given item is, and picking one would file a class
    Wikidata never stated.

    A date coarser than a day is dropped. Wikidata records the precision a
    date was stated at, ``format: date`` admits a full date only, and writing
    ``1930`` into it would publish an instance the schema does not describe.
    """
    kind = snak.get("kind")
    if kind == "item":
        target = f"wd:{snak['value']}"
        if value.get("anyOf"):
            return None
        embedded = value.get("$ref")
        return {"id": target, "type": [str(embedded)[: -len(SUFFIX)]]} if embedded else target
    if kind == "quantity":
        try:
            number = float(str(snak["value"]))
        except ValueError:
            return None
        return int(number) if number.is_integer() else number
    if kind == "time":
        if snak.get("precision") != DAY:
            return None
        return str(snak["value"])[1:11]
    return str(snak["value"]) if snak.get("value") else None


def build_instance(
    cls: WikidataClass,
    slots: Mapping[str, dict[str, Any]],
    *,
    name: str,
    terms: Mapping[str, str],
) -> dict[str, Any] | None:
    """One real item of the class, as a document its schema admits.

    ``None`` when the walk found no instance at all, which is what an abstract
    class looks like from here. Nothing is invented to fill the gap: an
    instance built from a plausible value would be the one thing in this
    module that is not a Wikidata statement.

    One item is embedded once in a document. Framing embeds a node at one
    place and returns every later occurrence as a bare reference, so a person
    who directed and wrote the same film would not come back as they went in;
    the second slot takes the next statement it has, or none.
    """
    if cls.example is None:
        return None
    document: dict[str, Any] = {
        "@context": name + SUFFIX,
        "$schema": name + SUFFIX,
        "id": f"wd:{cls.example.qid}",
        "type": [name],
    }
    embedded: set[str] = set()
    for prop, claims in sorted(cls.example.claims.items()):
        slot = slots.get(terms.get(prop, ""))
        if slot is None:
            continue
        for snak in claims:
            written = _written(snak, slot)
            if written is None:
                continue
            if isinstance(written, dict):
                if written["id"] in embedded:
                    continue
                embedded.add(str(written["id"]))
            document[terms[prop]] = written
            break
    return document


def build_module(
    classes: Sequence[WikidataClass],
    properties: Mapping[str, WikidataProperty],
    *,
    cap: int = DEFAULT_CAP,
    usage: float = DEFAULT_USAGE,
) -> Module:
    """The whole module, from what the harvest read and nothing else.

    Pure, so the rules can be tested without an hour of network. Everything
    that reaches a file is decided here; the harvest decides only what is
    available to decide from.
    """
    listed = sorted(classes, key=lambda cls: int(cls.qid[1:]))
    names = class_names(listed)
    terms = property_names(properties.values())

    chosen: dict[str, tuple[str, ...]] = {}
    rules: dict[str, str] = {}
    for cls in listed:
        chosen[cls.qid], rules[cls.qid] = properties_for(cls, properties, cap=cap, usage=usage)

    parents = {cls.qid: tuple(p for p in sorted(cls.parents, key=lambda q: int(q[1:])) if p in names) for cls in listed}

    declares: dict[str, list[str]] = {}
    embeds: dict[str, tuple[str, ...]] = {}
    dropped: dict[str, list[str]] = {}
    for cls in listed:
        above = _inherited(cls.qid, parents, chosen)
        mine = [prop for prop in chosen[cls.qid] if prop not in above]
        declares[cls.qid] = mine
        found: list[str] = []
        for prop in mine:
            if value_schema(properties[prop], names) is None:
                dropped.setdefault(properties[prop].datatype, []).append(prop)
            found += embed_targets(properties[prop], names)
        embeds[names[cls.qid] + SUFFIX] = tuple(sorted(set(found)))

    embeddable = _acyclic(embeds)
    schemas: dict[str, dict[str, Any]] = {}
    for cls in listed:
        own = names[cls.qid] + SUFFIX
        schemas[own] = build_schema(
            cls,
            declares[cls.qid],
            names=names,
            terms=terms,
            properties=properties,
            parents=parents[cls.qid],
            embeddable=embeddable[own],
        )

    # Instances come after every schema, because a class's slots include the
    # ones it takes from a parent and a parent is not always the lower QID.
    instances: dict[str, dict[str, Any]] = {}
    for cls in listed:
        slots = _effective(schemas[names[cls.qid] + SUFFIX], schemas)
        instance = build_instance(cls, slots, name=names[cls.qid], terms=terms)
        if instance is not None:
            instances[names[cls.qid] + INSTANCE_SUFFIX] = instance

    return Module(
        schemas=schemas,
        instances=instances,
        names=names,
        rules=rules,
        outside={
            cls.qid: sorted(set(cls.parents) - set(names), key=lambda q: int(q[1:])) for cls in listed if cls.parents
        },
        dropped={reason: sorted(set(found)) for reason, found in sorted(dropped.items())},
        demoted={
            source: sorted(set(targets) - embeddable[source])
            for source, targets in sorted(embeds.items())
            if set(targets) - embeddable[source]
        },
    )


def _effective(
    schema: Mapping[str, Any], built: Mapping[str, dict[str, Any]], seen: frozenset[str] = frozenset()
) -> dict[str, dict[str, Any]]:
    """A schema's slots including the ones it takes from above.

    The instance is built against these rather than against the class's own
    declarations, because a document about a film states the film's title and
    the title is declared two classes up.
    """
    found: dict[str, dict[str, Any]] = {}
    for entry in schema.get("allOf") or []:
        ref = str(entry.get("$ref"))
        if ref in built and ref not in seen:
            found |= _effective(built[ref], built, seen | {str(schema.get("$id"))})
    return found | {name: slot for name, slot in (schema.get("properties") or {}).items() if name != "type"}


def module_meta(
    module: Module,
    classes: Sequence[WikidataClass],
    properties: Mapping[str, WikidataProperty],
    *,
    roots: Sequence[str],
    depth: int,
    retrieved: str,
    sample: int,
    cap: int,
    usage: float,
    digest: str = "",
) -> dict[str, Any]:
    """What the module cannot say about itself.

    The roots and the depth, because the same generator produces a different
    module from different arguments. The revision of every entity read,
    because that is what pins the module to a state of Wikidata an editor
    cannot change under it. And the rule that chose each class's properties,
    because ``P1963`` and a sample are not the same claim.
    """
    by_rule = Counter(module.rules.values())
    return {
        "generator": "scripts/generate_wikidata_oold.py",
        "roots": sorted(roots, key=lambda q: int(q[1:])),
        "depth": depth,
        "retrieved_at": retrieved,
        "source": {"api": WIKIDATA_API, "sparql": SPARQL, "user_agent": USER_AGENT},
        "arguments": {"sample": sample, "max_properties": cap, "min_usage": usage},
        "counts": {
            "classes": len(module.schemas),
            "instances": len(module.instances),
            "properties": len(properties),
            "by_rule": dict(sorted(by_rule.items())),
            "parents_outside": sum(len(found) for found in module.outside.values()),
            "classes_without_instance": len(module.schemas) - len(module.instances),
            "demoted_embeds": sum(len(found) for found in module.demoted.values()),
        },
        "property_rule": dict(sorted(module.rules.items())),
        "parents_outside": {item: found for item, found in sorted(module.outside.items()) if found},
        "dropped_properties": module.dropped,
        "demoted_embeds": dict(sorted(module.demoted.items())),
        "names": dict(sorted(module.names.items())),
        "revisions": dict(
            sorted(
                (
                    {cls.qid: cls.revision for cls in classes if cls.revision}
                    | {prop.pid: prop.revision for prop in properties.values() if prop.revision}
                ).items()
            )
        ),
        "digest": digest,
    }


def write_module(module: Module, out: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Write the module, then pin it.

    The digest covers the schemas and is written into ``_meta.json``
    afterwards, which is why it is taken over ``*.schema.json`` and not over
    every file in the directory: a file that records the digest cannot be part
    of it.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name, document in sorted(module.schemas.items()):
        write_document(out / name, document)
    for name, document in sorted(module.instances.items()):
        write_document(out / name, document)
    found = corpus_digest(out, "*" + SUFFIX)
    recorded = {**meta, "digest": str(found), "counts": {**meta["counts"], "digested": found.files}}
    write_document(out / "_meta.json", recorded)
    return recorded


def _frontier(cache: Path, roots: Sequence[str], depth: int, limit: int, offline: bool) -> list[str]:
    """The class set, walked down ``P279`` one level at a time.

    Resumable at the level of one parent: the cache holds every parent already
    asked for, so a rerun asks only about the frontier it has not reached.
    """
    path = cache / "subclasses.json"
    known: dict[str, list[str]] = read_json(path) or {}
    found = list(roots)
    frontier = list(roots)
    for level in range(depth):
        todo = [item for item in frontier if item not in known]
        for start in range(0, len(todo), 50):
            chunk = todo[start : start + 50]
            if offline:
                raise SystemExit(f"{len(todo)} classes are not in {path} and --offline forbids fetching them")
            rows = sparql(SUBCLASS_QUERY % " ".join("wd:" + item for item in chunk))
            for item in chunk:
                known.setdefault(item, [])
            for row in rows:
                known[qid(row["parent"])].append(qid(row["child"]))
            write_json(path, known)
        children = sorted({child for item in frontier for child in known.get(item, [])}, key=lambda q: int(q[1:]))
        # Truncated here and not after the walk, so the ceiling bounds what the
        # next level is asked about as well as what is kept.
        frontier = [child for child in children if child not in found][: max(0, limit - len(found))]
        found += frontier
        print(f"  level {level + 1}: {len(frontier)} new classes, {len(found)} kept", flush=True)
        if len(found) >= limit:
            print(f"  stopping at the ceiling of {limit} classes", flush=True)
            break
    return sorted(set(found), key=lambda q: int(q[1:]))


def _entities(cache: Path, name: str, ids: Sequence[str], offline: bool) -> dict[str, Any]:
    """Read entities in batches of 50, skipping what the cache already holds.

    Only what was asked for comes back, never the rest of the cache. The cache
    accumulates across runs with different roots, and a module whose record
    counted every property a previous run happened to read would depend on
    what else the machine had already fetched.
    """
    path = cache / name
    known: dict[str, Any] = read_json(path) or {}
    todo = [item for item in dict.fromkeys(ids) if item not in known]
    if todo and offline:
        raise SystemExit(f"{len(todo)} entities are not in {path} and --offline forbids fetching them")
    for start in range(0, len(todo), 50):
        chunk = todo[start : start + 50]
        payload = api({
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": "info|claims|labels|descriptions|datatype",
            "languages": "en",
        })
        known |= payload.get("entities") or {}
        write_json(path, known)
        print(f"  {name} {start + len(chunk)}/{len(todo)}", flush=True)
    return {item: known[item] for item in dict.fromkeys(ids) if item in known}


def _sample(cache: Path, items: Sequence[str], size: int, offline: bool) -> dict[str, list[str]]:
    """A few instances of each class, which two things are read off.

    Which properties instances actually use, and which item the example
    instance is built from. One query per class, cached per class, because a
    class whose query timed out must not cost the ones already answered.
    """
    path = cache / "instances.json"
    known: dict[str, list[str]] = read_json(path) or {}
    todo = [item for item in items if item not in known]
    if todo and offline:
        raise SystemExit(f"{len(todo)} classes have no sample in {path} and --offline forbids fetching one")
    for index, item in enumerate(todo):
        try:
            rows = sparql(INSTANCE_QUERY % {"qid": item, "sample": size}, attempts=2)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            print(f"  {item} has no sample: {type(error).__name__}", flush=True)
            continue
        known[item] = sorted({qid(row["item"]) for row in rows}, key=lambda q: int(q[1:]))
        write_json(path, known)
        print(f"  sample {index + 1}/{len(todo)}: {item} has {len(known[item])} instances", flush=True)
    return {item: known[item] for item in items if item in known}


def _property_ranges(body: Mapping[str, Any]) -> tuple[str, ...]:
    """The classes a property's value-type constraint admits.

    ``P2302`` carries every kind of constraint a property has, and only
    ``Q21510865`` says anything about the range. The classes are its ``P2308``
    qualifiers.
    """
    found: list[str] = []
    for claim in (body.get("claims") or {}).get(PROPERTY_CONSTRAINT) or []:
        snak = _snak(claim.get("mainsnak") or {})
        if not snak or snak.get("value") != VALUE_TYPE_CONSTRAINT:
            continue
        for qualifier in (claim.get("qualifiers") or {}).get(CONSTRAINT_CLASS) or []:
            value = _snak(qualifier)
            if value and value["kind"] == "item":
                found.append(str(value["value"]))
    return tuple(sorted(set(found), key=lambda q: int(q[1:])))


def _english(body: Mapping[str, Any], field_name: str) -> str:
    return str(((body.get(field_name) or {}).get("en") or {}).get("value") or "")


def harvest(args: argparse.Namespace) -> tuple[list[WikidataClass], dict[str, WikidataProperty], str]:
    """Everything the module is built from, fetched once and cached.

    The order is forced by what depends on what: the walk names the classes,
    the classes name their sampled instances, the instances and the classes
    together name the properties, and only then can a property's datatype be
    read.
    """
    cache = args.cache
    items = args.classes or _frontier(cache, args.roots, args.depth, args.max_classes, args.offline)
    print(f"{len(items)} classes")

    bodies = _entities(cache, "classes.json", items, args.offline)
    sampled = _sample(cache, items, args.sample, args.offline)
    seen = sorted({item for found in sampled.values() for item in found}, key=lambda q: int(q[1:]))
    examples = _entities(cache, "examples.json", seen, args.offline)

    usage: dict[str, Counter[str]] = {item: Counter() for item in items}
    for item in items:
        for instance in sampled.get(item, []):
            usage[item].update(prop for prop in (examples.get(instance, {}).get("claims") or {}) if prop[0] == "P")

    wanted = {prop for body in bodies.values() for prop in _values(body, PROPERTIES_FOR_TYPE)}
    wanted |= {prop for counted in usage.values() for prop in counted}
    described = _entities(cache, "properties.json", sorted(wanted, key=lambda p: int(p[1:])), args.offline)

    properties = {
        prop: WikidataProperty(
            pid=prop,
            label=_english(body, "labels") or prop,
            datatype=str(body.get("datatype") or ""),
            description=_english(body, "descriptions"),
            ranges=_property_ranges(body),
            revision=int(body.get("lastrevid") or 0),
        )
        for prop, body in sorted(described.items())
        if body.get("datatype")
    }
    classes = [
        WikidataClass(
            qid=item,
            label=_english(bodies[item], "labels"),
            description=_english(bodies[item], "descriptions"),
            parents=tuple(dict.fromkeys(_values(bodies[item], SUBCLASS_OF))),
            declared=tuple(dict.fromkeys(_values(bodies[item], PROPERTIES_FOR_TYPE))),
            observed=tuple(sorted(usage[item].items())),
            sampled=len(sampled.get(item, [])),
            revision=int(bodies[item].get("lastrevid") or 0),
            example=_example(sampled.get(item, []), examples),
        )
        for item in items
        if item in bodies
    ]
    return classes, properties, _retrieved(cache)


def _example(sampled: Sequence[str], examples: Mapping[str, Any]) -> WikidataExample | None:
    """The sampled item with the most statements, which is the richest instance.

    Ties go to the lower QID, so the choice does not depend on the order the
    service returned the sample in.
    """
    found = [(item, examples[item]) for item in sampled if item in examples]
    if not found:
        return None
    item, body = max(found, key=lambda pair: (len(pair[1].get("claims") or {}), -int(pair[0][1:])))
    claims: dict[str, list[dict[str, Any]]] = {}
    for prop, statements in sorted((body.get("claims") or {}).items()):
        kept = [snak for snak in (_snak(claim.get("mainsnak") or {}) for claim in statements) if snak]
        if kept:
            claims[prop] = kept
    return WikidataExample(qid=item, label=_english(body, "labels"), claims=claims)


def _retrieved(cache: Path) -> str:
    """The date the cache was last written, which is the module's query date.

    Read off the files rather than from the clock, so re-emitting a module
    from a cache fetched last week dates it to last week and two runs over one
    cache agree byte for byte.
    """
    stamps = [path.stat().st_mtime for path in sorted(cache.glob("*.json"))]
    latest = max(stamps) if stamps else time.time()
    return datetime.fromtimestamp(latest, tz=UTC).date().isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--roots", nargs="*", default=["Q11424"], help="Wikidata classes to walk down from")
    parser.add_argument("--classes", nargs="*", default=[], help="an explicit class list, which skips the walk")
    parser.add_argument("--depth", type=int, default=1, help="how many P279 levels below the roots to take")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="instances read per class")
    parser.add_argument("--min-usage", type=float, default=DEFAULT_USAGE, help="share of the sample a property needs")
    parser.add_argument("--max-properties", type=int, default=DEFAULT_CAP, help="slots a schema may declare")
    parser.add_argument("--max-classes", type=int, default=DEFAULT_CLASSES, help="ceiling on the walk")
    parser.add_argument("--cache", type=Path, default=Path(".cache/wikidata_oold"))
    parser.add_argument("--out", type=Path, default=Path(".cache/wikidata_oold/schemas"))
    parser.add_argument("--offline", action="store_true", help="read the cache and refuse to fetch anything")
    parser.add_argument("--apply", action="store_true", help="write the module; without it nothing is written")
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    classes, properties, retrieved = harvest(args)
    module = build_module(classes, properties, cap=args.max_properties, usage=args.min_usage)
    meta = module_meta(
        module,
        classes,
        properties,
        roots=args.classes or args.roots,
        depth=0 if args.classes else args.depth,
        retrieved=retrieved,
        sample=args.sample,
        cap=args.max_properties,
        usage=args.min_usage,
    )
    counts = meta["counts"]
    print(
        f"{counts['classes']} schemas, {counts['instances']} instances, {counts['properties']} properties read, "
        f"{counts['by_rule']}, {counts['parents_outside']} parents outside the walk"
    )
    if module.dropped:
        counted = {reason: len(found) for reason, found in module.dropped.items()}
        print(f"  no JSON form: {counted}")
    if not args.apply:
        print("\ndry run; pass --apply")
        return 0

    recorded = write_module(module, args.out, meta)
    print(f"written to {args.out}, digest {recorded['digest']} over {recorded['counts']['digested']} schemas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
