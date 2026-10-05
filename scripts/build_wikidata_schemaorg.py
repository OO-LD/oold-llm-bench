"""Turn the Wikidata and Wikipedia harvests into a gradeable corpus.

Nothing here touches the network. ``harvest_wikidata_schemaorg.py`` produced
six files and this reads them, so the rule that decides what a document states
can be changed and remeasured in seconds instead of in an hour.

The work is one join and one decision. The join is Wikidata's: ``P1709`` says
which Wikidata item a schema.org class corresponds to and ``P1628`` says the
same for a property, so a statement about an item becomes a candidate value
for a schema.org slot. The decision is whether the article's lead states that
value, and that is the whole of the ground truth: a statement the lead does
not make is kept as a distractor and never scored, because an extractor cannot
read what the document does not say and grading it against the knowledge base
measures the knowledge base.

Two rules exist because Wikidata will otherwise hand over something false.
Statements are read at truthy rank, preferred where a property has one and
normal otherwise, and any statement carrying a start time, an end time or a
point in time is dropped whatever its rank. Vilnius carries 18 ``P17 country``
statements, and the lead states one of them.

Three more exist because real text has a failure a generated corpus cannot.
A slot can be answerable from the class rather than from the document, and the
wrong-document control is the only thing that sees it: 71% of the films are
from the United States, which took a Movie-only grid to 0.111 against a
ceiling of 0.01. :data:`GUESSABLE` takes such a property off every class,
:data:`VALUE_SHARE` keeps the committed sample flat in the values that are
left, and :data:`CELL_FLOOR` refuses to publish a class too small for the
control to be measuring the corpus rather than one document.

The build verifies itself before it writes. A perfect answer has to score 1.00
through the real grader, and the four negative controls have to stay at their
floor over the whole corpus and over each class on its own, on the corpus as
built and not on a fixture. The per-class check is the one that bites, because
a cell of this benchmark is one class. A corpus that fails is not written at
all, and the scores it passed with go into the file.

Two outputs, and the second is not committed. ``--out`` carries the truth, the
catalogue, the grounding rates and the provenance of each document: its url,
its revision id and the sha256 of the lead. ``--documents`` carries the leads
themselves, which are CC BY-SA 4.0 and stay out of an Apache-2.0 repository.
A revision id is immutable, so the leads can be fetched again exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oold_llm_bench.corpus.schemaorg import CATALOGUE_SLOTS, EXCLUDED_PROPERTIES, Kind
from oold_llm_bench.corpus.wikidata_schemaorg import (
    CLASS_QUERY,
    EQUIVALENT_CLASS,
    EQUIVALENT_PROPERTY,
    FACT_LICENCE,
    MIN_FACTS,
    MIN_SLOTS,
    NAME,
    TEXT_LICENCE,
    load_entities,
    stated_in,
    truthy,
    written_forms,
)
from oold_llm_bench.extract.json_answer import ID_KEYS, PROVENANCE_KEYS, UNIT_KEYS, VALUE_KEYS
from oold_llm_bench.grading import Dimension, TripleSet, make_triples
from oold_llm_bench.grading.score import score_task
from oold_llm_bench.results.record import Environment, ModelSpec
from oold_llm_bench.runner import Condition, ExperimentConfig
from oold_llm_bench.runner.preflight import CONTROL_CEILING, SPELLING_CEILING, control_scores

SCHEMA = "schema:"

RANGE_KIND: tuple[tuple[str, Kind], ...] = (
    ("Date", Kind.DATE),
    ("DateTime", Kind.DATETIME),
    ("Integer", Kind.INTEGER),
    ("Number", Kind.NUMBER),
    ("Float", Kind.NUMBER),
    ("Boolean", Kind.BOOLEAN),
    ("Text", Kind.TEXT),
    ("URL", Kind.URL),
    ("Time", Kind.TIME),
    ("Duration", Kind.DURATION),
)
"""Which datatype in a property's range decides the slot's kind, best first.

schema.org ranges are unions and a property often names several: ``duration``
is a ``Duration`` or a ``QuantitativeValue``, ``addressCountry`` is a
``Country`` or a ``Text``. The order is by how much the kind constrains an
answer, so a property that can be a date is a date slot and not a text slot.
"""

GROUNDABLE = (Kind.TEXT, Kind.DATE, Kind.DATETIME, Kind.NUMBER, Kind.INTEGER, Kind.ENUM)
"""Kinds a lead can be asked to have stated.

``URL`` and ``Duration`` are out and the reason is measured, not assumed: of
the mapped statements on the draw, no lead states a duration in a form that
resolves to the ISO 8601 the slot wants, and a lead does not print a url at
all. ``Boolean`` is out because Wikidata has no boolean statement to map.
``Time`` is out because a bare clock time has no subject in an encyclopaedia
lead.
"""

RESERVED = frozenset(
    {name for group in (ID_KEYS, VALUE_KEYS, UNIT_KEYS, PROVENANCE_KEYS) for name in group} | EXCLUDED_PROPERTIES
)
"""Property names the shared extractor reads as structure rather than content.

``value`` and ``unit`` on one entity are a quantity to
:func:`~oold_llm_bench.extract.json_answer.extract_json`, and ``id`` is the
key an edge points at. A corpus that put truth in those names would be graded
against the extractor's envelope. The schema.org bookkeeping properties go
with them, for the reason
:data:`~oold_llm_bench.corpus.schemaorg.EXCLUDED_PROPERTIES` states.
"""

VALUES_PER_PROPERTY = 4
"""How many values one property may contribute to one entity.

A film carries eleven ``genre`` statements and a lead names two. Keeping all
of them would make recall unreachable by construction on the properties with
the most statements, which are exactly the ones a lead summarises.
"""

GUESSABLE = 0.2
"""The share of one class's documents a property's commonest value may take.

A slot whose commonest value covers most of a class is answerable from the
class and not from the document. 71% of the films drawn here are from the
United States, so an answer copied wholesale from another film is right about
``countryOfOrigin`` seven times in ten, and the wrong-document control
measures exactly that: on a Movie-only grid it scored 0.111 with the slot in,
against a ceiling of 0.01.

What this removes is the categorical half of the mapping: country, genre,
language, gender, occupation, nationality, publisher. What it keeps is the
half a document has to be read for. The removed properties stay in the
per-property report with their grounding rate and their share, because a slot
dropped for being guessable is a finding about the corpus and not an absence.

A fifth and not a twentieth, because the two rules divide the work.
:data:`VALUE_SHARE` balances the sample, which handles a value that recurs a
little; this removes a slot that recurs so much that no sample of it could be
balanced. Measured: at a twentieth the rule also takes ``creator``, ``about``
and ``director``, Movie falls from 1,123 documents to 2, and what is left is
not a corpus.
"""

CELL_FLOOR = 30
"""How many documents a class needs before its entities are published.

Not a quality threshold on the documents, which have already cleared one, but
the point below which the negative controls stop measuring the corpus. The
wrong-document control answers one other document of the grid, so on a grid of
three it reports what a third of the grid happens to share: the three software
applications that survived the mapping scored 0.095 on it, against a ceiling
of 0.01, because two of them were made by Google.

A class under the floor keeps its line in the grounding statistics with the
count it reached, so a report can say which cells the corpus had and which it
did not.
"""

VALUE_SHARE = 0.04
"""How much of one class's committed sample may share one value.

The wrong-document control answers a different document of the same grid
correctly, so a value the donor holds and ``m`` others share puts about
``m / (n k)`` on the mean, where ``n`` is the grid and ``k`` the facts a
document carries. With a hundred documents of four facts, the 0.01 ceiling is
four documents sharing one value. This is that figure.

It is a sampling rule and not a truth rule. No fact is changed and no fact is
dropped: an entity whose values are already well represented in the sample is
passed over for one whose values are not, and the entity stays in the resolved
count the grounding rate is measured over. The effect is that the committed
corpus is flatter in its values than the draw was, which is recorded as the
number of entities the balance passed over.
"""


def _ids(node: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = node.get(key)
    if raw is None:
        return ()
    return tuple(one["@id"] for one in (raw if isinstance(raw, list) else [raw]) if isinstance(one, dict))


@dataclass
class Vocabulary:
    """schema.org's own JSON-LD, reduced to what a catalogue needs.

    The lineage and the enumeration members are resolved once at construction.
    schema.org declares 1,016 classes and 1,694 properties, every class asks
    for its ancestors several times, and walking the graph on each ask turned
    a build into a quarter of an hour.
    """

    nodes: dict[str, dict[str, Any]]
    parents_of: dict[str, tuple[str, ...]]
    ancestors_of: dict[str, frozenset[str]]
    members_of_type: dict[str, tuple[str, ...]]

    @classmethod
    def read(cls, doc: dict[str, Any]) -> Vocabulary:
        nodes = {node["@id"]: node for node in doc["@graph"]}
        parents = {
            name[len(SCHEMA) :]: tuple(p[len(SCHEMA) :] for p in _ids(node, "rdfs:subClassOf"))
            for name, node in nodes.items()
            if name.startswith(SCHEMA)
        }

        def walk(name: str, seen: frozenset[str]) -> frozenset[str]:
            found: set[str] = set()
            for parent in parents.get(name, ()):
                if parent in seen:
                    continue
                found.add(parent)
                found |= walk(parent, seen | {parent})
            return frozenset(found)

        members: dict[str, list[str]] = {}
        for name, node in nodes.items():
            kind = node.get("@type")
            for one in kind if isinstance(kind, list) else [kind]:
                if isinstance(one, str) and one.startswith(SCHEMA) and name.startswith(SCHEMA):
                    members.setdefault(one[len(SCHEMA) :], []).append(name[len(SCHEMA) :])
        return cls(
            nodes=nodes,
            parents_of=parents,
            ancestors_of={name: walk(name, frozenset()) for name in parents},
            members_of_type={name: tuple(sorted(found)) for name, found in members.items()},
        )

    def parents(self, name: str) -> tuple[str, ...]:
        return self.parents_of.get(name, ())

    def ancestors(self, name: str) -> frozenset[str]:
        return self.ancestors_of.get(name, frozenset())

    def is_class(self, name: str) -> bool:
        """Whether a ``P1709`` target is a class and not an enumeration member.

        23 of the targets are members such as Monday, Paperback and
        StudioAlbum, and ten more are datatypes. Both are asked of the
        vocabulary rather than listed here, so a schema.org release that
        promotes one of them needs no edit.
        """
        node = self.nodes.get(SCHEMA + name)
        if node is None or node.get("@type") != "rdfs:Class":
            return False
        return not ({"Enumeration", "DataType"} & (self.ancestors(name) | {name}))

    def description(self, name: str, limit: int = 160) -> str:
        comment = " ".join(str(self.nodes.get(SCHEMA + name, {}).get("rdfs:comment") or "").split())
        return comment if len(comment) <= limit else comment[: limit - 1].rsplit(" ", 1)[0] + "…"

    def kind_of(self, prop: str) -> tuple[Kind, tuple[str, ...]] | None:
        """The slot kind a property's range implies, and its members if closed.

        An item-valued range lands on text, because this corpus writes the
        target's name where schema.org would nest the target. A lead says
        "directed by Roman Polanski" and the answer is that string; nothing in
        a lead carries an entity a nested object could be built from.
        """
        node = self.nodes.get(SCHEMA + prop)
        if node is None:
            return None
        ranges = [r[len(SCHEMA) :] for r in _ids(node, "schema:rangeIncludes")]
        for name, kind in RANGE_KIND:
            if name in ranges:
                return kind, ()
        for name in ranges:
            members = self.members_of_type.get(name, ())
            if members and "Enumeration" in self.ancestors(name):
                return Kind.ENUM, members
        return (Kind.TEXT, ()) if ranges else None

    def domains_of(self, prop: str) -> set[str]:
        return {d[len(SCHEMA) :] for d in _ids(self.nodes.get(SCHEMA + prop, {}), "schema:domainIncludes")}

    def superseded(self, name: str) -> bool:
        return bool(_ids(self.nodes.get(SCHEMA + name, {}), "schema:supersededBy"))


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run harvest_wikidata_schemaorg.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def read_vocabulary(cache: Path) -> Vocabulary:
    return Vocabulary.read(read_json(cache / "schemaorg.jsonld"))


def class_map(cache: Path, vocabulary: Vocabulary) -> tuple[dict[str, list[str]], Counter[str]]:
    """Which Wikidata items each schema.org class is equivalent to."""
    found: dict[str, list[str]] = {}
    dropped: Counter[str] = Counter()
    for row in read_json(cache / "p1709.json"):
        target = row["target"]
        if not vocabulary.is_class(target):
            dropped["the P1709 target is an enumeration member or a datatype, not a class"] += 1
            continue
        found.setdefault(target, []).append(row["source"])
    return {k: sorted(v) for k, v in sorted(found.items())}, dropped


def property_map(
    cache: Path, vocabulary: Vocabulary
) -> tuple[dict[str, tuple[str, Kind, tuple[str, ...]]], Counter[str]]:
    """Which schema.org slot each Wikidata property fills, and of what kind."""
    found: dict[str, tuple[str, Kind, tuple[str, ...]]] = {}
    dropped: Counter[str] = Counter()
    for row in read_json(cache / "p1628.json"):
        source, target = row["source"], row["target"]
        if not source.startswith("P"):
            dropped["the P1628 subject is not a Wikidata property"] += 1
            continue
        if target in RESERVED:
            dropped["the schema.org property is a name the shared extractor reads as structure"] += 1
            continue
        if vocabulary.superseded(target):
            dropped["the schema.org property is superseded"] += 1
            continue
        resolved = vocabulary.kind_of(target)
        if resolved is None:
            dropped["the schema.org property declares no range"] += 1
            continue
        kind, choices = resolved
        if kind not in GROUNDABLE:
            dropped[f"the slot is a {kind.value}, which no lead states in a form that resolves"] += 1
            continue
        if not vocabulary.domains_of(target):
            dropped["the schema.org property declares no domain, so no class could offer it"] += 1
            continue
        found.setdefault(source, (target, kind, choices))
    return found, dropped


def usable_slots(
    vocabulary: Vocabulary, properties: dict[str, tuple[str, Kind, tuple[str, ...]]]
) -> tuple[dict[str, dict[str, tuple[Kind, tuple[str, ...]]]], dict[str, set[str]]]:
    """Every mapped property each class can carry, and which ones are its own.

    ``name`` is added to every class and is never its own. schema.org declares
    it on ``Thing``, no ``P1628`` reaches it because a Wikidata label is not a
    statement, and the subject's own name is the one thing an encyclopaedia
    lead always states.
    """
    by_slot: dict[str, tuple[Kind, tuple[str, ...]]] = {
        prop: (kind, choices) for prop, kind, choices in properties.values()
    }
    domains = {prop: vocabulary.domains_of(prop) for prop in by_slot}
    found: dict[str, dict[str, tuple[Kind, tuple[str, ...]]]] = {}
    declared: dict[str, set[str]] = {}
    for name in vocabulary.nodes:
        if not name.startswith(SCHEMA):
            continue
        short = name[len(SCHEMA) :]
        if not vocabulary.is_class(short):
            continue
        line = {short} | vocabulary.ancestors(short)
        found[short] = {prop: by_slot[prop] for prop, domain in domains.items() if domain & line}
        found[short][NAME] = (Kind.TEXT, ())
        declared[short] = {prop for prop, domain in domains.items() if short in domain}
    return found, declared


_PRECISION = {9: 4, 10: 7, 11: 10}
"""How many characters of an ISO date a Wikidata precision justifies.

Year, month, day. Anything coarser is a decade or a century, which schema.org
writes as a date and a reader does not.
"""


def flatten(claim: dict[str, Any], kind: Kind, labels: dict[str, str]) -> Any | None:
    """One statement's value as the slot would hold it, or ``None``."""
    raw = claim.get("value")
    if raw is None:
        return None
    if kind in (Kind.DATE, Kind.DATETIME):
        if claim.get("kind") != "time":
            return None
        width = _PRECISION.get(int(claim.get("precision") or 0))
        return str(raw).lstrip("+")[:width] if width else None
    if kind in (Kind.NUMBER, Kind.INTEGER):
        if claim.get("kind") != "quantity" or str(claim.get("unit") or "1") not in ("1", ""):
            # A magnitude with a unit has nowhere to put the unit, and a
            # number slot filled with centimetres reads as metres.
            return None
        number = float(str(raw).lstrip("+"))
        return int(number) if kind is Kind.INTEGER and number.is_integer() else number
    if claim.get("kind") == "string":
        return str(raw)
    if claim.get("kind") == "item":
        return labels.get(str(raw))
    return None


def candidates_of(
    body: dict[str, Any],
    properties: dict[str, tuple[str, Kind, tuple[str, ...]]],
    slots: dict[str, tuple[Kind, tuple[str, ...]]],
    lexicon: dict[str, Any],
    labels: dict[str, str],
    dropped: Counter[str],
) -> dict[str, list[tuple[Any, Kind, tuple[str, ...], str | None]]]:
    """Every value this item offers a slot its class declares."""
    found: dict[str, list[tuple[Any, Kind, tuple[str, ...], str | None]]] = {}
    for pid, claims in (body.get("claims") or {}).items():
        mapped = properties.get(pid)
        if mapped is None:
            continue
        prop, kind, choices = mapped
        if prop not in slots:
            dropped["the class does not declare the slot the property maps to"] += 1
            continue
        for claim in truthy(claims):
            value = flatten(claim, kind, labels)
            if value is None or value == "":
                dropped["the statement has no value the slot could hold"] += 1
                continue
            if kind is Kind.ENUM and choices and value not in choices:
                dropped["the value is not a member of the enumeration the slot closes"] += 1
                continue
            extra: list[str] = []
            # The QID of the thing the statement points at, kept beside the
            # surface form rather than instead of it. Flattening an item to
            # its label is right for grading a literal and loses the only
            # thing that makes an edge an edge: two documents naming the same
            # Q-number are talking about one entity, which is cross-document
            # identity with no judge and no annotation.
            target: str | None = None
            if claim.get("kind") == "item":
                target = str(claim["value"])
                entry = lexicon.get(target) or {}
                extra = [*(entry.get("aliases") or []), *(entry.get("demonyms") or [])]
            if value not in [v for v, _, _, _ in found.get(prop, [])]:
                found.setdefault(prop, []).append((value, kind, tuple(extra), target))
    return {prop: values[:VALUES_PER_PROPERTY] for prop, values in found.items()}


@dataclass
class Measured:
    """One class's draw, reduced to the entities that became tasks."""

    kept: list[dict[str, Any]]
    candidates: int = 0
    stated: int = 0
    entities: int = 0
    reached: int = 0


def build(  # noqa: C901 - one pass, one stated reason per branch
    cache: Path,
    cap: int | None,
    retrieved_at: str,
    guessable_at: float = GUESSABLE,
) -> tuple[dict[str, Any], dict[str, str]]:
    vocabulary = read_vocabulary(cache)
    classes, mapping_drops = class_map(cache, vocabulary)
    properties, property_drops = property_map(cache, vocabulary)
    mapping_drops.update(property_drops)
    slots_by_class, own_slots = usable_slots(vocabulary, properties)

    draw = read_json(cache / "draw.json")
    states = read_json(cache / "entities.json")
    lexicon = read_json(cache / "lexicon.json")
    labels = {item: (entry.get("label") or "") for item, entry in lexicon.items()}
    articles = read_json(cache / "articles.json")

    per_property: dict[str, Counter[str]] = {}
    measured: dict[str, Measured] = {}
    documents: dict[str, str] = {}
    seen: set[str] = set()
    entities_in = 0
    # Three counters and not one. An entity, a statement and a vocabulary
    # entry are three populations, and the arithmetic that says nothing was
    # lost is only checkable where the reasons it sums over all count the same
    # thing: `excluded` sums with `resolved` to the entities that went in.
    excluded: Counter[str] = Counter()
    statement_drops: Counter[str] = Counter()

    for cls in sorted(draw):
        offered = slots_by_class.get(cls)
        if offered is None or len(offered) < MIN_SLOTS:
            excluded["the class declares too few mapped properties to describe anything"] += len(draw[cls]["items"])
            entities_in += len(draw[cls]["items"])
            continue
        report = measured.setdefault(cls, Measured(kept=[]))
        for item in draw[cls]["items"]:
            entities_in += 1
            qid = item["qid"]
            if qid in seen:
                excluded["the item was already drawn for another class"] += 1
                continue
            seen.add(qid)
            body = states.get(qid)
            if body is None:
                excluded["the item's statements were never fetched"] += 1
                continue
            article = articles.get(qid)
            if article is None or not article.get("revision"):
                excluded["the item has no English Wikipedia article the harvest could read"] += 1
                continue
            lead = " ".join((article.get("extract") or "").split())
            if len(lead) < 120:
                excluded["the article's lead is too short to state three facts"] += 1
                continue

            found = candidates_of(body, properties, offered, lexicon, labels, statement_drops)
            label = body.get("label")
            if label:
                found[NAME] = [(label, Kind.TEXT, (), None)]
            facts: dict[str, list[Any]] = {}
            edges: dict[str, list[str | None]] = {}
            distractors: dict[str, list[Any]] = {}
            for prop, values in found.items():
                counter = per_property.setdefault(prop, Counter())
                for value, kind, extra, target in values:
                    counter["candidates"] += 1
                    forms = written_forms(value, kind, {str(value): list(extra)} if extra else None)
                    written = stated_in(lead, forms) if forms else None
                    if written is None:
                        distractors.setdefault(prop, []).append(value)
                        continue
                    counter["stated"] += 1
                    # A text answer is the form the lead states, not the label
                    # Wikidata files it under. A lead calling a film
                    # British-Austrian states its country twice, and a reader
                    # answers "British": grading that against "United Kingdom"
                    # marks a correct extraction wrong. Measured over the
                    # corpus as built, 228 of 4,401 values are an alias or a
                    # demonym rather than the label.
                    #
                    # Only text. A date is stated as "5 August 1930" and the
                    # answer is 1930-08-05, which is the resolution the task
                    # asks for, and an enumeration member is the member.
                    facts.setdefault(prop, []).append(written if kind is Kind.TEXT else value)
                    # Parallel to facts and in the same order, so a reader
                    # that does not want edges is unaffected by their being
                    # there. None where the value is a literal.
                    edges.setdefault(prop, []).append(target)

            report.entities += 1
            report.candidates += sum(len(v) for v in found.values())
            report.stated += sum(len(v) for v in facts.values())
            if len(facts) < MIN_SLOTS or sum(len(v) for v in facts.values()) < MIN_FACTS:
                excluded["the lead states fewer than three facts over three slots"] += 1
                continue
            report.reached += 1

            # The id and the url are derived at load time rather than stored.
            # Both follow from the QID and the title by a fixed rule, and
            # 1,328 copies of a 50-character url is 65 KiB of a file the hook
            # refuses over 500.
            record = {
                "qid": qid,
                "cls": cls,
                "title": article["title"],
                "revision": int(article["revision"]),
                "sha256": hashlib.sha256(lead.encode("utf-8")).hexdigest(),
                "facts": facts,
                "edges": {p: v for p, v in edges.items() if any(q is not None for q in v)},
                "distractors": distractors,
            }
            report.kept.append(record)
            documents[f"wds-{qid}"] = lead

    guessable = _guessable(measured)
    too_common = {prop for prop, (_, _, share) in guessable.items() if share > guessable_at}
    catalogue = _catalogue(vocabulary, slots_by_class, classes, per_property, measured, own_slots, too_common)
    kept = _trim_to_catalogue(measured, catalogue, excluded)
    kept = _cells_only(kept, excluded)
    # Ordered by the hash of the QID and not by the QID. Wikidata assigns ids
    # in creation order and an editor enters a discography in one sitting, so
    # a QID-sorted prefix of the albums is a prefix of a few artists: Madonna
    # held ten of the first hundred and one in sixty of the class. A cap on
    # that order samples a cluster, and the wrong-document control reads the
    # cluster as a corpus that answers itself. The order is still fixed, so
    # two callers asking for the first 120 still see the same 120.
    kept.sort(key=lambda record: (record["cls"], hashlib.sha256(record["qid"].encode()).hexdigest()))
    selected, passed_over = (kept, 0) if cap is None else _capped(kept, cap)
    documents = {f"wds-{record['qid']}": documents[f"wds-{record['qid']}"] for record in selected}

    per_class = Counter(record["cls"] for record in kept)
    payload = {
        "schema_version": "1",
        "name": "Wikidata-schema.org",
        "built_at": datetime.now(UTC).date().isoformat(),
        "retrieved_at": retrieved_at,
        "sources": {
            "schemaorg": {
                "url": "https://schema.org/version/latest/schemaorg-current-https.jsonld",
                "sha256": hashlib.sha256((cache / "schemaorg.jsonld").read_bytes()).hexdigest(),
                "classes": sum(1 for name in vocabulary.nodes if vocabulary.is_class(name.removeprefix(SCHEMA))),
            },
            "wikidata": {
                "equivalent_class": EQUIVALENT_CLASS,
                "equivalent_property": EQUIVALENT_PROPERTY,
                "mapped_classes": len(classes),
                "mapped_properties": len(properties),
                "endpoint": "https://query.wikidata.org/sparql",
            },
            "wikipedia": {"api": "https://en.wikipedia.org/w/api.php", "extract": "exintro, explaintext"},
        },
        "licence": {
            "facts": {"name": FACT_LICENCE, "url": "https://creativecommons.org/publicdomain/zero/1.0/"},
            "text": {"name": TEXT_LICENCE, "url": "https://creativecommons.org/licenses/by-sa/4.0/"},
            "note": (
                "The truth, the catalogue and the statistics are derived from Wikidata and are CC0 1.0. "
                "The documents are Wikipedia article leads under CC BY-SA 4.0 and are not published here: "
                "each record carries the url, the revision id and the sha256 of the lead it was measured "
                "on, which is what makes the corpus reproducible without redistributing anyone's prose."
            ),
        },
        "draw": {
            "query": CLASS_QUERY,
            "per_class": {
                cls: {"root": entry["root"], "floor": entry["floor"], "drawn": len(entry["items"])}
                for cls, entry in sorted(draw.items())
            },
        },
        "catalogue": catalogue,
        "properties": {
            prop: {
                "kind": _kind_in(catalogue, prop),
                "candidates": counts["candidates"],
                "stated": counts["stated"],
                "rate": round(counts["stated"] / counts["candidates"], 4) if counts["candidates"] else 0.0,
                "commonest": guessable.get(prop, ("", "", 0.0))[1],
                "commonest_share": round(guessable.get(prop, ("", "", 0.0))[2], 4),
                "offered": prop in {slot[0] for entry in catalogue.values() for slot in entry["slots"]},
            }
            for prop, counts in sorted(per_property.items())
        },
        "guessable_at": guessable_at,
        "guessable": {
            prop: {"on": cls, "commonest": value, "share": round(share, 4)}
            for prop, (cls, value, share) in sorted(guessable.items())
            if share > guessable_at
        },
        "grounding": {
            "candidates": sum(r.candidates for r in measured.values()),
            "stated": sum(r.stated for r in measured.values()),
            "rate": round(
                sum(r.stated for r in measured.values()) / max(sum(r.candidates for r in measured.values()), 1), 4
            ),
            # Two rates, because they answer two questions. The one above is
            # over every statement the mapping offered, which is what says how
            # much of a knowledge base a lead states. The one below is over the
            # entities and the slots this file publishes, which is what the
            # tasks in it were built from, and it is the higher of the two
            # because the guessable slots that went are the badly grounded ones.
            "published": {
                "candidates": sum(
                    len(v) for record in selected for part in ("facts", "distractors") for v in record[part].values()
                ),
                "stated": sum(len(v) for record in selected for v in record["facts"].values()),
            },
            "per_class": {
                cls: {
                    "entities": report.entities,
                    "candidates": report.candidates,
                    "stated": report.stated,
                    "rate": round(report.stated / report.candidates, 4) if report.candidates else 0.0,
                    "reached_minimum": report.reached,
                    "share_reaching_minimum": round(report.reached / report.entities, 4) if report.entities else 0.0,
                }
                for cls, report in sorted(measured.items())
            },
        },
        "entities_in": entities_in,
        "resolved": len(kept),
        "excluded": dict(excluded.most_common()),
        "mapping_excluded": dict(mapping_drops.most_common()),
        "statements_excluded": dict(statement_drops.most_common()),
        "per_class": dict(per_class.most_common()),
        "cap": cap,
        "value_share": VALUE_SHARE,
        "passed_over_for_balance": passed_over,
        "min_facts": MIN_FACTS,
        "min_slots": MIN_SLOTS,
        "controls": {},
        "entities": selected,
    }
    return payload, documents


CLASS_FLOOR = 40
STATEMENT_FLOOR = 25
"""How much evidence a guessability measurement needs to be one.

A class with two documents makes every value it holds a half, and a property
with two statements makes one of them a half. Neither is a corpus that can be
guessed; both are a sample too small to say anything. Below either floor the
pair is not measured, which keeps ``illustrator`` and its two statements out
of a rule meant for ``countryOfOrigin`` and its three thousand.
"""


def _guessable(measured: dict[str, Measured]) -> dict[str, tuple[str, str, float]]:
    """The class and value that make each property most guessable, and by how much.

    Per class, because a grid is one class. ``creator`` over the whole corpus
    is as distinctive as a name, and over the paintings alone it is Leonardo
    da Vinci once in four. Taking the worst class is what a cell will meet.

    Over the entities that reached the minimum, which is the population a grid
    is drawn from. Counting over the draw instead would let a property that
    only ever appears on documents nobody keeps decide whether a slot exists.
    """
    found: dict[str, tuple[str, str, float]] = {}
    for cls, report in measured.items():
        if len(report.kept) < CLASS_FLOOR:
            continue
        counts: dict[str, Counter[str]] = {}
        for record in report.kept:
            for prop, values in record["facts"].items():
                counts.setdefault(prop, Counter()).update(dict.fromkeys(map(str, values), 1))
        for prop, counter in counts.items():
            total = sum(counter.values())
            if total < STATEMENT_FLOOR:
                continue
            value, top = counter.most_common(1)[0]
            share = top / total
            if share > found.get(prop, ("", "", 0.0))[2]:
                found[prop] = (cls, value, share)
    return found


def _kind_in(catalogue: dict[str, Any], prop: str) -> str:
    for entry in catalogue.values():
        for slot in entry["slots"]:
            if slot[0] == prop:
                return str(slot[1])
    return Kind.TEXT.value


def _catalogue(
    vocabulary: Vocabulary,
    slots_by_class: dict[str, dict[str, tuple[Kind, tuple[str, ...]]]],
    classes: dict[str, list[str]],
    per_property: dict[str, Counter[str]],
    measured: dict[str, Measured],
    own_slots: dict[str, set[str]],
    guessable: set[str],
) -> dict[str, dict[str, Any]]:
    """Every class the prompt may offer, with the properties it can carry.

    The pool is the mapped classes that declare at least one mapped property
    of their own and carry at least three in all. Both halves do work. A class
    with three inherited properties and none of its own is indistinguishable
    from its parent on anything this corpus can state, so offering it adds a
    name no evidence selects; the measured catalogue is 405 classes with that
    half dropped and 85 with it. A class with two properties could not be what
    any document is, since an entity needs three.

    The classes the corpus answers are in whatever the rule says, because a
    catalogue that omitted one would make its documents unanswerable.

    Each class keeps at most :data:`~oold_llm_bench.corpus.schemaorg.CATALOGUE_SLOTS`
    properties, chosen by how often the corpus found one stated, because that
    number is the same for the catalogue entry, the union branches and the
    answer schema. The kept list is then sorted by name: ordering it by how
    often a slot is filled would put the answer's properties first in every
    entry.

    Every slot is marked as the class's own. schema.org declares ``name`` on
    ``Thing`` and ``genre`` on ``CreativeWork``, and the renderer shows a
    class's own slots in preference to its inherited ones, so an inherited slot
    would be withheld from the entry of the class that needs it. A document
    that names no property cannot be answered from a catalogue that hides half
    of it.

    A :data:`GUESSABLE` property is offered by nobody, including the classes
    this corpus never answers. Dropping it on the classes that were measured
    and leaving it on the rest would be a tell: the absence of
    ``countryOfOrigin`` from ``Movie`` alone says which classes the corpus has
    documents for.
    """
    answered = set(measured)
    found: dict[str, dict[str, Any]] = {}
    for name, declared in sorted(slots_by_class.items()):
        slots = {prop: value for prop, value in declared.items() if prop not in guessable}
        if name not in classes and name not in answered:
            continue
        if name not in answered and (len(slots) < MIN_SLOTS or not (own_slots.get(name, set()) - guessable)):
            continue
        ranked = sorted(slots, key=lambda prop: (-(per_property.get(prop, Counter())["stated"]), prop))
        chosen = sorted(ranked[:CATALOGUE_SLOTS])
        found[name] = {
            "parents": list(vocabulary.parents(name)),
            "label": name,
            "description": vocabulary.description(name),
            "slots": [
                [prop, slots[prop][0].value, True, *([list(slots[prop][1])] if slots[prop][1] else [])]
                for prop in chosen
            ],
        }
    return found


def _trim_to_catalogue(
    measured: dict[str, Measured],
    catalogue: dict[str, dict[str, Any]],
    excluded: Counter[str],
) -> list[dict[str, Any]]:
    """Drop any fact on a slot the class's catalogue entry does not carry.

    The catalogue keeps twelve properties per class and a class may map more,
    so a fact can survive grounding and still have nowhere to be shown. An
    entity that falls under the minimum once those are gone leaves with the
    same stated reason as one that never reached it.
    """
    kept: list[dict[str, Any]] = []
    for cls, report in measured.items():
        entry = catalogue.get(cls)
        if entry is None:
            excluded["the class has no catalogue entry"] += len(report.kept)
            continue
        offered = {slot[0] for slot in entry["slots"]}
        for record in report.kept:
            facts = {k: v for k, v in record["facts"].items() if k in offered}
            if len(facts) < MIN_SLOTS or sum(len(v) for v in facts.values()) < MIN_FACTS:
                excluded["the lead states fewer than three facts over three slots"] += 1
                continue
            record["facts"] = facts
            record["distractors"] = {k: v for k, v in record["distractors"].items() if k in offered}
            kept.append(record)
    return kept


def _cells_only(records: list[dict[str, Any]], excluded: Counter[str]) -> list[dict[str, Any]]:
    """Drop a class too small to be a grid, counting what it held.

    Below :data:`CELL_FLOOR` the negative controls stop measuring the corpus
    and start measuring one document: on three software applications the
    wrong-document control scored 0.095, because two of the three shared a
    creator and that is two thirds of everything there was. Publishing a class
    nobody can interpret a cell of is worse than reporting that it did not
    reach one, which the grounding statistics do.
    """
    sizes = Counter(record["cls"] for record in records)
    small = {cls for cls, size in sizes.items() if size < CELL_FLOOR}
    for cls in small:
        excluded[f"the class reached fewer documents than the {CELL_FLOOR} a grid needs"] += sizes[cls]
    return [record for record in records if record["cls"] not in small]


def _capped(records: list[dict[str, Any]], cap: int) -> tuple[list[dict[str, Any]], int]:
    """The committed sample: at most ``cap`` a class, flat in its values.

    An entity is passed over when one of its values already fills its share of
    the class, which is :data:`VALUE_SHARE` of whichever is smaller, the cap or
    the class. Of the class as well as of the cap, because the share that
    matters is of the grid a cell will be: four documents out of a cap of 120
    is a twentieth of a class that only has 76, and a twentieth is over the
    ceiling.

    The count of entities passed over is returned so the sample can be
    compared against the draw it came from.
    """
    sizes = Counter(record["cls"] for record in records)
    room = {cls: max(2, int(min(cap, size) * VALUE_SHARE)) for cls, size in sizes.items()}
    taken: Counter[str] = Counter()
    held: dict[str, Counter[str]] = {}
    selected: list[dict[str, Any]] = []
    passed = 0
    for record in records:
        cls = record["cls"]
        if taken[cls] >= cap:
            continue
        seen = held.setdefault(cls, Counter())
        values = {f"{prop}={value}" for prop, items in record["facts"].items() for value in map(str, items)}
        if any(seen[value] >= room[cls] for value in values):
            passed += 1
            continue
        taken[cls] += 1
        seen.update(values)
        selected.append(record)
    return selected, passed


def _perfect(task: Any) -> TripleSet:
    triples = frozenset()
    classes = {}
    for instance in task.expected:
        triples |= make_triples(instance.key, instance.fields)
        classes[instance.key] = instance.class_path
    return TripleSet(triples=triples, classes=classes, provenance={})


def _controls(tasks: list[Any], name: str, catalogue_size: int) -> dict[str, float]:
    grid = ExperimentConfig(
        name=name,
        conditions=[Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=catalogue_size)],
        models=[ModelSpec(model="m", provider_profile="openai", model_version="1")],
        tasks=tasks,
        runs_per_cell=1,
    )
    return control_scores(grid, Environment(benchmark_version="0.1.0", benchmark_sha="build"))


def _over_ceiling(scores: dict[str, float]) -> list[str]:
    return [
        f"{name} scored {score:.3f}"
        for name, score in scores.items()
        if score > (SPELLING_CEILING if name == "spelling" else CONTROL_CEILING)
    ]


def verify(path: Path, documents: dict[str, str], catalogue_size: int) -> dict[str, Any]:
    """Refuse to publish a corpus the grader cannot tell from a control.

    A perfect answer has to score 1.00 and the four controls have to stay at
    their floor, measured on the corpus as built.

    Per class as well as over the whole, and the per-class check is the one
    that bites. A cell of this benchmark is one class, and the strongest
    control answers a different document of the same grid correctly: over
    ten classes a value two films share is diluted by nine classes that do not
    have it, and over one class it is not. Checking the whole corpus alone
    reported 0.009 on a grid whose Movie half was at 0.111.
    """
    tasks = load_entities(documents, path=path)
    for task in tasks:
        result = score_task(task, _perfect(task))
        if result.primary < 1.0 or result.dimensions[Dimension.CLASS].f1 < 1.0:
            raise ValueError(
                f"{task.id} scores {result.primary:.3f} on a perfect answer, so the corpus is not gradeable"
            )

    overall = _controls(tasks, "wikidata-schemaorg", catalogue_size)
    by_class: dict[str, list[Any]] = {}
    for task in tasks:
        by_class.setdefault(task.expected[0].class_path, []).append(task)
    per_class = {
        cls: _controls(subset, cls, catalogue_size) for cls, subset in sorted(by_class.items()) if len(subset) > 1
    }

    failed = [f"over the whole corpus, {reason}" for reason in _over_ceiling(overall)]
    failed += [
        f"on a {cls}-only grid, {reason}" for cls, scores in per_class.items() for reason in _over_ceiling(scores)
    ]
    if failed:
        raise ValueError(
            "a negative control cleared its ceiling, so no cell would be interpretable: " + "; ".join(failed)
        )
    return {
        **{name: round(score, 4) for name, score in sorted(overall.items())},
        "per_class": {
            cls: {name: round(score, 4) for name, score in sorted(scores.items())} for cls, scores in per_class.items()
        },
    }


def dump(payload: dict[str, Any]) -> str:
    """The corpus as one line per entity, under an indented header.

    Written this way for two readers. A person reads a diff, and one entity
    per line makes a rebuild show which entities moved instead of reflowing
    the file. And the repository is not a data store: the hook refuses a file
    over 500 KiB and an indented entity is three times the size of a flat one.
    """
    placeholder = "<<entities>>"
    header = json.dumps({**payload, "entities": placeholder}, indent=1, ensure_ascii=False)
    body = ",\n  ".join(json.dumps(e, ensure_ascii=False, separators=(",", ":")) for e in payload["entities"])
    return header.replace(f'"{placeholder}"', "[\n  " + body + "\n ]") + "\n"


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(dump(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/wikidata_schemaorg"))
    parser.add_argument("--out", type=Path, default=Path("src/oold_llm_bench/data/wikidata_schemaorg.json"))
    parser.add_argument("--documents", type=Path, default=Path(".cache/wikidata_schemaorg/documents.json"))
    parser.add_argument("--cap", type=int, default=120, help="entities kept per class, 0 for all")
    parser.add_argument("--catalogue-size", type=int, default=25, help="classes a cell offers when verifying")
    parser.add_argument("--guessable-at", type=float, default=GUESSABLE, help="see GUESSABLE")
    parser.add_argument("--retrieved-at", default=datetime.now(UTC).date().isoformat())
    args = parser.parse_args()

    payload, documents = build(args.cache, None if args.cap == 0 else args.cap, args.retrieved_at, args.guessable_at)
    write(args.out, payload)
    args.documents.parent.mkdir(parents=True, exist_ok=True)
    args.documents.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8", newline="\n")

    payload["controls"] = verify(args.out, documents, args.catalogue_size)
    write(args.out, payload)

    grounding = payload["grounding"]
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")
    print(f"wrote {args.documents} ({args.documents.stat().st_size / 1024:.0f} KiB, not committed)")
    for reason, count in payload["mapping_excluded"].items():
        print(f"  mapping    {count:6}        : {reason}")
    for reason, count in payload["statements_excluded"].items():
        print(f"  statements {count:6}        : {reason}")
    print(f"  entities in                : {payload['entities_in']}")
    for reason, count in payload["excluded"].items():
        print(f"  excluded   {count:6}        : {reason}")
    print(f"  resolved                   : {payload['resolved']}")
    print(f"  written out                : {len(payload['entities'])} (cap {payload['cap']})")
    print(f"  passed over for balance    : {payload['passed_over_for_balance']}")
    print(f"  catalogue                  : {len(payload['catalogue'])} classes")
    print(f"  properties dropped         : {len(payload['guessable'])} guessable, see the record")
    print(f"  grounding, mapped          : {grounding['stated']}/{grounding['candidates']} = {grounding['rate']:.3f}")
    published = grounding["published"]
    rate = published["stated"] / max(published["candidates"], 1)
    print(f"  grounding, published       : {published['stated']}/{published['candidates']} = {rate:.3f}")
    written = Counter(record["cls"] for record in payload["entities"])
    controls = payload["controls"]
    print(f"  {'class':22} {'drawn':>6} {'resolved':>9} {'written':>8} {'grounded':>9} {'reach 3':>8} {'control':>8}")
    for cls, report in sorted(grounding["per_class"].items(), key=lambda kv: -kv[1]["reached_minimum"]):
        control = (controls["per_class"].get(cls) or {}).get("wrong-document")
        print(
            f"  {cls:22} {report['entities']:6} {payload['per_class'].get(cls, 0):9} {written[cls]:8}"
            f" {report['rate']:9.3f} {report['share_reaching_minimum']:8.3f}"
            f" {'-' if control is None else f'{control:8.4f}'}"
        )
    print(f"  controls over the whole    : { ({k: v for k, v in controls.items() if k != 'per_class'}) }")
    print(f"  {'property':24} {'rate':>9} {'stated':>8}")
    for prop, report in sorted(payload["properties"].items(), key=lambda kv: -kv[1]["stated"]):
        print(f"  {prop:24} {report['rate']:9.3f} {report['stated']:8}/{report['candidates']}")


if __name__ == "__main__":
    main()
