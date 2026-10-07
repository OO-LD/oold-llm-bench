"""Tasks read from Wikipedia leads, with Wikidata's facts as the ground truth.

The schema.org corpus this joins was generated from 75 prose frames, and the
frames name what they fill: "Logged as person. ... That is its additional
name." The class and the property slot are both readable off the page, so the
class-selection step the benchmark exists to measure was never exercised.
Measured over 50,000 Wikipedia leads, ordinary English names a schema.org class
or property in 2.9% of documents. This corpus is that measurement turned into
a corpus.

The construction is the one :mod:`oold_llm_bench.corpus.wiki_measurements`
uses, one join further along. There a published dataset supplied the pairing of
a sentence to a fact; here the pairing is made: Wikidata's ``P1709`` takes 471
items to a schema.org class, ``P1628`` takes 178 properties to a schema.org
property, and an item's sitelink takes it to the English Wikipedia article
whose lead is the document.

*What the ground truth is, and what it is not.* A Wikidata statement is a
candidate, not an answer. Only a statement the lead actually states becomes a
field, because an extractor cannot read what the document does not say and
grading it against the knowledge base measures the knowledge base. Every
statement that survives the rank and qualifier rules and is not stated is kept
beside the truth as a distractor, counted, and never scored. The share that is
stated is the grounding rate, and it is in the record per class and per
property because it qualifies every number the corpus produces.

A text answer is then the form the lead writes and not the label Wikidata
files it under, so every one of the 4,449 text values is a string its own
document contains. A lead calling a film British-Austrian states its country
twice and a reader answers "British"; grading that against "United Kingdom"
would mark a correct extraction wrong, and it did for 228 values before the
written form became the answer. A date is the exception and stays canonical:
the lead writes "5 August 1930", the answer is ``1930-08-05``, and resolving
the one to the other is what the task asks.

*Two things Wikidata will get wrong if they are not handled.*

``P1709`` points at the abstract superclass. ``schema:Person`` maps to
``Q215627`` "person", which has no direct instances at all: people are
``P31 wd:Q5``, and ``Q5`` carries no ``P1709``. Every draw therefore goes
through ``wdt:P31/wdt:P279*``, and :data:`CLASS_QUERY` is the only shape used.

A Wikidata statement is time-scoped and schema.org has no slot for that.
Vilnius carries 18 ``P17 country`` statements: Lithuania, the Soviet Union,
Poland, the Reichskommissariat Ostland. Flattened to ``addressCountry`` the
ground truth is simply wrong. So a statement is read at truthy rank, and any
statement carrying one of :data:`TIME_QUALIFIERS` is dropped whatever its rank,
because a value that was true in 1941 is not what the lead states now.

*One thing real text does that generated text cannot.* A slot can be
answerable from the class rather than from the document: 71% of the films
drawn here are from the United States, and an answer copied from any other
film is right about ``countryOfOrigin`` seven times in ten. The
wrong-document control measures it and nothing else does, and it took a
Movie-only grid to 0.111 against a ceiling of 0.01. Twelve properties are
therefore not offered by any class, and the committed sample caps how often
one value may recur. Both rules are the build's, both are measured, and both
are in the record: see :attr:`GroundedCorpus.guessable`.

*Licence posture, and why the documents are not here.* Wikidata is CC0 1.0 and
the extracted truth is derived from it. The Wikipedia lead is CC BY-SA 4.0.
This file publishes the url, the revision id, the sha256 of the lead and the
facts, and no page text: :func:`load_entities` is handed the documents by its
caller and checks each against the recorded hash. A revision id is immutable,
so :func:`document_request` rebuilds the exact bytes the corpus was measured
on, which an archived copy of the text would not guarantee.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from oold_llm_bench.corpus.schemaorg import (
    CATALOGUE_SLOTS,
    Kind,
    SchemaClass,
    Slot,
    answer_schema,
    branches_for,
    render_class,
)
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
    "ARTICLE_BASE",
    "CLASS_QUERY",
    "CORPUS_PATH",
    "EQUIVALENT_CLASS",
    "EQUIVALENT_PROPERTY",
    "FACT_LICENCE",
    "MIN_FACTS",
    "MIN_SLOTS",
    "NAME",
    "TEXT_LICENCE",
    "TIME_QUALIFIERS",
    "GroundedCorpus",
    "GroundedEntity",
    "classes_of",
    "document_request",
    "load_entities",
    "read_documents",
    "read_grounded_corpus",
    "spelled_date",
    "stated_in",
    "truthy",
    "written_forms",
]

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "wikidata_schemaorg.json"

ARTICLE_BASE = "https://en.wikipedia.org/wiki/"
"""Where a record's title becomes the url the attribution points at.

Stored once rather than per record. The file holds a thousand entities and
nothing about the url varies but the title, so keeping both would spend a
tenth of the large-file budget on a string a rule produces.
"""

TEXT_LICENCE = "CC BY-SA 4.0"
"""What the leads are under.

It reaches the sha256 and nothing else here, because no text is published.
A caller that supplies the documents is under it for the documents, and the
record carries the url whose history is the author list.
"""

FACT_LICENCE = "CC0 1.0"
"""What the truth is under. Wikidata waives everything, so the extracted facts,
the catalogue and the grounding statistics carry no condition."""

EQUIVALENT_CLASS = "P1709"
EQUIVALENT_PROPERTY = "P1628"
"""The two Wikidata properties the whole mapping rests on.

Neither is ours and neither is curated by us, which is the point: a hand-built
map between two vocabularies is an opinion, and this one is published, dated
and auditable through the item history.
"""

TIME_QUALIFIERS = ("P580", "P582", "P585")
"""Start time, end time, point in time.

A statement carrying any of these is scoped to a period schema.org cannot
express, so flattening it asserts of today what was true of 1941. Vilnius is
the case that makes it concrete, and it is not rare: a company's headquarters,
a person's employer and a territory's country are all written this way.
"""

NAME = "name"
"""The one field not taken from a mapped statement.

``schema:name`` has no ``P1628`` source, because a Wikidata label is not a
statement. The English label is used instead, which is the same string the
article is titled with, so it is stated in the lead by construction. Declared
here rather than left implicit: it is a free field in every task, and a reader
comparing this corpus against another has to know that one of the three
required facts is the subject's own name.
"""

MIN_FACTS = 3
MIN_SLOTS = 3
"""What an entity needs before it becomes a task.

Three stated values across three distinct properties. One value is a lookup,
two is a pair, and neither asks the model to decide which of a class's
properties a sentence fills. Measured against the draw, this is also the filter
that removes the entity whose article is a stub: 33% to 100% of a class's
entities reach it depending on the class, and the share is in the record.
"""

CLASS_QUERY = """SELECT DISTINCT ?item ?article WHERE {
  ?item wdt:P31/wdt:P279* wd:%(root)s ;
        wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= %(floor)d)
  FILTER NOT EXISTS { ?item wdt:P279 ?parent }
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
} LIMIT %(limit)d"""
"""The only shape an entity draw takes, recorded so the draw can be repeated.

``wdt:P31/wdt:P279*`` is mandatory and is the first trap: ``wd:Q5`` is four
steps below ``wd:Q215627`` and a draw on the mapped item alone returns nothing.

The sitelink floor is a declared bias, not a cleanup. Grounding is the binding
constraint on this corpus and an article nobody wrote states nothing, so the
draw is pushed towards subjects several languages cover. ``FILTER NOT EXISTS``
drops an item that is itself a class: the closure of ``Q215627`` contains
"Black men" and "Portuguese in Croatia", which are classes of people and not
people.

Ordering is left to the service. Adding ``ORDER BY`` makes the service
enumerate the whole closure before answering, which times out on the three
largest roots, and the draw is reproducible from the recorded QIDs rather than
from the order they arrived in.
"""


def truthy(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One property's statements, read the way a schema.org slot can hold them.

    Preferred rank where the property has one, normal otherwise, deprecated
    never, which is what ``wdt:`` means. Then every statement scoped to a
    period is dropped, because schema.org has no slot for when a value held
    and keeping one asserts of today what was true of 1941.

    It lives here rather than in the build because it is the rule the ground
    truth rests on, and a rule that cannot be tested without an hour of
    network is a rule nobody checks. The claim shape is the harvest's:
    ``rank`` as Wikidata reports it and ``qualifiers`` as the property ids a
    statement carries.
    """
    live = [claim for claim in claims if claim.get("rank") != "deprecated"]
    preferred = [claim for claim in live if claim.get("rank") == "preferred"]
    return [claim for claim in (preferred or live) if not set(claim.get("qualifiers") or ()) & set(TIME_QUALIFIERS)]


@dataclass(frozen=True)
class GroundedEntity:
    """One subject, its article, and what that article was found to state.

    Named for what separates it from the two corpora beside it, which each
    call their unit something shorter. Here the unit is a subject whose truth
    has been cut down to what one document states, and nothing else about it
    matters: the same entity under a different lead is a different record.
    """

    qid: str
    cls: str
    title: str
    revision: int
    sha256: str
    """Of the lead as the build read it, which is what a document is checked
    against before it is handed to a model."""
    facts: dict[str, list[Any]]
    """Property to the stated values, which is the whole of the ground truth.

    A text value is the form the lead writes. A date is the ISO date the lead
    writes some spelling of, because resolving the spelling is the task."""
    distractors: dict[str, list[Any]]
    """Statements that survived the rank and qualifier rules and that the lead
    does not state. Carried so the grounding rate can be recomputed from the
    file, and never scored: a value the document does not contain has no
    correct answer an extractor could give."""

    @property
    def id(self) -> str:
        """The record's identifier, derived rather than stored.

        It follows from the QID by a fixed rule, and a thousand copies of a
        string nothing decides is a tenth of the budget the large-file hook
        allows the whole file.
        """
        return f"wds-{self.qid}"

    @property
    def url(self) -> str:
        """The article, derived from its title the way MediaWiki derives it."""
        return ARTICLE_BASE + self.title.replace(" ", "_")

    @property
    def candidates(self) -> int:
        return sum(len(v) for v in self.facts.values()) + sum(len(v) for v in self.distractors.values())

    @property
    def stated(self) -> int:
        return sum(len(v) for v in self.facts.values())


@dataclass(frozen=True)
class GroundedCorpus:
    """The built corpus, its catalogue, and the arithmetic that accounts for it."""

    entities: tuple[GroundedEntity, ...]
    catalogue: dict[str, dict[str, Any]]
    """Every class the prompt may offer, with the properties it declares.

    Carried in the file rather than read from the generated schema.org module,
    because that module is an artefact held outside this repository and a
    corpus that cannot be loaded without it is a corpus nobody can run. The
    source is schema.org's own vocabulary, pinned in :attr:`sources`."""
    properties: dict[str, dict[str, Any]]
    grounding: dict[str, Any]
    excluded: dict[str, int]
    entities_in: int
    resolved: int
    per_class: dict[str, int]
    cap: int | None
    sources: dict[str, Any]
    licence: dict[str, Any]
    draw: dict[str, Any]
    controls: dict[str, float]
    """What each negative control scored when the build verified itself.

    Recorded rather than asserted in a comment. The build refuses to write a
    corpus whose controls clear their ceiling, so a file that exists is a file
    that passed, and this is the number it passed with."""
    controls_per_class: dict[str, dict[str, float]]
    """The same four, on a grid of one class.

    The check that bites. A cell of this benchmark is one class, and the
    wrong-document control answers a different document of the same grid: a
    value two films share is diluted by twelve other classes over the whole
    corpus and by nothing over one."""
    guessable: dict[str, dict[str, Any]]
    """Properties no class offers, and the value that cost them the slot.

    Kept in the record rather than dropped silently. A slot removed because
    71% of a class states the same value is a finding about the corpus, and a
    reader comparing a grounding rate here against one elsewhere has to know
    which properties are not in the denominator."""
    built_at: str
    retrieved_at_date: str

    @property
    def retrieved_at(self) -> datetime:
        return datetime.combine(date.fromisoformat(self.retrieved_at_date), datetime.min.time(), tzinfo=UTC)

    @property
    def dropped_by_cap(self) -> int:
        return self.resolved - len(self.entities)


def read_grounded_corpus(path: Path | None = None) -> GroundedCorpus:
    """Read the built corpus without needing a single document.

    Separate from :func:`load_entities` for the reason
    :func:`~oold_llm_bench.corpus.wiki_measurements.read_corpus` is separate:
    what went in and what came out have to be comparable without the rest of
    the corpus in hand, and here they have to be comparable without the
    Wikipedia text, which is not published with the file.

    The arithmetic is checked rather than reported. A corpus that has quietly
    lost half its draw still loads, and the grounding rate it reports would
    then describe a population nobody can name.
    """
    payload = json.loads((path or CORPUS_PATH).read_text(encoding="utf-8"))
    corpus = GroundedCorpus(
        entities=tuple(
            GroundedEntity(
                qid=entry["qid"],
                cls=entry["cls"],
                title=entry["title"],
                revision=int(entry["revision"]),
                sha256=entry["sha256"],
                facts={k: list(v) for k, v in entry["facts"].items()},
                distractors={k: list(v) for k, v in (entry.get("distractors") or {}).items()},
            )
            for entry in payload["entities"]
        ),
        catalogue={k: dict(v) for k, v in payload["catalogue"].items()},
        properties={k: dict(v) for k, v in payload["properties"].items()},
        grounding=dict(payload["grounding"]),
        excluded=dict(payload["excluded"]),
        entities_in=int(payload["entities_in"]),
        resolved=int(payload["resolved"]),
        per_class=dict(payload["per_class"]),
        cap=payload.get("cap"),
        sources=dict(payload["sources"]),
        licence=dict(payload["licence"]),
        draw=dict(payload["draw"]),
        controls={k: float(v) for k, v in (payload.get("controls") or {}).items() if k != "per_class"},
        controls_per_class={
            cls: {k: float(v) for k, v in scores.items()}
            for cls, scores in ((payload.get("controls") or {}).get("per_class") or {}).items()
        },
        guessable={k: dict(v) for k, v in (payload.get("guessable") or {}).items()},
        built_at=payload["built_at"],
        retrieved_at_date=payload["retrieved_at"],
    )
    accounted = corpus.resolved + sum(corpus.excluded.values())
    if accounted != corpus.entities_in:
        raise ValueError(
            f"{corpus.entities_in} entities went in and {accounted} are accounted for, "
            f"so {corpus.entities_in - accounted} left without a declared reason"
        )
    if sum(corpus.per_class.values()) != corpus.resolved:
        raise ValueError(
            f"{corpus.resolved} entities resolved but the per-class counts sum to {sum(corpus.per_class.values())}"
        )
    if len(corpus.entities) > corpus.resolved:
        raise ValueError(f"the file holds {len(corpus.entities)} entities but only {corpus.resolved} resolved")
    missing = sorted({entity.cls for entity in corpus.entities} - set(corpus.catalogue))
    if missing:
        raise ValueError(f"the catalogue does not offer classes the corpus answers: {missing}")
    return corpus


_KIND_BY_NAME = {kind.value: kind for kind in Kind}


def classes_of(corpus: GroundedCorpus) -> list[SchemaClass]:
    """The catalogue as the classes the prompt material is built from.

    Rebuilt into :class:`~oold_llm_bench.corpus.schemaorg.SchemaClass` so the
    catalogue entry, the union branches and the answer schema come out of the
    same three functions the generated corpus uses. A real document and a
    generated one are then offered the same surface, and a difference between
    them is a difference in the text and not in how the task was dressed.
    """
    return [
        SchemaClass(
            name=name,
            parents=tuple(entry.get("parents") or ()),
            label=entry.get("label") or name,
            description=entry.get("description") or "",
            slots=tuple(
                Slot(
                    name=slot[0],
                    kind=_KIND_BY_NAME.get(slot[1], Kind.TEXT),
                    choices=tuple(slot[3]) if len(slot) > 3 else (),
                    inherited=not slot[2],
                )
                for slot in entry.get("slots") or ()
            ),
        )
        for name, entry in sorted(corpus.catalogue.items())
    ]


def read_documents(path: Path) -> dict[str, str]:
    """The lead of every article, keyed by the record id that cites it.

    Written by the build beside its cache and never committed, because the
    leads are CC BY-SA 4.0 and this repository is Apache-2.0. A third party
    rebuilds it with :func:`document_request`, which asks for the exact
    revisions the corpus was measured on.
    """
    return {k: str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def document_request(entities: tuple[GroundedEntity, ...] | list[GroundedEntity], batch: int = 20) -> list[str]:
    """The Wikipedia API calls that fetch the documents this corpus cites.

    By revision id and not by title. A page moves and a lead is rewritten; the
    revision the sha256 was taken over does not, so this is what makes the
    corpus reproducible without republishing anyone's prose.
    """
    revisions = [str(entity.revision) for entity in entities]
    return [
        "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1"
        "&format=json&formatversion=2&revids=" + "|".join(revisions[start : start + batch])
        for start in range(0, len(revisions), batch)
    ]


_WHITESPACE = re.compile(r"\s+")
_DASHES = dict.fromkeys(range(0x2010, 0x2016), "-") | {0x2212: "-"}


def normalise(text: str) -> str:
    """Fold a document or a value to the form the matcher compares on.

    Compatibility normalisation, every dash onto the hyphen, whitespace
    collapsed, case folded. No punctuation stripping: "St. Louis" and "St
    Louis" are a real difference in a name and folding them would count a match
    the document did not make.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_DASHES)
    return _WHITESPACE.sub(" ", folded).strip().casefold()


_MONTHS = (
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

_ISO_DATE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")


def spelled_date(value: str) -> tuple[str, ...]:
    """Every way a lead writes one ISO date, including the bare year.

    A lead writes "5 August 1930", "August 5, 1930" or "in 1930", and the
    answer stays ``1930-08-05``. Without the spellings the date properties
    would ground at nearly zero, and they are the properties this corpus has
    most of.

    The bare year counts. A lead that says only the year has stated the date to
    the precision it chose to state it at, and refusing that would discard the
    commonest form a release year reaches the page in.
    """
    found = _ISO_DATE.match(value)
    if not found:
        return ()
    year, month, day = found.group(1), found.group(2), found.group(3)
    forms = [value, year]
    if month:
        name = _MONTHS[int(month) - 1]
        forms.append(f"{name} {year}")
        if day:
            number = str(int(day))
            forms += [f"{number} {name} {year}", f"{name} {number}, {year}", f"{name} {number} {year}"]
    return tuple(dict.fromkeys(forms))


def written_forms(value: Any, kind: Kind, lexicon: Mapping[str, list[str]] | None = None) -> tuple[str, ...]:
    """The surface forms a lead may state one value in.

    ``lexicon`` carries the aliases and the demonyms of an item-valued target,
    both taken from Wikidata rather than written here. It is what closes the
    measured gap on country and nationality: a lead calling a film a
    "British-Austrian drama" states ``countryOfOrigin`` twice and matches
    neither label, and ``P1549`` publishes "British" and "Austrian" as the
    demonyms of exactly those two items.
    """
    if isinstance(value, bool):
        return ()
    if kind in (Kind.DATE, Kind.DATETIME):
        return spelled_date(str(value))
    if isinstance(value, (int, float)):
        plain = f"{value:g}"
        grouped = f"{int(value):,}" if float(value).is_integer() else plain
        return tuple(dict.fromkeys((plain, grouped)))
    text = str(value)
    forms = [text, *(lexicon or {}).get(text, [])]
    return tuple(dict.fromkeys(form for form in forms if len(form) >= 3))


_WORD_EDGE = re.compile(r"[0-9a-z]", re.IGNORECASE)


def stated_in(document: str, forms: tuple[str, ...]) -> str | None:
    """The first form the document states, on word boundaries, or ``None``.

    The form is returned and not a boolean, because for a text slot it is the
    answer: the build files the form the lead writes rather than the label
    Wikidata keeps, so a value the document states as "British" is not graded
    against "United Kingdom".

    Word boundaries, because a bare substring test makes "Al" a match inside
    "Albania" and every short given name would ground. The boundary is built
    here instead of with ``\\b`` so that a form ending in a bracket or a dot
    still anchors: ``\\b`` after ")" never matches, and "Chinatown (1974 film)"
    is a title the lead states.
    """
    haystack = normalise(document)
    for form in forms:
        needle = normalise(form)
        if not needle:
            continue
        start = haystack.find(needle)
        while start >= 0:
            before = haystack[start - 1] if start else ""
            after = haystack[start + len(needle) : start + len(needle) + 1]
            if not _WORD_EDGE.match(before or " ") and not _WORD_EDGE.match(after or " "):
                return form
            start = haystack.find(needle, start + 1)
    return None


def _mentions_of(entity: GroundedEntity, document: str) -> tuple[str, ...]:
    """The words this document refers to the subject by.

    The article title and whatever ``name`` the subject was filed under, kept
    only where the document states them. Real text needs no model to supply
    this: the corpus already grounds every fact by checking the lead states
    some spelling of it, and the subject's own name is grounded the same way.

    A parenthetical qualifier is dropped from the title. "Mercury (planet)"
    disambiguates an encyclopaedia, and no lead refers to the subject that
    way.
    """
    offered = [entity.title.split(" (")[0].strip(), entity.title]
    offered += [str(value) for value in entity.facts.get(NAME, [])]
    seen: list[str] = []
    for name in offered:
        if name and name not in seen and stated_in(document, (name,)) is not None:
            seen.append(name)
    return tuple(seen)


def _check(entity: GroundedEntity, catalogue: Mapping[str, Any], documents: Mapping[str, str]) -> str | None:
    """What stopped one entity becoming a task, or ``None``."""
    offered = catalogue.get(entity.cls)
    if offered is None:
        return f"{entity.id} names the class {entity.cls!r}, which the catalogue does not offer"
    slots = {slot[0] for slot in offered.get("slots") or ()}
    unknown = sorted(set(entity.facts) - slots)
    if unknown:
        return f"{entity.id} fills {unknown} on {entity.cls}, which the class does not declare"
    if not entity.facts:
        return f"{entity.id} states nothing, so it has no answer"
    if len(entity.facts) < MIN_SLOTS or entity.stated < MIN_FACTS:
        return f"{entity.id} states {entity.stated} values over {len(entity.facts)} slots, under the minimum"
    document = documents.get(entity.id)
    if document is None:
        return f"{entity.id} has no document, and the corpus publishes none"
    if hashlib.sha256(document.encode("utf-8")).hexdigest() != entity.sha256:
        return f"{entity.id} was measured on revision {entity.revision} and the document supplied is not it"
    return None


def load_entities(
    documents: Mapping[str, str],
    *,
    path: Path | None = None,
    catalogue: tuple[str, ...] | None = None,
    describe_catalogue: bool = True,
    split: Split = Split.DEV,
    difficulty: Difficulty = Difficulty.HARD,
    limit: int | None = None,
) -> list[TaskRecord]:
    """The corpus as tasks, refusing to load rather than dropping one.

    ``documents`` maps a record id to the lead of the revision the truth was
    measured on. Every one is checked against the recorded sha256, and a
    mismatch stops the load: the grounding decision was taken against those
    exact bytes, and a lead that has since been rewritten would carry a truth
    nobody measured.

    ``describe_catalogue`` is on, unlike the generated corpus. A real document
    names neither its class nor its properties, so a catalogue of bare
    identifiers would leave the model nothing to choose on but the spelling of
    a class name, which is not the question.

    ``limit`` takes the first ``n`` per class in file order, which the build
    fixed by class and QID. A prefix and not a sample, so two callers asking
    for 120 tasks look at the same 120.

    ``difficulty`` is :attr:`~oold_llm_bench.tasks.models.Difficulty.HARD`
    because that is what the register is: an encyclopaedia lead written for a
    reader, with the facts spread through it in whatever order the sentence
    wanted.
    """
    corpus = read_grounded_corpus(path)
    problems = [p for p in (_check(e, corpus.catalogue, documents) for e in corpus.entities) if p]
    if problems:
        raise ValueError(
            "the Wikidata-schema.org corpus does not resolve against its own catalogue and documents: "
            + "; ".join(problems[:20])
        )

    classes = classes_of(corpus)
    offered = tuple(catalogue) if catalogue else tuple(cls.name for cls in classes)
    by_name = {cls.name: cls for cls in classes}
    present = [by_name[name] for name in offered if name in by_name]
    shape = answer_schema(present, Variant.NATIVE)
    narrowed = branches_for(present, Variant.NATIVE)
    lineage = {cls.name: [p for p in cls.parents if p in set(offered)] for cls in present}
    described = (
        {cls.name: render_class(cls, Variant.NATIVE, CATALOGUE_SLOTS) for cls in present}
        if describe_catalogue
        else None
    )
    # schema.org's own comment per property, where the corpus carries one.
    # Shown to the step that chooses between property names, which is a
    # vocabulary question that no wording of the question answers.
    described_properties = {
        name: body["description"]
        for name, body in (corpus.properties or {}).items()
        if isinstance(body, dict) and body.get("description")
    } or None

    chosen: list[GroundedEntity] = []
    taken: dict[str, int] = {}
    for entity in corpus.entities:
        if entity.cls not in by_name:
            continue
        if limit is not None and taken.get(entity.cls, 0) >= limit:
            continue
        taken[entity.cls] = taken.get(entity.cls, 0) + 1
        chosen.append(entity)

    rate = corpus.grounding.get("per_class") or {}
    return [
        TaskRecord(
            id=entity.id,
            document=documents[entity.id],
            expected=[
                ExpectedInstance(
                    key="e1",
                    class_path=entity.cls,
                    fields={
                        prop: values[0] if len(values) == 1 else list(values) for prop, values in entity.facts.items()
                    },
                    mentions=_mentions_of(entity, documents[entity.id]),
                )
            ],
            corpus=CorpusRef(
                name="wikidata-schemaorg",
                source=Source.BULK,
                document_id=entity.qid,
                content_hash=entity.sha256,
                url=entity.url,
                retrieved_at=corpus.retrieved_at,
                licence=TEXT_LICENCE,
            ),
            split=split,
            difficulty=difficulty,
            notes=(
                f"corpus=wikidata-schemaorg,class={entity.cls},qid={entity.qid},"
                f"revision={entity.revision},stated={entity.stated}/{entity.candidates},"
                f"grounded={(rate.get(entity.cls) or {}).get('rate', 0.0):.3f},licence={TEXT_LICENCE}"
            ),
            catalogue=list(offered),
            catalogue_text=described,
            property_text=described_properties,
            answer_schema=shape,
            branches=narrowed,
            class_parents=lineage,
        )
        for entity in chosen
    ]
