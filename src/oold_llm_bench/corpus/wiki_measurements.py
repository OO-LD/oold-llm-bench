"""Tasks read from Wiki-Measurements, human-written text we may republish.

Every number this benchmark has produced was measured on prose the benchmark
wrote itself, and the twenty ELN notes are a register check rather than a
corpus. This is the first source that is real text, carries structured ground
truth, and comes under a licence that lets the corpus itself be published
instead of a list of urls and hashes.

The licence pair is why it is first and it is not free. The sentences are
Wikipedia's under CC BY-SA 4.0, the facts beside them are Wikidata's under
CC0 1.0, and this repository is Apache-2.0. So the data file states its own
terms, every record carries the article it was taken from, and anything
derived from the sentences stays CC BY-SA 4.0. Attribution is per record and
not per corpus, because a task set trimmed to 120 documents has to carry the
attribution for those 120.

*What the ground truth is, and what it is not.* The dataset pairs each
sentence with the Wikidata fact it was matched against, and the two disagree
often: over the strict set the fact's value equals the value written in the
sentence in 43.4% of examples, because Wikipedia writes 255,541 sq mi where
Wikidata stores square kilometres. The fact is therefore not the answer. The
annotated spans are: the magnitude is the value span, the unit is the unit
span, and both are text a person wrote. The fact is used for exactly two
things, the quantity kind its property denotes and a second opinion on the
unit: an example is kept only where the unit the sentence writes and the unit
the fact carries resolve to the same QUDT unit.

*What the model is up against.* The document is one sentence of an article,
so the unit reaches the page as ``km2`` or ``square kilometres`` while the
answer stays ``kilo_meter_squared``, and the magnitude reaches it as
``214,000``. That is the same task
:attr:`~oold_llm_bench.corpus.quantities.Notation.WRITTEN` generates, here
without anyone having chosen the surface form. What is new is that the
sentence names its own property: "in area", "at an elevation of". Class
selection is therefore closer to a lookup than to an inference on this
corpus, which is a fact about Wikipedia and is reported rather than corrected.

*What is left out.* Every exclusion is a declared reason with a count, and
:func:`read_corpus` refuses a file whose counts do not add up to the examples
that went in. The precedent is :mod:`oold_llm_bench.corpus.eln`, where twenty
notes quietly becoming twelve tasks is the failure being guarded against; the
same failure here is 26,009 examples quietly becoming a number nobody can
place. 5,576 survive, and four kinds clear the 120 a cell needs: Area 3,090,
Length 1,494, Altitude 488, Diameter 186. The other eight are carried and
counted rather than dropped, so a report can say which cells it had.

The largest single exclusion after the unmapped properties is a sentence that
states a second reading the annotation does not cover, 1,772 of them. The
answer shape asks for every measurement in the document, so "the Anadyr is
1,150 kilometres long and has a basin of 74,000 sq mi" has no answer that
scores: one of the two readings is unannotated and would come off precision.
The detector needs a space between the magnitude and the unit, which is what
the style guide asks for and what these sentences almost always do, so a
reading written "1750m" is the one it can miss.

Trimming the catalogue and ordering it stay where they already are, in
:mod:`oold_llm_bench.runner.adapter`, so one of these documents is offered
exactly what a synthetic task under the same condition is offered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from oold_llm_bench.corpus.catalogue import CatalogueEntry, render_catalogue
from oold_llm_bench.corpus.quantities import QuantityKind
from oold_llm_bench.grading.triples import Quantity
from oold_llm_bench.tasks.models import (
    CorpusRef,
    Difficulty,
    ExpectedInstance,
    Source,
    Split,
    TaskRecord,
)

__all__ = [
    "CORPUS_PATH",
    "DATASET_DOI",
    "MISLABELLED_UNITS",
    "PROPERTY_KIND",
    "TEXT_LICENCE",
    "Corpus",
    "Example",
    "load_examples",
    "read_corpus",
]

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "wiki_measurements.json"

DATASET_DOI = "10.5281/zenodo.14858280"
"""Wiki-Quantities and Wiki-Measurements, FZJ ICE-2, 2025.

Pinned by DOI rather than by download url. The ids the dataset assigns are
stated to be reassigned between versions, so the build records the archive's
sha256 beside this and a record id means nothing without it.
"""

TEXT_LICENCE = "CC BY-SA 4.0"
"""What the sentences are under, and therefore what this corpus is under.

Share-alike reaches anything derived from the text, which includes the task
records. It does not reach the Wikidata facts, which are CC0 1.0, and the
build records both.
"""

PROPERTY_KIND: dict[str, str] = {
    "P2046": "Area",
    "P2053": "Area",
    "P2043": "Length",
    "P2044": "Altitude",
    "P2386": "Diameter",
    "P2048": "Height",
    "P2049": "Width",
    "P2261": "Width",
    "P2120": "Radius",
    "P4511": "Depth",
    "P2067": "Mass",
    "P2047": "Time",
    "P2146": "Time",
    "P2109": "Power",
    "P2234": "Volume",
    "P2052": "Speed",
}
"""The quantity kind each Wikidata property denotes, where the corpus has one.

The one piece of this corpus that is authored here, and it is authored because
nothing published supplies it. A Wikidata property carries no QUDT quantity
kind, and the alternative is the property span in the sentence, which is a
word Wikipedia chose: taking the class from "in area" would make class
selection a substring match on the document by construction rather than by
accident.

Kept to properties whose kind is not arguable. ``P2044`` is elevation above
sea level and lands on ``Altitude``, which QUDT defines as height above sea
level; ``P2660``, topographic prominence, is a height difference and is left
out rather than guessed at. A property absent here is an exclusion with a
count, never a silent drop, and the build reports how many examples each
absent property held.
"""

MISLABELLED_UNITS: dict[str, tuple[str, str]] = {
    "Area": ("year", "deca_year"),
}
"""Enumeration members whose name contradicts the QUDT unit they denote.

``Area.schema.json`` names ``qunit:ARE`` as ``year`` and ``qunit:DecaARE`` as
``deca_year``. The are is 100 square metres; ``year`` is what the same corpus
calls ``qunit:YR`` under ``Time``, and pint parses it as time. Area is the
largest cell this corpus has, so the defect is reached rather than avoided.

The schemas are not ours, so nothing is patched: the two names stay in the
enumeration, stay in the catalogue a task offers, and are refused as an
answer. An example whose unit resolves to one of them is excluded with that
reason and counted, and the build re-checks the upstream ``@context`` so the
declaration cannot outlive the defect.

Same class of defect, outside the kinds this corpus reaches:
``MassFlowRate`` names ``qunit:KiloTONNE-PER-YR`` as ``knot_per_year``.
"""


@dataclass(frozen=True)
class Example:
    """One sentence and the reading its annotation marks in it."""

    id: str
    text: str
    """The sentence with the annotation marks removed, as the model reads it."""
    kind: str
    unit: str
    magnitude: float
    value_text: str
    """The magnitude as the sentence writes it, ``214,000`` for 214000.0."""
    unit_text: str
    """The unit as the sentence writes it, ``km2`` for ``kilo_meter_squared``."""
    article: str
    """The Wikipedia article the sentence is from. This is the attribution."""
    qudt_unit: str
    wikidata: dict[str, str]
    """The entity, property and unit of the fact, as bare Wikidata ids.

    The fact is not the answer, and it is carried so a reader can check that.
    :attr:`Corpus.wikidata_base` is the namespace the three ids sit in.
    """


@dataclass(frozen=True)
class Corpus:
    """The built corpus and the arithmetic that says nothing was lost."""

    examples: tuple[Example, ...]
    excluded: dict[str, int]
    """How many examples each declared reason removed."""
    examples_in: int
    resolved: int
    """How many survived every reason, before the per-kind cap."""
    per_kind: dict[str, int]
    """How many survived per kind, before the cap."""
    cap: int | None
    dataset: dict[str, Any]
    licence: dict[str, Any]
    built_at: str
    wikidata_base: str = "http://www.wikidata.org/entity/"

    @property
    def retrieved_at(self) -> datetime:
        """When the archive this was built from was downloaded."""
        stamp = str(self.dataset.get("retrieved_at") or self.built_at)
        return datetime.combine(date.fromisoformat(stamp), datetime.min.time(), tzinfo=UTC)

    @property
    def dropped_by_cap(self) -> int:
        return self.resolved - len(self.examples)


def read_corpus(path: Path | None = None) -> Corpus:
    """Read the built corpus without consulting the quantity schemas.

    Separate from :func:`load_examples` for the reason
    :func:`~oold_llm_bench.corpus.eln.read_notes` is: the count that went in
    and the count that came out have to be comparable without a corpus in
    hand. The arithmetic is checked here rather than reported, because a
    corpus that has silently lost a third of itself still loads.
    """
    payload = json.loads((path or CORPUS_PATH).read_text(encoding="utf-8"))
    corpus = Corpus(
        examples=tuple(
            Example(
                id=entry["id"],
                text=entry["text"],
                kind=entry["kind"],
                unit=entry["unit"],
                magnitude=float(entry["magnitude"]),
                value_text=entry["value_text"],
                unit_text=entry["unit_text"],
                article=entry["article"],
                qudt_unit=entry["qudt_unit"],
                wikidata=dict(entry["wikidata"]),
            )
            for entry in payload["examples"]
        ),
        excluded=dict(payload["excluded"]),
        examples_in=int(payload["examples_in"]),
        resolved=int(payload["resolved"]),
        per_kind=dict(payload["per_kind"]),
        cap=payload.get("cap"),
        dataset=dict(payload["dataset"]),
        licence=dict(payload["licence"]),
        built_at=payload["built_at"],
        wikidata_base=payload.get("wikidata_base") or "http://www.wikidata.org/entity/",
    )
    accounted = corpus.resolved + sum(corpus.excluded.values())
    if accounted != corpus.examples_in:
        raise ValueError(
            f"{corpus.examples_in} examples went in and {accounted} are accounted for, "
            f"so {corpus.examples_in - accounted} left without a declared reason"
        )
    if sum(corpus.per_kind.values()) != corpus.resolved:
        raise ValueError(
            f"{corpus.resolved} examples resolved but the per-kind counts sum to {sum(corpus.per_kind.values())}"
        )
    if len(corpus.examples) > corpus.resolved:
        raise ValueError(f"the file holds {len(corpus.examples)} examples but only {corpus.resolved} resolved")
    return corpus


def _check(example: Example, by_name: dict[str, QuantityKind], offered: frozenset[str]) -> str | None:
    """What stopped one example becoming a task, or ``None``."""
    kind = by_name.get(example.kind)
    if kind is None:
        return f"{example.id} names the class {example.kind!r}, which the corpus does not have"
    if example.unit not in kind.units:
        return f"{example.id} gives {example.unit!r}, which is not a unit of {kind.name}"
    if example.unit in MISLABELLED_UNITS.get(kind.name, ()):
        return f"{example.id} answers {example.unit!r}, whose name contradicts the QUDT unit it denotes"
    if kind.name not in offered:
        return f"{example.id} needs {kind.name}, which the catalogue does not offer"
    if example.value_text not in example.text:
        return f"{example.id} expects {example.value_text!r}, which its document does not contain"
    return None


def load_examples(
    kinds: list[QuantityKind],
    *,
    path: Path | None = None,
    entries: dict[str, CatalogueEntry] | None = None,
    catalogue: tuple[str, ...] | None = None,
    split: Split = Split.DEV,
    difficulty: Difficulty = Difficulty.HARD,
    limit: int | None = None,
) -> list[TaskRecord]:
    """The corpus as tasks, refusing to load rather than dropping one.

    Every class and every unit is checked against the same corpus the
    catalogue is built from and the grader compares against. Anything that
    does not check out is reported together with everything else that did not,
    because fixing a corpus one raised error at a time is how a corpus loses
    examples.

    ``limit`` takes the first ``n`` per kind in file order, which the build
    fixed by document id. Taking a prefix rather than a sample keeps two
    callers asking for 120 tasks looking at the same 120.

    ``difficulty`` is :attr:`~oold_llm_bench.tasks.models.Difficulty.HARD`
    because that is what the register is: the reading sits inside a sentence
    written for a reader, not for an extractor.
    """
    corpus = read_corpus(path)
    by_name = {kind.name: kind for kind in kinds}
    offered = tuple(catalogue) if catalogue else tuple(kind.name for kind in kinds if kind.has_units)

    problems = [p for p in (_check(e, by_name, frozenset(offered)) for e in corpus.examples) if p]
    if problems:
        raise ValueError(
            "the Wiki-Measurements corpus does not resolve against the quantity corpus: " + "; ".join(problems[:20])
        )

    chosen: list[Example] = []
    if limit is None:
        chosen = list(corpus.examples)
    else:
        taken: dict[str, int] = {}
        for example in corpus.examples:
            if taken.get(example.kind, 0) < limit:
                taken[example.kind] = taken.get(example.kind, 0) + 1
                chosen.append(example)

    units = {kind.name: sorted(set(kind.units)) for kind in kinds if kind.name in set(offered) and kind.units}
    # Nothing is withheld. A synthetic task hides the annotation its document
    # was generated from, and these documents were generated from nothing.
    described = dict(zip(offered, render_catalogue(offered, entries), strict=True)) if entries else None
    enums = dict(zip(offered, render_catalogue(offered, entries, annotations=False), strict=True)) if entries else None
    notes = (
        f"corpus=wiki-measurements,dataset={corpus.dataset.get('file', '')},"
        f"doi={DATASET_DOI},notation=written,licence={TEXT_LICENCE}"
    )

    return [
        TaskRecord(
            id=example.id,
            document=example.text,
            expected=[
                ExpectedInstance(
                    key="q1",
                    class_path=example.kind,
                    fields={"value": Quantity(magnitude=example.magnitude, unit=example.unit)},
                )
            ],
            corpus=CorpusRef(
                name="wiki-measurements",
                source=Source.BULK,
                document_id=example.id,
                content_hash=hashlib.sha256(example.text.encode("utf-8")).hexdigest(),
                url=example.article,
                retrieved_at=corpus.retrieved_at,
                licence=TEXT_LICENCE,
            ),
            split=split,
            difficulty=difficulty,
            notes=notes,
            catalogue=list(offered),
            unit_catalogue=units,
            catalogue_text=described,
            catalogue_enums=enums,
        )
        for example in chosen
    ]
