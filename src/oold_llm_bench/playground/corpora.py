"""The class collection the playground extracts against, and the tasks it loads.

schema.org is the default and the one the graph view is designed around: it
has a class hierarchy and object-valued properties, so a document can carry an
edge and a picture of the answer is worth drawing. The quantity corpora have
neither, and are offered beside it because they cost nothing to offer.

The schema.org module is a generated artefact held outside this repository, so
a fresh clone does not have it. Every entry point here therefore names the
directory it wanted and what to set, rather than failing on a missing file
several frames down. :func:`~oold_llm_bench.corpus.provenance.corpus_digest`
is reported with it, because a directory that half arrived still loads.

Nothing here decides what a run looks like. It produces task records, which is
the only thing the runner and the grader accept.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from oold_llm_bench.corpus import schemaorg as so
from oold_llm_bench.corpus.provenance import CorpusDigest, corpus_digest
from oold_llm_bench.corpus.quantities import QuantityKind
from oold_llm_bench.corpus.wiki_measurements import load_examples, read_corpus
from oold_llm_bench.tasks.models import (
    CorpusRef,
    Difficulty,
    ExpectedInstance,
    Source,
    Split,
    TaskRecord,
)

__all__ = [
    "ALL_CLASSES",
    "CORPORA",
    "NAME_SLOT",
    "SCHEMAS_ENV",
    "MissingSchemas",
    "SchemaCorpus",
    "catalogue_sets",
    "is_scoreable",
    "load_schemaorg",
    "paste_task",
    "schemaorg_tasks",
    "schemas_directory",
    "wiki_tasks",
]

SCHEMAS_ENV = "OOLD_BENCH_SCHEMAS"
"""Where the generated schema.org module sits.

Named the way the rest of this package names what it reads from the
environment, and required rather than guessed: a default path that happens to
exist on one machine is how two people compare answers measured against two
different generations of a corpus.
"""

ALL_CLASSES = "all"
"""The set offering every describable class, and the default for a paste."""

CORPORA = ("schemaorg", "wiki-measurements")
"""The task sources the playground offers, default first."""

MIN_OWN_SLOTS = 3
"""How many properties of its own a class needs to be worth describing.

The same threshold the fine-tuning pipeline uses, so the 122 classes here are
the 122 classes there and a playground run is comparable to a grid cell.
"""


class MissingSchemas(RuntimeError):
    """The schema collection is not where it was expected.

    Its own type so a caller can say what to do about it, which is a different
    message from a corpus that is present and unreadable.
    """


@dataclass(frozen=True)
class SchemaCorpus:
    """The schema.org module, read once and reported with its digest."""

    directory: Path
    digest: CorpusDigest
    classes: list[so.SchemaClass]
    """Everything the module declares, which is what supplies the hierarchy."""
    describable: list[so.SchemaClass]
    """Those with enough properties of their own to write a document about."""
    linked: list[so.SchemaClass]
    """Those that can carry an edge, which is the pool a linked draw uses."""

    @property
    def catalogue(self) -> tuple[str, ...]:
        return tuple(sorted(cls.name for cls in self.describable))

    def describe(self) -> dict[str, object]:
        return {
            "directory": str(self.directory),
            "digest": str(self.digest),
            "files": self.digest.files,
            "classes": len(self.classes),
            "describable": len(self.describable),
            "linked": len(self.linked),
        }


def schemas_directory(explicit: str | Path | None = None) -> Path:
    """The schema collection's directory, or a message saying how to set it."""
    raw = explicit or os.environ.get(SCHEMAS_ENV)
    if not raw:
        raise MissingSchemas(
            f"no schema collection: pass a directory or set {SCHEMAS_ENV} to the generated "
            f"schema.org module, the one holding *.schema.json"
        )
    directory = Path(raw).expanduser()
    if not directory.is_dir():
        raise MissingSchemas(f"{directory} is not a directory, so there is no schema collection to read")
    if not any(directory.glob("*.schema.json")):
        raise MissingSchemas(f"{directory} holds no *.schema.json, so it is not a generated schema module")
    return directory


def load_schemaorg(directory: str | Path | None = None) -> SchemaCorpus:
    """Read the schema.org module and the three pools a draw needs."""
    path = schemas_directory(directory)
    classes = so.load_classes(path)
    if not classes:
        raise MissingSchemas(f"{path} parsed to no classes, so the module is not the one this expects")
    describable = so.describable_classes(classes, min_own_slots=MIN_OWN_SLOTS)
    return SchemaCorpus(
        directory=path,
        digest=corpus_digest(path, "*.schema.json"),
        classes=classes,
        describable=describable,
        linked=so.linked_classes(describable),
    )


def schemaorg_tasks(
    corpus: SchemaCorpus,
    *,
    count: int = 12,
    seed: int = 1,
    linked: bool = True,
    n_entities: int = 2,
    n_slots: int = 4,
    split: Split = Split.DEV,
) -> list[TaskRecord]:
    """Generated schema.org documents, linked by default.

    Linked because the edge is what this view was built to show. A flat draw
    expects one entity per document and asserts no edge, which makes the graph
    a list with extra steps.

    The seed is the whole of the randomness, so a task id here names the same
    document as a task id in a grid run with the same seed.
    """
    pool = corpus.linked if linked else corpus.describable
    if not pool:
        raise MissingSchemas("the schema collection has no class that can carry an edge")
    catalogue = corpus.catalogue
    return [
        so.generate_task(
            corpus.classes,
            task_id=f"pg-schemaorg-{seed + index}",
            seed=seed + index,
            n_entities=n_entities if linked else 1,
            n_slots=n_slots,
            linked=linked,
            draw_from=pool,
            catalogue=catalogue,
            describe_catalogue=True,
            split=split,
            difficulty=Difficulty.MEDIUM,
        )
        for index in range(count)
    ]


def wiki_tasks(*, count: int = 12, split: Split = Split.DEV) -> list[TaskRecord]:
    """Real Wikipedia sentences, with the reading their annotation marks.

    The quantity kinds are read off the corpus itself rather than off the QUDT
    module, so this option needs no second directory. Every unit a kind admits
    here is a unit some example states, which is narrower than the QUDT
    enumeration and is stated so nobody reads the catalogue as the full one.
    """
    corpus = read_corpus()
    units: dict[str, set[str]] = {}
    for example in corpus.examples:
        units.setdefault(example.kind, set()).add(example.unit)
    kinds = [QuantityKind(name=name, units=tuple(sorted(seen))) for name, seen in sorted(units.items())]
    tasks = load_examples(kinds, split=split)
    return tasks[:count]


NAME_SLOT = "name"
"""The property a pasted document names its entities by.

Withheld by the generated corpus on purpose: ``name`` is Thing's, every class
inherits it, and a document built from it would describe most classes with the
same four properties, so the choice of class would stop mattering. The slot cap
then cuts the inherited slots off entirely and no offered class carries it.

A pasted sentence is not a generated one. It says "Andrea works at Siemens",
and a catalogue with nowhere to put "Andrea" pushed the model into
``additionalName`` on one run and ``address`` on the next. The label the graph
draws and the value two turns are compared on both come from here, so the slot
is added back for a paste and for nothing else.

It is also the field the placeholder expectation carries. A task record refuses
an expectation with no values, and a pasted document has no ground truth to put
there, so the placeholder holds this slot empty and nothing scores it.
"""


def paste_task(
    corpus: SchemaCorpus,
    document: str,
    *,
    catalogue_set: str | Sequence[str] = ALL_CLASSES,
    task_id: str = "pg-pasted",
) -> TaskRecord:
    """A pasted document, offered the named class sets and no ground truth.

    The catalogue is chosen by name and never by count. A condition trims
    around the classes an expectation names, which is right where the answer
    is known and impossible where the expectation is a placeholder. Trimming
    here on a fixed shuffle instead makes the offered list arbitrary: "Andrea
    works at Siemens" is shown 25 classes holding neither Person nor
    Organization, the plan picks Thing, which declares no links, so the edge
    cannot be expressed and the answer carries none.

    Several sets may be named and the catalogue is their union, because the
    sets are branches of Thing and a sentence rarely stays inside one.
    "Andrea works at Siemens" needs Person and Organization, which are two
    branches, and `all` is the only single set holding both: 122 classes whose
    property union is an answer schema of several hundred slots that a small
    model fills with nulls.

    The reference draw is linked, so the answer schema carries the link
    properties of the offered classes and the ranges that pin them. Without
    that an edge has no slot to be written in, which is how a pasted document
    came back with its relation flattened into a text field.

    The record carries no usable expectation, so nothing scores it. That is
    why the playground shows the score panel only for a corpus task.
    """
    catalogue = _catalogue_for(corpus, catalogue_set)
    offered = set(catalogue)
    # Drawn from the offered classes only. A draw from the whole pool can land
    # on a class the set left out, and a task record refuses a catalogue that
    # omits a class its expectation names.
    describable = [cls for cls in corpus.describable if cls.name in offered]
    linkable = [cls for cls in corpus.linked if cls.name in offered]
    linked = len(linkable) >= 2
    reference = so.generate_task(
        corpus.classes,
        task_id=task_id,
        seed=0,
        n_entities=2 if linked else 1,
        linked=linked,
        draw_from=linkable if linked else describable,
        catalogue=catalogue,
        describe_catalogue=True,
    )
    return reference.model_copy(
        update={
            "document": document,
            "expected": [ExpectedInstance(key="pasted", class_path=catalogue[0], fields={NAME_SLOT: ""})],
            "corpus": CorpusRef(
                source=Source.MANUAL,
                document_id=task_id,
                content_hash=_hash(document),
            ),
            "notes": "pasted into the playground; the expectation is a placeholder and is never scored",
            **_naming(reference, catalogue),
        }
    )


def _naming(reference: TaskRecord, catalogue: tuple[str, ...]) -> dict[str, object]:
    """The task fields that carry :data:`NAME_SLOT` into every offered class."""
    slot = {"type": "string", "description": "The name the document gives this entity."}
    shape = reference.answer_schema
    if shape is not None:
        shape = copy.deepcopy(shape)
        shape["properties"]["entities"]["items"]["properties"].setdefault(NAME_SLOT, slot)
    branches = None
    if reference.branches is not None:
        branches = {name: {NAME_SLOT: dict(slot), **copy.deepcopy(props)} for name, props in reference.branches.items()}
    described = reference.catalogue_text
    if described:
        described = {name: _named_entry(text) for name, text in described.items() if name in set(catalogue)}
    return {"answer_schema": shape, "branches": branches, "catalogue_text": described}


def _named_entry(text: str) -> str:
    """One catalogue entry with the name slot listed among its properties."""
    lines = text.splitlines()
    line = f"  {NAME_SLOT} (text)"
    if line in lines:
        return text
    after = next((i for i, one in enumerate(lines) if one.startswith("  parent:")), 0)
    return "\n".join([*lines[: after + 1], line, *lines[after + 1 :]])


def is_scoreable(task: TaskRecord) -> bool:
    """Whether this task carries ground truth worth showing a score against."""
    return not (len(task.expected) == 1 and task.expected[0].fields == {NAME_SLOT: ""})


def _hash(document: str) -> str:
    import hashlib

    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def catalogue_sets(corpus: SchemaCorpus) -> dict[str, tuple[str, ...]]:
    """The named class sets a paste may be offered, largest first.

    A count is the wrong way to choose a catalogue. Twenty-five of 122 leaves
    the answer in or out by accident, and "Andrea works at Siemens" was once
    shown 25 classes holding neither Person nor Organization, so the plan
    picked Thing, which declares no links, and the edge had nowhere to go.

    The names are the ontology's own: the child of Thing each class descends
    through. A reader choosing "Organization" knows what they are asking for in
    a way that "25" cannot express.

    The sets do not overlap and a document does not respect them, so they are
    meant to be combined: Person holds one class and Organization five, and the
    two together are the six a sentence about employment needs.
    """
    sets = {ALL_CLASSES: tuple(corpus.catalogue)}
    grouped = so.top_level_sets(corpus.classes, corpus.describable)
    for name, members in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        sets[name] = members
    return sets


def _catalogue_for(corpus: SchemaCorpus, chosen: str | Sequence[str]) -> tuple[str, ...]:
    """The classes the named sets offer between them, in corpus order.

    Sorted rather than concatenated, so naming two sets in either order gives
    the same catalogue and two runs of one choice are the same run.
    """
    names = [chosen] if isinstance(chosen, str) else list(chosen)
    if not names:
        names = [ALL_CLASSES]
    sets = catalogue_sets(corpus)
    unknown = [name for name in names if name not in sets]
    if unknown:
        raise KeyError(f"unknown class set {unknown[0]!r}, expected one of {sorted(sets)}")
    return tuple(sorted({member for name in names for member in sets[name]}))
