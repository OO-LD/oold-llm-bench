"""Turning a schema module into the task set a grid runs on.

Each grid names a corpus rather than a file, because what a corpus needs to be
obtained differs. The quantity corpus is a tracked table of Wikipedia
sentences plus a schema module, so it builds from a clone and a download. The
Wikidata schema.org corpus needs the page text, which may not be
redistributed, so a third party harvests it and the loader says so rather than
failing on a missing path.

Tasks are balanced across classes in both. An unbalanced pool makes a mean a
report on whichever class the corpus collected most of: ``Person`` alone is a
third of the Wikidata draw.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oold_llm_bench.tasks.models import TaskRecord

__all__ = [
    "DOCUMENTS",
    "DOCUMENTS_CACHE",
    "DOCUMENTS_REVISION",
    "SYNTHETIC_CLASSES",
    "balance",
    "documents_for",
    "quantity_tasks",
    "synthetic_schemaorg_tasks",
    "wikidata_schemaorg_tasks",
]

SYNTHETIC_CLASSES = (
    "Event",
    "CollegeOrUniversity",
    "Movie",
    "MusicAlbum",
    "Organization",
    "Painting",
    "Periodical",
    "Person",
    "TVSeries",
    "VideoGame",
)
"""The ten classes the Wikidata corpus draws, held fixed here too.

A generated task is comparable to a harvested one only if both answer the
same question over the same catalogue. Drawing from a wider or narrower set
of classes would make a precision gap readable as a corpus effect when it was
a catalogue effect."""

DOCUMENTS = "OO-LD/oold-wikidata-schemaorg-documents"
DOCUMENTS_REVISION = "ed302be3d95fc21e2be21001f858b45b4d15c0b9"
"""Pinned, so a published corpus is not swapped for Wikipedia drift one way
and Hub drift the other. The per-lead sha256 would catch a changed document
anyway; this makes the download reproducible rather than merely checked."""
"""The leads the corpus cites, published under the licence they carry.

Not shipped in this repository, which is Apache-2.0, because the text is
CC BY-SA 4.0 and the two do not mix in one tree. Published as a dataset of
its own with attribution instead, which is how a Wikipedia derivative is
normally released.

Published rather than re-fetched, because the corpus pins a rendering and not
only a revision. A revision is retrievable: ``action=parse&oldid=`` returns
the one asked for. The sha256 was taken over ``prop=extracts`` output, and
that renderer only runs on the current text, so the lead as it was rendered
then cannot be reproduced once the article moves on. A later corpus can hash
something derivable from ``oldid`` and be drift-proof; this one carries its
bytes instead.
"""

DOCUMENTS_CACHE = Path(".cache/wikidata_schemaorg/documents.json")
"""Where a local harvest leaves the same text. Preferred when it is there, so
a corpus being rebuilt is read from the rebuild and not from the Hub."""


def documents_for(path: Path | None = None) -> dict[str, str]:
    """The leads, from a local cache if there is one and the Hub otherwise.

    Every lead is checked against the sha256 the corpus recorded, wherever it
    came from. A document that does not hash as recorded is a different
    document wearing the same id, and scoring against it would score answers
    against prose nobody measured.
    """
    from oold_llm_bench.corpus.wikidata_schemaorg import read_documents

    local = path or DOCUMENTS_CACHE
    if local.is_file():
        return read_documents(local)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            f"{local} is absent, so the leads come from the Hub: uv sync --extra corpora, "
            f"or rebuild the cache with scripts/fetch_wikidata_documents.py"
        ) from exc
    return read_documents(
        Path(hf_hub_download(DOCUMENTS, "documents.json", repo_type="dataset", revision=DOCUMENTS_REVISION))
    )


def balance(tasks: list[TaskRecord], per_class: int, classes: tuple[str, ...] = ()) -> list[TaskRecord]:
    """At most ``per_class`` tasks of each class, in a fixed order.

    ``classes`` restricts to a named set. A grid that compares a context
    ladder wants the same few classes throughout, so a model is never credited
    for a rung that happened to draw easier ones.
    """
    pooled: dict[str, list[TaskRecord]] = collections.defaultdict(list)
    for task in tasks:
        pooled[task.expected[0].class_path].append(task)
    wanted = classes or tuple(sorted(pooled))
    missing = [name for name in wanted if name not in pooled]
    if missing:
        raise ValueError(f"the corpus offers no task for {', '.join(missing)}")
    return [task for name in wanted for task in pooled[name][:per_class]]


def quantity_tasks(schemas: Path, per_class: int, classes: tuple[str, ...] = ()) -> list[TaskRecord]:
    """Wiki-Measurements, as tasks, against one generation of the schemas.

    The catalogue is built from the same module the grammar is built from, so
    a class offered in the prompt is a class the decoder will accept. Building
    them apart is how a catalogue comes to list a name no branch admits.
    """
    from oold_llm_bench.corpus import load_kinds
    from oold_llm_bench.corpus.catalogue import load_entries
    from oold_llm_bench.corpus.wiki_measurements import load_examples

    kinds = load_kinds(schemas)
    entries = load_entries(schemas, kinds)
    return balance(load_examples(kinds, entries=entries), per_class, classes)


def wikidata_schemaorg_tasks(per_class: int, documents: Path | None = None) -> list[TaskRecord]:
    """Wikidata-grounded schema.org entities, as tasks.

    The leads come from :func:`documents_for`, so a clone needs no harvest.
    """
    from oold_llm_bench.corpus import load_entities

    return balance(load_entities(documents_for(documents)), per_class)


def synthetic_schemaorg_tasks(per_class: int, schemas: Path | None = None, n_slots: int = 6) -> list[TaskRecord]:
    """Generated schema.org entities, as tasks, over the same ten classes
    :data:`wikidata_schemaorg_tasks` draws from real leads.

    The entity is written from a known set of facts rather than read from a
    page, so what the document states and what the corpus expects are the
    same thing by construction. Run beside the Wikidata corpus, the gap
    between them is the share of fillable's precision ceiling that belongs to
    Wikidata's coverage rather than to the step: see
    :mod:`oold_llm_bench.corpus.wikidata_schemaorg`'s module docstring for the
    measurement this is the control for.

    ``n_slots`` at 6 rather than the generator's own default of 4, because the
    harvested entities it is compared against hold more: the Wikidata corpus's
    grounded facts run from a handful to over a dozen per entity, and a
    generated document stating fewer would understate the fillable task.

    ``schemas`` resolves the pinned schema.org module the same way a quantity
    grid resolves its own, rather than through ``needs_schemas``: that flag is
    wired to the quantities module specifically, and a second corpus needing
    a different one would have to fight it for the same path.
    """
    from oold_llm_bench.corpus import SCHEMAORG, generate_schemaorg_task, load_classes, resolve_module
    from oold_llm_bench.tasks import Difficulty, Split

    directory = schemas or resolve_module(SCHEMAORG)
    classes = load_classes(directory)
    offered = [cls for cls in classes if cls.name in SYNTHETIC_CLASSES]
    by_name = {cls.name: cls for cls in offered}
    missing = set(SYNTHETIC_CLASSES) - set(by_name)
    if missing:
        raise ValueError(f"the schema.org module at {directory} has no class for {sorted(missing)}")

    tasks = [
        generate_schemaorg_task(
            classes,
            task_id=f"synthetic-{name}-{seed}",
            seed=seed,
            draw_from=[by_name[name]],
            n_entities=1,
            n_slots=n_slots,
            split=Split.DEV,
            difficulty=Difficulty.MEDIUM,
            catalogue=SYNTHETIC_CLASSES,
            describe_catalogue=True,
        )
        # Seeded off the class's position rather than drawn from one running
        # counter, so adding a class at the end never reseeds every task
        # already taken from the others. Not `hash(name)`: string hashing is
        # salted per process, and a seed that changes between runs reproduces
        # nothing.
        for index, name in enumerate(SYNTHETIC_CLASSES)
        for seed in range(index * 10_000, index * 10_000 + per_class)
    ]
    return balance(tasks, per_class)
