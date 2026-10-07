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
    "balance",
    "documents_for",
    "quantity_tasks",
    "wikidata_schemaorg_tasks",
]

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
