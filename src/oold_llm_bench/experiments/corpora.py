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

__all__ = ["DOCUMENTS_CACHE", "balance", "quantity_tasks", "wikidata_schemaorg_tasks"]

DOCUMENTS_CACHE = Path(".cache/wikidata_schemaorg/documents.json")
"""Where the harvest leaves the page text, and the reason it is not shipped.

The truth was measured against exact bytes of a Wikipedia lead, and those
bytes belong to their authors. The corpus table carries the ids and the
sha256 of each; the text is fetched, checked against the hash, and never
redistributed.
"""


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

    Raises with the command that produces the cache rather than with a
    missing-file traceback, because the harvest is an hour of polite fetching
    and a reader meeting this for the first time needs to know that.
    """
    from oold_llm_bench.corpus import load_entities, read_documents

    path = documents or DOCUMENTS_CACHE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} holds the page text this corpus is scored against, and it is not redistributable. "
            "Produce it with: uv run python scripts/harvest_wikidata_schemaorg.py"
        )
    return balance(load_entities(read_documents(path)), per_class)
