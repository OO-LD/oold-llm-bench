"""Where a tuning or evaluation task comes from.

Two corpora answer the same question differently, and H4 needs the one that
can answer it at all.

schema.org names every property in the document, so a task is a transcription
and no per-class knowledge is required on either half of a split. That was
measured: over 240 tasks the class name appeared 240 times and every one of
480 property names appeared, in readable form. A held-out drop cannot appear
there because nothing per class was ever needed.

QUDT states a magnitude and a unit and points at the kind with an elucidation
that does not name it. Deciding which kind admits ``bar`` is per-class
knowledge, it is absent from the document, and under a bare catalogue it can
only come from weights. That is the contrast H4 is about.

A source is a callable, not a class hierarchy. Everything that varies is
closed over when it is built, so :func:`~oold_llm_bench.finetune.build_corpus`
and the evaluation both stay ignorant of which corpus they are running.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from oold_llm_bench.corpus import quantities as qu
from oold_llm_bench.corpus import schemaorg as so
from oold_llm_bench.corpus.signals import Signal, SignalData, Vocabulary
from oold_llm_bench.finetune.dataset import CATALOGUE_SIZE, offered_catalogue
from oold_llm_bench.finetune.split import ClassSplit
from oold_llm_bench.tasks.models import Difficulty, TaskRecord, Variant

if TYPE_CHECKING:
    from typing import Any

__all__ = [
    "QUDT_CATALOGUE_SIZE",
    "QUDT_SIGNAL",
    "QUDT_VOCABULARY",
    "WIKI_LIMIT",
    "TaskDraw",
    "quantities_pool",
    "quantities_source",
    "schemaorg_source",
]

TaskDraw = Callable[[str, int], TaskRecord]
"""A task id and a seed in, one task out. The only thing a corpus has to be."""

QUDT_SIGNAL = Signal.ELUCIDATION
QUDT_VOCABULARY = Vocabulary.CONSENSUS
"""What the QUDT grid already runs, so the base numbers are known.

An elucidation points at the kind without naming it, which is the property
that makes the unit enumeration load-bearing. A named signal would put the
answer in the document and reproduce the schema.org defect.
"""

QUDT_CATALOGUE_SIZE = 100
"""The catalogue the enforcement grid offered, kept so the base is comparable."""


def _names(answers: Sequence[Any], variant: Variant = Variant.NATIVE) -> list[str]:
    return [so.class_identifier(entity.cls.name, variant) for entity in answers]


def schemaorg_source(
    classes: list[so.SchemaClass],
    answer_classes: list[so.SchemaClass],
    split: ClassSplit,
    *,
    catalogue_size: int = CATALOGUE_SIZE,
    difficulty: Difficulty = Difficulty.HARD,
    n_slots: int = 4,
    notation: qu.Notation = qu.Notation.CANONICAL,
) -> TaskDraw:
    """Tasks whose answer is one of ``answer_classes`` and whose catalogue is mixed.

    A written document costs no class here, unlike the quantity pool below.
    Every slot kind schema.org declares has a surface form, so the answer set
    and the split are the same under either notation.
    """

    def draw(task_id: str, seed: int) -> TaskRecord:
        entities = so.draw(answer_classes, random.Random(seed), 1, n_slots)  # noqa: S311 - reproducible, not secure
        catalogue = offered_catalogue(
            _names(entities),
            split,
            random.Random(f"catalogue:{seed}"),  # noqa: S311 - reproducible, not secure
            catalogue_size,
        )
        task, _ = so.generate_pair(
            classes,
            task_id=task_id,
            seed=seed,
            difficulty=difficulty,
            n_slots=n_slots,
            catalogue=catalogue,
            describe_catalogue=True,
            draw_from=answer_classes,
            notation=notation,
        )
        return task

    return draw


def quantities_pool(
    kinds: list[qu.QuantityKind],
    data: SignalData,
    notation: qu.Notation = qu.Notation.CANONICAL,
) -> list[qu.QuantityKind]:
    """The kinds a task can be built from under the signal H4 runs on.

    ``candidates`` already drops kinds with no identifier and, for an
    elucidation, kinds that share their signal text with another. Two kinds
    described by one sentence cannot both be the answer, so keeping them would
    cap every arm on those tasks and the loss would be read as a model
    failure.

    A written document narrows it further: a kind whose every unit has no
    writable symbol cannot be rendered, so it drops out. Measured over this
    pool, 42 of 323 go, 20 from the training half and 22 from the held-out
    one, and they are the dimensionless ratios QUDT writes as a mark for no
    unit at all. The two halves stay within one of each other, so the split
    survives the narrowing and does not have to be redrawn.
    """
    found = qu.candidates(kinds, QUDT_SIGNAL, data, QUDT_VOCABULARY)
    return qu.writable(found) if notation is qu.Notation.WRITTEN else found


def quantities_source(
    kinds: list[qu.QuantityKind],
    answer_kinds: list[qu.QuantityKind],
    split: ClassSplit,
    data: SignalData,
    entries: dict[str, Any],
    *,
    catalogue_size: int = QUDT_CATALOGUE_SIZE,
    difficulty: Difficulty = Difficulty.HARD,
    notation: qu.Notation = qu.Notation.CANONICAL,
) -> TaskDraw:
    """Tasks whose answer is one of ``answer_kinds`` and whose catalogue is mixed.

    The kind is drawn here before the task is generated, for the same reason
    the schema.org source does it: the catalogue has to contain the answer and
    which kind the answer is only becomes known once the draw has happened.
    The draw is repeated inside ``generate_task`` from the same seed over the
    same list, so the two agree.
    """
    usable = qu.candidates(answer_kinds, QUDT_SIGNAL, data, QUDT_VOCABULARY)

    def draw(task_id: str, seed: int) -> TaskRecord:
        chosen = random.Random(seed).sample(usable, 1)  # noqa: S311 - reproducible, not secure
        needed = [qu.identifier_for(kind, QUDT_VOCABULARY, data) or kind.name for kind in chosen]
        catalogue = offered_catalogue(
            needed,
            split,
            random.Random(f"catalogue:{seed}"),  # noqa: S311 - reproducible, not secure
            catalogue_size,
        )
        return qu.generate_task(
            kinds,
            task_id=task_id,
            seed=seed,
            n_entities=1,
            signal=QUDT_SIGNAL,
            vocabulary=QUDT_VOCABULARY,
            difficulty=difficulty,
            data=data,
            catalogue=catalogue,
            entries=entries,
            draw_from=answer_kinds,
            notation=notation,
        )

    return draw


WIKI_LIMIT = 80
"""How many tasks per kind the real-text run reads.

The corpus is 1,254 examples over twelve kinds and the distribution is very
uneven, 250 for ``Area`` against 4 for ``Speed``. Taking a prefix per kind
keeps the run affordable without letting the three largest kinds decide the
number: at 80 the held-out half is 168 tasks, which clears the 120 the power
table asks for, and the training half is 364.
"""


def wiki_tasks(
    kinds: list[qu.QuantityKind],
    split: ClassSplit,
    entries: dict[str, Any],
    *,
    catalogue_size: int = QUDT_CATALOGUE_SIZE,
    limit: int | None = WIKI_LIMIT,
) -> dict[str, list[TaskRecord]]:
    """Wiki-Measurements, grouped by which side of the split each kind is on.

    Three groups and not two. Some kinds the corpus carries are in neither
    half, because ``writable()`` narrowed them out of the pool the split was
    drawn over, and folding them into either one would report a number about
    kinds that half does not contain.

    One catalogue for the whole corpus, since the loader offers one. It holds
    every kind the corpus answers with, so no task is unanswerable, and fills
    the rest from both halves so a held-out kind is always on the list.
    """
    from oold_llm_bench.corpus.wiki_measurements import load_examples, read_corpus

    answered = sorted({example.kind for example in read_corpus().examples})
    catalogue = offered_catalogue(
        answered,
        split,
        random.Random("wiki-catalogue"),  # noqa: S311 - reproducible, not secure
        catalogue_size,
    )
    tasks = load_examples(kinds, entries=entries, catalogue=catalogue, limit=limit)

    train, heldout = set(split.train), set(split.heldout)
    grouped: dict[str, list[TaskRecord]] = {"train": [], "heldout": [], "unsplit": []}
    for task in tasks:
        kind = task.expected[0].class_path
        where = "train" if kind in train else ("heldout" if kind in heldout else "unsplit")
        grouped[where].append(task)
    return grouped
