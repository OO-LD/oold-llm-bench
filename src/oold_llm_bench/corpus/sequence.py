"""Splitting one entity's facts across several documents.

Step four reads several documents about one entity and has to decide that
they are one entity (dedup) and fold their facts into one record (merge,
scored on :attr:`~oold_llm_bench.grading.triples.Dimension.PATCH`). Neither
has ground truth on the corpora this benchmark already has: a single-document
corpus names one entity once, and nothing records what a merge of several
documents should produce.

A generated sequence gives both for free. The entity is drawn the way
:func:`~oold_llm_bench.corpus.schemaorg.draw` already draws one, and its
facts are partitioned across documents instead of written into one, so the
true merge is exactly the entity the draw started from and dedup truth is
true by construction: every document's local entity *is* the same entity,
because there was only ever one.

Two parameters earn their keep because they are the two ways a merge can go
wrong. ``overlap`` repeats a field across adjacent documents, which is what
makes "kept" a real question: a merge that only ever adds has nothing to
prove it also keeps. ``conflict_rate`` perturbs a repeated field's second
occurrence, which is what makes "contradicted" a real question. Both are
reported in :class:`SequenceItem` rather than left for a scorer to rediscover
from the documents.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from oold_llm_bench.corpus.schemaorg import (
    Difficulty,
    Entity,
    Labelling,
    SchemaClass,
    Slot,
    Variant,
    answer_schema,
    branches_for,
    class_identifier,
    designating_slots,
    draw,
    mentions_for,
    property_identifier,
    render,
    spell,
)
from oold_llm_bench.grading.triples import Scalar
from oold_llm_bench.tasks.models import CorpusRef, ExpectedInstance, Source, Split, TaskRecord

__all__ = ["SequenceItem", "draw_sequence", "generate_sequence", "tasks_of"]


@dataclass(frozen=True)
class SequenceItem:
    """One entity's facts, split across documents, with what each document
    alone states and what folding all of them in should produce.
    """

    key: str
    cls: SchemaClass
    documents: tuple[str, ...]
    """One per step of the sequence, in presentation order. Each one is the
    whole document a step would be given; nothing outside this tuple is
    withheld from it."""
    per_document: tuple[dict[str, Any], ...]
    """What each document alone states, property identifier to value, in the
    same order as :attr:`documents`. The oracle input and output for running
    identify, fillable or extract on one step of the sequence standalone."""
    expected_merge: ExpectedInstance
    """The full entity, fields resolved the way :attr:`contradicted` records,
    mentions carried over from the original draw."""
    overlapping: frozenset[str]
    """Property identifiers stated in more than one document."""
    contradicted: dict[str, tuple[Scalar, Scalar]]
    """Property identifier to ``(earlier value, later value)``, for every
    overlapping property whose documents disagree. The later value is the one
    :attr:`expected_merge` carries: recency is the resolution rule this draws
    under, chosen because it is the one a reader applies without being told
    one, and declared here rather than left for a merge to assume silently."""


def _perturb(value: Scalar) -> Scalar:
    """A value of the same kind, clearly not the one given.

    Deterministic and type-preserving rather than drawn from the schema's own
    generators: a contradiction only has to be detectably different, and
    reaching into another slot's generator for a value that merely sounds
    plausible would make the corpus about plausibility rather than about
    disagreement.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return f"{value} (alternate)"
    return value


def draw_sequence(
    classes: list[SchemaClass],
    rng: random.Random,
    *,
    n_slots: int = 6,
    n_documents: int = 3,
    overlap: int = 1,
    conflict_rate: float = 0.0,
    key: str = "e1",
) -> SequenceItem:
    """Draw one entity and split it into ``n_documents`` partial sightings.

    ``overlap`` fields are each repeated in the document immediately after
    the one that first states them, so a merge has to recognise a restated
    fact and not only a new one. ``conflict_rate`` is the share of those
    repeats perturbed by :func:`_perturb` instead of restated unchanged; `0.0`
    draws an overlap with no contradiction in it, which is its own condition
    and not a lesser one.

    Every document also carries the entity's designator, the same value every
    document would carry it by in a real sequence: without it, nothing in a
    later document reads as being about the entity the first one named.
    """
    if n_documents < 1:
        raise ValueError(f"a sequence needs at least one document, asked for {n_documents}")
    entity = draw(classes, rng, 1, n_slots, Labelling.NAMED, named=True)[0]
    naming = {slot.name for slot in designating_slots(entity.cls)}
    designated = next(((slot, value) for slot, value in entity.values if slot.name in naming), None)
    if designated is None:
        raise ValueError(
            f"{entity.cls.name} offers no slot whose value reads as a name, so no document in the "
            "sequence could say a later one is about the entity the first one introduced"
        )
    designator = designated
    rest = [(slot, value) for slot, value in entity.values if slot.name not in naming]

    buckets: list[list[tuple[Slot, Scalar]]] = [[] for _ in range(n_documents)]
    for index, field in enumerate(rest):
        buckets[index % n_documents].append(field)

    overlapping: set[str] = set()
    contradicted: dict[str, tuple[Scalar, Scalar]] = {}
    repeatable = [(i, slot, value) for i, bucket in enumerate(buckets[:-1]) for slot, value in bucket]
    for i, slot, value in rng.sample(repeatable, min(overlap, len(repeatable))):
        name = property_identifier(slot.name, Variant.NATIVE)
        later = _perturb(value) if rng.random() < conflict_rate else value
        buckets[i + 1].append((slot, later))
        overlapping.add(name)
        if later != value:
            contradicted[name] = (value, later)

    documents: list[str] = []
    per_document: list[dict[str, Any]] = []
    for bucket in buckets:
        doc_entity = Entity(key=key, cls=entity.cls, values=(designator, *bucket))
        documents.append(render(doc_entity, Difficulty.MEDIUM, rng, Variant.NATIVE, Labelling.NAMED))
        fields = {property_identifier(slot.name, Variant.NATIVE): spell(value) for slot, value in doc_entity.values}
        per_document.append(fields)

    merged: dict[str, Any] = {}
    for fields in per_document:
        merged.update(fields)
    for name, (_, later) in contradicted.items():
        merged[name] = spell(later)

    expected_merge = ExpectedInstance(
        key=key,
        class_path=class_identifier(entity.cls.name, Variant.NATIVE),
        fields=merged,
        mentions=mentions_for(entity),
    )
    return SequenceItem(
        key=key,
        cls=entity.cls,
        documents=tuple(documents),
        per_document=tuple(per_document),
        expected_merge=expected_merge,
        overlapping=frozenset(overlapping),
        contradicted=contradicted,
    )


def generate_sequence(
    classes: list[SchemaClass],
    *,
    seed: int,
    n_slots: int = 6,
    n_documents: int = 3,
    overlap: int = 1,
    conflict_rate: float = 0.0,
) -> SequenceItem:
    """:func:`draw_sequence`, reproducible from a seed alone."""
    return draw_sequence(
        classes,
        random.Random(seed),  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        n_slots=n_slots,
        n_documents=n_documents,
        overlap=overlap,
        conflict_rate=conflict_rate,
    )


def tasks_of(item: SequenceItem, task_id: str) -> list[TaskRecord]:
    """One task per document, scoreable the way every other task here is.

    Each document's expected instance is what that document alone states,
    not the merge: running identify, fillable or extract on step ``i`` of a
    sequence is scored against step ``i``'s own facts, the same way a step
    grid scores one document and not a corpus.

    The catalogue offered is ``item.cls`` alone. ``item.cls.slots`` is
    already the class's effective set, inherited slots included, the way
    :func:`~oold_llm_bench.corpus.schemaorg.load_classes` resolves it, so
    nothing is lost by not widening the offer to its ancestors as well.
    """
    offered = [item.cls]
    shape = answer_schema(offered, Variant.NATIVE)
    narrowed = branches_for(offered, Variant.NATIVE)
    tasks: list[TaskRecord] = []
    for index, (document, fields) in enumerate(zip(item.documents, item.per_document, strict=True)):
        tasks.append(
            TaskRecord(
                id=f"{task_id}-seq{index}",
                document=document,
                expected=[
                    ExpectedInstance(
                        key=item.key,
                        class_path=class_identifier(item.cls.name, Variant.NATIVE),
                        fields=fields,
                    )
                ],
                corpus=CorpusRef(
                    name="sequence",
                    source=Source.SYNTHETIC,
                    document_id=f"{task_id}-seq{index}",
                    content_hash=hashlib.sha256(document.encode("utf-8")).hexdigest(),
                ),
                split=Split.DEV,
                catalogue=[class_identifier(item.cls.name, Variant.NATIVE)],
                answer_schema=shape,
                branches=narrowed,
                notes=f"sequence={task_id},step={index},of={len(item.documents)}",
            )
        )
    return tasks
