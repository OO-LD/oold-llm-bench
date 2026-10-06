"""Scoring one step against the answer the corpus already holds.

Each step answers a different question, so each gets its own counts rather
than a share of the end-to-end triple F1. The alignment problem is the same
one the triple grader has: the step's keys are its own, and they have to be
matched to the corpus's keys before anything can be compared.

Matched on the class and the mention together, greedily, best first. An
entity the step named correctly but filed under another key is a correct
answer, and treating it as two errors would report the key scheme rather than
the step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from oold_llm_bench.grading.compare import normalise_text
from oold_llm_bench.grading.score import Score
from oold_llm_bench.grading.triples import Dimension

if TYPE_CHECKING:
    from oold_llm_bench.tasks.models import TaskRecord

__all__ = ["match_entities", "score_fillable", "score_identify"]


def _said(value: object) -> str:
    """One mention, folded the way every other comparison here folds text."""
    return normalise_text(value)


def _mention_hit(produced: object, expected: tuple[str, ...]) -> bool:
    """Whether the words point at the right entity, loosely.

    Containment either way, not equality. "Ada Lovelace" and "Ada" denote the
    same entity in a document that calls her both, and a step that answers
    one where the corpus recorded the other has not made a mistake worth
    failing an entity over.
    """
    got = _said(produced)
    if not got:
        return False
    return any(got == want or got in want or want in got for want in (_said(name) for name in expected) if want)


def match_entities(
    task: TaskRecord,
    produced: dict[str, tuple[str, ...]],
    mentions: dict[str, str] | None = None,
) -> dict[str, str]:
    """Which produced key answers which expected key.

    Greedy and best first: a pair agreeing on class and mention is taken
    before one agreeing on class alone, so a document holding two entities of
    one class is not matched by whichever happened to come first.
    """
    said = mentions or {}
    scored: list[tuple[int, str, str]] = []
    for key, classes in produced.items():
        for instance in task.expected:
            weight = 0
            if instance.class_path in classes:
                weight += 1
            wanted = tuple(getattr(instance, "mentions", ()) or ()) or (instance.key,)
            if _mention_hit(said.get(key, ""), wanted):
                weight += 2
            if weight:
                scored.append((weight, key, instance.key))
    taken: dict[str, str] = {}
    used: set[str] = set()
    for _, key, expected_key in sorted(scored, key=lambda row: (-row[0], row[1], row[2])):
        if key in taken or expected_key in used:
            continue
        taken[key] = expected_key
        used.add(expected_key)
    return taken


def score_identify(
    task: TaskRecord,
    produced: dict[str, tuple[str, ...]],
    mentions: dict[str, str] | None = None,
) -> dict[Dimension, Score]:
    """Step one: were the entities found, under the right class and words.

    Three dimensions and not one. An entity missed, an entity invented and an
    entity found under the wrong class are different failures, and the step
    after this one fails differently for each.
    """
    matched = match_entities(task, produced, mentions)
    entity = Score(
        true_positives=len(matched),
        false_positives=len(produced) - len(matched),
        false_negatives=len(task.expected) - len(matched),
    )
    by_key = {instance.key: instance for instance in task.expected}
    right, wrong, said_right, said_wrong = 0, 0, 0, 0
    for key, expected_key in matched.items():
        instance = by_key[expected_key]
        if instance.class_path in produced[key]:
            right += 1
        else:
            wrong += 1
        wanted = tuple(getattr(instance, "mentions", ()) or ())
        if not wanted:
            # The corpus recorded none, so there is nothing to be right or
            # wrong about. Absent rather than zero, as DUPLICATE is.
            continue
        if _mention_hit((mentions or {}).get(key, ""), wanted):
            said_right += 1
        else:
            said_wrong += 1
    out = {
        Dimension.ENTITY: entity,
        Dimension.CLASS: Score(true_positives=right, false_positives=wrong, false_negatives=wrong),
    }
    if said_right or said_wrong:
        out[Dimension.MENTION] = Score(
            true_positives=said_right, false_positives=said_wrong, false_negatives=said_wrong
        )
    return out


def score_fillable(
    task: TaskRecord,
    produced: dict[str, tuple[str, ...]],
    mentions: dict[str, str] | None = None,
    *,
    expected: dict[str, tuple[str, ...]] | None = None,
) -> dict[Dimension, Score]:
    """Step two: were the slots the document fills the ones it named.

    Precision and recall over the set, pooled across entities. Naming a slot
    the document never fills is a false positive, because the extract step is
    then made to demand a value that is not there; missing one is a false
    negative, because that value can no longer be reached at all.
    """
    from oold_llm_bench.steps.oracle import fillable_of

    wanted = expected if expected is not None else fillable_of(task)
    matched = match_entities(task, dict.fromkeys(produced, ()), mentions) if mentions else {}
    hit = over = under = 0
    seen: set[str] = set()
    for key, names in produced.items():
        expected_key = matched.get(key, key)
        seen.add(expected_key)
        want = set(wanted.get(expected_key, ()))
        got = set(names)
        hit += len(got & want)
        over += len(got - want)
        under += len(want - got)
    for key, want in wanted.items():
        if key not in seen:
            # An entity the step never answered for still owes its slots.
            under += len(want)
    return {Dimension.FILLABLE: Score(true_positives=hit, false_positives=over, false_negatives=under)}
