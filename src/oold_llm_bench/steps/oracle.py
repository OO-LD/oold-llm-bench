"""The answer each step should have given, read off the task record.

Every stage of the pipeline is already in the corpus: it generated the
entities, so it knows their classes, which slots it filled, and what it called
them. These functions name those stages, so a step can be handed the one
before it without a model having produced it.

An oracle is the point, not a convenience. A step fed real upstream output is
measured together with every mistake made before it, and a chain of four such
measurements cannot say which step lost the entity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oold_llm_bench.tasks.models import TaskRecord

__all__ = ["fillable_of", "mentions_of", "plan_of", "shortlist_of"]


def shortlist_of(task: TaskRecord) -> dict[str, tuple[str, ...]]:
    """Step one's answer: the class of each expected entity.

    A shortlist of one. The step may keep k, and the corpus knows the right
    one, so handing it a longer list would be inventing distractors the
    document does not support.
    """
    return {instance.key: (instance.class_path,) for instance in task.expected}


def mentions_of(task: TaskRecord) -> dict[str, str]:
    """Step one's other answer: the words each entity was written from.

    Falls back to the key where the corpus recorded nothing, so a step given
    the oracle is never handed an empty string. A corpus with no mentions
    should not silently become a corpus whose mentions are blank.
    """
    out: dict[str, str] = {}
    for instance in task.expected:
        mentions = getattr(instance, "mentions", ()) or ()
        out[instance.key] = mentions[0] if mentions else instance.key
    return out


def fillable_of(task: TaskRecord, *, optional: bool = False) -> dict[str, tuple[str, ...]]:
    """Step two's answer: the slots the document states for each entity.

    ``optional`` adds the slots that are scored when present and not counted
    as missing when absent. Off by default, because the step is asked which
    properties the document fills and an optional slot that the document did
    not fill is not one of them.
    """
    out: dict[str, tuple[str, ...]] = {}
    for instance in task.expected:
        names = list(instance.fields)
        if optional:
            names += [name for name in instance.optional_fields if name not in names]
        out[instance.key] = tuple(names)
    return out


def plan_of(task: TaskRecord, *, mentions: dict[str, str] | None = None) -> tuple[Any, ...]:
    """The plan steps two and three take as input, from the oracle.

    Built here rather than in each caller so the key, the class and the
    mention always travel together: a plan whose keys do not match the
    expected keys makes every edge read as dangling, which would look like a
    finding about links.
    """
    from oold.agent.extraction import PlannedEntity

    said = mentions if mentions is not None else mentions_of(task)
    return tuple(
        PlannedEntity(key=instance.key, classes=(instance.class_path,), mention=said.get(instance.key, instance.key))
        for instance in task.expected
    )
