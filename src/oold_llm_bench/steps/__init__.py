"""One step of the graph pipeline, run and scored on its own.

The pipeline is four steps and the benchmark has only ever scored the end of
it, so a number that fell could not be attributed. A step given the output of
the one before it is measured together with every mistake upstream; a step
given the **corpus's own answer** is measured on its own question.

    identify            document, catalogue      -> class and mention per entity
    fillable properties document, class, mention -> the slots the document fills
    extract             document, class, slots   -> the entity, ids and links
    dedup               entity, existing graph   -> match and merge, or mint

Oracle input comes from the task record, which already carries every stage:
:attr:`~oold_llm_bench.tasks.models.ExpectedInstance.class_path` is step one's
answer, ``set(fields)`` is step two's, and ``fields`` with the edges is step
three's. Nothing here invents ground truth.

Running a prefix of the chain for real and the rest from the oracle makes
every prefix a condition of its own, so the drop a step contributes is the
difference between two measured cells rather than an attribution argument.
"""

from __future__ import annotations

from oold_llm_bench.steps.oracle import (
    fillable_of,
    mentions_of,
    plan_of,
    shortlist_of,
)
from oold_llm_bench.steps.score import score_fillable, score_identify

__all__ = [
    "fillable_of",
    "mentions_of",
    "plan_of",
    "score_fillable",
    "score_identify",
    "shortlist_of",
]
