"""Deterministic scoring.

Output from every arm is reduced to triples first, then scored. No judge model
is used anywhere, so the grader is reproducible and can be unit tested against
known-wrong cases.
"""

from oold_llm_bench.grading.align import Alignment, Pairing, align, duplicates, overlap
from oold_llm_bench.grading.compare import (
    MatchMode,
    normalise_text,
    same_quantity,
    same_value,
)
from oold_llm_bench.grading.score import Score, TaskScore, score_task
from oold_llm_bench.grading.triples import (
    TRIPLE_SCHEMA_VERSION,
    Dimension,
    Quantity,
    Reference,
    Scalar,
    Triple,
    TripleSet,
    make_triple,
    make_triples,
    merge,
    normalise_property,
)

__all__ = [
    "TRIPLE_SCHEMA_VERSION",
    "Alignment",
    "Dimension",
    "MatchMode",
    "Pairing",
    "Quantity",
    "Reference",
    "Scalar",
    "Score",
    "TaskScore",
    "Triple",
    "TripleSet",
    "align",
    "duplicates",
    "make_triple",
    "make_triples",
    "merge",
    "normalise_property",
    "normalise_text",
    "overlap",
    "same_quantity",
    "same_value",
    "score_task",
]
