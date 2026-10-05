"""Declarative task records.

Expectations live in data, not in the code that runs them. A task carries the input
document, the instances expected from it with their values, and the labels
that decide which cell a result belongs to.
"""

from oold_llm_bench.tasks.models import (
    CorpusRef,
    Difficulty,
    ExpectedInstance,
    Source,
    Split,
    TaskRecord,
    TaskSet,
    Variant,
)

__all__ = [
    "CorpusRef",
    "Difficulty",
    "ExpectedInstance",
    "Source",
    "Split",
    "TaskRecord",
    "TaskSet",
    "Variant",
]
