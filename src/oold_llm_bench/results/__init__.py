"""Self-describing result records.

Every record names the condition it was produced under, and separates what
reproduces a run from what identifies one subscription.
"""

from oold_llm_bench.results.record import (
    PUBLISHED_ENVIRONMENT_FIELDS,
    PUBLISHED_MODEL_FIELDS,
    Environment,
    ModelSpec,
    RunRecord,
    config_hash,
)
from oold_llm_bench.results.store import ResultStore, read_records

__all__ = [
    "ResultStore",
    "read_records",
    Environment,
    ModelSpec,
    PUBLISHED_ENVIRONMENT_FIELDS,
    PUBLISHED_MODEL_FIELDS,
    RunRecord,
    config_hash,
]
