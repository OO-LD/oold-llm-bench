"""What a study is configured with.

Model identity and pinned versions are committed. Deployments, endpoints and
anything else that names one account are supplied at run time and never
recorded in a published result.
"""

from oold_llm_bench.config.models import (
    LADDER,
    ModelEntry,
    drifted,
    entry_for,
    load_ladder,
    spec_for,
)

__all__ = [
    "LADDER",
    "ModelEntry",
    "drifted",
    "entry_for",
    "load_ladder",
    "spec_for",
]
