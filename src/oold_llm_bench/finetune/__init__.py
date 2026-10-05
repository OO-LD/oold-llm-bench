"""Tuning a small model on the schema set, and the split that makes it a test.

H4 predicts that a model tuned on the schemas answers a lean prompt as well as
a base model answers a prompt carrying the whole schema, at a fraction of the
prompt tokens. On its own that is a demonstration and not a result: a tuned
model could reach it by having memorised the corpus.

What makes it a test is the second half of the prediction. Half the classes
are held out of tuning, so the tuned model should keep the base model's score
on the training half and lose it on the held-out half. One half is the claim
and the other is its control, and the partition both refer to is written once
to ``data/schemaorg_split.json`` and read from there afterwards.

Evaluation is not in this package, but it has one requirement that is decided
here: the catalogue offered at evaluation time must contain both halves. A
held-out class that is not on the list cannot be chosen, the tuned model would
score zero on it for a reason that has nothing to do with tuning, and the
contrast would be unmeasurable. Tuning prompts already offer both halves for
the same reason, so the two look alike.
"""

from oold_llm_bench.finetune.arm import (
    BASELINE_ARM,
    LEAN_ARM,
    lean_enforcement,
    register_lean_arm,
)
from oold_llm_bench.finetune.azure import (
    DEFAULT_API_VERSION,
    TERMINAL_STATES,
    DataPlane,
    FineTuneError,
    data_plane_endpoint,
)
from oold_llm_bench.finetune.dataset import (
    CATALOGUE_SIZE,
    TRAIN_SEED_BASE,
    VALIDATION_SEED_BASE,
    Corpus,
    answer_text,
    build_corpus,
    build_example,
    lean_condition,
    offered_catalogue,
    round_trip_score,
    write_jsonl,
)
from oold_llm_bench.finetune.job import DEVELOPER_TIER, JobRequest, poll, summarise
from oold_llm_bench.finetune.split import (
    SPLIT_PATH,
    SPLIT_SEED,
    SPLIT_VERSION,
    ClassSplit,
    load_split,
    partition,
)

__all__ = [
    "BASELINE_ARM",
    "CATALOGUE_SIZE",
    "DEFAULT_API_VERSION",
    "DEVELOPER_TIER",
    "LEAN_ARM",
    "SPLIT_PATH",
    "SPLIT_SEED",
    "SPLIT_VERSION",
    "TERMINAL_STATES",
    "TRAIN_SEED_BASE",
    "VALIDATION_SEED_BASE",
    "ClassSplit",
    "Corpus",
    "DataPlane",
    "FineTuneError",
    "JobRequest",
    "answer_text",
    "build_corpus",
    "build_example",
    "data_plane_endpoint",
    "lean_condition",
    "lean_enforcement",
    "load_split",
    "offered_catalogue",
    "partition",
    "poll",
    "register_lean_arm",
    "round_trip_score",
    "summarise",
    "write_jsonl",
]
