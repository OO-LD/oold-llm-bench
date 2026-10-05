"""Reading an arm's answer into triples.

Every arm is read by the same extractor, so a difference between arms is a
difference between arms and not between the ways their output was parsed.
"""

from oold_llm_bench.extract.json_answer import (
    CLASS_KEYS,
    UNIT_KEYS,
    VALUE_KEYS,
    extract_json,
)
from oold_llm_bench.extract.prose_answer import extract_prose, parse_loss

__all__ = [
    "CLASS_KEYS",
    "UNIT_KEYS",
    "VALUE_KEYS",
    "extract_json",
    "extract_prose",
    "parse_loss",
]
