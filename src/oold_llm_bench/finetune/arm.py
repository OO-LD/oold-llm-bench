"""The arm the tuned model is trained and evaluated on.

H4 compares a tuned model given a lean prompt against a base model given the
whole schema. The lean prompt is the gated arm with the schema section removed: the
described catalogue is still shown, because a model that is not told which
classes are on offer is answering a different question, and no decode-time
constraint is applied, because a constraint would enforce the shape the tuning
is supposed to have taught.

The arm is registered rather than defined locally so that one name resolves to
one condition on both sides of the experiment. A training prompt built here
and an evaluation prompt built from :class:`~oold_llm_bench.runner.Condition`
have to be the same bytes, or the result measures the wording.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oold.agent.enforcement import Enforcement

__all__ = ["BASELINE_ARM", "LEAN_ARM", "lean_enforcement", "register_lean_arm"]

LEAN_ARM = "catalog-not-enforced-gated"
"""The prompt the tuned model sees: catalogue, no schema."""

BASELINE_ARM = "schema-dump-catalog-not-enforced-gated"
"""What the lean arm is derived from, and what it is compared against.

The baseline shows the schema and applies no decode-time constraint, so the
only difference between the two is the section H4 is about.
"""


def register_lean_arm() -> Enforcement:
    """Add the lean arm to the shared registry, once.

    Idempotent, so importing this package twice does not redefine the arm, and
    a caller that registered it already keeps the condition it registered.
    """
    from oold.agent.enforcement import ARMS

    if LEAN_ARM not in ARMS:
        ARMS[LEAN_ARM] = replace(ARMS[BASELINE_ARM], schema_in_prompt=False)
    return ARMS[LEAN_ARM]


def lean_enforcement() -> Enforcement:
    """The lean condition, with the empty catalogue a run fills in."""
    return register_lean_arm()
