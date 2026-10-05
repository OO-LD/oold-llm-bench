"""Arms this benchmark adds to the shared registry.

The library ships the ladder; a study sometimes needs a rung the library has
no opinion about. Registered here rather than in the library because the
library should not carry an arm that exists to answer one question.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

__all__ = ["UNION_BLIND_ARM", "UNION_ENFORCED_ARM", "register_union_arms"]

UNION_ENFORCED_ARM = "catalog-enforced"
"""The union constraint applied and never shown.

The union enforces an ``anyOf`` over one branch per class, each branch carrying the
unit enumeration that class admits, so a unit that cannot occur with the
chosen class cannot be produced. That is the constraint this corpus actually
states, and a flat enum does not: measured on a hundred offered classes, the
flat grammar is two independent enumerations, a hundred class names
and six hundred and five unit identifiers, so ``Altitude`` with ``kilogram``
validates.

The union pays for that correctness in context. Printing a hundred discriminated
branches costs 32,401 tokens where the prose catalogue states the same
per-class mapping in 7,204, so the JSON is four times the size of the prose
that says the same thing. Separating the constraint from its presentation is
what lets the prompt shrink while the grammar stays correct, which is the only
way to ask whether a tune can pay for what the prompt stops saying.
"""


UNION_BLIND_ARM = "no-catalog-enforced"
"""The union applied and the catalogue never mentioned.

The floor of the context ladder, and the one that asks whether a grammar can
carry a task on its own. The decode constraint still admits one branch per
class with that class's units, so every answer it can produce is a valid one;
the prompt says only what the task is and names no class at all.

If a model scores here, the answer space was all it needed. If it scores
zero, it needs to be told what it is choosing among even when it could not
produce anything else, which is a different and more interesting kind of
failure than getting the choice wrong.

``no-catalog-not-enforced`` is not this. That arm has no grammar either, so it measures a
model with no constraint and no catalogue; this one holds the constraint and
removes only the telling.
"""


def register_union_arms() -> Any:
    """Add the arm to the shared registry, once.

    Idempotent, so importing twice does not redefine it and a caller that
    registered it already keeps what it registered.
    """
    from oold.agent.enforcement import ARMS

    if UNION_ENFORCED_ARM not in ARMS:
        ARMS[UNION_ENFORCED_ARM] = replace(ARMS["schema-dump-catalog-enforced"], schema_in_prompt=False)
    if UNION_BLIND_ARM not in ARMS:
        ARMS[UNION_BLIND_ARM] = replace(
            ARMS["schema-dump-catalog-enforced"], schema_in_prompt=False, catalogue_in_prompt=False
        )
    return ARMS[UNION_ENFORCED_ARM]
