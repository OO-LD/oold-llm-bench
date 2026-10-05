"""What a cell showed and what it enforced, named as one thing.

An arm name is composed of what varies, so a table can be read without a
legend. The context half names what the prompt carries and the enforcement
half what the decoder will accept::

    no-catalog-not-enforced             the document alone
    no-catalog-enforced                 the document, union enforced
    schema-prose-catalog-enforced       the quantities with their unit lists
    schema-prose-full-catalog-enforced  the same plus descriptions
    schema-dump-catalog-enforced        the JSON Schema printed out

The two halves are independent by construction, which is the point: holding
the enforcement fixed and walking the context down is the only way to ask what
the prompt is still doing once the grammar already forbids a wrong answer. It
also shows which pairs were never run, because a name that describes nothing
is a gap rather than an omission.

A cell's label is parsed from its arm rather than looked up, so an arm added
to the registry has a published name the moment it is named correctly. The one
thing the arm cannot say is how much the catalogue tells about each class,
because ``describe_catalogue`` is a property of the run: an arm says
``catalog`` and a label resolves that to ``schema-prose-catalog`` or
``schema-prose-full-catalog``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CONTEXT", "ENFORCED", "LEGACY_ARM_NAMES", "label_of"]

CONTEXT = {
    "no-catalog": "the document, no class list",
    "schema-prose-catalog": "the quantities, each with its unit list",
    "schema-prose-full-catalog": "the quantities, each with descriptions and its unit list",
    "schema-dump-catalog": "the JSON Schema printed in full, plus the document",
}

ENFORCED = {
    "not-enforced": "nothing",
    "flat-enforced": "class and unit as two independent enumerations",
    "enforced": "anyOf over one branch per class, each with that class's unit enum",
}

LEGACY_ARM_NAMES = {
    "A0-prose": "no-catalog-not-enforced-prose",
    "A0-json": "no-catalog-not-enforced",
    "A1": "schema-dump-catalog-not-enforced-gated",
    "A1-lean": "catalog-not-enforced-gated",
    "A2": "schema-dump-catalog-flat-enforced",
    "A2-strict": "schema-dump-catalog-flat-enforced-strict",
    "A2-enforced-only": "catalog-flat-enforced",
    "A3": "schema-dump-catalog-flat-enforced-grounded",
    "A4": "schema-dump-catalog-enforced",
    "A4-strict": "schema-dump-catalog-enforced-strict",
    "A4-enforced-only": "catalog-enforced",
    "A4-blind": "no-catalog-enforced",
}
"""What a published record written before the rename carries in ``arm``.

Results are not rewritten when a name changes, because a record is evidence
and editing it to match current vocabulary is the one thing a result file must
never do. The reader translates instead, so a run from either side of the
rename reaches the same label.
"""

_CARRIES = ("schema-dump-catalog", "no-catalog", "catalog")
_ACCEPTS = ("not-enforced", "flat-enforced", "enforced")


def label_of(arm: str, enforcement: dict[str, Any]) -> str:
    """The composed name for one cell, as ``context-enforcement``.

    ``enforcement`` is the record's own enforcement dict. Only
    ``describe_catalogue`` is read from it, because that is the one part of
    the context the arm does not fix.
    """
    name = LEGACY_ARM_NAMES.get(arm, arm)
    carries = next((c for c in _CARRIES if name == c or name.startswith(f"{c}-")), None)
    rest = name[len(carries) :].lstrip("-") if carries else ""
    accepts = next((a for a in _ACCEPTS if rest == a or rest.startswith(f"{a}-")), None)
    if carries is None or accepts is None:
        # Not a default. An unparseable arm falling back to "not-enforced"
        # labels a constrained cell as unconstrained, which is wrong in the
        # half that matters and reads as a deliberate condition.
        raise KeyError(f"arm {arm!r} does not name what it carries and what it enforces")
    qualifiers = rest[len(accepts) :]
    if carries == "catalog":
        # The arm says the catalogue is in the prompt. How much it says about
        # each class is a condition, so the label resolves it here.
        carries = "schema-prose-full-catalog" if enforcement.get("describe_catalogue") else "schema-prose-catalog"
    return f"{carries}-{accepts}{qualifiers}"
