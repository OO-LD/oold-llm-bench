"""Turning a cell into a configured agent.

This is the join between the two packages. ``oold.agent`` knows what an
enforcement condition is, not what a benchmark is.
``oold_llm_bench`` knows what a cell is, not what a provider is.
One of them has to bridge that, and doing it here keeps the bridge visible
instead of spreading provider knowledge through the runner.

Everything a cell needs comes from the cell. The arm decides the enforcement,
the model spec names the provider profile, and the task supplies the catalogue
the gate and the decode-time constraint both work against. Nothing is read
from the environment. A variable that is not declared does not reach the
record, and a condition missing from the record cannot be checked.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from oold_llm_bench.runner.config import Cell

if TYPE_CHECKING:
    from oold.agent.client import ChatClient
    from oold.agent.enforcement import Enforcement
    from oold.agent.extraction import ExtractionAgent
    from oold.agent.prompts import ExtractionRequest

__all__ = [
    "ANSWER_SCHEMA",
    "agent_factory",
    "build_agent",
    "build_enforcement",
    "build_request",
]

ANSWER_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QuantityValues",
    "description": "Every measurement stated in the document.",
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "The quantity kind this measurement is an instance of.",
                    },
                    "value": {
                        "type": "number",
                        "description": "The magnitude, as written in the document.",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit the magnitude is given in.",
                    },
                },
                "required": ["type", "value", "unit"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}
"""The answer shape every arm is scored against.

Flat on purpose: a measurement is a magnitude and a unit, and nothing here
attaches one to a subject. A later stage adds that, once a quantity belongs
to a sample and not just to a document.

The property named ``type`` is the slot the decode-time constraint pins and
the commit gate checks, so an arm that carries a catalogue has somewhere to
put it.
"""


def _catalogue_of(cell: Cell) -> tuple[str, ...] | None:
    """The class list this cell offers, trimmed to the declared size.

    The classes a task needs are always kept. Trimming them out would change
    what the task is asking, not how large the catalogue is, and the catalogue
    size is the variable under test.

    The result is then shuffled on the task id. Keeping the needed classes at
    the front, the order selection takes them in, would put the answer at
    position zero of every catalogue and of every enum: the ordering would
    carry the answer, and a model biased towards early options would score for
    that bias. Shuffling on the task id rather than at random keeps two runs
    of one config identical.
    """
    catalogue = list(cell.task.catalogue or ())
    if not catalogue:
        return None
    limit = cell.condition.catalogue_size
    if limit is None or limit >= len(catalogue):
        return _shuffled(catalogue, cell.task.id)

    needed = [i.class_path for i in cell.task.expected if i.class_path]
    kept = list(dict.fromkeys(needed))
    for name in catalogue:
        if len(kept) >= limit:
            break
        if name not in kept:
            kept.append(name)
    if len(kept) > limit:
        raise ValueError(
            f"task {cell.task.id} needs {len(kept)} classes but the condition offers a catalogue of {limit}"
        )
    return _shuffled(kept, cell.task.id)


def _shuffled(names: list[str], task_id: str) -> tuple[str, ...]:
    """A stable order that does not depend on which class the answer is."""
    rng = random.Random(f"catalogue:{task_id}")  # noqa: S311 - reproducible, not secure
    ordered = list(names)
    rng.shuffle(ordered)
    return tuple(ordered)


def _units_of(cell: Cell, catalogue: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """The units the offered classes admit, and no others.

    Trimming the catalogue trims the units with it. A 25-class catalogue
    carrying the units of the whole corpus would constrain almost nothing
    while reporting a closed slot.
    """
    per_class = cell.task.unit_catalogue
    if not per_class:
        return None
    names = catalogue if catalogue is not None else tuple(per_class)
    units = {unit for name in names for unit in per_class.get(name, ())}
    return tuple(sorted(units)) or None


def _catalogue_text_of(cell: Cell, catalogue: tuple[str, ...]) -> tuple[str, ...] | None:
    """The rendered entries for the classes this cell offers, in its order.

    Two renderings, chosen by what the condition enforces rather than by one
    flag. ``describe_catalogue`` decides whether the prose is shown, because
    a label and a description are checkable by nobody. ``pin_units`` decides
    whether the unit enumeration is shown, because that one is exactly what
    the grammar will accept, and a schema presented without what it enforces
    asks the model to guess at a vocabulary it was never given.

    Moved together, the bare catalogue lists a hundred class identifiers in
    full and withholds every unit identifier while the grammar enforces them.
    Class accuracy then reads 0.98 against unit accuracy 0.02 on the same
    cell, which is a fact about the prompt and reads as a fact about models.

    Falls back to a bare identifier for a class the task carries no text for,
    so the catalogue the model sees is never shorter than the one the
    condition declares.
    """
    described = cell.condition.describe_catalogue
    rendered = cell.task.catalogue_text if described else cell.task.catalogue_enums
    if not rendered:
        return None
    return tuple(rendered.get(name) or f"- {name}" for name in catalogue)


def build_enforcement(cell: Cell) -> Enforcement:
    """The enforcement condition this cell runs under.

    Separate from :func:`build_agent` because a caller may need the condition
    without a provider. Building the prompt for a fine-tuning corpus is that
    case: the training turns have to be the turns an evaluation sends, and the
    only way to guarantee that is for both to come from here.
    """
    from oold.agent.enforcement import ARMS, DecodeConstraint, arm

    if cell.condition.arm not in ARMS:
        raise KeyError(f"unknown arm {cell.condition.arm!r}, expected one of {sorted(ARMS)}")
    # A no-catalog arm is shown no class list at all, so a declared size has
    # nothing to trim and offering one would be refused.
    offers_catalogue = ARMS[cell.condition.arm].catalogue is not None
    catalogue = _catalogue_of(cell) if offers_catalogue else None
    enforcement = arm(cell.condition.arm, catalogue)
    if offers_catalogue and cell.condition.pin_units:
        enforcement = enforcement.with_units(_units_of(cell, catalogue))
    # Shown when the condition describes the catalogue, and also when it only
    # closes the unit slot: the enumeration the grammar enforces belongs in
    # the schema the prompt presents.
    #
    # Unless the prompt already carries the schema. An arm with
    # schema_in_prompt prints the enumerations in JSON, so rendering them a
    # second time in prose is the same content twice: measured on the quantity
    # catalogue, 6,300 context tokens to repeat 605 unit identifiers the
    # schema block already listed. The schema is the cheaper carrier because
    # it is flat, listing each unit once where the catalogue repeats a set per
    # class, and that cheapness is also its weakness: flat means the grammar
    # accepts a unit with a class it cannot occur with.
    # Only an arm that both shows the schema and constrains with it prints the
    # enumerations. An unenforced arm shows a schema and enforces nothing, so its schema
    # carries no enum to duplicate, and suppressing its catalogue left it with
    # neither the prose nor the JSON: 818 tokens naming a hundred classes and
    # not one unit.
    arm_now = ARMS[cell.condition.arm]
    shows_enums = (
        getattr(arm_now, "schema_in_prompt", False)
        and getattr(arm_now, "decode_constraint", None) is not DecodeConstraint.NONE
    )
    wants_text = cell.condition.describe_catalogue or (cell.condition.pin_units and not shows_enums)
    if offers_catalogue and catalogue and wants_text:
        text = _catalogue_text_of(cell, catalogue)
        if text:
            enforcement = enforcement.with_catalogue_text(text)
    return enforcement


def build_agent(cell: Cell, client: ChatClient, *, attempts: int = 1) -> ExtractionAgent:
    """The agent this cell runs, with nothing decided outside the cell."""
    from oold.agent.enforcement import Orchestration
    from oold.agent.extraction import ExtractionAgent
    from oold.agent.provider import profile_for

    enforcement = build_enforcement(cell)

    return ExtractionAgent(
        client=client,
        enforcement=enforcement,
        profile=profile_for(cell.model.provider_profile),
        orchestration=Orchestration(cell.condition.orchestration),
        attempts=attempts,
        shortlist_k=cell.condition.shortlist_k or 3,
        plan_retry=cell.condition.plan_retry,
    )


def _catalogue_for(cell: Cell) -> tuple[str, ...] | None:
    """The classes this cell offers, or None when the arm offers none."""
    from oold.agent.enforcement import ARMS

    if cell.condition.arm not in ARMS or ARMS[cell.condition.arm].catalogue is None:
        return None
    return _catalogue_of(cell)


def _branches_of(cell: Cell) -> dict[str, dict[str, Any]] | None:
    """What each offered class narrows, for a union constraint.

    The task's own branches when it carries them, because only the corpus
    knows what a class narrows. Falling back to the unit enumeration keeps the
    quantity corpus working without having to describe itself twice.
    """
    if cell.task.branches:
        return dict(cell.task.branches)
    per_class = cell.task.unit_catalogue
    if not per_class:
        return None
    return {name: {"unit": {"type": "string", "enum": list(units)}} for name, units in per_class.items() if units}


def _answer_schema_of(cell: Cell, catalogue: tuple[str, ...] | None) -> dict[str, Any] | None:
    """The task's answer shape, trimmed to the classes this cell offers.

    A corpus whose classes decide which properties exist writes an answer
    schema covering every class it has. Sending all of it when a quarter of
    the classes are on offer overstates what the arm was asked to handle, and
    leaves an unconstrained arm free to answer with a property no offered
    class defines.
    """
    schema = cell.task.answer_schema
    branches = cell.task.branches
    if not schema or not branches or catalogue is None:
        return schema

    offered = {name for name in catalogue if name in branches}
    if not offered:
        return schema
    keep = {prop for name in offered for prop in branches[name]}

    trimmed = copy.deepcopy(schema)
    items = trimmed.get("properties", {}).get("entities", {}).get("items")
    if not isinstance(items, dict):
        return trimmed
    properties = items.get("properties") or {}
    # A property no branch defines is structural, like the class slot or a
    # magnitude every class shares, so it stays.
    defined = {prop for branch in branches.values() for prop in branch}
    items["properties"] = {
        name: definition for name, definition in properties.items() if name in keep or name not in defined
    }
    return trimmed


def _parents_of(cell: Cell) -> dict[str, tuple[str, ...]] | None:
    """The class hierarchy, when the corpus has one worth expressing."""
    lineage = cell.task.class_parents
    if not lineage:
        return None
    return {name: tuple(values) for name, values in lineage.items()}


def _ranges_of(cell: Cell) -> dict[str, tuple[str, ...]] | None:
    """Which classes each link property may point at.

    What `_pin_references` needs to constrain a reference slot to the ids of
    planned entities whose class fits. Supplied per request because it is
    corpus material, the way branches and parents already are.
    """
    declared = cell.task.property_ranges
    if not declared:
        return None
    return {name: tuple(values) for name, values in declared.items()}


def build_request(cell: Cell) -> ExtractionRequest:
    """The document and the schema material this cell is allowed to show.

    The schema is handed over for every arm. Which arm actually shows it, and
    which one sends it to the provider, is the enforcement's decision, so the
    difference between the arms stays in one place.
    """
    from oold.agent.prompts import ExtractionRequest

    return ExtractionRequest(
        document=cell.task.document,
        schema=_answer_schema_of(cell, _catalogue_for(cell)) or ANSWER_SCHEMA,
        branches=_branches_of(cell),
        parents=_parents_of(cell),
        ranges=_ranges_of(cell),
        property_text=cell.task.property_text or None,
        property_evidence=cell.condition.property_evidence,
    )


def agent_factory(client_for: Callable[[Cell], ChatClient], *, attempts: int = 1) -> Callable[[Cell], ExtractionAgent]:
    """An ``agent_for`` callback for :func:`~oold_llm_bench.runner.execute.run_experiment`.

    Only the client is left to the caller, since that is the one part that
    needs credentials.
    """
    return lambda cell: build_agent(cell, client_for(cell), attempts=attempts)
