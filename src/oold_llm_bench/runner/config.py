"""Declaring what a study runs.

A cell is one condition, one model, one task, one repetition. Naming all four
lets a result be attributed, and naming them in configuration instead of in
code stops the grid drifting between runs.

Runs per cell is fixed here and written into the record before anything is
executed. Choosing it after seeing results is the oldest way to turn noise
into a finding.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product

from oold_llm_bench.results.record import ModelSpec, config_hash
from oold_llm_bench.tasks.models import Split, TaskRecord

__all__ = ["ORCHESTRATIONS", "Cell", "Condition", "ExperimentConfig"]

ORCHESTRATIONS = ("single_shot", "select_then_fill", "segmented", "multi_step")
"""Every way of breaking the work up that a cell can declare.

Named here rather than inline so the documentation can be held to it: each
one owes a recorded chain showing what it sends, and a pipeline added without
one fails the docs test.

``recursive`` is in the library's axis and is not here. It has no runner
implementation, so a condition naming it would build an agent that cannot
run.
"""


@dataclass(frozen=True)
class Condition:
    """One point in the space the study varies.

    The arm decides where structure is enforced. The rest decides what the
    document and the catalogue look like, and each is held constant while
    another moves, so a difference has one candidate cause.
    """

    arm: str
    catalogue_size: int | None = None
    """``None`` offers the whole catalogue. A number trims it, always keeping
    the classes a task needs."""
    signal: str = "unit"
    vocabulary: str = "consensus"
    language: str | None = None
    unit_match: str = "exact"
    """How strictly a unit has to be spelled for the primary metric. Exact by
    default: pint leniency lands on the arm with no enum, so making it decide
    the headline number would flatter the unconstrained arm. Physical equality
    is reported either way as its own dimension."""

    orchestration: str = "single_shot"
    """How the work is broken up. ``select_then_fill`` shortlists the class in
    one call and fills what that shortlist implies in a second."""

    shortlist_k: int | None = None
    """How many classes the select step may keep. ``1`` is the commit case,
    where a wrong selection cannot be recovered. Ignored by single-shot."""

    reasoning: str | None = None
    """Whether the model was asked to think before answering.

    ``None`` takes the provider's default and is what every result before
    2026-10-03 was measured under. ``"off"`` suppresses it.

    A condition and not a tuning knob, which is why it is in :meth:`key` and
    gets its own row. Measured on one Wiki-Measurements task at a 100-class
    catalogue: the 4B answers in 44.9s thinking and 4.6s not, the 9B in 305.8s
    thinking and 4.1s not. Those are the same answers, so the time is not
    buying accuracy here, but it is a different treatment and a number taken
    under one must never be pooled with a number taken under the other.

    The 9B is the reason this is not optional. With thinking on it spends the
    whole 4,000-token budget reasoning and returns no answer at all, so the
    arm is unrunnable rather than slow."""

    property_evidence: bool = False
    """Whether the property step must quote the words it read each one from.

    A condition rather than a default: requiring it lowers the score on five
    models of six, and what a model writes there says what it thought it was
    reading, so it is worth being able to ask."""
    plan_retry: bool = False
    """Whether a plan call that answered nothing is asked again.

    Its own field because the mitigation is not a property of any pipeline.
    gpt-5-nano returns the selection schema instead of a plan on about a third
    of plan calls, 37 of 120 under segmented against 0 of 120 under single
    shot, and that costs the whole cell. A retry recovers it.

    Separable so the comparison can be three arms rather than two: segmented,
    segmented with the retry, and multi-step. Folded into the orchestration,
    any gain would read as the pipeline's when it belongs to the retry."""

    describe_catalogue: bool = True
    """Whether the catalogue is shown with labels, descriptions and units, or
    as bare identifiers. On by default: identifiers alone are not a baseline a
    person would accept. Off is the condition that measures how much of a
    score comes from the identifier being self-describing."""

    pin_units: bool = True
    """Whether a constrained arm also closes the unit slot. On by default,
    because a quantity corpus closes it and leaving it open enforces half of
    what the schema says. Off is a condition worth measuring, not an
    oversight."""

    output_form: str = "json"
    """``prose`` routes the answer through the prose parser instead of the
    JSON one. Declared instead of read off the arm name, because which
    extractor scores an arm is part of the treatment."""

    embed_nested: bool = False
    """Whether an object-valued property is offered as an object to fill.

    Off by default, and that is the whole reason it is a field. Every result
    so far was taken against an answer shape with no slot an entity could be
    written into: a `Person` had nowhere to put a `PostalAddress`, so a plan
    that found one produced an entity nothing on the page pointed at. Turning
    that on silently would move the answer shape under every arm at once and
    leave no record of which numbers were taken under which.

    On, the offered classes' embeddings are added to the schema and to their
    branches, so what the model is asked for is the entity in place rather
    than an edge to one it has to state separately. Refused where the task
    carries no embeddings to add, since a condition that quietly does nothing
    writes a second row identical to the first.

    On, it also chooses the narrower of the task's two catalogues: a class the
    schema only ever embeds stops being a class an entity may be planned for,
    because offered both routes a model takes the top-level one and leaves the
    slot empty. Off, the same exclusion would leave that class unreachable.
    :func:`~oold_llm_bench.runner.adapter.offered_catalogue` resolves which
    one ran and ``catalogue_hash`` records it."""

    @property
    def key(self) -> str:
        """A short label, stable enough to group results by.

        Every axis the study varies is in it, and the list below is the whole
        of it. A bare catalogue and a described one are two different prompts
        under one arm; an answer shape with a slot an entity can be written
        into is a different question than one without.

        Four axes were missing here while the report key carried them, so two
        conditions differing only in ``pin_units`` produced the same run_id,
        wrote into one jsonl, and collapsed to one row in ``run.describe()``.
        preflight reads the same key: a control scoring 0.50 on one and 0.00
        on the other reports 0.25, which is exactly ``SPELLING_CEILING``, so
        the gate that refuses a corpus could be defeated by the pooling rather
        than by the corpus.
        """
        optional = (
            (not self.describe_catalogue, "bare"),
            (self.orchestration != "single_shot", f"{self.orchestration}(k{self.shortlist_k})"),
            (self.catalogue_size is not None, f"n{self.catalogue_size}"),
            (not self.pin_units, "open-units"),
            (self.unit_match == "physical", "physical"),
            (self.output_form != "json", self.output_form),
            (self.embed_nested, "embedded"),
            (self.plan_retry, "retry"),
            (bool(self.reasoning), f"think:{self.reasoning}"),
            (bool(self.language), self.language or ""),
        )
        parts = [self.arm, self.signal, self.vocabulary, *(part for varied, part in optional if varied)]
        return "/".join(parts)

    def describe(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "reasoning": self.reasoning,
            "catalogue_size": self.catalogue_size,
            "signal": self.signal,
            "vocabulary": self.vocabulary,
            "language": self.language,
            "unit_match": self.unit_match,
            "orchestration": self.orchestration,
            "shortlist_k": self.shortlist_k,
            "plan_retry": self.plan_retry,
            "property_evidence": self.property_evidence,
            "describe_catalogue": self.describe_catalogue,
            "pin_units": self.pin_units,
            "output_form": self.output_form,
            "embed_nested": self.embed_nested,
        }

    def __post_init__(self) -> None:
        if self.orchestration not in ORCHESTRATIONS:
            raise ValueError(f"unknown orchestration {self.orchestration!r}")
        if self.shortlist_k is not None and self.shortlist_k < 1:
            raise ValueError("a shortlist of nothing leaves the second step no class to fill")
        if self.plan_retry and self.orchestration not in {"segmented", "multi_step"}:
            raise ValueError("plan_retry needs an orchestration with a plan call to retry")
        if self.orchestration == "select_then_fill" and self.shortlist_k is None:
            raise ValueError("select_then_fill needs a shortlist_k, or the second step gains nothing")
        if self.unit_match not in {"exact", "physical"}:
            raise ValueError(f"unit_match must be exact or physical, not {self.unit_match!r}")
        if self.output_form not in {"json", "prose"}:
            raise ValueError(f"output_form must be json or prose, not {self.output_form!r}")


@dataclass(frozen=True)
class Cell:
    """One condition, one model, one task, one repetition."""

    condition: Condition
    model: ModelSpec
    task: TaskRecord
    repetition: int

    @property
    def key(self) -> str:
        return f"{self.condition.key}|{self.model.model}|{self.task.id}|r{self.repetition}"


@dataclass
class ExperimentConfig:
    """The grid, fixed before anything runs."""

    name: str
    conditions: list[Condition]
    models: list[ModelSpec]
    tasks: list[TaskRecord]
    runs_per_cell: int = 20
    """Twenty for an interaction, against the three the predecessor used. A
    binary outcome at three runs has four possible values, which is not
    enough resolution to see the effect the study is about."""
    split: Split = Split.DEV
    """Tuning touches dev. Every report states how often test was read."""
    max_attempts: int = 1
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.runs_per_cell < 1:
            raise ValueError("runs_per_cell must be at least 1")
        if not self.conditions:
            raise ValueError("an experiment needs at least one condition")
        if not self.models:
            raise ValueError("an experiment needs at least one model")
        wrong_split = [t.id for t in self.tasks if t.split is not self.split]
        if wrong_split:
            raise ValueError(f"tasks are not all in the {self.split.value} split: {wrong_split[:3]}")

    def cells(self) -> Iterator[Cell]:
        """Every cell, in a stable order so two runs are comparable."""
        for condition, model, task, repetition in product(
            self.conditions,
            self.models,
            self.tasks,
            range(1, self.runs_per_cell + 1),
        ):
            yield Cell(condition, model, task, repetition)

    @property
    def size(self) -> int:
        return len(self.conditions) * len(self.models) * len(self.tasks) * self.runs_per_cell

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "split": self.split.value,
            "runs_per_cell": self.runs_per_cell,
            "max_attempts": self.max_attempts,
            "conditions": [c.describe() for c in self.conditions],
            "models": [m.publish() for m in self.models],
            "n_tasks": len(self.tasks),
            "n_cells": self.size,
            "notes": self.notes,
        }

    @property
    def config_sha256(self) -> str:
        """Hash of the grid, so a rerun can prove it ran the same study."""
        return config_hash(self.describe())
