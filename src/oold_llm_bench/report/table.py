"""Turning records into a table somebody can read.

Every row is keyed on the whole condition, never on the arm alone. A mean over
arms pooled across models and catalogue sizes is the number that made the
predecessor's result unreadable: it could move because an arm helped, because
one model was added, or because a catalogue grew.

Dispersion is reported beside every mean. A cell mean with no spread next to
it cannot be told apart from noise, and at one run per cell there is no spread
to report, which is itself worth seeing.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Cell", "Table", "tabulate"]


@dataclass
class Cell:
    """Every score that landed in one condition and model."""

    condition: str
    model: str
    scores: list[float] = field(default_factory=list)
    dimensions: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    """Every dimension by name, holding F1, plus ``name:precision`` and
    ``name:recall`` beside it.

    Kept apart rather than averaged into the F1, because the three failures a
    multi-entity answer can have are a missed entity, an invented one and a
    repeated one, and an F1 that moved is consistent with all of them."""
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    document_chars: list[int] = field(default_factory=list)
    failures: int = 0

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores) if self.scores else 0.0

    @property
    def stdev(self) -> float:
        """Zero-width at one observation, which is the honest answer."""
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def input_per_task(self) -> float:
        """Input tokens one task costs, every step of it included.

        The number to compare orchestrations on. A two-step run sends the
        document twice, so a saving measured on the enum alone would say it is
        cheaper when the total says otherwise."""
        return self.input_tokens / self.n if self.n else 0.0

    @property
    def output_per_task(self) -> float:
        return self.output_tokens / self.n if self.n else 0.0

    @property
    def calls_per_task(self) -> float:
        return self.calls / self.n if self.n else 0.0

    @property
    def document_size(self) -> float:
        """Mean document length in characters, so the token figures can be
        read against the input they were spent on."""
        return statistics.fmean(self.document_chars) if self.document_chars else 0.0

    def dimension(self, name: str) -> float | None:
        """The cell mean for one dimension, or ``None`` where it does not apply.

        A dimension no record in the cell reported is not a zero. Reading it
        as one would drag down every cell that predates the dimension, and a
        column of real zeros would be indistinguishable from a column that
        was never filled in.
        """
        values = self.dimensions.get(name) or []
        return statistics.fmean(values) if values else None

    def describe(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "model": self.model,
            "n": self.n,
            "mean": round(self.mean, 4),
            "stdev": round(self.stdev, 4),
            "failures": self.failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_per_task": round(self.input_per_task, 1),
            "output_per_task": round(self.output_per_task, 1),
            "calls_per_task": round(self.calls_per_task, 2),
            "document_chars": round(self.document_size, 1),
            "dimensions": {
                name: round(mean, 4) for name in sorted(self.dimensions) if (mean := self.dimension(name)) is not None
            },
        }


@dataclass
class Table:
    """Cells, and the questions a reader asks of them first."""

    cells: dict[tuple[str, str], Cell] = field(default_factory=dict)

    def conditions(self) -> list[str]:
        return sorted({key[0] for key in self.cells})

    def models(self) -> list[str]:
        return sorted({key[1] for key in self.cells})

    def get(self, condition: str, model: str) -> Cell | None:
        return self.cells.get((condition, model))

    def saturated(self, ceiling: float = 0.99) -> list[tuple[str, str]]:
        """Cells with no headroom, which cannot show an effect."""
        return [key for key, cell in self.cells.items() if cell.n and cell.mean >= ceiling]

    def single_observation(self) -> list[tuple[str, str]]:
        """Cells with one run, where no dispersion can be reported."""
        return [key for key, cell in self.cells.items() if cell.n == 1]

    def render(
        self,
        dimensions: tuple[str, ...] = (
            "class",
            "value",
            "unit",
            "entity:precision",
            "entity:recall",
            "duplicate",
        ),
    ) -> str:
        """A fixed-width table. Condition and model both in the key.

        Class and value are always shown, and the default omits neither. A
        single accuracy number cannot distinguish a constraint that fixed the
        form of an answer from one that fixed its content, and the two come
        apart here: enforcement moves value by 0.10 to 0.37 and moves class by
        0.00 on one model and 0.04 on another.

        Entity precision, entity recall and duplicate are shown for the same
        reason one step further out. On a document holding several entities,
        missing one, inventing one and emitting one twice are three different
        failures with three different fixes, and every one of them lands in
        the same triple F1. Recall falls for the first, precision for the
        second and third, duplicate only for the third, so the three columns
        read together name which happened.

        A dimension no record reported prints as a dash. Printing 0.00 would
        read as a measured failure of something that was never measured.

        The condition column is as wide as the longest key, and the dimension
        columns as wide as the longest name. A fixed width misaligned every
        column to its right as soon as orchestration and fidelity joined the
        key, and a misaligned table is read wrong.
        """
        width = max((len(key) for key in self.conditions()), default=0)
        width = max(width, len("condition")) + 2
        column = max(max((len(name) for name in dimensions), default=0) + 2, 16)
        header = f"{'condition':<{width}}{'model':<24}{'n':>3}{'mean':>7}{'sd':>7}{'fail':>6}"
        for name in dimensions:
            header += f"{name:>{column}}"
        header += f"{'in/task':>10}{'out/task':>10}{'calls':>7}{'doc':>7}"
        lines = [header, "-" * len(header)]
        for condition in self.conditions():
            for model in self.models():
                cell = self.get(condition, model)
                if cell is None:
                    continue
                row = f"{condition:<{width}}{model:<24}{cell.n:>3}{cell.mean:>7.2f}{cell.stdev:>7.2f}{cell.failures:>6}"
                for name in dimensions:
                    mean = cell.dimension(name)
                    row += f"{mean:>{column}.2f}" if mean is not None else f"{'-':>{column}}"
                row += (
                    f"{cell.input_per_task:>10.0f}{cell.output_per_task:>10.0f}"
                    f"{cell.calls_per_task:>7.1f}{cell.document_size:>7.0f}"
                )
                lines.append(row)
        return "\n".join(lines)

    def describe(self) -> dict[str, Any]:
        return {
            "cells": [cell.describe() for cell in self.cells.values()],
            "saturated": [list(key) for key in self.saturated()],
            "single_observation": [list(key) for key in self.single_observation()],
        }


def tabulate(records: list[dict[str, Any]]) -> Table:
    """Group records by condition and model.

    Reads the published shape, so a table can be built from a file a third
    party was given. Anything it cannot find is left at zero instead of
    guessed at.

    Fidelity is resolved per condition and model before anything is grouped,
    not read off each record. A record whose run stopped before the schema was
    prepared carries no fidelity, and keying on what that record happens to
    hold put it in a row of its own, away from the cells it belongs to. That is
    how 179 answers of zero ended up in separate rows and left the rows above
    them reading 0.20 too high.
    """
    bands = _bands_by_condition(records)
    table = Table()
    for record in records:
        enforcement = record.get("enforcement") or {}
        model = (record.get("model") or {}).get("model", "unknown")
        base = _condition_key(record, enforcement)
        band = bands.get((base, model))
        condition = f"{base}/fid{band}" if band else base
        cell = table.cells.setdefault((condition, model), Cell(condition=condition, model=model))

        scores = record.get("scores") or []
        if not scores:
            cell.failures += 1
            continue
        for score in scores:
            cell.scores.append(float(score.get("primary_f1", 0.0)))
            for name, value in (score.get("dimensions") or {}).items():
                cell.dimensions[name].append(float(value.get("f1", 0.0)))
                for metric in ("precision", "recall"):
                    if metric in value:
                        cell.dimensions[f"{name}:{metric}"].append(float(value[metric]))

        cell.document_chars.append(int(record.get("document_chars") or 0))
        for call in (record.get("calls") or {}).get("calls", []):
            cell.calls += 1
            cell.input_tokens += int(call.get("input_tokens") or 0)
            cell.output_tokens += int(call.get("output_tokens") or 0)
    return table


def _bands_by_condition(records: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """The fidelity band each condition and model ran at.

    One band per group, taken from the records that reached the provider. A
    group that genuinely spans two bands keeps the widest, which shows up as a
    row that should not have been one treatment and is meant to be noticed.
    """
    seen: dict[tuple[str, str], set[str]] = {}
    for record in records:
        fidelity = (record.get("degradation") or {}).get("fidelity")
        if fidelity is None:
            continue
        key = (
            _condition_key(record, record.get("enforcement") or {}),
            (record.get("model") or {}).get("model", "unknown"),
        )
        seen.setdefault(key, set()).add(_fidelity_band(float(fidelity)))
    return {key: "|".join(sorted(bands)) for key, bands in seen.items()}


def _fidelity_band(value: float) -> str:
    """Which band a schema transform landed in.

    Four bands separate the cases that matter. Almost nothing survived, most
    of it survived, all of it survived, or inlining added keywords.
    """
    if value < 0.5:
        return "<0.5"
    if value < 0.95:
        return "0.5-0.95"
    if value <= 1.0:
        return "~1.0"
    return ">1.0"


def _part_of(name: str, enforcement: dict[str, Any]) -> str | None:
    """One axis as it appears in the key, or ``None`` where it was not varied.

    A table rather than a chain of branches, because a chain grows past what
    anyone rereads and the axis added last is the one that gets missed. An
    axis that reaches :attr:`~oold_llm_bench.runner.config.Condition.key` and
    not this function pools a thinking-off cell with a thinking-on one under a
    name that fits neither.

    ``plan_retry`` gets its own part rather than riding on the orchestration.
    Pooled with the plain arm, a gain that belongs to re-asking a failed plan
    call would read as the orchestration's.
    """
    value = enforcement.get(name)
    if name in ("signal", "vocabulary"):
        return str(value) if value else None
    if name == "catalogue_size":
        return f"n{value}" if value else None
    if name == "pin_units":
        return "units" if value else None
    if name == "describe_catalogue":
        return "described" if value else None
    if name == "orchestration":
        return f"{value}(k{enforcement.get('shortlist_k')})" if value and value != "single_shot" else None
    if name == "plan_retry":
        return "retry" if value else None
    if name == "unit_match":
        return "physical" if value == "physical" else None
    if name == "reasoning":
        return f"think:{value}" if value else None
    return str(value) if value else None


_KEY_AXES = (
    "signal",
    "vocabulary",
    "catalogue_size",
    "pin_units",
    "describe_catalogue",
    "orchestration",
    "plan_retry",
    "unit_match",
    "reasoning",
    "language",
)
"""Every axis the report separates on, in the order the key reads them.

Adding an axis to :class:`~oold_llm_bench.runner.config.Condition` and not to
this tuple pools two treatments under one row, which is the one thing the
report is not allowed to do.
"""


def _condition_key(record: dict[str, Any], enforcement: dict[str, Any]) -> str:
    """Arm plus everything that was varied around it."""
    parts = [str(record.get("arm", "?"))]
    parts += [part for part in (_part_of(name, enforcement) for name in _KEY_AXES) if part]
    return "/".join(parts)
