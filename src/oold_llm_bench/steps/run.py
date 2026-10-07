"""Running one step of the pipeline as a measurable cell.

A step is given the corpus's own answer for everything before it, so what it
scores is its own question. The record it writes is the same shape every other
cell writes, so a step cell and a whole-pipeline cell sit in one table and the
drop between them is read off rather than argued about.

The primary metric is the step's own dimension: whether the entities were
found for identify, whether the slots were named for fillable. There is no
triple F1 for a step that produces no triples, and reporting one would be
reporting the dimension it happened to be derived from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from oold_llm_bench.grading.triples import Dimension
from oold_llm_bench.steps.oracle import plan_of
from oold_llm_bench.steps.score import score_fillable, score_identify

if TYPE_CHECKING:
    from oold_llm_bench.grading.score import Score
    from oold_llm_bench.runner.config import Cell

__all__ = ["STEPS", "StepOutcome", "run_step"]


@dataclass
class StepOutcome:
    """What one step produced and what that was worth."""

    step: str
    produced: Any = None
    dimensions: dict[Dimension, Score] = field(default_factory=dict)
    primary: Dimension = Dimension.ENTITY
    calls: Any = None
    error: str | None = None

    def describe(self) -> dict[str, Any]:
        """The shape a score takes in a record.

        ``primary_f1`` is the step's own dimension and not a triple F1. A step
        that produces no triples has none, and borrowing one would report the
        dimension it was derived from rather than the step.
        """
        scored = self.dimensions.get(self.primary)
        return {
            "task_id": self.step,
            "primary_f1": round(scored.f1, 4) if scored else 0.0,
            "dimensions": {dimension.value: score.describe() for dimension, score in self.dimensions.items()},
        }


def _identify(agent: Any, cell: Cell, request: Any) -> StepOutcome:
    from oold.agent.client import CallLog

    log = CallLog()
    selected, mentions = agent.identify(request, log)
    return StepOutcome(
        step="identify",
        produced={"selected": selected, "mentions": mentions},
        dimensions=score_identify(cell.task, selected, mentions),
        primary=Dimension.ENTITY,
        calls=log,
    )


def _fillable(agent: Any, cell: Cell, request: Any) -> StepOutcome:
    """Step two, handed the true classes and mentions.

    Oracle input on purpose: a step fed a real plan is measured together with
    whatever the plan got wrong, and the two cannot be separated afterwards.
    """
    from oold.agent.client import CallLog

    log = CallLog()
    plan = plan_of(cell.task)
    chosen = agent.fillable_properties(request, plan, log)
    return StepOutcome(
        step="fillable",
        produced=chosen,
        dimensions=score_fillable(cell.task, chosen),
        primary=Dimension.FILLABLE,
        calls=log,
    )


def _extract(agent: Any, cell: Cell, request: Any) -> StepOutcome:
    """Step three, handed the true plan, scored by the whole-answer grader.

    The ids come from the plan and are pinned into the schema, so a link has a
    name to point at. What this measures against the single-shot arm is
    therefore enforced ids against model-minted ones, with the classes held.
    """
    from oold_llm_bench.grading.compare import UnitMatch
    from oold_llm_bench.grading.score import score_task
    from oold_llm_bench.runner.execute import read_answer

    plan = plan_of(cell.task)
    result = agent.extract(request, plan=plan, filling=tuple(entity.key for entity in plan))
    scored = score_task(
        cell.task,
        read_answer(cell, result),
        unit_match=UnitMatch(cell.condition.unit_match),
    )
    return StepOutcome(
        step="extract",
        produced=getattr(result, "payload", None),
        dimensions=dict(scored.dimensions),
        primary=Dimension.VALUE,
        calls=getattr(result, "calls", None),
    )


STEPS = {"identify": _identify, "fillable": _fillable, "extract": _extract}
"""Each step of the pipeline, runnable alone against oracle input.

``dedup`` is absent: its judge is measured by
:mod:`oold_llm_bench.grading.identity` against its own corpus, and its merge
has no ground truth until a sequence corpus exists.
"""


def run_step(step: str, cell: Cell, agent: Any) -> StepOutcome:
    """One step, one cell. Errors are returned rather than raised.

    A step that fell over is a cell that scored nothing, which is a different
    thing from a cell that scored zero, and a grid of hundreds should not stop
    because one of them did.
    """
    if step not in STEPS:
        raise KeyError(f"no step {step!r}; there are {', '.join(STEPS)}")
    from oold_llm_bench.runner.adapter import build_request

    try:
        return STEPS[step](agent, cell, build_request(cell))
    except Exception as exc:
        return StepOutcome(step=step, error=f"{type(exc).__name__}: {exc}")
