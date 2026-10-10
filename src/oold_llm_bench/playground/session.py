"""One run of the playground, from a choice of condition to a scored answer.

The orchestration is the thing this exists to make selectable, so it is a
value on :class:`Options` and never a branch in this module. What an option
builds is decided by :func:`~oold_llm_bench.runner.adapter.build_agent`, which
is where provider knowledge lives; instantiating an agent here would put a
second copy of that knowledge in a tool whose whole point is to show what the
first one does.

Scoring mirrors :func:`~oold_llm_bench.runner.execute.run_cell` rather than
calling it, because a cell outcome keeps the score and drops the result, and
the result is what carries the links, the per-step cost and the degradation
this view is built to show.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from oold_llm_bench.config.models import entry_for, spec_for
from oold_llm_bench.grading.compare import UnitMatch
from oold_llm_bench.grading.score import TaskScore, score_task
from oold_llm_bench.grading.triples import TripleSet
from oold_llm_bench.playground.corpora import ALL_CLASSES, is_scoreable
from oold_llm_bench.playground.graph import Graph, build_graph, links_of
from oold_llm_bench.runner.adapter import build_agent, build_request
from oold_llm_bench.runner.config import Cell, Condition
from oold_llm_bench.runner.execute import read_answer
from oold_llm_bench.tasks.models import TaskRecord

if TYPE_CHECKING:
    from oold.agent.client import ChatClient

__all__ = [
    "ARM_ORDER",
    "ORCHESTRATIONS",
    "Options",
    "Outcome",
    "arm_names",
    "build_cell",
    "model_names",
    "run_once",
]

ORCHESTRATIONS = ("single_shot", "select_then_fill", "segmented", "multi_step")
"""The four that are implemented.

``recursive`` is declared in the agent package and refuses to be built, so
offering it would be offering a choice that raises.
"""

ARM_ORDER = (
    "schema-dump-catalog-flat-enforced",
    "catalog-flat-enforced",
    "schema-dump-catalog-enforced",
    "schema-dump-catalog-not-enforced-gated",
    "schema-dump-catalog-flat-enforced-grounded",
    "no-catalog-not-enforced",
    "no-catalog-not-enforced-prose",
)
"""Arms in the order a person picking one wants them.

The flat-enforced arm first because it is the decode-time arm most cells run.
Its blind form and the union follow because they are the two contrasts
against it: one removes the
prompt copy of the schema, the other replaces the enum with a union, and both
are easier to read off a picture than off a table.
"""


def arm_names() -> tuple[str, ...]:
    """Every arm, preferred order first, then whatever else is declared."""
    from oold.agent.enforcement import ARMS

    known = [name for name in ARM_ORDER if name in ARMS]
    return tuple(known + sorted(set(ARMS) - set(known)))


def model_names() -> tuple[str, ...]:
    """The ladder, as the study names it."""
    from oold_llm_bench.config.models import LADDER

    return tuple(entry.model for entry in LADDER)


@dataclass(frozen=True)
class Options:
    """What a person chose, in the vocabulary the condition already has."""

    orchestration: str = "segmented"
    arm: str = "schema-dump-catalog-flat-enforced"
    model: str = "claude-haiku-4-5"
    shortlist_k: int = 3
    catalogue_set: tuple[str, ...] = (ALL_CLASSES,)
    """The named class sets a pasted document is offered.

    The union of the sets, and never a count. Trimming to a number leaves the
    answer in or out by accident, which is how "Andrea works at ExampleCorp" came
    to be shown 25 classes holding neither Person nor Organization."""
    catalogue_size: int | None = 25
    """How far a corpus task's own catalogue is trimmed.

    Twenty-five because that is what a schema.org cell offers, and a playground
    that silently ran a wider catalogue than the grid would give a reading
    nothing in the grid can be compared to.

    It applies to a corpus task and never to a paste. A trim keeps the classes
    an expectation names and fills the rest from the front, which is right
    where the answer is known; a pasted document's expectation is a
    placeholder, so the trim would cut :attr:`catalogue_set` back to an
    arbitrary 25 and the choice of sets would mean nothing."""
    judge_model: str | None = None
    """The model asked whether two entities are one, or ``None`` for no judge.

    Separate from :attr:`model` on purpose. A judge that is the model under
    test confounds deduplication with extraction, so the two are chosen apart
    and the interface says when they are the same."""
    describe_catalogue: bool = True
    pin_units: bool = False
    """Off, because schema.org declares no unit enumeration. A quantity task
    turns it on, and the condition refuses it where there is nothing to pin."""
    unit_match: str = "exact"
    attempts: int = 1

    def condition(self) -> Condition:
        """The condition these choices are, refusing an impossible pairing."""
        if self.orchestration not in ORCHESTRATIONS:
            raise ValueError(f"unknown orchestration {self.orchestration!r}, expected one of {list(ORCHESTRATIONS)}")
        return Condition(
            arm=self.arm,
            catalogue_size=self.catalogue_size,
            orchestration=self.orchestration,
            shortlist_k=self.shortlist_k,
            describe_catalogue=self.describe_catalogue,
            pin_units=self.pin_units,
            unit_match=self.unit_match,
            output_form="prose" if self.arm == "no-catalog-not-enforced-prose" else "json",
        )

    def label(self) -> str:
        return self.condition().key

    def with_orchestration(self, orchestration: str) -> Options:
        return replace(self, orchestration=orchestration)


def build_cell(task: TaskRecord, options: Options, *, repetition: int = 1) -> Cell:
    """One cell: this condition, this model, this document."""
    return Cell(
        condition=options.condition(),
        model=spec_for(entry_for(options.model)),
        task=task,
        repetition=repetition,
    )


@dataclass
class Outcome:
    """One answer, and everything the view reports beside it."""

    cell: Cell | None
    """None when the interface refused before building one, which is how an
    empty document is reported without raising at the widget."""
    result: Any = None
    produced: TripleSet | None = None
    score: TaskScore | None = None
    graph: Graph | None = None
    sent_schema: dict[str, Any] | None = None
    """The schema the request carried, after the catalogue trim and before the
    provider subset. The subset's own loss is in the degradation."""
    declared_schema: dict[str, Any] | None = None
    shortlist: list[str] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def primary(self) -> float | None:
        return self.score.primary if self.score else None

    def summary(self) -> str:
        if self.error:
            return f"failed: {self.error}"
        counts = self.graph.describe() if self.graph else {}
        parts = [
            f"{counts.get('entities', 0)} entities",
            f"{counts.get('links', 0)} links",
            f"{counts.get('dangling', 0)} dangling",
            f"{counts.get('self_loops', 0)} self-loops",
            f"{counts.get('isolated', 0)} isolated",
        ]
        if self.primary is not None:
            parts.insert(0, f"F1 {self.primary:.3f}")
        return ", ".join(parts)


def run_once(cell: Cell, client: ChatClient, *, attempts: int = 1) -> Outcome:
    """Run one cell against one client and reduce what came back.

    A failure is returned rather than raised. A playground that dies on a
    provider error loses the graph that was already on screen, and the error
    is usually the more interesting of the two.
    """
    outcome = Outcome(cell=cell, declared_schema=cell.task.answer_schema)
    try:
        request = build_request(cell)
        outcome.sent_schema = request.schema
        agent = build_agent(cell, client, attempts=attempts)
        outcome.result = agent.run(request)
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome

    outcome.produced = read_answer(cell, outcome.result)
    links, dangling = links_of(outcome.result)
    # Read off the result and not scored. The mention is what the plan step
    # read the entity from, and it is the only designator there is for an
    # entity whose property step declined `name`, which is the usual answer:
    # a document saying "Jane works at ExampleCorp" does not state that Jane
    # is named Jane. Empty for an orchestration with no plan step.
    mentions = getattr(outcome.result, "mentions", None) or {}
    outcome.graph = build_graph(outcome.produced, links=links, dangling=dangling, mentions=mentions)

    selected = getattr(outcome.result, "selected", None) or {}
    pooled = sorted({name for names in selected.values() for name in names})
    outcome.shortlist = pooled or None
    if is_scoreable(cell.task):
        outcome.score = score_task(
            cell.task,
            outcome.produced,
            unit_match=UnitMatch(cell.condition.unit_match),
            shortlist=outcome.shortlist,
        )
    return outcome
