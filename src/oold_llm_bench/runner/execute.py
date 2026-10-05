"""Running the cells and recording what happened.

The runner owns no judgement. It builds the condition the config declares,
asks the agent, reads the answer with the shared extractor, scores it with
the shared grader, and writes a record that names every one of those choices.
Anything it decided on its own would be a variable nobody declared.

Nothing here mutates the environment. The predecessor wrote each model's
configuration into ``os.environ`` before calling it, which is why it kept a
snapshot of the original environment to undo the damage and why two arms
could never run at the same time. A model is passed in as a value instead.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from oold_llm_bench.extract import extract_json, extract_prose
from oold_llm_bench.grading.compare import UnitMatch
from oold_llm_bench.grading.score import TaskScore, score_task
from oold_llm_bench.grading.triples import TripleSet
from oold_llm_bench.results.record import Environment, RunRecord, config_hash
from oold_llm_bench.runner.config import Cell, ExperimentConfig
from oold_llm_bench.runner.training_match import TrainingMatch, match_of

__all__ = ["CellOutcome", "ExperimentRun", "run_cell", "run_experiment"]


class Agent(Protocol):
    """What the runner needs from whatever answers a document."""

    def run(self, request: Any) -> Any: ...


@dataclass
class CellOutcome:
    """One cell: what came back, what it scored, and what went wrong."""

    cell: Cell
    score: TaskScore | None = None
    record: RunRecord | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.score is not None

    @property
    def primary(self) -> float:
        return self.score.primary if self.score else 0.0


@dataclass
class ExperimentRun:
    """Every outcome, plus the discipline checks the plan requires."""

    config: ExperimentConfig
    outcomes: list[CellOutcome] = field(default_factory=list)
    test_touches: int = 0
    """How many times the held-out split was read. Always reported: an
    unstated number is one nobody checks."""

    def by_condition(self) -> dict[str, list[CellOutcome]]:
        grouped: dict[str, list[CellOutcome]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.cell.condition.key, []).append(outcome)
        return grouped

    def mean_primary(self, condition_key: str) -> float:
        cells = [o for o in self.by_condition().get(condition_key, []) if o.ok]
        return sum(o.primary for o in cells) / len(cells) if cells else 0.0

    def saturated(self, ceiling: float = 0.99) -> list[str]:
        """Conditions where every model is at the ceiling.

        A cell with no headroom cannot show an effect, so reporting one is
        reporting the task set instead of the arms.
        """
        return [key for key in self.by_condition() if self.mean_primary(key) >= ceiling]

    def failures(self) -> list[CellOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def error_kinds(self) -> dict[str, int]:
        """How the failed cells failed, counted by exception type.

        Reported with every run. A mean over the cells that came back, with no
        count of the ones that did not, is a mean over whatever the provider
        felt like serving: one sweep here lost 40 of 120 cells to rate limiting
        and the summary said nothing.
        """
        kinds: dict[str, int] = {}
        for outcome in self.failures():
            kind = str(outcome.error or "").split(":", 1)[0] or "unknown"
            kinds[kind] = kinds.get(kind, 0) + 1
        return kinds

    def describe(self) -> dict[str, Any]:
        return {
            "config": self.config.describe(),
            "config_sha256": self.config.config_sha256,
            "test_touches": self.test_touches,
            "n_outcomes": len(self.outcomes),
            "n_failures": len(self.failures()),
            "error_kinds": self.error_kinds(),
            "saturated_conditions": self.saturated(),
            "mean_primary": {key: round(self.mean_primary(key), 4) for key in self.by_condition()},
        }


def read_answer(cell: Cell, result: Any) -> TripleSet:
    """Turn what came back into triples, using the extractor the arm declares.

    A prose arm is read by the prose parser, whose own loss is measured by
    :func:`~oold_llm_bench.extract.parse_loss` and has to be reported beside
    any score it produces.
    """
    if cell.condition.output_form == "prose":
        return extract_prose(getattr(result, "text", "") or "")

    payload = getattr(result, "payload", None)
    if payload is None:
        return TripleSet(triples=frozenset(), classes={}, provenance={}, parse_errors=1)
    return extract_json(payload)


def run_cell(
    cell: Cell,
    agent: Agent,
    environment: Environment,
    *,
    make_request: Callable[[Cell], Any],
    subclasses: dict[str, set[str]] | None = None,
) -> CellOutcome:
    """Run one cell and score it.

    A cell that raises is recorded as a failure and the run continues. Losing
    the rest of a grid to one provider error would cost more than the cell.
    """
    try:
        result = agent.run(make_request(cell))
    except Exception as exc:
        return CellOutcome(cell=cell, error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    produced = read_answer(cell, result)
    # Pooled across entities, because one union covers the whole fill call.
    selected = getattr(result, "selected", None) or {}
    shortlist = sorted({name for names in selected.values() for name in names}) or None
    score = score_task(
        cell.task,
        produced,
        subclasses=subclasses,
        unit_match=UnitMatch(cell.condition.unit_match),
        shortlist=shortlist,
    )

    calls = getattr(result, "calls", None)
    degradation = getattr(result, "degradation", None)

    record = RunRecord(
        run_id=cell.key,
        arm=cell.condition.arm,
        model=cell.model,
        environment=environment,
        enforcement=cell.condition.describe(),
        corpus_hash=cell.task.corpus.content_hash,
        catalogue_hash=config_hash(cell.task.catalogue or []),
        # The assembled turns when the agent hands them back, the document
        # otherwise. Folding changes the bytes one model receives, so the
        # document alone would say two models were sent the same prompt.
        prompt_hash=getattr(result, "prompt_sha256", None) or config_hash(cell.task.document),
        schema_hash=getattr(result, "schema_sha256", None),
        split=cell.task.split.value,
        variant=cell.task.variant.value,
        repetition=cell.repetition,
        scores=[score.describe()],
        calls=calls.describe() if calls is not None else {},
        degradation=degradation.describe() if degradation is not None else {},
        selected={key: list(names) for key, names in selected.items()},
        document_chars=len(cell.task.document),
        links=[edge.describe() for edge in getattr(result, "links", ()) or ()],
        dangling=[edge.describe() for edge in getattr(result, "dangling", ()) or ()],
        unpinned=list(getattr(result, "unpinned", ()) or ()),
        invalid=list(getattr(result, "invalid", ()) or ()),
        repairs=int(getattr(result, "repairs", 0) or 0),
        answer_text=getattr(result, "text", None),
        answer_payload=getattr(result, "payload", None),
        notes=cell.task.notes,
        training_match=_training_match_of(cell).describe(),
    )
    return CellOutcome(cell=cell, score=score, record=record)


def _training_match_of(cell: Cell) -> TrainingMatch:
    """Whether this cell asks the model the question it was trained on.

    Resolved here rather than by the caller, because the caller is whoever
    wrote a run script and the whole point is that it should not be theirs to
    remember. A model the catalogue does not know is a base model.
    """
    from oold_llm_bench.config.models import entry_for

    try:
        entry = entry_for(cell.model.model)
    except (KeyError, ValueError):
        return TrainingMatch(status="base")
    trained_on = getattr(entry, "trained_on", None)
    if not trained_on:
        return TrainingMatch(status="base")
    # The corpus comes from the task, because the condition has no idea which
    # corpus it is running over and the model's record names one. Compared
    # here rather than left undeclared: an adapter tuned on generated prose
    # and scored on human prose is being asked to cross a register as well as
    # recall a vocabulary, and one number cannot separate the two.
    wanted = trained_on.get("corpus")
    found = getattr(cell.task.corpus, "name", "") or ""
    if wanted and found and wanted != found:
        return TrainingMatch(status="mismatched", differing=("corpus",))
    rest = {k: v for k, v in trained_on.items() if k != "corpus"} if wanted and found else trained_on
    return match_of(rest, cell.condition)


def run_experiment(
    config: ExperimentConfig,
    agent_for: Callable[[Cell], Agent],
    environment: Environment,
    *,
    make_request: Callable[[Cell], Any],
    subclasses: dict[str, set[str]] | None = None,
    on_outcome: Callable[[CellOutcome], None] | None = None,
    workers: int = 1,
) -> ExperimentRun:
    """Run every cell in the declared order.

    ``agent_for`` builds the agent a cell needs, which is where a model and a
    condition become a configured client. Keeping it a callback means the
    runner never learns what a provider is.

    ``workers`` above one runs cells concurrently. Outcomes are still appended
    in the declared order, so two runs of one config stay comparable and a
    result never depends on which provider answered first. A cell holds no
    shared state: the agent is built per cell and the model is a value, which
    is what the environment mutation this runner replaced made impossible.
    """
    run = ExperimentRun(config=config)
    cells = list(config.cells())
    run.test_touches = sum(1 for cell in cells if cell.task.split is not config.split)

    if workers <= 1:
        for cell in cells:
            outcome = run_cell(cell, agent_for(cell), environment, make_request=make_request, subclasses=subclasses)
            run.outcomes.append(outcome)
            if on_outcome is not None:
                on_outcome(outcome)
        return run

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_cell,
                cell,
                agent_for(cell),
                environment,
                make_request=make_request,
                subclasses=subclasses,
            )
            for cell in cells
        ]
        for future in futures:
            outcome = future.result()
            run.outcomes.append(outcome)
            if on_outcome is not None:
                on_outcome(outcome)
    return run
