"""What has to hold before a number is worth reporting.

Conditions on reporting, checked in code rather than left to whoever writes
the results up. Each catches a fault that is invisible in the result it
corrupts: a grader that scores a random agent above zero, a task set every
model saturates, a stable alias that moves underneath a deployment, and a
test split read during tuning.

Nothing here needs credentials. The controls answer without a provider, and
the version check takes what the caller already read from one, so a preflight
can run before any budget is spent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from oold_llm_bench.config.models import drifted
from oold_llm_bench.results.record import Environment
from oold_llm_bench.runner.config import ExperimentConfig
from oold_llm_bench.runner.controls import (
    EmptyClient,
    RandomClient,
    SpellingClient,
    WrongDocumentClient,
)
from oold_llm_bench.runner.execute import run_experiment

__all__ = ["CONTROL_CEILING", "Preflight", "control_scores", "preflight"]

SPELLING_CEILING = 0.25
"""What the spelling control may score and still count as a corpus.

Higher than the others, and deliberately. The other three answer something
wrong, so any score at all is the grader rewarding shape. This one answers the
document without reading a vocabulary, and on real text it will not reach zero:
the hand-authored ELN notes score 0.06, on a single note reading "4.7 ohm"
where the identifier is `ohm`. That is language, not a defect, and no rendering
choice of ours caused it.

What the threshold separates is a corpus where a few readings happen to spell
their own identifier from one where that is how the corpus is written. A
corpus that writes a unit as its identifier scores 0.96 on this control, and
no other control detects it.
"""

CONTROL_CEILING = 0.01
"""What a control may score and still count as zero.

Not exactly zero, because a control that happens to name a plausible unit
should not fail a run outright. Anything above this is the grader rewarding
shape, and the plan makes that a stop.
"""


class _ControlAgent:
    """Runs a control client down the path a model takes.

    The same request, extractor, alignment and grader. A control that
    skipped part of that path would not check the path a model takes.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def run(self, request: Any) -> Any:
        reply = self.client.invoke([])
        try:
            payload = json.loads(reply.text)
        except json.JSONDecodeError:
            payload = None
        return type(
            "ControlResult",
            (),
            {
                "payload": payload,
                "text": reply.text,
                "calls": None,
                "degradation": None,
                "schema_sha256": None,
            },
        )()


@dataclass
class Preflight:
    """Every check, and whether anything blocks."""

    controls: dict[str, float] = field(default_factory=dict)
    saturated: list[str] = field(default_factory=list)
    drift: dict[str, tuple[str, str]] = field(default_factory=dict)
    blocked: list[str] = field(default_factory=list)

    @property
    def clear(self) -> bool:
        return not self.blocked

    def describe(self) -> dict[str, Any]:
        return {
            "clear": self.clear,
            "controls": self.controls,
            "saturated": self.saturated,
            "drift": {k: list(v) for k, v in self.drift.items()},
            "blocked": self.blocked,
        }


def _answers_for(task: Any) -> list[dict[str, Any]]:
    """A task's own answer, in the shape the JSON extractor reads.

    Every field, not only ``value``. Reading ``value`` alone makes this
    control degenerate on any corpus whose instances do not carry one: a
    schema.org task would answer ``{"value": null, "unit": null}``, which
    cannot score whatever the grader does.

    A link is written as the target nested inside its source, which is the
    only shape the extractor reads as an edge. Flattening it to the target's
    key would answer with a string no document contains, and the control would
    be unable to fail on the one dimension a linked corpus exists to measure.

    A cycle is nested once and then left alone, so every instance reaches the
    answer exactly once. A control that returns nothing cannot fail, which is
    the same defect as a control that returns only nulls.

    A property holding several links is nested value by value, because a
    ``Reference`` is not JSON-serialisable.
    """
    bodies = {instance.key: _body_of(instance) for instance in task.expected}
    parent: dict[str, str] = {}

    def nestable(source: str, target: str) -> bool:
        """Whether nesting ``target`` under ``source`` keeps this a forest."""
        if target not in bodies or target in parent or target == source:
            return False
        walk: str | None = source
        while walk is not None:
            if walk == target:
                return False
            walk = parent.get(walk)
        return True

    for instance in task.expected:
        for name, value in instance.fields.items():
            targets = list(value) if isinstance(value, (list, tuple)) else [value]
            keys = [k for k in (getattr(v, "key", None) for v in targets) if k is not None]
            taken = [k for k in keys if nestable(instance.key, k)]
            if not taken:
                continue
            for k in taken:
                parent[k] = instance.key
            nested = [bodies[k] for k in taken]
            bodies[instance.key][name] = nested if isinstance(value, (list, tuple)) else nested[0]

    return [body for key, body in bodies.items() if key not in parent]


def _body_of(instance: Any) -> dict[str, Any]:
    """One instance as JSON, with its links left for the caller to fill in."""
    answer: dict[str, Any] = {"type": instance.class_path}
    for name, value in instance.fields.items():
        magnitude = getattr(value, "magnitude", None)
        if magnitude is not None:
            # A quantity carries its unit beside it, the shape the extractor
            # folds back into one triple.
            answer[name] = magnitude
            answer["unit"] = getattr(value, "unit", None)
        elif getattr(value, "key", None) is None:
            answer[name] = value
    return answer


def _wrong_document_agent(config: ExperimentConfig, cell: Any) -> _ControlAgent:
    """Answer some other document in the grid, never this one.

    Picking the answers once for the whole run is the mistake to avoid here.
    A fixed answer is the right answer whenever the cell happens to be the
    task it came from, so the control would report the grid size.
    """
    other = next(t for t in config.tasks if t.id != cell.task.id)
    return _ControlAgent(WrongDocumentClient(_answers_for(other)))


def _spelling_agent(cell: Any) -> _ControlAgent:
    """Copy the number and the words beside it, and underscore them.

    Given the expected class on purpose. Class is a separate dimension, and
    withholding it would let a class failure disguise how much of the value
    dimension a reader recovers without knowing any vocabulary.
    """
    expected = cell.task.expected[0].class_path if cell.task.expected else ""
    return _ControlAgent(SpellingClient(cell.task.document, expected))


def control_scores(config: ExperimentConfig, environment: Environment) -> dict[str, float]:
    """What each negative control scores on this task set.

    The third one is the strongest control. It answers a different document of the
    same corpus correctly, so every value it gives is a real value with a
    real unit and a real class, and anything it scores comes from the grader
    matching shape.
    """
    catalogue = list(config.tasks[0].catalogue or ()) if config.tasks else []
    factories = {
        "empty": lambda cell: _ControlAgent(EmptyClient()),
        "random": lambda cell: _ControlAgent(RandomClient(catalogue, seed=0)),
        "spelling": _spelling_agent,
    }
    if len(config.tasks) > 1:
        factories["wrong-document"] = lambda cell: _wrong_document_agent(config, cell)

    scores: dict[str, float] = {}
    for name, factory in factories.items():
        run = run_experiment(
            config,
            factory,
            environment,
            make_request=lambda cell: cell.task.id,
        )
        scores[name] = max((run.mean_primary(k) for k in run.by_condition()), default=0.0)
    return scores


def preflight(
    config: ExperimentConfig,
    environment: Environment,
    *,
    live_versions: dict[str, str] | None = None,
    run_controls: bool = True,
) -> Preflight:
    """Check everything that can be checked before spending a budget.

    A blocked preflight stops the run. The plan makes each of these a
    condition on reporting, so a caller that runs anyway is reporting a
    number it cannot defend.

    Saturation is not checked here, because it needs results. It is reported
    by :meth:`~oold_llm_bench.runner.execute.ExperimentRun.saturated` once a
    run exists, and the same rule applies to it.
    """
    result = Preflight()

    if run_controls:
        result.controls = control_scores(config, environment)
        if "wrong-document" not in result.controls:
            result.blocked.append(
                "the grid has one task, so the wrong-document control cannot be "
                "built and the sharpest check on the grader is missing"
            )
        for name, score in result.controls.items():
            ceiling = SPELLING_CEILING if name == "spelling" else CONTROL_CEILING
            if score > ceiling:
                result.blocked.append(
                    f"the {name} control scored {score:.3f}, above {ceiling}, "
                    f"so the grader is rewarding shape and no cell is interpretable"
                )

    if live_versions is not None:
        result.drift = drifted(live_versions)
        for model, (pinned, reported) in result.drift.items():
            result.blocked.append(f"{model} is pinned at {pinned} and the provider now reports {reported}")
        for spec in config.models:
            if spec.model not in live_versions:
                result.blocked.append(f"{spec.model} is in the grid but the provider does not offer it")

    if config.split.value != "dev":
        result.blocked.append(
            f"the grid runs on the {config.split.value} split, which is held out and "
            f"read once, so this needs a deliberate decision and not a default"
        )

    return result
