"""The grids, declared.

A grid's conditions are the part that must not move: they are what the figure
is about, and changing one makes the result a different result under the same
name. Everything else, the models, the task count, the endpoint, is an
argument, because it is what a reader supplies.

Each condition is labelled by :func:`~oold_llm_bench.report.axes.label_of`, so
a grid's rungs can be selected and read by the names the published tables use
rather than by the arm ids in the records.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from oold_llm_bench.runner import Condition
from oold_llm_bench.runner.arms import UNION_BLIND_ARM, UNION_ENFORCED_ARM

if TYPE_CHECKING:
    from oold_llm_bench.tasks.models import TaskRecord

__all__ = ["GRIDS", "Grid"]

# The arm names below are strings and need no registry to be declared. The
# registry is the library's, so registering here would make importing a grid
# require the agent extra, which the grader and the corpus deliberately do
# not. `run_grid` registers before it builds an agent.

_LADDER_CLASSES = ("Altitude", "Area", "Length", "Diameter")
"""Four kinds, held across every rung.

The ladder's whole claim is that the context changed and nothing else did, so
the tasks have to be the same tasks at every rung.
"""


@dataclass(frozen=True)
class Grid:
    """One published comparison, and how to build the tasks it runs on."""

    name: str
    summary: str
    conditions: tuple[Condition, ...]
    tasks: Callable[[int, Path | None], list[TaskRecord]]
    """Given a per-class count and a resolved schema directory, the task set."""
    per_class: int
    dimensions: tuple[str, ...] = ("class", "value", "unit")
    needs_schemas: bool = True
    workers: int = 4
    notes: str = ""
    step: str | None = None
    """Which step of the pipeline this grid runs alone, or the whole arm.

    A step grid hands the step the corpus's own answer for everything before
    it, so what it scores is its own question. Named here rather than on the
    condition, because it decides how a cell is run and not what the model is
    asked."""

    def rungs(self) -> list[str]:
        """The published name of each condition, in declared order."""
        from oold_llm_bench.report.axes import label_of

        return [label_of(c.arm, c.describe()) for c in self.conditions]

    def select(self, wanted: tuple[str, ...]) -> Grid:
        """The same grid with only the named rungs.

        A reader who wants one row of a table should not have to pay for the
        context-heavy rungs beside it: the ladder's top rung is 45,000 input
        tokens per call and its floor is 81.
        """
        if not wanted:
            return self
        labels = self.rungs()
        unknown = [name for name in wanted if name not in labels]
        if unknown:
            raise KeyError(f"{self.name} has no rung {', '.join(unknown)}; it has {', '.join(labels)}")
        kept = tuple(c for c, label in zip(self.conditions, labels, strict=True) if label in wanted)
        return replace(self, conditions=kept)


def _quantity(per_class: int, schemas: Path | None) -> list[TaskRecord]:
    from oold_llm_bench.experiments.corpora import quantity_tasks

    if schemas is None:
        raise ValueError("a quantity grid needs a schema module; run `oold-bench fetch-corpus quantities`")
    return quantity_tasks(schemas, per_class, _LADDER_CLASSES)


def _wikidata(per_class: int, schemas: Path | None) -> list[TaskRecord]:
    from oold_llm_bench.experiments.corpora import wikidata_schemaorg_tasks

    return wikidata_schemaorg_tasks(per_class)


def _synthetic(per_class: int, schemas: Path | None) -> list[TaskRecord]:
    """Generated entities over the same ten classes :func:`_wikidata` draws
    from real leads, so a grid can run the same step on both and read the gap
    as the corpus's, not the catalogue's. ``schemas`` is accepted, not
    ``needs_schemas``: that flag resolves the quantities module by name, and
    a second corpus needing its own would have nothing to resolve it with."""
    from oold_llm_bench.experiments.corpora import synthetic_schemaorg_tasks

    return synthetic_schemaorg_tasks(per_class, schemas)


def _wikidata_own_words(per_class: int, schemas: Path | None) -> list[TaskRecord]:
    """The same tasks, with each slot described as Wikidata describes it.

    Ground truth comes from Wikidata's statements and the step is shown
    schema.org's words, and the two vocabularies disagree about what a slot
    means. Wikidata's `creator` is "maker of this creative work (where no
    more specific property exists)", which excludes a film that has a
    director; schema.org's permits it. Its `contentLocation` is where the
    narrative is set, not where the painting hangs.

    So this is not a prompt variant. It asks whether the step was being given
    one definition and graded against another.
    """
    import json

    from oold_llm_bench.corpus.wikidata_schemaorg import CORPUS_PATH

    path = CORPUS_PATH.parent / "wikidata_property_text.json"
    described = json.loads(path.read_text(encoding="utf-8"))
    return [task.model_copy(update={"property_text": described}) for task in _wikidata(per_class, schemas)]


_CONTEXT_LADDER = (
    # Nothing at all, and no grammar either. The floor, and what says what the
    # catalogue is worth: never told a class name, the base model scores zero.
    Condition(arm="no-catalog-not-enforced", catalogue_size=100, signal="named", reasoning="off"),
    # The same 81 tokens, with a decoder that admits one branch per class. The
    # model is still never shown a class name.
    Condition(
        arm=UNION_BLIND_ARM,
        catalogue_size=100,
        signal="named",
        pin_units=True,
        describe_catalogue=False,
        reasoning="off",
    ),
    # The catalogue, each class with the units it admits, no prose.
    Condition(
        arm=UNION_ENFORCED_ARM,
        catalogue_size=100,
        signal="named",
        pin_units=True,
        describe_catalogue=False,
        reasoning="off",
    ),
    # The same plus descriptions, five thousand tokens more.
    Condition(
        arm=UNION_ENFORCED_ARM,
        catalogue_size=100,
        signal="named",
        pin_units=True,
        describe_catalogue=True,
        reasoning="off",
    ),
    # The union printed out in full, which is the same constraint at four
    # times the tokens the prose catalogue states it in.
    Condition(
        arm="schema-dump-catalog-enforced",
        catalogue_size=100,
        signal="named",
        pin_units=True,
        describe_catalogue=True,
        reasoning="off",
    ),
)

_ENFORCEMENT_ABLATION = (
    # Presented and not enforced.
    Condition(
        arm="catalog-not-enforced-gated",
        catalogue_size=100,
        signal="named",
        pin_units=True,
        describe_catalogue=False,
        reasoning="off",
    ),
    # Presented and enforced, at the same prompt length. The difference is the
    # grammar and nothing else.
    Condition(
        arm="catalog-flat-enforced",
        catalogue_size=100,
        signal="named",
        pin_units=True,
        describe_catalogue=False,
        reasoning="off",
    ),
    # The schema printed on top of a catalogue that already states the
    # enumerations, which is what the token cost of restating buys.
    Condition(
        arm="schema-dump-catalog-flat-enforced",
        catalogue_size=100,
        signal="named",
        pin_units=True,
        describe_catalogue=False,
        reasoning="off",
    ),
)

_STEP_CONDITIONS = (
    # The class schema and the range enums bind whatever the prompt carries:
    # presenting the catalogue is a condition on a step, knowing it is not.
    # `reasoning` is left unset. Turning thinking off is a vLLM control sent
    # as chat_template_kwargs, which Azure rejects outright, and a step grid
    # should run wherever the model does.
    Condition(
        arm=UNION_ENFORCED_ARM,
        catalogue_size=25,
        signal="named",
        describe_catalogue=True,
        orchestration="multi_step",
        shortlist_k=3,
    ),
)

_SCHEMAORG_UNION = tuple(
    # pin_units is meaningless here: schema.org declares no unit slot, so
    # there is nothing for the enum to close. Left at the default and said
    # rather than set, because a reader coming from the quantity grids will
    # look for it.
    Condition(arm=arm, catalogue_size=25, signal="named", describe_catalogue=True, reasoning="off")
    for arm in ("catalog-flat-enforced", UNION_ENFORCED_ARM)
)

GRIDS: dict[str, Grid] = {
    grid.name: grid
    for grid in (
        Grid(
            name="context-ladder",
            summary="One grammar, the prompt walked from 45,000 tokens down to 81",
            conditions=_CONTEXT_LADDER,
            tasks=_quantity,
            per_class=30,
            notes=(
                "The enforcement is identical in four of the five rungs, so what the table reads "
                "is what the prompt is still doing once the grammar already forbids a wrong answer. "
                "Run it twice, base and tuned, to read what an adapter pays for."
            ),
        ),
        Grid(
            name="enforcement-ablation",
            summary="Presenting a schema against enforcing it, at one prompt length",
            conditions=_ENFORCEMENT_ABLATION,
            tasks=_quantity,
            per_class=30,
        ),
        Grid(
            name="step-identify",
            summary="Which entities a document holds, and what it calls them",
            conditions=_STEP_CONDITIONS,
            tasks=_wikidata,
            per_class=20,
            dimensions=("class", "mention", "entity:recall"),
            needs_schemas=False,
            workers=12,
            step="identify",
            notes=(
                "Real text, where the corpus records the subject alone. Entity precision is left out: "
                "a lead names the subject's founder and its city too, and counting those as inventions "
                "measures how much the document says."
            ),
        ),
        Grid(
            name="step-fillable",
            summary="Which slots the document fills, given the true class",
            conditions=_STEP_CONDITIONS,
            tasks=_wikidata,
            per_class=20,
            dimensions=("fillable", "fillable:precision", "fillable:recall"),
            needs_schemas=False,
            workers=12,
            step="fillable",
            notes="Handed the true classes and mentions, so a wrong plan cannot be blamed for it.",
        ),
        Grid(
            name="step-fillable-evidence",
            summary="The same step, made to quote the words it read each property from",
            conditions=(replace(_STEP_CONDITIONS[0], property_evidence=True),),
            tasks=_wikidata,
            per_class=20,
            dimensions=("fillable", "fillable:precision", "fillable:recall"),
            needs_schemas=False,
            workers=12,
            step="fillable",
            notes="Scores lower than step-fillable on five models of six. Run to read what the step quotes.",
        ),
        Grid(
            name="step-fillable-wikidata",
            summary="The same step, each slot described as the vocabulary the truth came from describes it",
            conditions=_STEP_CONDITIONS,
            tasks=_wikidata_own_words,
            per_class=20,
            dimensions=("fillable", "fillable:precision", "fillable:recall"),
            needs_schemas=False,
            workers=12,
            step="fillable",
            notes="Against step-fillable, which shows schema.org's words for the same slots.",
        ),
        Grid(
            name="step-fillable-synthetic",
            summary="The same step, on generated entities where every stated fact is known",
            conditions=_STEP_CONDITIONS,
            tasks=_synthetic,
            per_class=20,
            dimensions=("fillable", "fillable:precision", "fillable:recall"),
            needs_schemas=False,
            workers=12,
            step="fillable",
            notes=(
                "Against step-fillable: a generated document states exactly its entity's known facts, "
                "so precision is read against ground truth and not against Wikidata's coverage of it."
            ),
        ),
        Grid(
            name="step-extract",
            summary="The entity itself, given the true plan and ids pinned to it",
            conditions=_STEP_CONDITIONS,
            tasks=_wikidata,
            per_class=20,
            dimensions=("class", "property", "value"),
            needs_schemas=False,
            workers=12,
            step="extract",
            notes="Ids come from the plan and are pinned, so a link has a name to point at.",
        ),
        Grid(
            name="multi-step-chain",
            summary="Detect, choose properties, extract: the whole orchestration, no oracle",
            conditions=_STEP_CONDITIONS,
            tasks=_wikidata,
            per_class=20,
            dimensions=("class", "property", "value"),
            needs_schemas=False,
            workers=12,
            notes=(
                "The same arm and the same tasks as step-identify, step-fillable and "
                "step-extract, run end to end instead of one at a time against oracle "
                "input. The drop against the three step grids is what a wrong plan or a "
                "wrong property choice costs once nothing downstream corrects it."
            ),
        ),
        Grid(
            name="multi-step-chain-synthetic",
            summary="The same chain, on generated entities where every stated fact is known",
            conditions=_STEP_CONDITIONS,
            tasks=_synthetic,
            per_class=20,
            dimensions=("class", "property", "value"),
            needs_schemas=False,
            workers=12,
            notes=(
                "Against multi-step-chain: a generated document states exactly its "
                "entity's known facts, so a drop here is the chain's own error "
                "compounding and not Wikidata's coverage showing up a second time."
            ),
        ),
        Grid(
            name="schemaorg-union",
            summary="Flat enumeration against a discriminated union, on human prose",
            conditions=_SCHEMAORG_UNION,
            tasks=_wikidata,
            per_class=40,
            dimensions=("class", "value", "property"),
            needs_schemas=False,
            # Two, not four. One llama.cpp process serves one model and a
            # described catalogue is a large prefill: four concurrent calls
            # completed no cell in forty minutes where a single call answered
            # in 18.7 seconds.
            workers=2,
        ),
    )
}
