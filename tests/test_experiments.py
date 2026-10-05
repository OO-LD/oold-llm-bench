"""The grids a published figure rests on, and the path from one to a record.

Every number this benchmark reports came from a grid. A grid that lives in a
scratch script is a grid nobody else can run, so these pin that the declared
ones resolve, that their rungs are the names the tables use, and that a run
reaches a record and a report without a provider.
"""

from __future__ import annotations

import importlib.util

import pytest

from oold_llm_bench.corpus.quantities import Notation, QuantityKind, generate_task
from oold_llm_bench.experiments import GRIDS, report, run_grid
from oold_llm_bench.experiments.corpora import balance
from oold_llm_bench.experiments.grids import Grid
from oold_llm_bench.results.record import Environment, ModelSpec
from oold_llm_bench.runner import Condition

KINDS = [
    QuantityKind(name="Length", units=("meter", "kilo_meter")),
    QuantityKind(name="Mass", units=("gram", "kilo_gram")),
]
needs_agent = pytest.mark.skipif(
    importlib.util.find_spec("oold") is None,
    reason="running a grid builds agents from the library, which is the `agent` extra",
)
"""Marks a test that runs a grid rather than only declaring one.

Declaring and reading a grid needs nothing but the core, which is the point
of the extra being optional. Running one reaches preflight, and preflight
builds control agents from the library like any other cell.
"""

MODEL = "gpt-5-nano"
"""A real catalogue entry, because a spec is built before any call is made.
No call is made here: the agent is injected."""


class TestTheDeclaredGrids:
    def test_every_rung_has_a_published_name(self):
        """A rung nobody can name is a row nobody can cite."""
        for grid in GRIDS.values():
            assert grid.rungs()

    def test_a_grid_names_each_rung_once(self):
        """Two rungs sharing a label pool into one row, and the comparison the
        grid exists to make disappears into a mean."""
        for name, grid in GRIDS.items():
            labels = grid.rungs()
            assert len(set(labels)) == len(labels), f"{name} repeats a rung: {labels}"

    def test_the_context_ladder_holds_its_enforcement_fixed(self):
        """Its whole claim is that the prompt moved and the grammar did not,
        so every rung but the floor enforces the same union."""
        rungs = GRIDS["context-ladder"].rungs()
        assert rungs[0] == "no-catalog-not-enforced"
        assert all(rung.endswith("-enforced") for rung in rungs[1:])

    def test_a_rung_can_be_run_on_its_own(self):
        """The top rung is 45,000 input tokens per call and the floor is 81.
        A reader who wants the floor should not pay for the rest."""
        grid = GRIDS["context-ladder"].select(("no-catalog-enforced",))
        assert grid.rungs() == ["no-catalog-enforced"]
        assert len(grid.conditions) == 1

    def test_an_unknown_rung_names_the_ones_there_are(self):
        with pytest.raises(KeyError, match="no-catalog-enforced"):
            GRIDS["context-ladder"].select(("no-such-rung",))

    def test_selecting_nothing_keeps_the_whole_grid(self):
        grid = GRIDS["context-ladder"]
        assert grid.select(()).rungs() == grid.rungs()


class TestBalancing:
    """An unbalanced pool makes a mean a report on whichever class the corpus
    collected most of."""

    def _tasks(self):
        return [generate_task(KINDS, task_id=f"t{i}", seed=i, n_entities=1) for i in range(12)]

    def test_each_class_contributes_at_most_the_cap(self):
        pooled = balance(self._tasks(), 2)
        counts = {}
        for task in pooled:
            counts[task.expected[0].class_path] = counts.get(task.expected[0].class_path, 0) + 1
        assert set(counts.values()) <= {1, 2}

    def test_a_named_class_that_is_absent_raises(self):
        """Silently dropping it would run a ladder on three classes at one
        rung and four at the next."""
        with pytest.raises(ValueError, match="Nonexistent"):
            balance(self._tasks(), 2, ("Length", "Nonexistent"))


class PerfectAgent:
    """Answers each task correctly, so the path is what is under test."""

    def run(self, request):
        task = request.task if hasattr(request, "task") else None
        return type(
            "Result",
            (),
            {
                "payload": {"entities": self._entities(task)},
                "calls": None,
                "degradation": None,
                "schema_sha256": None,
                "text": "",
            },
        )()

    @staticmethod
    def _entities(task):
        if task is None:
            return []
        return [
            {
                "type": instance.class_path,
                "value": instance.fields["value"].magnitude,
                "unit": instance.fields["value"].unit,
            }
            for instance in task.expected
        ]


@needs_agent
def test_a_grid_whose_corpus_spells_its_answers_is_refused(tmp_path):
    """Preflight is consulted before a cell is billed, not after.

    A corpus that writes a unit as its identifier lets a control that copies
    the words beside the number score 1.00, and no cell measured on it means
    anything.
    """
    tasks = [generate_task(KINDS, task_id=f"t{i}", seed=i, n_entities=1) for i in range(4)]
    grid = Grid(
        name="spelled",
        summary="a corpus the spelling control can read off the page",
        conditions=(Condition(arm="no-catalog-not-enforced", catalogue_size=2, signal="named"),),
        tasks=lambda per_class, schemas: balance(tasks, per_class),
        per_class=2,
        needs_schemas=False,
        workers=1,
    )
    with pytest.raises(RuntimeError, match="spelling control"):
        run_grid(grid, [MODEL], out=tmp_path, agents=lambda spec: PerfectAgent())


@needs_agent
def test_a_grid_runs_to_a_record_and_a_report(tmp_path):
    """The claim the whole package exists for: a declaration reaches a record.

    No provider and no schema module, because what is being pinned is the
    path, not the answers.
    """
    tasks = [generate_task(KINDS, task_id=f"t{i}", seed=i, n_entities=1, notation=Notation.WRITTEN) for i in range(4)]
    grid = Grid(
        name="smoke",
        summary="one rung, four tasks",
        conditions=(Condition(arm="no-catalog-not-enforced", catalogue_size=2, signal="named"),),
        tasks=lambda per_class, schemas: balance(tasks, per_class),
        per_class=2,
        needs_schemas=False,
        workers=1,
    )
    result = run_grid(
        grid,
        [MODEL],
        out=tmp_path,
        agents=lambda spec: PerfectAgent(),
    )
    assert result.cells == len(grid.conditions) * len(balance(tasks, 2))
    assert result.records.exists()
    rendered = report(result.published, ("class", "value", "unit"))
    assert "no-catalog-not-enforced" in rendered


def test_the_environment_records_the_revision_a_run_was_made_at():
    """A result that does not name the code that made it cannot be rerun: the
    grader, the catalogue rendering and the arm registry all move."""
    from oold_llm_bench.experiments.run import _revision, _version

    assert _version()
    assert _revision()


def test_a_model_spec_is_what_the_record_carries():
    assert ModelSpec(model="x", provider_profile="openai").describe()["model"] == "x"
    assert Environment(benchmark_version="0.1.0", benchmark_sha="abc").benchmark_sha == "abc"
