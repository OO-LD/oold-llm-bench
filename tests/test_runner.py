"""Declaring a study, running it, and proving the grader can fail.

The control tests are the ones that matter. If an agent answering a different
document scores above zero, nothing else measured here means anything.
"""

import pytest

from oold_llm_bench.corpus.quantities import QuantityKind, generate_task
from oold_llm_bench.extract import extract_json
from oold_llm_bench.grading.score import score_task
from oold_llm_bench.results.record import Environment, ModelSpec
from oold_llm_bench.runner import (
    Condition,
    EmptyClient,
    ExperimentConfig,
    RandomClient,
    WrongDocumentClient,
    run_experiment,
)
from oold_llm_bench.tasks.models import Split

KINDS = [
    QuantityKind(name="Length", units=("meter",)),
    QuantityKind(name="Mass", units=("gram",)),
    QuantityKind(name="Duration", units=("second",)),
]
CATALOGUE = [k.name for k in KINDS]


def tasks(n: int = 2, split: Split = Split.DEV):
    return [generate_task(KINDS, task_id=f"t{i}", seed=i, split=split) for i in range(n)]


def model(name: str = "fake-1") -> ModelSpec:
    return ModelSpec(model=name, provider_profile="openai", model_version="1")


def environment() -> Environment:
    return Environment(benchmark_version="0.1.0", benchmark_sha="abc123")


class PerfectAgent:
    """Answers every task correctly. The ceiling the controls sit under."""

    def __init__(self, task_by_id):
        self.task_by_id = task_by_id

    def run(self, request):
        task = self.task_by_id[request]
        return type(
            "Result",
            (),
            {
                "payload": {
                    "entities": [
                        {
                            "type": i.class_path,
                            "value": i.fields["value"].magnitude,
                            "unit": i.fields["value"].unit,
                        }
                        for i in task.expected
                    ]
                },
                "calls": None,
                "degradation": None,
                "schema_sha256": None,
                "text": "",
            },
        )()


class ClientAgent:
    """Wraps a control client so it travels the same path a model does."""

    def __init__(self, client):
        self.client = client

    def run(self, request):
        reply = self.client.invoke([])
        import json

        try:
            payload = json.loads(reply.text)
        except json.JSONDecodeError:
            payload = None
        return type(
            "Result",
            (),
            {
                "payload": payload,
                "calls": None,
                "degradation": None,
                "schema_sha256": None,
                "text": reply.text,
            },
        )()


def config(
    conditions: list[Condition] | None = None,
    models: list[ModelSpec] | None = None,
    task_list: list | None = None,
    runs_per_cell: int = 2,
) -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke",
        conditions=[Condition(arm="schema-dump-catalog-flat-enforced")] if conditions is None else conditions,
        models=[model()] if models is None else models,
        tasks=tasks() if task_list is None else task_list,
        runs_per_cell=runs_per_cell,
    )


class TestTheGrid:
    def test_the_cell_count_is_the_product(self):
        grid = config(
            conditions=[
                Condition(arm="schema-dump-catalog-not-enforced-gated"),
                Condition(arm="schema-dump-catalog-flat-enforced"),
            ],
            models=[model("a"), model("b")],
            task_list=tasks(3),
            runs_per_cell=5,
        )
        assert grid.size == 2 * 2 * 3 * 5
        assert len(list(grid.cells())) == grid.size

    def test_cells_come_out_in_a_stable_order(self):
        grid = config()
        assert [c.key for c in grid.cells()] == [c.key for c in grid.cells()]

    def test_a_condition_key_names_what_varies(self):
        key = Condition(
            arm="schema-dump-catalog-flat-enforced", catalogue_size=25, signal="translated", language="de"
        ).key
        assert key == "schema-dump-catalog-flat-enforced/translated/consensus/n25/de"

    def test_two_conditions_differing_only_in_the_embedding_keep_two_keys(self):
        """Four axes were once missing from this key, so two conditions wrote
        into one jsonl and collapsed to one row. An answer shape with a slot an
        entity can be written into is a fifth."""
        plain = Condition(arm="schema-dump-catalog-flat-enforced")
        embedded = Condition(arm="schema-dump-catalog-flat-enforced", embed_nested=True)
        assert plain.key != embedded.key
        assert embedded.key.endswith("embedded")

    def test_runs_per_cell_is_fixed_before_running(self):
        """Choosing it after seeing results turns noise into a finding."""
        assert config().describe()["runs_per_cell"] == 2

    def test_the_grid_is_hashed(self):
        assert len(config().config_sha256) == 64

    def test_changing_the_grid_changes_the_hash(self):
        assert config().config_sha256 != config(runs_per_cell=3).config_sha256

    def test_an_empty_grid_is_refused(self):
        with pytest.raises(ValueError, match="at least one condition"):
            config(conditions=[])
        with pytest.raises(ValueError, match="at least one model"):
            config(models=[])

    def test_a_task_from_the_wrong_split_is_refused(self):
        """Held-out tasks do not wander into a dev run by accident."""
        with pytest.raises(ValueError, match="not all in the dev split"):
            config(task_list=tasks(1, split=Split.TEST))


class TestRunning:
    def run_with(self, agent_factory, grid=None):
        grid = grid or config()
        by_id = {t.id: t for t in grid.tasks}
        return run_experiment(
            grid,
            agent_factory,
            environment(),
            make_request=lambda cell: cell.task.id,
        ), by_id

    def test_a_perfect_agent_scores_one(self):
        grid = config()
        by_id = {t.id: t for t in grid.tasks}
        run, _ = self.run_with(lambda cell: PerfectAgent(by_id), grid)
        assert len(run.outcomes) == grid.size
        assert all(o.ok for o in run.outcomes)
        assert run.mean_primary("schema-dump-catalog-flat-enforced/unit/consensus") == pytest.approx(1.0)

    def test_every_outcome_carries_a_record(self):
        grid = config()
        by_id = {t.id: t for t in grid.tasks}
        run, _ = self.run_with(lambda cell: PerfectAgent(by_id), grid)
        record = run.outcomes[0].record
        assert record is not None
        assert record.arm == "schema-dump-catalog-flat-enforced"
        assert len(record.corpus_hash) == 64

    def test_a_published_record_carries_no_account_identifier(self):
        grid = config(
            models=[
                ModelSpec(
                    model="m",
                    provider_profile="openai",
                    deployment="my-deployment",
                    subscription="secret",
                )
            ]
        )
        by_id = {t.id: t for t in grid.tasks}
        run, _ = self.run_with(lambda cell: PerfectAgent(by_id), grid)
        import json

        published = json.dumps(run.outcomes[0].record.publish())
        assert "my-deployment" not in published
        assert "secret" not in published

    def test_an_agent_that_raises_is_recorded_not_fatal(self):
        class Broken:
            def run(self, request):
                raise RuntimeError("provider said no")

        run, _ = self.run_with(lambda cell: Broken())
        assert len(run.failures()) == len(run.outcomes)
        assert "provider said no" in run.outcomes[0].error

    def test_one_broken_cell_does_not_lose_the_grid(self):
        grid = config()
        by_id = {t.id: t for t in grid.tasks}
        seen = {"n": 0}

        def agent_for(cell):
            seen["n"] += 1
            if seen["n"] == 1:

                class Broken:
                    def run(self, request):
                        raise RuntimeError("transient")

                return Broken()
            return PerfectAgent(by_id)

        run, _ = self.run_with(agent_for, grid)
        assert len(run.outcomes) == grid.size
        assert len(run.failures()) == 1

    def test_the_run_is_serialisable(self):
        import json

        grid = config()
        by_id = {t.id: t for t in grid.tasks}
        run, _ = self.run_with(lambda cell: PerfectAgent(by_id), grid)
        assert json.dumps(run.describe())

    def test_the_catalogue_hash_names_the_catalogue_the_condition_offered(self):
        """One task supplies two catalogues, so a hash taken from the task
        would report the embedding pair as having run against one."""
        task = tasks(1)[0].model_copy(
            update={"catalogue": CATALOGUE, "embedded_branches": {}, "value_objects": ["Mass"]}
        )
        grid = config(
            conditions=[
                Condition(arm="schema-dump-catalog-flat-enforced"),
                Condition(arm="schema-dump-catalog-flat-enforced", embed_nested=True),
            ],
            task_list=[task],
            runs_per_cell=1,
        )
        by_id = {t.id: t for t in grid.tasks}
        run, _ = self.run_with(lambda cell: PerfectAgent(by_id), grid)
        hashes = {o.cell.condition.embed_nested: o.record.catalogue_hash for o in run.outcomes}
        assert hashes[False] != hashes[True]


class TestTheControls:
    """If any of these scores, nothing else measured here means anything."""

    def run_control(self, client):
        grid = config(runs_per_cell=1)
        run = run_experiment(
            grid,
            lambda cell: ClientAgent(client),
            environment(),
            make_request=lambda cell: cell.task.id,
        )
        return run

    def test_an_empty_answer_scores_zero(self):
        assert self.run_control(EmptyClient()).mean_primary("schema-dump-catalog-flat-enforced/unit/consensus") == 0.0

    def test_a_random_answer_scores_zero(self):
        run = self.run_control(RandomClient(CATALOGUE, seed=1))
        assert run.mean_primary("schema-dump-catalog-flat-enforced/unit/consensus") == 0.0

    def test_answering_a_different_document_scores_zero(self):
        """Every value is real. Anything it scores is the grader rewarding shape."""
        other = generate_task(KINDS, task_id="other", seed=9999)
        answers = [
            {
                "type": i.class_path,
                "value": i.fields["value"].magnitude,
                "unit": i.fields["value"].unit,
            }
            for i in other.expected
        ]
        run = self.run_control(WrongDocumentClient(answers))
        assert run.mean_primary("schema-dump-catalog-flat-enforced/unit/consensus") == 0.0

    def test_a_control_travels_the_same_path_a_model_does(self):
        """Not a shortcut: same extractor, same alignment, same score."""
        reply = RandomClient(CATALOGUE, seed=2).invoke([])
        import json

        produced = extract_json(json.loads(reply.text))
        task = tasks(1)[0]
        assert score_task(task, produced).primary == 0.0


class TestTheProseArm:
    """A0-prose is read by the prose parser, not by the JSON one."""

    def run_prose(self, answer: str):
        class ProseAgent:
            def run(self, request):
                return type(
                    "Result",
                    (),
                    {
                        "payload": None,
                        "calls": None,
                        "degradation": None,
                        "schema_sha256": None,
                        "text": answer,
                    },
                )()

        grid = config(
            conditions=[Condition(arm="no-catalog-not-enforced-prose", output_form="prose")],
            task_list=tasks(1),
            runs_per_cell=1,
        )
        return run_experiment(
            grid,
            lambda cell: ProseAgent(),
            environment(),
            make_request=lambda cell: cell.task.id,
        )

    def test_a_prose_answer_is_scored(self):
        task = tasks(1)[0]
        value = task.expected[0].fields["value"]
        answer = f"The reading was {value.magnitude} {value.unit.replace('_', ' ')}."
        assert self.run_prose(answer).mean_primary("no-catalog-not-enforced-prose/unit/consensus/prose") > 0.0

    def test_a_prose_answer_naming_the_wrong_value_scores_zero(self):
        assert (
            self.run_prose("The reading was 99999.0 parsec.").mean_primary(
                "no-catalog-not-enforced-prose/unit/consensus/prose"
            )
            == 0.0
        )

    def test_an_output_form_nobody_implements_is_refused(self):
        with pytest.raises(ValueError, match="json or prose"):
            Condition(arm="A0", output_form="yaml")


class TestDiscipline:
    def test_saturation_is_reported(self):
        grid = config()
        by_id = {t.id: t for t in grid.tasks}
        run = run_experiment(
            grid,
            lambda cell: PerfectAgent(by_id),
            environment(),
            make_request=lambda cell: cell.task.id,
        )
        assert run.saturated() == ["schema-dump-catalog-flat-enforced/unit/consensus"]

    def test_a_control_run_is_not_saturated(self):
        assert self.__class__ and True
        grid = config(runs_per_cell=1)
        run = run_experiment(
            grid,
            lambda cell: ClientAgent(EmptyClient()),
            environment(),
            make_request=lambda cell: cell.task.id,
        )
        assert run.saturated() == []

    def test_test_touches_start_at_zero(self):
        grid = config()
        by_id = {t.id: t for t in grid.tasks}
        run = run_experiment(
            grid,
            lambda cell: PerfectAgent(by_id),
            environment(),
            make_request=lambda cell: cell.task.id,
        )
        assert run.test_touches == 0
        assert run.describe()["test_touches"] == 0


class TestRunningConcurrently:
    """Cells are independent, so they can run at once."""

    def grid(self):
        return config(task_list=tasks(4), runs_per_cell=2)

    def test_concurrent_and_serial_agree(self):
        grid = self.grid()
        by_id = {t.id: t for t in grid.tasks}
        serial = run_experiment(
            grid,
            lambda c: PerfectAgent(by_id),
            environment(),
            make_request=lambda c: c.task.id,
        )
        concurrent = run_experiment(
            grid,
            lambda c: PerfectAgent(by_id),
            environment(),
            make_request=lambda c: c.task.id,
            workers=4,
        )
        assert [o.cell.key for o in concurrent.outcomes] == [o.cell.key for o in serial.outcomes]
        assert concurrent.mean_primary("schema-dump-catalog-flat-enforced/unit/consensus") == serial.mean_primary(
            "schema-dump-catalog-flat-enforced/unit/consensus"
        )

    def test_outcomes_keep_the_declared_order(self):
        """Not the order providers answered in, or two runs stop comparing."""
        grid = self.grid()
        by_id = {t.id: t for t in grid.tasks}
        run = run_experiment(
            grid,
            lambda c: PerfectAgent(by_id),
            environment(),
            make_request=lambda c: c.task.id,
            workers=4,
        )
        assert [o.cell.key for o in run.outcomes] == [c.key for c in grid.cells()]

    def test_every_outcome_is_reported_once(self):
        grid = self.grid()
        by_id = {t.id: t for t in grid.tasks}
        seen = []
        run_experiment(
            grid,
            lambda c: PerfectAgent(by_id),
            environment(),
            make_request=lambda c: c.task.id,
            workers=4,
            on_outcome=seen.append,
        )
        assert len(seen) == grid.size

    def test_one_failing_cell_does_not_lose_the_others(self):
        grid = self.grid()
        by_id = {t.id: t for t in grid.tasks}

        class Broken:
            def run(self, request):
                raise RuntimeError("provider said no")

        calls = {"n": 0}

        def agent_for(cell):
            calls["n"] += 1
            return Broken() if calls["n"] == 3 else PerfectAgent(by_id)

        run = run_experiment(grid, agent_for, environment(), make_request=lambda c: c.task.id, workers=4)
        assert len(run.outcomes) == grid.size
        assert len(run.failures()) == 1

    def test_test_touches_are_counted_once(self):
        grid = self.grid()
        by_id = {t.id: t for t in grid.tasks}
        run = run_experiment(
            grid,
            lambda c: PerfectAgent(by_id),
            environment(),
            make_request=lambda c: c.task.id,
            workers=4,
        )
        assert run.test_touches == 0
