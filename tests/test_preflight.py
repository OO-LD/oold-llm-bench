"""The gate that runs before a budget is spent.

Each check here stands for something that went wrong once, so the tests are
written as the failure, not as the happy path.
"""

import json

import pytest

from oold_llm_bench.corpus.quantities import Notation, QuantityKind, generate_task
from oold_llm_bench.corpus.schemaorg import Kind, Link, SchemaClass, Slot
from oold_llm_bench.corpus.schemaorg import generate_task as schemaorg_task
from oold_llm_bench.results.record import Environment, ModelSpec
from oold_llm_bench.runner import Condition, ExperimentConfig
from oold_llm_bench.runner.preflight import CONTROL_CEILING, _answers_for, control_scores, preflight
from oold_llm_bench.tasks.models import CorpusRef, ExpectedInstance, Source, Split, TaskRecord

KINDS = [
    QuantityKind(name="Length", units=("meter",)),
    QuantityKind(name="Mass", units=("gram",)),
    QuantityKind(name="Duration", units=("second",)),
]
"""Units with a published symbol, so the grid can be rendered as a document
writes it. Under the canonical rendering the spelling control scores 1.00 and
preflight blocks, which is the point of having it."""
CATALOGUE = tuple(k.name for k in KINDS)


def _linkable(name: str, target: str | None = None) -> SchemaClass:
    """One class with three slots of its own, optionally pointing at another."""
    return SchemaClass(
        name=name,
        links=(Link(name=f"{name.lower()}Of", ranges=(target,)),) if target else (),
        slots=(
            Slot(name=f"{name.lower()}Name", kind=Kind.TEXT),
            Slot(name=f"{name.lower()}Count", kind=Kind.INTEGER),
            Slot(name=f"{name.lower()}On", kind=Kind.DATE),
        ),
    )


ONE_PAIR = [_linkable("Book", "Person"), _linkable("Person")]
"""The smallest corpus that carries an edge, and too small to be a grid."""

FOUR_PAIRS = [
    *ONE_PAIR,
    _linkable("Course", "School"),
    _linkable("School"),
    _linkable("Ticket", "Venue"),
    _linkable("Venue"),
    _linkable("Award", "Judge"),
    _linkable("Judge"),
]
"""Wide enough that two documents of a grid rarely take the same class pair."""


def environment() -> Environment:
    return Environment(benchmark_version="0.1.0", benchmark_sha="abc")


def grid(split: Split = Split.DEV, models: list[ModelSpec] | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        name="preflight",
        conditions=[Condition(arm="schema-dump-catalog-flat-enforced")],
        models=models or [ModelSpec(model="gpt-5-nano", provider_profile="openai", model_version="2025-08-07")],
        tasks=[
            generate_task(KINDS, task_id=f"t{i}", seed=i, split=split, catalogue=CATALOGUE, notation=Notation.WRITTEN)
            for i in range(3)
        ],
        runs_per_cell=1,
        split=split,
    )


class TestTheControls:
    def test_every_control_scores_zero_on_a_real_task_set(self):
        scores = control_scores(grid(), environment())
        assert set(scores) == {"empty", "random", "spelling", "wrong-document"}
        assert all(s <= CONTROL_CEILING for s in scores.values())

    def test_a_clean_grid_passes(self):
        assert preflight(grid(), environment()).clear

    def test_the_control_scores_are_recorded_not_just_judged(self):
        """The number is recorded, not just the verdict. Nobody can audit a
        verdict.
        """
        assert preflight(grid(), environment()).describe()["controls"]["random"] == 0.0


class TestVersionDrift:
    def test_a_matching_version_passes(self):
        result = preflight(grid(), environment(), live_versions={"gpt-5-nano": "2025-08-07"})
        assert result.clear

    def test_a_moved_alias_blocks_the_run(self):
        result = preflight(grid(), environment(), live_versions={"gpt-5-nano": "2026-02-01"})
        assert not result.clear
        assert "pinned at 2025-08-07" in result.blocked[0]

    def test_a_model_the_provider_no_longer_offers_blocks_the_run(self):
        result = preflight(grid(), environment(), live_versions={"something-else": "1"})
        assert not result.clear
        assert "does not offer it" in " ".join(result.blocked)

    def test_versions_are_only_checked_when_the_caller_supplies_them(self):
        assert preflight(grid(), environment()).drift == {}


class TestHeldOutDiscipline:
    def test_a_grid_on_the_test_split_blocks(self):
        result = preflight(grid(split=Split.TEST), environment())
        assert not result.clear
        assert "held out" in " ".join(result.blocked)

    def test_the_dev_split_does_not_block(self):
        assert "held out" not in " ".join(preflight(grid(), environment()).blocked)


class TestWhatItReports:
    def test_the_result_is_serialisable(self):
        import json

        assert json.dumps(preflight(grid(), environment()).describe())

    def test_several_problems_are_all_reported(self):
        """Fixing one and rerunning to find the next wastes a run each time."""
        result = preflight(
            grid(split=Split.TEST),
            environment(),
            live_versions={"gpt-5-nano": "2026-02-01"},
        )
        assert len(result.blocked) >= 2

    def test_controls_can_be_skipped_when_they_have_already_run(self):
        result = preflight(grid(), environment(), run_controls=False)
        assert result.controls == {}
        assert result.clear


class TestTheCeilingItself:
    def test_the_ceiling_is_not_exactly_zero(self):
        """A control naming one plausible unit should not fail a whole run."""
        assert 0.0 < CONTROL_CEILING < 0.1

    def test_a_grader_that_rewards_shape_would_be_caught(self):
        """The check has to be able to fire, or it is decoration."""
        from oold_llm_bench.runner.preflight import Preflight

        result = Preflight(controls={"random": 0.4})
        assert result.controls["random"] > CONTROL_CEILING


@pytest.mark.parametrize("control", ["empty", "random", "wrong-document"])
def test_no_control_scores_on_any_task_in_the_grid(control):
    assert control_scores(grid(), environment())[control] == 0.0


class TestTheSpellingControl:
    """The control that was missing, and what it caught.

    Three controls scored 0.00 and the quantity grid was declared
    interpretable on that basis. This one reads nothing: it finds a number,
    takes the words beside it, replaces the spaces with underscores, and
    submits that as the unit. On the corpus as it was rendered until
    2026-09-28 it scored a perfect primary on 115 of 120 tasks.

    So the primary metric was a spelling convention, and no other control
    could say so, because each of the other three answers something wrong.
    This one answers the document without understanding any of it.
    """

    def _scores(self, notation):
        config = ExperimentConfig(
            name="spelling",
            conditions=[Condition(arm="schema-dump-catalog-flat-enforced")],
            models=[ModelSpec(model="gpt-5-nano", provider_profile="openai", model_version="2025-08-07")],
            tasks=[
                generate_task(KINDS, task_id=f"t{i}", seed=i, catalogue=CATALOGUE, notation=notation) for i in range(6)
            ],
            runs_per_cell=1,
        )
        return control_scores(config, environment())

    def test_it_defeats_the_canonical_rendering(self):
        assert self._scores(Notation.CANONICAL)["spelling"] > 0.9

    def test_it_scores_nothing_once_units_are_written(self):
        assert self._scores(Notation.WRITTEN)["spelling"] <= CONTROL_CEILING

    def test_the_other_controls_cannot_see_it(self):
        """Why three zeros were not enough.

        Each of the other controls answers a wrong document, a random class or
        nothing at all, so all three score zero under either rendering and
        neither rendering is distinguishable from the other by them.
        """
        canonical = self._scores(Notation.CANONICAL)
        for name in ("empty", "random", "wrong-document"):
            assert canonical[name] <= CONTROL_CEILING

    def test_a_grid_it_defeats_does_not_pass_preflight(self):
        config = ExperimentConfig(
            name="spelling",
            conditions=[Condition(arm="schema-dump-catalog-flat-enforced")],
            models=[ModelSpec(model="gpt-5-nano", provider_profile="openai", model_version="2025-08-07")],
            tasks=[
                generate_task(KINDS, task_id=f"t{i}", seed=i, catalogue=CATALOGUE, notation=Notation.CANONICAL)
                for i in range(6)
            ],
            runs_per_cell=1,
        )
        result = preflight(config, environment())
        assert not result.clear
        assert any("spelling" in reason for reason in result.blocked)


class TestTheWrongDocumentControlCanActuallyFail:
    """A control that cannot score is not a control.

    It read `fields["value"]` and nothing else, so on a corpus whose instances
    carry no such field it answered `{"value": null, "unit": null}`. That
    cannot score whatever the grader does, and four schema.org grids were
    reported as controls-clear on it.

    It now answers with every field the other document carries, so a grader
    rewarding shape would be caught. It still scores 0.00 there, which is what
    makes those grids defensible rather than merely unchallenged.
    """

    def test_it_answers_with_every_field_not_only_value(self):
        task = generate_task(KINDS, task_id="t1", seed=1, catalogue=CATALOGUE, notation=Notation.WRITTEN)
        (answer,) = _answers_for(task)
        assert answer["type"] == task.expected[0].class_path
        assert answer.get("value") is not None

    def test_an_instance_with_no_value_field_still_answers(self):
        """The shape that made it degenerate."""
        instance = ExpectedInstance(key="e1", class_path="Book", fields={"isbn": "978-0-00-000000-0"})
        task = TaskRecord(
            id="t1",
            document="A book.",
            expected=[instance],
            corpus=CorpusRef(source=Source.SYNTHETIC, document_id="d1", content_hash="0" * 64),
            split=Split.DEV,
        )
        (answer,) = _answers_for(task)
        assert answer["isbn"] == "978-0-00-000000-0"
        assert set(answer) > {"type"}


class TestTheControlsOnALinkedGrid:
    """Entities that point at one another, and four agents that cannot read.

    A flat answer carries no edge, so a control that emitted one would be
    unable to fail on the dimension a linked corpus exists to measure. The
    wrong-document control therefore writes its links the way a correct answer
    writes them, and what that exposes is the test below.
    """

    def grid(self, classes) -> ExperimentConfig:
        return ExperimentConfig(
            name="linked",
            conditions=[Condition(arm="schema-dump-catalog-flat-enforced")],
            models=[ModelSpec(model="gpt-5-nano", provider_profile="openai", model_version="2025-08-07")],
            tasks=[
                schemaorg_task(
                    classes,
                    task_id=f"t{i}",
                    seed=i,
                    n_entities=2,
                    n_slots=3,
                    linked=True,
                    catalogue=tuple(cls.name for cls in classes),
                    describe_catalogue=True,
                )
                for i in range(6)
            ],
            runs_per_cell=1,
        )

    def test_the_wrong_document_control_answers_the_edge_too(self):
        """Flattening it would leave the control unable to fail on a link."""
        task = self.grid(ONE_PAIR).tasks[0]
        (answer,) = _answers_for(task)
        assert isinstance(answer["bookOf"], dict)
        assert answer["bookOf"]["type"] == task.expected[1].class_path

    def test_a_pool_of_one_pair_lets_a_wrong_document_answer_the_edge(self):
        """The constraint a linked grid is under, as the control that catches it.

        Every document here is a Book pointing at a Person, so the alignment
        puts each of a wrong document's entities on the entity of its own
        class, and the edge between them is right without anything having been
        read. Measured on the schema.org corpus, whose 62 source classes give
        397 drawable pairs: a wrong document's edge reaches the right entity in
        5 of 200 draws, and the pairwise floor is zero in 195 of 200.
        """
        scores = control_scores(self.grid(ONE_PAIR), environment())
        assert scores["wrong-document"] > CONTROL_CEILING
        assert not preflight(self.grid(ONE_PAIR), environment()).clear

    def test_a_pool_that_varies_its_pairs_clears(self):
        scores = control_scores(self.grid(FOUR_PAIRS), environment())
        assert set(scores) == {"empty", "random", "spelling", "wrong-document"}
        assert all(score <= CONTROL_CEILING for score in scores.values())
        assert preflight(self.grid(FOUR_PAIRS), environment()).clear

    def test_the_other_three_cannot_see_the_difference(self):
        """They answer a random class, or nothing, so no pool changes them."""
        narrow = control_scores(self.grid(ONE_PAIR), environment())
        for name in ("empty", "random", "spelling"):
            assert narrow[name] == 0.0


class TestTheWrongDocumentControlCanAlwaysFail:
    """Shapes that made the control answer nothing, or crash.

    A control that cannot fail reports "clear" on a corpus nobody checked.
    That has now happened twice in this repo, so each shape gets a test
    rather than a comment.
    """

    @staticmethod
    def _task(instances):
        import types

        return types.SimpleNamespace(expected=instances)

    @staticmethod
    def _person(key, **fields):
        from oold_llm_bench.tasks.models import ExpectedInstance

        return ExpectedInstance(key=key, class_path="Person", fields=fields)

    def test_a_mutual_link_still_answers(self):
        from oold_llm_bench.grading import Reference
        from oold_llm_bench.runner.preflight import _answers_for

        bodies = _answers_for(
            self._task([
                self._person("a", knows=Reference(key="b")),
                self._person("b", knows=Reference(key="a")),
            ])
        )
        assert bodies, "both entities were pointed at, so the control answered nothing"
        json.dumps(bodies)

    def test_a_cycle_of_three_still_answers(self):
        from oold_llm_bench.grading import Reference
        from oold_llm_bench.runner.preflight import _answers_for

        bodies = _answers_for(
            self._task([
                self._person("a", knows=Reference(key="b")),
                self._person("b", knows=Reference(key="c")),
                self._person("c", knows=Reference(key="a")),
            ])
        )
        assert bodies
        json.dumps(bodies)

    def test_a_property_holding_several_links_serialises(self):
        from oold_llm_bench.grading import Reference
        from oold_llm_bench.runner.preflight import _answers_for

        bodies = _answers_for(
            self._task([
                self._person("a", knows=[Reference(key="b"), Reference(key="c")]),
                self._person("b", name="B"),
                self._person("c", name="C"),
            ])
        )
        assert bodies
        json.dumps(bodies), "a Reference reached json.dumps and killed the run"
