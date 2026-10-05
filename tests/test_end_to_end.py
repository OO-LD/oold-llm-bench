"""Corpus to task to score, with no model involved.

These pin the seam between generation and grading. A generator that emits an
answer the grader cannot recognise would otherwise look fine on both sides
and score every arm at zero.
"""

import pytest

from oold_llm_bench.corpus.quantities import QuantityKind, generate_task
from oold_llm_bench.grading import Dimension, Quantity, TripleSet, make_triple, make_triples
from oold_llm_bench.grading.compare import UnitMatch
from oold_llm_bench.grading.score import score_task
from oold_llm_bench.tasks import Difficulty

KINDS = [
    QuantityKind(name="Length", units=("meter", "centi_meter", "kilo_meter")),
    QuantityKind(name="Mass", units=("gram", "kilo_gram")),
    QuantityKind(name="Temperature", units=("degree_Celsius", "kelvin")),
]


def perfect_answer(task) -> TripleSet:
    """What a model that read the document correctly would produce."""
    triples = frozenset()
    classes = {}
    for instance in task.expected:
        triples |= make_triples(instance.key, instance.fields)
        classes[instance.key] = instance.class_path
    return TripleSet(triples=triples, classes=classes, provenance={})


class TestTheSeamCloses:
    @pytest.mark.parametrize("difficulty", [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD])
    def test_a_perfect_answer_scores_one(self, difficulty):
        task = generate_task(KINDS, task_id="t1", seed=11, difficulty=difficulty, n_entities=3)
        result = score_task(task, perfect_answer(task))
        assert result.primary == pytest.approx(1.0)
        assert result.dimensions[Dimension.CLASS].f1 == pytest.approx(1.0)
        assert result.dimensions[Dimension.UNIT].f1 == pytest.approx(1.0)

    def test_an_equivalent_unit_still_scores_one(self):
        """Physical equality is reported, but it is not the primary metric.

        The corpus closes the unit slot, so the answer is the name the corpus
        uses. Conversion is a different answer that denotes the same quantity,
        and the gap between the two is a dimension rather than a free pass.
        """
        task = generate_task([QuantityKind(name="Length", units=("meter",))], task_id="t1", seed=1)
        expected = task.expected[0].fields["value"]
        converted = Quantity(magnitude=expected.magnitude * 100, unit="centi_meter")
        answer = TripleSet(
            triples=make_triples("a", {"value": converted}),
            classes={"a": "Length"},
            provenance={},
        )
        assert score_task(task, answer).primary == pytest.approx(0.0)
        physical = score_task(task, answer, unit_match=UnitMatch.PHYSICAL)
        assert physical.primary == pytest.approx(1.0)

    def test_a_task_that_asks_for_normalisation_accepts_a_rescaled_answer(self):
        """The task says what a correct answer is, not the grader.

        Only units convert. A closed enumeration of class identifiers has no
        equivalent, so this stays a unit-only special case.
        """
        task = generate_task([QuantityKind(name="Length", units=("meter",))], task_id="t1", seed=1)
        task = task.model_copy(update={"accepts_conversion": True})
        expected = task.expected[0].fields["value"]
        converted = Quantity(magnitude=expected.magnitude * 100, unit="centi_meter")
        answer = TripleSet(
            triples=make_triples("a", {"value": converted}),
            classes={"a": "Length"},
            provenance={},
        )
        assert score_task(task, answer).primary == pytest.approx(1.0)

    def test_an_ordinary_task_still_refuses_a_rescaled_answer(self):
        task = generate_task([QuantityKind(name="Length", units=("meter",))], task_id="t1", seed=1)
        assert task.accepts_conversion is False

    def test_a_wrong_unit_is_caught(self):
        """The whole reason this corpus leads: unit errors are detectable."""
        task = generate_task([QuantityKind(name="Mass", units=("gram",))], task_id="t1", seed=1)
        expected = task.expected[0].fields["value"]
        answer = TripleSet(
            triples=make_triples("a", {"value": Quantity(magnitude=expected.magnitude, unit="kilo_gram")}),
            classes={"a": "Mass"},
            provenance={},
        )
        result = score_task(task, answer)
        assert result.primary == 0.0

    def test_a_right_number_with_no_unit_does_not_pass(self):
        task = generate_task([QuantityKind(name="Mass", units=("gram",))], task_id="t1", seed=1)
        expected = task.expected[0].fields["value"]
        answer = TripleSet(
            triples=make_triples("a", {"value": expected.magnitude}),
            classes={"a": "Mass"},
            provenance={},
        )
        assert score_task(task, answer).primary == 0.0

    def test_the_right_class_with_the_wrong_number_separates(self):
        task = generate_task([QuantityKind(name="Length", units=("meter",))], task_id="t1", seed=1)
        answer = TripleSet(
            triples=make_triples("a", {"value": Quantity(magnitude=999999.0, unit="meter")}),
            classes={"a": "Length"},
            provenance={},
        )
        result = score_task(task, answer)
        assert result.dimensions[Dimension.CLASS].f1 == pytest.approx(1.0)
        assert result.dimensions[Dimension.VALUE].f1 == 0.0

    def test_an_answer_from_a_different_document_scores_zero(self):
        """The negative control, on generated tasks instead of fixtures."""
        task = generate_task(KINDS, task_id="t1", seed=12, n_entities=2)
        other = generate_task(KINDS, task_id="t2", seed=9999, n_entities=2)
        assert score_task(task, perfect_answer(other)).primary == 0.0

    def test_an_empty_answer_scores_zero(self):
        task = generate_task(KINDS, task_id="t1", seed=13, n_entities=2)
        empty = TripleSet(triples=frozenset(), classes={}, provenance={})
        assert score_task(task, empty).primary == 0.0


class TestOrderDoesNotMatter:
    """A list of quantity values has no subject, so order carries nothing."""

    def test_answering_in_a_different_order_scores_the_same(self):
        task = generate_task(KINDS, task_id="t1", seed=21, n_entities=3)
        forward = perfect_answer(task)
        reversed_keys = {instance.key: f"r{index}" for index, instance in enumerate(reversed(task.expected), start=1)}
        shuffled = TripleSet(
            triples=frozenset(make_triple(reversed_keys[t.entity], t.prop, t.value) for t in forward.triples),
            classes={reversed_keys[k]: v for k, v in forward.classes.items()},
            provenance={},
        )
        assert score_task(task, shuffled).primary == pytest.approx(1.0)
