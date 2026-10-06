"""Scoring one step of the pipeline against the answer the corpus holds.

A step fed real upstream output is measured together with every mistake made
before it. These pin that each step can be handed the corpus's own answer and
scored on its own question, which is what makes a drop attributable.
"""

from __future__ import annotations

from oold_llm_bench.grading.triples import Dimension
from oold_llm_bench.steps import fillable_of, mentions_of, score_fillable, score_identify, shortlist_of
from oold_llm_bench.tasks.models import CorpusRef, ExpectedInstance, Source, Split, TaskRecord


def task(*instances: ExpectedInstance) -> TaskRecord:
    return TaskRecord(
        id="t1",
        document="Ada works for Acme.",
        expected=list(instances),
        catalogue=("Person", "Organization"),
        corpus=CorpusRef(source=Source.SYNTHETIC, document_id="d1", content_hash="0" * 64),
        split=Split.DEV,
    )


def person(key: str = "p1", **fields) -> ExpectedInstance:
    return ExpectedInstance(key=key, class_path="Person", fields=fields or {"name": "Ada"})


class TestTheOracle:
    """Every stage is already in the task record. Nothing here invents truth."""

    def test_the_shortlist_is_the_expected_class(self):
        assert shortlist_of(task(person())) == {"p1": ("Person",)}

    def test_the_fillable_set_is_what_the_document_states(self):
        got = fillable_of(task(person(name="Ada", jobTitle="engineer")))
        assert set(got["p1"]) == {"name", "jobTitle"}

    def test_an_optional_slot_is_left_out_unless_asked_for(self):
        """The step is asked which properties the document fills, and an
        optional slot the document did not fill is not one of them."""
        instance = ExpectedInstance(
            key="p1", class_path="Person", fields={"name": "Ada"}, optional_fields={"award": "x"}
        )
        assert set(fillable_of(task(instance))["p1"]) == {"name"}
        assert set(fillable_of(task(instance), optional=True)["p1"]) == {"name", "award"}

    def test_a_corpus_with_no_mentions_does_not_hand_over_blanks(self):
        assert mentions_of(task(person())) == {"p1": "p1"}


class TestScoringIdentify:
    def test_the_right_entity_under_the_right_class_scores_one(self):
        out = score_identify(task(person()), {"e1": ("Person",)})
        assert out[Dimension.ENTITY].true_positives == 1
        assert out[Dimension.CLASS].true_positives == 1

    def test_an_invented_entity_costs_precision_not_recall(self):
        out = score_identify(task(person()), {"e1": ("Person",), "e2": ("Organization",)})
        assert out[Dimension.ENTITY].false_positives == 1
        assert out[Dimension.ENTITY].false_negatives == 0

    def test_a_missed_entity_costs_recall_not_precision(self):
        out = score_identify(task(person("p1"), person("p2")), {"e1": ("Person",)})
        assert out[Dimension.ENTITY].false_negatives == 1
        assert out[Dimension.ENTITY].false_positives == 0

    def test_the_wrong_class_is_a_class_failure_and_not_a_missing_entity(self):
        """Found under the wrong class is a different failure from not found,
        and the step after this one fails differently for each."""
        instance = ExpectedInstance(key="p1", class_path="Person", fields={"name": "Ada"}, mentions=("Ada",))
        out = score_identify(task(instance), {"e1": ("Organization",)}, {"e1": "Ada"})
        assert out[Dimension.ENTITY].true_positives == 1
        assert out[Dimension.CLASS].true_positives == 0

    def test_the_key_the_step_chose_does_not_matter(self):
        instance = ExpectedInstance(key="p1", class_path="Person", fields={"name": "Ada"}, mentions=("Ada",))
        out = score_identify(task(instance), {"whatever": ("Person",)}, {"whatever": "Ada"})
        assert out[Dimension.ENTITY].true_positives == 1


class TestScoringTheMention:
    def _task(self):
        return task(ExpectedInstance(key="p1", class_path="Person", fields={"name": "Ada"}, mentions=("Ada Lovelace",)))

    def test_a_shorter_form_of_the_same_name_counts(self):
        """A document calling her both has not been misread by a step that
        answers one where the corpus recorded the other."""
        out = score_identify(self._task(), {"e1": ("Person",)}, {"e1": "Ada"})
        assert out[Dimension.MENTION].true_positives == 1

    def test_words_naming_something_else_do_not(self):
        out = score_identify(self._task(), {"e1": ("Person",)}, {"e1": "Acme"})
        assert out[Dimension.MENTION].false_positives == 1

    def test_it_is_absent_where_the_corpus_recorded_none(self):
        """Absent rather than zero, so a corpus without mentions does not pull
        down every cell it appears in."""
        assert Dimension.MENTION not in score_identify(task(person()), {"e1": ("Person",)}, {"e1": "Ada"})

    def test_a_wrong_mention_does_not_fail_the_entity(self):
        out = score_identify(self._task(), {"e1": ("Person",)}, {"e1": "Acme"})
        assert out[Dimension.ENTITY].true_positives == 1
        assert out[Dimension.CLASS].true_positives == 1


class TestScoringFillable:
    def _task(self):
        return task(person(name="Ada", jobTitle="engineer"))

    def test_naming_exactly_what_the_document_fills_scores_one(self):
        out = score_fillable(self._task(), {"p1": ("name", "jobTitle")})
        assert out[Dimension.FILLABLE].precision == 1.0
        assert out[Dimension.FILLABLE].recall == 1.0

    def test_a_slot_the_document_never_fills_costs_precision(self):
        """The extract step is then made to demand a value that is not there."""
        out = score_fillable(self._task(), {"p1": ("name", "jobTitle", "award")})
        assert out[Dimension.FILLABLE].false_positives == 1

    def test_a_slot_left_out_costs_recall(self):
        """That value can no longer be reached at all."""
        out = score_fillable(self._task(), {"p1": ("name",)})
        assert out[Dimension.FILLABLE].false_negatives == 1

    def test_an_entity_never_answered_for_still_owes_its_slots(self):
        out = score_fillable(self._task(), {})
        assert out[Dimension.FILLABLE].false_negatives == 2
