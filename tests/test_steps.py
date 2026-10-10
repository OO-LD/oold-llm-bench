"""Scoring one step of the pipeline against the answer the corpus holds.

A step fed real upstream output is measured together with every mistake made
before it. These pin that each step can be handed the corpus's own answer and
scored on its own question, which is what makes a drop attributable.
"""

from __future__ import annotations

from oold_llm_bench.grading.triples import Dimension
from oold_llm_bench.grading.vocabulary import PropertyHierarchy, Relation, read_hierarchy
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


class TestAVocabularyThatOffersTwoNamesForOneReading:
    """ "Written by Jane Doe" licenses `author` and `creator` both, and
    Wikidata says so: P50 is a subproperty of P170. Charging a false positive
    and a false negative for the wider name charges two errors for one answer.
    """

    def _hierarchy(self):
        return PropertyHierarchy(
            parents={"author": frozenset({"creator"}), "creator": frozenset()},
            contested=frozenset({frozenset({"affiliation", "memberOf"})}),
        )

    def _task(self):
        return task(ExpectedInstance(key="p1", class_path="Person", fields={"author": "Jane Doe"}))

    def test_strict_scoring_is_unchanged_by_the_hierarchy(self):
        """The headline number must not move when leniency is added beside it."""
        out = score_fillable(self._task(), {"p1": ("creator",)}, vocabulary=self._hierarchy())
        assert out[Dimension.FILLABLE].true_positives == 0
        assert out[Dimension.FILLABLE].false_positives == 1
        assert out[Dimension.FILLABLE].false_negatives == 1

    def test_the_wider_name_is_a_hit_under_the_vocabulary(self):
        out = score_fillable(self._task(), {"p1": ("creator",)}, vocabulary=self._hierarchy())
        assert out[Dimension.FILLABLE_NEAR].true_positives == 1
        assert out[Dimension.FILLABLE_NEAR].false_positives == 0
        assert out[Dimension.FILLABLE_NEAR].false_negatives == 0

    def test_an_unrelated_name_stays_wrong(self):
        out = score_fillable(self._task(), {"p1": ("award",)}, vocabulary=self._hierarchy())
        assert out[Dimension.FILLABLE_NEAR].true_positives == 0
        assert out[Dimension.FILLABLE_NEAR].false_positives == 1

    def test_one_wider_name_answers_one_slot_and_not_two(self):
        """Two expected slots under one parent are not both paid for by naming it once."""
        both = task(ExpectedInstance(key="p1", class_path="Person", fields={"author": "Jane", "illustrator": "Max"}))
        hierarchy = PropertyHierarchy(
            parents={"author": frozenset({"creator"}), "illustrator": frozenset({"creator"})},
        )
        out = score_fillable(both, {"p1": ("creator",)}, vocabulary=hierarchy)
        assert out[Dimension.FILLABLE_NEAR].true_positives == 1
        assert out[Dimension.FILLABLE_NEAR].false_negatives == 1

    def test_no_hierarchy_reports_no_lenient_dimension(self):
        """It would be the strict count under a second name."""
        out = score_fillable(self._task(), {"p1": ("creator",)}, vocabulary=PropertyHierarchy(parents={}))
        assert Dimension.FILLABLE_NEAR not in out


class TestTheBuiltHierarchy:
    def test_it_closes_over_both_vocabularies(self):
        hierarchy = read_hierarchy()
        assert "creator" in hierarchy.ancestors("author")
        assert hierarchy.relation("creator", "author") is Relation.BROADER
        assert hierarchy.relation("author", "creator") is Relation.NARROWER
        assert hierarchy.relation("award", "author") is Relation.NONE

    def test_a_chain_through_an_unoffered_property_still_connects(self):
        """`birthDate` reaches `startDate` through `inception`, offered as neither."""
        assert "startDate" in read_hierarchy().ancestors("birthDate")

    def test_the_contested_pair_is_mutual_rather_than_dropped(self):
        """Wikidata and schema.org order these opposite ways. Both edges stay."""
        hierarchy = read_hierarchy()
        assert frozenset({"affiliation", "memberOf"}) in hierarchy.contested
        assert hierarchy.near("affiliation", "memberOf")
        assert hierarchy.near("memberOf", "affiliation")

    def test_a_missing_file_is_an_empty_hierarchy_rather_than_an_error(self, tmp_path):
        assert read_hierarchy(tmp_path / "absent.json").parents == {}


class TestACorpusThatDoesNotRecordEveryEntity:
    """A Wikipedia lead names the subject's founder, its city and its parent
    organisation, and only the subject is expected. Counting the rest as
    inventions measures how much the document says, not how well the step
    read it: measured, a step at entity recall 1.00 scored precision 0.11.
    """

    def _task(self, exhaustive: bool):
        one = task(person())
        return one.model_copy(update={"corpus": one.corpus.model_copy(update={"exhaustive": exhaustive})})

    def test_a_generated_corpus_still_reports_f1(self):
        from oold_llm_bench.steps.run import StepOutcome

        out = StepOutcome(step="identify", dimensions=score_identify(self._task(True), {"e1": ("Person",)}))
        assert out.metric == "f1"

    def test_a_harvested_one_reports_recall(self):
        """Not zero and not hidden: the counts are in the record, and the
        number the cell reports is the one that can be read."""
        from oold_llm_bench.steps.run import StepOutcome

        found = score_identify(self._task(False), {"e1": ("Person",), "e2": ("Organization",)})
        out = StepOutcome(step="identify", dimensions=found, metric="recall")
        assert out.describe()["primary_f1"] == 1.0
        assert found[Dimension.ENTITY].false_positives == 1, "the invention is still counted"
