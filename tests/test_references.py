"""Scoring a link between two entities.

Until this existed, a perfect graph and a correct bag of unlinked entities
scored identically. A nested object became its own entity and the edge that
pointed at it was discarded, and a link written as an id became an ordinary
string, compared literally against a key the model could not have guessed.

These tests fix both halves: the edge survives extraction, and it is judged by
where it points rather than by what it is called.
"""

from __future__ import annotations

from oold_llm_bench.extract.json_answer import extract_json
from oold_llm_bench.grading import Reference, make_triple
from oold_llm_bench.grading.score import score_task
from oold_llm_bench.tasks.models import CorpusRef, ExpectedInstance, Source, Split, TaskRecord


def _task(expected: list[ExpectedInstance]) -> TaskRecord:
    return TaskRecord(
        id="t1",
        document="Ada works for Example Lab.",
        expected=expected,
        corpus=CorpusRef(source=Source.SYNTHETIC, document_id="d1", content_hash="0" * 64),
        split=Split.DEV,
    )


def _people() -> list[ExpectedInstance]:
    return [
        ExpectedInstance(
            key="p1",
            class_path="Person",
            fields={"name": "Ada", "worksFor": Reference(key="o1")},
        ),
        ExpectedInstance(key="o1", class_path="Organization", fields={"name": "Example Lab"}),
    ]


class TestTheEdgeSurvivesExtraction:
    def test_a_nested_object_leaves_a_link_behind(self):
        produced = extract_json({
            "type": "Person",
            "name": "Ada",
            "worksFor": {"type": "Organization", "name": "Example Lab"},
        })
        links = [t for t in produced.triples if isinstance(t.value, Reference)]
        assert len(links) == 1
        assert links[0].prop == "works_for"

    def test_the_link_points_at_the_entity_that_was_read(self):
        produced = extract_json({
            "type": "Person",
            "name": "Ada",
            "worksFor": {"type": "Organization", "name": "Example Lab"},
        })
        link = next(t.value for t in produced.triples if isinstance(t.value, Reference))
        assert produced.classes[link.key] == "Organization"

    def test_the_child_is_still_its_own_entity(self):
        """The edge is added, nothing is taken away."""
        produced = extract_json({
            "type": "Person",
            "name": "Ada",
            "worksFor": {"type": "Organization", "name": "Example Lab"},
        })
        assert len(produced.entities()) == 2


class TestALinkIsJudgedByWhereItPoints:
    def test_a_correct_graph_scores_one(self):
        produced = extract_json([
            {"type": "Person", "name": "Ada", "worksFor": {"type": "Organization", "name": "Example Lab"}},
        ])
        assert score_task(_task(_people()), produced).primary == 1.0

    def test_the_model_may_name_its_entities_anything(self):
        """The produced key is the model's own and alignment rewrites it.

        This pinned the opposite until the id form was read: a link written as
        a planned id stayed an ordinary string, so a segmented orchestration
        could emit correct edges and score none of them. See
        TestALinkWrittenAsAPlannedId for the rule that closed it.
        """
        produced = extract_json({
            "entities": [
                {"id": "zzz", "type": "Organization", "name": "Example Lab"},
                {"id": "aaa", "type": "Person", "name": "Ada", "worksFor": "zzz"},
            ]
        })
        link = next(t for t in produced.triples if isinstance(t.value, Reference))
        assert link.entity == "aaa"
        assert link.value == Reference(key="zzz")

    def test_the_same_entities_without_the_link_score_lower(self):
        """The finding this whole change exists for."""
        linked = extract_json([
            {"type": "Person", "name": "Ada", "worksFor": {"type": "Organization", "name": "Example Lab"}}
        ])
        unlinked = extract_json([
            {"type": "Person", "name": "Ada"},
            {"type": "Organization", "name": "Example Lab"},
        ])
        task = _task(_people())
        assert score_task(task, linked).primary > score_task(task, unlinked).primary

    def test_a_link_to_an_invented_entity_is_wrong_and_not_missing(self):
        produced = extract_json([
            {"type": "Person", "name": "Ada", "worksFor": {"type": "Organization", "name": "Somewhere Else"}},
        ])
        result = score_task(_task(_people()), produced)
        assert result.primary < 1.0


def test_a_reference_reads_as_an_arrow():
    assert str(Reference(key="o1")) == "-> o1"


def test_a_reference_is_a_triple_value():
    triple = make_triple("p1", "worksFor", Reference(key="o1"))
    assert triple.value == Reference(key="o1")


class TestALinkWrittenAsAPlannedId:
    """The shape a segmented orchestration answers in.

    It plans the document, pins each entity to a planned id and writes an edge
    as that id. Read as a string the edge is compared literally against a key
    the model could not have guessed, which is the case this module's docstring
    calls unscoreable.
    """

    def test_an_id_another_entity_claimed_becomes_a_link(self):
        produced = extract_json({
            "entities": [
                {"id": "e1", "type": "Person", "name": "Ada", "worksFor": "e2"},
                {"id": "e2", "type": "Organization", "name": "Example Lab"},
            ]
        })
        link = next(t for t in produced.triples if isinstance(t.value, Reference))
        assert link.entity == "e1"
        assert link.value == Reference(key="e2")

    def test_only_an_id_an_entity_claimed_resolves(self):
        """A key this reader invented was never visible to the answer.

        Resolving against one would turn a value that happens to read ``e2``
        into an edge nobody asserted.
        """
        produced = extract_json({
            "entities": [
                {"type": "Person", "name": "Ada", "note": "e2"},
                {"type": "Organization", "name": "Example Lab"},
            ]
        })
        assert not [t for t in produced.triples if isinstance(t.value, Reference)]

    def test_an_entity_pointing_at_itself_is_not_a_link(self):
        """A defect in the answer, not a self-loop to be scored."""
        produced = extract_json({"entities": [{"id": "e1", "type": "Person", "knows": "e1"}]})
        assert not [t for t in produced.triples if isinstance(t.value, Reference)]

    def test_the_stated_id_becomes_the_entity_key(self):
        produced = extract_json({"entities": [{"id": "zzz", "type": "Person", "name": "Ada"}]})
        assert produced.entities() == frozenset({"zzz"})

    def test_the_id_is_not_also_read_as_a_value(self):
        produced = extract_json({"entities": [{"id": "e1", "type": "Person", "name": "Ada"}]})
        assert {t.prop for t in produced.triples} == {"name"}
