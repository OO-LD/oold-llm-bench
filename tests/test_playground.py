"""The playground, without a provider and without an interface.

Everything here runs offline. What the playground shows a person is a
conversion from an answer to a picture, four read-outs computed from the
result, and an orchestration switch, and none of those needs a model. The one
test that does need a model is the end-to-end one in
``tests/test_playground_ui.py``, which is opt-in.

The agent package is optional, so the tests that build a real agent skip
without it rather than testing a stand-in against the shape the stand-in
assumes.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
from pathlib import Path
from typing import Never

import pytest

from oold_llm_bench.extract import extract_json
from oold_llm_bench.grading.triples import Quantity, Reference, TripleSet, make_triple
from oold_llm_bench.playground import (
    ALL_CLASSES,
    CLOSE_MATCH,
    CORPORA,
    DIFFERENT,
    EXACT_MATCH,
    ORCHESTRATIONS,
    SCHEMAS_ENV,
    Comparable,
    GraphState,
    MissingSchemas,
    Options,
    ReplayClient,
    build_cell,
    build_graph,
    catalogue_sets,
    cost_rows,
    cost_total,
    coverage,
    decide,
    degradation_rows,
    load_schemaorg,
    paste_task,
    run_once,
    schema_rows,
    schemaorg_tasks,
    schemas_directory,
    score_rows,
    validation_rows,
    wiki_tasks,
)
from oold_llm_bench.playground.app import STATE_STYLE, state_html
from oold_llm_bench.playground.corpora import NAME_SLOT, is_scoreable
from oold_llm_bench.playground.graph import CLOSE_MATCH_COLOUR, PARALLEL_ROUNDNESS, SELF_ANGLE
from oold_llm_bench.playground.replay import _ids_in
from oold_llm_bench.tasks.models import (
    CorpusRef,
    Difficulty,
    ExpectedInstance,
    Source,
    Split,
    TaskRecord,
)

oold_agent = pytest.importorskip("oold.agent.extraction")


class FakeLink:
    """One edge, in the shape the agent records it and nothing more."""

    def __init__(self, source: str, prop: str, target: str) -> None:
        self.source = source
        self.prop = prop
        self.target = target


def answer_schema(properties: int) -> dict:
    """An answer schema carrying a given number of entity properties."""
    shape = {name: {"type": "string"} for name in (f"p{index}" for index in range(properties))}
    return {"properties": {"entities": {"items": {"properties": shape}}}}


def drawn(outcome) -> dict:
    """The graph one outcome drew, refusing an outcome that drew none."""
    assert outcome.graph is not None, f"nothing was drawn: {outcome.error}"
    return outcome.graph.describe()


def triples(*items) -> TripleSet:
    return TripleSet(
        triples=frozenset(make_triple(entity, prop, value) for entity, prop, value in items),
        classes={},
        provenance={},
    )


def linked_answer() -> TripleSet:
    """Two entities and an edge, the shape a linked task expects."""
    return TripleSet(
        triples=frozenset({
            make_triple("e1", "abstract", "Mickdros Calnor"),
            make_triple("e1", "publisherImprint", Reference(key="e2")),
            make_triple("e2", "address", "Rhoder Mickvale"),
        }),
        classes={"e1": "CreativeWork", "e2": "Organization"},
        provenance={},
    )


def quantity_task(task_id: str = "t1") -> TaskRecord:
    return TaskRecord(
        id=task_id,
        document="The sample was 1.75 m long.",
        expected=[
            ExpectedInstance(key="q1", class_path="Length", fields={"value": Quantity(magnitude=1.75, unit="meter")})
        ],
        corpus=CorpusRef(source=Source.SYNTHETIC, document_id=task_id, content_hash="0" * 64),
        split=Split.DEV,
        difficulty=Difficulty.EASY,
        catalogue=["Length", "Mass", "Duration"],
        unit_catalogue={"Length": ["meter"], "Mass": ["gram"], "Duration": ["second"]},
    )


def linked_task(task_id: str = "t-link") -> TaskRecord:
    """Two entities and one edge, written by hand so no corpus is needed."""
    return TaskRecord(
        id=task_id,
        document="Filed under creative work, published by the imprint recorded as Rhoder Mickvale.",
        expected=[
            ExpectedInstance(
                key="e1",
                class_path="CreativeWork",
                fields={"abstract": "Mickdros Calnor", "publisherImprint": Reference(key="e2")},
            ),
            ExpectedInstance(key="e2", class_path="Organization", fields={"address": "Rhoder Mickvale"}),
        ],
        corpus=CorpusRef(source=Source.SYNTHETIC, document_id=task_id, content_hash="1" * 64),
        split=Split.DEV,
        catalogue=["CreativeWork", "Organization", "Person"],
        property_ranges={"publisherImprint": ["Organization"]},
    )


def named_answer(*rows: tuple[str, str | None, str], link: tuple[str, str, str] | None = None) -> TripleSet:
    """Entities with a name and, where the answer stated one, a class."""
    triples = {make_triple(key, "name", name) for key, _, name in rows}
    if link is not None:
        source, prop, target = link
        triples.add(make_triple(source, prop, Reference(key=target)))
    return TripleSet(
        triples=frozenset(triples),
        classes={key: class_path for key, class_path, _ in rows if class_path is not None},
        provenance={},
    )


class TestGraph:
    def test_an_entity_becomes_a_node_the_document_named(self):
        graph = build_graph(named_answer(("e1", "Person", "Alice")))
        assert graph.nodes["e1"].label == "Alice"
        assert graph.nodes["e1"].class_path == "Person"
        assert graph.nodes["e1"].kind == "entity"

    def test_an_entity_the_answer_did_not_name_falls_back_to_its_class(self):
        graph = build_graph(linked_answer())
        assert sorted(graph.nodes) == ["e1", "e2"]
        assert graph.nodes["e1"].label == "CreativeWork"
        assert graph.nodes["e1"].class_path == "CreativeWork"
        assert graph.nodes["e1"].kind == "entity"

    def test_two_entities_of_one_class_are_two_nodes_reading_differently(self):
        """The label was the class, so an edge between two people read
        ``Person -> Person`` and looked like a self-loop that was not one."""
        graph = build_graph(
            named_answer(("e1", "Person", "Alice"), ("e2", "Person", "Bob"), link=("e1", "knows", "e2"))
        )
        assert sorted(node.label for node in graph.of_kind("entity")) == ["Alice", "Bob"]
        edge = graph.edges_of_kind("link")[0]
        assert (graph.nodes[edge.source].label, graph.nodes[edge.target].label) == ("Alice", "Bob")

    def test_a_class_node_does_not_read_as_a_loop_on_its_own_instance(self):
        graph = build_graph(named_answer(("e1", "Person", "Alice")), show_classes=True)
        edge = graph.edges_of_kind("type")[0]
        assert graph.nodes[edge.source].label == "Alice"
        assert graph.nodes[edge.target].label == "Person"

    def test_a_reference_becomes_an_edge_and_not_a_value(self):
        graph = build_graph(linked_answer())
        edges = graph.edges_of_kind("link")
        assert [(e.source, e.target) for e in edges] == [("e1", "e2")]
        assert edges[0].label == "publisher_imprint"

    def test_values_live_in_the_tooltip_and_not_in_the_picture(self):
        graph = build_graph(linked_answer())
        assert graph.nodes["e1"].data["abstract"] == "Mickdros Calnor"
        assert "publisher_imprint" not in graph.nodes["e1"].data
        assert graph.describe()["nodes"] == 2

    def test_a_quantity_reads_as_a_magnitude_and_a_unit(self):
        graph = build_graph(triples(("q1", "value", Quantity(magnitude=1.75, unit="meter"))))
        assert graph.nodes["q1"].data["value"] == "1.75 meter"

    def test_a_class_node_is_off_by_default_and_available(self):
        assert build_graph(linked_answer()).of_kind("class") == []
        with_classes = build_graph(linked_answer(), show_classes=True)
        assert sorted(node.label for node in with_classes.of_kind("class")) == ["CreativeWork", "Organization"]
        assert [e.kind for e in with_classes.edges_of_kind("type")] == ["type", "type"]

    def test_a_literal_node_is_off_by_default_and_available(self):
        assert build_graph(linked_answer()).of_kind("literal") == []
        with_literals = build_graph(linked_answer(), show_literals=True)
        assert len(with_literals.of_kind("literal")) == 2


class TestDanglingEdges:
    """A link reaching nothing is the failure this view exists to show."""

    def test_a_link_whose_target_was_never_emitted_is_dangling(self):
        answer = triples(("e1", "abstract", "Mickdros Calnor"))
        graph = build_graph(
            answer,
            links=[FakeLink("e1", "publisherImprint", "e2")],
            dangling=[FakeLink("e1", "publisherImprint", "e2")],
        )
        assert len(graph.dangling) == 1
        assert graph.edges_of_kind("link") == []

    def test_the_missing_target_still_gets_a_node(self):
        graph = build_graph(
            triples(("e1", "abstract", "x")),
            links=[FakeLink("e1", "publisherImprint", "e2")],
            dangling=[FakeLink("e1", "publisherImprint", "e2")],
        )
        assert [node.id for node in graph.of_kind("missing")] == ["e2"]

    def test_a_dangling_edge_is_drawn_differently_from_a_resolved_one(self):
        broken = build_graph(
            triples(("e1", "abstract", "x")),
            links=[FakeLink("e1", "publisherImprint", "e2")],
            dangling=[FakeLink("e1", "publisherImprint", "e2")],
        )
        resolved = build_graph(linked_answer())
        assert broken.dangling[0].action()["dashed"] is True
        assert resolved.edges_of_kind("link")[0].action()["dashed"] is False
        assert broken.of_kind("missing")[0].overlay() is not None
        assert resolved.nodes["e2"].overlay() is None

    def test_a_link_the_agent_reports_does_not_double_the_edge(self):
        """The agent names the property as written, the extractor as normalised."""
        graph = build_graph(linked_answer(), links=[FakeLink("e1", "publisherImprint", "e2")])
        assert len(graph.edges) == 1

    def test_a_link_whose_source_was_never_emitted_is_dangling_too(self):
        """The far end is not the only one an answer can leave out.

        An edge starting at an entity nothing reported is the same failure as
        one ending at it, and used to be drawn as an ordinary link from a node
        that was not there.
        """
        graph = build_graph(
            named_answer(("e2", "Organization", "Siemens")),
            links=[FakeLink("e1", "worksFor", "e2")],
        )
        assert [node.id for node in graph.of_kind("missing")] == ["e1"]
        assert [edge.id for edge in graph.dangling] == ["e1|e2|works_for"]
        assert graph.edges_of_kind("link") == []
        assert graph.describe()["missing"] == 1

    def test_an_edge_with_one_end_unreported_can_still_be_folded_in(self):
        """The crash this reproduces: the fold places an edge under the ids it
        resolved, and an end with no node of its own had none to be placed."""
        state = GraphState()
        diff = state.update(
            build_graph(
                named_answer(("e2", "Organization", "Siemens")),
                links=[FakeLink("e1", "worksFor", "e2")],
            )
        )
        assert sorted(diff.created_nodes) == ["e1", "e2"]
        assert state.describe()["dangling"] == ["e1|e2|works_for"]
        assert [overlay["id"] for overlay in state.overlays()] == ["e1"]

    def test_an_entity_typed_and_left_empty_is_drawn_and_not_invented_as_missing(self):
        """The other half of the same crash, read through the extractor.

        A triple carries a value, so an entity that stated a class and no
        values leaves none, and ``multi_step`` arrives there by design. That
        entity was dropped from the picture, which left its class edge with no
        source and anything pointing at it looking like a link into nothing.
        """
        payload = {
            "entities": [
                {"id": "e1", "type": "Person", "name": None},
                {"id": "e2", "type": "Organization", "name": "Siemens", "employee": "e1"},
            ]
        }
        graph = build_graph(extract_json(payload), links=[FakeLink("e2", "employee", "e1")], show_classes=True)
        state = GraphState()
        state.update(graph)
        described = state.describe()
        assert described["labels"] == {"e1": "Person", "e2": "Siemens"}
        assert described["links"] == ["e2|e1|employee"]
        assert described["dangling"] == []
        assert described["n_missing"] == 0

    def test_a_class_edge_of_an_empty_entity_has_a_node_to_start_from(self):
        """With no link to it, nothing else would have drawn that node."""
        payload = {"entities": [{"id": "e1", "type": "Person", "name": None}]}
        state = GraphState()
        state.update(build_graph(extract_json(payload), show_classes=True))
        assert state.describe()["entities"] == ["e1"]
        assert state.describe()["classes"] == ["Person"]


class TestSelfLoops:
    """An entity pointing at itself asserts nothing and is not a link.

    A segmented run offers the plan's ids as an enum, and gpt-5-nano answered
    ``worksFor`` with the organisation's own id on five of the seven links it
    asserted about one document.
    """

    def test_an_edge_from_an_entity_to_itself_is_not_counted_as_a_link(self):
        graph = build_graph(named_answer(("e2", "Organization", "Siemens")), links=[FakeLink("e2", "worksFor", "e2")])
        assert graph.edges_of_kind("link") == []
        assert [e.label for e in graph.edges_of_kind("self")] == ["works_for"]
        assert graph.describe()["self_loops"] == 1

    def test_a_self_loop_does_not_invent_a_missing_node(self):
        graph = build_graph(named_answer(("e2", "Organization", "Siemens")), links=[FakeLink("e2", "worksFor", "e2")])
        assert graph.of_kind("missing") == []

    def test_a_self_loop_is_drawn_apart_from_a_resolved_link(self):
        graph = build_graph(named_answer(("e2", "Organization", "Siemens")), links=[FakeLink("e2", "worksFor", "e2")])
        assert graph.edges_of_kind("self")[0].action()["dashed"] is True


class TestParallelEdges:
    """Two edges between one pair of nodes have to be two lines on screen.

    ``multi_step`` states the relation both ways round, so "Andrea works at
    Siemens" comes back as ``worksFor`` one way and ``employee`` the other.
    Drawn on the renderer's default path those are one line carrying two
    labels on top of each other.

    The curve is read off :meth:`GraphState.edge_overlays`, because that is
    what the interface sends: the flat action format the edges arrive in
    carries a label and a dash and nothing that bends a line.
    """

    def folded(self, *links: FakeLink) -> GraphState:
        state = GraphState()
        state.update(
            build_graph(
                named_answer(("e1", "Person", "Andrea"), ("e2", "Organization", "Siemens")),
                links=list(links),
            )
        )
        return state

    def curves(self, state: GraphState) -> dict[str, dict]:
        return {overlay["id"]: overlay["smooth"] for overlay in state.edge_overlays() if "smooth" in overlay}

    def test_an_edge_alone_between_its_nodes_keeps_the_straight_path(self):
        """Nothing is bent that nothing would cover, so one answer in one
        direction draws exactly what it drew before."""
        state = self.folded(FakeLink("e1", "worksFor", "e2"))
        assert state.edge_overlays() == []

    def test_an_edge_and_its_inverse_are_bent_to_either_side(self):
        """Both are turned the same way and they end up apart.

        The renderer turns a curve relative to the direction of the edge, so
        one handedness applied to two opposite directions is two opposite
        sides of the same straight line.
        """
        curves = self.curves(self.folded(FakeLink("e1", "worksFor", "e2"), FakeLink("e2", "employee", "e1")))
        assert sorted(curves) == ["e1|e2|works_for", "e2|e1|employee"]
        assert {curve["type"] for curve in curves.values()} == {"curvedCW"}
        assert {curve["roundness"] for curve in curves.values()} == {PARALLEL_ROUNDNESS}

    def test_two_edges_the_same_way_round_are_turned_against_each_other(self):
        """Here the directions agree, so the handedness has to differ instead."""
        curves = self.curves(self.folded(FakeLink("e1", "worksFor", "e2"), FakeLink("e1", "memberOf", "e2")))
        assert sorted(curves) == ["e1|e2|member_of", "e1|e2|works_for"]
        assert {curve["type"] for curve in curves.values()} == {"curvedCW", "curvedCCW"}

    def test_a_third_edge_between_one_pair_is_bent_further_out(self):
        curves = self.curves(
            self.folded(
                FakeLink("e1", "worksFor", "e2"),
                FakeLink("e1", "memberOf", "e2"),
                FakeLink("e2", "employee", "e1"),
            )
        )
        assert sorted(curve["roundness"] for curve in curves.values()) == [
            PARALLEL_ROUNDNESS,
            PARALLEL_ROUNDNESS,
            PARALLEL_ROUNDNESS * 2,
        ]

    def test_two_loops_on_one_node_are_turned_apart_rather_than_bent(self):
        """A loop is not a curve between two points, so roundness does nothing
        to it and the renderer's own angle is what separates two."""
        state = self.folded(FakeLink("e2", "worksFor", "e2"), FakeLink("e2", "employee", "e2"))
        angles = {overlay["id"]: overlay["selfReference"]["angle"] for overlay in state.edge_overlays()}
        assert sorted(angles) == ["e2|e2|employee", "e2|e2|works_for"]
        assert len(set(angles.values())) == 2
        assert set(angles.values()) == {math.pi / 4, math.pi / 4 + SELF_ANGLE}

    def test_every_curve_names_the_edge_it_bends(self):
        """The overlay goes out as an edge update, matched by id and not by
        position, so it has to name an edge that was drawn."""
        state = self.folded(FakeLink("e1", "worksFor", "e2"), FakeLink("e2", "employee", "e1"))
        drawn = {edge.id: edge for edge in state.edges.values()}
        for overlay in state.edge_overlays():
            edge = drawn[overlay["id"]]
            assert (overlay["from"], overlay["to"]) == (edge.source, edge.target)


class TestGraphState:
    def test_the_first_answer_is_all_additions(self):
        state = GraphState()
        diff = state.update(build_graph(linked_answer()))
        assert sorted(diff.created_nodes) == ["e1", "e2"]
        assert len(diff.created_edges) == 1
        assert [a["action"] for a in diff.actions].count("addNode") == 2

    def test_an_unchanged_node_is_stored_rather_than_re_added(self):
        state = GraphState()
        state.update(build_graph(linked_answer()))
        diff = state.update(build_graph(linked_answer()))
        assert diff.created_nodes == []
        assert diff.updated_nodes == []
        assert sorted(diff.stored_nodes) == ["e1", "e2"]

    def test_a_second_answer_adds_to_the_first(self):
        state = GraphState()
        state.update(build_graph(triples(("e1", "abstract", "x"))))
        diff = state.update(build_graph(triples(("e3", "name", "y"))))
        assert diff.created_nodes == ["e3"]
        assert state.describe()["entities"] == ["e1", "e3"]

    def test_the_state_read_out_names_the_entities_and_the_edges(self):
        state = GraphState()
        state.update(build_graph(linked_answer()))
        described = state.describe()
        assert described["entities"] == ["e1", "e2"]
        assert described["links"] == ["e1|e2|publisher_imprint"]
        assert described["dangling"] == []

    def test_a_missing_node_carries_an_overlay_the_flat_format_cannot(self):
        state = GraphState()
        state.update(
            build_graph(
                triples(("e1", "abstract", "x")),
                links=[FakeLink("e1", "p", "ghost")],
                dangling=[FakeLink("e1", "p", "ghost")],
            )
        )
        assert [o["id"] for o in state.overlays()] == ["ghost"]


class TestTheStateReadOut:
    """The graph published as data, which is also what a reader sees.

    It is the one read-out that is not a table, and it was written as a single
    line of JSON. An empty graph is already 450 characters of it, and a
    ``code`` element does not break a line nobody put a space in, so it ran
    off the side of the panel before anything had been extracted at all.
    """

    def folded(self) -> GraphState:
        state = GraphState()
        state.update(build_graph(linked_answer()))
        return state

    def read(self, state: GraphState) -> str:
        """The text the browser shows, which is what the test there reads."""
        found = re.search(r"<code[^>]*>(.*)</code>", state_html(state), flags=re.S)
        assert found is not None
        return html.unescape(found.group(1))

    def test_the_payload_still_reads_back_as_the_state(self):
        """What the end-to-end test does with it, and the reason the JSON
        stays inside the element rather than moving to a widget."""
        payload = json.loads(self.read(self.folded()))
        assert payload["entities"] == ["e1", "e2"]
        assert payload["links"] == ["e1|e2|publisher_imprint"]

    def test_the_payload_is_indented_rather_than_one_line(self):
        body = self.read(GraphState())
        assert body.count("\n") > 10
        assert "\n  " in body

    def test_the_read_out_is_bounded_and_scrolls_instead_of_growing(self):
        assert f'<code style="{STATE_STYLE}">' in state_html(GraphState())
        assert "max-height" in STATE_STYLE
        assert "overflow:auto" in STATE_STYLE

    def test_the_hooks_the_end_to_end_test_waits_on_are_still_there(self):
        markup = state_html(self.folded())
        assert 'data-testid="graph-state"' in markup
        assert 'data-nodes="2"' in markup
        assert 'data-edges="1"' in markup
        assert 'data-turns="1"' in markup


class StubJudge:
    """A judge that answers from a table, so a test spends no call."""

    def __init__(self, answers: dict[tuple[str, str], str], name: str = "stub") -> None:
        self.answers = answers
        self.name = name
        self.asked: list[tuple[str, str]] = []

    def decide(self, left, right) -> tuple[str, str]:
        pair = (left.values.get("name", left.key), right.values.get("name", right.key))
        self.asked.append(pair)
        return self.answers.get(pair, DIFFERENT), "stubbed"


class TestIdentityDecision:
    """Whether two entities are one, and which route decided it."""

    def entity(self, key: str, class_path: str | None, **values: str) -> Comparable:
        return Comparable(key=key, class_path=class_path, values=values)

    def test_exact_agreement_merges_without_spending_a_call(self):
        """Same properties, same values. A restatement, not a judgement."""
        judge = StubJudge({}, name="unused")
        decision = decide(
            self.entity("a", "Organization", name="Example Corp"),
            self.entity("b", "Organization", name="Example Corp"),
            judge,
        )
        assert (decision.outcome, decision.route, decision.judge) == (EXACT_MATCH, "agreement", None)
        assert judge.asked == []

    def test_one_side_carrying_more_is_asked_rather_than_merged(self):
        """A superset is a judgement, and the cheap route must be the safe one.

        A merge cannot be undone from the interface, so agreement fires only
        where there is nothing to decide. Whether an organisation named
        Example Corp with an address is the Example Corp without one is a real
        question, and it goes to whoever can answer it.
        """
        judge = StubJudge({}, name="asked")
        decision = decide(
            self.entity("a", "Organization", name="Example Corp"),
            self.entity("b", "Organization", name="Example Corp", address="Berlin"),
            judge,
        )
        assert decision.route == "judge"
        assert judge.asked

    def test_one_agreeing_property_no_longer_merges_two_people(self):
        """The case the old rule got wrong, kept as the reason for the new one.

        Agreement used to mean every *shared* value, so two people who had
        only a job title in common were merged with no call and no prompt.
        """
        judge = StubJudge({}, name="asked")
        decision = decide(
            self.entity("a", "Person", name="Alice Smith", jobTitle="Engineer"),
            self.entity("b", "Person", jobTitle="Engineer", email="bob@example.com"),
            judge,
        )
        assert decision.outcome != EXACT_MATCH or decision.route == "judge"
        assert judge.asked

    def test_agreement_is_case_insensitive_but_still_deterministic(self):
        decision = decide(
            self.entity("a", "Organization", name="example corp"),
            self.entity("b", "Organization", name="Example Corp"),
            StubJudge({}),
        )
        assert decision.outcome == EXACT_MATCH

    def test_an_entity_that_states_no_class_contradicts_none(self):
        """A nested object written into a link slot arrives with no class.

        Reading that absence as a class of its own drew "Example Corp" twice:
        once as the organisation and once as the object in ``worksFor``.
        """
        judge = StubJudge({})
        decision = decide(
            self.entity("a", None, name="Example Corp"),
            self.entity("b", "Organization", name="Example Corp"),
            judge,
        )
        assert (decision.outcome, decision.route) == (EXACT_MATCH, "agreement")
        assert judge.asked == []

    def test_a_class_mismatch_is_decided_here_and_never_asked_about(self):
        judge = StubJudge({})
        decision = decide(
            self.entity("a", "Person", name="Example Corp"),
            self.entity("b", "Organization", name="Example Corp"),
            judge,
        )
        assert (decision.outcome, decision.route) == (DIFFERENT, "agreement")
        assert judge.asked == []

    def test_a_disagreement_goes_to_the_judge_and_names_the_conflict(self):
        judge = StubJudge({("Example Corp", "Example Corporation"): CLOSE_MATCH})
        decision = decide(
            self.entity("a", "Organization", name="Example Corp"),
            self.entity("b", "Organization", name="Example Corporation"),
            judge,
        )
        assert (decision.outcome, decision.route, decision.judge) == (CLOSE_MATCH, "judge", "stub")
        assert decision.conflicts == ("name",)

    def test_only_an_exact_match_merges(self):
        for outcome in (CLOSE_MATCH, DIFFERENT):
            judge = StubJudge({("a", "b"): outcome})
            decision = decide(self.entity("a", "Person"), self.entity("b", "Person"), judge)
            assert decision.merges is False
        judge = StubJudge({("a", "b"): EXACT_MATCH})
        assert decide(self.entity("a", "Person"), self.entity("b", "Person"), judge).merges is True

    def test_coverage_counts_only_the_pairs_that_were_decided(self):
        judge = StubJudge({("a", "b"): CLOSE_MATCH})
        deferred = decide(self.entity("a", "Person"), self.entity("b", "Person"), judge)
        settled = decide(self.entity("c", "Person"), self.entity("d", "Organization"), judge)
        assert coverage([deferred]) == 0.0
        assert coverage([settled]) == 1.0
        assert coverage([deferred, settled]) == 0.5


def turn(*rows: tuple[str, str, str], link: tuple[str, str, str] | None = None):
    return build_graph(named_answer(*rows, link=link))


ALICE = ("e1", "Person", "Alice")
BOB = ("e1", "Person", "Bob")
CORP = ("e2", "Organization", "Example Corp")


class TestTwoTurnsGrowOneGraph:
    """Turn two must add to turn one rather than overwrite it.

    Every orchestration hands out the same ids, so folding a second answer in
    by key alone wrote its ``e1`` over the first one's and the graph stopped
    growing at two nodes however many turns were sent.
    """

    def judged(self, answers: dict[tuple[str, str], str]) -> tuple[GraphState, StubJudge]:
        judge = StubJudge(answers)
        return GraphState(judge=judge), judge

    def test_a_second_person_is_added_and_the_shared_employer_is_not(self):
        state, _ = self.judged({("Bob", "Alice"): DIFFERENT})
        state.update(turn(ALICE, CORP, link=("e1", "worksFor", "e2")))
        state.update(turn(BOB, CORP, link=("e1", "worksFor", "e2")))
        described = state.describe()
        assert sorted(described["labels"].values()) == ["Alice", "Bob", "Example Corp"]
        assert sorted(described["types"].values()) == ["Organization", "Person", "Person"]
        assert len(described["links"]) == 2
        assert described["self_loops"] == []

    def test_both_employment_edges_reach_the_one_organisation(self):
        state, _ = self.judged({("Bob", "Alice"): DIFFERENT})
        state.update(turn(ALICE, CORP, link=("e1", "worksFor", "e2")))
        state.update(turn(BOB, CORP, link=("e1", "worksFor", "e2")))
        described = state.describe()
        targets = {link.split("|")[1] for link in described["links"]}
        assert len(targets) == 1
        assert described["labels"][targets.pop()] == "Example Corp"

    def test_the_identical_organisation_takes_the_fast_path_and_spends_nothing(self):
        state, judge = self.judged({("Bob", "Alice"): DIFFERENT})
        state.update(turn(ALICE, CORP, link=("e1", "worksFor", "e2")))
        state.update(turn(BOB, CORP, link=("e1", "worksFor", "e2")))
        merges = [d for d in state.ledger.decisions if d.merges]
        assert [(d.outcome, d.route, d.judge) for d in merges] == [(EXACT_MATCH, "agreement", None)]
        assert ("Example Corp", "Example Corp") not in judge.asked

    def test_a_judge_deciding_the_same_thing_merges_too_and_says_it_paid(self):
        state, _ = self.judged({("Bob", "Alice"): DIFFERENT, ("Example Corporation", "Example Corp"): EXACT_MATCH})
        state.update(turn(ALICE, CORP))
        state.update(turn(BOB, ("e2", "Organization", "Example Corporation")))
        merged = [d for d in state.ledger.decisions if d.merges]
        assert [(d.route, d.judge) for d in merged] == [("judge", "stub")]
        assert len([n for n in state.nodes.values() if n.kind == "entity"]) == 3

    def test_an_unclear_pair_stays_two_nodes_with_a_relation_between_them(self):
        state, _ = self.judged({("Bob", "Alice"): CLOSE_MATCH})
        state.update(turn(ALICE))
        state.update(turn(BOB))
        described = state.describe()
        assert len(described["entities"]) == 2
        assert len(described["close_matches"]) == 1
        assert described["close_matches"][0].endswith(f"|{CLOSE_MATCH}")
        assert described["links"] == []

    def test_a_deferred_relation_is_drawn_apart_from_every_other_edge(self):
        state, _ = self.judged({("Bob", "Alice"): CLOSE_MATCH})
        state.update(turn(ALICE, CORP, link=("e1", "worksFor", "e2")))
        state.update(turn(BOB))
        overlays = state.edge_overlays()
        assert [o["color"] for o in overlays] == [CLOSE_MATCH_COLOUR]
        assert all(e.action()["dashed"] for e in state.edges.values() if e.kind == "closeMatch")

    def test_a_merge_records_the_conflict_rather_than_taking_the_newer_value(self):
        state, _ = self.judged({("Example Corp", "Example Corp"): EXACT_MATCH})
        state.update(turn(("e2", "Organization", "Example Corp")))
        second = TripleSet(
            triples=frozenset({
                make_triple("e2", "name", "Example Corp"),
                make_triple("e2", "address", "Berlin"),
            }),
            classes={"e2": "Organization"},
            provenance={},
        )
        third = TripleSet(
            triples=frozenset({
                make_triple("e2", "name", "Example Corp"),
                make_triple("e2", "address", "Munich"),
            }),
            classes={"e2": "Organization"},
            provenance={},
        )
        state.update(build_graph(second))
        state.update(build_graph(third))
        described = state.describe()
        assert len(described["entities"]) == 1
        assert described["conflicts"] == {"e2": ["address"]}
        node = state.nodes["e2"]
        assert node.data["address"] == "Berlin"
        assert node.data["address (conflict)"] == "Munich"

    def test_one_answer_reporting_one_thing_twice_draws_one_node(self):
        """Nano reported the organisation twice in a single answer.

        The same rule settles it: the graph is one graph, and an answer is not
        a scope the identity step is allowed to skip.
        """
        state, judge = self.judged({})
        state.update(
            build_graph(
                named_answer(ALICE, CORP, ("e3", "Organization", "Example Corp"), link=("e1", "worksFor", "e2"))
            )
        )
        described = state.describe()
        assert sorted(described["labels"].values()) == ["Alice", "Example Corp"]
        assert [d.route for d in state.ledger.decisions if d.merges] == ["agreement"]
        assert judge.asked == []

    def test_a_merged_node_takes_the_class_the_other_half_stated(self):
        state, _ = self.judged({})
        state.update(build_graph(named_answer(("e1", None, "Example Corp"), ("e2", "Organization", "Example Corp"))))
        described = state.describe()
        assert list(described["types"].values()) == ["Organization"]
        assert list(described["labels"].values()) == ["Example Corp"]

    def test_different_classes_are_decided_without_asking(self):
        state, judge = self.judged({})
        state.update(turn(ALICE))
        state.update(turn(CORP))
        crossed = [d for d in state.ledger.decisions if {d.left, d.right} == {"e1", "e2"}]
        assert [(d.outcome, d.route) for d in crossed] == [(DIFFERENT, "agreement")]
        assert judge.asked == []

    def test_coverage_falls_when_a_judge_defers(self):
        state, _ = self.judged({("Bob", "Alice"): CLOSE_MATCH})
        state.update(turn(ALICE))
        state.update(turn(BOB))
        assert state.describe()["identity"]["coverage"] == 0.0

    def test_with_no_judge_nothing_but_exact_agreement_merges(self):
        """And the no-op is not billed as a call.

        `JUDGE` is documented as the route that spent one, and the read-out
        reports the two routes apart so the cost is on screen rather than in
        a bill. A session with no judge configured makes no calls, so it must
        not read as "decided by judge".
        """
        state = GraphState()
        state.update(turn(ALICE))
        state.update(turn(BOB))
        described = state.describe()
        assert len(described["entities"]) == 2
        assert [d.judge for d in state.ledger.decisions] == [None]
        assert described["identity"]["counts"]["judge"] == 0
        assert described["identity"]["coverage"] == 0.0


class TestReadOuts:
    def test_the_score_panel_reports_every_dimension_primary_first(self):
        from oold_llm_bench.grading.score import score_task

        task = quantity_task()
        produced = TripleSet(
            triples=frozenset({make_triple("q1", "value", Quantity(magnitude=1.75, unit="meter"))}),
            classes={"q1": "Length"},
            provenance={},
        )
        rows = score_rows(score_task(task, produced))
        assert rows[0]["dimension"] == "value"
        assert rows[0]["f1"] == 1.0
        assert {row["dimension"] for row in rows} >= {"value", "entity", "class", "unit"}

    def test_a_wrong_answer_scores_below_a_right_one(self):
        from oold_llm_bench.grading.score import score_task

        task = quantity_task()
        wrong = TripleSet(
            triples=frozenset({make_triple("q1", "value", Quantity(magnitude=99.0, unit="gram"))}),
            classes={"q1": "Mass"},
            provenance={},
        )
        rows = {row["dimension"]: row for row in score_rows(score_task(task, wrong))}
        assert rows["value"]["f1"] == 0.0
        assert rows["class"]["f1"] == 0.0

    def test_the_cost_panel_attributes_tokens_to_the_step_that_spent_them(self):
        from oold.agent.client import CallLog, TokenUsage

        log = CallLog()
        with log.timed("plan", "m") as sink:
            sink.append(TokenUsage(input_tokens=100, output_tokens=10))
        for _ in range(2):
            with log.timed("fill", "m") as sink:
                sink.append(TokenUsage(input_tokens=200, output_tokens=20))

        rows = {row["step"]: row for row in cost_rows(log)}
        assert rows["plan"]["calls"] == 1
        assert rows["fill"]["calls"] == 2
        assert rows["fill"]["input_tokens"] == 400
        assert cost_total(cost_rows(log))["total_tokens"] == 550

    def test_the_cost_panel_is_empty_without_a_log(self):
        assert cost_rows(None) == []
        assert cost_total([]) == {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "errors": 0,
        }

    def test_the_schema_panel_names_the_catalogue_trim_and_the_provider_cap(self):
        from oold.agent.provider import profile_for

        declared = answer_schema(157)
        sent = answer_schema(40)
        rows = {row["measure"]: row["value"] for row in schema_rows(None, declared=declared, sent=sent)}
        assert rows["entity properties after catalogue trim"] == "40 of 157"

        with_profile = schema_rows(None, declared=declared, sent=sent, profile=profile_for("anthropic"))
        assert {"measure": "optional properties anthropic accepts", "value": 24} in with_profile

    def test_the_schema_panel_reports_what_the_provider_subset_removed(self):
        from oold.agent.provider import prepare, profile_for

        schema = {
            "type": "object",
            "properties": {"a": {"type": "string", "minLength": 2}, "b": {"anyOf": [{"type": "string"}]}},
        }
        _, degradation = prepare(schema, profile_for("anthropic"), grounding=False)
        rows = {row["measure"]: row["value"] for row in schema_rows(degradation)}
        assert "anyOf" in degradation.dropped
        assert rows["constraints described"] >= 1
        assert "of" in rows["schema keywords"]
        assert [row["keyword"] for row in degradation_rows(degradation)] == sorted(degradation.dropped)

    def test_the_validation_panel_names_why_an_answer_fails(self):
        class Result:
            def __init__(self) -> None:
                self.payload = {"entities": []}
                self.invalid = ["at /entities/0: 'type' is a required property"]
                self.repairs = 2
                self.dropped = ["NotAClass"]
                self.unpinned = ["publisherImprint"]

        rows = {(row["check"], str(row["detail"])) for row in validation_rows(Result())}
        assert ("repair attempts", "2") in rows
        assert ("schema errors", "1") in rows
        assert ("gate dropped", "NotAClass") in rows
        assert ("reference left open", "publisherImprint") in rows

    def test_the_validation_panel_says_nothing_without_a_result(self):
        assert validation_rows(None) == []


class TestOrchestrationSwitch:
    """Every option has to build the agent it claims, through the adapter."""

    @pytest.mark.parametrize("orchestration", ORCHESTRATIONS)
    def test_each_option_builds_the_orchestration_it_names(self, orchestration):
        from oold.agent.enforcement import Orchestration

        from oold_llm_bench.runner.adapter import build_agent

        cell = build_cell(quantity_task(), Options(orchestration=orchestration, model="gpt-5-mini"))
        agent = build_agent(cell, ReplayClient(quantity_task()))
        assert agent.orchestration is Orchestration(orchestration)

    def test_an_unknown_orchestration_is_refused_before_a_call(self):
        with pytest.raises(ValueError, match="unknown orchestration"):
            Options(orchestration="recursive").condition()

    @pytest.mark.parametrize(
        "arm", ["schema-dump-catalog-flat-enforced", "catalog-flat-enforced", "schema-dump-catalog-enforced"]
    )
    def test_each_arm_builds_the_enforcement_it_names(self, arm):
        from oold.agent.enforcement import ARMS

        from oold_llm_bench.runner.adapter import build_enforcement

        cell = build_cell(quantity_task(), Options(arm=arm, model="gpt-5-mini"))
        enforcement = build_enforcement(cell)
        assert enforcement.decode_constraint is ARMS[arm].decode_constraint
        assert enforcement.schema_in_prompt is ARMS[arm].schema_in_prompt

    def test_the_condition_key_names_the_orchestration(self):
        assert "segmented" in Options(orchestration="segmented").label()
        assert "select_then_fill(k3)" in Options(orchestration="select_then_fill").label()


class TestReplayRuns:
    """A whole run, end to end, with the ground truth standing in for a model."""

    def test_single_shot_makes_one_call(self):
        task = linked_task()
        client = ReplayClient(task)
        outcome = run_once(build_cell(task, Options(orchestration="single_shot", model="gpt-5-mini")), client)
        assert outcome.error is None
        assert client.calls == ["extract"]
        assert len(list(outcome.result.calls)) == 1

    def test_segmented_makes_one_call_plus_one_per_shortlist(self):
        task = linked_task()
        client = ReplayClient(task)
        outcome = run_once(build_cell(task, Options(orchestration="segmented", model="gpt-5-mini")), client)
        assert client.calls == ["select", "fill", "fill"]
        steps = {row["step"]: row["calls"] for row in cost_rows(outcome.result.calls)}
        assert steps == {"plan": 1, "fill": 2}

    def test_only_segmented_can_express_the_edge(self):
        task = linked_task()
        graphs = {}
        for orchestration in ORCHESTRATIONS:
            options = Options(orchestration=orchestration, model="gpt-5-mini")
            graphs[orchestration] = drawn(run_once(build_cell(task, options), ReplayClient(task)))
        assert graphs["single_shot"]["links"] == 0
        assert graphs["select_then_fill"]["links"] == 0
        assert graphs["segmented"]["links"] == 1

    def test_an_omitted_target_leaves_the_edge_dangling(self):
        task = linked_task()
        options = Options(orchestration="segmented", model="gpt-5-mini")
        outcome = run_once(build_cell(task, options), ReplayClient(task, omit=["e2"]))
        assert len(outcome.result.dangling) == 1
        assert drawn(outcome)["dangling"] == 1
        assert drawn(outcome)["missing"] == 1

    def test_the_score_is_shown_for_a_task_with_ground_truth(self):
        task = linked_task()
        options = Options(orchestration="segmented", model="gpt-5-mini")
        outcome = run_once(build_cell(task, options), ReplayClient(task))
        assert outcome.score is not None
        assert outcome.primary == pytest.approx(1.0)

    def test_a_missing_entity_costs_the_score(self):
        task = linked_task()
        options = Options(orchestration="segmented", model="gpt-5-mini")
        outcome = run_once(build_cell(task, options), ReplayClient(task, omit=["e2"]))
        assert outcome.score is not None
        assert outcome.score.primary < 1.0

    def test_a_provider_failure_is_returned_and_not_raised(self):
        class Broken:
            model = "broken"

            def invoke(self, messages, *, response_format=None, strict=False) -> Never:
                raise RuntimeError("no route to host")

        outcome = run_once(build_cell(quantity_task(), Options(model="gpt-5-mini")), Broken())
        assert outcome.error is not None
        assert "no route to host" in outcome.error
        assert outcome.graph is None


class TestReplayClient:
    def test_the_plan_offers_every_entity_its_own_class(self):
        client = ReplayClient(linked_task())
        plan = client._plan()["entities"]
        assert [entry["id"] for entry in plan] == ["e1", "e2"]
        assert [entry["candidates"] for entry in plan] == [["CreativeWork"], ["Organization"]]

    def test_an_omitted_entity_is_not_planned(self):
        assert [entry["id"] for entry in ReplayClient(linked_task(), omit=["e1"])._plan()["entities"]] == ["e2"]

    def test_a_fill_call_reports_only_the_ids_it_was_asked_for(self):
        mentions = {"e1": "Alpha (e1)", "e2": "Beta (e2)"}
        prompt = 'Read the document and report these entities, each under the id given here: e1 ("Alpha (e1)"). Give'
        assert _ids_in(prompt, mentions) == {"e1"}

    def test_a_prose_arm_is_answered_with_the_document(self):
        task = quantity_task()
        client = ReplayClient(task)

        class Message:
            def __init__(self, content):
                self.content = content

        reply = client.invoke([Message("Answer in plain prose. State each entity.")])
        assert reply.text == task.document
        assert client.calls == ["prose"]


class TestCorpora:
    def test_a_missing_schema_collection_says_what_to_set(self, monkeypatch):
        monkeypatch.delenv(SCHEMAS_ENV, raising=False)
        with pytest.raises(MissingSchemas, match=SCHEMAS_ENV):
            schemas_directory()

    def test_a_directory_that_is_not_a_module_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.delenv(SCHEMAS_ENV, raising=False)
        (tmp_path / "notes.txt").write_text("nothing here", encoding="utf-8")
        with pytest.raises(MissingSchemas, match=re.escape("no *.schema.json")):
            schemas_directory(tmp_path)

    def test_the_default_corpus_is_schemaorg(self):
        assert CORPORA[0] == "schemaorg"

    def test_wiki_measurements_needs_no_second_directory(self):
        tasks = wiki_tasks(count=3)
        assert len(tasks) == 3
        assert all(task.expected[0].fields["value"].unit for task in tasks)
        assert is_scoreable(tasks[0])


SCHEMAS = os.environ.get(SCHEMAS_ENV)
needs_schemaorg = pytest.mark.skipif(
    not (SCHEMAS and Path(SCHEMAS).is_dir()),
    reason=f"set {SCHEMAS_ENV} to the generated schema.org module",
)


@needs_schemaorg
class TestSchemaOrgCorpus:
    """Only where the generated module is present, which a clone is not."""

    def test_the_corpus_reports_its_digest_and_its_pools(self):
        corpus = load_schemaorg()
        described = {key: value for key, value in corpus.describe().items() if isinstance(value, int)}
        assert described["files"] > 0
        assert described["describable"] > 0
        assert described["linked"] <= described["describable"]

    def test_a_linked_draw_gives_two_entities_and_an_edge(self):
        task = schemaorg_tasks(load_schemaorg(), count=1, seed=1)[0]
        assert len(task.expected) == 2
        edges = [
            value for instance in task.expected for value in instance.fields.values() if isinstance(value, Reference)
        ]
        assert len(edges) == 1

    def test_a_pasted_document_carries_no_score(self):
        corpus = load_schemaorg()
        task = paste_task(corpus, "Some prose nobody wrote a ground truth for.")
        assert not is_scoreable(task)
        assert task.catalogue

    def test_two_pastes_of_one_set_are_offered_the_same_classes(self):
        corpus = load_schemaorg()
        first = paste_task(corpus, "one", catalogue_set="Organization")
        second = paste_task(corpus, "two", catalogue_set="Organization")
        assert first.catalogue is not None
        assert first.catalogue == second.catalogue


@needs_schemaorg
class TestAPastedDocumentChoosesANamedSet:
    """A count is the wrong way to choose a catalogue.

    "Andrea works at Siemens" was shown 25 classes of 122, holding neither
    Person nor Organization. Thing was the only one that fitted, Thing declares
    no links, so the plan shortlisted it for both entities, they shared one
    fill call, and the edge had no slot to go in. The model was right and the
    catalogue was wrong.

    The sets are the ontology's own: the child of Thing each class descends
    through. Asking for "Organization" says what you want in a way that "25"
    cannot.
    """

    def test_every_describable_class_is_offered_by_default(self):
        corpus = load_schemaorg()
        task = paste_task(corpus, "Andrea works at Siemens")
        assert set(task.catalogue or ()) == set(corpus.catalogue)

    def test_the_classes_a_reader_would_pick_are_there(self):
        corpus = load_schemaorg()
        offered = set(paste_task(corpus, "Andrea works at Siemens").catalogue or ())
        assert {"Person", "Organization"} <= offered

    def test_a_named_set_narrows_to_its_branch(self):
        corpus = load_schemaorg()
        task = paste_task(corpus, "Andrea works at Siemens", catalogue_set="Organization")
        offered = set(task.catalogue or ())
        assert "Organization" in offered
        assert "Person" not in offered

    def test_several_sets_are_offered_together_because_a_sentence_crosses_them(self):
        corpus = load_schemaorg()
        sets = catalogue_sets(corpus)
        task = paste_task(corpus, "Andrea works at Siemens", catalogue_set=["Person", "Organization"])
        offered = set(task.catalogue or ())
        assert offered == set(sets["Person"]) | set(sets["Organization"])
        assert {"Person", "Organization"} <= offered
        assert len(offered) < len(sets[ALL_CLASSES])

    def test_the_offered_classes_do_not_depend_on_the_order_the_sets_were_named(self):
        corpus = load_schemaorg()
        one = paste_task(corpus, "x", catalogue_set=["Person", "Organization"]).catalogue
        other = paste_task(corpus, "x", catalogue_set=["Organization", "Person"]).catalogue
        assert one == other

    def test_a_paste_can_express_an_edge_because_its_schema_declares_one(self):
        """The reference draw is linked, so the link properties are in it.

        Without that the answer schema carries no link slot at all, and a
        relation the document states has nowhere to go but a text field.
        """
        corpus = load_schemaorg()
        task = paste_task(corpus, "Andrea works at Siemens", catalogue_set=["Person", "Organization"])
        slots = (task.answer_schema or {})["properties"]["entities"]["items"]["properties"]
        assert "worksFor" in slots
        assert "worksFor" in (task.property_ranges or {})

    def test_a_paste_is_offered_the_slot_its_entities_are_named_by(self):
        """The generated corpus withholds ``name`` and a pasted sentence needs it.

        Asked for "Andrea works at Siemens" without it, gpt-5-nano answered
        with ``additionalName`` on one run and ``address`` on the next.
        """
        corpus = load_schemaorg()
        task = paste_task(corpus, "Andrea works at Siemens", catalogue_set=["Person", "Organization"])
        slots = (task.answer_schema or {})["properties"]["entities"]["items"]["properties"]
        assert NAME_SLOT in slots
        assert all(NAME_SLOT in props for props in (task.branches or {}).values())
        assert all(f"  {NAME_SLOT} (text)" in text for text in (task.catalogue_text or {}).values())

    def test_the_sets_are_named_after_the_hierarchy(self):
        sets = catalogue_sets(load_schemaorg())
        assert sets[ALL_CLASSES] == load_schemaorg().catalogue
        assert {"Intangible", "CreativeWork", "Organization", "Person"} <= set(sets)

    def test_an_unknown_set_says_what_it_expected(self):
        corpus = load_schemaorg()
        with pytest.raises(KeyError, match="unknown class set"):
            paste_task(corpus, "anything", catalogue_set="NotASet")


@needs_schemaorg
class TestTheConditionTheInterfaceAdvertises:
    """What the pane promises has to be what the submission runs.

    Two choices are overruled past the widget. A pasted document is never
    trimmed, because the trim keeps the classes an expectation names and a
    paste has none, and offline replay never asks a judge, because asking one
    is a call. Both were applied where the run was built and nowhere where it
    was reported, so the pane advertised a trim of 25 over a paste that ran
    untrimmed and named a judge that decided nothing.
    """

    @pytest.fixture
    def playground(self):
        pytest.importorskip("panel")
        pytest.importorskip("panelini")
        from oold_llm_bench.playground.app import Playground

        return Playground(load_schemaorg(), task_count=2)

    def test_a_paste_is_not_advertised_with_a_trim_it_will_not_run(self, playground):
        pasted = paste_task(load_schemaorg(), "Andrea works at Siemens")
        assert playground.effective_options(pasted).catalogue_size is None
        assert "n25" not in playground._condition_html()

    def test_a_loaded_corpus_task_is_advertised_with_the_trim_it_runs(self, playground):
        """Published on the load and not only on the next control change,
        because loading a document is one of the ways the condition moves."""
        playground._on_load(None)
        assert playground.effective_options(playground.loaded).catalogue_size == 25
        assert "n25" in str(playground.condition_pane.object)

    def test_offline_replay_reports_the_judge_it_will_not_ask(self, playground):
        playground.judge_choice.value = "gpt-5-nano"
        assert playground.effective_options().judge_model is None
        assert 'data-judge="none"' in playground._condition_html()
        assert "gpt-5-nano" in playground._identity_html()

    def test_the_judge_is_reported_once_a_provider_can_be_asked(self, playground):
        from oold_llm_bench.playground.app import PROVIDER

        playground.judge_choice.value = "gpt-5-nano"
        playground.client_choice.value = PROVIDER
        assert playground.effective_options().judge_model == "gpt-5-nano"
        assert 'data-judge="gpt-5-nano"' in playground._condition_html()


class TestLinksAreNotIdentityEvidence:
    """A reference value is an id the pipeline invented, not a property."""

    def test_a_link_slot_is_left_out_of_the_comparison(self):
        """Two records of one thing carry different ids for the same relation.

        Comparing them asks whether two answers agreed about a name the plan
        handed out, which is not a question about identity. Excluded here so
        that the merge rule sees the literals only.
        """
        from oold_llm_bench.playground.graph import Edge, Node, _comparable, _link_props

        node = Node(
            id="e1",
            label="Andrea",
            kind="entity",
            class_path="Person",
            data={"id": "e1", "type": "Person", "name": "Andrea", "works_for": "e2"},
        )
        edges = [Edge(source="e1", target="e2", label="works_for", kind="link")]
        assert _link_props(edges, "e1") == frozenset({"works_for"})
        assert dict(_comparable("e1", node, _link_props(edges, "e1")).values) == {"name": "Andrea"}
        assert "works_for" in _comparable("e1", node).values

    def test_two_entities_agreeing_only_on_a_link_are_not_merged(self):
        """Without the exclusion this is an exact match on nothing real."""
        from oold_llm_bench.playground.identity import EXACT_MATCH, Comparable, decide

        left = Comparable(key="a", class_path="Person", values={"works_for": "e2"})
        right = Comparable(key="b", class_path="Person", values={"works_for": "e2"})
        assert decide(left, right, StubJudge({}, name="asked")).outcome == EXACT_MATCH

        bare = Comparable(key="a", class_path="Person", values={})
        judge = StubJudge({}, name="asked")
        assert decide(bare, Comparable(key="b", class_path="Person", values={}), judge).route == "judge"


class TestTheSmallerReadOutFaults:
    """Four places the interface said something it did not mean."""

    def test_coverage_of_nothing_is_not_full_coverage(self):
        """The number whose job is to make abstention visible, at rest."""
        from oold_llm_bench.playground.identity import coverage

        assert coverage([]) == 0.0

    def test_a_label_that_is_only_a_class_is_upgraded_by_a_real_name(self):
        """A fallback reads as a label while being the absence of one.

        A node drawn before any name arrived is labelled by its class, and
        treating that as present meant the node stayed "Organization" for
        good however many turns carried the real name.
        """
        from oold_llm_bench.playground.graph import Node, _fold, _is_named

        standing = Node(id="e2", label="Organization", kind="entity", class_path="Organization", data={"id": "e2"})
        named = Node(id="e2", label="Siemens", kind="entity", class_path="Organization", data={"id": "e2"})
        assert not _is_named(standing)
        assert _is_named(named)
        assert _fold(standing, named, []).label == "Siemens"
        assert _fold(named, standing, []).label == "Siemens"

    def test_a_property_stated_twice_is_still_compared(self):
        """`_data_of` keeps both readings as a list, and a list was skipped.

        The property most likely to settle an identity was the one most
        likely to be dropped, because a name emitted twice is exactly the
        shape a duplicate arrives in.
        """
        from oold_llm_bench.playground.graph import Node, _comparable

        node = Node(
            id="e1",
            label="Acme",
            kind="entity",
            class_path="Organization",
            data={"id": "e1", "type": "Organization", "name": ["Acme", "Acme Inc"]},
        )
        assert "name" in _comparable("e1", node).values

    def test_an_empty_submission_is_refused_rather_than_answered(self):
        """Running the example and drawing it as the user's answer is worse
        than saying there is nothing to run."""
        from oold_llm_bench.playground.corpora import SCHEMAS_ENV

        if not os.environ.get(SCHEMAS_ENV):
            pytest.skip(f"set {SCHEMAS_ENV}")
        pytest.importorskip("panel")
        pytest.importorskip("panelini")
        from oold_llm_bench.playground.app import Playground

        outcome = Playground(load_schemaorg(), task_count=2).extract("   ")
        assert outcome.error and "no document" in outcome.error
        assert outcome.graph is None
