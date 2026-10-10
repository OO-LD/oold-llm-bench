"""A wikilink between two articles this corpus already has entities for.

The fixture below is small and hand-written, the way
``test_corpus_wikidata_schemaorg.py`` keeps its own documents out of the
harvest: a link task needs a grounded corpus, a document and a set of
harvested rows, and all three are cheaper to control here than to carry a
slice of the real cache into a test.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from oold_llm_bench.corpus.linked_articles import (
    LINK_PROPERTY,
    LINKS_CACHE,
    GroundedLink,
    ground_links,
    load_linked_articles,
)
from oold_llm_bench.corpus.wikidata_schemaorg import NAME, read_grounded_corpus
from oold_llm_bench.experiments.corpora import DOCUMENTS_CACHE, documents_for
from oold_llm_bench.grading.triples import Reference
from oold_llm_bench.tasks.models import TaskRecord

CATALOGUE = {
    "CollegeOrUniversity": {
        "parents": [],
        "label": "CollegeOrUniversity",
        "description": "A university.",
        "slots": [[NAME, "text", True], ["city", "text", True]],
    },
    "Person": {
        "parents": [],
        "label": "Person",
        "description": "A person.",
        "slots": [[NAME, "text", True], ["occupation", "text", True]],
    },
}

DOCUMENT = (
    "Harvard University is a private research university primarily in Cambridge. "
    "It is deeply linked with Radcliffe College, historically known simply as "
    "Radcliffe, and its history includes the educator Charles W. Eliot."
)

ENTITIES = {
    "Q1": ("CollegeOrUniversity", "Harvard University", {NAME: ["Harvard University"], "city": ["Cambridge"]}),
    "Q2": ("CollegeOrUniversity", "Radcliffe College", {NAME: ["Radcliffe College"]}),
    "Q3": ("Person", "Charles William Eliot", {NAME: ["Charles William Eliot"]}),
    "Q4": ("CollegeOrUniversity", "Johns Hopkins University", {NAME: ["Johns Hopkins University"]}),
}

ROWS = [
    # Grounded, kept: two anchors for the same edge, which have to collapse
    # into one GroundedLink and not two.
    {"source": "Q1", "target": "Q2", "target_title": "Radcliffe College", "anchor": "Radcliffe College"},
    {"source": "Q1", "target": "Q2", "target_title": "Radcliffe College", "anchor": "Radcliffe"},
    # Grounded, kept: the second distinct target the source names.
    {"source": "Q1", "target": "Q3", "target_title": "Charles William Eliot", "anchor": "Charles W. Eliot"},
    # Never reaches the plain-text lead: an infobox-only anchor.
    {"source": "Q1", "target": "Q4", "target_title": "Johns Hopkins University", "anchor": "Founded 1636"},
    # Grounded, but the same QID on both ends: a redirect-shaped self-link.
    {"source": "Q1", "target": "Q1", "target_title": "Harvard University", "anchor": "Harvard"},
]


def _entity(qid: str, **changes):
    cls, title, facts = ENTITIES[qid]
    built = {
        "qid": qid,
        "cls": cls,
        "title": title,
        "revision": 1000 + int(qid.removeprefix("Q")),
        "sha256": hashlib.sha256(DOCUMENT.encode("utf-8")).hexdigest(),
        "facts": facts,
        "distractors": {},
    }
    built.update(changes)
    return built


def fixture_payload(**changes):
    payload = {
        "schema_version": "1",
        "name": "Wikidata-schema.org",
        "built_at": "2026-10-08",
        "retrieved_at": "2026-10-08",
        "sources": {"schemaorg": {}, "wikidata": {}},
        "licence": {"text": {}, "facts": {}, "note": ""},
        "draw": {"query": "", "per_class": {}},
        "catalogue": CATALOGUE,
        "properties": {},
        "grounding": {},
        "entities_in": 4,
        "resolved": 4,
        "excluded": {},
        "per_class": {"CollegeOrUniversity": 3, "Person": 1},
        "cap": None,
        "entities": [_entity(qid) for qid in ENTITIES],
    }
    payload.update(changes)
    return payload


def fixture(tmp_path, **changes):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(fixture_payload(**changes)), encoding="utf-8")
    return path


def documents(**changes):
    docs = {"wds-Q1": DOCUMENT}
    docs.update(changes)
    return docs


def entities_by_qid(tmp_path):
    return {e.qid: e for e in read_grounded_corpus(fixture(tmp_path)).entities}


class TestGroundLinks:
    """What a row needs before it becomes an edge this corpus will score."""

    def test_an_infobox_only_anchor_is_dropped(self, tmp_path):
        kept = ground_links(ROWS, entities_by_qid(tmp_path), documents())
        assert all(link.target != "Q4" for link in kept)

    def test_a_self_link_is_dropped(self, tmp_path):
        kept = ground_links(ROWS, entities_by_qid(tmp_path), documents())
        assert all(link.source != link.target for link in kept)

    def test_two_anchors_for_one_edge_collapse_into_one_link(self, tmp_path):
        kept = ground_links(ROWS, entities_by_qid(tmp_path), documents())
        to_q2 = [link for link in kept if link.target == "Q2"]
        assert len(to_q2) == 1
        assert set(to_q2[0].anchors) == {"Radcliffe College", "Radcliffe"}

    def test_a_second_distinct_target_is_its_own_link(self, tmp_path):
        kept = ground_links(ROWS, entities_by_qid(tmp_path), documents())
        assert {link.target for link in kept} == {"Q2", "Q3"}

    def test_exactly_two_links_survive_this_fixture(self, tmp_path):
        """Two grounded cross-entity edges: Q1->Q2 and Q1->Q3. Pinned so a
        change to the filter shows up here before it shows up on the real
        harvest."""
        kept = ground_links(ROWS, entities_by_qid(tmp_path), documents())
        assert kept == [
            GroundedLink(source="Q1", target="Q2", anchors=("Radcliffe College", "Radcliffe")),
            GroundedLink(source="Q1", target="Q3", anchors=("Charles W. Eliot",)),
        ]

    def test_a_missing_document_stops_the_load(self, tmp_path):
        with pytest.raises(ValueError, match="has no document"):
            ground_links(ROWS, entities_by_qid(tmp_path), {})

    def test_a_document_that_is_not_the_revision_measured_stops_the_load(self, tmp_path):
        changed = documents(**{"wds-Q1": DOCUMENT + " A later edit."})
        with pytest.raises(ValueError, match="the document given is not it"):
            ground_links(ROWS, entities_by_qid(tmp_path), changed)

    def test_a_target_the_corpus_no_longer_has_is_left_out_rather_than_guessed_at(self, tmp_path):
        rows = [*ROWS, {"source": "Q1", "target": "Q999", "target_title": "x", "anchor": "Radcliffe"}]
        kept = ground_links(rows, entities_by_qid(tmp_path), documents())
        assert all(link.target != "Q999" for link in kept)


class TestLoadLinkedArticles:
    """The task a kept link becomes."""

    def test_one_task_for_the_one_source_that_has_a_kept_link(self, tmp_path):
        tasks = load_linked_articles(documents(), corpus_path=fixture(tmp_path), rows=ROWS)
        assert [task.id for task in tasks] == ["wdl-Q1"]

    def test_the_source_keeps_its_own_facts(self, tmp_path):
        task = load_linked_articles(documents(), corpus_path=fixture(tmp_path), rows=ROWS)[0]
        source = next(i for i in task.expected if i.key == "wds-Q1")
        assert source.class_path == "CollegeOrUniversity"
        assert source.fields["city"] == "Cambridge"

    def test_the_link_field_holds_a_reference_per_distinct_target(self, tmp_path):
        task = load_linked_articles(documents(), corpus_path=fixture(tmp_path), rows=ROWS)[0]
        source = next(i for i in task.expected if i.key == "wds-Q1")
        links = source.fields[LINK_PROPERTY]
        assert isinstance(links, list)
        assert set(links) == {Reference(key="wds-Q2"), Reference(key="wds-Q3")}

    def test_a_reference_points_at_the_targets_own_catalogue_identity(self, tmp_path):
        """The target's key and class are the ones `load_entities` would
        publish for the same entity elsewhere, not a parallel scheme."""
        task = load_linked_articles(documents(), corpus_path=fixture(tmp_path), rows=ROWS)[0]
        target = next(i for i in task.expected if i.key == "wds-Q2")
        assert target.class_path == "CollegeOrUniversity"
        assert target.fields[NAME] == "Radcliffe College"

    def test_the_targets_mentions_are_the_anchors_the_source_named_it_by(self, tmp_path):
        task = load_linked_articles(documents(), corpus_path=fixture(tmp_path), rows=ROWS)[0]
        target = next(i for i in task.expected if i.key == "wds-Q2")
        assert set(target.mentions) == {"Radcliffe College", "Radcliffe"}

    def test_every_expected_key_is_unique(self, tmp_path):
        task = load_linked_articles(documents(), corpus_path=fixture(tmp_path), rows=ROWS)[0]
        keys = [i.key for i in task.expected]
        assert len(keys) == len(set(keys))

    def test_the_link_property_is_offered_on_every_branch(self, tmp_path):
        task = load_linked_articles(documents(), corpus_path=fixture(tmp_path), rows=ROWS)[0]
        assert LINK_PROPERTY in (task.branches or {})["CollegeOrUniversity"]
        assert LINK_PROPERTY in (task.branches or {})["Person"]

    def test_a_task_round_trips_through_task_record_validation(self, tmp_path):
        """A ``Reference`` survives a JSON round trip as ``{"key": ...}``, per
        ``ExpectedInstance.fields``'s own docstring, and not as the same
        Python object: what is checked here is that re-validating the dumped
        shape raises nothing and keeps that shape, not identity with the
        original instances."""
        task = load_linked_articles(documents(), corpus_path=fixture(tmp_path), rows=ROWS)[0]
        dumped = task.model_dump(mode="json")
        source = next(f for f in dumped["expected"] if f["key"] == "wds-Q1")
        assert {ref["key"] for ref in source["fields"][LINK_PROPERTY]} == {"wds-Q2", "wds-Q3"}
        reloaded = TaskRecord.model_validate(dumped)
        assert reloaded.id == task.id
        assert [i.key for i in reloaded.expected] == [i.key for i in task.expected]

    def test_no_source_without_a_kept_link_becomes_a_task(self, tmp_path):
        """Q2, Q3 and Q4 name no grounded link of their own in this fixture,
        so none of them is a source and the task set does not grow to one
        per entity the way the standalone corpus does."""
        tasks = load_linked_articles(documents(), corpus_path=fixture(tmp_path), rows=ROWS)
        assert {task.corpus.document_id for task in tasks} == {"Q1"}

    def test_the_licence_and_provenance_match_the_standalone_corpus(self, tmp_path):
        task = load_linked_articles(documents(), corpus_path=fixture(tmp_path), rows=ROWS)[0]
        assert task.corpus.licence == "CC BY-SA 4.0"
        assert task.corpus.exhaustive is False
        assert task.corpus.url and task.corpus.url.startswith("https://en.wikipedia.org/wiki/")


LINKS = LINKS_CACHE
needs_real_harvest = pytest.mark.skipif(
    not (LINKS.is_file() and DOCUMENTS_CACHE.is_file()),
    reason="run scripts/harvest_wikidata_links.py and scripts/fetch_wikidata_documents.py first",
)


@pytest.fixture(scope="module")
def rows():
    return json.loads(LINKS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def by_qid():
    return {e.qid: e for e in read_grounded_corpus().entities}


@pytest.fixture(scope="module")
def docs():
    return documents_for()


@pytest.fixture(scope="module")
def grounded_rows(rows, by_qid, docs):
    """Every harvested row whose anchor is in the document, self-links
    included: what :func:`ground_links` drops before it does, kept here once
    so every count below is read off the same pass."""
    from oold_llm_bench.corpus.wikidata_schemaorg import stated_in

    return [
        row
        for row in rows
        if row["source"] in by_qid
        and row["target"] in by_qid
        and stated_in(docs[by_qid[row["source"]].id], (row["anchor"],)) is not None
    ]


@pytest.fixture(scope="module")
def kept(rows, by_qid, docs):
    return ground_links(rows, by_qid, docs)


@needs_real_harvest
class TestAgainstTheRealHarvest:
    """The numbers in the module docstring, recomputed rather than trusted.

    Pinned as exact counts and not a tolerance: the harvest cache and the
    documents it grounds against are both static files on disk, so a changed
    count here means one of them moved, which is exactly what this is meant
    to catch, not noise to be tolerant of.
    """

    def test_the_harvest_still_holds_513_rows(self, rows):
        assert len(rows) == 513

    def test_380_of_513_rows_ground_in_the_plain_text_lead(self, grounded_rows):
        assert len(grounded_rows) == 380

    def test_31_of_the_grounded_rows_are_self_links(self, grounded_rows):
        assert sum(1 for row in grounded_rows if row["source"] == row["target"]) == 31

    def test_349_of_the_grounded_rows_are_between_two_distinct_entities(self, grounded_rows):
        assert sum(1 for row in grounded_rows if row["source"] != row["target"]) == 349

    def test_260_distinct_source_target_pairs_remain(self, kept):
        assert len(kept) == 260

    def test_208_sources_become_tasks(self, kept):
        assert len({link.source for link in kept}) == 208

    def test_every_kept_target_is_a_known_entity(self, kept, by_qid):
        assert all(link.target in by_qid for link in kept)

    def test_every_kept_pair_is_between_two_distinct_entities(self, kept):
        assert all(link.source != link.target for link in kept)
