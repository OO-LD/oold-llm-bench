"""Wikipedia leads as tasks, with Wikidata's facts as the truth.

Real text joined to a knowledge base, so the failures to test for are the ones
neither a generated corpus nor an annotated one can have: a fact the lead does
not state, a statement that was true in 1941, a document that is not the
revision the truth was measured on, and a corpus whose grounding rate nobody
can recompute from the file.

The end-to-end checks run on a fixture whose documents are written here. The
committed corpus carries no page text, because the leads are CC BY-SA 4.0, so
a test that needed them would need the network. What the real corpus can be
checked on without them is checked on it: the arithmetic, the catalogue, the
provenance and the control scores the build recorded.
"""

import hashlib
import json

import pytest

from oold_llm_bench.corpus.schemaorg import Kind
from oold_llm_bench.corpus.wikidata_schemaorg import (
    CLASS_QUERY,
    MIN_FACTS,
    MIN_SLOTS,
    NAME,
    TEXT_LICENCE,
    TIME_QUALIFIERS,
    classes_of,
    document_request,
    load_entities,
    read_documents,
    read_grounded_corpus,
    spelled_date,
    stated_in,
    truthy,
    written_forms,
)
from oold_llm_bench.grading import Dimension, TripleSet, make_triples
from oold_llm_bench.grading.score import score_task
from oold_llm_bench.results.record import Environment, ModelSpec
from oold_llm_bench.runner import Cell, Condition, ExperimentConfig
from oold_llm_bench.runner.adapter import _catalogue_of
from oold_llm_bench.runner.preflight import (
    CONTROL_CEILING,
    SPELLING_CEILING,
    control_scores,
    preflight,
)
from oold_llm_bench.tasks.models import Difficulty, Source, Split

CATALOGUE = {
    "Thing": {
        "parents": [],
        "label": "Thing",
        "description": "The most generic type of item.",
        "slots": [[NAME, "text", True]],
    },
    "CreativeWork": {
        "parents": ["Thing"],
        "label": "CreativeWork",
        "description": "The most generic kind of creative work.",
        "slots": [[NAME, "text", True], ["datePublished", "date", True], ["genre", "text", True]],
    },
    "Movie": {
        "parents": ["CreativeWork"],
        "label": "Movie",
        "description": "A movie.",
        "slots": [
            [NAME, "text", True],
            ["datePublished", "date", True],
            ["director", "text", True],
            ["countryOfOrigin", "text", True],
            ["genre", "text", True],
            ["duration", "duration", True],
        ],
    },
    "Person": {
        "parents": ["Thing"],
        "label": "Person",
        "description": "A person, alive, dead, undead or fictional.",
        "slots": [
            [NAME, "text", True],
            ["birthDate", "date", True],
            ["birthPlace", "text", True],
            ["deathDate", "date", True],
            ["nationality", "text", True],
        ],
    },
    "MusicAlbum": {
        "parents": ["CreativeWork"],
        "label": "MusicAlbum",
        "description": "A collection of music tracks.",
        "slots": [
            [NAME, "text", True],
            ["datePublished", "date", True],
            ["byArtist", "text", True],
            ["numTracks", "integer", True],
        ],
    },
}
"""A catalogue the way the build writes one: every slot is the class's own.

Marked own because the renderer shows a class's own slots in preference to its
inherited ones, and ``name`` is declared on ``Thing``. An inherited slot
withheld from the entry of the class that needs it is a slot no document could
be answered on, since the document names no property.
"""

DOCUMENTS = {
    "wds-Q11": (
        "Harbour Lights is a 1974 drama film from Portugal directed by Ines Varga. "
        "Shot over a single winter in the fishing town of Nazare, it follows a widowed "
        "net-mender through the season that closes the harbour. It was released on "
        "20 June 1974 and ran for eleven weeks."
    ),
    "wds-Q12": (
        "The Quiet Ledger is a 1988 comedy film from Canada directed by Olaf Brennt. "
        "A municipal auditor is sent to a mining town and finds the books in order and "
        "the town emptying. It opened on 3 March 1988 to small houses and was revived "
        "a decade later."
    ),
    "wds-Q13": (
        "Pale Signal is a 2003 thriller film from Iceland directed by Hedda Larusson. "
        "A coastguard radio operator hears a distress call that no vessel admits to "
        "sending. Released on 11 September 2003, it was the first Icelandic feature "
        "shot entirely at sea."
    ),
    "wds-Q21": (
        "Ines Varga was a film director born on 5 August 1930 in Coimbra. She trained "
        "as a stage designer, moved into television in the late fifties, and directed "
        "four features before leaving the industry. She died on 2 January 1998."
    ),
    "wds-Q22": (
        "Olaf Brennt was a screenwriter and director born on 14 April 1941 in Winnipeg. "
        "He wrote for radio for a decade before his first feature and taught at a film "
        "school for the rest of his working life. He died on 30 November 2011."
    ),
    "wds-Q23": (
        "Hedda Larusson was a director and editor born on 9 July 1962 in Akureyri. She "
        "cut documentaries for the national broadcaster, then directed three features, "
        "each of them at sea. She retired in 2015."
    ),
}
"""Prose written for this test, not taken from anywhere.

The corpus publishes no page text, so a fixture that quoted an article would
be the one place a lead reached the repository. Each document states exactly
the facts its record claims and none of another record's, which is what lets
the wrong-document control be asserted at zero rather than merely observed
near it.
"""

FACTS = {
    "wds-Q11": (
        "Movie",
        "Harbour Lights",
        {
            NAME: ["Harbour Lights"],
            "director": ["Ines Varga"],
            "datePublished": ["1974-06-20"],
            "countryOfOrigin": ["Portugal"],
        },
        {"genre": ["drame"]},
    ),
    "wds-Q12": (
        "Movie",
        "The Quiet Ledger",
        {
            NAME: ["The Quiet Ledger"],
            "director": ["Olaf Brennt"],
            "datePublished": ["1988-03-03"],
            "countryOfOrigin": ["Canada"],
        },
        {"genre": ["satire"]},
    ),
    "wds-Q13": (
        "Movie",
        "Pale Signal",
        {
            NAME: ["Pale Signal"],
            "director": ["Hedda Larusson"],
            "datePublished": ["2003-09-11"],
            "countryOfOrigin": ["Iceland"],
        },
        {},
    ),
    "wds-Q21": (
        "Person",
        "Ines Varga",
        {NAME: ["Ines Varga"], "birthDate": ["1930-08-05"], "birthPlace": ["Coimbra"], "deathDate": ["1998-01-02"]},
        {"nationality": ["Portuguese Republic"]},
    ),
    "wds-Q22": (
        "Person",
        "Olaf Brennt",
        {NAME: ["Olaf Brennt"], "birthDate": ["1941-04-14"], "birthPlace": ["Winnipeg"], "deathDate": ["2011-11-30"]},
        {},
    ),
    "wds-Q23": (
        "Person",
        "Hedda Larusson",
        {NAME: ["Hedda Larusson"], "birthDate": ["1962-07-09"], "birthPlace": ["Akureyri"]},
        {"nationality": ["Republic of Iceland"]},
    ),
}

CELL_MINIMUM = 120
"""Tasks a cell needs before its number is worth reporting."""


def entity(record_id: str, **changes):
    cls, title, facts, distractors = FACTS[record_id]
    built = {
        "qid": record_id.removeprefix("wds-"),
        "cls": cls,
        "title": title,
        "revision": 1000 + int(record_id.removeprefix("wds-Q")),
        "sha256": hashlib.sha256(DOCUMENTS[record_id].encode("utf-8")).hexdigest(),
        "facts": facts,
        "distractors": distractors,
    }
    built.update(changes)
    return built


def fixture_payload(**changes):
    entities = [entity(key) for key in sorted(FACTS)]
    payload = {
        "schema_version": "1",
        "name": "Wikidata-schema.org",
        "built_at": "2026-10-03",
        "retrieved_at": "2026-10-03",
        "sources": {"schemaorg": {"sha256": "x"}, "wikidata": {"equivalent_class": "P1709"}},
        "licence": {"text": {"name": TEXT_LICENCE}, "facts": {"name": "CC0 1.0"}, "note": "CC BY-SA 4.0"},
        "draw": {"query": CLASS_QUERY, "per_class": {}},
        "catalogue": CATALOGUE,
        "properties": {NAME: {"kind": "text", "candidates": 6, "stated": 6, "rate": 1.0}},
        "grounding": {
            "candidates": 30,
            "stated": 22,
            "rate": 0.7333,
            "published": {"candidates": 24, "stated": 22},
            "per_class": {
                "Movie": {"entities": 4, "candidates": 16, "stated": 12, "rate": 0.75, "reached_minimum": 3},
                "Person": {"entities": 4, "candidates": 14, "stated": 10, "rate": 0.7143, "reached_minimum": 3},
            },
        },
        "entities_in": 8,
        "resolved": 6,
        "excluded": {"the lead states fewer than three facts over three slots": 2},
        "per_class": {"Movie": 3, "Person": 3},
        "cap": None,
        "min_facts": MIN_FACTS,
        "min_slots": MIN_SLOTS,
        "controls": {
            "empty": 0.0,
            "random": 0.0,
            "spelling": 0.0,
            "wrong-document": 0.0,
            "per_class": {"Movie": {"wrong-document": 0.0}, "Person": {"wrong-document": 0.0}},
        },
        "guessable": {"inLanguage": {"commonest": "English", "share": 0.71}},
        "entities": entities,
    }
    payload.update(changes)
    return payload


def fixture(tmp_path, **changes):
    path = tmp_path / "wikidata_schemaorg.json"
    path.write_text(json.dumps(fixture_payload(**changes)), encoding="utf-8")
    return path


def one(tmp_path, **changes):
    """A corpus file holding a single entity, with whatever this test needs."""
    record = entity("wds-Q11", **changes)
    payload = fixture_payload(
        entities=[record],
        entities_in=2,
        resolved=1,
        per_class={"Movie": 1},
        excluded={"the lead states fewer than three facts over three slots": 1},
    )
    path = tmp_path / "one.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def perfect_answer(task) -> TripleSet:
    """What a model that read the lead correctly would produce."""
    triples = frozenset()
    classes = {}
    for instance in task.expected:
        triples |= make_triples(instance.key, instance.fields)
        classes[instance.key] = instance.class_path
    return TripleSet(triples=triples, classes=classes, provenance={})


@pytest.fixture(scope="module")
def built():
    return read_grounded_corpus()


@pytest.fixture
def tasks(tmp_path):
    return load_entities(DOCUMENTS, path=fixture(tmp_path))


def _committed() -> dict:
    from oold_llm_bench.corpus.wikidata_schemaorg import CORPUS_PATH

    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def written(tmp_path, **changes):
    """The committed corpus with one field changed, so a guard can be tried."""
    payload = _committed()
    payload.update(changes)
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestNothingIsLostBetweenTheDrawAndTheTasks:
    """A draw of several thousand items becoming a few hundred tasks is only
    acceptable if every one of the missing carries a stated reason."""

    def test_every_entity_is_either_resolved_or_excluded_for_a_stated_reason(self, built):
        assert built.resolved + sum(built.excluded.values()) == built.entities_in

    def test_every_declared_reason_removed_something(self, built):
        assert built.excluded
        assert all(count > 0 for count in built.excluded.values())
        assert all(reason.strip() for reason in built.excluded)

    def test_the_per_class_counts_account_for_everything_that_resolved(self, built):
        assert sum(built.per_class.values()) == built.resolved

    def test_the_file_never_holds_more_than_resolved(self, built):
        assert len(built.entities) <= built.resolved

    def test_the_cap_is_declared_and_is_what_the_file_obeys(self, built):
        """At most the cap, and under it where balancing the sample's values
        passed an entity over. See ``VALUE_SHARE`` in the build."""
        assert built.cap is not None
        for cls, total in built.per_class.items():
            held = sum(1 for e in built.entities if e.cls == cls)
            assert 0 < held <= min(total, built.cap)

    def test_the_cap_reports_what_it_held_back(self, built):
        assert built.dropped_by_cap == built.resolved - len(built.entities)

    def test_an_entity_that_left_without_a_reason_is_refused(self, tmp_path):
        path = written(tmp_path, resolved=99999)
        with pytest.raises(ValueError, match="without a declared reason"):
            read_grounded_corpus(path)

    def test_per_class_counts_that_do_not_add_up_are_refused(self, tmp_path):
        payload = _committed()
        first = next(iter(payload["per_class"]))
        path = written(tmp_path, per_class={**payload["per_class"], first: 1})
        with pytest.raises(ValueError, match="per-class counts sum to"):
            read_grounded_corpus(path)

    def test_a_file_holding_more_than_it_resolved_is_refused(self, tmp_path):
        payload = _committed()
        excluded = dict(payload["excluded"])
        reason = next(iter(excluded))
        excluded[reason] += payload["resolved"] - 1
        cls = next(iter(payload["per_class"]))
        path = written(tmp_path, resolved=1, per_class={cls: 1}, excluded=excluded)
        with pytest.raises(ValueError, match="but only 1 resolved"):
            read_grounded_corpus(path)

    def test_a_class_the_catalogue_does_not_offer_is_refused(self, tmp_path):
        payload = _committed()
        entities = [dict(payload["entities"][0], cls="Enthusiasm")]
        path = written(tmp_path, entities=entities)
        with pytest.raises(ValueError, match="does not offer classes the corpus answers"):
            read_grounded_corpus(path)


class TestTheGroundTruthIsWhatTheLeadStates:
    """A statement the document does not make has no correct answer, so it is
    carried apart from the truth and never scored."""

    def test_the_grounding_rate_is_in_the_record_and_not_in_a_comment(self, built):
        assert 0.0 < built.grounding["rate"] <= 1.0
        assert built.grounding["stated"] + 0 <= built.grounding["candidates"]

    def test_every_class_reports_its_own_grounding_rate(self, built):
        per_class = built.grounding["per_class"]
        assert set(built.per_class) <= set(per_class)
        for report in per_class.values():
            assert 0.0 <= report["rate"] <= 1.0
            assert 0.0 <= report["share_reaching_minimum"] <= 1.0

    def test_every_property_reports_its_own_grounding_rate(self, built):
        assert built.properties
        for report in built.properties.values():
            assert report["stated"] <= report["candidates"]
            assert 0.0 <= report["rate"] <= 1.0

    def test_the_published_rate_can_be_recomputed_from_the_file(self, built):
        """A statistic nobody can recompute is a claim, not a measurement.

        The published rate is over the entities and the slots the file holds;
        the rate beside it is over every statement the mapping offered, and
        the two differ because a guessable slot left after it was measured."""
        published = built.grounding["published"]
        assert sum(e.stated for e in built.entities) == published["stated"]
        assert sum(e.candidates for e in built.entities) == published["candidates"]
        assert 0 < published["stated"] <= published["candidates"]

    def test_no_distractor_is_also_a_fact(self, built):
        for item in built.entities:
            for prop, values in item.distractors.items():
                assert not set(map(str, values)) & set(map(str, item.facts.get(prop, [])))

    def test_every_entity_clears_the_minimum(self, built):
        for item in built.entities:
            assert item.stated >= MIN_FACTS
            assert len(item.facts) >= MIN_SLOTS

    def test_a_distractor_never_reaches_a_task(self, tasks):
        for task in tasks:
            _cls, _title, facts, distractors = FACTS[task.id]
            assert set(task.expected[0].fields) == set(facts)
            assert not set(task.expected[0].fields) & set(distractors)

    def test_every_expected_value_is_stated_by_its_own_document(self, tasks):
        for task in tasks:
            for prop, value in task.expected[0].fields.items():
                kind = Kind.DATE if prop.endswith("Date") or prop == "datePublished" else Kind.TEXT
                for one_value in value if isinstance(value, list) else [value]:
                    assert stated_in(task.document, written_forms(one_value, kind)), (task.id, prop, one_value)

    def test_an_entity_under_the_minimum_stops_the_load(self, tmp_path):
        path = one(tmp_path, facts={NAME: ["Harbour Lights"], "director": ["Ines Varga"]})
        with pytest.raises(ValueError, match="under the minimum"):
            load_entities(DOCUMENTS, path=path)

    def test_a_fact_on_a_slot_the_class_does_not_declare_stops_the_load(self, tmp_path):
        facts = dict(FACTS["wds-Q11"][2], isbn=["978-3-16-148410-0"])
        path = one(tmp_path, facts=facts)
        with pytest.raises(ValueError, match="which the class does not declare"):
            load_entities(DOCUMENTS, path=path)


class TestTheTimeScopedStatement:
    """Vilnius carries 18 country statements and schema.org has one slot. The
    rule that stops the wrong one becoming truth is tested, not assumed."""

    def test_the_three_qualifiers_are_the_declared_ones(self):
        assert TIME_QUALIFIERS == ("P580", "P582", "P585")

    def test_a_statement_scoped_to_a_period_is_dropped(self):
        claims = [
            {"value": "Q37", "rank": "normal", "qualifiers": []},
            {"value": "Q15180", "rank": "normal", "qualifiers": ["P580", "P582"]},
            {"value": "Q36", "rank": "normal", "qualifiers": ["P585"]},
        ]
        assert [c["value"] for c in truthy(claims)] == ["Q37"]

    def test_a_preferred_statement_wins_over_a_normal_one(self):
        claims = [
            {"value": "Q1", "rank": "normal", "qualifiers": []},
            {"value": "Q2", "rank": "preferred", "qualifiers": []},
        ]
        assert [c["value"] for c in truthy(claims)] == ["Q2"]

    def test_a_preferred_statement_that_is_time_scoped_leaves_nothing(self):
        """Preferred and scoped is the Vilnius shape, and the right answer is
        no value rather than the normal-ranked one underneath it."""
        claims = [
            {"value": "Q1", "rank": "normal", "qualifiers": []},
            {"value": "Q2", "rank": "preferred", "qualifiers": ["P580"]},
        ]
        assert truthy(claims) == []

    def test_a_deprecated_statement_is_never_read(self):
        claims = [
            {"value": "Q1", "rank": "deprecated", "qualifiers": []},
            {"value": "Q2", "rank": "normal", "qualifiers": []},
        ]
        assert [c["value"] for c in truthy(claims)] == ["Q2"]


class TestTheGroundingMatcher:
    """It decides the whole of the ground truth, so what it does and does not
    count is tested rather than described."""

    def test_a_date_is_matched_in_the_forms_a_lead_writes(self):
        forms = spelled_date("1930-08-05")
        assert "5 August 1930" in forms
        assert "August 5, 1930" in forms
        assert "1930" in forms

    def test_a_date_stated_only_as_a_year_counts(self):
        assert stated_in("The film opened in 1974 to small houses.", spelled_date("1974-06-20"))

    def test_a_month_precision_date_writes_no_day(self):
        assert spelled_date("1930-08") == ("1930-08", "1930", "August 1930")

    def test_a_value_is_matched_on_word_boundaries(self):
        assert stated_in("Born in Albania.", ("Albania",))
        assert not stated_in("Born in Albania.", ("Alban",))

    def test_a_title_in_brackets_still_anchors(self):
        """A plain word-boundary regex never matches after a bracket, and a
        bracketed disambiguator is how half of Wikipedia titles end."""
        assert stated_in("See Chinatown (1974 film) for the sequel.", ("Chinatown (1974 film)",))

    def test_an_alias_from_wikidata_is_matched(self):
        forms = written_forms("United Kingdom", Kind.TEXT, {"United Kingdom": ["British", "UK"]})
        assert stated_in("a British-Austrian drama film", forms)

    def test_the_form_the_document_states_is_what_comes_back(self):
        """It is the answer for a text slot. Grading "British" against
        "United Kingdom" marks a correct extraction wrong."""
        forms = written_forms("United Kingdom", Kind.TEXT, {"United Kingdom": ["British", "UK"]})
        assert stated_in("a British-Austrian drama film", forms) == "British"
        assert stated_in("made in the United Kingdom", forms) == "United Kingdom"

    def test_a_value_the_document_does_not_state_comes_back_as_nothing(self):
        assert stated_in("a French drama film", written_forms("United Kingdom", Kind.TEXT)) is None

    def test_a_form_shorter_than_three_characters_is_refused(self):
        assert written_forms("US", Kind.TEXT, {"US": ["U"]}) == ()

    def test_a_number_is_matched_with_and_without_its_grouping(self):
        assert stated_in("It sold 1,200,000 copies.", written_forms(1200000, Kind.INTEGER))
        assert stated_in("It ran 142 minutes.", written_forms(142, Kind.INTEGER))

    def test_a_dash_the_document_writes_is_folded(self):
        """The dash is an escape, so what is being tested is unambiguous in
        the source and survives a reformat of this file."""
        assert stated_in("the Austro\u2013Hungarian empire", ("Austro-Hungarian",))


class TestTheDocumentsAreNotPublished:
    """The leads are CC BY-SA 4.0 and this repository is Apache-2.0, so the
    file carries what identifies a lead and never the lead."""

    def test_no_record_carries_page_text(self, built):
        """A record is provenance and truth. The longest string in one is a
        url or a title, and a lead is several hundred characters."""
        for item in built.entities:
            assert len(item.url) < 300
            assert len(item.title) < 200

    def test_every_record_carries_the_revision_it_was_measured_on(self, built):
        for item in built.entities:
            assert item.revision > 0
            assert len(item.sha256) == 64

    def test_the_licence_note_says_what_is_published_and_what_is_not(self, built):
        assert built.licence["facts"]["name"] == "CC0 1.0"
        assert built.licence["text"]["name"] == TEXT_LICENCE
        assert "not published here" in built.licence["note"]

    def test_the_documents_can_be_asked_for_by_revision_id(self, built):
        calls = document_request(built.entities[:25])
        assert calls
        assert all("revids=" in call for call in calls)
        assert str(built.entities[0].revision) in calls[0]

    def test_a_document_store_round_trips(self, tmp_path):
        """What the build writes beside its cache, and what a third party
        rebuilds from the revision ids. Not committed, so the only thing that
        has to hold is that reading it back gives what was written."""
        path = tmp_path / "documents.json"
        path.write_text(json.dumps(DOCUMENTS, ensure_ascii=False), encoding="utf-8")
        assert read_documents(path) == DOCUMENTS

    def test_a_document_that_is_not_the_revision_measured_stops_the_load(self, tmp_path):
        documents = dict(DOCUMENTS, **{"wds-Q11": DOCUMENTS["wds-Q11"] + " It was restored in 2011."})
        with pytest.raises(ValueError, match="the document supplied is not it"):
            load_entities(documents, path=fixture(tmp_path))

    def test_a_missing_document_stops_the_load(self, tmp_path):
        documents = {k: v for k, v in DOCUMENTS.items() if k != "wds-Q21"}
        with pytest.raises(ValueError, match="has no document"):
            load_entities(documents, path=fixture(tmp_path))


class TestProvenance:
    """A report has to tell these from generated prose off the record alone,
    and a reader has to be able to find the article they came from."""

    def test_every_record_is_marked_as_coming_from_a_bulk_dataset(self, tasks):
        assert {t.corpus.source for t in tasks} == {Source.BULK}

    def test_no_record_claims_to_be_synthetic(self, tasks):
        assert Source.SYNTHETIC not in {t.corpus.source for t in tasks}

    def test_every_record_names_the_article_it_came_from(self, tasks):
        for task in tasks:
            assert (task.corpus.url or "").startswith("https://en.wikipedia.org/wiki/")

    def test_every_record_names_the_wikidata_item(self, tasks):
        for task in tasks:
            assert task.corpus.document_id.startswith("Q")
            assert f"qid={task.corpus.document_id}" in (task.notes or "")

    def test_every_record_carries_the_share_alike_licence(self, tasks):
        assert {t.corpus.licence for t in tasks} == {TEXT_LICENCE}

    def test_the_content_hash_covers_the_document(self, tasks):
        for task in tasks:
            assert task.corpus.content_hash == hashlib.sha256(task.document.encode("utf-8")).hexdigest()

    def test_the_notes_carry_the_corpus_the_revision_and_the_grounding(self, tasks):
        for task in tasks:
            assert "corpus=wikidata-schemaorg" in (task.notes or "")
            assert "revision=" in (task.notes or "")
            assert "grounded=" in (task.notes or "")

    def test_the_register_is_recorded_as_hard(self, tasks):
        assert {t.difficulty for t in tasks} == {Difficulty.HARD}

    def test_the_split_can_be_chosen(self, tmp_path):
        taken = load_entities(DOCUMENTS, path=fixture(tmp_path), split=Split.TEST)
        assert {t.split for t in taken} == {Split.TEST}

    def test_the_draw_query_is_recorded_with_the_corpus(self, built):
        assert "wdt:P31/wdt:P279*" in built.draw["query"]


class TestTheCatalogue:
    """A document that names neither its class nor its properties is only
    answerable from the catalogue, so what the catalogue carries is the task."""

    def test_every_class_the_corpus_answers_is_offered(self, tasks):
        offered = set(tasks[0].catalogue or ())
        assert {t.expected[0].class_path for t in tasks} <= offered

    def test_the_catalogue_is_described_by_default(self, tasks):
        described = tasks[0].catalogue_text or {}
        assert "A movie." in described["Movie"]
        assert "director" in described["Movie"]

    def test_a_catalogue_entry_shows_every_slot_the_corpus_can_fill(self, tasks):
        """Nothing is withheld. The generated corpus withholds whatever the
        document's signal was drawn from; a lead was drawn from nothing."""
        entry = (tasks[0].catalogue_text or {})["Person"]
        for slot in ("birthDate", "birthPlace", "deathDate", NAME):
            assert slot in entry

    def test_bare_identifiers_can_be_asked_for(self, tmp_path):
        taken = load_entities(DOCUMENTS, path=fixture(tmp_path), describe_catalogue=False)
        assert taken[0].catalogue_text is None

    def test_the_answer_schema_covers_every_offered_property(self, tasks):
        properties = tasks[0].answer_schema["properties"]["entities"]["items"]["properties"]
        assert "type" in properties
        for task in tasks:
            assert set(task.expected[0].fields) <= set(properties)

    def test_each_class_narrows_the_union_to_its_own_properties(self, tasks):
        branches = tasks[0].branches or {}
        assert "birthDate" in branches["Person"]
        assert "birthDate" not in branches["Movie"]

    def test_the_hierarchy_is_carried(self, tasks):
        assert (tasks[0].class_parents or {})["Movie"] == ["CreativeWork"]

    def test_a_caller_may_choose_the_catalogue(self, tmp_path):
        taken = load_entities(DOCUMENTS, path=fixture(tmp_path), catalogue=("Movie", "Person"))
        assert taken[0].catalogue == ["Movie", "Person"]

    def test_a_caller_may_take_a_fixed_number_per_class(self, tmp_path):
        taken = load_entities(DOCUMENTS, path=fixture(tmp_path), limit=2)
        counts: dict[str, int] = {}
        for task in taken:
            counts[task.expected[0].class_path] = counts.get(task.expected[0].class_path, 0) + 1
        assert max(counts.values()) == 2

    def test_the_committed_catalogue_offers_more_classes_than_it_answers(self, built):
        """Class selection has to be a choice. A catalogue holding only the
        classes the corpus answers would make the wrong answer unreachable."""
        assert len(built.catalogue) > len(built.per_class)

    def test_every_committed_class_declares_enough_to_be_describable(self, built):
        for name, entry in built.catalogue.items():
            assert len(entry["slots"]) >= MIN_SLOTS, name

    def test_the_committed_catalogue_rebuilds_into_classes(self, built):
        classes = classes_of(built)
        assert {cls.name for cls in classes} == set(built.catalogue)
        assert all(cls.slots for cls in classes)


class TestTheCatalogueOrder:
    """The ordering must not carry the answer. It did once, for a whole run."""

    def cells(self, tasks, size=4):
        model = ModelSpec(model="m", provider_profile="openai", model_version="1")
        return [
            Cell(
                condition=Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=size),
                model=model,
                task=task,
                repetition=1,
            )
            for task in tasks
        ]

    def positions(self, tasks, size=4):
        found = []
        for cell in self.cells(tasks, size):
            offered = _catalogue_of(cell) or ()
            found.append(offered.index(cell.task.expected[0].class_path))
        return found

    def test_the_answer_is_not_always_first(self, tasks):
        assert set(self.positions(tasks)) != {0}

    def test_the_class_the_document_needs_is_always_offered(self, tasks):
        for cell in self.cells(tasks, size=1):
            assert cell.task.expected[0].class_path in (_catalogue_of(cell) or ())

    def test_the_order_is_stable_for_one_task(self, tasks):
        cell = self.cells(tasks)[0]
        assert _catalogue_of(cell) == _catalogue_of(cell)


class TestTheSeamCloses:
    """GroundedCorpus to task to score, with no model involved."""

    def test_a_perfect_answer_scores_one(self, tasks):
        for task in tasks:
            result = score_task(task, perfect_answer(task))
            assert result.primary == pytest.approx(1.0)
            assert result.dimensions[Dimension.CLASS].f1 == pytest.approx(1.0)

    def test_an_empty_answer_scores_zero(self, tasks):
        empty = TripleSet(triples=frozenset(), classes={}, provenance={})
        assert score_task(tasks[0], empty).primary == 0.0

    def test_a_perfect_answer_grounds_every_text_value_and_no_date(self, tasks):
        """A text answer is a string the lead contains, which is what makes
        this corpus different from one whose answers are canonical forms.

        A date is the exception and it is by construction: the answer is
        ``1974-06-20`` and the lead writes "20 June 1974". The grounded
        dimension is a rate and not a score for exactly this reason, and this
        is the rate on this corpus.
        """
        kinds = {slot[0]: slot[1] for entry in CATALOGUE.values() for slot in entry["slots"]}
        for task in tasks:
            fields = task.expected[0].fields
            counted = {
                kind: sum(len(v) if isinstance(v, list) else 1 for p, v in fields.items() if kinds[p] == kind)
                for kind in ("text", "date")
            }
            grounded = score_task(task, perfect_answer(task)).dimensions[Dimension.GROUNDED]
            assert grounded.true_positives == counted["text"]
            assert grounded.false_positives == counted["date"]

    def test_every_negative_control_stays_at_its_floor(self, tasks):
        """All four measure 0.00, the wrong-document control included.

        It is the one that can fail on real text: it answers another document
        of this corpus correctly, so a property whose values repeat across
        documents, a genre or a language, would score for it. The build
        verifies this on the committed corpus and refuses to write one that
        does not clear it.
        """
        grid = ExperimentConfig(
            name="wikidata-schemaorg",
            conditions=[Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=5)],
            models=[ModelSpec(model="m", provider_profile="openai", model_version="1")],
            tasks=tasks,
            runs_per_cell=1,
        )
        scores = control_scores(grid, Environment(benchmark_version="0.1.0", benchmark_sha="abc"))
        assert scores == {"empty": 0.0, "random": 0.0, "spelling": 0.0, "wrong-document": 0.0}

    def test_the_preflight_clears_on_the_dev_split(self, tasks):
        grid = ExperimentConfig(
            name="wikidata-schemaorg",
            conditions=[Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=5)],
            models=[ModelSpec(model="m", provider_profile="openai", model_version="1")],
            tasks=tasks,
            runs_per_cell=1,
        )
        check = preflight(grid, Environment(benchmark_version="0.1.0", benchmark_sha="abc"))
        assert check.clear, check.blocked

    def test_the_build_recorded_what_the_controls_scored_on_the_real_corpus(self, built):
        """The committed corpus carries no documents, so the controls cannot
        be rerun here. The build runs them on the corpus as built, refuses to
        write one that fails, and records what it passed with."""
        assert set(built.controls) == {"empty", "random", "spelling", "wrong-document"}
        for name, score in built.controls.items():
            assert score <= (SPELLING_CEILING if name == "spelling" else CONTROL_CEILING), name

    def test_every_class_was_controlled_as_the_grid_it_will_be(self, built):
        """A cell is one class, and the wrong-document control answers another
        document of the same grid. Over ten classes a value two films share is
        diluted by nine that do not have it; over one it is not, and that is
        where the number was 0.111 before the corpus was fixed."""
        assert set(built.controls_per_class) == set(built.per_class)
        for cls, scores in built.controls_per_class.items():
            for name, score in scores.items():
                assert score <= (SPELLING_CEILING if name == "spelling" else CONTROL_CEILING), (cls, name)


class TestWhatAClassGivesAway:
    """A slot answerable from the class is not an extraction. Only real text
    has the problem and only one control can see it."""

    def test_a_guessable_property_is_offered_by_no_class(self, built):
        assert built.guessable
        offered = {slot[0] for entry in built.catalogue.values() for slot in entry["slots"]}
        assert not offered & set(built.guessable)

    def test_no_entity_is_graded_on_one(self, built):
        for item in built.entities:
            assert not set(item.facts) & set(built.guessable)

    def test_each_one_names_the_value_and_the_share_that_cost_it_the_slot(self, built):
        for prop, report in built.guessable.items():
            assert report["commonest"], prop
            assert 0.0 < report["share"] <= 1.0

    def test_the_country_a_film_comes_from_is_one_of_them(self, built):
        """The case the rule was written for. Measured at 0.71 on the films."""
        assert built.guessable["countryOfOrigin"]["share"] > 0.5

    def test_a_guessable_property_keeps_its_grounding_rate_in_the_record(self, built):
        """A slot dropped for being guessable is a finding, not an absence."""
        for prop in built.guessable:
            assert built.properties[prop]["stated"] > 0
            assert built.properties[prop]["offered"] is False

    def test_no_value_fills_more_than_its_share_of_a_class(self, built):
        """What the committed sample is balanced on. The share is of the grid
        a cell will be, so it is measured per class and not over the whole."""
        per_class: dict[str, list] = {}
        for item in built.entities:
            per_class.setdefault(item.cls, []).append(item)
        for cls, items in per_class.items():
            counts: dict[str, int] = {}
            for item in items:
                for prop, values in item.facts.items():
                    for value in dict.fromkeys(map(str, values)):
                        counts[f"{prop}={value}"] = counts.get(f"{prop}={value}", 0) + 1
            assert max(counts.values()) <= max(2, int(len(items) * 0.04)) + 1, cls


class TestTheCellsThatClearTheMinimum:
    """A class below the minimum is carried and not hidden, so a report can say
    which cells it had and which it did not."""

    def test_at_least_one_class_clears_the_minimum(self, built):
        assert any(count >= CELL_MINIMUM for count in built.per_class.values())

    def test_a_class_below_the_minimum_is_still_carried_with_its_count(self, built):
        """Which classes those are is a finding about Wikipedia, not a defect:
        a book's lead names its author and little else, and a city's lead names
        no mapped property at all."""
        reported = built.grounding["per_class"]
        assert set(built.per_class) <= set(reported)
        assert all(reported[cls]["entities"] > 0 for cls in built.per_class)

    def test_a_class_too_small_to_be_a_grid_is_not_published(self, built):
        """Below thirty documents the controls read one document's luck: three
        software applications scored 0.095 on the wrong-document control
        because two of them were made by Google."""
        for cls in built.per_class:
            assert sum(1 for e in built.entities if e.cls == cls) >= 30, cls

    def test_a_class_that_was_drawn_and_published_nothing_says_so(self, built):
        drawn = set(built.draw["per_class"])
        assert drawn - set(built.per_class)
        for cls in drawn - set(built.per_class):
            assert built.grounding["per_class"][cls]["entities"] >= 0


class TestAgainstTheAdapter:
    """A task the adapter cannot read is a task no arm can run."""

    def test_build_request_accepts_the_records(self, tasks):
        pytest.importorskip("oold.agent.prompts")
        from oold_llm_bench.runner.adapter import build_request

        cell = Cell(
            condition=Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=4),
            model=ModelSpec(model="m", provider_profile="openai", model_version="1"),
            task=tasks[0],
            repetition=1,
        )
        request = build_request(cell)
        assert request.document == tasks[0].document
        assert request.branches
        assert request.parents
