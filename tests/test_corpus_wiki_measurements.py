"""Wiki-Measurements as tasks.

Real text with annotated spans, so the failures to test for are the ones a
generated corpus cannot have: an example naming a class or a unit the quantity
corpus does not have, a ground truth its own document does not contain, and
26,009 examples quietly becoming a number nobody can place.
"""

import hashlib
import json

import pytest

from oold_llm_bench.corpus.catalogue import CatalogueEntry
from oold_llm_bench.corpus.quantities import QuantityKind
from oold_llm_bench.corpus.wiki_measurements import (
    DATASET_DOI,
    MISLABELLED_UNITS,
    PROPERTY_KIND,
    TEXT_LICENCE,
    load_examples,
    read_corpus,
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

_LENGTH_UNITS = (
    "meter",
    "fermi",
    "pico_meter",
    "nano_meter",
    "micro_meter",
    "milli_meter",
    "centi_meter",
    "deci_meter",
    "deca_meter",
    "hecto_meter",
    "kilo_meter",
    "astronomical_unit",
)
"""What ``Length`` admits, written once because seven kinds here inherit it.

That inheritance is the reason class selection on this corpus is not decidable
from the unit: a sentence reading "410 m" is an ``Altitude`` or a ``Height``
or a ``Depth`` and only the words around it say which.
"""


def _of_length(name: str) -> QuantityKind:
    return QuantityKind(name=name, units=_LENGTH_UNITS, own_units=False, parent="Length")


KINDS = [
    QuantityKind(name="Length", units=_LENGTH_UNITS),
    _of_length("Altitude"),
    _of_length("Depth"),
    _of_length("Diameter"),
    _of_length("Height"),
    _of_length("Radius"),
    _of_length("Width"),
    QuantityKind(
        name="Area",
        units=(
            "meter_squared",
            "nano_meter_squared",
            "micro_meter_squared",
            "milli_meter_squared",
            "centi_meter_squared",
            "deci_meter_squared",
            "year",
            "deca_year",
            "hectare",
            "kilo_meter_squared",
        ),
    ),
    QuantityKind(name="Mass", units=("kilo_gram", "dalton", "gram", "pound", "carat", "metric_ton", "mega_gram")),
    QuantityKind(name="Power", units=("watt", "milli_watt", "kilo_watt", "mega_watt", "giga_watt")),
    QuantityKind(
        name="Speed",
        units=("meter_per_second", "centi_meter_per_second", "kilo_meter_per_second", "hertz_meter"),
    ),
    QuantityKind(name="Time", units=("second", "minute", "hour", "day", "week", "month", "year", "kilo_year")),
]
"""The kinds the corpus uses, with the enumerations the real schemas give them.

``Length`` and ``Area`` carry theirs whole, because those are the two largest
cells and ``Area`` carries the defect the corpus has to survive. The other
four carry the members this corpus draws and a few more: ``Power`` enumerates
36 names of which two appear here, and writing out the other 34 would test
nothing this file does not already test.
"""

ENTRIES = {
    "Area": CatalogueEntry(
        identifier="Area",
        label="Area",
        description="The two-dimensional size of a defined part of a surface.",
        units=("meter_squared", "kilo_meter_squared"),
    )
}

CELL_MINIMUM = 120
"""Tasks a cell needs before its number is worth reporting."""


def perfect_answer(task) -> TripleSet:
    """What a model that read the sentence correctly would produce."""
    triples = frozenset()
    classes = {}
    for instance in task.expected:
        triples |= make_triples(instance.key, instance.fields)
        classes[instance.key] = instance.class_path
    return TripleSet(triples=triples, classes=classes, provenance={})


def _committed() -> dict:
    from oold_llm_bench.corpus.wiki_measurements import CORPUS_PATH

    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def written(tmp_path, **changes):
    """The committed corpus with one field changed, so a guard can be tried."""
    payload = _committed()
    payload.update(changes)
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def one(tmp_path, **changes):
    """A corpus file holding a single example, with whatever this test needs."""
    example = {
        "id": "wm-1",
        "text": "The lake covers 12 km2 in area.",
        "kind": "Area",
        "unit": "kilo_meter_squared",
        "magnitude": 12.0,
        "value_text": "12",
        "unit_text": "km2",
        "article": "https://en.wikipedia.org/wiki/Example",
        "qudt_unit": "KiloM2",
        "wikidata": {"entity": "", "property": "http://www.wikidata.org/entity/P2046", "unit": ""},
    }
    example.update(changes)
    payload = {
        "schema_version": "1",
        "name": "Wiki-Measurements",
        "dataset": {"file": "x", "retrieved_at": "2026-09-28"},
        "licence": {"text": {"name": TEXT_LICENCE}},
        "built_at": "2026-09-28",
        "examples_in": 1,
        "resolved": 1,
        "excluded": {},
        "per_kind": {example["kind"]: 1},
        "cap": None,
        "examples": [example],
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def corpus():
    return read_corpus()


@pytest.fixture(scope="module")
def tasks():
    return load_examples(KINDS, entries=ENTRIES)


class TestNothingIsLostBetweenTheArchiveAndTheTasks:
    """26,009 examples becoming 1,254 tasks is only acceptable if every one of
    the missing carries a stated reason."""

    def test_every_example_is_either_resolved_or_excluded_for_a_stated_reason(self, corpus):
        assert corpus.resolved + sum(corpus.excluded.values()) == corpus.examples_in

    def test_every_declared_reason_removed_something(self, corpus):
        assert corpus.excluded
        assert all(count > 0 for count in corpus.excluded.values())
        assert all(reason.strip() for reason in corpus.excluded)

    def test_the_per_kind_counts_account_for_everything_that_resolved(self, corpus):
        assert sum(corpus.per_kind.values()) == corpus.resolved

    def test_the_file_never_holds_more_than_resolved(self, corpus):
        assert len(corpus.examples) <= corpus.resolved

    def test_the_cap_is_declared_and_is_what_the_file_obeys(self, corpus):
        assert corpus.cap is not None
        for kind, total in corpus.per_kind.items():
            held = sum(1 for e in corpus.examples if e.kind == kind)
            assert held == min(total, corpus.cap)

    def test_an_example_that_left_without_a_reason_is_refused(self, tmp_path):
        path = written(tmp_path, resolved=4000)
        with pytest.raises(ValueError, match="without a declared reason"):
            read_corpus(path)

    def test_the_cap_reports_what_it_held_back(self, corpus):
        assert corpus.dropped_by_cap == corpus.resolved - len(corpus.examples)

    def test_a_file_holding_more_than_it_resolved_is_refused(self, tmp_path):
        payload = _committed()
        excluded = dict(payload["excluded"])
        excluded["the value span is not a plain number"] += payload["resolved"] - 1
        path = written(tmp_path, resolved=1, per_kind={"Area": 1}, excluded=excluded)
        with pytest.raises(ValueError, match="but only 1 resolved"):
            read_corpus(path)

    def test_per_kind_counts_that_do_not_add_up_are_refused(self, tmp_path):
        path = written(tmp_path, per_kind={**_committed()["per_kind"], "Area": 1})
        with pytest.raises(ValueError, match="per-kind counts sum to"):
            read_corpus(path)

    def test_a_class_the_corpus_does_not_have_stops_the_load(self, tmp_path):
        path = one(tmp_path, kind="Enthusiasm")
        with pytest.raises(ValueError, match="which the corpus does not have"):
            load_examples(KINDS, path=path)

    def test_a_unit_the_kind_does_not_admit_stops_the_load(self, tmp_path):
        path = one(tmp_path, unit="furlong")
        with pytest.raises(ValueError, match="not a unit of Area"):
            load_examples(KINDS, path=path)

    def test_a_ground_truth_the_document_does_not_contain_stops_the_load(self, tmp_path):
        path = one(tmp_path, value_text="99")
        with pytest.raises(ValueError, match="which its document does not contain"):
            load_examples(KINDS, path=path)

    def test_a_kind_the_catalogue_does_not_offer_stops_the_load(self, tmp_path):
        path = one(tmp_path)
        with pytest.raises(ValueError, match="which the catalogue does not offer"):
            load_examples(KINDS, path=path, catalogue=("Length",))


class TestTheMislabelledUnitNames:
    """``Area`` calls the are ``year``. It is upstream, so it is refused rather
    than renamed, and the refusal is checked rather than assumed."""

    def test_the_declaration_names_the_two_area_units(self):
        assert MISLABELLED_UNITS == {"Area": ("year", "deca_year")}

    def test_the_names_are_still_in_the_enumeration_this_test_runs_against(self):
        area = next(k for k in KINDS if k.name == "Area")
        assert set(MISLABELLED_UNITS["Area"]) <= set(area.units)

    def test_no_task_answers_one_of_them(self, tasks):
        for task in tasks:
            for instance in task.expected:
                assert instance.fields["value"].unit not in MISLABELLED_UNITS.get(instance.class_path, ())

    def test_one_of_them_as_an_answer_stops_the_load(self, tmp_path):
        path = one(tmp_path, unit="year")
        with pytest.raises(ValueError, match="contradicts the QUDT unit"):
            load_examples(KINDS, path=path)

    def test_the_catalogue_still_offers_them_because_the_schema_still_has_them(self, tasks):
        assert set(MISLABELLED_UNITS["Area"]) <= set((tasks[0].unit_catalogue or {})["Area"])


class TestTheGroundTruthComesFromTheSpans:
    """The Wikidata value disagrees with the sentence more often than it
    agrees, so nothing here may come from the fact."""

    def test_every_expected_class_is_a_corpus_identifier(self, tasks):
        names = {k.name for k in KINDS}
        assert {i.class_path for t in tasks for i in t.expected} <= names

    def test_every_expected_unit_comes_from_its_own_kinds_enumeration(self, tasks):
        allowed = {k.name: set(k.units) for k in KINDS}
        for task in tasks:
            for instance in task.expected:
                assert instance.fields["value"].unit in allowed[instance.class_path]

    def test_the_value_appears_in_the_document_as_the_sentence_writes_it(self, corpus):
        for example in corpus.examples:
            assert example.value_text in example.text

    def test_the_magnitude_is_that_value_with_its_grouping_removed(self, corpus):
        for example in corpus.examples:
            assert float(example.value_text.replace(",", "")) == example.magnitude

    def test_the_unit_reaches_the_page_written_and_not_canonical(self, corpus):
        """The task is a surface form resolved to an identifier, so the two
        agreeing everywhere would mean the corpus measures nothing."""
        written_as_canonical = sum(1 for e in corpus.examples if e.unit_text == e.unit)
        assert written_as_canonical < len(corpus.examples) // 2

    def test_the_document_carries_no_annotation_mark(self, corpus):
        marks = {"\U0001f336", "\U0001f34a", "\U0001f34f", "\U0001f350", "\U0001f353", "\ufe0f"}
        for example in corpus.examples:
            assert not marks & set(example.text)

    def test_every_kind_comes_from_a_declared_wikidata_property(self, corpus):
        assert {e.kind for e in corpus.examples} <= set(PROPERTY_KIND.values())
        for example in corpus.examples:
            pid = example.wikidata["property"].rsplit("/", 1)[-1]
            assert PROPERTY_KIND[pid] == example.kind


class TestProvenance:
    """A report has to tell these from synthetic prose off the record alone,
    and a reader has to be able to find the article they came from."""

    def test_every_record_is_marked_as_coming_from_a_bulk_dataset(self, tasks):
        assert {t.corpus.source for t in tasks} == {Source.BULK}

    def test_no_record_claims_to_be_synthetic(self, tasks):
        assert Source.SYNTHETIC not in {t.corpus.source for t in tasks}

    def test_every_record_names_the_article_it_came_from(self, tasks):
        """The attribution CC BY-SA asks for. The article's history is the
        author list, so the url is what has to survive a trim to 120 tasks."""
        for task in tasks:
            assert (task.corpus.url or "").startswith("https://")
            assert ".wikipedia.org/wiki/" in (task.corpus.url or "")

    def test_every_record_carries_the_share_alike_licence(self, tasks):
        assert {t.corpus.licence for t in tasks} == {TEXT_LICENCE}

    def test_every_record_says_when_the_archive_was_taken(self, tasks):
        assert all(t.corpus.retrieved_at is not None for t in tasks)

    def test_the_content_hash_covers_the_document(self, tasks):
        for task in tasks:
            assert task.corpus.content_hash == hashlib.sha256(task.document.encode("utf-8")).hexdigest()

    def test_the_notes_carry_the_corpus_the_doi_and_the_licence(self, tasks):
        for task in tasks:
            assert "corpus=wiki-measurements" in (task.notes or "")
            assert DATASET_DOI in (task.notes or "")
            assert TEXT_LICENCE in (task.notes or "")

    def test_the_corpus_records_both_licences_and_what_share_alike_reaches(self, corpus):
        assert corpus.licence["text"]["name"] == TEXT_LICENCE
        assert corpus.licence["facts"]["name"] == "CC0 1.0"
        assert "CC BY-SA 4.0" in corpus.licence["note"]

    def test_the_corpus_pins_the_archive_it_was_built_from(self, corpus):
        assert corpus.dataset["doi"] == DATASET_DOI
        assert len(corpus.dataset["archive_sha256"]) == 64

    def test_the_register_is_recorded_as_hard(self, tasks):
        assert {t.difficulty for t in tasks} == {Difficulty.HARD}

    def test_the_split_can_be_chosen(self):
        assert load_examples(KINDS, split=Split.TEST, limit=1)[0].split is Split.TEST


class TestTheCellsThatClearTheMinimum:
    """A kind below the minimum is carried and not hidden, so a report can say
    which cells it had and which it did not."""

    def test_four_kinds_clear_the_minimum(self, corpus):
        clearing = {kind for kind, n in corpus.per_kind.items() if n >= CELL_MINIMUM}
        assert clearing == {"Area", "Length", "Altitude", "Diameter"}

    def test_the_file_holds_enough_of_each_to_fill_one(self, corpus):
        for kind in ("Area", "Length", "Altitude", "Diameter"):
            assert sum(1 for e in corpus.examples if e.kind == kind) >= CELL_MINIMUM

    def test_a_kind_below_the_minimum_is_still_carried(self, corpus):
        assert any(0 < n < CELL_MINIMUM for n in corpus.per_kind.values())

    def test_a_caller_may_take_a_fixed_number_per_kind(self):
        taken = load_examples(KINDS, limit=5)
        counts = {}
        for task in taken:
            counts[task.expected[0].class_path] = counts.get(task.expected[0].class_path, 0) + 1
        assert max(counts.values()) == 5


class TestTheCatalogue:
    """Comparable to what a synthetic task is offered, or the two do not read
    side by side."""

    def test_every_kind_with_units_is_offered_by_default(self, tasks):
        assert set(tasks[0].catalogue or ()) == {k.name for k in KINDS if k.has_units}

    def test_the_units_each_offered_class_admits_are_carried(self, tasks):
        per_class = tasks[0].unit_catalogue or {}
        assert per_class["Length"] == sorted(set(_LENGTH_UNITS))

    def test_the_catalogue_is_described_when_the_schemas_are_given(self, tasks):
        assert "two-dimensional size" in (tasks[0].catalogue_text or {})["Area"]

    def test_bare_identifiers_are_offered_when_no_schemas_are_given(self):
        assert load_examples(KINDS, limit=1)[0].catalogue_text is None

    def test_a_caller_may_choose_the_catalogue(self):
        offered = tuple(k.name for k in KINDS)
        assert load_examples(KINDS, catalogue=offered, limit=1)[0].catalogue == list(offered)


class TestTheCatalogueOrder:
    """The ordering must not carry the answer. It did once, for a whole run."""

    def cells(self, tasks, size=8):
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

    def positions(self, tasks, size=8):
        found = []
        for cell in self.cells(tasks, size):
            offered = _catalogue_of(cell) or ()
            found.append(offered.index(cell.task.expected[0].class_path))
        return found

    def test_the_answer_is_not_at_one_fixed_index(self, tasks):
        assert len(set(self.positions(tasks[:60]))) > 1

    def test_the_answer_is_not_always_first(self, tasks):
        assert set(self.positions(tasks[:60])) != {0}

    def test_the_class_the_document_needs_is_always_offered(self, tasks):
        for cell in self.cells(tasks[:60], size=1):
            assert cell.task.expected[0].class_path in (_catalogue_of(cell) or ())

    def test_the_order_is_stable_for_one_task(self, tasks):
        cell = self.cells(tasks)[0]
        assert _catalogue_of(cell) == _catalogue_of(cell)


class TestTheSeamCloses:
    """Corpus to task to score, with no model involved."""

    def test_a_perfect_answer_scores_one(self, tasks):
        for task in tasks:
            result = score_task(task, perfect_answer(task))
            assert result.primary == pytest.approx(1.0)
            assert result.dimensions[Dimension.CLASS].f1 == pytest.approx(1.0)
            assert result.dimensions[Dimension.UNIT].f1 == pytest.approx(1.0)

    def test_an_empty_answer_scores_zero(self, tasks):
        empty = TripleSet(triples=frozenset(), classes={}, provenance={})
        assert score_task(tasks[0], empty).primary == 0.0

    def test_every_negative_control_stays_at_its_floor(self, tasks):
        """All four measure 0.00 over the whole corpus, spelling included.

        The spelling control is allowed up to 0.25 on text a person wrote,
        because a sentence writing `ohm` spells its own identifier and no
        rendering choice of ours caused that. The hand-authored ELN notes
        score 0.06 for exactly that reason. This corpus scores 0.00, and the
        reason is worth stating: the quantity schemas enumerate SI names, and
        every SI unit in these sentences reaches the page as a symbol or as a
        plural, `km2` and `metres` and `MW`, never as `kilo_meter_squared`.
        The value dimension here is therefore a resolution and not a
        de-underscoring, which is the whole claim the control exists to test.
        """
        grid = ExperimentConfig(
            name="wiki-measurements",
            conditions=[Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=25)],
            models=[ModelSpec(model="m", provider_profile="openai", model_version="1")],
            tasks=tasks,
            runs_per_cell=1,
        )
        scores = control_scores(grid, Environment(benchmark_version="0.1.0", benchmark_sha="abc"))
        assert scores == {"empty": 0.0, "random": 0.0, "spelling": 0.0, "wrong-document": 0.0}
        assert scores["spelling"] <= SPELLING_CEILING
        assert all(score <= CONTROL_CEILING for name, score in scores.items() if name != "spelling")

    def test_the_preflight_clears_on_the_dev_split(self, tasks):
        grid = ExperimentConfig(
            name="wiki-measurements",
            conditions=[Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=25)],
            models=[ModelSpec(model="m", provider_profile="openai", model_version="1")],
            tasks=tasks[:200],
            runs_per_cell=1,
        )
        check = preflight(grid, Environment(benchmark_version="0.1.0", benchmark_sha="abc"))
        assert check.clear, check.blocked


class TestAgainstTheAdapter:
    """A task the adapter cannot read is a task no arm can run."""

    def test_build_request_accepts_the_records(self, tasks):
        pytest.importorskip("oold.agent.prompts")
        from oold_llm_bench.runner.adapter import build_request

        cell = Cell(
            condition=Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=8),
            model=ModelSpec(model="m", provider_profile="openai", model_version="1"),
            task=tasks[0],
            repetition=1,
        )
        request = build_request(cell)
        assert request.document == tasks[0].document
        assert request.branches

    def test_score_task_reads_what_build_request_was_given(self, tasks):
        task = tasks[0]
        assert score_task(task, perfect_answer(task)).primary == pytest.approx(1.0)
