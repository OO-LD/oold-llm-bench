"""The hand-authored notes as tasks.

This corpus is annotated and not constructed, so the failure the generated
ones cannot have is the one to test for: an annotation naming a class or a
unit the corpus does not have, and the loader quietly returning fewer tasks
than there are notes.
"""

import json

import pytest

from oold_llm_bench.corpus.catalogue import CatalogueEntry
from oold_llm_bench.corpus.eln import Note, load_notes, read_notes, resolve_unit
from oold_llm_bench.corpus.quantities import QuantityKind
from oold_llm_bench.grading import Dimension, TripleSet, make_triples
from oold_llm_bench.grading.score import score_task
from oold_llm_bench.results.record import Environment, ModelSpec
from oold_llm_bench.runner import Cell, Condition, ExperimentConfig
from oold_llm_bench.runner.adapter import _catalogue_of
from oold_llm_bench.runner.preflight import CONTROL_CEILING, SPELLING_CEILING, control_scores
from oold_llm_bench.tasks.models import Difficulty, Source, Split

KINDS = [
    QuantityKind(name="Pressure", units=("bar", "pascal", "kilo_pascal")),
    QuantityKind(name="Mass", units=("gram", "kilo_gram", "milli_gram")),
    QuantityKind(name="ThermodynamicTemperature", units=("kelvin", "Celsius", "milli_kelvin")),
    QuantityKind(
        name="CelsiusTemperature",
        units=("kelvin", "Celsius", "milli_kelvin"),
        own_units=False,
        parent="ThermodynamicTemperature",
    ),
    QuantityKind(name="ElectrolyticConductivity", units=("siemens_per_meter", "milli_siemens_per_centi_meter")),
    QuantityKind(name="VolumeFlowRate", units=("liter_per_minute", "milli_liter_per_minute")),
    QuantityKind(name="Thickness", units=("meter", "milli_meter", "nano_meter"), own_units=False, parent="Length"),
    QuantityKind(name="Time", units=("second", "minute", "hour")),
    QuantityKind(name="Volume", units=("liter", "milli_liter", "meter_cubed")),
    QuantityKind(name="Voltage", units=("volt", "milli_volt")),
    QuantityKind(name="Density", units=("gram_per_centi_meter_cubed", "kilo_gram_per_meter_cubed")),
    QuantityKind(name="Force", units=("newton", "kilo_newton")),
    QuantityKind(
        name="TemperatureRateOfChange",
        units=("kelvin_per_minute", "Celsius_per_minute"),
        own_units=False,
        parent="TemperaturePerTime",
    ),
    QuantityKind(name="Wavelength", units=("meter", "nano_meter"), own_units=False, parent="Length"),
    QuantityKind(name="Resistance", units=("ohm", "kilo_ohm")),
    QuantityKind(name="Energy", units=("joule", "kilo_joule")),
    QuantityKind(name="AngularVelocity", units=("radian_per_second", "radian_per_minute")),
    QuantityKind(name="Length", units=("meter", "kilo_meter")),
    QuantityKind(name="Luminance", units=("candela_per_meter_squared",)),
]
"""The kinds the notes need, with the units the real corpus gives them.

``AngularVelocity`` keeps its three radian units and ``Volume`` keeps no
dimensionless one, because those two absences are what the declared
exclusions rest on.
"""

ENTRIES = {
    "VolumeFlowRate": CatalogueEntry(
        identifier="VolumeFlowRate",
        label="Volume Flow Rate",
        description="The volume of fluid passing a point per unit time.",
        units=("liter_per_minute",),
    )
}

NOTE_COUNT = 20
"""What was written down. The loader is only allowed to return fewer when the
annotation declares why, and the difference is checked here rather than left
to whoever reads the results."""


def written(tmp_path, *entries):
    """A notes file with whatever this test needs in it."""
    path = tmp_path / "notes.json"
    path.write_text(
        json.dumps({"source": "hand-authored", "written": "2026-09-27", "note": "", "notes": list(entries)}),
        encoding="utf-8",
    )
    return path


def one(note_id="n1", cls="Pressure", value=1.0, unit="bar", **extra):
    return {
        "id": note_id,
        "text": f"Read {value} {unit}.",
        "expected": [{"class": cls, "value": value, "unit": unit}],
        **extra,
    }


def perfect_answer(task) -> TripleSet:
    """What a model that read the note correctly would produce."""
    triples = frozenset()
    classes = {}
    for instance in task.expected:
        triples |= make_triples(instance.key, instance.fields)
        classes[instance.key] = instance.class_path
    return TripleSet(triples=triples, classes=classes, provenance={})


@pytest.fixture(scope="module")
def tasks():
    return load_notes(KINDS, entries=ENTRIES)


class TestNothingIsLostBetweenTheFileAndTheTasks:
    """Twenty notes becoming twelve tasks is the failure being guarded."""

    def test_every_note_in_the_file_is_read(self):
        assert len(read_notes().notes) == NOTE_COUNT

    def test_the_tasks_and_the_declared_exclusions_account_for_all_of_them(self, tasks):
        excluded = read_notes().declared_skips
        assert len(tasks) + len(excluded) == NOTE_COUNT

    def test_an_exclusion_says_why_in_the_data(self):
        for note in read_notes().declared_skips:
            assert note.skip
            assert not note.scorable

    def test_a_class_the_corpus_does_not_have_stops_the_load(self, tmp_path):
        path = written(tmp_path, one(cls="Enthusiasm"))
        with pytest.raises(ValueError, match="which the corpus does not have"):
            load_notes(KINDS, path=path)

    def test_a_unit_the_kind_does_not_admit_stops_the_load(self, tmp_path):
        path = written(tmp_path, one(unit="furlong"))
        with pytest.raises(ValueError, match="not a unit of Pressure"):
            load_notes(KINDS, path=path)

    def test_every_broken_note_is_named_at_once(self, tmp_path):
        """Fixing an annotation one raised error at a time loses notes."""
        path = written(tmp_path, one("n1", cls="Enthusiasm"), one("n2", unit="furlong"))
        with pytest.raises(ValueError, match=r"n1.*n2"):
            load_notes(KINDS, path=path)

    def test_an_exclusion_marker_cannot_hide_a_working_note(self, tmp_path):
        path = written(tmp_path, one(skip="looked hard"))
        with pytest.raises(ValueError, match="declared unscorable"):
            load_notes(KINDS, path=path)

    def test_a_note_the_catalogue_does_not_offer_stops_the_load(self, tmp_path):
        path = written(tmp_path, one())
        with pytest.raises(ValueError, match="which the catalogue does not offer"):
            load_notes(KINDS, path=path, catalogue=("Mass",))


class TestResolvingWhatWasWrittenDown:
    """What a person spells is not what the grader compares against."""

    def test_a_unit_written_as_a_label_resolves_to_the_identifier(self, tasks):
        record = next(t for t in tasks if t.id == "eln-17")
        assert record.expected[0].fields["value"].unit == "Celsius"

    def test_the_degree_prefix_is_the_only_thing_dropped(self):
        kind = next(k for k in KINDS if k.name == "ThermodynamicTemperature")
        assert resolve_unit(kind, "degree_Celsius") == "Celsius"
        assert resolve_unit(kind, "degree_Fahrenheit") is None

    def test_a_unit_spelled_exactly_is_taken_as_written(self):
        kind = next(k for k in KINDS if k.name == "Pressure")
        assert resolve_unit(kind, "kilo_pascal") == "kilo_pascal"

    def test_a_class_written_as_its_label_resolves(self, tmp_path):
        path = written(tmp_path, one(cls="Volume Flow Rate", unit="liter_per_minute"))
        record = load_notes(KINDS, path=path, entries=ENTRIES)[0]
        assert record.expected[0].class_path == "VolumeFlowRate"

    def test_every_expected_class_is_a_corpus_identifier(self, tasks):
        names = {k.name for k in KINDS}
        assert {i.class_path for t in tasks for i in t.expected} <= names

    def test_every_expected_unit_comes_from_its_own_kinds_enumeration(self, tasks):
        allowed = {k.name: set(k.units) for k in KINDS}
        for task in tasks:
            for instance in task.expected:
                assert instance.fields["value"].unit in allowed[instance.class_path]

    def test_the_value_appears_in_the_note(self, tasks):
        """Ground truth the document does not contain is unanswerable."""
        for task in tasks:
            for instance in task.expected:
                magnitude = instance.fields["value"].magnitude
                assert str(magnitude) in task.document or str(int(magnitude)) in task.document


class TestProvenance:
    """A report has to tell these from synthetic prose off the record alone."""

    def test_every_record_is_marked_hand_authored(self, tasks):
        assert {t.corpus.source for t in tasks} == {Source.MANUAL}

    def test_no_record_claims_to_be_synthetic(self, tasks):
        assert Source.SYNTHETIC not in {t.corpus.source for t in tasks}

    def test_the_record_names_the_note_it_came_from(self, tasks):
        assert {t.corpus.document_id for t in tasks} == {t.id for t in tasks}

    def test_the_content_hash_covers_the_note_text(self, tasks):
        import hashlib

        for task in tasks:
            assert task.corpus.content_hash == hashlib.sha256(task.document.encode("utf-8")).hexdigest()

    def test_the_notes_field_carries_the_corpus_and_the_date(self, tasks):
        assert all("corpus=eln" in (t.notes or "") for t in tasks)
        assert all(read_notes().written in (t.notes or "") for t in tasks)

    def test_the_document_is_the_note_and_nothing_else(self, tasks):
        texts = {n.text for n in read_notes().notes}
        assert {t.document for t in tasks} <= texts

    def test_the_register_is_recorded_as_hard(self, tasks):
        """The reading sits inside a sentence that carries no answer."""
        assert {t.difficulty for t in tasks} == {Difficulty.HARD}

    def test_the_split_can_be_chosen(self):
        assert load_notes(KINDS, split=Split.TEST)[0].split is Split.TEST


class TestTheCatalogue:
    """Comparable to what a synthetic task is offered, or the two do not read
    side by side."""

    def test_every_kind_with_units_is_offered_by_default(self, tasks):
        offered = set(tasks[0].catalogue or ())
        assert offered == {k.name for k in KINDS if k.has_units}

    def test_the_units_each_offered_class_admits_are_carried(self, tasks):
        per_class = tasks[0].unit_catalogue or {}
        assert per_class["Pressure"] == sorted({"bar", "pascal", "kilo_pascal"})

    def test_the_catalogue_is_described_when_the_schemas_are_given(self, tasks):
        described = tasks[0].catalogue_text or {}
        assert "volume of fluid passing a point" in described["VolumeFlowRate"]

    def test_nothing_is_withheld_from_the_description(self, tasks):
        """No document here was generated from an annotation, so no annotation
        in the catalogue can be a copy of one."""
        assert "units: liter_per_minute" in (tasks[0].catalogue_text or {})["VolumeFlowRate"]

    def test_bare_identifiers_are_offered_when_no_schemas_are_given(self):
        assert load_notes(KINDS)[0].catalogue_text is None

    def test_a_caller_may_choose_the_catalogue(self):
        offered = tuple(k.name for k in KINDS)
        assert load_notes(KINDS, catalogue=offered)[0].catalogue == list(offered)


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
        assert len(set(self.positions(tasks))) > 1

    def test_the_answer_is_not_always_first(self, tasks):
        assert set(self.positions(tasks)) != {0}

    def test_the_class_the_note_needs_is_always_offered(self, tasks):
        for cell in self.cells(tasks, size=1):
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

    def test_an_answer_from_another_note_scores_zero(self, tasks):
        for task in tasks:
            for other in tasks:
                if other.id != task.id:
                    assert score_task(task, perfect_answer(other)).primary == 0.0

    def test_an_empty_answer_scores_zero(self, tasks):
        empty = TripleSet(triples=frozenset(), classes={}, provenance={})
        assert score_task(tasks[0], empty).primary == 0.0

    def test_every_negative_control_stays_at_the_floor(self, tasks):
        grid = ExperimentConfig(
            name="eln",
            conditions=[Condition(arm="schema-dump-catalog-flat-enforced")],
            models=[ModelSpec(model="m", provider_profile="openai", model_version="1")],
            tasks=tasks,
            runs_per_cell=1,
        )
        scores = control_scores(grid, Environment(benchmark_version="0.1.0", benchmark_sha="abc"))
        assert set(scores) == {"empty", "random", "spelling", "wrong-document"}
        # The spelling control is not expected to reach zero on text a person
        # wrote. One note reads "4.7 ohm", where the identifier is `ohm`.
        assert scores["spelling"] <= SPELLING_CEILING
        assert all(score <= CONTROL_CEILING for name, score in scores.items() if name != "spelling")


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


class TestTheNoteItself:
    def test_a_note_without_a_reason_is_meant_to_become_a_task(self):
        assert Note(id="n", text="t", readings=()).scorable
