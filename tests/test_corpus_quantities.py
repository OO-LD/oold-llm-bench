"""Round-trip task generation from the quantity corpus."""

import hashlib

import pytest

from oold_llm_bench.corpus.quantities import (
    UNWRITABLE,
    Notation,
    QuantityKind,
    Signal,
    UnitSymbols,
    candidates,
    canonical_number,
    generate_task,
    load_unit_symbols,
    readable_kind,
    render,
    unit_identifiable,
    writable,
    written_number,
)
from oold_llm_bench.grading.triples import Quantity
from oold_llm_bench.tasks import Difficulty, Split

KINDS = [
    QuantityKind(name="Length", units=("meter", "centi_meter", "kilo_meter")),
    QuantityKind(name="IonicStrength", units=("milli_mole_per_kilo_gram",)),
    QuantityKind(name="Mass", units=("gram", "kilo_gram")),
    QuantityKind(name="Inherited", units=("gram", "kilo_gram"), own_units=False, parent="Mass"),
]

CORPUS = [
    QuantityKind(
        name="ActivityConcentration",
        units=(
            "becquerel_per_meter_cubed",
            "nano_becquerel_per_liter",
            "micro_becquerel_per_liter",
            "milli_becquerel_per_liter",
            "becquerel_per_liter",
        ),
        uuid="815fd908-9afe-56a9-9cdf-0e0ca0c04543",
    ),
    QuantityKind(
        name="ForcePerAreaTime",
        units=("pascal_per_second", "pascal_per_hour", "pascal_per_minute", "hecto_pascal_per_hour"),
        uuid="5ffdbf4d-b0e9-5c7e-8ce0-298e48ee3197",
    ),
    QuantityKind(
        name="InverseMass",
        units=("per_kilo_gram", "per_gram", "per_milli_gram"),
        uuid="01d40c43-6663-572d-80de-0e503436a120",
    ),
    QuantityKind(
        name="InverseSquareEnergy",
        units=("per_joule_squared", "per_giga_electron_volt_squared", "per_electron_volt_squared"),
        uuid="2ff0e06e-1f99-5bd0-82b2-fb855c1b8963",
    ),
    QuantityKind(
        name="LinearThermalExpansion",
        units=(
            "meter_per_kelvin",
            "micro_meter_per_kelvin",
            "milli_meter_per_kelvin",
            "centi_meter_per_kelvin",
        ),
        uuid="cbd09a3b-3c53-56f6-b501-044b7a9cd13b",
    ),
    QuantityKind(
        name="LuminousIntensity",
        units=("candela", "millicandela", "candlepower", "kilo_candela"),
        uuid="cd5084e9-8d8a-541e-b5d1-21ae983fad3d",
    ),
    QuantityKind(
        name="MolarMass",
        units=("kilo_gram_per_mole", "gram_per_mole", "kilo_gram_per_kilo_mole"),
        uuid="57dc5920-ca59-5bc5-b7c1-f41b0a79bad3",
    ),
    QuantityKind(
        name="Permittivity",
        units=(
            "farad_per_meter",
            "pico_farad_per_meter",
            "nano_farad_per_meter",
            "micro_farad_per_kilo_meter",
            "micro_farad_per_meter",
            "farad_per_kilo_meter",
        ),
        uuid="a2c1b304-0a2b-51e7-a8eb-fa4524755702",
    ),
    QuantityKind(
        name="PressureCoefficient",
        units=(
            "pascal_per_kelvin",
            "hecto_pascal_per_kelvin",
            "kilo_pascal_per_kelvin",
            "mega_pascal_per_kelvin",
        ),
        uuid="a1c670ea-697d-58d8-8f1a-79b2c8aca63e",
    ),
    QuantityKind(
        name="PressureGradient",
        units=(
            "pascal_per_meter",
            "pico_pascal_per_kilo_meter",
            "milli_pascal_per_meter",
            "hecto_pascal_per_meter",
            "kilo_pascal_per_meter",
            "kilo_pascal_per_milli_meter",
        ),
        uuid="731e73ec-9945-5a1f-bc5a-6f5c4f67b422",
    ),
    QuantityKind(
        name="VolumeThermalExpansion",
        units=(
            "meter_cubed_per_kelvin",
            "centi_meter_cubed_per_kelvin",
            "milli_liter_per_kelvin",
            "liter_per_kelvin",
        ),
        uuid="fa99fdcf-e84d-5a00-8472-386406ac1416",
    ),
]
"""Eleven kinds copied out of the corpus, not invented.

Every one is unit-identifiable and every unit of every one has a published
symbol, so the same fixture serves both notations and the difference between
them is the rendering and nothing else. The committed signal data covers these
names, so the signals can be exercised on them too.
"""


class TestReadableKind:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Length", "length"),
            ("IonicStrength", "ionic strength"),
            ("ElectricChargeDensity", "electric charge density"),
            ("PHValue", "ph value"),
            ("Volume", "volume"),
        ],
    )
    def test_a_class_name_reads_as_prose(self, name, expected):
        assert readable_kind(name) == expected

    def test_it_never_produces_a_word_no_document_contains(self):
        """Lowercasing whole gives "ionicstrength", which nothing matches."""
        assert readable_kind("IonicStrength") != "ionicstrength"


class TestRender:
    def test_the_value_and_unit_reach_the_prose(self):
        import random

        text = render("Length", 1.75, "centi_meter", Difficulty.EASY, random.Random(1))  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        assert "1.75" in text
        assert "centi meter" in text

    def test_underscores_do_not_survive_into_prose(self):
        import random

        text = render("Length", 1.0, "milli_mole_per_kilo_gram", Difficulty.EASY, random.Random(1))  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        assert "_" not in text


class TestSignal:
    """Whether the document names the class it is asking for.

    Naming it makes class selection a substring match, which would flatten
    the arm comparison the whole study is built around.
    """

    def test_an_implied_document_never_contains_the_class_name(self):
        for seed in range(30):
            task = generate_task(KINDS, task_id="t", seed=seed, n_entities=1, signal=Signal.UNIT)
            flat = task.document.lower().replace(" ", "")
            for instance in task.expected:
                assert instance.class_path.lower() not in flat

    def test_a_named_document_does_contain_it(self):
        """The ceiling condition, kept deliberately and reported apart."""
        task = generate_task([KINDS[1]], task_id="t", seed=1, signal=Signal.NAMED)
        assert "ionic strength" in task.document

    def test_implied_is_the_default(self):
        assert (
            generate_task(KINDS, task_id="t", seed=1).document
            == generate_task(KINDS, task_id="t", seed=1, signal=Signal.UNIT).document
        )

    def test_implied_draws_only_from_unit_identifiable_kinds(self):
        identifiable = {k.name for k in unit_identifiable(KINDS)}
        used = {
            instance.class_path
            for seed in range(30)
            for instance in generate_task(KINDS, task_id="t", seed=seed, signal=Signal.UNIT).expected
        }
        assert used <= identifiable

    def test_a_shared_unit_disqualifies_a_kind_from_implied_tasks(self):
        """Two kinds sharing a unit cannot be told apart from it."""
        shared = [
            QuantityKind(name="AlphaThing", units=("per_meter",)),
            QuantityKind(name="BetaThing", units=("per_meter",)),
        ]
        assert unit_identifiable(shared) == []
        with pytest.raises(ValueError, match="no quantity kind is usable"):
            generate_task(shared, task_id="t", seed=1, signal=Signal.UNIT)

    def test_a_shared_unit_is_still_fine_for_a_named_task(self):
        shared = [QuantityKind(name="AlphaThing", units=("per_meter",))]
        task = generate_task(shared, task_id="t", seed=1, signal=Signal.NAMED)
        assert "alpha thing" in task.document

    def test_the_condition_is_recorded_on_the_task(self):
        task = generate_task(KINDS, task_id="t", seed=1, signal=Signal.NAMED)
        assert task.notes == "signal=named,vocabulary=consensus"


class TestGenerateTask:
    def test_a_task_carries_a_quantity_as_its_expected_value(self):
        """A bare number cannot be scored for unit correctness."""
        task = generate_task(KINDS, task_id="t1", seed=1)
        value = task.expected[0].fields["value"]
        assert isinstance(value, Quantity)
        assert value.unit

    def test_the_expected_unit_comes_from_the_schema_enumeration(self):
        task = generate_task(KINDS, task_id="t1", seed=3, n_entities=3, signal=Signal.NAMED)
        allowed = {k.name: set(k.units) for k in KINDS}
        for instance in task.expected:
            value = instance.fields["value"]
            assert value.unit in allowed[instance.class_path]

    def test_the_value_appears_in_the_document(self):
        """Ground truth the document does not contain is unanswerable."""
        task = generate_task(KINDS, task_id="t1", seed=5, n_entities=2)
        for instance in task.expected:
            assert str(instance.fields["value"].magnitude) in task.document

    def test_a_seed_reproduces_the_document_exactly(self):
        first = generate_task(KINDS, task_id="t1", seed=9)
        second = generate_task(KINDS, task_id="t1", seed=9)
        assert first.document == second.document
        assert first.corpus.content_hash == second.corpus.content_hash

    def test_a_different_seed_gives_a_different_document(self):
        first = generate_task(KINDS, task_id="t1", seed=9)
        second = generate_task(KINDS, task_id="t1", seed=10)
        assert first.document != second.document

    def test_the_content_hash_covers_the_document(self):
        task = generate_task(KINDS, task_id="t1", seed=2)
        import hashlib

        assert task.corpus.content_hash == hashlib.sha256(task.document.encode("utf-8")).hexdigest()

    @pytest.mark.parametrize("difficulty", [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD])
    def test_every_difficulty_still_contains_its_answer(self, difficulty):
        task = generate_task(KINDS, task_id="t1", seed=4, difficulty=difficulty, n_entities=2)
        assert task.difficulty is difficulty
        for instance in task.expected:
            assert str(instance.fields["value"].magnitude) in task.document

    def test_harder_prose_is_longer_than_plain_prose(self):
        plain = generate_task(KINDS, task_id="t1", seed=6, difficulty=Difficulty.EASY)
        embedded = generate_task(KINDS, task_id="t1", seed=6, difficulty=Difficulty.HARD)
        assert len(embedded.document) > len(plain.document)

    def test_a_kind_that_inherits_its_units_is_never_a_unit_task(self):
        """Its parent and every sibling carry the same enumeration."""
        used = {
            instance.class_path
            for seed in range(20)
            for instance in generate_task(KINDS, task_id="t", seed=seed, signal=Signal.UNIT).expected
        }
        assert "Inherited" not in used

    def test_a_kind_that_inherits_its_units_can_still_be_a_named_task(self):
        """505 of 943 schemas inherit, and dropping them loses Diameter."""
        used = {
            instance.class_path
            for seed in range(30)
            for instance in generate_task(KINDS, task_id="t", seed=seed, signal=Signal.NAMED).expected
        }
        assert "Inherited" in used

    def test_asking_for_more_entities_than_exist_is_refused(self):
        with pytest.raises(ValueError, match="corpus offers"):
            generate_task(KINDS, task_id="t1", seed=1, n_entities=99)

    def test_a_corpus_that_constrains_nothing_is_refused(self):
        with pytest.raises(ValueError, match="no quantity kind is usable"):
            generate_task([QuantityKind(name="X", units=())], task_id="t1", seed=1)

    def test_the_split_label_is_carried(self):
        task = generate_task(KINDS, task_id="t1", seed=1, split=Split.TEST)
        assert task.split is Split.TEST

    def test_entity_keys_are_unique(self):
        task = generate_task(KINDS, task_id="t1", seed=8, n_entities=3, signal=Signal.NAMED)
        keys = [instance.key for instance in task.expected]
        assert len(set(keys)) == len(keys)


class TestSubclassesAreReachable:
    """505 of 943 schemas are subclasses that inherit their units.

    No unit can identify one, because the parent and every sibling carry the
    same enumeration, so a translated label or a description is the only way
    to point at one without naming it.
    """

    def test_a_subclass_inherits_its_parents_units(self):
        inherited = next(k for k in KINDS if k.name == "Inherited")
        parent = next(k for k in KINDS if k.name == "Mass")
        assert inherited.units == parent.units
        assert inherited.own_units is False
        assert inherited.parent == "Mass"

    def test_a_subclass_is_never_unit_identifiable(self):
        assert "Inherited" not in {k.name for k in unit_identifiable(KINDS)}

    def test_sharing_a_unit_also_disqualifies_the_parent(self):
        """If a subclass carries 'gram' too, 'gram' no longer names Mass."""
        assert "Mass" not in {k.name for k in unit_identifiable(KINDS)}


class TestMultipleEntities:
    """Several measurements are a list of quantity values, not a narrative.

    Nothing is attached to a subject yet. That makes the list
    unambiguous: with no sample or product to attribute a reading to, each
    reading is identified by its own value and unit, so order carries no
    information and cannot be got wrong. Attaching a measurement to a subject
    comes later, and the schema.org corpus already has that shape.
    """

    def test_several_entities_render_as_a_list(self):
        task = generate_task(KINDS, task_id="t", seed=5, n_entities=3, signal=Signal.NAMED)
        assert task.document.startswith("Measurements:")
        assert task.document.count("\n- ") == 3

    def test_one_entity_stays_a_sentence(self):
        task = generate_task(KINDS, task_id="t", seed=5, signal=Signal.NAMED)
        assert "Measurements:" not in task.document
        assert task.document.endswith(".")

    def test_every_reading_keeps_its_own_value_and_unit(self):
        task = generate_task(KINDS, task_id="t", seed=5, n_entities=3, signal=Signal.NAMED)
        for instance in task.expected:
            value = instance.fields["value"]
            assert str(value.magnitude) in task.document
            assert value.unit.replace("_", " ") in task.document

    def test_the_readings_are_distinguishable_from_each_other(self):
        """Two readings with the same value and unit could not be told apart."""
        task = generate_task(KINDS, task_id="t", seed=5, n_entities=3, signal=Signal.NAMED)
        pairs = {(i.fields["value"].magnitude, i.fields["value"].unit) for i in task.expected}
        assert len(pairs) == len(task.expected)


class TestDroppingAmbiguousKinds:
    """Two kinds described by one sentence cannot both be the answer."""

    def data(self):
        from oold_llm_bench.corpus.signals import SignalData

        return SignalData(
            labels={},
            descriptions={"Alpha": "same text", "Beta": "same text", "Gamma": "its own"},
            symbols={},
            synonyms={},
            elucidations={},
            emmo_iris={},
            qudt_version="test",
            emmo_version="test",
            source="test",
        )

    def kinds(self):
        return [
            QuantityKind(name="Alpha", units=("meter",)),
            QuantityKind(name="Beta", units=("gram",)),
            QuantityKind(name="Gamma", units=("second",)),
        ]

    def test_a_shared_description_removes_every_kind_that_shares_it(self):
        """No arm can be right on them more than half the time, so the loss is
        the corpus and not the model."""
        pool = candidates(self.kinds(), Signal.DESCRIPTION, self.data())
        assert [k.name for k in pool] == ["Gamma"]

    def test_a_kind_with_its_own_text_survives(self):
        pool = candidates(self.kinds(), Signal.DESCRIPTION, self.data())
        assert "Gamma" in [k.name for k in pool]

    def test_case_and_spacing_do_not_hide_a_duplicate(self):
        from oold_llm_bench.corpus.signals import SignalData

        data = SignalData(
            labels={},
            descriptions={"Alpha": "Same Text ", "Beta": "same text"},
            symbols={},
            synonyms={},
            elucidations={},
            emmo_iris={},
            qudt_version="t",
            emmo_version="t",
            source="t",
        )
        pool = candidates(self.kinds()[:2], Signal.DESCRIPTION, data)
        assert pool == []


def grid(**overrides):
    """Every task the regression guard and the notation comparison walk.

    720 records: eleven kinds, forty seeds, three difficulties, one to three
    readings, both signals the fixture supports throughout. Wide enough that
    drift in any frame, any pool or any field of the record shows up.
    """
    for seed in range(40):
        for difficulty in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
            for n_entities in (1, 2, 3):
                for signal in (Signal.UNIT, Signal.NAMED):
                    yield generate_task(
                        CORPUS,
                        task_id=f"t{seed}",
                        seed=seed,
                        difficulty=difficulty,
                        n_entities=n_entities,
                        signal=signal,
                        **overrides,
                    )


class TestTheDefaultNotationHasNotMoved:
    """The guard every measured result rests on.

    A corpus that changes underneath a run makes the runs incomparable, so the
    old rendering is pinned over the whole record and not only the document.

    Moved deliberately on 2026-10-04, and this is the record of why. The
    rendered catalogue now carries every unit the grammar will accept rather
    than eight of them and a count, and it carries them whether or not the
    descriptions are shown. A presented schema has to contain what it
    enforces; before this the bare catalogue listed a hundred class
    identifiers in full and no unit identifier at all, which is why class
    accuracy read 0.98 and unit accuracy read 0.02 on the same cell.

    Every quantity number measured before this date was taken under the old
    rendering and is not comparable with one taken after it.
    """

    def test_the_whole_grid_of_records_hashes_as_it_did(self):
        digest = hashlib.sha256()
        for task in grid():
            digest.update(task.model_dump_json().encode("utf-8"))
        assert digest.hexdigest() == "1feb33efaa95d3de74ad907e4b6194234a9026348e70b3c6316dec411b2bfcca"

    def test_canonical_is_what_a_caller_gets_without_asking(self):
        assert generate_task(CORPUS, task_id="t", seed=3) == generate_task(
            CORPUS, task_id="t", seed=3, notation=Notation.CANONICAL
        )

    def test_the_unit_still_reaches_the_page_de_underscored(self):
        """The fault the written notation exists to fix, stated as a test."""
        task = generate_task(KINDS, task_id="t", seed=9, signal=Signal.NAMED)
        assert task.expected[0].fields["value"].unit.replace("_", " ") in task.document


class TestWrittenNotation:
    def test_a_fixed_seed_writes_the_symbol_where_it_wrote_the_identifier(self):
        canonical = generate_task(CORPUS, task_id="t", seed=4, signal=Signal.NAMED)
        written = generate_task(CORPUS, task_id="t", seed=4, signal=Signal.NAMED, notation=Notation.WRITTEN)
        assert canonical.document == "A inverse square energy of 11.2 per giga electron volt squared was recorded."
        assert written.document == "A inverse square energy of 11.2 /GeV² was recorded."
        assert canonical.expected == written.expected
        assert written.expected[0].fields["value"].unit == "per_giga_electron_volt_squared"

    def test_the_document_writes_the_published_symbol(self):
        task = generate_task(CORPUS, task_id="t", seed=4, signal=Signal.NAMED, notation=Notation.WRITTEN)
        unit = task.expected[0].fields["value"].unit
        symbol = load_unit_symbols().written(task.expected[0].class_path, unit)
        assert symbol is not None
        assert symbol in task.document
        assert unit.replace("_", " ") not in task.document

    def test_no_underscore_survives_into_prose(self):
        for task in grid(notation=Notation.WRITTEN):
            assert "_" not in task.document

    def test_the_notation_is_recorded_on_the_task(self):
        task = generate_task(CORPUS, task_id="t", seed=1, notation=Notation.WRITTEN)
        assert task.notes == "signal=unit,vocabulary=consensus,notation=written"

    def test_every_document_in_the_grid_moves(self):
        pairs = zip(grid(), grid(notation=Notation.WRITTEN), strict=True)
        moved = sum(1 for old, new in pairs if old.document != new.document)
        assert moved == 720

    def test_a_kind_with_no_writable_unit_is_refused(self):
        unwritable = [QuantityKind(name="Countable", units=("unitless",))]
        assert writable(unwritable) == []
        with pytest.raises(ValueError, match="no quantity kind is usable"):
            generate_task(unwritable, task_id="t", seed=1, notation=Notation.WRITTEN)

    def test_a_unit_with_no_symbol_is_never_drawn(self):
        """The kind survives on the units it can write rather than being lost."""
        mixed = [QuantityKind(name="Mixed", units=("unitless", "kilo_meter"))]
        assert [k.name for k in writable(mixed)] == ["Mixed"]
        drawn = {
            generate_task(mixed, task_id="t", seed=seed, notation=Notation.WRITTEN).expected[0].fields["value"].unit
            for seed in range(20)
        }
        assert drawn == {"kilo_meter"}

    def test_rendering_a_unit_no_document_can_write_is_refused(self):
        import random

        with pytest.raises(ValueError, match="no document could write"):
            render("Countable", 1.0, "unitless", Difficulty.EASY, random.Random(1), notation=Notation.WRITTEN)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one


class TestWrittenNumbers:
    @pytest.mark.parametrize(
        ("value", "written"),
        [(76037.1179, "76,037.1179"), (48.7, "48.7"), (999.999, "999.999"), (1000.0, "1,000.0")],
    )
    def test_digits_are_grouped_in_threes(self, value, written):
        assert written_number(value) == written

    def test_nothing_is_rounded_away(self):
        assert canonical_number(written_number(76037.1179)) == 76037.1179


class TestRecoveringTheAnswerFromTheText:
    """Every written document has to be readable back to its own answer.

    Not spot-checked: the whole grid, both fields, by the documented rule and
    nothing else. A surface form the answer cannot be reached from is a task
    with no correct answer, and on this corpus that is the primary metric.
    """

    def test_every_reading_reads_back_to_what_it_was(self):
        symbols = load_unit_symbols()
        checked = 0
        for task in grid(notation=Notation.WRITTEN):
            for instance in task.expected:
                value = instance.fields["value"]
                written = written_number(value.magnitude)
                symbol = symbols.written(instance.class_path, value.unit)
                assert symbol is not None
                assert written in task.document
                assert symbol in task.document
                assert canonical_number(written) == value.magnitude
                assert symbols.canonical(symbol) == value.unit
                checked += 1
        assert checked == 1440


class TestUnitSymbols:
    def test_a_symbol_names_one_unit_and_no_other(self):
        symbols = load_unit_symbols()
        assert symbols.canonical("km") == "kilo_meter"
        assert symbols.canonical("J/K") == "joule_per_kelvin"
        assert symbols.canonical("F/m") == "farad_per_meter"

    def test_a_table_two_units_share_a_symbol_in_is_refused(self, tmp_path):
        import json

        path = tmp_path / "clash.json"
        path.write_text(
            json.dumps({"symbols": {"alpha": "x", "beta": "x"}, "by_kind": {}, "source": "test"}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="claimed by both"):
            load_unit_symbols(path)

    def test_a_name_meaning_two_units_is_resolved_by_its_kind(self):
        """``meter`` is ``m`` under Length and ``m²/m`` under AreaPerLength."""
        symbols = load_unit_symbols()
        assert symbols.written("Length", "meter") == "m"
        assert symbols.written("AreaPerLength", "meter") == "m²/m"

    @pytest.mark.parametrize("symbol", ["cal{IT}", "1000", "一", "#"])
    def test_a_published_symbol_no_document_writes_is_refused(self, symbol):
        assert any(rejects(symbol) for _, rejects in UNWRITABLE)

    def test_a_supplied_table_replaces_the_committed_one(self):
        supplied = UnitSymbols(symbols={"kilo_meter": "KM"}, by_kind={}, names={"KM": "kilo_meter"}, source="test")
        task = generate_task(
            [QuantityKind(name="Length", units=("kilo_meter",))],
            task_id="t",
            seed=1,
            notation=Notation.WRITTEN,
            symbols=supplied,
        )
        assert "KM" in task.document
