"""Grading.

Several of these are regression tests for ways the predecessor benchmark could
score a wrong answer as a pass. Those are marked in their docstrings.
"""

import pytest

from oold_llm_bench.grading import Dimension, Quantity, TripleSet, make_triples
from oold_llm_bench.grading.align import align, duplicates, overlap
from oold_llm_bench.grading.compare import (
    MatchMode,
    UnitMatch,
    normalise_unit,
    same_value,
    unit_forms,
)
from oold_llm_bench.grading.score import Score, score_task
from oold_llm_bench.tasks import (
    CorpusRef,
    ExpectedInstance,
    Source,
    Split,
    TaskRecord,
)


def corpus() -> CorpusRef:
    return CorpusRef(source=Source.SYNTHETIC, document_id="d1", content_hash="0" * 64)


def task(*expected: ExpectedInstance, class_parents: dict[str, list[str]] | None = None) -> TaskRecord:
    return TaskRecord(
        id="t1",
        document="Jane Doe works at Example Lab.",
        expected=list(expected),
        corpus=corpus(),
        split=Split.DEV,
        class_parents=class_parents,
    )


def person(key="p1", **fields) -> ExpectedInstance:
    return ExpectedInstance(
        key=key,
        class_path="schemaorg.Person",
        fields=fields or {"name": "Jane Doe"},
    )


def produced(entities: dict[str, dict], classes=None, provenance=None) -> TripleSet:
    triples = frozenset()
    for key, fields in entities.items():
        triples |= make_triples(key, fields)
    return TripleSet(
        triples=triples,
        classes=classes or {},
        provenance=provenance or {},
    )


class TestCompare:
    def test_substring_is_not_a_match(self):
        """ "Doe" scored as "Doe-Smith" in the predecessor."""
        assert same_value("Doe-Smith", "Doe") is False
        assert same_value("Doe", "Doe-Smith") is False

    def test_contains_has_to_be_asked_for(self):
        assert same_value("Doe", "Doe-Smith", MatchMode.CONTAINS) is True

    def test_case_and_whitespace_do_not_matter(self):
        assert same_value("Jane  Doe", " jane doe ") is True

    def test_numbers_match_across_representations(self):
        assert same_value(1000, "1000") is True
        assert same_value(1000, "1,000") is True
        assert same_value(1000, 1001) is False

    def test_booleans_are_not_numbers(self):
        assert same_value(True, 1) is False

    def test_a_converted_quantity_is_not_the_same_answer_by_default(self):
        """The corpus closes the unit slot, so the spelling is the answer."""
        metre = Quantity(magnitude=1.0, unit="m")
        centimetre = Quantity(magnitude=100.0, unit="cm")
        assert same_value(metre, centimetre) is False

    def test_a_converted_quantity_matches_when_physical_equality_is_asked_for(self):
        metre = Quantity(magnitude=1.0, unit="m")
        centimetre = Quantity(magnitude=100.0, unit="cm")
        assert same_value(metre, centimetre, unit_match=UnitMatch.PHYSICAL) is True

    def test_a_spelling_variant_is_refused_by_default(self):
        """An arm with no enum emits MeV, an arm with one cannot. Accepting it
        hands the unconstrained arm a mark the constrained arm cannot earn."""
        assert (
            same_value(
                Quantity(magnitude=1.0, unit="mega_electron_volt"),
                Quantity(magnitude=1.0, unit="MeV"),
            )
            is False
        )

    def test_a_spelling_variant_is_visible_under_physical_equality(self):
        assert (
            same_value(
                Quantity(magnitude=1.0, unit="mega_electron_volt"),
                Quantity(magnitude=1.0, unit="MeV"),
                unit_match=UnitMatch.PHYSICAL,
            )
            is True
        )

    def test_a_wrong_unit_is_not_a_match(self):
        assert (
            same_value(
                Quantity(magnitude=1.0, unit="m"),
                Quantity(magnitude=1.0, unit="s"),
            )
            is False
        )

    def test_a_bare_number_does_not_answer_a_quantity(self):
        assert same_value(Quantity(magnitude=1.0, unit="m"), 1.0) is False


class TestQudtUnitSpellings:
    """QUDT vocabularies write SI prefixes with an underscore.

    pint parses none of those, so without a candidate ladder every prefixed
    unit in the Tier 1 corpus would be scored wrong.
    """

    @pytest.mark.parametrize(
        ("prefixed", "plain", "factor"),
        [
            ("centi_meter", "meter", 100.0),
            ("milli_meter", "meter", 1000.0),
            ("kilo_meter", "meter", 0.001),
            ("micro_meter", "meter", 1_000_000.0),
            ("kilo_gram", "gram", 0.001),
        ],
    )
    def test_a_prefixed_unit_converts_under_physical_equality(self, prefixed, plain, factor):
        """Reported as a dimension. Not the primary metric, because the
        document said one of them and the enum offers both."""
        assert same_value(
            Quantity(magnitude=1.0, unit=plain),
            Quantity(magnitude=factor, unit=prefixed),
            unit_match=UnitMatch.PHYSICAL,
        )

    @pytest.mark.parametrize(
        ("british", "american"),
        [
            ("metre", "meter"),
            ("kilo_metre", "kilo_meter"),
            ("centi_metre", "centi_meter"),
            ("litre", "liter"),
        ],
    )
    def test_spelling_is_reconciled_under_physical_equality(self, british, american):
        """The curated module writes metre, the QUDT corpus writes meter.

        Two source vocabularies disagreeing is a corpus problem, so it is not
        solved by making the grader lenient for everyone. Until the corpus
        layer reconciles them, the gap shows up in this dimension.
        """
        assert same_value(
            Quantity(magnitude=1.0, unit=british),
            Quantity(magnitude=1.0, unit=american),
            unit_match=UnitMatch.PHYSICAL,
        )

    def test_a_spelling_difference_is_not_a_match_by_default(self):
        assert (
            same_value(
                Quantity(magnitude=1.0, unit="metre"),
                Quantity(magnitude=1.0, unit="meter"),
            )
            is False
        )

    def test_conversion_works_across_spellings_too(self):
        assert same_value(
            Quantity(magnitude=1.0, unit="metre"),
            Quantity(magnitude=100.0, unit="centi_meter"),
            unit_match=UnitMatch.PHYSICAL,
        )

    @pytest.mark.parametrize("unit", ["degree_Celsius", "revolutions_per_minute"])
    def test_units_that_need_their_underscores_still_parse(self, unit):
        """The literal spelling is tried first, so these are not mangled."""
        assert same_value(Quantity(magnitude=2.0, unit=unit), Quantity(magnitude=2.0, unit=unit))

    def test_incompatible_dimensions_still_fail(self):
        assert (
            same_value(
                Quantity(magnitude=1.0, unit="kilo_meter"),
                Quantity(magnitude=1.0, unit="kilo_gram"),
            )
            is False
        )

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("kilo_meter", "kilometer"),
            ("centi_meter", "centimeter"),
            ("meter_per_second", "meter per second"),
            ("per_second", "1 / second"),
            ("number", "dimensionless"),
            ("degree_Celsius", "degree celsius"),
        ],
    )
    def test_the_rewrite_matches_the_upstream_rule(self, name, expected):
        """Ported from opensemantic.characteristics.quantitative, which
        generates the corpus. A naive underscore strip resolves 50% of the
        corpus unit names; this rule resolves 89%."""
        from oold_llm_bench.grading.compare import pint_name

        assert pint_name(name) == expected

    def test_compound_units_are_not_mangled_into_one_word(self):
        from oold_llm_bench.grading.compare import pint_name

        assert pint_name("kilo_gram_per_meter_cubed") == "kilogram per meter cubed"

    def test_an_unknown_unit_fails_rather_than_passing_silently(self):
        assert (
            same_value(
                Quantity(magnitude=1.0, unit="meter"),
                Quantity(magnitude=1.0, unit="nano_becquerel_per_widget"),
            )
            is False
        )


class TestWrittenUnitSpellings:
    """What a person writes, folded to the form a lookup can use.

    Real text supplies the cases: Wikipedia writes `km2` and `square
    kilometres` for one unit and a data sheet writes an exponent with the
    mathematical minus. Each rule folds one difference a reader would not
    notice, and none folds a difference that carries a unit.
    """

    @pytest.mark.parametrize(
        ("written", "expected"),
        [
            ("km²", "km2"),
            ("m³", "m3"),
            ("mW m\u22122", "mW m-2"),
            ("(km)", "km"),
            ("[m]", "m"),
            ("m.", "m"),
            ("km\u00a0", "km"),
            ("MW", "MW"),
        ],
    )
    def test_the_written_form_is_folded_without_losing_case(self, written, expected):
        assert normalise_unit(written) == expected

    def test_case_survives_because_it_carries_a_prefix(self):
        """`mW` is a milliwatt and `MW` a megawatt, so folding them here would
        throw away the answer before the symbol table ever sees it."""
        assert normalise_unit("mW") != normalise_unit("MW")

    @pytest.mark.parametrize(
        ("written", "key"),
        [
            ("square kilometres", "kilometer2"),
            ("square kilometers", "kilometer2"),
            ("sq mi", "mi2"),
            ("kilo_meter_squared", "kilometer2"),
            ("cubic metres", "meter3"),
            ("metres", "meter"),
            ("hectares", "hectare"),
            ("μm", "um"),
            ("km²", "km2"),
        ],
    )
    def test_two_spellings_of_one_unit_share_a_key(self, written, key):
        assert key in unit_forms(written)

    def test_the_exponent_moves_to_the_same_side_from_either_spelling(self):
        assert set(unit_forms("square kilometres")) & set(unit_forms("kilo_meter_squared"))

    def test_a_short_name_keeps_its_own_spelling(self):
        """Stripping a plural from `m` would leave nothing, so it is not
        stripped: the rule only applies where a stem survives it."""
        assert unit_forms("m") == ("m",)

    def test_nothing_is_produced_for_an_empty_unit(self):
        assert unit_forms("  ") == ()


class TestAlignment:
    def test_the_best_pairing_wins_not_the_first(self):
        """The predecessor took the first dict-order match and stopped."""
        expected = [person("p1", name="Jane Doe", email="jane@x.org")]
        candidates = {
            "a": make_triples("a", {"name": "Jane Doe"}),
            "b": make_triples("b", {"name": "Jane Doe", "email": "jane@x.org"}),
        }
        assert align(expected, candidates).matched() == {"p1": "b"}

    def test_alignment_does_not_depend_on_insertion_order(self):
        expected = [person("p1", name="Jane Doe", email="jane@x.org")]
        good = make_triples("b", {"name": "Jane Doe", "email": "jane@x.org"})
        poor = make_triples("a", {"name": "Jane Doe"})
        first = align(expected, {"a": poor, "b": good}).matched()
        second = align(expected, {"b": good, "a": poor}).matched()
        assert first == second == {"p1": "b"}

    def test_two_expectations_do_not_take_the_same_entity(self):
        expected = [person("p1", name="Jane"), person("p2", name="John")]
        candidates = {
            "a": make_triples("a", {"name": "Jane"}),
            "b": make_triples("b", {"name": "John"}),
        }
        assert align(expected, candidates).matched() == {"p1": "a", "p2": "b"}

    def test_an_entity_with_nothing_in_common_is_left_unmatched(self):
        result = align(
            [person("p1", name="Jane")],
            {"a": make_triples("a", {"name": "Someone Else"})},
        )
        assert result.matched() == {}
        assert result.unmatched_expected == ("p1",)
        assert result.unmatched_produced == ("a",)

    def test_an_expectation_after_a_gap_can_still_be_matched(self):
        """Permuting only the produced side never reached the last one.

        With fewer answers than expectations the pairing was truncated, so an
        answer that left out the middle entity was scored as having left out
        the last one as well."""
        expected = [person(f"p{i}", name=f"Name {i}") for i in range(1, 6)]
        candidates = {f"e{i}": make_triples(f"e{i}", {"name": f"Name {i}"}) for i in (1, 2, 4, 5)}
        result = align(expected, candidates)
        assert result.matched() == {"p1": "e1", "p2": "e2", "p4": "e4", "p5": "e5"}
        assert result.unmatched_expected == ("p3",)

    def test_overlap_is_a_share_of_the_required_fields(self):
        instance = person("p1", name="Jane", email="jane@x.org")
        half = make_triples("a", {"name": "Jane"})
        assert overlap(instance, half) == pytest.approx(0.5)

    def test_large_tasks_fall_back_and_say_so(self):
        expected = [person(f"p{i}", name=f"n{i}") for i in range(10)]
        candidates = {f"a{i}": make_triples(f"a{i}", {"name": f"n{i}"}) for i in range(10)}
        result = align(expected, candidates)
        assert result.exhaustive is False
        assert len(result.matched()) == 10


class TestScore:
    def test_the_right_class_with_every_field_null_still_credits_class(self):
        """Found live: narrowing the union to the true class made an entity's
        answer collapse to the class alone with every property null. With no
        triple to carry it, the entity was invisible to alignment entirely,
        and a class stated correctly scored as a miss on every dimension."""
        from oold_llm_bench.grading import Dimension

        result = score_task(
            task(person("p1", name="Jane Doe")),
            produced({}, classes={"a": "schemaorg.Person"}),
        )
        assert result.dimensions[Dimension.CLASS].f1 == pytest.approx(1.0)
        assert result.dimensions[Dimension.VALUE].f1 == 0.0

    def test_f1_of_a_perfect_answer_is_one(self):
        result = score_task(
            task(person("p1", name="Jane Doe")),
            produced({"a": {"name": "Jane Doe"}}, classes={"a": "schemaorg.Person"}),
        )
        assert result.primary == pytest.approx(1.0)

    def test_a_banana_scores_zero_on_value(self):
        """The predecessor passed a LaboratoryProcess named "banana"."""
        result = score_task(
            task(person("p1", name="Jane Doe")),
            produced({"a": {"name": "banana"}}, classes={"a": "schemaorg.Person"}),
        )
        assert result.primary == 0.0
        assert result.dimensions[list(result.dimensions)[1]].f1 >= 0.0

    def test_the_right_class_with_the_wrong_value_still_fails_on_value(self):
        result = score_task(
            task(person("p1", name="Jane Doe")),
            produced({"a": {"name": "banana"}}, classes={"a": "schemaorg.Person"}),
        )
        from oold_llm_bench.grading import Dimension

        assert result.dimensions[Dimension.CLASS].f1 == pytest.approx(1.0)
        assert result.dimensions[Dimension.VALUE].f1 == 0.0

    def test_an_arm_with_no_class_is_still_scored_on_value(self):
        """A0 assigns no class. That is the condition, not a failure."""
        from oold_llm_bench.grading import Dimension

        result = score_task(
            task(person("p1", name="Jane Doe")),
            produced({"a": {"name": "Jane Doe"}}, classes={}),
        )
        assert result.dimensions[Dimension.VALUE].f1 == pytest.approx(1.0)
        assert result.dimensions[Dimension.CLASS].f1 == 0.0

    def test_an_invented_entity_costs_precision(self):
        result = score_task(
            task(person("p1", name="Jane Doe")),
            produced({
                "a": {"name": "Jane Doe"},
                "b": {"name": "Nobody", "email": "x@y.z"},
            }),
        )
        assert result.primary < 1.0
        from oold_llm_bench.grading import Dimension

        assert result.dimensions[Dimension.ENTITY].false_positives == 1

    def test_a_missing_entity_costs_recall_once_not_twice(self):
        """The predecessor raised both UNEXPECTED_NEW_ENTITY and
        WRONG_NEW_COUNT for one mistake."""
        from oold_llm_bench.grading import Dimension

        result = score_task(
            task(person("p1", name="Jane"), person("p2", name="John")),
            produced({"a": {"name": "Jane"}}),
        )
        entity = result.dimensions[Dimension.ENTITY]
        assert entity.false_negatives == 1
        assert entity.false_positives == 0

    def test_optional_fields_are_scored_when_present_never_when_absent(self):
        instance = ExpectedInstance(
            key="p1",
            class_path="schemaorg.Person",
            fields={"name": "Jane"},
            optional_fields={"email": "jane@x.org"},
        )
        without = score_task(task(instance), produced({"a": {"name": "Jane"}}))
        assert without.primary == pytest.approx(1.0)

        wrong = score_task(
            task(instance),
            produced({"a": {"name": "Jane", "email": "other@x.org"}}),
        )
        assert wrong.primary < 1.0

    def test_a_subclass_satisfies_the_expectation_when_allowed(self):
        from oold_llm_bench.grading import Dimension

        result = score_task(
            task(person("p1", name="Jane")),
            produced({"a": {"name": "Jane"}}, classes={"a": "schemaorg.Patient"}),
            subclasses={"schemaorg.Patient": {"schemaorg.Person"}},
        )
        assert result.dimensions[Dimension.CLASS].f1 == pytest.approx(1.0)

    def test_a_subclass_is_refused_when_the_task_forbids_it(self):
        from oold_llm_bench.grading import Dimension

        strict = ExpectedInstance(
            key="p1",
            class_path="schemaorg.Person",
            fields={"name": "Jane"},
            allow_subclass=False,
        )
        result = score_task(
            task(strict),
            produced({"a": {"name": "Jane"}}, classes={"a": "schemaorg.Patient"}),
            subclasses={"schemaorg.Patient": {"schemaorg.Person"}},
        )
        assert result.dimensions[Dimension.CLASS].f1 == 0.0

    def test_provenance_is_its_own_dimension(self):
        from oold_llm_bench.grading import Dimension

        with_span = score_task(
            task(person("p1", name="Jane")),
            produced({"a": {"name": "Jane"}}, provenance={"a": "line 3"}),
        )
        without = score_task(task(person("p1", name="Jane")), produced({"a": {"name": "Jane"}}))
        assert with_span.dimensions[Dimension.PROVENANCE].f1 == pytest.approx(1.0)
        assert without.dimensions[Dimension.PROVENANCE].f1 == 0.0

    def test_parse_errors_are_carried_through(self):
        answer = produced({"a": {"name": "Jane"}})
        answer = TripleSet(
            triples=answer.triples,
            classes=answer.classes,
            provenance=answer.provenance,
            parse_errors=7,
        )
        assert score_task(task(person("p1", name="Jane")), answer).parse_errors == 7

    def test_the_result_is_serialisable(self):
        result = score_task(task(person("p1", name="Jane")), produced({"a": {"name": "Jane"}}))
        described = result.describe()
        assert described["primary_f1"] == 1.0
        assert set(described["dimensions"]) >= {"value", "class", "entity"}


class TestVocabularyAwareValueAndProperty:
    """A value correctly read but filed under a defensible synonym should not
    be charged a miss on the value and an invention on the name, for one
    answer. See `PropertyHierarchy` and `Dimension.VALUE_NEAR_PROPERTY` /
    `Dimension.PROPERTY_NEAR`."""

    def _hierarchy(self):
        from oold_llm_bench.grading.vocabulary import PropertyHierarchy

        return PropertyHierarchy(parents={"author": frozenset({"creator"}), "creator": frozenset()})

    def test_strict_value_and_property_are_unmoved_by_the_hierarchy(self):
        """The lenient dimensions exist beside the strict ones; a hierarchy
        in play must never move what the strict ones already counted."""
        from oold_llm_bench.grading import Dimension

        instance = ExpectedInstance(key="p1", class_path="schemaorg.Person", fields={"author": "Jane Doe"})
        result = score_task(
            task(instance),
            produced({"a": {"creator": "Jane Doe"}}, classes={"a": "schemaorg.Person"}),
            vocabulary=self._hierarchy(),
        )
        assert result.dimensions[Dimension.VALUE].f1 == 0.0
        assert result.dimensions[Dimension.PROPERTY].f1 == 0.0

    def test_a_broader_name_recovers_the_value_and_the_property(self):
        from oold_llm_bench.grading import Dimension

        instance = ExpectedInstance(key="p1", class_path="schemaorg.Person", fields={"author": "Jane Doe"})
        result = score_task(
            task(instance),
            produced({"a": {"creator": "Jane Doe"}}, classes={"a": "schemaorg.Person"}),
            vocabulary=self._hierarchy(),
        )
        assert result.dimensions[Dimension.VALUE_NEAR_PROPERTY].f1 == pytest.approx(1.0)
        assert result.dimensions[Dimension.PROPERTY_NEAR].f1 == pytest.approx(1.0)

    def test_an_unrelated_name_recovers_nothing(self):
        from oold_llm_bench.grading import Dimension

        instance = ExpectedInstance(key="p1", class_path="schemaorg.Person", fields={"author": "Jane Doe"})
        result = score_task(
            task(instance),
            produced({"a": {"award": "Jane Doe"}}, classes={"a": "schemaorg.Person"}),
            vocabulary=self._hierarchy(),
        )
        assert result.dimensions[Dimension.VALUE_NEAR_PROPERTY].f1 == 0.0
        assert result.dimensions[Dimension.PROPERTY_NEAR].f1 == 0.0

    def test_a_broader_name_with_the_wrong_value_is_not_forgiven(self):
        """The relation excuses the name, never the value underneath it."""
        from oold_llm_bench.grading import Dimension

        instance = ExpectedInstance(key="p1", class_path="schemaorg.Person", fields={"author": "Jane Doe"})
        result = score_task(
            task(instance),
            produced({"a": {"creator": "banana"}}, classes={"a": "schemaorg.Person"}),
            vocabulary=self._hierarchy(),
        )
        assert result.dimensions[Dimension.VALUE_NEAR_PROPERTY].f1 == 0.0

    def test_no_hierarchy_reports_neither_near_dimension(self):
        """An explicit empty hierarchy, not the omitted default: the default
        reads the real built file, which is populated and would make this
        assertion pass for the wrong reason."""
        from oold_llm_bench.grading import Dimension
        from oold_llm_bench.grading.vocabulary import PropertyHierarchy

        result = score_task(
            task(person("p1", name="Jane")),
            produced({"a": {"name": "Jane"}}),
            vocabulary=PropertyHierarchy(parents={}),
        )
        assert Dimension.VALUE_NEAR_PROPERTY not in result.dimensions
        assert Dimension.PROPERTY_NEAR not in result.dimensions


class TestClassLineage:
    """Answering `Person` for an `Actor` is under-specified, not an invention.
    Read from the task's own `class_parents`, never a property vocabulary."""

    def _lineage(self):
        return {"schemaorg.Actor": ["schemaorg.Person"]}

    def test_strict_class_is_unmoved_by_the_lineage(self):
        from oold_llm_bench.grading import Dimension

        instance = ExpectedInstance(key="p1", class_path="schemaorg.Actor", fields={"name": "Jane"})
        result = score_task(
            task(instance, class_parents=self._lineage()),
            produced({"a": {"name": "Jane"}}, classes={"a": "schemaorg.Person"}),
        )
        assert result.dimensions[Dimension.CLASS].f1 == 0.0

    def test_the_ancestor_is_forgiven_under_class_near(self):
        from oold_llm_bench.grading import Dimension

        instance = ExpectedInstance(key="p1", class_path="schemaorg.Actor", fields={"name": "Jane"})
        result = score_task(
            task(instance, class_parents=self._lineage()),
            produced({"a": {"name": "Jane"}}, classes={"a": "schemaorg.Person"}),
        )
        assert result.dimensions[Dimension.CLASS_NEAR].f1 == pytest.approx(1.0)

    def test_an_unrelated_class_is_not_forgiven(self):
        from oold_llm_bench.grading import Dimension

        instance = ExpectedInstance(key="p1", class_path="schemaorg.Actor", fields={"name": "Jane"})
        result = score_task(
            task(instance, class_parents=self._lineage()),
            produced({"a": {"name": "Jane"}}, classes={"a": "schemaorg.Organization"}),
        )
        assert result.dimensions[Dimension.CLASS_NEAR].f1 == 0.0

    def test_a_descendant_is_forgiven_too(self):
        """The same relation in the other direction: `Actor` for a `Person`."""
        from oold_llm_bench.grading import Dimension

        instance = ExpectedInstance(key="p1", class_path="schemaorg.Person", fields={"name": "Jane"})
        result = score_task(
            task(instance, class_parents=self._lineage()),
            produced({"a": {"name": "Jane"}}, classes={"a": "schemaorg.Actor"}),
        )
        assert result.dimensions[Dimension.CLASS_NEAR].f1 == pytest.approx(1.0)

    def test_an_instance_that_forbids_subclassing_is_not_forgiven_either(self):
        from oold_llm_bench.grading import Dimension

        strict = ExpectedInstance(key="p1", class_path="schemaorg.Actor", fields={"name": "Jane"}, allow_subclass=False)
        result = score_task(
            task(strict, class_parents=self._lineage()),
            produced({"a": {"name": "Jane"}}, classes={"a": "schemaorg.Person"}),
        )
        assert result.dimensions[Dimension.CLASS_NEAR].f1 == 0.0

    def test_no_lineage_reports_no_class_near_dimension(self):
        from oold_llm_bench.grading import Dimension

        result = score_task(task(person("p1", name="Jane")), produced({"a": {"name": "Jane"}}))
        assert Dimension.CLASS_NEAR not in result.dimensions


class TestMultiEntity:
    """What a document holding several entities can get wrong.

    Missing one, inventing one and emitting one twice all land in the same
    triple F1. Each case below asserts which dimension moves and, as
    importantly, which does not.
    """

    @staticmethod
    def five() -> TaskRecord:
        return task(*(person(f"p{i}", name=f"Name {i}", email=f"n{i}@x.org") for i in range(1, 6)))

    @staticmethod
    def all_five() -> dict[str, dict]:
        return {f"e{i}": {"name": f"Name {i}", "email": f"n{i}@x.org"} for i in range(1, 6)}

    def test_a_perfect_multi_entity_answer_scores_one_everywhere(self):
        from oold_llm_bench.grading import Dimension

        instances = [
            ExpectedInstance(
                key=f"p{i}",
                class_path="schemaorg.Person",
                fields={"name": f"Name {i}", "height": Quantity(magnitude=float(i), unit="meter")},
            )
            for i in range(1, 6)
        ]
        answer = produced(
            {f"e{i}": {"name": f"Name {i}", "height": Quantity(magnitude=float(i), unit="meter")} for i in range(1, 6)},
            classes={f"e{i}": "schemaorg.Person" for i in range(1, 6)},
            provenance={f"e{i}": f"line {i}" for i in range(1, 6)},
        )
        result = score_task(task(*instances), answer, shortlist=["schemaorg.Person"])
        assert set(result.dimensions) >= {Dimension.ENTITY, Dimension.DUPLICATE}
        # GROUNDED is a rate and not a correctness score: it asks whether the
        # value appears in the document, and this task's document does not
        # spell the values at all. Averaging it in with the rest is the
        # mistake its own docstring warns about, so it is excluded here.
        assert all(
            score.f1 == pytest.approx(1.0)
            for name, score in result.dimensions.items()
            if name is not Dimension.GROUNDED
        )
        assert result.dimensions[Dimension.ENTITY].precision == pytest.approx(1.0)
        assert result.dimensions[Dimension.ENTITY].recall == pytest.approx(1.0)

    def test_a_missing_entity_moves_recall_and_leaves_precision_alone(self):
        from oold_llm_bench.grading import Dimension

        answer = self.all_five()
        del answer["e3"]
        result = score_task(self.five(), produced(answer))
        entity = result.dimensions[Dimension.ENTITY]
        assert entity.recall == pytest.approx(0.8)
        assert entity.precision == pytest.approx(1.0)
        assert result.dimensions[Dimension.DUPLICATE].f1 == pytest.approx(1.0)

    def test_an_invented_entity_moves_precision_and_leaves_recall_alone(self):
        from oold_llm_bench.grading import Dimension

        answer = self.all_five()
        answer["e9"] = {"name": "Nobody At All", "email": "nobody@elsewhere.org"}
        result = score_task(self.five(), produced(answer))
        entity = result.dimensions[Dimension.ENTITY]
        assert entity.precision == pytest.approx(5 / 6)
        assert entity.recall == pytest.approx(1.0)
        assert result.dimensions[Dimension.DUPLICATE].f1 == pytest.approx(1.0)

    def test_a_repeated_entity_moves_duplicate_and_precision_but_not_recall(self):
        from oold_llm_bench.grading import Dimension

        answer = self.all_five()
        answer["e9"] = dict(answer["e3"])
        result = score_task(self.five(), produced(answer))
        entity = result.dimensions[Dimension.ENTITY]
        assert entity.precision == pytest.approx(5 / 6)
        assert entity.recall == pytest.approx(1.0)
        assert result.dimensions[Dimension.DUPLICATE].f1 == pytest.approx(10 / 11)

    def test_repeating_and_inventing_are_told_apart(self):
        """Both cost the same entity precision. Only one is a duplicate."""
        from oold_llm_bench.grading import Dimension

        repeated = self.all_five()
        repeated["e9"] = dict(repeated["e3"])
        invented = self.all_five()
        invented["e9"] = {"name": "Nobody At All", "email": "nobody@elsewhere.org"}

        first = score_task(self.five(), produced(repeated))
        second = score_task(self.five(), produced(invented))
        assert first.dimensions[Dimension.ENTITY].precision == pytest.approx(
            second.dimensions[Dimension.ENTITY].precision
        )
        assert first.dimensions[Dimension.DUPLICATE].f1 < second.dimensions[Dimension.DUPLICATE].f1

    def test_sharing_only_a_class_is_not_a_duplicate(self):
        """A document naming two people is the ordinary case."""
        from oold_llm_bench.grading import Dimension

        answer = self.all_five()
        answer["e9"] = {"name": "Nobody At All", "email": "nobody@elsewhere.org"}
        classes = dict.fromkeys(answer, "schemaorg.Person")
        result = score_task(self.five(), produced(answer, classes=classes))
        assert result.dimensions[Dimension.DUPLICATE].f1 == pytest.approx(1.0)

    def test_two_identical_inventions_are_two_inventions(self):
        """They agree with each other and with nothing in the corpus."""
        from oold_llm_bench.grading import Dimension

        answer = self.all_five()
        answer["e8"] = {"name": "Nobody At All", "email": "nobody@elsewhere.org"}
        answer["e9"] = {"name": "Nobody At All", "email": "nobody@elsewhere.org"}
        result = score_task(self.five(), produced(answer))
        assert result.dimensions[Dimension.ENTITY].false_positives == 2
        assert result.dimensions[Dimension.DUPLICATE].f1 == pytest.approx(1.0)

    def test_a_single_entity_task_reads_one_not_zero(self):
        """Historical cells hold one entity each. A dimension that read 0.00
        where it does not apply would drag every one of them down."""
        from oold_llm_bench.grading import Dimension

        result = score_task(task(person("p1", name="Jane Doe")), produced({"a": {"name": "Jane Doe"}}))
        assert result.dimensions[Dimension.DUPLICATE].f1 == pytest.approx(1.0)

    def test_an_answer_that_found_nothing_carries_no_duplicate_score(self):
        """Absent, not zero. There is no entity for a duplicate to be of."""
        from oold_llm_bench.grading import Dimension

        result = score_task(
            task(person("p1", name="Jane Doe")),
            produced({"a": {"name": "Someone Else"}}),
        )
        assert Dimension.DUPLICATE not in result.dimensions

    def test_a_stub_restating_the_value_under_another_property_is_a_duplicate(self):
        """The commonest way a model duplicates an entity, and it was invisible.

        Asked to link, a model writes the target again as a stub that names
        it. The name lands in whatever property the model reaches for, and the
        corpus filed the same string somewhere else, so property agreement is
        zero and the restatement scored as an invention. Over the 1,920 cells
        of e4 the dimension counted 200 restatements and missed 405.
        """
        from oold_llm_bench.grading import Dimension

        answer = self.all_five()
        answer["e9"] = {"alternateName": "Name 3"}
        result = score_task(self.five(), produced(answer))
        assert result.dimensions[Dimension.DUPLICATE].false_positives == 1

        entities = {key: make_triples(key, fields) for key, fields in answer.items()}
        expected = list(self.five().expected)
        alignment = align(expected, entities)
        assert duplicates(expected, entities, alignment)["e9"] == "p3"

    def test_a_short_or_numeric_agreement_is_coincidence_and_not_a_duplicate(self):
        """A value has to be distinctive before agreeing on it means anything.

        A boolean, a small enumeration and a short code recur across unrelated
        entities by chance. Counting those would make every document with two
        booleans in it look like a repetition.
        """
        from oold_llm_bench.grading import Dimension

        answer = self.all_five()
        answer["e9"] = {"status": "n3@x.org"[:4]}
        result = score_task(self.five(), produced(answer))
        assert result.dimensions[Dimension.DUPLICATE].false_positives == 0

    def test_the_duplicate_names_the_entity_it_restates(self):
        answer = self.all_five()
        answer["e9"] = dict(answer["e3"])
        entities = {key: make_triples(key, fields) for key, fields in answer.items()}
        expected = list(self.five().expected)
        alignment = align(expected, entities)
        assert set(duplicates(expected, entities, alignment).values()) == {"p3"}


class TestNegativeControl:
    def test_an_agent_answering_from_a_different_document_scores_near_zero(self):
        """If this passes, the grader is not measuring what it claims."""
        result = score_task(
            task(person("p1", name="Jane Doe"), person("p2", name="John Roe")),
            produced(
                {
                    "x": {"name": "Wolfgang Pauli", "email": "w@zurich.ch"},
                    "y": {"name": "Lise Meitner", "email": "l@berlin.de"},
                },
                classes={"x": "schemaorg.Product", "y": "schemaorg.Event"},
            ),
        )
        assert result.primary == 0.0

    def test_an_empty_answer_scores_zero_not_one(self):
        result = score_task(
            task(person("p1", name="Jane Doe")),
            TripleSet(triples=frozenset(), classes={}, provenance={}),
        )
        assert result.primary == 0.0


class TestScoreArithmetic:
    def test_scores_add(self):
        total = Score(1, 2, 3) + Score(10, 20, 30)
        assert (total.true_positives, total.false_positives, total.false_negatives) == (
            11,
            22,
            33,
        )

    def test_an_empty_score_is_zero_not_undefined(self):
        empty = Score()
        assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)


class TestADurationIsComparedAsAnInterval:
    """Markup stores a duration in ISO 8601 and a page writes it for a reader.

    Measured over the Web Data Commons corpus: of 324 expected ``prepTime``,
    ``cookTime`` and ``totalTime`` values, none appear in the page verbatim.
    Compared as text, every one of them is wrong whatever the model answered,
    so the dimension reported the two notations and not the extraction.
    """

    def test_the_two_notations_for_one_interval_agree(self):
        from oold_llm_bench.grading.compare import same_value

        assert same_value("PT5M", "5 minutes")
        assert same_value("PT1H30M", "1 hour 30 minutes")
        assert same_value("PT1H30M", "90 minutes")
        assert same_value("P1D", "24 hours")

    def test_a_different_interval_still_fails(self):
        from oold_llm_bench.grading.compare import same_value

        assert not same_value("PT5M", "10 minutes")
        assert not same_value("PT1H", "1 minute")

    def test_prose_that_merely_holds_a_duration_is_not_one(self):
        """Otherwise it matches any answer stating five minutes of anything."""
        from oold_llm_bench.grading.compare import same_value

        assert not same_value("PT5M", "ready in 5 minutes, serves 4")

    def test_a_bare_number_is_a_count_until_something_says_what_of(self):
        from oold_llm_bench.grading.compare import _as_duration

        assert _as_duration("5") is None
        assert _as_duration("hello") is None
        assert _as_duration("") is None

    def test_text_that_is_not_a_duration_compares_as_text(self):
        from oold_llm_bench.grading.compare import same_value

        assert same_value("Breakfast", "Breakfast")
        assert not same_value("Breakfast", "Dinner")


class TestAValueNamedInMoreWords:
    """A corpus files a short designation and a document gives the full one.

    Wikidata records the founder as "Orange" where the lead says "Orange
    Group", and as "Christian Albert" where the lead says "Christian Albert,
    Duke of Holstein-Gottorp". Strict scoring calls both wrong and reports an
    answer that read the document correctly as having invented a value.
    """

    def _scored(self, expected_fields, produced_fields):
        record = task(ExpectedInstance(key="e1", class_path="Organization", fields=expected_fields))
        produced = TripleSet(triples=make_triples("e1", produced_fields), classes={"e1": "Organization"}, provenance={})
        return score_task(record, produced).dimensions

    def test_the_fuller_designation_is_recovered(self):
        dims = self._scored(
            {"name": "Orange", "founder": "Christian Albert"},
            {"name": "Orange Group", "founder": "Christian Albert, Duke of Holstein-Gottorp"},
        )
        assert dims[Dimension.VALUE].f1 == 0.0
        assert dims[Dimension.VALUE_NEAR].f1 == 1.0

    def test_another_entity_is_still_wrong(self):
        dims = self._scored({"name": "Fashoda Incident"}, {"name": "Europe"})
        assert dims[Dimension.VALUE_NEAR].f1 == 0.0

    def test_a_designation_too_short_to_find_inside_another_is_not_credited(self):
        """ "Acme" is a substring of a great many strings, so containment there
        would be a coincidence rather than the same thing named in full."""
        dims = self._scored({"name": "Acme"}, {"name": "Acme Holdings International"})
        assert dims[Dimension.VALUE_NEAR].f1 == 0.0

    def test_it_is_never_smaller_than_the_strict_reading(self):
        dims = self._scored({"name": "Orange Group"}, {"name": "Orange Group"})
        assert dims[Dimension.VALUE].f1 == 1.0
        assert dims[Dimension.VALUE_NEAR].f1 == 1.0

    def test_the_reverse_containment_is_not_credited(self):
        """A fragment of the right name is not the right name in more words."""
        dims = self._scored({"name": "Christian Albert, Duke of Holstein-Gottorp"}, {"name": "Christian Albert"})
        assert dims[Dimension.VALUE_NEAR].f1 == 0.0


class TestADateIsComparedAsACalendarDate:
    """The corpus files a date as ISO 8601 and the document writes it out.

    Measured over the Wikidata leads, a date in the wrong notation is half of
    every wrong value the chain produces, and none of those answers was wrong
    about the date.
    """

    def test_the_two_notations_for_one_date_agree(self):
        from oold_llm_bench.grading.compare import same_value

        assert same_value("1898-11-03", "3 November 1898")
        assert same_value("1898-11-03", "November 3, 1898")
        assert same_value("1898-11-03", "3rd November 1898")

    def test_a_different_date_still_fails(self):
        from oold_llm_bench.grading.compare import same_value

        assert not same_value("2016-04-14", "2016-04-22")
        assert not same_value("1898-11-03", "3 November 1899")

    def test_a_year_alone_is_less_precise_and_not_another_spelling(self):
        from oold_llm_bench.grading.compare import same_value

        assert not same_value("1901-01-28", "1901")

    def test_a_slash_form_is_refused_rather_than_guessed(self):
        """03/11/1898 is November or March depending on where it was written,
        and a grader that picks one decides what the document did not."""
        from oold_llm_bench.grading.compare import _as_date

        assert _as_date("03/11/1898") is None
