"""Reading an arm's JSON answer.

One extractor serves every arm, so these cases are the shapes a grounded arm
and an unconstrained one both produce. If the two needed different readers, a
measured difference between them would be partly a difference between readers.
"""

import pytest

from oold_llm_bench.extract import extract_json
from oold_llm_bench.grading import Quantity


def values(result):
    return {(t.prop, t.value) for t in result.triples}


class TestShapesEveryArmProduces:
    def test_a_grounded_instance(self):
        result = extract_json({"type": ["Length"], "value": 1.75, "unit": "meter"})
        assert result.classes == {"e1": "Length"}
        assert values(result) == {("value", Quantity(magnitude=1.75, unit="meter"))}

    def test_a_list_behind_a_wrapper(self):
        result = extract_json({
            "entities": [
                {"type": "Length", "value": 1.75, "unit": "meter"},
                {"type": "Mass", "value": 2.0, "unit": "kilo_gram"},
            ]
        })
        assert set(result.classes.values()) == {"Length", "Mass"}
        assert len(result.triples) == 2

    def test_a_bare_list(self):
        result = extract_json([{"kind": "Length", "magnitude": 1.75, "uom": "meter"}])
        assert result.classes == {"e1": "Length"}

    def test_entities_keyed_by_name(self):
        result = extract_json({"q1": {"class": "Length", "value": 1.75, "unit": "m"}})
        assert result.classes == {"q1": "Length"}

    def test_an_unconstrained_answer_with_no_class(self):
        """A0 names no class, and the values still have to be readable.

        The key is recorded as the class it asserts, because an arm with no
        schema to fix the envelope states its class there and nowhere else.
        Dropping it discarded the one field the answer carried, which made
        91 per cent of one arm's class failures an artefact.

        Recording it cannot credit an arm that did not name a class:
        comparison is exact, so a property-ish key never matches a class
        path. It scores exactly as an absent class did.
        """
        result = extract_json({"length": {"amount": 1.75, "units": "m"}})
        assert result.classes == {"length": "length"}
        assert values(result) == {("value", Quantity(magnitude=1.75, unit="m"))}

    def test_a_key_that_names_the_class_is_credited(self):
        """The shape every schema-free arm actually answers in."""
        result = extract_json({"Length": {"value": 1.75, "unit": "meter"}})
        assert result.classes == {"Length": "Length"}

    def test_an_entity_that_states_its_class_overrides_the_key(self):
        result = extract_json({"e1": {"type": "Length", "value": 1.75, "unit": "meter"}})
        assert result.classes == {"e1": "Length"}

    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "Length", "value": 1.75, "unit": "meter"},
            {"class_path": "Length", "magnitude": 1.75, "units": "meter"},
            {"@type": "Length", "reading": "1.75", "uom": "meter"},
        ],
    )
    def test_synonymous_keys_give_the_same_triple(self, payload):
        """An arm should not be scored on its choice of key names."""
        result = extract_json(payload)
        assert result.classes == {"e1": "Length"}
        assert values(result) == {("value", Quantity(magnitude=1.75, unit="meter"))}


class TestQuantities:
    def test_value_and_unit_become_one_quantity(self):
        result = extract_json({"value": 1.75, "unit": "meter"})
        ((prop, value),) = values(result)
        assert prop == "value"
        assert isinstance(value, Quantity)

    def test_a_value_with_no_unit_stays_a_number(self):
        result = extract_json({"type": "Length", "value": 1.75})
        assert values(result) == {("value", 1.75)}

    def test_a_numeric_string_is_read_as_a_number(self):
        result = extract_json({"value": "1,750.5", "unit": "meter"})
        ((_, value),) = values(result)
        assert value == Quantity(magnitude=1750.5, unit="meter")

    def test_a_magnitude_that_is_not_a_number_is_an_error(self):
        result = extract_json({"value": "about a metre", "unit": "meter"})
        assert result.triples == frozenset()
        assert result.parse_errors == 1


class TestLeniencyIsCounted:
    """A lenient read must never inflate a score without saying so."""

    @pytest.mark.parametrize("payload", [{}, {"foo": None}, "I could not find anything", 42])
    def test_an_unreadable_answer_costs_a_parse_error(self, payload):
        result = extract_json(payload)
        assert result.triples == frozenset()
        assert result.parse_errors >= 1

    def test_a_null_field_yields_no_triple(self):
        """Scoring a null would let an answer of nothing earn triples."""
        result = extract_json({"type": "Length", "value": 1.75, "unit": "m", "note": None})
        assert len(result.triples) == 1

    def test_a_clean_answer_costs_nothing(self):
        result = extract_json({"type": "Length", "value": 1.75, "unit": "meter"})
        assert result.parse_errors == 0


class TestListsAndNesting:
    def test_a_repeated_field_becomes_one_triple_per_element(self):
        result = extract_json({"type": "Person", "email": ["a@x.org", "b@x.org"]})
        assert len([t for t in result.triples if t.prop == "email"]) == 2

    def test_a_nested_object_becomes_its_own_entity(self):
        result = extract_json({"type": "Product", "name": "widget", "offer": {"type": "Offer", "price": 9}})
        assert set(result.classes.values()) == {"Product", "Offer"}

    def test_provenance_is_picked_up_when_offered(self):
        result = extract_json({"type": "Length", "value": 1.0, "unit": "m", "source": "line 3"})
        assert result.provenance == {"e1": "line 3"}


class TestRoundTrip:
    def test_a_generated_task_answered_correctly_scores_one(self):
        """The full loop with no model: generate, answer, read, score."""
        from oold_llm_bench.corpus.quantities import QuantityKind, generate_task
        from oold_llm_bench.grading.score import score_task

        kinds = [QuantityKind(name="Length", units=("meter",))]
        task = generate_task(kinds, task_id="t1", seed=1)
        expected = task.expected[0].fields["value"]

        answer = extract_json({
            "entities": [
                {
                    "type": task.expected[0].class_path,
                    "value": expected.magnitude,
                    "unit": expected.unit,
                }
            ]
        })
        assert score_task(task, answer).primary == pytest.approx(1.0)

    def test_the_same_answer_in_another_shape_scores_the_same(self):
        """A grounded arm and an unconstrained one are read alike."""
        from oold_llm_bench.corpus.quantities import QuantityKind, generate_task
        from oold_llm_bench.grading.score import score_task

        kinds = [QuantityKind(name="Length", units=("meter",))]
        task = generate_task(kinds, task_id="t1", seed=1)
        expected = task.expected[0].fields["value"]

        loose = extract_json({"measurement": {"amount": expected.magnitude, "units": expected.unit}})
        result = score_task(task, loose)
        from oold_llm_bench.grading import Dimension

        assert result.dimensions[Dimension.VALUE].f1 == pytest.approx(1.0)
        assert result.dimensions[Dimension.CLASS].f1 == 0.0


class TestErrorsAreNotDoubleCounted:
    """One failure, one error.

    The predecessor raised both UNEXPECTED_NEW_ENTITY and WRONG_NEW_COUNT for
    a single mistake, which made its error counts unusable for comparison.
    """

    def test_an_unreadable_quantity_counts_once(self):
        result = extract_json({"value": "about a metre", "unit": "meter"})
        assert result.parse_errors == 1

    def test_an_empty_entity_counts_once(self):
        assert extract_json({}).parse_errors == 1

    def test_two_bad_entities_count_twice(self):
        result = extract_json({"entities": [{}, {}]})
        assert result.parse_errors == 2

    def test_a_good_entity_beside_a_bad_one_counts_once(self):
        result = extract_json({"entities": [{"type": "Length", "value": 1.0, "unit": "m"}, {}]})
        assert result.parse_errors == 1
        assert len(result.triples) == 1


class TestInventedKeysDoNotCollideWithStatedOnes:
    """The prompt asks for "a short identifier for this entity, e.g. e1", so
    the model's ids and the reader's come from one namespace. An answer that
    ids some entities and not others used to merge them silently."""

    def test_three_entities_with_one_missing_id_stay_three(self):
        from oold_llm_bench.extract.json_answer import extract_json

        found = extract_json({
            "entities": [
                {"id": "e1", "type": "Person", "name": "Ada"},
                {"type": "Organization", "name": "Acme"},
                {"id": "e3", "type": "Book", "name": "Zed"},
            ]
        })
        assert len(found.entities()) == 3
        assert found.classes["e1"] == "Person", "the first entity's class was overwritten"
        assert set(found.classes.values()) == {"Person", "Organization", "Book"}

    def test_a_stated_id_matching_an_invented_one_is_kept_apart(self):
        from oold_llm_bench.extract.json_answer import extract_json

        found = extract_json({
            "entities": [
                {"type": "Organization", "name": "Acme"},
                {"id": "e1", "type": "Person", "name": "Ada"},
            ]
        })
        assert len(found.entities()) == 2
        assert set(found.classes.values()) == {"Organization", "Person"}
