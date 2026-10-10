"""Triple schema and property normalisation."""

import pytest
from pydantic import ValidationError

from oold_llm_bench.grading import (
    Dimension,
    Quantity,
    Triple,
    TripleSet,
    make_triple,
    make_triples,
    merge,
    normalise_property,
)


class TestNormaliseProperty:
    @pytest.mark.parametrize(
        "spelling",
        [
            "first_name",
            "firstName",
            "FirstName",
            "first-name",
            "first name",
            "  first name  ",
            "first.name",
        ],
    )
    def test_spellings_of_one_property_collide(self, spelling):
        assert normalise_property(spelling) == "first_name"

    def test_distinct_properties_stay_distinct(self):
        assert normalise_property("name") != normalise_property("username")

    def test_digits_start_a_new_word(self):
        assert normalise_property("address2Line") == "address2_line"

    def test_acronyms_do_not_explode(self):
        assert normalise_property("URL") == "url"


class TestTriple:
    def test_make_triple_normalises_the_property(self):
        assert make_triple("e1", "firstName", "Jane").prop == "first_name"

    def test_a_raw_unnormalised_property_is_refused(self):
        """Constructing directly must not bypass normalisation."""
        with pytest.raises(ValidationError, match="not normalised"):
            Triple(entity="e1", prop="firstName", value="Jane")

    def test_triples_are_hashable_and_deduplicate(self):
        one = make_triple("e1", "name", "Jane")
        same = make_triple("e1", "name", "Jane")
        assert len({one, same}) == 1

    def test_a_quantity_keeps_its_unit(self):
        triple = make_triple("e1", "length", Quantity(magnitude=1.0, unit="m"))
        assert isinstance(triple.value, Quantity)
        assert triple.value.unit == "m"
        assert str(triple.value) == "1.0 m"


class TestMakeTriples:
    def test_a_list_becomes_one_triple_per_element(self):
        triples = make_triples("e1", {"email": ["a@example.org", "b@example.org"]})
        assert len(triples) == 2

    def test_order_within_a_list_does_not_matter(self):
        forward = make_triples("e1", {"email": ["a@x.org", "b@x.org"]})
        backward = make_triples("e1", {"email": ["b@x.org", "a@x.org"]})
        assert forward == backward

    def test_scalars_and_lists_are_indistinguishable_afterwards(self):
        assert make_triples("e1", {"email": "a@x.org"}) == make_triples("e1", {"email": ["a@x.org"]})


class TestTripleSet:
    def build(
        self,
        triples: frozenset[Triple] | None = None,
        classes: dict[str, str] | None = None,
    ) -> TripleSet:
        return TripleSet(
            triples=(make_triples("e1", {"name": "Jane", "age": 31}) if triples is None else triples),
            classes={"e1": "schemaorg.Person"} if classes is None else classes,
            provenance={},
        )

    def test_entities_are_derived_from_the_triples(self):
        assert self.build().entities() == frozenset({"e1"})

    def test_an_entity_known_only_by_class_still_counts(self):
        """A class pinned at decode time and then filled with nothing
        produces no triple at all. Without this, align.CLASS_AGREEMENT could
        never fire for it: the entity would not even be a candidate."""
        empty = TripleSet(triples=frozenset(), classes={"e1": "schemaorg.Person"}, provenance={})
        assert empty.entities() == frozenset({"e1"})

    def test_an_entity_known_only_by_provenance_still_counts(self):
        spanned = TripleSet(triples=frozenset(), classes={}, provenance={"e1": "line 3"})
        assert spanned.entities() == frozenset({"e1"})

    def test_an_arm_may_assign_no_class(self):
        """A0 emits no class. That is the condition, not a malformed result."""
        assert self.build(classes={}).classes == {}

    def test_for_entity_selects_only_that_entity(self):
        triples = make_triples("e1", {"name": "Jane"}) | make_triples("e2", {"name": "John"})
        subject = self.build(triples=triples)
        assert len(subject.for_entity("e1")) == 1

    def test_parse_errors_default_to_none(self):
        assert self.build().parse_errors == 0


class TestMerge:
    def test_merging_unions_triples_and_sums_parse_errors(self):
        first = TripleSet(
            triples=make_triples("e1", {"name": "Jane"}),
            classes={"e1": "schemaorg.Person"},
            provenance={},
            parse_errors=1,
        )
        second = TripleSet(
            triples=make_triples("e2", {"name": "John"}),
            classes={"e2": "schemaorg.Person"},
            provenance={},
            parse_errors=2,
        )
        merged = merge([first, second])
        assert merged.entities() == frozenset({"e1", "e2"})
        assert merged.parse_errors == 3


def test_every_dimension_is_reported_separately():
    assert [d.value for d in Dimension] == [
        "entity",
        "class",
        "class_near",
        "property",
        "property_near",
        "value",
        "value_near",
        "unit",
        "unit_physical",
        "shortlist",
        "duplicate",
        "mention",
        "fillable",
        "fillable_near",
        "patch",
        "provenance",
        "grounded",
    ]
