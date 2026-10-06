"""Round-trip task generation from the schema.org corpus."""

import hashlib
import json
import random

import pytest

from oold_llm_bench.corpus import schemaorg
from oold_llm_bench.corpus.quantities import readable_kind
from oold_llm_bench.corpus.schemaorg import (
    CATALOGUE_SLOTS,
    DECLINED,
    FRAME_COUNT,
    IMPLIED_FRAME_COUNT,
    Form,
    Kind,
    Labelling,
    Link,
    Notation,
    SchemaClass,
    Slot,
    answer_schema,
    canonical_date,
    canonical_duration,
    canonical_time,
    canonical_value,
    class_identifier,
    describable_classes,
    designating_slots,
    draw,
    draw_linked,
    form_of,
    generate_pair,
    generate_task,
    identifying_sets,
    implied_classes,
    legible_slots,
    linked_classes,
    load_classes,
    opaque_name,
    property_identifier,
    rename_map,
    render,
    render_class,
    resolvable_links,
    seed_values,
    spell,
    written_value,
)
from oold_llm_bench.extract.json_answer import extract_json
from oold_llm_bench.grading import Dimension, Reference, TripleSet, make_triples
from oold_llm_bench.grading.compare import same_value
from oold_llm_bench.grading.score import score_task
from oold_llm_bench.tasks import Difficulty, Split
from oold_llm_bench.tasks.models import CorpusRef, ExpectedInstance, Source, TaskRecord, Variant

BOOK = SchemaClass(
    name="Book",
    parents=("CreativeWork",),
    label="Book",
    description="A book.",
    links=(Link(name="author", ranges=("Person",)),),
    slots=(
        Slot(name="isbn", kind=Kind.TEXT),
        Slot(name="bookEdition", kind=Kind.TEXT),
        Slot(name="numberOfPages", kind=Kind.INTEGER),
        Slot(name="abridged", kind=Kind.BOOLEAN),
        Slot(name="bookFormat", kind=Kind.ENUM, choices=("Hardcover", "Paperback", "EBook")),
        Slot(name="name", kind=Kind.TEXT, inherited=True),
        Slot(name="datePublished", kind=Kind.DATE, inherited=True),
    ),
)

PERSON = SchemaClass(
    name="Person",
    parents=("Thing",),
    label="Person",
    description="A person.",
    links=(Link(name="worksFor", ranges=("Organization",)),),
    slots=(
        Slot(name="familyName", kind=Kind.TEXT),
        Slot(name="givenName", kind=Kind.TEXT),
        Slot(name="birthDate", kind=Kind.DATE),
        Slot(name="email", kind=Kind.EMAIL),
        Slot(name="height", kind=Kind.NUMBER),
        Slot(name="name", kind=Kind.TEXT, inherited=True),
    ),
)

EVENT = SchemaClass(
    name="Event",
    parents=("Thing",),
    label="Event",
    description="An event.",
    links=(Link(name="superEvent", ranges=("Event",)),),
    slots=(
        Slot(name="startDate", kind=Kind.DATE),
        Slot(name="maximumAttendeeCapacity", kind=Kind.INTEGER),
        Slot(name="isAccessibleForFree", kind=Kind.BOOLEAN),
        Slot(name="eventStatus", kind=Kind.ENUM, choices=("EventScheduled", "EventPostponed")),
        Slot(name="duration", kind=Kind.DURATION),
        Slot(name="name", kind=Kind.TEXT, inherited=True),
    ),
)

PROPERTY_VALUE = SchemaClass(
    name="PropertyValue",
    parents=("Thing",),
    label="Property value",
    description="A property-value pair.",
    slots=(
        Slot(name="value", kind=Kind.NUMBER),
        Slot(name="unitCode", kind=Kind.TEXT),
        Slot(name="propertyID", kind=Kind.TEXT),
        Slot(name="name", kind=Kind.TEXT, inherited=True),
    ),
)
"""The real class, copied because the collision is in these two slot names."""

CLASSES = [BOOK, PERSON, EVENT]

SERVICE_PERIOD = SchemaClass(
    name="ServicePeriod",
    parents=("Thing",),
    label="Service period",
    description="A service period.",
    slots=(
        Slot(name="businessDays", kind=Kind.ENUM, choices=("Monday", "Tuesday", "Wednesday")),
        Slot(name="cutoffTime", kind=Kind.TIME),
        Slot(name="duration", kind=Kind.DURATION),
        Slot(name="name", kind=Kind.TEXT, inherited=True),
    ),
)

DELIVERY_EVENT = SchemaClass(
    name="DeliveryEvent",
    parents=("Thing",),
    label="Delivery event",
    description="A delivery event.",
    slots=(
        Slot(name="accessCode", kind=Kind.TEXT),
        Slot(name="availableFrom", kind=Kind.DATETIME),
        Slot(name="hasDeliveryMethod", kind=Kind.ENUM, choices=("ParcelService", "OnSitePickup")),
        Slot(name="name", kind=Kind.TEXT, inherited=True),
    ),
)

WRITTEN_CLASSES = [*CLASSES, SERVICE_PERIOD, DELIVERY_EVENT]
"""The three above plus the two real classes that carry a clock.

``ServicePeriod`` and ``DeliveryEvent`` are copied out of the corpus for the
same reason ``PropertyValue`` is: schema.org declares 3 time slots and 22
datetime slots over the describable classes, none of them on a Book, a Person
or an Event, and a written grid without either would leave the two forms that
carry a clock untested. The three above stay as they are, because the pinned
grid is what every measured result rests on.
"""


def perfect_answer(task) -> TripleSet:
    """What a model that read the document correctly would produce."""
    triples = frozenset()
    classes = {}
    for instance in task.expected:
        triples |= make_triples(instance.key, instance.fields)
        classes[instance.key] = instance.class_path
    return TripleSet(triples=triples, classes=classes, provenance={})


def write_corpus(directory) -> None:
    """A miniature corpus with every shape the real one uses.

    Written to disk rather than mocked, because what is being tested is that
    the loader reads the projection schema.org was translated into, and a
    mock would only restate the loader's own assumptions.
    """
    files = {
        "Text": {"title": "Text", "type": "string"},
        "Number": {"title": "Number", "type": "number"},
        "Integer": {"title": "Integer", "$ref": "Number.schema.json", "type": "integer"},
        "Boolean": {"title": "Boolean", "type": "boolean"},
        "Date": {"title": "Date", "type": "string", "format": "date"},
        "Email": {"title": "Email", "$ref": "Text.schema.json", "format": "email"},
        "Thing": {
            "title": "Thing",
            "type": "object",
            "properties": {
                "type": {"type": "array", "items": {"type": "string"}},
                "identifier": {"type": "array", "items": {"$ref": "Text.schema.json"}},
                "name": {"title": "Name", "type": "array", "items": {"$ref": "Text.schema.json"}},
            },
        },
        "Organization": {
            "title": "Organization",
            "description": "An organization.",
            "allOf": [{"$ref": "Thing.schema.json"}],
            "properties": {
                "email": {"type": "array", "items": {"$ref": "Email.schema.json"}},
                "duns": {"type": "array", "items": {"$ref": "Text.schema.json"}},
            },
        },
        "Place": {
            "title": "Place",
            "allOf": [{"$ref": "Thing.schema.json"}],
            "properties": {
                "branchCode": {"type": "array", "items": {"$ref": "Text.schema.json"}},
                "maximumAttendeeCapacity": {"type": "array", "items": {"$ref": "Integer.schema.json"}},
            },
        },
        "LocalBusiness": {
            "title": "LocalBusiness",
            "allOf": [{"$ref": "Organization.schema.json"}, {"$ref": "Place.schema.json"}],
            "properties": {"priceRange": {"type": "array", "items": {"$ref": "Text.schema.json"}}},
        },
        "BookFormatType": {
            "title": "BookFormatType",
            "allOf": [{"$ref": "Thing.schema.json"}],
            "$defs": {
                "member": {
                    "type": "string",
                    "format": "iri-reference",
                    "enum": ["schema:Hardcover", "schema:Paperback"],
                    "x-oold-ui-enum-titles": ["Hardcover", "Paperback"],
                }
            },
        },
        "Book": {
            "title": "Book",
            "description": "A book.",
            "allOf": [{"$ref": "Thing.schema.json"}],
            "properties": {
                "isbn": {"type": "array", "items": {"$ref": "Text.schema.json"}},
                "numberOfPages": {"type": "array", "items": {"$ref": "Integer.schema.json"}},
                "abridged": {"type": "array", "items": {"$ref": "Boolean.schema.json"}},
                "bookFormat": {
                    "type": "array",
                    "items": {"$ref": "BookFormatType.schema.json#/$defs/member"},
                },
                "author": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "format": "iri-reference",
                        "x-oold-range": "Organization.schema.json",
                    },
                },
            },
        },
        "Ticket": {
            "title": "Ticket",
            "allOf": [{"$ref": "Thing.schema.json"}],
            "properties": {
                "issuedBy": {
                    "type": "array",
                    "items": {"type": "string", "format": "iri-reference", "x-oold-range": "Organization.schema.json"},
                },
                "issuedBy_text": {"type": "array", "items": {"$ref": "Text.schema.json"}},
                "totalPrice": {
                    "type": "array",
                    "items": {
                        "anyOf": [{"$ref": "Number.schema.json"}, {"$ref": "Organization.schema.json"}],
                    },
                },
                "status": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "format": "iri-reference",
                        "enum": ["schema:Valid", "schema:Void"],
                        "x-oold-ui-enum-titles": ["Valid", "Void"],
                    },
                },
            },
        },
    }
    for name, schema in files.items():
        (directory / f"{name}.schema.json").write_text(json.dumps(schema), encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    write_corpus(tmp_path)
    return {cls.name: cls for cls in load_classes(tmp_path)}


class TestLoadClasses:
    def test_a_datatype_leaf_is_not_a_class(self, corpus):
        """Text and Integer are what properties hold, not what they describe."""
        assert "Text" not in corpus
        assert "Integer" not in corpus
        assert "Book" in corpus

    def test_the_parent_comes_from_all_of(self, corpus):
        assert corpus["Organization"].parents == ("Thing",)
        assert corpus["Organization"].parent == "Thing"

    def test_both_parents_of_a_multiply_inherited_class_are_kept(self, corpus):
        """LocalBusiness is an Organization and a Place, and needs both."""
        business = corpus["LocalBusiness"]
        assert business.parents == ("Organization", "Place")
        names = {slot.name for slot in business.slots}
        assert {"email", "duns", "branchCode", "maximumAttendeeCapacity"} <= names

    def test_an_inherited_slot_is_marked_as_one(self, corpus):
        book = corpus["Book"]
        assert {slot.name for slot in book.own_slots} == {"isbn", "numberOfPages", "abridged", "bookFormat"}
        assert next(slot for slot in book.slots if slot.name == "name").inherited is True

    def test_a_refined_datatype_resolves_to_what_it_refines(self, corpus):
        """Email is Text with a format, and reading only its type finds nothing."""
        email = next(slot for slot in corpus["Organization"].slots if slot.name == "email")
        assert email.kind is Kind.EMAIL

    def test_an_integer_is_not_a_plain_number(self, corpus):
        pages = next(slot for slot in corpus["Book"].slots if slot.name == "numberOfPages")
        assert pages.kind is Kind.INTEGER

    def test_an_enumerated_slot_carries_its_members(self, corpus):
        book_format = next(slot for slot in corpus["Book"].slots if slot.name == "bookFormat")
        assert book_format.kind is Kind.ENUM
        assert book_format.choices == ("Hardcover", "Paperback")

    def test_an_inline_enumeration_is_read_too(self, corpus):
        """A range spanning several enumerations inlines the union instead."""
        status = next(slot for slot in corpus["Ticket"].slots if slot.name == "status")
        assert status.kind is Kind.ENUM
        assert status.choices == ("Valid", "Void")

    def test_a_node_reference_is_not_a_slot(self, corpus):
        """An author is another entity, and no prose writes down its IRI."""
        assert "author" not in {slot.name for slot in corpus["Book"].slots}
        assert "issuedBy" not in {slot.name for slot in corpus["Ticket"].slots}

    def test_a_node_reference_is_a_link(self, corpus):
        """The same property, read as the edge it is rather than dropped."""
        author = next(link for link in corpus["Book"].links if link.name == "author")
        assert author.ranges == ("Organization",)
        assert {link.name for link in corpus["Ticket"].own_links} == {"issuedBy"}

    def test_a_property_that_resolves_to_a_literal_is_never_a_link(self, corpus):
        """``totalPrice`` is a Number or an Organization, and the number wins."""
        assert "totalPrice" not in {link.name for link in corpus["Ticket"].links}
        assert next(slot for slot in corpus["Ticket"].slots if slot.name == "totalPrice").kind is Kind.NUMBER

    def test_an_enumeration_member_stays_a_closed_list(self, corpus):
        """schema.org models it as a node, and reading it as one would take
        the enumerations out of the slots that make them worth asking about."""
        assert "bookFormat" not in {link.name for link in corpus["Book"].links}
        assert "status" not in {link.name for link in corpus["Ticket"].links}

    def test_the_projection_companion_key_is_not_a_link_either(self, corpus):
        assert "issuedBy_text" not in {link.name for link in corpus["Ticket"].links}

    def test_the_projection_companion_key_is_left_out(self, corpus):
        """``issuedBy_text`` is an artefact of the projection, not vocabulary."""
        assert "issuedBy_text" not in {slot.name for slot in corpus["Ticket"].slots}

    def test_a_mixed_range_keeps_its_literal_branch(self, corpus):
        total = next(slot for slot in corpus["Ticket"].slots if slot.name == "totalPrice")
        assert total.kind is Kind.NUMBER

    def test_serialisation_properties_are_never_slots(self, corpus):
        names = {slot.name for cls in corpus.values() for slot in cls.slots}
        assert "type" not in names
        assert "identifier" not in names

    def test_a_class_carries_its_label_and_description(self, corpus):
        assert corpus["Book"].label == "Book"
        assert corpus["Book"].description == "A book."


class TestDescribableClasses:
    def test_a_class_with_too_few_declared_slots_is_left_out(self):
        """It would be described by Thing's properties and by nothing else."""
        thin = SchemaClass(name="Thin", slots=(Slot(name="name", kind=Kind.TEXT, inherited=True),))
        assert describable_classes([*CLASSES, thin], 3) == CLASSES


class TestSeedValues:
    def test_declared_slots_are_taken_before_inherited_ones(self):
        chosen = {slot.name for slot, _ in seed_values(BOOK, random.Random(3), 5)}  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        assert chosen == {slot.name for slot in BOOK.own_slots}

    def test_a_seed_reproduces_the_values(self):
        first = seed_values(PERSON, random.Random(7), 4)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        second = seed_values(PERSON, random.Random(7), 4)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        assert first == second

    def test_every_value_matches_the_kind_of_its_slot(self):
        for seed in range(20):
            for slot, value in seed_values(EVENT, random.Random(seed), 5):  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
                if slot.kind is Kind.INTEGER:
                    assert isinstance(value, int) and not isinstance(value, bool)
                elif slot.kind is Kind.BOOLEAN:
                    assert isinstance(value, bool)
                elif slot.kind is Kind.ENUM:
                    assert value in slot.choices
                else:
                    assert isinstance(value, str)

    def test_an_email_value_looks_like_one(self):
        values = seed_values(PERSON, random.Random(1), 5)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        email = next(value for slot, value in values if slot.kind is Kind.EMAIL)
        assert "@" in str(email)

    def test_asking_for_more_slots_than_a_class_has_is_refused(self):
        with pytest.raises(ValueError, match="asked for"):
            seed_values(BOOK, random.Random(1), 99)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one


class TestRender:
    def test_the_corpus_writes_from_more_frames_than_the_quantity_one(self):
        """Sixteen frames over thousands of documents are learnable."""
        assert FRAME_COUNT > 40

    def test_every_value_reaches_the_prose(self):
        entity = draw(CLASSES, random.Random(4), 1, 4)[0]  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        text = render(entity, Difficulty.EASY, random.Random(4))  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        for _, value in entity.values:
            assert spell(value) in text

    def test_a_boolean_is_written_as_a_document_writes_it(self):
        assert spell(True) == "true"
        assert spell(False) == "false"

    def test_a_class_name_reads_as_prose(self):
        entity = draw([BOOK], random.Random(1), 1, 3)[0]  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        text = render(entity, Difficulty.EASY, random.Random(1))  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        assert "book" in text
        assert "Book" not in text

    def test_the_catalogue_entry_of_a_renamed_class_carries_no_english(self):
        """Its description would hand back the vocabulary the rename removed."""
        entry = render_class(BOOK, Variant.RENAMED)
        assert "A book." not in entry
        assert "isbn" not in entry
        assert "Hardcover" in entry


class TestVariants:
    """The pretraining-exposure contrast, which is why documents come in pairs."""

    def test_a_renamed_document_contains_no_schema_org_name(self):
        for seed in range(20):
            entities = draw(CLASSES, random.Random(seed), 1, 4)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
            _, renamed = generate_pair(CLASSES, task_id="t", seed=seed)
            for entity in entities:
                assert readable_kind(entity.cls.name) not in renamed.document
                for slot, _ in entity.values:
                    assert readable_kind(slot.name) not in renamed.document

    def test_both_variants_carry_the_same_values(self):
        native, renamed = generate_pair(CLASSES, task_id="t", seed=5)
        assert sorted(map(str, native.expected[0].fields.values())) == sorted(
            map(str, renamed.expected[0].fields.values())
        )

    def test_both_variants_use_the_same_sentences(self):
        """A pair that drifted would look like a vocabulary effect by accident."""
        seed = 11
        entity = draw(CLASSES, random.Random(seed), 1, 4)[0]  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        native, renamed = generate_pair(CLASSES, task_id="t", seed=seed)
        back = renamed.document.replace(
            class_identifier(entity.cls.name, Variant.RENAMED), readable_kind(entity.cls.name)
        )
        for slot, _ in entity.values:
            back = back.replace(property_identifier(slot.name, Variant.RENAMED), readable_kind(slot.name))
        assert back == native.document

    def test_the_variant_is_recorded_on_the_task(self):
        native, renamed = generate_pair(CLASSES, task_id="t", seed=2)
        assert native.variant is Variant.NATIVE
        assert renamed.variant is Variant.RENAMED
        assert native.id.endswith("-native")
        assert renamed.id.endswith("-renamed")

    def test_an_opaque_identifier_does_not_move_between_runs(self):
        """A per-task renaming would make a shared catalogue impossible."""
        assert opaque_name("Book", "Q") == opaque_name("Book", "Q")
        assert opaque_name("Book", "Q") != opaque_name("Person", "Q")

    def test_a_class_and_a_property_of_the_same_name_stay_apart(self):
        assert class_identifier("name", Variant.RENAMED) != property_identifier("name", Variant.RENAMED)

    def test_the_renamed_class_is_what_the_answer_is_graded_against(self):
        _, renamed = generate_pair(CLASSES, task_id="t", seed=3)
        assert renamed.expected[0].class_path.startswith("Q")

    def test_a_collision_is_refused_rather_than_merged(self, monkeypatch):
        """Two names folded onto one identifier would look like a class error.

        Forced, because forty bits over a corpus of this size will not collide
        on its own, which is exactly why the guard has to be exercised rather
        than trusted.
        """
        monkeypatch.setattr(schemaorg, "opaque_name", lambda name, prefix, salt="": f"{prefix}0")
        with pytest.raises(ValueError, match="claimed by both"):
            rename_map(["Book", "Person"], "Q")

    def test_renaming_the_whole_corpus_produces_no_collision(self):
        names = [cls.name for cls in CLASSES]
        assert len(set(rename_map(names, "Q").values())) == len(names)


class TestGenerateTask:
    def test_a_seed_reproduces_the_document_exactly(self):
        first = generate_task(CLASSES, task_id="t1", seed=9)
        second = generate_task(CLASSES, task_id="t1", seed=9)
        assert first.document == second.document
        assert first.corpus.content_hash == second.corpus.content_hash

    def test_a_different_seed_gives_a_different_document(self):
        first = generate_task(CLASSES, task_id="t1", seed=9)
        second = generate_task(CLASSES, task_id="t1", seed=10)
        assert first.document != second.document

    def test_the_content_hash_covers_the_document(self):
        task = generate_task(CLASSES, task_id="t1", seed=2)
        assert task.corpus.content_hash == hashlib.sha256(task.document.encode("utf-8")).hexdigest()

    @pytest.mark.parametrize("difficulty", [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD])
    def test_every_difficulty_still_contains_its_answer(self, difficulty):
        task = generate_task(CLASSES, task_id="t1", seed=4, difficulty=difficulty, n_entities=2)
        assert task.difficulty is difficulty
        for instance in task.expected:
            for value in instance.fields.values():
                assert spell(value) in task.document

    def test_harder_prose_is_longer_than_plain_prose(self):
        plain = generate_task(CLASSES, task_id="t1", seed=6, difficulty=Difficulty.EASY)
        separated = generate_task(CLASSES, task_id="t1", seed=6, difficulty=Difficulty.MEDIUM)
        embedded = generate_task(CLASSES, task_id="t1", seed=6, difficulty=Difficulty.HARD)
        assert len(plain.document) < len(separated.document) < len(embedded.document)

    def test_the_split_label_is_carried(self):
        task = generate_task(CLASSES, task_id="t1", seed=1, split=Split.TEST)
        assert task.split is Split.TEST

    def test_entity_keys_are_unique(self):
        task = generate_task(CLASSES, task_id="t1", seed=8, n_entities=3)
        keys = [instance.key for instance in task.expected]
        assert len(set(keys)) == len(keys)

    def test_the_class_each_entity_came_from_is_recorded(self):
        """A renamed task's class path is opaque, so a report needs this."""
        task = generate_task(CLASSES, task_id="t1", seed=1)
        assert task.notes is not None
        assert task.notes.startswith("variant=native,classes=")

    def test_asking_for_more_entities_than_exist_is_refused(self):
        with pytest.raises(ValueError, match="corpus offers"):
            generate_task(CLASSES, task_id="t1", seed=1, n_entities=99)

    def test_a_corpus_with_no_usable_class_is_refused(self):
        with pytest.raises(ValueError, match=r"no schema\.org class"):
            generate_task([SchemaClass(name="Bare", slots=())], task_id="t1", seed=1)

    def test_a_catalogue_that_omits_the_answer_is_refused(self):
        with pytest.raises(ValueError, match="catalogue omits"):
            generate_task(CLASSES, task_id="t1", seed=1, catalogue=("Nothing",))

    def test_a_described_catalogue_reaches_the_task(self):
        task = generate_task(
            CLASSES,
            task_id="t1",
            seed=1,
            catalogue=tuple(cls.name for cls in CLASSES),
            describe_catalogue=True,
        )
        assert task.catalogue_text is not None
        assert "A book." in task.catalogue_text["Book"]


class TestMultipleEntities:
    """Several records are a list, and each one names the class it belongs to.

    With more than one subject on the page a value that is not attached to one
    is unattributable, which is the failure the quantity corpus sidesteps by
    having no subjects at all.
    """

    def test_several_entities_render_as_a_list(self):
        task = generate_task(CLASSES, task_id="t", seed=5, n_entities=3)
        assert task.document.startswith("Records:")
        assert task.document.count("\n- ") == 3

    def test_one_entity_stays_a_paragraph(self):
        task = generate_task(CLASSES, task_id="t", seed=5)
        assert "Records:" not in task.document

    def test_each_entity_keeps_its_own_values(self):
        task = generate_task(CLASSES, task_id="t", seed=5, n_entities=3)
        for instance in task.expected:
            for value in instance.fields.values():
                assert spell(value) in task.document


class TestTheSeamCloses:
    """Corpus to task to score, with no model involved.

    A generator that emits an answer the grader cannot recognise would look
    fine on both sides and score every arm at zero.
    """

    @pytest.mark.parametrize("difficulty", [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD])
    @pytest.mark.parametrize("variant", [Variant.NATIVE, Variant.RENAMED])
    def test_a_perfect_answer_scores_one(self, difficulty, variant):
        task = generate_task(CLASSES, task_id="t1", seed=11, difficulty=difficulty, n_entities=3, variant=variant)
        result = score_task(task, perfect_answer(task))
        assert result.primary == pytest.approx(1.0)
        assert result.dimensions[Dimension.CLASS].f1 == pytest.approx(1.0)
        assert result.dimensions[Dimension.PROPERTY].f1 == pytest.approx(1.0)

    def test_an_answer_from_a_different_document_scores_zero(self):
        task = generate_task(CLASSES, task_id="t1", seed=12, n_entities=2)
        other = generate_task(CLASSES, task_id="t2", seed=9999, n_entities=2)
        assert score_task(task, perfect_answer(other)).primary == 0.0

    def test_an_empty_answer_scores_zero(self):
        task = generate_task(CLASSES, task_id="t1", seed=13, n_entities=2)
        empty = TripleSet(triples=frozenset(), classes={}, provenance={})
        assert score_task(task, empty).primary == 0.0

    def test_the_native_answer_does_not_pass_for_the_renamed_task(self):
        """The two vocabularies are different answers to the same facts."""
        native, renamed = generate_pair(CLASSES, task_id="t1", seed=14)
        assert score_task(renamed, perfect_answer(native)).dimensions[Dimension.CLASS].f1 == 0.0

    @pytest.mark.parametrize("difficulty", [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD])
    def test_a_perfect_answer_scores_one_through_the_extractor(self, difficulty):
        """The same check, but down the path a model's answer takes.

        The test above hands the grader triples built straight from the task,
        so the extractor never runs and an extractor that mangles a shape the
        generator emits stays invisible. That gap hid a real ceiling: see
        :class:`TestValueAndUnitOnOneEntity`.
        """
        task = generate_task(CLASSES, task_id="t1", seed=11, difficulty=difficulty, n_entities=3)
        result = score_task(task, extract_json(json.loads(_as_json_answer(task))))
        assert result.primary == pytest.approx(1.0)


def _as_json_answer(task) -> str:
    """The task's own answer, written the way a perfect model would write it."""
    return json.dumps(
        [{"type": instance.class_path, **instance.fields} for instance in task.expected],
        default=str,
    )


class TestValueAndUnitOnOneEntity:
    """A class whose own properties are named ``value`` and ``unitCode``.

    The extractor folds any value and unit found together into one quantity
    triple, which is what an unconstrained arm's ``{"amount": 1.75, "units":
    "m"}`` has to become. schema.org has classes that declare those two as
    ordinary properties, and for those the folding is wrong: three expected
    triples arrive as one and the task cannot score above 0.67.

    It is left in rather than fixed, and the reason is that the extractor is
    deliberately blind to the schema. One extractor reads every arm, so it
    cannot consult the class to decide whether a value and a unit belong
    together. Excluding these classes from the pool is the fix, and it has to
    wait for a grid that runs native and renamed together, because changing
    the pool under one half of that pair makes the contrast unreadable.

    Measured on the 120 tasks of ``g3b``: two are affected, both
    ``PropertyValue``, so the ceiling is 0.9945 rather than 1.00. It applies
    to every arm and model equally and the design is paired, so it shifts no
    comparison.
    """

    def test_value_and_unit_collapse_into_one_triple(self):
        task = generate_task([PROPERTY_VALUE], task_id="t1", seed=3, difficulty=Difficulty.HARD, n_entities=1)
        result = score_task(task, extract_json(json.loads(_as_json_answer(task))))
        assert result.primary < 1.0

    def test_the_triples_the_task_declares_are_still_three(self):
        """The loss is the extractor's, not the generator's."""
        task = generate_task([PROPERTY_VALUE], task_id="t1", seed=3, difficulty=Difficulty.HARD, n_entities=1)
        assert score_task(task, perfect_answer(task)).primary == pytest.approx(1.0)


class TestTheCatalogueFollowsTheVariant:
    """A caller builds one catalogue and gets both variants back."""

    def pool(self):
        return [
            SchemaClass(
                name=name,
                slots=(
                    Slot(name=f"{name.lower()}A", kind=Kind.TEXT),
                    Slot(name=f"{name.lower()}B", kind=Kind.NUMBER),
                    Slot(name=f"{name.lower()}C", kind=Kind.TEXT),
                    Slot(name=f"{name.lower()}D", kind=Kind.NUMBER),
                ),
            )
            for name in ("Alpha", "Beta", "Gamma")
        ]

    def catalogue(self, pool):
        return tuple(class_identifier(cls.name, Variant.NATIVE) for cls in pool)

    def test_the_renamed_task_is_offered_renamed_classes(self):
        """Otherwise its answer is opaque and its catalogue is not."""
        pool = self.pool()
        _, renamed = generate_pair(pool, task_id="t", seed=1, n_slots=3, catalogue=self.catalogue(pool))
        assert all(name.startswith("Q") for name in renamed.catalogue or [])

    def test_each_variant_holds_its_own_answer(self):
        pool = self.pool()
        for seed in range(12):
            for task in generate_pair(pool, task_id=f"t{seed}", seed=seed, n_slots=3, catalogue=self.catalogue(pool)):
                assert task.expected[0].class_path in (task.catalogue or [])

    def test_both_catalogues_are_the_same_size(self):
        """The pair differs in the names and in nothing else."""
        pool = self.pool()
        native, renamed = generate_pair(pool, task_id="t", seed=1, n_slots=3, catalogue=self.catalogue(pool))
        assert len(native.catalogue or []) == len(renamed.catalogue or []) == 3

    def test_a_renamed_entry_keeps_no_english(self):
        """Describing it with the label would hand back what renaming removed."""
        pool = self.pool()
        _, renamed = generate_pair(
            pool, task_id="t", seed=1, n_slots=3, catalogue=self.catalogue(pool), describe_catalogue=True
        )
        text = " ".join((renamed.catalogue_text or {}).values())
        assert "Alpha" not in text
        assert "alphaA" not in text


TWIN = SchemaClass(
    name="Twin",
    slots=(
        Slot(name="pages", kind=Kind.INTEGER),
        Slot(name="short", kind=Kind.BOOLEAN),
        Slot(name="format", kind=Kind.ENUM, choices=("Hardcover", "Paperback")),
    ),
)
"""A class Book cannot be told apart from by the forms of its values."""


class TestTheLabelledCorpusDoesNotMove:
    """Every measured result rests on these exact sentences.

    A generator that drifted would leave the runs behind it incomparable
    without any other test failing, so the documents are pinned rather than
    described.
    """

    @pytest.mark.parametrize(
        ("difficulty", "document"),
        [
            (
                Difficulty.EASY,
                "The sheet covers a single entry of type event. The maximum attendee capacity was noted as 8852. "
                "The is accessible for free is given as false. Under event status the sheet has EventScheduled. "
                "It carries the duration PT1H16M.",
            ),
            (
                Difficulty.MEDIUM,
                "The sheet covers a single entry of type event. The value 8852 was entered. Its label, further "
                "down, is maximum attendee capacity. The is accessible for free is given as false. There is "
                "EventScheduled on the sheet. The event status field is where it goes. It carries the duration "
                "PT1H16M.",
            ),
            (
                Difficulty.HARD,
                "The sheet covers a single entry of type event. The value 8852 was entered. Its label, further "
                "down, is maximum attendee capacity. The is accessible for free is given as false. The entry was "
                "copied into the ledger unchanged. Someone noted EventScheduled. Which field that was is given "
                "below as event status. It was stamped PT1H16M for duration. No correction was made afterwards.",
            ),
        ],
    )
    def test_a_fixed_seed_writes_what_it_wrote(self, difficulty, document):
        assert generate_task(CLASSES, task_id="t", seed=17, difficulty=difficulty).document == document

    def test_the_whole_grid_of_documents_hashes_as_it_did(self):
        """720 documents, both variants, so drift anywhere is caught."""
        digest = hashlib.sha256()
        for seed in range(40):
            for difficulty in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
                for n_entities in (1, 2, 3):
                    pair = generate_pair(
                        CLASSES, task_id=f"t{seed}", seed=seed, difficulty=difficulty, n_entities=n_entities
                    )
                    for task in pair:
                        digest.update(task.document.encode("utf-8"))
        assert digest.hexdigest() == "cc4b2a506336db3570957fa0b319346e60af3db58da4c175bcdee6fa09c84e46"


class TestLegibleSlots:
    """Which slots a value can point at when nothing names it."""

    def test_two_slots_of_one_form_are_both_dropped(self):
        """Which of the pair a text value belongs to would be a coin toss."""
        assert {slot.name for slot in legible_slots(BOOK)} == {"numberOfPages", "abridged", "bookFormat"}

    def test_an_integer_and_a_number_are_one_form(self):
        """An integer satisfies a number slot, so it rules neither out."""
        assert form_of(Slot(name="pages", kind=Kind.INTEGER)) is Form.NUMERIC
        assert form_of(Slot(name="height", kind=Kind.NUMBER)) is Form.NUMERIC
        mixed = SchemaClass(
            name="Mixed",
            slots=(
                Slot(name="count", kind=Kind.INTEGER),
                Slot(name="size", kind=Kind.NUMBER),
                Slot(name="when", kind=Kind.DATE),
            ),
        )
        assert {slot.name for slot in legible_slots(mixed)} == {"when"}

    def test_enumerations_that_never_meet_both_stay(self):
        """A member of a closed list points at the one list that holds it."""
        assert len(legible_slots(EVENT)) == 5

    def test_enumerations_that_share_a_member_both_go(self):
        overlapping = SchemaClass(
            name="Overlapping",
            slots=(
                Slot(name="first", kind=Kind.ENUM, choices=("Red", "Blue")),
                Slot(name="second", kind=Kind.ENUM, choices=("Blue", "Green")),
                Slot(name="third", kind=Kind.BOOLEAN),
            ),
        )
        assert {slot.name for slot in legible_slots(overlapping)} == {"third"}

    def test_a_text_slot_does_not_contest_an_enumeration(self):
        """Otherwise no closed list is legible and the mode has no signal left."""
        assert "bookFormat" in {slot.name for slot in legible_slots(BOOK)}

    def test_an_inherited_slot_is_never_drawn_from(self):
        """A document made of Thing's properties points at no class at all."""
        assert all(not slot.inherited for cls in CLASSES for slot in legible_slots(cls))


class TestIdentifyingSets:
    """A document that names no class has to be told apart by its values."""

    def test_a_set_another_class_could_take_is_not_offered(self):
        assert identifying_sets(BOOK, [BOOK, TWIN], 3) == ()
        assert identifying_sets(BOOK, CLASSES, 3) != ()

    def test_a_class_no_set_identifies_is_not_in_the_pool(self):
        assert implied_classes([BOOK, TWIN], 3) == []

    def test_the_pool_thins_as_a_document_asks_for_more_slots(self):
        """Stated because it is the price of the mode, not a defect of it."""
        assert [cls.name for cls in implied_classes(CLASSES, 3)] == ["Book", "Person", "Event"]
        assert [cls.name for cls in implied_classes(CLASSES, 4)] == ["Event"]

    def test_a_held_out_class_is_told_apart_from_the_catalogue_and_not_the_pool(self):
        assert implied_classes([BOOK], 3) == [BOOK]
        assert implied_classes([BOOK], 3, against=[BOOK, TWIN]) == []


class TestImpliedDocuments:
    """Prose that states its values and names neither class nor property."""

    def implied(self, seed, difficulty=Difficulty.HARD, n_entities=1):
        return generate_task(
            CLASSES,
            task_id=f"t{seed}",
            seed=seed,
            difficulty=difficulty,
            n_entities=n_entities,
            n_slots=3,
            labelling=Labelling.IMPLIED,
        )

    def test_no_property_name_reaches_the_prose(self):
        for seed in range(30):
            task = self.implied(seed)
            for name in task.expected[0].fields:
                assert readable_kind(name) not in task.document

    def test_no_class_name_reaches_the_prose(self):
        for seed in range(30):
            task = self.implied(seed)
            assert readable_kind(task.expected[0].class_path) not in task.document

    def test_every_value_still_reaches_the_prose(self):
        for seed in range(30):
            task = self.implied(seed)
            for value in task.expected[0].fields.values():
                assert spell(value) in task.document

    def test_the_two_variants_are_the_same_document(self):
        """With no vocabulary on the page the pair differs in the catalogue alone."""
        native, renamed = generate_pair(CLASSES, task_id="t", seed=5, n_slots=3, labelling=Labelling.IMPLIED)
        assert native.document == renamed.document
        assert native.expected[0].class_path != renamed.expected[0].class_path

    def test_the_answer_surface_is_what_the_class_declares(self):
        """``name`` takes any text, so offering it makes every text value arguable."""
        catalogue = tuple(class_identifier(cls.name, Variant.NATIVE) for cls in CLASSES)
        task = generate_task(
            CLASSES,
            task_id="t",
            seed=5,
            n_slots=3,
            catalogue=catalogue,
            describe_catalogue=True,
            labelling=Labelling.IMPLIED,
        )
        assert set((task.branches or {})["Book"]) == {slot.name for slot in BOOK.own_slots}
        assert task.class_parents is None

    def test_a_catalogue_the_model_cannot_read_is_refused(self):
        with pytest.raises(ValueError, match="describe_catalogue"):
            generate_task(CLASSES, task_id="t", seed=1, n_slots=3, catalogue=("Book",), labelling=Labelling.IMPLIED)

    def test_a_corpus_no_document_can_be_built_from_is_refused(self):
        with pytest.raises(ValueError, match="its values alone identify"):
            generate_task([BOOK, TWIN], task_id="t", seed=1, n_slots=3, labelling=Labelling.IMPLIED)

    def test_asking_for_more_entities_than_the_thinned_pool_holds_is_refused(self):
        with pytest.raises(ValueError, match="corpus offers"):
            generate_task(CLASSES, task_id="t", seed=1, n_slots=3, n_entities=9, labelling=Labelling.IMPLIED)

    def test_a_class_with_no_identifying_set_cannot_be_filled(self):
        """Reached when a caller draws from a pool wider than the catalogue."""
        with pytest.raises(ValueError, match="tell it apart"):
            schemaorg.implied_values(BOOK, random.Random(1), 3, [BOOK, TWIN])  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one

    def test_harder_prose_is_longer_than_plain_prose(self):
        plain = self.implied(6, Difficulty.EASY)
        separated = self.implied(6, Difficulty.MEDIUM)
        embedded = self.implied(6, Difficulty.HARD)
        assert len(plain.document) < len(separated.document) < len(embedded.document)

    def test_the_corpus_writes_from_more_frames_than_the_quantity_one(self):
        assert IMPLIED_FRAME_COUNT > 40
        assert IMPLIED_FRAME_COUNT < FRAME_COUNT

    def test_several_entities_carry_no_class_line(self):
        task = self.implied(5, n_entities=3)
        assert task.document.startswith("Records:")
        assert task.document.count("\n- ") == 3
        for cls in CLASSES:
            assert readable_kind(cls.name) not in task.document

    def test_a_boolean_gets_a_frame_that_can_hold_one(self):
        """A bare "the sheet has true on it" reads as a generation artefact."""
        entity = draw(CLASSES, random.Random(5), 1, 3, Labelling.IMPLIED)[0]  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        text = render(entity, Difficulty.EASY, random.Random(5), labelling=Labelling.IMPLIED)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        assert "has true on it" not in text
        assert "has false on it" not in text


class TestTheImpliedSeamCloses:
    """Corpus to task to score, for a document that names nothing."""

    def implied(self, seed, difficulty=Difficulty.HARD, n_entities=1):
        return generate_task(
            CLASSES,
            task_id=f"t{seed}",
            seed=seed,
            difficulty=difficulty,
            n_entities=n_entities,
            n_slots=3,
            labelling=Labelling.IMPLIED,
        )

    @pytest.mark.parametrize("difficulty", [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD])
    @pytest.mark.parametrize("variant", [Variant.NATIVE, Variant.RENAMED])
    def test_a_perfect_answer_scores_one(self, difficulty, variant):
        """What the ambiguity decision buys: no slot is drawn that form cannot place."""
        pair = generate_pair(
            CLASSES, task_id="t", seed=11, difficulty=difficulty, n_entities=3, n_slots=3, labelling=Labelling.IMPLIED
        )
        task = pair[0] if variant is Variant.NATIVE else pair[1]
        result = score_task(task, perfect_answer(task))
        assert result.primary == pytest.approx(1.0)
        assert result.dimensions[Dimension.CLASS].f1 == pytest.approx(1.0)
        assert result.dimensions[Dimension.PROPERTY].f1 == pytest.approx(1.0)

    @pytest.mark.parametrize("difficulty", [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD])
    def test_a_perfect_answer_scores_one_through_the_extractor(self, difficulty):
        task = self.implied(11, difficulty, n_entities=3)
        assert score_task(task, extract_json(json.loads(_as_json_answer(task)))).primary == pytest.approx(1.0)

    def test_an_empty_answer_scores_zero(self):
        empty = TripleSet(triples=frozenset(), classes={}, provenance={})
        assert score_task(self.implied(13, n_entities=2), empty).primary == 0.0

    def test_an_answer_from_a_different_document_scores_zero(self):
        task = self.implied(12, n_entities=2)
        other = self.implied(1012, n_entities=2)
        assert score_task(task, perfect_answer(other)).primary == 0.0

    def test_the_floor_two_documents_share_is_the_value_space(self):
        """A booleans-and-enumerations pool agrees by accident now and then.

        Measured rather than engineered away, the way the labelled corpus
        states its own floor: the slots that make this happen are the closed
        lists, which are most of what this mode has left to draw from.
        """
        scores = [
            score_task(self.implied(seed), perfect_answer(self.implied(1000 + seed))).primary for seed in range(200)
        ]
        assert sum(score == 0.0 for score in scores) == 177
        assert max(scores) == pytest.approx(2 / 3)


RESPELLED = {Kind.DATE, Kind.DATETIME, Kind.TIME, Kind.DURATION, Kind.BOOLEAN, Kind.NUMBER, Kind.INTEGER}
"""The kinds the written notation gives a second form to."""


def _answered(instance, variant):
    """The slot behind each field of one expected instance, and its value.

    A renamed task carries opaque class and property identifiers, so the slot a
    field came from has to be looked up under the variant the task was written
    in and not by its schema.org name.
    """
    by_identifier = {class_identifier(cls.name, variant): cls for cls in WRITTEN_CLASSES}
    cls = by_identifier[instance.class_path]
    slots = {property_identifier(slot.name, variant): slot for slot in cls.slots}
    return [(slots[name], value) for name, value in instance.fields.items()]


def grid(classes=None, **overrides):
    """Every task the notation guard and the round trip walk.

    1,440 records: forty seeds, three difficulties, one to three entities, both
    labellings, both variants. Wide enough that drift in any frame, any pool or
    any field of the record shows up.
    """
    for seed in range(40):
        for difficulty in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
            for n_entities in (1, 2, 3):
                for labelling in (Labelling.NAMED, Labelling.IMPLIED):
                    yield from generate_pair(
                        classes if classes is not None else CLASSES,
                        task_id=f"t{seed}",
                        seed=seed,
                        difficulty=difficulty,
                        n_entities=n_entities,
                        n_slots=3,
                        labelling=labelling,
                        **overrides,
                    )


class TestTheDefaultNotationHasNotMoved:
    """The guard every measured result rests on.

    The grid above pins the documents. This pins the whole record, so a change
    to the catalogue, the branches, the answer schema or the notes is caught
    too, and none of those is visible in the prose.
    """

    def test_the_whole_grid_of_records_hashes_as_it_did(self):
        """Defaults excluded, so a field added to the record with a default
        cannot move this. The guard is about what the corpus says, and a slot
        nothing fills says nothing: re-pinning a digest for a schema addition
        is how a guard stops guarding."""
        digest = hashlib.sha256()
        for task in grid():
            digest.update(task.model_dump_json(exclude_defaults=True).encode("utf-8"))
        # Moved 2026-10-04 because TaskRecord gained catalogue_enums, which
        # every record serialises. schema.org declares no unit slot, so
        # nothing it shows a model changed; only the shape of the record did.
        # The quantity pin moved for a different and larger reason, which is
        # recorded there.
        assert digest.hexdigest() == "5920fe063146625c805031ab039d5bc9ea44815815a23efb0b21d415a3057af4"

    def test_canonical_is_what_a_caller_gets_without_asking(self):
        assert generate_task(CLASSES, task_id="t", seed=3) == generate_task(
            CLASSES, task_id="t", seed=3, notation=Notation.CANONICAL
        )

    def test_the_schema_form_still_reaches_the_page(self):
        """The fault the written notation exists to fix, stated as a test."""
        task = generate_task(CLASSES, task_id="t", seed=17)
        assert "PT1H16M" in task.document
        assert "false" in task.document


class TestWrittenValues:
    """One slot kind at a time, with the inverse asserted beside the form."""

    @pytest.mark.parametrize(
        ("kind", "value", "written"),
        [
            (Kind.DATE, "2008-11-05", "5 November 2008"),
            (Kind.DATE, "1998-01-31", "31 January 1998"),
            (Kind.DATETIME, "2000-05-13T22:55:18Z", "13 May 2000 at 10:55:18 pm UTC"),
            (Kind.TIME, "22:55:18", "10:55:18 pm"),
            (Kind.TIME, "00:00:00", "12:00:00 am"),
            (Kind.TIME, "12:00:00", "12:00:00 pm"),
            (Kind.DURATION, "PT3H18M", "3 h 18 min"),
            (Kind.DURATION, "PT45M", "45 min"),
            (Kind.DURATION, "P1M", "1 mo"),
            (Kind.BOOLEAN, True, "yes"),
            (Kind.BOOLEAN, False, "no"),
            (Kind.NUMBER, 76037.1179, "76,037.1179"),
            (Kind.INTEGER, 8852, "8,852"),
            (Kind.INTEGER, 42, "42"),
        ],
    )
    def test_a_value_is_written_as_a_document_writes_it(self, kind, value, written):
        assert written_value(kind, value) == written

    @pytest.mark.parametrize(
        ("kind", "value"),
        [
            (Kind.DATE, "2008-11-05"),
            (Kind.DATETIME, "2000-05-13T22:55:18Z"),
            (Kind.TIME, "00:00:00"),
            (Kind.TIME, "12:00:00"),
            (Kind.DURATION, "PT3H18M"),
            (Kind.BOOLEAN, True),
            (Kind.NUMBER, 76037.1179),
            (Kind.INTEGER, 8852),
        ],
    )
    def test_the_inverse_returns_the_value_and_its_type(self, kind, value):
        back = canonical_value(kind, written_value(kind, value))
        assert back == value
        assert type(back) is type(value)

    @pytest.mark.parametrize("kind", [Kind.TEXT, Kind.URL, Kind.EMAIL, Kind.ENUM])
    def test_a_kind_a_document_already_writes_is_left_alone(self, kind):
        """Rewriting these would invent a convention instead of following one."""
        assert written_value(kind, "Calder Wynn") == "Calder Wynn"
        assert canonical_value(kind, "Calder Wynn") == "Calder Wynn"

    def test_every_hour_of_the_day_survives_the_twelve_hour_clock(self):
        """Midnight and noon are the two a twelve-hour clock usually loses."""
        clocks = [f"{hour:02d}:17:04" for hour in range(24)]
        assert len({written_value(Kind.TIME, clock) for clock in clocks}) == 24
        assert [canonical_value(Kind.TIME, written_value(Kind.TIME, clock)) for clock in clocks] == clocks

    def test_a_duration_no_document_can_write_is_refused(self):
        with pytest.raises(ValueError, match="no document could write"):
            written_value(Kind.DURATION, "P")

    def test_a_written_value_needs_the_kind_of_its_slot(self):
        """A mode that fell back to the canonical form would report the fallback."""
        with pytest.raises(ValueError, match="needs the kind"):
            spell("2008-11-05", None, Notation.WRITTEN)


class TestWhyTheOtherFormsAreDeclined:
    """Each entry of :data:`DECLINED` is a collision, not a preference."""

    def test_the_table_gives_a_reason_for_every_form(self):
        assert len(DECLINED) == 7
        assert all(form and reason for form, reason in DECLINED)

    def test_a_slash_separated_date_is_two_dates(self):
        """The reason a long form is used, spelled as the ambiguity it avoids."""
        assert canonical_date("5 November 2008") != canonical_date("11 May 2008")
        assert "05/11/2008" in dict(DECLINED)

    def test_a_bare_clock_duration_is_two_durations(self):
        """``3:18`` is three hours and eighteen minutes, and three minutes and
        eighteen seconds, and it is also how this module writes a time."""
        assert canonical_duration("3 h 18 min") == "PT3H18M"
        assert canonical_duration("3 min 18 s") == "PT3M18S"
        assert canonical_time("3:18:00 am") == "03:18:00"

    def test_a_month_and_a_minute_stay_apart(self):
        """ISO 8601 spells both ``M`` and tells them apart by position."""
        assert canonical_duration("1 mo") == "P1M"
        assert canonical_duration("1 min") == "PT1M"

    def test_dropping_the_seconds_would_lose_a_third_of_the_answer(self):
        assert written_value(Kind.DATETIME, "2000-05-13T22:55:18Z").endswith(":18 pm UTC")


class TestRecoveringTheAnswerFromTheText:
    """Every written document has to be readable back to its own answer.

    Not spot-checked: the whole grid, every field, by the documented rule and
    nothing else. An expected value no rule reaches from the page is a task
    with no correct answer.
    """

    def test_every_field_reads_back_to_what_it_was(self):
        checked = 0
        for task in grid(WRITTEN_CLASSES, notation=Notation.WRITTEN):
            for instance in task.expected:
                for slot, value in _answered(instance, task.variant):
                    written = written_value(slot.kind, value)
                    assert written in task.document
                    assert canonical_value(slot.kind, written) == value
                    checked += 1
        assert checked == 8640

    def test_the_grid_exercises_every_kind_the_mode_respells(self):
        """A round trip that never saw a clock would vouch for nothing."""
        drawn = {
            slot.kind
            for task in grid(WRITTEN_CLASSES, notation=Notation.WRITTEN)
            for instance in task.expected
            for slot, _ in _answered(instance, task.variant)
        }
        assert drawn >= RESPELLED


class TestWrittenDocuments:
    def test_a_fixed_seed_writes_the_form_where_it_wrote_the_schema(self):
        canonical = generate_task(CLASSES, task_id="t", seed=17)
        written = generate_task(CLASSES, task_id="t", seed=17, notation=Notation.WRITTEN)
        assert canonical.document == (
            "The sheet covers a single entry of type event. The maximum attendee capacity was noted as 8852. "
            "The is accessible for free is given as false. Under event status the sheet has EventScheduled. "
            "It carries the duration PT1H16M."
        )
        assert written.document == (
            "The sheet covers a single entry of type event. The maximum attendee capacity was noted as 8,852. "
            "The is accessible for free is given as no. Under event status the sheet has EventScheduled. "
            "It carries the duration 1 h 16 min."
        )
        assert canonical.expected == written.expected

    def test_the_notation_is_recorded_on_the_task(self):
        task = generate_task(CLASSES, task_id="t", seed=1, notation=Notation.WRITTEN)
        assert task.notes is not None
        assert task.notes.endswith(",notation=written")

    def test_most_of_the_grid_moves_and_the_rest_is_counted(self):
        """The documents that stay are the draws with no respelled slot in them.

        Reported rather than engineered away. A class whose declared properties
        are text, a URL and an enumeration is written the same either way, and
        the alternative is to stop drawing those slots, which would narrow the
        corpus to buy a number.
        """
        moved = sum(
            1
            for old, new in zip(grid(WRITTEN_CLASSES), grid(WRITTEN_CLASSES, notation=Notation.WRITTEN), strict=True)
            if old.document != new.document
        )
        assert moved == 1422

    def test_an_enumeration_member_is_left_canonical(self):
        """Out of scope on purpose: nothing deterministic publishes a phrase
        for ``EventScheduled``, so inventing one would be authoring."""
        task = generate_task(CLASSES, task_id="t", seed=17, notation=Notation.WRITTEN)
        assert "EventScheduled" in task.document

    def test_a_catalogue_that_never_states_a_kind_is_refused(self):
        """The form the answer takes is no longer on the page."""
        with pytest.raises(ValueError, match="what form an answer takes"):
            generate_task(CLASSES, task_id="t", seed=1, catalogue=("Book",), notation=Notation.WRITTEN)

    def test_a_described_catalogue_is_accepted(self):
        task = generate_task(
            CLASSES,
            task_id="t",
            seed=1,
            catalogue=tuple(cls.name for cls in CLASSES),
            describe_catalogue=True,
            notation=Notation.WRITTEN,
        )
        assert task.catalogue_text is not None
        assert "(duration)" in task.catalogue_text["Event"]


class TestTheWrittenModeComposesWithTheImplied:
    """The interaction that would have gone wrong quietly.

    An implied document says which class it is by the form of its values, so a
    change to how a value is written could have broken the identification the
    pool promises. It does not, and the reason is that the pool is read off the
    schema while the forms stay pairwise apart on the page.
    """

    def test_the_pool_is_the_same_set_of_classes(self):
        """:func:`legible_slots` reads the schema and never the page."""
        assert [cls.name for cls in implied_classes(WRITTEN_CLASSES, 3)] == [
            "Book",
            "Person",
            "Event",
            "ServicePeriod",
            "DeliveryEvent",
        ]

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (Kind.DATE, Kind.DATETIME),
            (Kind.TIME, Kind.DATETIME),
            (Kind.TIME, Kind.DURATION),
            (Kind.NUMBER, Kind.DATE),
            (Kind.BOOLEAN, Kind.ENUM),
        ],
    )
    def test_two_forms_a_reader_has_to_tell_apart_stay_apart(self, left, right):
        rng = random.Random(7)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        slots = {
            Kind.DATE: Slot(name="a", kind=Kind.DATE),
            Kind.DATETIME: Slot(name="b", kind=Kind.DATETIME),
            Kind.TIME: Slot(name="c", kind=Kind.TIME),
            Kind.DURATION: Slot(name="d", kind=Kind.DURATION),
            Kind.NUMBER: Slot(name="e", kind=Kind.NUMBER),
            Kind.BOOLEAN: Slot(name="f", kind=Kind.BOOLEAN),
            Kind.ENUM: Slot(name="g", kind=Kind.ENUM, choices=("Hardcover", "EventScheduled")),
        }
        written = {
            kind: {
                written_value(kind, value)
                for _ in range(40)
                for _, value in seed_values(SchemaClass(name="One", slots=(slots[kind],)), rng, 1)
            }
            for kind in (left, right)
        }
        assert not written[left] & written[right]

    def test_no_enumeration_member_reads_as_a_boolean(self):
        """Measured over the corpus at 530 members, and over the fixture here."""
        members = {m for cls in WRITTEN_CLASSES for s in cls.slots if s.kind is Kind.ENUM for m in s.choices}
        assert not members & {"yes", "no"}

    def test_an_implied_written_document_still_names_nothing(self):
        for seed in range(30):
            task = generate_task(
                WRITTEN_CLASSES,
                task_id="t",
                seed=seed,
                n_slots=3,
                labelling=Labelling.IMPLIED,
                notation=Notation.WRITTEN,
            )
            assert readable_kind(task.expected[0].class_path) not in task.document
            for name in task.expected[0].fields:
                assert readable_kind(name) not in task.document


class TestTheWrittenSeamCloses:
    """Corpus to task to score, for a document that writes its values out."""

    @pytest.mark.parametrize("difficulty", [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD])
    @pytest.mark.parametrize("labelling", [Labelling.NAMED, Labelling.IMPLIED])
    def test_a_perfect_answer_scores_one(self, difficulty, labelling):
        task = generate_task(
            WRITTEN_CLASSES,
            task_id="t",
            seed=11,
            difficulty=difficulty,
            n_entities=3,
            n_slots=3,
            labelling=labelling,
            notation=Notation.WRITTEN,
        )
        result = score_task(task, perfect_answer(task))
        assert result.primary == pytest.approx(1.0)
        assert result.dimensions[Dimension.CLASS].f1 == pytest.approx(1.0)
        assert result.dimensions[Dimension.PROPERTY].f1 == pytest.approx(1.0)

    @pytest.mark.parametrize("difficulty", [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD])
    def test_a_perfect_answer_scores_one_through_the_extractor(self, difficulty):
        task = generate_task(
            WRITTEN_CLASSES,
            task_id="t",
            seed=11,
            difficulty=difficulty,
            n_entities=3,
            n_slots=3,
            notation=Notation.WRITTEN,
        )
        assert score_task(task, extract_json(json.loads(_as_json_answer(task)))).primary == pytest.approx(1.0)

    def test_an_answer_copied_off_the_page_does_not_pass(self):
        """What the mode buys, as a score, and where the grader is lenient.

        Every value is what the document says, which is what an arm that reads
        without normalising produces. The enumeration member passes because it
        was never respelled, and the integer passes because
        :func:`~oold_llm_bench.grading.compare.same_value` strips digit
        separators before comparing numbers. The boolean and the duration do
        not, and those are the slots the mode is measuring.
        """
        task = generate_task(CLASSES, task_id="t", seed=17, n_slots=4, notation=Notation.WRITTEN)
        slots = {slot.name: slot for slot in EVENT.slots}
        copied = TripleSet(
            triples=make_triples(
                "e1",
                {name: written_value(slots[name].kind, value) for name, value in task.expected[0].fields.items()},
            ),
            classes={"e1": task.expected[0].class_path},
            provenance={},
        )
        assert score_task(task, copied).primary == pytest.approx(0.5)


def _copied_off_the_page(task, notation):
    """What an arm that transcribes and never normalises would answer.

    The schema.org analogue of
    :class:`~oold_llm_bench.runner.controls.SpellingClient`, which is
    quantity-shaped: it looks for a number and the words beside it and submits
    them as a magnitude and a unit, which is not the shape any answer here
    takes. On this corpus it scores 0.00 under either notation and so says
    nothing about either, and the same goes for the wrong-document control,
    which reads ``fields["value"]`` and finds nothing to read. The control that
    can see the difference is this one, written against the corpus it measures.
    """
    by_name = {cls.name: cls for cls in WRITTEN_CLASSES}
    triples = frozenset()
    classes = {}
    for instance in task.expected:
        slots = {slot.name: slot for slot in by_name[instance.class_path].slots}
        triples |= make_triples(
            instance.key,
            {name: spell(value, slots[name].kind, notation) for name, value in instance.fields.items()},
        )
        classes[instance.key] = instance.class_path
    return TripleSet(triples=triples, classes=classes, provenance={})


class TestWhatCopyingThePageIsWorth:
    """How much of the corpus is answerable without normalising anything.

    The quantity corpus found 115 of 120 and that is why it stopped writing
    units as identifiers. This is the same measurement for schema.org, and the
    canonical number is of the same order: a transcriber that copies each value
    as the document states it is perfect on 88 of 120 labelled documents.

    The written residual is the enumeration decision and the grader's numeric
    leniency, both stated rather than engineered away.
    """

    def scores(self, labelling, notation, n_slots):
        return [
            score_task(
                task := generate_task(
                    WRITTEN_CLASSES,
                    task_id=f"t{seed}",
                    seed=seed,
                    difficulty=Difficulty.HARD,
                    n_slots=n_slots,
                    labelling=labelling,
                    notation=notation,
                ),
                _copied_off_the_page(task, notation),
            ).primary
            for seed in range(120)
        ]

    @pytest.mark.parametrize(
        ("labelling", "n_slots", "notation", "mean", "perfect"),
        [
            (Labelling.NAMED, 4, Notation.CANONICAL, 0.9333, 88),
            (Labelling.NAMED, 4, Notation.WRITTEN, 0.6604, 10),
            (Labelling.IMPLIED, 3, Notation.CANONICAL, 0.9056, 86),
            (Labelling.IMPLIED, 3, Notation.WRITTEN, 0.5472, 0),
        ],
    )
    def test_transcription_stops_being_enough(self, labelling, n_slots, notation, mean, perfect):
        scores = self.scores(labelling, n_slots=n_slots, notation=notation)
        assert sum(scores) / len(scores) == pytest.approx(mean, abs=5e-5)
        assert sum(score == 1.0 for score in scores) == perfect

    def test_the_kinds_a_transcriber_still_gets_right_are_the_ones_left_canonical(self):
        """Two exceptions, and both are worth naming.

        An enumeration member is copied correctly because nothing respells it,
        which is the out-of-scope decision showing up as a number. An integer
        and a number are copied correctly although they are respelled, because
        the grader strips digit separators before comparing. Five of the seven
        respelled kinds are normalisation the score depends on; those two are
        surface variation and nothing more.
        """
        passes = set()
        for seed in range(120):
            task = generate_task(
                WRITTEN_CLASSES,
                task_id=f"t{seed}",
                seed=seed,
                difficulty=Difficulty.HARD,
                n_slots=4,
                notation=Notation.WRITTEN,
            )
            for slot, value in _answered(task.expected[0], task.variant):
                if same_value(value, written_value(slot.kind, value)):
                    passes.add(slot.kind)
        assert passes == {Kind.TEXT, Kind.EMAIL, Kind.ENUM, Kind.INTEGER, Kind.NUMBER}


def _linked(seed, n_entities=2, classes=None, **overrides):
    return generate_task(
        classes if classes is not None else CLASSES,
        task_id=f"t{seed}",
        seed=seed,
        n_entities=n_entities,
        linked=True,
        **overrides,
    )


def _body(instance):
    """One expected instance as JSON, with its links left out."""
    return {
        "type": instance.class_path,
        **{name: value for name, value in instance.fields.items() if not isinstance(value, Reference)},
    }


def _answer(task, *, links=True, points_at=None):
    """What a model writes: a link is the entity it reaches, nested in its source.

    ``points_at`` sends every edge somewhere else. ``"another"`` puts it on a
    different record of the same draw, ``"nothing"`` on a record that is not
    on the page at all.
    """
    by_key = {instance.key: instance for instance in task.expected}
    reached = {v.key for i in task.expected for v in i.fields.values() if isinstance(v, Reference)}
    if not links:
        return json.loads(json.dumps([_body(i) for i in task.expected], default=str))

    written = []
    for instance in task.expected:
        if instance.key in reached:
            continue
        body = _body(instance)
        for name, value in instance.fields.items():
            if not isinstance(value, Reference):
                continue
            if points_at == "nothing":
                body[name] = {"type": "Thing", "name": "Nobody At All"}
                continue
            elsewhere = [key for key in sorted(reached) if key != value.key]
            body[name] = _body(by_key[elsewhere[0] if points_at == "another" and elsewhere else value.key])
        written.append(body)
    return json.loads(json.dumps(written, default=str))


def _emitted(task, *, without=None, invented=False, twice=None):
    """A perfect answer as triples, with one entity moved out of place."""
    triples = frozenset()
    classes = {}
    for instance in task.expected:
        if instance.key == without:
            continue
        triples |= make_triples(instance.key, instance.fields)
        classes[instance.key] = instance.class_path
        if instance.key == twice:
            triples |= make_triples(f"{instance.key}-again", instance.fields)
            classes[f"{instance.key}-again"] = instance.class_path
    if invented:
        triples |= make_triples("ghost", {"name": "Nobody At All"})
        classes["ghost"] = "Thing"
    return TripleSet(triples=triples, classes=classes, provenance={})


class TestWhatCanBeLinked:
    """Only the configurations a reader can resolve, and the cost of saying so."""

    def test_only_text_can_name_an_entry(self):
        """ "The entry filed as 8852" is a record number and not a record."""
        assert {slot.name for slot in designating_slots(PERSON)} == {"familyName", "givenName"}

    def test_a_code_cannot_name_an_entry(self):
        """It reads as the link's own value rather than as another record."""
        assert {slot.name for slot in designating_slots(BOOK)} == {"bookEdition"}

    def test_an_inherited_slot_cannot_name_an_entry(self):
        """Every class has ``name``, so naming entries by it names Things."""
        assert all(not slot.inherited for cls in CLASSES for slot in designating_slots(cls))

    def test_a_range_outside_the_pool_is_not_drawable(self):
        """No Organization is written down, so the link would reach nothing."""
        assert resolvable_links(PERSON, CLASSES) == ()

    def test_a_range_no_document_can_name_is_not_drawable(self):
        """An Event declares no text slot, so nothing on its record designates it."""
        assert resolvable_links(EVENT, CLASSES) == ()

    def test_the_pool_is_what_survives_both_rules(self):
        assert [cls.name for cls in linked_classes(CLASSES)] == ["Book"]
        assert [(link.name, target.name) for link, target in resolvable_links(BOOK, CLASSES)] == [("author", "Person")]


class TestLinkedDocuments:
    """Two entities on the page, and the relation between them in prose."""

    def test_the_document_states_both_entities_and_the_relation(self):
        task = _linked(17)
        assert "type book" in task.document
        assert "person entry" in task.document
        assert "Under author the sheet points at the entry for" in task.document

    def test_the_target_is_named_by_a_value_its_own_record_shows(self):
        """The whole of how a reader resolves the edge."""
        for seed in range(30):
            entities = draw_linked(CLASSES, random.Random(seed), 2, 4)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
            source, target = entities
            edge = source.edges[0]
            assert edge.target == target.key
            assert edge.designator in {spell(value) for _, value in target.values}
            assert edge.designator in _linked(seed).document

    def test_the_name_that_designates_the_target_belongs_to_no_one_else(self):
        """Unshared, which is all the rule asks. Two entries agreeing on a
        page number leaves the edge recoverable and is not interfered with."""
        for seed in range(60):
            entities = draw_linked(CLASSES, random.Random(seed), 4, 4)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
            for edge in (edge for entity in entities for edge in entity.edges):
                holders = {e.key for e in entities for _, v in e.values if spell(v) == edge.designator}
                assert holders == {edge.target}

    def test_each_record_is_its_own_paragraph(self):
        """A link sentence says "its" of the subject the paragraph opened with."""
        task = _linked(5, n_entities=4)
        assert len(task.document.split("\n\n")) == 4
        assert "Records:" not in task.document

    def test_the_edge_count_follows_the_entity_count(self):
        counted = [
            sum(isinstance(v, Reference) for i in _linked(3, n).expected for v in i.fields.values())
            for n in range(2, 7)
        ]
        assert counted == [1, 1, 2, 2, 3]

    def test_an_odd_entity_is_drawn_unlinked(self):
        task = _linked(3, n_entities=3)
        assert len(task.expected) == 3
        assert sum(isinstance(v, Reference) for i in task.expected for v in i.fields.values()) == 1

    def test_a_renamed_linked_document_contains_no_schema_org_name(self):
        for seed in range(20):
            _, renamed = generate_pair(CLASSES, task_id="t", seed=seed, n_entities=2, linked=True)
            assert "author" not in renamed.document
            assert readable_kind("Person") not in renamed.document

    def test_the_link_property_is_graded_under_the_variant_that_wrote_it(self):
        native, renamed = generate_pair(CLASSES, task_id="t", seed=4, n_entities=2, linked=True)
        assert native.expected[0].fields["author"] == Reference(key="e2")
        assert renamed.expected[0].fields[property_identifier("author", Variant.RENAMED)] == Reference(key="e2")

    def test_the_catalogue_entry_says_what_a_link_points_at(self):
        task = _linked(1, catalogue=tuple(cls.name for cls in CLASSES), describe_catalogue=True)
        assert "author (link: Person)" in (task.catalogue_text or {})["Book"]
        assert "author" in (task.branches or {})["Book"]

    def test_an_unlinked_task_says_nothing_about_links(self):
        """Every measured result was taken against a catalogue without them."""
        task = generate_task(CLASSES, task_id="t", seed=1, catalogue=("Book",), describe_catalogue=True)
        assert "author" not in (task.catalogue_text or {})["Book"]
        assert "author" not in (task.branches or {})["Book"]

    def test_the_implied_mode_is_refused(self):
        """Nothing on an implied page separates a value an entry holds from a
        value that names another entry, so the edge is not recoverable."""
        with pytest.raises(ValueError, match="the implied mode withholds"):
            _linked(1, n_slots=3, labelling=Labelling.IMPLIED)

    def test_one_entity_cannot_carry_a_link(self):
        with pytest.raises(ValueError, match="a link needs two entities"):
            _linked(1, n_entities=1)

    def test_a_corpus_with_no_drawable_pair_is_refused(self):
        with pytest.raises(ValueError, match="links to a class a document can name"):
            _linked(1, classes=[PERSON, EVENT])

    def test_a_shared_name_is_refused_rather_than_written_down(self, monkeypatch):
        """Forced, because 65,536 appellations will not collide on their own.

        Four entities and not two, because only the slot a draw designates its
        target by takes the appellation form. One link has one such value on
        the page, so a clash needs two links, and forcing the form on a single
        pair no longer reaches the guard it is here to exercise.
        """
        monkeypatch.setattr(schemaorg, "_proper_noun", lambda rng: "Calder Wynn")
        with pytest.raises(ValueError, match="is carried by"):
            _linked(1, n_entities=4)


class TestTheLinkedSeamCloses:
    """Corpus to task to score, for entities that point at one another."""

    def test_a_perfect_answer_scores_one(self):
        for seed in range(30):
            task = _linked(seed, n_entities=4)
            result = score_task(task, perfect_answer(task))
            assert result.primary == pytest.approx(1.0)
            assert result.dimensions[Dimension.ENTITY].f1 == pytest.approx(1.0)

    def test_the_reference_triples_are_what_a_perfect_answer_gets_right(self):
        """Asserted on the edges and not only on the primary number."""
        task = _linked(11, n_entities=4)
        produced = extract_json(_answer(task))
        result = score_task(task, produced)
        assert len([t for t in produced.triples if isinstance(t.value, Reference)]) == 2
        assert result.primary == pytest.approx(1.0)
        assert result.dimensions[Dimension.PROPERTY].f1 == pytest.approx(1.0)

    def test_the_same_entities_without_the_links_score_lower(self):
        """The finding the reference work exists for, on generated tasks.

        The gap is the two edges against the eighteen triples a four-entity
        draw of this pool carries, so it is the same number on every seed.
        """
        for seed in range(60):
            task = _linked(seed, n_entities=4)
            linked = score_task(task, extract_json(_answer(task))).primary
            unlinked = score_task(task, extract_json(_answer(task, links=False))).primary
            assert linked == pytest.approx(1.0)
            assert unlinked == pytest.approx(16 / 17)
            assert linked > unlinked

    def test_a_link_at_the_wrong_record_scores_worse_than_no_link(self):
        """Which way round this falls is a finding and not a requirement."""
        for seed in range(60):
            task = _linked(seed, n_entities=4)
            missing = score_task(task, extract_json(_answer(task, links=False))).primary
            wrong = score_task(task, extract_json(_answer(task, points_at="another"))).primary
            invented = score_task(task, extract_json(_answer(task, points_at="nothing"))).primary
            assert wrong < missing
            assert invented < wrong

    def test_a_missing_entity_moves_recall_and_not_precision(self):
        entity = score_task(_linked(3, 4), _emitted(_linked(3, 4), without="e2")).dimensions[Dimension.ENTITY]
        assert entity.precision == pytest.approx(1.0)
        assert entity.recall == pytest.approx(0.75)

    def test_an_invented_entity_moves_precision_and_not_recall(self):
        entity = score_task(_linked(3, 4), _emitted(_linked(3, 4), invented=True)).dimensions[Dimension.ENTITY]
        assert entity.precision == pytest.approx(0.8)
        assert entity.recall == pytest.approx(1.0)

    def test_an_entity_emitted_twice_moves_the_duplicate_dimension(self):
        task = _linked(3, n_entities=4)
        clean = score_task(task, _emitted(task))
        restated = score_task(task, _emitted(task, twice="e1"))
        assert clean.dimensions[Dimension.DUPLICATE].f1 == pytest.approx(1.0)
        assert restated.dimensions[Dimension.DUPLICATE].f1 == pytest.approx(0.8889, abs=5e-5)
        assert restated.dimensions[Dimension.ENTITY].recall == pytest.approx(1.0)

    def test_a_written_linked_document_still_names_its_target(self):
        """Text is a kind the written notation leaves alone, so the two compose."""
        task = _linked(11, n_entities=4, notation=Notation.WRITTEN)
        assert score_task(task, perfect_answer(task)).primary == pytest.approx(1.0)
        assert score_task(task, extract_json(_answer(task))).primary == pytest.approx(1.0)

    def test_a_task_written_to_json_and_read_back_scores_the_same(self):
        """The reference reaches disk as ``{"key": ...}`` and comes back a link."""
        task = _linked(11, n_entities=4)
        reloaded = TaskRecord.model_validate_json(task.model_dump_json())
        produced = extract_json(_answer(task))
        assert score_task(reloaded, produced).primary == score_task(task, produced).primary

    def test_a_link_at_an_instance_the_task_does_not_hold_is_refused(self):
        """Ground truth that points nowhere has no correct answer."""
        with pytest.raises(ValueError, match="point at no expected instance"):
            TaskRecord(
                id="t",
                document="d",
                expected=[ExpectedInstance(key="e1", class_path="Book", fields={"author": Reference(key="e9")})],
                corpus=CorpusRef(source=Source.SYNTHETIC, document_id="d", content_hash="0" * 64),
                split=Split.DEV,
            )


class TestTheDrawStaysInsideTheCatalogueCap:
    """Every expected field has to be in the schema the arm is given.

    `CATALOGUE_SLOTS` caps the catalogue entry, the union branches and the
    answer schema. The named mode drew from `own_slots` uncapped, so 36 of 480
    expected fields fell outside the schema the arm was sent, 26 of 120 tasks
    were unanswerable under a union before a model saw them, and every named
    A4-against-A2 figure was biased against A4 by 0.060. The implied mode never
    had it: `legible_slots` caps first.

    The two grid digests did not catch this because they are computed over the
    fixtures below, whose classes all declare fewer slots than the cap. They
    pin the renderer, not the corpus.
    """

    WIDE = SchemaClass(
        name="Wide",
        parents=("Thing",),
        label="Wide",
        description="A class declaring more slots than the cap allows.",
        slots=tuple(Slot(name=f"prop{n}", kind=Kind.TEXT) for n in range(CATALOGUE_SLOTS + 8)),
    )

    def test_a_drawn_slot_is_one_the_catalogue_entry_shows(self):
        task = generate_task([self.WIDE], task_id="t1", seed=5, n_slots=4, catalogue=("Wide",))
        shown = {slot.name for slot in self.WIDE.own_slots[:CATALOGUE_SLOTS]}
        for instance in task.expected:
            assert set(instance.fields) <= shown

    def test_every_expected_field_reaches_the_answer_schema(self):
        """The property the fault actually broke."""
        task = generate_task([self.WIDE], task_id="t1", seed=6, n_slots=4, catalogue=("Wide",))
        rendered = json.dumps(answer_schema([self.WIDE], Variant.NATIVE))
        for instance in task.expected:
            for field in instance.fields:
                assert f'"{field}"' in rendered

    def test_a_class_under_the_cap_is_unaffected(self):
        """Which is why the fixtures never saw it."""
        assert len(BOOK.own_slots) < CATALOGUE_SLOTS
        task = generate_task([BOOK], task_id="t1", seed=7, n_slots=3, catalogue=("Book",))
        assert task.expected
