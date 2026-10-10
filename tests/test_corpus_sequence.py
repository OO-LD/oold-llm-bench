"""Splitting one entity across several documents.

Dedup truth is true by construction here: every document's local entity is
the one entity the draw started from, so these tests are about what the
split records, not about whether the pieces really are one entity.
"""

from __future__ import annotations

import random

import pytest

from oold_llm_bench.corpus.schemaorg import Kind, Link, SchemaClass, Slot, spell
from oold_llm_bench.corpus.sequence import draw_sequence, generate_sequence, tasks_of

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

CLASSES = [BOOK, PERSON]


class TestDrawSequence:
    def test_every_document_names_the_entity(self):
        """Without the designator in every document, a later one is not
        readable as being about the first one's subject at all."""
        item = draw_sequence(CLASSES, random.Random(1), n_documents=3)  # noqa: S311
        mention = item.expected_merge.mentions[0]
        assert all(mention in document for document in item.documents)

    def test_the_merge_is_the_entity_the_draw_started_from(self):
        """No overlap, no conflict: the merge is just every field, once."""
        item = draw_sequence(CLASSES, random.Random(2), n_slots=4, n_documents=4, overlap=0)  # noqa: S311
        stated = {}
        for fields in item.per_document:
            stated.update(fields)
        assert item.expected_merge.fields == stated

    def test_an_overlapping_field_is_stated_in_two_documents(self):
        item = draw_sequence(CLASSES, random.Random(3), n_slots=4, n_documents=3, overlap=1)  # noqa: S311
        name = next(iter(item.overlapping))
        holders = [fields for fields in item.per_document if name in fields]
        assert len(holders) == 2

    def test_a_full_conflict_rate_always_contradicts_the_overlap(self):
        rng = random.Random(4)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        item = draw_sequence(CLASSES, rng, n_slots=4, n_documents=3, overlap=2, conflict_rate=1.0)
        assert item.contradicted
        assert set(item.contradicted) == item.overlapping

    def test_a_zero_conflict_rate_never_contradicts(self):
        rng = random.Random(5)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        item = draw_sequence(CLASSES, rng, n_slots=4, n_documents=3, overlap=2, conflict_rate=0.0)
        assert item.contradicted == {}

    def test_the_merge_resolves_a_contradiction_to_the_later_value(self):
        rng = random.Random(4)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
        item = draw_sequence(
            CLASSES,
            rng,
            n_slots=4,
            n_documents=3,
            overlap=2,
            conflict_rate=1.0,
        )
        for name, (_, later) in item.contradicted.items():
            assert item.expected_merge.fields[name] == spell(later)

    def test_overlap_cannot_exceed_what_a_sequence_can_carry(self):
        """Asking for more repeats than there are fields to repeat is not an
        error: it is clamped, since the corpus is asked for a draw and not
        for a specific count."""
        item = draw_sequence(CLASSES, random.Random(6), n_slots=2, n_documents=5, overlap=50)  # noqa: S311
        assert len(item.overlapping) <= 2

    def test_zero_documents_is_refused(self):
        with pytest.raises(ValueError, match="at least one document"):
            draw_sequence(CLASSES, random.Random(7), n_documents=0)  # noqa: S311

    def test_one_document_is_the_whole_entity_with_nothing_to_overlap(self):
        item = draw_sequence(CLASSES, random.Random(8), n_slots=4, n_documents=1, overlap=2)  # noqa: S311
        assert len(item.documents) == 1
        assert item.overlapping == frozenset()
        assert item.per_document[0] == item.expected_merge.fields

    def test_generate_sequence_is_reproducible_from_the_seed_alone(self):
        first = generate_sequence(CLASSES, seed=9, n_documents=3, overlap=1, conflict_rate=0.5)
        second = generate_sequence(CLASSES, seed=9, n_documents=3, overlap=1, conflict_rate=0.5)
        assert first.documents == second.documents
        assert first.expected_merge == second.expected_merge


class TestPerturb:
    """The conflict value has to differ and keep its type, or a contradiction
    could not be told apart from a value a grader would call unparsed."""

    def test_a_boolean_flips(self):
        from oold_llm_bench.corpus.sequence import _perturb

        assert _perturb(True) is False
        assert _perturb(False) is True

    def test_a_number_moves(self):
        from oold_llm_bench.corpus.sequence import _perturb

        assert _perturb(3) == 4
        assert _perturb(1.5) == 2.5

    def test_text_is_marked_rather_than_replaced(self):
        """Marked and not redrawn: the point is a detectable difference, not
        a second plausible-sounding value."""
        from oold_llm_bench.corpus.sequence import _perturb

        assert _perturb("Paris") == "Paris (alternate)"


class TestTasksOf:
    def test_one_task_per_document(self):
        item = draw_sequence(CLASSES, random.Random(10), n_slots=4, n_documents=3)  # noqa: S311
        tasks = tasks_of(item, "t")
        assert len(tasks) == 3
        assert [t.document for t in tasks] == list(item.documents)

    def test_each_task_expects_only_what_its_own_document_states(self):
        item = draw_sequence(CLASSES, random.Random(11), n_slots=4, n_documents=3, overlap=1)  # noqa: S311
        tasks = tasks_of(item, "t")
        for task, fields in zip(tasks, item.per_document, strict=True):
            assert task.expected[0].fields == fields

    def test_the_catalogue_is_the_one_true_class(self):
        item = draw_sequence(CLASSES, random.Random(12), n_documents=2)  # noqa: S311
        tasks = tasks_of(item, "t")
        assert all(task.catalogue == [item.cls.name] for task in tasks)

    def test_tasks_are_valid_task_records(self):
        """Round-trips through the same validation every other task does."""
        item = draw_sequence(CLASSES, random.Random(13), n_documents=2)  # noqa: S311
        tasks = tasks_of(item, "t")
        assert all(task.expected for task in tasks)


class TestANamelessClass:
    """Found live: a class offering only non-appellation slots crashed with a
    bare `StopIteration` instead of a clear error, because `draw(..., named=True)`
    only pins a designator in when the class has one to offer."""

    NUMBERS_ONLY = SchemaClass(
        name="NumbersOnly",
        parents=("Thing",),
        label="Numbers only",
        description="A class with nothing a document could name it by.",
        slots=(
            Slot(name="height", kind=Kind.NUMBER),
            Slot(name="weight", kind=Kind.NUMBER),
            Slot(name="count", kind=Kind.INTEGER),
            Slot(name="verified", kind=Kind.BOOLEAN),
        ),
    )

    def test_a_clear_error_names_the_class_and_the_reason(self):
        with pytest.raises(ValueError, match="NumbersOnly offers no slot whose value reads as a name"):
            draw_sequence([self.NUMBERS_ONLY], random.Random(14), n_slots=4, n_documents=2)  # noqa: S311
