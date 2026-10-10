"""Task record schema.

Frozen interface. Corpus generators write it, the runner reads it, the grader
scores against it. Changing a field here invalidates every task file and every
result produced from one, so bump ``TASK_SCHEMA_VERSION`` when it happens.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TASK_SCHEMA_VERSION = "1"


class Split(str, Enum):
    """Which half of the corpus a task belongs to.

    Tuning touches ``dev`` only. Every report states how often ``test`` was
    read.
    """

    DEV = "dev"
    TEST = "test"


class Variant(str, Enum):
    """Whether the vocabulary is one the models have seen in pretraining.

    ``renamed`` carries the same structure under opaque class and property
    names, which is the contrast that keeps schema.org exposure from
    flattering the arms that get no schema.
    """

    NATIVE = "native"
    RENAMED = "renamed"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Source(str, Enum):
    """Where the document came from."""

    SYNTHETIC = "synthetic"
    BULK = "bulk"
    CRAWL = "crawl"
    MANUAL = "manual"


class CorpusRef(BaseModel):
    """Provenance of one document, recorded so a run can be repeated."""

    model_config = ConfigDict(frozen=True)

    name: str = ""
    exhaustive: bool = True
    """Whether everything the document states is recorded as expected.

    True of a generated corpus, which wrote the document from the entities.
    False of a harvested one, where it fails twice over.

    For entities: a Wikipedia lead about a university names its founder, its
    city and its parent organisation, and the corpus records the subject
    alone. Measured, a step reaching an entity recall of 1.00 scored an entity
    precision of 0.11 on those tasks.

    For properties: the expected slots are the ones Wikidata happens to hold
    for that item and whose value the lead also states, which is a subset of
    what the lead states. Measured over 4,568 rejected property names, 76%
    name a property Wikidata records nothing about for that entity, against
    14% that are a vocabulary near miss and 10% that Wikidata holds but the
    lead does not state. So fillable precision on this corpus is mostly a
    reading of Wikidata's coverage.

    One flag and not two, because no corpus has been met that records every
    entity and only some properties. A report declines to headline a number
    nobody can read rather than each report deciding for itself."""
    """Which corpus this document came from, as one word.

    Added 2026-10-04 because nothing carried it. The corpus was readable only
    out of ``TaskRecord.notes``, which three corpora spelled three different
    ways, so no check could compare the corpus a model was tuned on against
    the corpus it was being scored on. One was tuned on generated quantity
    prose and scored on human encyclopedic prose, and the gap between it and
    a model tuned on the register it was tested on was read for a while as a
    fact about the two models.
    """

    source: Source
    document_id: str
    content_hash: str
    """sha256 of the bytes the document was derived from."""
    url: str | None = None
    retrieved_at: datetime | None = None
    licence: str | None = None
    robots_allowed: bool | None = None

    @model_validator(mode="after")
    def _crawled_documents_carry_their_origin(self) -> CorpusRef:
        if self.source in (Source.BULK, Source.CRAWL):
            if not self.url:
                raise ValueError(f"{self.source.value} document needs a url")
            if self.retrieved_at is None:
                raise ValueError(f"{self.source.value} document needs a retrieval time")
        return self


class ExpectedInstance(BaseModel):
    """One instance the document should yield, with the values it should carry.

    ``fields`` may not be empty. Matching on class alone lets a
    ``LaboratoryProcess`` named "banana" pass, as the predecessor
    benchmark scored four of its six expected entities.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    """Stable identifier within the task, used to report per-instance results."""
    class_path: str
    fields: dict[str, Any]
    """Property name to the value it carries.

    A value may be a :class:`~oold_llm_bench.grading.triples.Reference` to
    another instance's :attr:`key`, which is how a task asserts an edge rather
    than a literal. It survives a JSON round trip as ``{"key": ...}``, and
    ``Triple`` reads that shape back, so a task written to disk and scored
    from disk scores the same."""
    optional_fields: dict[str, Any] = Field(default_factory=dict)
    """Scored when present, not counted as a miss when absent."""
    mentions: tuple[str, ...] = ()
    """The words the document refers to this entity by.

    A list, because one entity is named several ways in the same document and
    a step answering any of them has read it correctly. Empty where the corpus
    does not record it, which is not the same as empty words: a step is then
    scored on class and count and not on the mention at all.

    What the steps after the first are told the entity by. A plan that finds
    the right number of entities and names them wrongly hands the next step a
    question about something else, and that failure is invisible in a score
    over the values."""
    allow_subclass: bool = True

    @field_validator("fields")
    @classmethod
    def _at_least_one_value(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("expected instances must assert at least one field value")
        return value


class TaskRecord(BaseModel):
    """One extraction task: an input document and what it should produce."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = TASK_SCHEMA_VERSION
    id: str
    document: str
    """The text handed to the model, with any structured markup removed."""
    expected: list[ExpectedInstance]
    corpus: CorpusRef
    split: Split
    variant: Variant = Variant.NATIVE
    difficulty: Difficulty = Difficulty.MEDIUM
    catalogue: list[str] | None = None
    """Class paths this task needs. The catalogue-size sweep adds distractors
    around these, so a trimmed catalogue never removes a required class."""

    answer_schema: dict[str, Any] | None = None
    """The shape an answer to this task takes.

    Carried by the task because the corpus knows it and the runner does not. A
    quantity is a class, a magnitude and a unit; a schema.org entity is a class
    and whatever properties that class defines. One hardcoded shape fits the
    first and silently misdescribes the second."""

    branches: dict[str, dict[str, Any]] | None = None
    """What each offered class narrows, for a union constraint. Keyed by the
    identifier the catalogue uses, so a trim trims these with it."""

    class_parents: dict[str, list[str]] | None = None
    """Which classes each class inherits from.

    Carried when the corpus has a hierarchy. A union then states a property
    where it is declared instead of repeating every inherited one in every
    branch, and a class with two parents keeps both."""

    property_ranges: dict[str, list[str]] | None = None
    """Which classes each link property may point at.

    Carried when the corpus declares object-valued properties. It is what lets
    a reference slot be pinned at decode time to the ids of entities whose
    class fits the range, so a link is constrained to a real target rather
    than to any string."""

    property_text: dict[str, str] | None = None
    """What each offered property name means, where the vocabulary says.

    Read by the step that chooses between property names and by no other. A
    bare name does not say whether the gallery holding a painting is its
    ``contentLocation`` or its ``provider``, and that is a vocabulary
    question rather than a reading one."""
    catalogue_text: dict[str, str] | None = None
    catalogue_enums: dict[str, str] | None = None
    """The same entries with the prose removed and the enumerations kept.

    What a presented schema enforces has to be in it; what it merely
    annotates does not. So a condition that closes the unit slot shows the
    unit enumeration whether or not it shows descriptions, and the bare
    catalogue stops hiding the one thing the grammar will check.
    """
    """Each offered class rendered for the prompt, keyed by identifier.

    A catalogue of bare identifiers asks a model to choose between names it
    was told nothing about. What the entries may carry, and what has to be
    withheld because the document was built from it, is decided when the task
    is generated and recorded with it."""

    accepts_conversion: bool = False
    """Whether a rescaled answer in another unit is a correct answer.

    Off by default. A document states a value in a unit and the corpus closes
    the unit slot, so the answer is that magnitude in that unit. A task that
    asks for normalisation is the case where a converted answer is right, and
    it says so here.

    Only units convert. A closed slot of class identifiers, or of any other
    enumeration, has no equivalent, so enum conformance is the general
    question and physical equality is a unit-only special case."""

    unit_catalogue: dict[str, list[str]] | None = None
    """Units each offered class admits, keyed by the class identifier.

    A mapping and not a flat list, so trimming the catalogue trims the units
    with it. A 25-class catalogue paired with the units of all 943 would
    constrain almost nothing while looking constrained."""
    notes: str | None = None

    @field_validator("expected")
    @classmethod
    def _keys_are_unique(cls, value: list[ExpectedInstance]) -> list[ExpectedInstance]:
        if not value:
            raise ValueError("a task must expect at least one instance")
        keys = [instance.key for instance in value]
        duplicates = {k for k in keys if keys.count(k) > 1}
        if duplicates:
            raise ValueError(f"duplicate instance keys: {sorted(duplicates)}")
        return value

    @model_validator(mode="after")
    def _links_reach_an_expected_instance(self) -> TaskRecord:
        """An edge naming no instance of this task has no correct answer.

        ``Reference`` is imported here rather than at the top of the module
        because the grader reads this file, and naming it at module scope
        would close the loop.
        """
        from oold_llm_bench.grading.triples import Reference

        keys = {instance.key for instance in self.expected}
        dangling = sorted(
            f"{instance.key}.{prop}"
            for instance in self.expected
            for prop, value in (instance.fields | instance.optional_fields).items()
            for item in (value if isinstance(value, (list, tuple)) else [value])
            if isinstance(item, Reference) and item.key not in keys
        )
        if dangling:
            raise ValueError(f"links point at no expected instance: {dangling}")
        return self

    @model_validator(mode="after")
    def _catalogue_covers_expected_classes(self) -> TaskRecord:
        if self.catalogue is None:
            return self
        offered = set(self.catalogue)
        missing = sorted({i.class_path for i in self.expected} - offered)
        if missing:
            raise ValueError(f"catalogue omits expected classes: {missing}")
        return self


class TaskSet(BaseModel):
    """A named collection of tasks, the unit a run is configured against."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = TASK_SCHEMA_VERSION
    name: str
    tasks: list[TaskRecord]

    def split(self, split: Split) -> TaskSet:
        return TaskSet(
            name=f"{self.name}:{split.value}",
            tasks=[t for t in self.tasks if t.split is split],
        )
