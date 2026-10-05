"""Task record schema: what it accepts and what it refuses.

The refusals are the point. Each one corresponds to a way the predecessor
benchmark could score a wrong answer as a pass.
"""

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from oold_llm_bench.tasks import (
    CorpusRef,
    Difficulty,
    ExpectedInstance,
    Source,
    Split,
    TaskRecord,
    TaskSet,
    Variant,
)


def synthetic_corpus(document_id: str = "doc-1") -> CorpusRef:
    return CorpusRef(
        source=Source.SYNTHETIC,
        document_id=document_id,
        content_hash="0" * 64,
    )


def instance(
    key: str = "p1",
    class_path: str = "schemaorg.Person",
    fields: dict[str, Any] | None = None,
    optional_fields: dict[str, Any] | None = None,
) -> ExpectedInstance:
    return ExpectedInstance(
        key=key,
        class_path=class_path,
        fields={"name": "Jane Doe"} if fields is None else fields,
        optional_fields={} if optional_fields is None else optional_fields,
    )


def task(
    task_id: str = "t1",
    document: str = "Jane Doe works at Example Lab.",
    expected: list[ExpectedInstance] | None = None,
    split: Split = Split.DEV,
    catalogue: list[str] | None = None,
) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        document=document,
        expected=[instance()] if expected is None else expected,
        corpus=synthetic_corpus(),
        split=split,
        catalogue=catalogue,
    )


class TestExpectedInstance:
    def test_accepts_an_asserted_value(self):
        assert instance().fields == {"name": "Jane Doe"}

    def test_refuses_empty_fields(self):
        """Four of six expected entities in the predecessor had fields={}."""
        with pytest.raises(ValidationError, match="at least one field value"):
            instance(fields={})

    def test_optional_fields_may_be_empty(self):
        assert instance(optional_fields={}).optional_fields == {}

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            setattr(instance(), "key", "other")  # noqa: B010


class TestTaskRecord:
    def test_defaults_are_the_conservative_ones(self):
        record = task()
        assert record.variant is Variant.NATIVE
        assert record.difficulty is Difficulty.MEDIUM
        assert record.catalogue is None
        assert record.schema_version == "1"

    def test_refuses_no_expectations(self):
        with pytest.raises(ValidationError, match="at least one instance"):
            task(expected=[])

    def test_refuses_duplicate_instance_keys(self):
        with pytest.raises(ValidationError, match="duplicate instance keys"):
            task(expected=[instance("p1"), instance("p1")])

    def test_refuses_a_catalogue_missing_an_expected_class(self):
        """A trimmed catalogue must never remove a class the task needs."""
        with pytest.raises(ValidationError, match="catalogue omits"):
            task(catalogue=["schemaorg.Organization"])

    def test_accepts_a_catalogue_that_covers_the_expected_classes(self):
        record = task(catalogue=["schemaorg.Person", "schemaorg.Organization"])
        assert record.catalogue == ["schemaorg.Person", "schemaorg.Organization"]


class TestCorpusRef:
    def test_synthetic_needs_no_url(self):
        assert synthetic_corpus().url is None

    @pytest.mark.parametrize("source", [Source.BULK, Source.CRAWL])
    def test_fetched_documents_need_a_url(self, source):
        with pytest.raises(ValidationError, match="needs a url"):
            CorpusRef(source=source, document_id="d", content_hash="0" * 64)

    @pytest.mark.parametrize("source", [Source.BULK, Source.CRAWL])
    def test_fetched_documents_need_a_retrieval_time(self, source):
        with pytest.raises(ValidationError, match="retrieval time"):
            CorpusRef(
                source=source,
                document_id="d",
                content_hash="0" * 64,
                url="https://example.org/a",
            )

    def test_a_complete_crawl_reference_is_accepted(self):
        ref = CorpusRef(
            source=Source.CRAWL,
            document_id="d",
            content_hash="0" * 64,
            url="https://example.org/a",
            retrieved_at=datetime(2026, 9, 26, tzinfo=timezone.utc),
            licence="CC-BY-4.0",
            robots_allowed=True,
        )
        assert ref.robots_allowed is True


class TestTaskSet:
    def test_split_selects_only_that_half(self):
        dev = task(task_id="a", split=Split.DEV)
        held_out = task(task_id="b", split=Split.TEST)
        every = TaskSet(name="tier1", tasks=[dev, held_out])

        assert [t.id for t in every.split(Split.DEV).tasks] == ["a"]
        assert [t.id for t in every.split(Split.TEST).tasks] == ["b"]
        assert every.split(Split.TEST).name == "tier1:test"
