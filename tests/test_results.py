"""Result records, and the boundary between local and published.

The leak tests are the point. A field added later must not reach the published
form by being forgotten.
"""

import json

import pytest

from oold_llm_bench.results import (
    PUBLISHED_MODEL_FIELDS,
    Environment,
    ModelSpec,
    RunRecord,
    config_hash,
)

# A deployment name is not the model name. They diverge in practice, which is
# exactly why the model name can be published and the deployment cannot.
DEPLOYMENT = "my-sonnet-deployment-01"
ENDPOINT = "https://some-resource.cognitiveservices.azure.com/"
SUBSCRIPTION = "96001274-0000-0000-0000-000000000000"
RESOURCE_GROUP = "some-rg"

ACCOUNT_SCOPED = {
    "deployment": DEPLOYMENT,
    "endpoint": ENDPOINT,
    "subscription": SUBSCRIPTION,
    "resource_group": RESOURCE_GROUP,
}


def model() -> ModelSpec:
    return ModelSpec(
        model="claude-sonnet-5",
        provider_profile="anthropic",
        model_version="2",
        region="swedencentral",
        temperature=0.0,
        deployment=DEPLOYMENT,
        endpoint=ENDPOINT,
        subscription=SUBSCRIPTION,
        resource_group=RESOURCE_GROUP,
    )


def record(spec: ModelSpec | None = None) -> RunRecord:
    return RunRecord(
        run_id="r1",
        arm="schema-dump-catalog-flat-enforced",
        model=model() if spec is None else spec,
        environment=Environment(benchmark_version="0.1.0", benchmark_sha="abc123"),
        enforcement={"decode_constraint": "json_schema+enum", "catalogue_size": 135},
        corpus_hash="c" * 64,
        catalogue_hash="d" * 64,
        prompt_hash="e" * 64,
    )


class TestPublishedForm:
    @pytest.mark.parametrize("field_name", sorted(ACCOUNT_SCOPED))
    def test_account_scoped_fields_never_reach_the_published_form(self, field_name):
        assert field_name not in record().publish()["model"]

    @pytest.mark.parametrize("value", sorted(ACCOUNT_SCOPED.values()))
    def test_no_account_identifier_survives_anywhere_in_the_json(self, value):
        """Serialise the whole thing and search it, not just the model block."""
        assert value not in json.dumps(record().publish())

    def test_what_reproduces_a_run_does_survive(self):
        published = record().publish()["model"]
        assert published["model"] == "claude-sonnet-5"
        assert published["model_version"] == "2"
        assert published["temperature"] == 0.0
        assert published["provider_profile"] == "anthropic"

    def test_the_model_version_is_kept_and_the_deployment_name_is_not(self):
        """A deployment name identifies one subscription. The version pins the run."""
        published = record().publish()["model"]
        assert "model_version" in published
        assert "deployment" not in published

    def test_a_new_model_field_does_not_leak_by_default(self):
        """The allowlist decides, so forgetting to classify a field is safe."""
        assert PUBLISHED_MODEL_FIELDS.isdisjoint(ACCOUNT_SCOPED)
        every_field = set(model().describe())
        unclassified = every_field - PUBLISHED_MODEL_FIELDS
        published = set(record().publish()["model"])
        assert published.isdisjoint(unclassified)

    def test_the_condition_survives_in_full(self):
        published = record().publish()
        assert published["arm"] == "schema-dump-catalog-flat-enforced"
        assert published["enforcement"]["catalogue_size"] == 135
        for key in ("corpus_hash", "catalogue_hash", "prompt_hash"):
            assert len(published[key]) == 64

    def test_withheld_names_what_was_dropped(self):
        assert record().withheld() == set(ACCOUNT_SCOPED)

    def test_withheld_is_empty_when_nothing_was_set(self):
        bare = record(ModelSpec(model="m", provider_profile="openai"))
        assert bare.withheld() == set()


class TestLocalForm:
    def test_the_local_record_keeps_everything(self):
        described = record().describe()["model"]
        for key, value in ACCOUNT_SCOPED.items():
            assert described[key] == value

    def test_both_forms_agree_on_the_condition(self):
        subject = record()
        local, published = subject.describe(), subject.publish()
        for key in ("run_id", "arm", "enforcement", "corpus_hash", "split"):
            assert local[key] == published[key]


class TestConfigHash:
    def test_key_order_does_not_change_the_hash(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_a_different_value_changes_the_hash(self):
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_it_is_a_sha256(self):
        assert len(config_hash({"a": 1})) == 64


class TestEnvironment:
    def test_it_records_what_produced_the_result(self):
        described = Environment(benchmark_version="0.1.0").describe()
        assert described["python_version"].count(".") == 2
        assert described["started_at"].endswith("+00:00")

    def test_the_published_form_carries_the_git_sha(self):
        published = Environment(benchmark_version="0.1.0", benchmark_sha="abc123").publish()
        assert published["benchmark_sha"] == "abc123"


class TestAGraphResultIsRecorded:
    """Every edge, and every edge that went nowhere.

    `e2` measured a segmented run over 120 linked tasks and the record could
    not say whether a single edge was produced, because `ExtractionResult`
    carried `links`, `dangling` and `unpinned` and `RunRecord` carried none of
    them. That is the same gap that made a two-step failure unattributable
    before `selected` was kept.
    """

    def _record(self, **kwargs):
        return RunRecord(
            run_id="r",
            arm="schema-dump-catalog-enforced",
            model=ModelSpec(model="gpt-5-nano", provider_profile="openai"),
            environment=Environment(benchmark_version="0.1.0"),
            enforcement={},
            corpus_hash="0" * 64,
            catalogue_hash="0" * 64,
            prompt_hash="0" * 64,
            **kwargs,
        )

    def test_an_edge_survives_into_the_record(self):
        record = self._record(links=[{"source": "e1", "prop": "worksFor", "target": "e2"}])
        assert record.describe()["links"] == [{"source": "e1", "prop": "worksFor", "target": "e2"}]

    def test_a_dangling_edge_is_kept_beside_the_links_not_instead(self):
        """Pointing at an entity never reported is not the same as no edge."""
        edge = {"source": "e1", "prop": "worksFor", "target": "e9"}
        record = self._record(links=[edge], dangling=[edge])
        described = record.describe()
        assert described["links"] == [edge]
        assert described["dangling"] == [edge]

    def test_an_unpinned_property_is_named(self):
        """A constraint that never engaged and one obeyed look alike without it."""
        record = self._record(unpinned=["acceptedPaymentMethod"])
        assert record.describe()["unpinned"] == ["acceptedPaymentMethod"]

    def test_a_run_with_no_graph_records_empty_rather_than_absent(self):
        described = self._record().describe()
        assert described["links"] == []
        assert described["dangling"] == []
        assert described["unpinned"] == []

    def test_the_published_form_keeps_them(self):
        """A third party rescoring a graph result needs the edges."""
        edge = {"source": "e1", "prop": "worksFor", "target": "e2"}
        published = self._record(links=[edge]).publish()
        assert published["links"] == [edge]
