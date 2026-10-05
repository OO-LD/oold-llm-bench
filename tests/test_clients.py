"""Building a client without calling one.

No test here sends a request. The tests check that every model in the
catalogue resolves to a transport that exists, that missing credentials raise
before the first request, and that nothing secret reaches a record.
"""

import pytest

from oold_llm_bench.clients import TRANSPORTS, Credentials, build_client, client_factory
from oold_llm_bench.config import LADDER, entry_for

pytest.importorskip("langchain_openai")
pytest.importorskip("langchain_anthropic")
pytest.importorskip("langchain_mistralai")

ENV = {
    "OOLD_BENCH_API_KEY": "key-not-real",
    "OOLD_BENCH_ENDPOINT": "https://example.invalid/openai/v1",
    "OOLD_BENCH_ANTHROPIC_ENDPOINT": "https://example.invalid/anthropic",
    "OOLD_BENCH_API_VERSION": "2024-02-15",
    # Hosted transports are separate accounts with separate keys, so they are
    # named here rather than falling back to the Azure one. A model pointing
    # at one of them without its own credentials refuses to build, which is
    # the behaviour TestHostedTransports pins.
    "UNSLOTH_API_ENDPOINT": "https://example.invalid/unsloth",
    "UNSLOTH_API_KEY": "unsloth-not-real",
    "MISTRALAI_API_ENDPOINT": "https://example.invalid/mistral",
    "MISTRALAI_API_KEY": "mistral-not-real",
    "VLLM_4B_ENDPOINT": "https://example.invalid/vllm-4b",
    "VLLM_9B_ENDPOINT": "https://example.invalid/vllm-9b",
    "VLLM_27B_ENDPOINT": "https://example.invalid/vllm-27b",
    "VLLM_API_KEY": "vllm-not-real",
}


def credentials() -> Credentials:
    return Credentials.from_env(dict(ENV))


class TestReadingCredentials:
    def test_a_missing_key_fails_before_any_request(self):
        """Without this check the run fails at the first call, when the grid up
        to that point has already cost money.
        """
        with pytest.raises(RuntimeError, match="OOLD_BENCH_API_KEY"):
            Credentials.from_env({"OOLD_BENCH_ENDPOINT": "https://example.invalid"})

    def test_a_missing_endpoint_fails_too(self):
        with pytest.raises(RuntimeError, match="OOLD_BENCH_ENDPOINT"):
            Credentials.from_env({"OOLD_BENCH_API_KEY": "k"})

    def test_both_missing_are_named_together(self):
        with pytest.raises(RuntimeError, match="OOLD_BENCH_API_KEY, OOLD_BENCH_ENDPOINT"):
            Credentials.from_env({})

    def test_the_mistral_endpoint_falls_back_to_the_openai_one(self):
        """One resource serves both on this deployment. Another may not."""
        assert credentials().endpoint_for("mistral") == ENV["OOLD_BENCH_ENDPOINT"]

    def test_the_anthropic_endpoint_does_not_fall_back(self):
        """A different host. A guessed endpoint would send the request to the
        wrong service.
        """
        creds = Credentials.from_env({
            "OOLD_BENCH_API_KEY": "k",
            "OOLD_BENCH_ENDPOINT": "https://example.invalid",
        })
        with pytest.raises(RuntimeError, match="OOLD_BENCH_ANTHROPIC_ENDPOINT"):
            creds.endpoint_for("anthropic")

    def test_describe_reports_presence_and_not_contents(self):
        described = credentials().describe()
        assert described["OOLD_BENCH_API_KEY"] is True
        assert "key-not-real" not in str(described)


class TestEveryModelResolves:
    @pytest.mark.parametrize("entry", LADDER, ids=lambda e: e.model)
    def test_the_transport_is_one_that_exists(self, entry):
        assert entry.transport in TRANSPORTS

    @pytest.mark.parametrize("entry", LADDER, ids=lambda e: e.model)
    def test_a_client_can_be_built_for_it(self, entry):
        client = build_client(entry, credentials())
        assert client.model == entry.model

    def test_the_recorded_identifier_is_the_model_not_the_deployment(self):
        """They match on this resource and will not on another."""
        assert build_client(entry_for("gpt-5-nano"), credentials()).model == "gpt-5-nano"

    def test_a_dropped_system_message_turns_folding_on(self):
        """Otherwise the model is scored on a question it never received."""
        assert build_client(entry_for("phi-4-mini-instruct"), credentials()).fold_system is True

    def test_folding_stays_off_where_the_system_message_arrives(self):
        assert build_client(entry_for("llama-3.3-70b-instruct"), credentials()).fold_system is False

    def test_a_model_that_rejects_temperature_is_not_sent_one(self):
        entry = entry_for("claude-sonnet-5")
        client = build_client(entry, credentials(), temperature=0.0)
        assert getattr(client.llm, "temperature", None) is None

    def test_an_unknown_transport_is_refused(self):
        from dataclasses import replace

        broken = replace(entry_for("gpt-5-nano"), transport="carrier-pigeon")
        with pytest.raises(ValueError, match="carrier-pigeon"):
            build_client(broken, credentials())


class TestTheFactory:
    def test_it_builds_from_a_cell(self):
        from oold_llm_bench.corpus.quantities import QuantityKind, generate_task
        from oold_llm_bench.results.record import ModelSpec
        from oold_llm_bench.runner import Cell, Condition

        kinds = [QuantityKind(name="Length", units=("meter",))]
        cell = Cell(
            condition=Condition(arm="schema-dump-catalog-flat-enforced"),
            model=ModelSpec(model="gpt-5-nano", provider_profile="openai", temperature=0.0),
            task=generate_task(kinds, task_id="t", seed=1, catalogue=("Length",)),
            repetition=1,
        )
        assert client_factory(credentials())(cell).model == "gpt-5-nano"

    def test_a_model_outside_the_catalogue_is_refused(self):
        from oold_llm_bench.corpus.quantities import QuantityKind, generate_task
        from oold_llm_bench.results.record import ModelSpec
        from oold_llm_bench.runner import Cell, Condition

        kinds = [QuantityKind(name="Length", units=("meter",))]
        cell = Cell(
            condition=Condition(arm="schema-dump-catalog-flat-enforced"),
            model=ModelSpec(model="gpt-4", provider_profile="openai"),
            task=generate_task(kinds, task_id="t", seed=1, catalogue=("Length",)),
            repetition=1,
        )
        with pytest.raises(KeyError, match="unknown model"):
            client_factory(credentials())(cell)


class TestHostedTransports:
    """A provider that is not Azure brings its own endpoint and its own key."""

    def test_a_hosted_transport_uses_its_own_key_and_not_the_azure_one(self):
        creds = credentials()
        assert creds.key_for("unsloth") == "unsloth-not-real"
        assert creds.key_for("mistral_api") == "mistral-not-real"
        assert creds.key_for("openai") == creds.api_key
        assert creds.endpoint_for("unsloth") != creds.endpoint

    def test_a_hosted_transport_without_credentials_refuses_to_build(self):
        """Sending the Azure key to another provider authenticates nothing,
        so the failure has to name what is missing rather than 401 later."""
        bare = Credentials.from_env({
            "OOLD_BENCH_API_KEY": "k",
            "OOLD_BENCH_ENDPOINT": "https://example.invalid",
        })
        with pytest.raises(RuntimeError, match="UNSLOTH_API_ENDPOINT"):
            bare.endpoint_for("unsloth")
        assert bare.key_for("unsloth") == "k"


class TestNarrowingToTheTransportsInUse:
    """A grid on a self-hosted endpoint should not need an Azure account.

    Demanding every credential puts a subscription between a third party and
    a reproduction of a figure measured on a model they can serve themselves.
    """

    def test_a_self_hosted_grid_needs_no_azure_pair(self):
        from oold_llm_bench.clients import Credentials

        env = {"VLLM_9B_ENDPOINT": "https://host/v1", "VLLM_API_KEY": "k"}
        assert Credentials.from_env(env, transports=["vllm_9b"]).endpoint_for("vllm_9b") == "https://host/v1"

    def test_an_azure_transport_still_needs_it(self):
        import pytest

        from oold_llm_bench.clients import Credentials

        with pytest.raises(RuntimeError, match="OOLD_BENCH_API_KEY"):
            Credentials.from_env({"VLLM_9B_ENDPOINT": "x", "VLLM_API_KEY": "k"}, transports=["openai"])

    def test_a_named_hosted_transport_without_its_pair_is_named(self):
        import pytest

        from oold_llm_bench.clients import Credentials

        with pytest.raises(RuntimeError, match="VLLM_9B_ENDPOINT, VLLM_API_KEY"):
            Credentials.from_env({}, transports=["vllm_9b"])

    def test_a_transport_the_catalogue_cannot_name_is_refused(self):
        import pytest

        from oold_llm_bench.clients import Credentials

        with pytest.raises(RuntimeError, match="unknown transport"):
            Credentials.from_env({}, transports=["vllm_70b"])

    def test_asking_for_nothing_in_particular_keeps_the_old_contract(self):
        """Unnamed, only the Azure pair is required and a hosted pair that
        happens to be set is still picked up."""
        from oold_llm_bench.clients import Credentials

        env = {"OOLD_BENCH_API_KEY": "k", "OOLD_BENCH_ENDPOINT": "https://azure/"}
        assert Credentials.from_env(env).hosted == {}
