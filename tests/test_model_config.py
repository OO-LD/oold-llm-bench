"""Tests for the model catalogue: generic facts in, account details out.

No other test catches an account identifier leaking into the committed
catalogue, so the leak tests come first.
"""

import json

import pytest

from oold_llm_bench.config import LADDER, drifted, entry_for, load_ladder, spec_for


class TestWhatMayBeCommitted:
    def test_no_entry_has_a_field_for_an_account_identifier(self):
        """The catalogue is public. Deployments and endpoints are not.

        Checked on the field names, not on the serialised text, so a note
        that mentions an endpoint does not read as one.
        """
        fields = set(LADDER[0].__dict__)
        assert fields.isdisjoint({
            "endpoint",
            "deployment",
            "subscription",
            "resource_group",
            "api_key",
        })

    def test_no_entry_names_a_resource_or_a_region_host(self):
        text = json.dumps([e.__dict__ for e in LADDER]).lower()
        for marker in (".openai.azure.com", ".services.ai.azure.com", "-rg", "http"):
            assert marker not in text

    def test_a_published_spec_drops_what_the_caller_supplied(self):
        spec = spec_for(entry_for("gpt-5-nano"), deployment="some-deployment", endpoint="https://x")
        published = json.dumps(spec.publish())
        assert "some-deployment" not in published
        assert "https://x" not in published

    def test_a_published_spec_keeps_what_reproduces_the_run(self):
        published = spec_for(entry_for("gpt-5-nano")).publish()
        assert published["model"] == "gpt-5-nano"
        assert published["model_version"] == "2025-08-07"


class TestTheLadder:
    def test_every_entry_pins_a_version(self):
        assert all(e.model_version for e in LADDER)

    def test_every_entry_declares_a_provider_profile(self):
        from oold.agent.provider import PROFILES

        assert all(e.provider_profile in PROFILES for e in LADDER)

    def test_the_ladder_spans_more_than_one_size(self):
        """H3 asks whether constraint helps most where the model is weakest."""
        assert len({e.tier for e in LADDER}) >= 3

    def test_the_ladder_spans_more_than_one_family(self):
        assert len({e.family for e in LADDER}) >= 4

    def test_model_names_are_unique(self):
        names = [e.model for e in LADDER]
        assert len(names) == len(set(names))

    def test_an_unknown_model_is_refused(self):
        with pytest.raises(KeyError, match="unknown model"):
            entry_for("gpt-4")

    def test_the_ladder_can_be_filtered_by_tier(self):
        assert {s.model for s in load_ladder(tiers=("frontier",))} == {"claude-sonnet-5"}

    def test_the_ladder_can_be_filtered_by_family(self):
        assert all(s.model.startswith("gpt-5") for s in load_ladder(families=("gpt-5",)))

    def test_a_ladder_with_no_resolver_reaches_nothing(self):
        """A dry run names models and versions and calls no provider."""
        assert all(s.endpoint is None for s in load_ladder())

    def test_a_resolver_supplies_the_account_details(self):
        specs = load_ladder(
            lambda e: {"endpoint": "https://example", "region": "swedencentral"},
            tiers=("frontier",),
        )
        assert specs[0].endpoint == "https://example"
        assert specs[0].region == "swedencentral"

    def test_temperature_is_recorded_even_at_the_default(self):
        assert spec_for(entry_for("gpt-5-nano")).temperature == 0.0

    def test_a_model_that_rejects_temperature_records_none(self):
        """claude-sonnet-5 returns 400 on it. Recording 0.0 would be a lie."""
        assert entry_for("claude-sonnet-5").accepts_temperature is False
        assert spec_for(entry_for("claude-sonnet-5")).temperature is None

    def test_a_deployment_that_drops_the_system_message_is_recorded(self):
        """Measured 2026-09-27: it accepts one, returns 200 and discards it."""
        assert entry_for("phi-4-mini-instruct").system_role is False

    def test_every_other_model_keeps_the_system_message(self):
        others = [e for e in LADDER if e.model != "phi-4-mini-instruct"]
        assert all(e.system_role for e in others)


class TestVersionDrift:
    def test_a_matching_version_is_not_drift(self):
        assert drifted({"gpt-5-nano": "2025-08-07"}) == {}

    def test_a_changed_version_is_reported_with_both_values(self):
        assert drifted({"gpt-5-nano": "2026-01-01"}) == {"gpt-5-nano": ("2025-08-07", "2026-01-01")}

    def test_a_model_the_provider_does_not_report_is_not_drift(self):
        assert drifted({}) == {}

    def test_drift_in_one_model_does_not_hide_another(self):
        changes = drifted({"gpt-5-nano": "x", "claude-sonnet-5": "y"})
        assert set(changes) == {"gpt-5-nano", "claude-sonnet-5"}

    def test_a_model_outside_the_ladder_is_ignored(self):
        assert drifted({"some-other-model": "1"}) == {}
