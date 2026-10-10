class TestAModelThatThinksUnlessTold:
    """A thinking model spends its completion budget before it answers.

    Measured on `qwen3.5-9b` at a 64-token budget: 64 reasoning tokens,
    `finish_reason` "length", `content` None. Every cell comes back empty and
    nothing about it reads as a thinking problem, because the deployment is
    answering each request in milliseconds.
    """

    def test_the_local_qwen_deployments_declare_it(self):
        from oold_llm_bench.config.models import LADDER

        thinking = {e.model for e in LADDER if e.thinks_unless_told}
        assert "qwen3.5-9b" in thinking
        assert "qwen3.5-4b" in thinking

    def test_a_hosted_model_does_not(self):
        """Azure rejects the parameter outright, so it must never be sent."""
        from oold_llm_bench.config.models import LADDER

        by_name = {e.model: e for e in LADDER}
        for name in ("gpt-5-nano", "gpt-5-mini", "claude-haiku-4-5", "llama-3.3-70b-instruct"):
            assert not by_name[name].thinks_unless_told, name

    def test_it_is_silenced_with_no_condition_asking(self):
        assert _decide(reasoning=None, thinks=True) is True

    def test_a_condition_asking_for_thinking_is_obeyed(self):
        assert _decide(reasoning="high", thinks=True) is False

    def test_a_model_that_does_not_think_is_left_alone(self):
        assert _decide(reasoning=None, thinks=False) is False

    def test_a_condition_asking_for_none_silences_anything(self):
        assert _decide(reasoning="off", thinks=False) is True


def _decide(*, reasoning, thinks):
    """What the client decides about the thinking switch."""
    from dataclasses import replace

    from oold_llm_bench.clients.azure import thinking_disabled
    from oold_llm_bench.config.models import LADDER

    entry = replace(next(e for e in LADDER if e.model == "qwen3.5-9b"), thinks_unless_told=thinks)
    return thinking_disabled(entry, reasoning)
