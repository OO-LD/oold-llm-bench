"""Capturing what an arm sends, so a pipeline page is a recording.

A pipeline described in prose is a pipeline nobody can check, and no result
record holds the messages or the schema body. These pin that a recorder is a
pass-through, that the steps are named from the call log rather than guessed,
and that a cut says how much it cut.
"""

from __future__ import annotations

import pytest

from oold_llm_bench.chains import Chain, Exchange, Recorder, chain_of, excerpt, render

# The recorder implements the library's client protocol, so the real message
# type is what a test should hand it. The library is an extra, and a module
# that cannot be collected without it would make a core-only run report an
# error where it should report a skip.
Message = pytest.importorskip("oold.agent.client").Message


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.parsed = None


class FakeClient:
    model = "fake-1"

    def __init__(self) -> None:
        self.seen: list[tuple] = []

    def invoke(self, messages, *, response_format=None, strict=False):
        self.seen.append((tuple(m.role for m in messages), response_format, strict))
        return FakeResponse('{"entities": []}')


class TestTheRecorder:
    def test_it_passes_the_call_through_unchanged(self):
        """A chain has to be a recording of a real call, not a reconstruction."""
        client = FakeClient()
        recorder = Recorder(client)
        schema = {"type": "object"}
        reply = recorder.invoke([Message("system", "s"), Message("user", "u")], response_format=schema, strict=True)
        assert reply.text == '{"entities": []}'
        assert client.seen == [(("system", "user"), schema, True)]

    def test_it_keeps_both_turns_and_the_schema(self):
        recorder = Recorder(FakeClient())
        recorder.invoke([Message("system", "catalogue"), Message("user", "document")], response_format={"a": 1})
        (exchange,) = recorder.exchanges
        assert exchange.system == "catalogue"
        assert exchange.user == "document"
        assert exchange.response_format == {"a": 1}
        assert exchange.model == "fake-1"

    def test_an_unconstrained_call_records_no_schema(self):
        recorder = Recorder(FakeClient())
        recorder.invoke([Message("user", "d")])
        assert recorder.exchanges[0].response_format is None
        assert recorder.exchanges[0].system == ""


class TestNamingTheSteps:
    """The client is not told which step it serves, so a name is read from the
    call log rather than invented."""

    def _recorder(self, n: int) -> Recorder:
        recorder = Recorder(FakeClient())
        for _ in range(n):
            recorder.invoke([Message("user", "d")])
        return recorder

    def _task(self):
        return type("Task", (), {"document": "doc", "expected": [], "notes": ""})()

    def test_steps_come_from_the_log(self):
        recorder = self._recorder(2)
        result = type("Result", (), {"calls": [_call("select"), _call("fill")], "payload": {}})()
        chain = chain_of(result, recorder, self._task(), pipeline="select_then_fill", arm="a", label="l")
        assert [e.step for e in chain.exchanges] == ["select", "fill"]

    def test_a_log_that_missed_a_call_is_refused(self):
        """Zipping a short log onto long exchanges would label the wrong call,
        which is worse than saying nothing."""
        recorder = self._recorder(3)
        result = type("Result", (), {"calls": [_call("select")], "payload": {}})()
        with pytest.raises(ValueError, match="not seeing every call"):
            chain_of(result, recorder, self._task(), pipeline="p", arm="a", label="l")

    def test_without_a_log_the_steps_stay_unnamed(self):
        recorder = self._recorder(1)
        result = type("Result", (), {"calls": [], "payload": {}})()
        chain = chain_of(result, recorder, self._task(), pipeline="p", arm="a", label="l")
        assert chain.exchanges[0].step == "?"


def _call(step: str):
    return type("Call", (), {"step": step})()


class TestExcerpting:
    def test_a_short_message_is_untouched(self):
        assert excerpt("abc", 10) == "abc"

    def test_a_cut_says_how_much_it_cut(self):
        """A reader has to be able to tell a prompt that is mostly catalogue
        from one that is mostly instruction."""
        cut = excerpt("x" * 100, 10)
        assert cut.startswith("x" * 10)
        assert "90 of 100 characters cut" in cut


def test_a_page_shows_the_prompt_the_schema_and_the_reply():
    chain = Chain(
        pipeline="single_shot",
        arm="catalog-flat-enforced",
        label="schema-prose-catalog-flat-enforced",
        model="fake-1",
        document="It is 880 metres high.",
        expected=[{"key": "q1", "class": "Altitude", "fields": {"value": "880.0 meter"}}],
        exchanges=[
            Exchange(
                index=0,
                model="fake-1",
                system="Choose a class:\n- Altitude",
                user="It is 880 metres high.",
                response_format={"type": "object"},
                strict=True,
                reply='{"entities": [{"type": "Altitude"}]}',
                step="extract",
            )
        ],
        produced={"entities": [{"type": "Altitude"}]},
        scores={"primary": 1.0},
        attribution="CC BY-SA 4.0",
    )
    page = render(chain)
    assert "Step 1: extract" in page
    assert "Choose a class:" in page
    assert "as a grammar" in page
    assert "880 metres" in page
    assert "CC BY-SA 4.0" in page
    assert "| primary | 1.000 |" in page


def test_an_unconstrained_step_says_so_rather_than_printing_nothing():
    """A page that omits the line cannot be told from one whose arm enforces
    a schema the renderer failed to capture."""
    chain = Chain(
        pipeline="single_shot",
        arm="no-catalog-not-enforced",
        label="no-catalog-not-enforced",
        model="fake-1",
        document="d",
        expected=[],
        exchanges=[
            Exchange(index=0, model="fake-1", system="", user="d", response_format=None, strict=False, reply="{}")
        ],
    )
    assert "**enforced schema**: none" in render(chain)


class TestEveryPipelineHasAPage:
    """A pipeline with no recorded chain is a pipeline nobody can check.

    The pages are generated by `oold-bench chains` and committed, so this
    fails when an orchestration is added and its page is not.
    """

    def _pages(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent / "docs" / "pipelines"
        return {path.stem for path in root.glob("*.md")} - {"index"}

    def test_each_runnable_orchestration_is_documented(self):
        from oold_llm_bench.runner.config import ORCHESTRATIONS

        missing = sorted(set(ORCHESTRATIONS) - self._pages())
        assert not missing, f"no chain recorded for {', '.join(missing)}"

    def test_a_page_shows_a_prompt_a_schema_and_a_reply(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent / "docs" / "pipelines"
        for name in self._pages():
            page = (root / f"{name}.md").read_text(encoding="utf-8")
            assert "**system message**" in page, name
            assert "**enforced schema**" in page, name
            assert "**reply**" in page, name
            assert "### Step 1:" in page, name
