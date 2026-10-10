"""The playground end to end, driving the real interface in a real browser.

Opt-in and never part of ``make ci``. It needs the ``playground`` extra, a
browser Playwright has downloaded, and the generated schema.org module:

.. code-block:: bash

    uv sync --extra playground
    uv run playwright install chromium
    OOLD_BENCH_SCHEMAS=/path/to/schemaorg/generated uv run pytest tests/test_playground_ui.py -m ui

The default path answers from the loaded task's own ground truth and calls
nothing. ``OOLD_BENCH_UI_LIVE=1`` switches the interface to the provider,
which costs a call per run and is the only way this reaches a model.

What "settled" means here, because a fixed sleep would be flaky on a slow call
and fast enough to pass on a broken one. Two signals, neither a timer:

* the published turn count goes up, which happens after the graph pane has
  been rewritten and never before. Waiting on the run status instead would
  read the previous answer back on a second submission, because the status is
  still ``done`` from the first.
* the renderer's canvas, read as a data URL, is identical across two
  consecutive polls and differs from the one taken before the run. vis-network
  stops redrawing once its layout stabilises, so an unchanging image is the
  picture having settled, and a changed one is something having been drawn.

The screenshot is the artefact a human looks at when this fails. The assertion
is the node and edge sets read back out of the page.

What is asserted depends on who answered. Offline the answer is the ground
truth, so the graph can be required to be exactly it. Against a model it
cannot: gpt-5-nano reported six entities for a two-entity task, so a test
demanding a perfect answer could never pass live and would say nothing about
the interface.

The live assertions are therefore the ones the interface owes for any answer.
Every class drawn was one the catalogue offered, every link is a property the
offered classes declare, an entity pointing at itself is drawn and is not
counted as a link, and the graph after two turns is the two turns and not the
second one. Whether the model called Andrea a Person, and whether the range of
the link it chose admits the class at the far end, are printed: those are what
the benchmark's score is for, and a test asserting them would be a sampler
rather than a check.
"""

from __future__ import annotations

import json
import os
import socket
from collections import Counter
from pathlib import Path

import pytest

from oold_llm_bench.corpus.linked_articles import LINKS_CACHE
from oold_llm_bench.experiments.corpora import DOCUMENTS_CACHE
from oold_llm_bench.grading.triples import Reference, normalise_property

pytest.importorskip("panel")
pytest.importorskip("panelini")
pytest.importorskip("playwright.sync_api")

from oold_llm_bench.playground.corpora import SCHEMAS_ENV, load_schemaorg, paste_task, schemaorg_tasks
from oold_llm_bench.playground.graph import NAME_KEYS
from oold_llm_bench.playground.identity import CLOSE_MATCH, EXACT_MATCH

SCHEMAS = os.environ.get(SCHEMAS_ENV)
LIVE = os.environ.get("OOLD_BENCH_UI_LIVE") == "1"
ARTIFACTS = os.environ.get("OOLD_BENCH_UI_ARTIFACTS")
MODEL = os.environ.get("OOLD_BENCH_UI_MODEL", "gpt-5-nano")

pytestmark = [
    pytest.mark.ui,
    pytest.mark.skipif(
        not (SCHEMAS and Path(SCHEMAS).is_dir()),
        reason=f"set {SCHEMAS_ENV} to the generated schema.org module",
    ),
]

SETTLE_POLLS = 2
"""How many consecutive identical readings count as a settled picture.

One is not enough: a single reading can catch the layout at a frame that
happens to repeat, and the run costs nothing extra to confirm.
"""

POLL_MS = 400
POLL_LIMIT = 60

PEOPLE_AND_FIRMS = ("Person", "Organization")
"""The named class sets the pasted documents here are offered.

Two sets and not one. They are branches of Thing, and a sentence about
employment crosses both: Person holds one class, Organization five, and the
only single set holding both is `all`, whose 122 classes union into an answer
schema of several hundred slots that a small model fills with nulls.
"""

_CANVAS = """
() => {
  const find = (root) => {
    for (const el of root.querySelectorAll('*')) {
      if (el.tagName === 'CANVAS') return el;
      if (el.shadowRoot) { const hit = find(el.shadowRoot); if (hit) return hit; }
    }
    return null;
  };
  const canvas = find(document);
  return canvas ? canvas.toDataURL() : null;
}
"""
"""The renderer's own pixels.

Written as a walk rather than a selector because the component sits several
open shadow roots deep, and the canvas is the only place vis-network's state
is observable at all: it draws nodes, it does not create elements for them.
"""


def _free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


@pytest.fixture(scope="module")
def corpus():
    return load_schemaorg()


@pytest.fixture(scope="module")
def expected_task(corpus):
    """The task the interface loads first, rebuilt from the same seed.

    Generated here rather than read off the page, so the expectation comes
    from ground truth and not from whatever the interface happened to show.
    """
    return schemaorg_tasks(corpus, count=1, seed=1)[0]


@pytest.fixture(scope="module")
def server():
    from oold_llm_bench.playground.app import serve

    port = _free_port()
    running = serve(port=port, show=False, threaded=True, address="127.0.0.1")
    yield f"http://127.0.0.1:{port}"
    running.stop()


@pytest.fixture
def opened(server, page):
    """A loaded page on the provider or the replay client, graph still blank."""
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.goto(server, wait_until="networkidle")
    page.wait_for_selector('[data-testid="run-status"]', timeout=60_000)
    if LIVE:
        page.locator(".pg-client select").select_option(index=1)
        page.locator(".pg-model select").select_option(MODEL)
    return page.evaluate(_CANVAS)


@pytest.fixture
def ready(opened, page):
    """A page with the first corpus task in the chat box."""
    page.locator(".pg-load button").first.click()
    page.wait_for_selector('[data-testid="run-status"][data-status="loaded"]', timeout=30_000)
    return opened


def _graph_state(page) -> dict:
    return json.loads(page.locator('[data-testid="graph-state"] code').inner_text())


def _turns(page) -> int:
    return int(page.locator('[data-testid="graph-state"]').get_attribute("data-turns") or 0)


def _choose(page, *, orchestration: str | None = None, sets: tuple[str, ...] = (), judge: str | None = None) -> None:
    """Set the controls and wait for the server to have them.

    The wait is the point. A choice reaches the browser at once and the server
    a round trip later, so clicking Send straight after selected the catalogue
    in the page and ran the previous one, which for a paste is all 122 classes.
    """
    if orchestration is not None:
        page.locator(".pg-orchestration select").select_option(orchestration)
    if sets:
        labels = page.locator(".pg-catalogue select option").all_inner_texts()
        wanted = [label for label in labels if label.rsplit(" (", 1)[0] in sets]
        assert wanted, f"none of {sets} is offered among {labels}"
        page.locator(".pg-catalogue select").select_option(wanted)
    if judge is not None:
        page.locator(".pg-judge select").select_option(judge)

    wants = {"data-orchestration": orchestration, "data-sets": ",".join(sorted(sets)) if sets else None}
    wants["data-judge"] = judge
    for name, value in wants.items():
        if value is not None:
            page.wait_for_selector(f'[data-testid="condition"][{name}="{value}"]', timeout=30_000)


INPUT = ".chat-interface-input-widget textarea"
"""Panel's own class for the chat input.

A class of ours does not survive: the interface rebuilds its input row and
sets ``css_classes`` on both the row and the widget itself, so anything passed
in is replaced.
"""


def _paste(page, text: str) -> None:
    page.locator(INPUT).first.fill(text)


def _submit_and_settle(page, blank: str | None) -> dict:
    expected = _turns(page) + 1
    page.get_by_role("button", name="Send").click()
    # Selectors and not a page function: Panel renders each pane into its own
    # shadow root, which `document.querySelector` does not reach and a
    # Playwright selector does.
    page.wait_for_selector(
        f'[data-testid="graph-state"][data-turns="{expected}"], [data-testid="run-status"][data-status="error"]',
        timeout=300_000,
    )

    status = page.locator('[data-testid="run-status"]').get_attribute("data-status")
    assert status != "error", page.locator('[data-testid="run-status"]').inner_text()

    state = _graph_state(page)
    assert state["nodes"], "the answer drew no node, so there is nothing to look at"

    previous = None
    agreed = 0
    for _ in range(POLL_LIMIT):
        page.wait_for_timeout(POLL_MS)
        current = page.evaluate(_CANVAS)
        # The reading has to differ from the one taken before the run as well
        # as repeat. vis-network draws a frame or two after the state pane is
        # rewritten, and two identical readings of the old picture are a
        # settled picture of the wrong answer.
        if current is not None and current == previous and current != blank:
            agreed += 1
            if agreed >= SETTLE_POLLS - 1:
                break
        else:
            agreed = 0
        previous = current
    else:
        # Not a failure. The state above already says what was drawn, and a
        # layout that is still moving after the window makes a blurrier
        # screenshot rather than a wrong assertion.
        print(f"the layout was still moving after {POLL_LIMIT * POLL_MS / 1000:.0f}s")

    assert previous != blank, "the state names nodes but the canvas is unchanged, so nothing reached the renderer"
    return state


def _open_tab(page, name: str) -> None:
    page.locator(".bk-tab", has_text=name).first.click()
    page.wait_for_timeout(POLL_MS)


def _shoot(page, tmp_path: Path, name: str) -> Path:
    target = Path(ARTIFACTS) if ARTIFACTS else tmp_path
    target.mkdir(parents=True, exist_ok=True)
    shot = target / name
    page.locator(".pg-graph").first.screenshot(path=str(shot))
    print(f"graph screenshot: {shot}")
    return shot


def _expected_edge(task) -> tuple[str, str, str]:
    for instance in task.expected:
        for prop, value in instance.fields.items():
            if isinstance(value, Reference):
                return instance.key, value.key, normalise_property(prop)
    pytest.fail(f"task {task.id} asserts no edge, so there is nothing to check a graph against")


def _designator(instance) -> str:
    """What the graph will have labelled this expected instance by.

    A stated name where the instance states one. Otherwise its class, which
    is what the stand-in label of an entity nothing named is built from. The
    class alone does not distinguish two instances of one class; `_node_for`
    refuses to pick between them rather than guessing.
    """
    held = {normalise_property(prop): value for prop, value in instance.fields.items()}
    for key in NAME_KEYS:
        found = held.get(key)
        if isinstance(found, str) and found.strip():
            return found.strip()
    return str(instance.class_path).rsplit("/", 1)[-1]


def _nodes_for(state: dict, wanted: str) -> list[str]:
    """Every node whose label names this thing, in id order."""
    return sorted(key for key, label in state["labels"].items() if wanted.casefold() in str(label).casefold())


def _node_for(state: dict, wanted: str) -> str:
    """The one node that names this thing.

    Several is a finding about the answer and not about the picture: reporting
    one entity three times is what an extractor does wrong, and the graph
    showing three nodes is it doing right. The caller decides whether that
    fails its own assertion; this only refuses to pick one of the three.
    """
    hits = _nodes_for(state, wanted)
    assert hits, f"no node is labelled {wanted!r}: {state['labels']}"
    assert len(hits) == 1, f"{wanted!r} was reported {len(hits)} times, as {hits}"
    return hits[0]


def _links_between(state: dict, source: str, target: str) -> list[str]:
    found = []
    for link in state["links"]:
        head, tail, prop = link.split("|")
        if head == source and tail == target:
            found.append(prop)
    return found


def _ranges(task) -> dict[str, set[str]]:
    return {normalise_property(name): set(admitted) for name, admitted in (task.property_ranges or {}).items()}


def _is_declared(task, prop: str) -> bool:
    """Whether the offered classes declare a link property by this name.

    Read off the task's own ranges rather than off a list written here, so a
    model that picks a different but defensible link passes and a model that
    invents a property name does not.
    """
    return normalise_property(prop) in _ranges(task)


def _admits(task, prop: str, class_path: str | None) -> bool:
    """Whether the property's range admits the class at the far end.

    Reported and not asserted. The decode constraint pins a reference slot to
    the ids the plan shortlisted a compatible class for, and the fill step may
    then answer with a narrower class than the shortlist held, so a range the
    answer violates is the agent's behaviour rather than the picture's.
    """
    admitted = _ranges(task).get(normalise_property(prop))
    if admitted is None:
        return False
    if class_path is None:
        return True
    lineage = task.class_parents or {}
    seen, queue = set(), [class_path]
    while queue:
        current = queue.pop()
        if current in admitted:
            return True
        if current in seen:
            continue
        seen.add(current)
        queue.extend(lineage.get(current, ()))
    return False


class TestACorpusTask:
    """The document the interface loads first, against its own ground truth."""

    def test_the_graph_holds_the_entities_and_the_edge_the_task_asserts(self, ready, expected_task, tmp_path, page):
        state = _submit_and_settle(page, ready)
        _shoot(page, tmp_path, "playground-graph.png")

        source_key, sink_key, prop = _expected_edge(expected_task)

        if LIVE:
            # A model is not required to answer perfectly, only correctly
            # about what it did report. Requiring the exact expected set here
            # is what made this test red by construction: nano reported six
            # entities for a two-entity task and the assertion could not pass.
            assert state["entities"], "the model reported no entity at all"
            assert state["self_loops"] == [], f"an entity points at itself: {state['self_loops']}"
        else:
            # Offline replay echoes the task's own data under the graph's own
            # ids, not the task's keys, so a task key has to be resolved to
            # the id the graph drew it under. `_designator` is what the label
            # will say: a stated name, or the class a stand-in is built from.
            resolved = {instance.key: _node_for(state, _designator(instance)) for instance in expected_task.expected}
            assert sorted(state["entities"]) == sorted(resolved.values()), (
                f"graph holds {state['entities']}, the task expects {sorted(resolved.values())}"
            )
            assert state["dangling"] == [], f"an edge reaches nothing: {state['dangling']}"
            source, sink = resolved[source_key], resolved[sink_key]
            assert f"{source}|{sink}|{prop}" in state["links"], f"the edge is missing from {state['links']}"
            assert state["nodes"] == len(resolved)

    def test_the_score_panel_reports_a_perfect_run_on_a_corpus_task(self, ready, page):
        """Only assertable offline, where the answer is the ground truth."""
        if LIVE:
            pytest.skip("a live model has no guaranteed score to assert")

        _submit_and_settle(page, ready)
        _open_tab(page, "Score")
        reported = page.locator(".pg-score .pg-primary").first.inner_text()
        assert "Primary value F1" in reported
        assert "1.000" in reported

    def test_the_near_dimensions_are_shown_beside_the_strict_ones(self, ready, page):
        """`class_near`, `value_near_property` and `property_near` widen a strict
        dimension with a vocabulary- or lineage-aware reading of a hit; the
        score panel's fixed row order used to name only the three they widen."""
        if LIVE:
            pytest.skip("a live model has no guaranteed score to assert")

        _submit_and_settle(page, ready)
        _open_tab(page, "Score")
        table = page.locator(".pg-score .pg-table").first.inner_text()
        for dimension in ("value_near_property", "class_near", "property_near"):
            assert dimension in table, f"{dimension} is missing from the score panel: {table}"

    def test_the_cost_panel_separates_the_plan_call_from_the_fill_calls(self, ready, page):
        """A segmented run is one plan call and one fill call per shortlist."""
        _submit_and_settle(page, ready)
        _open_tab(page, "Cost")
        table = page.locator(".pg-cost .pg-table").first.inner_text()
        assert "plan" in table
        assert "fill" in table

    def test_the_schema_panel_says_how_much_of_the_schema_was_sent(self, ready, page):
        _submit_and_settle(page, ready)
        _open_tab(page, "Schema sent")
        table = page.locator(".pg-schema .pg-table").first.inner_text()
        assert "entity properties after catalogue trim" in table
        if not LIVE:
            # The fidelity is read off the degradation the agent recorded, and
            # a segmented run whose plan step came back empty never reaches the
            # fill step that records one.
            assert "schema fidelity" in table


NEW_CORPUS_PREFIX = {
    "wikidata-schemaorg": "wds-",
    "linked-articles": "wdl-",
    "sequence": "pg-sequence-",
}
"""The three sources built this session, and the id every one of their own
tasks starts with. Checked against the task selector rather than trusted,
because nothing else on screen says which corpus a loaded task came from."""

needs_wikidata_documents = pytest.mark.skipif(
    not DOCUMENTS_CACHE.is_file(),
    reason=f"needs {DOCUMENTS_CACHE}: scripts/fetch_wikidata_documents.py, or uv sync --extra corpora for the Hub",
)

needs_wikidata_links = pytest.mark.skipif(
    not LINKS_CACHE.is_file(),
    reason=f"needs {LINKS_CACHE}: scripts/harvest_wikidata_links.py",
)


def _select_corpus_and_wait(page, name: str) -> list[str]:
    """Choose a corpus and wait for the task selector to offer its own tasks.

    Nothing reflects a corpus change the way the condition pane reflects
    every other control, so this polls the task list itself: clicking Load
    straight after selecting the corpus loaded the previous corpus's first
    task on a slow rerender, the same race `_choose` already guards against
    for the condition pane.
    """
    page.locator(".pg-corpus select").select_option(name)
    prefix = NEW_CORPUS_PREFIX[name]
    ids: list[str] = []
    for _ in range(POLL_LIMIT):
        ids = [v for v in page.locator(".pg-task select option").all_inner_texts() if v]
        if ids and all(task_id.startswith(prefix) for task_id in ids):
            return ids
        page.wait_for_timeout(POLL_MS)
    pytest.fail(f"the task selector never settled on {name}'s own tasks: {ids}")


def _load_and_settle(page, blank, task_id: str | None = None) -> dict:
    """Load a task, by id where one is named, and submit it."""
    if task_id is not None:
        page.locator(".pg-task select").select_option(task_id)
    page.locator(".pg-load button").first.click()
    page.wait_for_selector('[data-testid="run-status"][data-status="loaded"]', timeout=30_000)
    return _submit_and_settle(page, blank)


class TestTheCorporaBuiltThisSession:
    """Three task sources the selector did not offer before this session.

    Offline throughout: every one of these corpora carries its own ground
    truth, so the replay client answers all three the way it answers the
    default schema.org corpus, and nothing here needs a live model or
    `OOLD_BENCH_UI_LIVE`.
    """

    @pytest.mark.parametrize(
        "corpus_name",
        [
            pytest.param("wikidata-schemaorg", marks=needs_wikidata_documents),
            pytest.param("linked-articles", marks=[needs_wikidata_documents, needs_wikidata_links]),
            "sequence",
        ],
    )
    def test_the_corpus_is_offered_and_its_first_task_draws_a_graph(self, opened, page, corpus_name):
        assert corpus_name in page.locator(".pg-corpus select option").all_inner_texts()
        _select_corpus_and_wait(page, corpus_name)
        state = _load_and_settle(page, opened)
        assert state["nodes"], f"{corpus_name}'s first task drew no node"

    @needs_wikidata_documents
    @needs_wikidata_links
    def test_a_linked_articles_task_resolves_its_edge_in_one_turn(self, opened, page):
        """The edge is cross-document because of what it reaches past: the
        document is one article's lead, and the link it states names an
        entity whose own record lives in a different article. Offline replay
        answers both ends from this one task's own ground truth, so the edge
        the graph draws is a resolved link and not a dangling one, after one
        submission and with no second document loaded.
        """
        _select_corpus_and_wait(page, "linked-articles")
        state = _load_and_settle(page, opened)
        assert state["links"], "no link was drawn at all"
        assert state["dangling"] == [], f"a cross-document link was left dangling: {state['dangling']}"

    def test_stepping_through_a_sequence_grows_one_graph_and_asks_identity(self, opened, page):
        """Loading each step and sending it in turn is the corpus's own
        natural use: the turn count advances one at a time and an identity
        comparison is made at the second step, which is the thing the
        Identity tab exists to show rather than a log line.
        """
        ids = _select_corpus_and_wait(page, "sequence")
        assert len(ids) >= 2, f"only one step was offered: {ids}"

        first = _load_and_settle(page, opened, ids[0])
        assert first["turns"] == 1

        second = _load_and_settle(page, page.evaluate(_CANVAS), ids[1])
        assert second["turns"] == 2

        _open_tab(page, "Identity")
        identity_table = page.locator(".pg-identity").first.inner_text()
        assert "No two entities have been compared yet" not in identity_table, (
            "stepping to the second document made no identity comparison"
        )


ANDREA = "Andrea works at ExampleCorp"

needs_a_model = pytest.mark.skipif(
    not LIVE,
    reason=(
        "a pasted document has no ground truth, so the replay client has nothing to answer it with "
        "and draws an empty graph; set OOLD_BENCH_UI_LIVE=1"
    ),
)


@needs_a_model
class TestAPastedSentence:
    """Two people, one sentence, and the classes a reader would pick.

    The reported failure was on this document: two entities, no link, and a
    shortlist of ``Thing``, because the interface trimmed the catalogue to 25
    of 122 classes and neither Person nor Organization survived the cut.

    What is asserted is what the interface owes for any answer: a class it
    draws was offered, an entity pointing at itself is not a link, and a link
    is a property the offered classes declare whose range admits the class at
    the other end. Whether gpt-5-nano calls Andrea a Person is the model's
    business and the benchmark's score, so it is printed and not asserted.
    """

    @pytest.mark.parametrize("orchestration", ["single_shot", "select_then_fill", "segmented"])
    def test_the_sentence_is_drawn_as_named_and_typed_entities(self, opened, corpus, tmp_path, page, orchestration):
        _choose(page, orchestration=orchestration, sets=PEOPLE_AND_FIRMS)
        _paste(page, ANDREA)
        state = _submit_and_settle(page, opened)
        shot = _shoot(page, tmp_path, f"pasted-andrea-{orchestration}.png")
        print(f"{orchestration}: {state['labels']}\ntypes {state['types']}")
        print(f"links {state['links']}\nself-loops {state['self_loops']}\n{shot}")

        task = paste_task(corpus, ANDREA, catalogue_set=PEOPLE_AND_FIRMS)
        offered = set(task.catalogue or ())
        # Every class drawn is one the catalogue offered. A class the model
        # invented would be the interface reporting an answer nobody could
        # have given, which is a different failure from a wrong answer.
        claimed = [name for name in state["types"].values() if name is not None]
        assert claimed, f"every entity was drawn without a class: {state['types']}"
        assert set(claimed) <= offered, f"a class nobody offered was drawn: {sorted(set(claimed) - offered)}"

        # A self-loop is the model's own answer and not a drawing mistake: a
        # segmented run offers the plan's ids as an enum and nano answers
        # `worksFor` with the organisation's own id. The graph has to show
        # that, and it must not count it as a link, because an answer about
        # nothing is not an answer with edges.
        for loop in state["self_loops"]:
            source, target, _ = loop.split("|")
            assert source == target
            assert loop not in state["links"]

        for link in state["links"]:
            source, target, prop = link.split("|")
            assert source != target, f"a self-reference was drawn as a link: {link}"
            assert _is_declared(task, prop), f"{prop} is not a link property any offered class declares"
            if not _admits(task, prop, state["types"][target]):
                print(f"{prop} points at a {state['types'][target]}, which its range does not admit")

        # Only an orchestration that hands out ids can express an edge, so the
        # other two are recorded rather than skipped: the difference between
        # them on this document is the thing worth writing down.
        if orchestration == "segmented":
            assert state["links"], "a segmented run expressed no edge at all"


@needs_a_model
class TestTheGraphSurvivesTheAnswer:
    """The path the reported crash took, which is only reachable here.

    The fold placed every edge under the ids it had resolved and an edge with
    an end nothing drew had none, so one answer raised a ``KeyError`` out of
    the chat callback and took the whole response with it. Nothing below the
    interface shows that: the extraction succeeded and the graph was built.

    ``multi_step`` because it is the orchestration that states a relation both
    ways round, so one sentence about employment is two edges between one pair
    of nodes, which is the other thing the picture has to survive.
    """

    def test_the_answer_is_drawn_and_every_edge_has_both_its_ends(self, opened, tmp_path, page):
        _choose(page, orchestration="multi_step", sets=PEOPLE_AND_FIRMS)
        _paste(page, ANDREA)
        state = _submit_and_settle(page, opened)
        shot = _shoot(page, tmp_path, "andrea-multi-step.png")
        print(f"labels {state['labels']}\ntypes {state['types']}")
        print(f"links {state['links']}\ndangling {state['dangling']}\nself-loops {state['self_loops']}")
        print(f"screenshot: {shot}")

        assert state["turns"] == 1
        assert state["entities"], "the model reported no entity at all"

        # The invariant the crash broke. An edge with an end the graph never
        # drew is one the renderer cannot place and the fold could not map,
        # and it reaches here whether the answer left the end out or the
        # extraction and the agent disagreed about which ids exist.
        drawn = set(state["node_ids"])
        for kind in ("links", "dangling", "self_loops", "close_matches"):
            for edge in state[kind]:
                source, target, _ = edge.split("|")
                assert {source, target} <= drawn, f"{kind} edge {edge} reaches outside {sorted(drawn)}"

        # Printed and not asserted: whether the model states the inverse is
        # the answer's business and the benchmark's. That two edges between
        # one pair are drawn apart is settled in the unit tests, which do not
        # need a model to agree to produce the pair.
        pairs = Counter(frozenset(edge.split("|")[:2]) for edge in [*state["links"], *state["dangling"]])
        parallel = {tuple(sorted(pair)): count for pair, count in pairs.items() if count > 1}
        print(f"pairs carrying more than one edge: {parallel}")


ALICE = "Alice works at Example Corp"
BOB = "Bob works at Example Corp"


@needs_a_model
class TestTwoTurnsGrowOneGraph:
    """The thing every earlier check missed: state across turns.

    Every other test submits once. The playground exists to watch a graph grow,
    and nothing asserted that turn two does not clobber turn one, duplicate it,
    or lose the edge from it.

    The organisation is named identically in both turns, so it is decided by
    exact agreement and no judge call is spent on it. The path is asserted and
    not only the node count, because reaching the same picture by spending a
    call on an identical match is a cost rather than a behaviour.
    """

    @pytest.mark.parametrize("orchestration", ["single_shot", "segmented"])
    def test_a_second_person_joins_the_organisation_already_drawn(self, opened, corpus, tmp_path, page, orchestration):
        _choose(page, orchestration=orchestration, sets=PEOPLE_AND_FIRMS, judge=MODEL)
        _paste(page, ALICE)
        first = _submit_and_settle(page, opened)
        _shoot(page, tmp_path, f"two-turns-{orchestration}-1.png")
        drawn = page.evaluate(_CANVAS)
        _paste(page, BOB)
        second = _submit_and_settle(page, drawn)
        shot = _shoot(page, tmp_path, f"two-turns-{orchestration}-2.png")
        print(f"turn one: {first['labels']}\nturn two: {second['labels']}\nscreenshot: {shot}")
        print(f"types {second['types']}\nlinks {second['links']}\nself-loops {second['self_loops']}")
        print(f"identity {second['identity']['counts']} coverage {second['identity']['coverage']}")
        for decision in second["identity"]["decisions"]:
            print(f"  {decision}")

        assert second["turns"] == 2
        for loop in second["self_loops"]:
            assert loop not in second["links"], "a self-reference was counted as an edge"

        # The organisation is found by its label and the people by elimination.
        # Both turns name the employer identically, which is what the merge
        # turns on; a person's own name is not, and nano leaves it out often
        # enough that looking Alice up by name would test the model.
        corp = _node_for(second, "Example Corp")
        people = [key for key in second["entities"] if key != corp]
        assert len(second["entities"]) == 3, f"expected three entities, got {second['labels']}"
        assert len(people) == 2, f"expected two people beside the employer: {second['labels']}"
        # Typed, and typed from the catalogue. Which class the model picked is
        # the score's business: nano called all three Organization on one run
        # and Person, Person, Organization on the next.
        offered = set(paste_task(corpus, ALICE, catalogue_set=PEOPLE_AND_FIRMS).catalogue or ())
        assert set(second["types"].values()) <= offered, f"a class nobody offered was drawn: {second['types']}"

        merges = [d for d in second["identity"]["decisions"] if d["outcome"] == EXACT_MATCH]
        assert merges, f"nothing merged, so the organisation was drawn twice: {second['identity']}"
        assert all(d["route"] == "agreement" for d in merges), f"an identical name spent a judge call: {merges}"

        if orchestration == "segmented":
            for person in people:
                assert _links_between(second, person, corp), (
                    f"{second['labels'][person]} has no edge: {second['links']}"
                )

    @pytest.mark.parametrize("orchestration", ["single_shot", "segmented"])
    def test_every_deferred_pair_is_drawn_as_a_relation(self, opened, page, orchestration):
        """A closeMatch keeps both nodes and asserts a relation between them.

        Whether the judge defers on these two sentences is the judge's
        business, so the count is not fixed here. What is fixed is that a
        deferral and an edge are the same thing: one drawn without the other
        would leave a question nobody can see or a relation nobody asked for.
        """
        _choose(page, orchestration=orchestration, sets=PEOPLE_AND_FIRMS, judge=MODEL)
        _paste(page, ALICE)
        _submit_and_settle(page, opened)
        _paste(page, BOB)
        state = _submit_and_settle(page, page.evaluate(_CANVAS))

        deferred = {(d["left"], d["right"]) for d in state["identity"]["decisions"] if d["outcome"] == CLOSE_MATCH}
        print(f"judge decisions: {state['identity']['counts']}, coverage {state['identity']['coverage']}")
        assert len(state["close_matches"]) == len(deferred)
        assert all(link.endswith(f"|{CLOSE_MATCH}") for link in state["close_matches"])
