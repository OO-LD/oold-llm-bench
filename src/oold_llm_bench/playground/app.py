"""The interface: a document goes in, a graph and four read-outs come out.

The graph is the part a person looks at and the read-outs are the part that
makes this an evaluation tool rather than a demo. An answer that looks right
in a picture can still have been scored against a schema a quarter of which
never reached the model, repaired twice, and still be invalid; none of that is
visible in the answer and all of it is visible here.

Panel and panelini are imported inside the functions that use them and never
at module scope, so importing this module costs nothing and the benchmark
stays installable without either. This is the only module in the package that
needs them at all.

The app publishes its own state into the DOM under ``data-testid``. A test
that read pixels could only say that something rendered; reading the node and
edge sets back out is what lets it say that the right thing rendered.
"""

from __future__ import annotations

import asyncio
import html
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from oold_llm_bench.config.models import entry_for
from oold_llm_bench.playground.corpora import (
    ALL_CLASSES,
    CORPORA,
    SEQUENCE_DOCUMENTS,
    SchemaCorpus,
    catalogue_sets,
    is_scoreable,
    linked_articles_tasks,
    load_schemaorg,
    paste_task,
    schemaorg_tasks,
    sequence_tasks,
    wiki_tasks,
    wikidata_schemaorg_tasks,
)
from oold_llm_bench.playground.graph import VIS_OPTIONS, Diff, GraphState, build_graph, links_of
from oold_llm_bench.playground.identity import DEFER, MERGE, Judge, ModelJudge, action_for
from oold_llm_bench.playground.panels import (
    cost_rows,
    cost_total,
    document_yaml,
    schema_rows,
    score_rows,
    session_call_rows,
    validation_rows,
)
from oold_llm_bench.playground.replay import ReplayClient
from oold_llm_bench.playground.session import (
    ORCHESTRATIONS,
    Options,
    Outcome,
    arm_names,
    build_cell,
    model_names,
    run_once,
)
from oold_llm_bench.tasks.models import TaskRecord

if TYPE_CHECKING:
    from oold.agent.client import ChatClient

__all__ = ["Playground", "build_app", "serve"]

REPLAY = "replay (offline)"
PROVIDER = "provider (calls the model)"

NO_JUDGE = "none (only exact agreement merges)"

EXAMPLE = (
    "The entry is an image file named Harbour At Dusk, 2.4 MB in size, "
    "and its caption refers to the record of Calder Wynn, a video file "
    "of 00:04:12 uploaded on 2024-03-18."
)

STATE_PANE_HEIGHT = 220
STATE_STYLE = "display:block;white-space:pre;overflow:auto;max-height:190px;font-size:11px;line-height:1.35;tab-size:2"
"""The read-out's own box, written on the element.

Inline because the interface ships no stylesheet: every other read-out is
plain HTML with a class on it for the tests, and one rule in a file of its own
would be a second place to look for the look of one element.
"""


def _table(rows: list[dict[str, Any]], *, empty: str) -> str:
    """One read-out as a table, escaped, with no dependency on pandas."""
    if not rows:
        return f'<p class="pg-empty">{html.escape(empty)}</p>'
    columns = list(rows[0])
    head = "".join(f"<th>{html.escape(str(name))}</th>" for name in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(name, '')))}</td>" for name in columns) + "</tr>"
        for row in rows
    )
    return f"<table class='pg-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def state_html(state: GraphState) -> str:
    """The graph as data, so a test asserts on it instead of on pixels.

    The turn count is an attribute of its own. A second submission leaves the
    status at ``done`` from the first one, so a test waiting on the status
    alone would read the previous answer back and pass on it.

    Indented and scrolled rather than handed to a JSON widget. The payload has
    to stay inside this element, because the attributes here and the text in
    it are what the end-to-end test reads the graph out of and a widget of its
    own would carry neither. On one line it ran off the panel with nothing in
    the graph at all, so the box is bounded in both directions and scrolls.
    """
    payload = json.dumps(state.describe(), indent=2, sort_keys=True)
    return (
        f'<div data-testid="graph-state" data-nodes="{len(state.nodes)}" '
        f'data-edges="{len(state.edges)}" data-turns="{state.turn}">'
        f'<code style="{STATE_STYLE}">{html.escape(payload)}</code></div>'
    )


class Playground:
    """One session: the widgets, the graph state, and the last outcome."""

    def __init__(
        self,
        corpus: SchemaCorpus,
        *,
        options: Options | None = None,
        client_for: Any = None,
        task_count: int = 12,
    ) -> None:
        import panel as pn

        self.pn = pn
        self.corpus = corpus
        self.options = options or Options()
        self.client_for = client_for or default_client
        self.state = GraphState()
        self.tasks: list[TaskRecord] = []
        self.loaded: TaskRecord | None = None
        """The corpus task whose document is in the box.

        Kept so a document that was loaded rather than typed keeps its ground
        truth. A submission that no longer matches it is a pasted document,
        and a pasted document has no answer to score against."""
        self.outcome: Outcome | None = None
        self.task_count = task_count
        self.call_log: list[dict[str, Any]] = []
        """Every call made this session, oldest first, across every turn.

        Appended to and never replaced, unlike the cost table beside it: that
        one is one turn's bill, and this is the session's history of what was
        asked, so a reader can scroll back through it after the turn that made
        a call is no longer the current one.
        """
        self._build_widgets()
        self._load_tasks()

    def _build_widgets(self) -> None:
        pn = self.pn
        from panelini.panels.visnetwork import VisNetwork

        self.orchestration = pn.widgets.Select(
            name="Orchestration",
            options=list(ORCHESTRATIONS),
            value=self.options.orchestration,
            css_classes=["pg-orchestration"],
        )
        self.arm = pn.widgets.Select(name="Arm", options=list(arm_names()), value=self.options.arm)
        self.model = pn.widgets.Select(
            name="Model", options=list(model_names()), value=self.options.model, css_classes=["pg-model"]
        )
        # Named sets and not a count. Twenty-five of 122 leaves the answer in
        # or out by accident; the sets are the ontology's own branches, so a
        # reader asks for Person and Organization rather than for a number.
        # Several at once, because a sentence does not stay inside one branch.
        self.sets = catalogue_sets(self.corpus)
        self.catalogue_choice = pn.widgets.MultiSelect(
            name="Catalogue (pasted documents)",
            options=[f"{name} ({len(members)})" for name, members in self.sets.items()],
            value=[self._set_label(name) for name in self.options.catalogue_set if name in self.sets],
            size=8,
            css_classes=["pg-catalogue"],
        )
        self.shortlist_k = pn.widgets.IntInput(name="Shortlist k", value=self.options.shortlist_k, start=1, end=10)
        self.client_choice = pn.widgets.Select(
            name="Client", options=[REPLAY, PROVIDER], value=REPLAY, css_classes=["pg-client"]
        )
        self.judge_choice = pn.widgets.Select(
            name="Identity judge",
            options=[NO_JUDGE, *model_names()],
            value=self.options.judge_model or NO_JUDGE,
            css_classes=["pg-judge"],
        )
        self.show_classes = pn.widgets.Checkbox(name="Draw class nodes", value=False)
        self.auto_merge_toggle = pn.widgets.Checkbox(
            name="Auto-merge exact match",
            value=True,
            css_classes=["pg-auto-merge-exact-match"],
        )

        self.corpus_choice = pn.widgets.Select(
            name="Corpus", options=list(CORPORA), value=CORPORA[0], css_classes=["pg-corpus"]
        )
        self.task_choice = pn.widgets.Select(name="Task", options=[], css_classes=["pg-task"])
        self.load_button = pn.widgets.Button(name="Load task", button_type="primary", css_classes=["pg-load"])
        self.reset_button = pn.widgets.Button(name="Clear graph", css_classes=["pg-reset"])
        self.corpus_choice.param.watch(lambda event: self._load_tasks(), "value")
        self.load_button.on_click(self._on_load)
        self.reset_button.on_click(self._on_reset)

        self.vis = VisNetwork(nodes=[], edges=[], options=VIS_OPTIONS, sizing_mode="stretch_both", min_height=420)

        self.chat = pn.chat.ChatInterface(
            callback=self._on_message,
            user="You",
            show_rerun=False,
            show_undo=False,
            show_timestamp=False,
            show_reaction_icons=False,
            show_button_name=True,
            callback_exception="verbose",
            widgets=pn.widgets.TextAreaInput(placeholder=EXAMPLE, auto_grow=True, rows=3),
            sizing_mode="stretch_both",
            min_height=320,
        )

        # A class per read-out. Panel keeps every tab in the document and only
        # hides the inactive ones, so a selector that names the table alone
        # reads whichever pane happens to come first.
        self.score_pane = pn.pane.HTML("", sizing_mode="stretch_width", css_classes=["pg-score"])
        self.cost_pane = pn.pane.HTML("", sizing_mode="stretch_width", css_classes=["pg-cost"])
        self.schema_pane = pn.pane.HTML("", sizing_mode="stretch_width", css_classes=["pg-schema"])
        self.validation_pane = pn.pane.HTML("", sizing_mode="stretch_width", css_classes=["pg-validation"])
        self.corpus_pane = pn.pane.HTML(self._corpus_html(), sizing_mode="stretch_width", css_classes=["pg-corpus"])
        self.identity_pane = pn.pane.HTML("", sizing_mode="stretch_width", css_classes=["pg-identity"])
        self.calls_pane = pn.pane.HTML("", sizing_mode="stretch_width", css_classes=["pg-calls"])
        self.documents_pane = pn.pane.HTML("", sizing_mode="stretch_width", css_classes=["pg-documents"])
        self.state_pane = pn.pane.HTML(state_html(self.state), sizing_mode="stretch_width", height=STATE_PANE_HEIGHT)
        self.status_pane = pn.pane.HTML(self._status_html("idle", ""), sizing_mode="stretch_width", height=40)
        # The condition the next submission will run, republished on every
        # change. A reader sees what is about to be sent rather than inferring
        # it from seven controls, and a test can wait for a choice to have
        # reached the server instead of clicking Send into a race.
        self.condition_pane = pn.pane.HTML(self._condition_html(), sizing_mode="stretch_width", height=56)
        for control in (
            self.orchestration,
            self.arm,
            self.model,
            self.catalogue_choice,
            self.shortlist_k,
            self.client_choice,
            self.judge_choice,
            self.auto_merge_toggle,
        ):
            control.param.watch(lambda event: self._publish_condition(), "value")

        self.tabs = pn.Tabs(
            ("Score", self.score_pane),
            ("Cost", self.cost_pane),
            ("Schema sent", self.schema_pane),
            ("Validation", self.validation_pane),
            ("Identity", self.identity_pane),
            ("Calls", self.calls_pane),
            ("Documents", self.documents_pane),
            ("Graph state", self.state_pane),
            ("Corpus", self.corpus_pane),
            sizing_mode="stretch_width",
        )
        self._render_readouts()

    def _set_label(self, name: str) -> str:
        return f"{name} ({len(self.sets[name])})"

    def chosen_sets(self) -> tuple[str, ...]:
        """The named class sets ticked, falling back to every describable one."""
        chosen = tuple(label.rsplit(" (", 1)[0] for label in self.catalogue_choice.value or ())
        return tuple(name for name in chosen if name in self.sets) or (ALL_CLASSES,)

    def current_options(self) -> Options:
        """The choices as they stand, read off the widgets rather than stored."""
        judge = self.judge_choice.value
        return replace(
            self.options,
            orchestration=self.orchestration.value,
            arm=self.arm.value,
            model=self.model.value,
            catalogue_set=self.chosen_sets(),
            shortlist_k=int(self.shortlist_k.value or 1),
            judge_model=None if judge == NO_JUDGE else judge,
        )

    def effective_options(self, task: TaskRecord | None = None) -> Options:
        """The choices as a submission will actually use them.

        Two of the choices are settled past the widget and every read-out of
        them has to say so, or the interface reports a condition nothing ran.
        The catalogue trim keeps the classes an expectation names, and a
        pasted document's expectation is a placeholder, so a paste is never
        trimmed. A judge is a call, so offline replay never asks one.

        Without a task the trim is left as chosen, because which of the two a
        document is cannot be decided before there is one.
        """
        options = self.current_options()
        if task is not None and not is_scoreable(task):
            options = replace(options, catalogue_size=None)
        if self.client_choice.value == REPLAY:
            options = replace(options, judge_model=None)
        return options

    def typed(self) -> str:
        """Whatever is in the chat box, which may be nothing at all."""
        return str(getattr(self.chat.active_widget, "value", "") or "")

    def _load_tasks(self) -> None:
        choice = self.corpus_choice.value
        if choice == "wiki-measurements":
            self.tasks = wiki_tasks(count=self.task_count)
        elif choice == "wikidata-schemaorg":
            self.tasks = wikidata_schemaorg_tasks(count=self.task_count)
        elif choice == "linked-articles":
            self.tasks = linked_articles_tasks(count=self.task_count)
        elif choice == "sequence":
            # A count of documents in the selector, not of sequences: the
            # control the rest of this class exposes is `task_count`, and a
            # second number meaning something else would need its own widget
            # to explain it.
            self.tasks = sequence_tasks(self.corpus, count=max(1, self.task_count // SEQUENCE_DOCUMENTS))
        else:
            self.tasks = schemaorg_tasks(self.corpus, count=self.task_count)
        self.task_choice.options = [task.id for task in self.tasks]
        if self.tasks:
            self.task_choice.value = self.tasks[0].id

    def _task_by_id(self, task_id: str) -> TaskRecord | None:
        return next((task for task in self.tasks if task.id == task_id), None)

    def _on_load(self, _event: Any) -> None:
        task = self._task_by_id(self.task_choice.value)
        if task is None:
            return
        self.loaded = task
        self.chat.active_widget.value = task.document
        # The document is half the condition, so loading one changes what the
        # next submission runs and the pane has to say so.
        self._publish_condition()
        self.status_pane.object = self._status_html("loaded", task.id)

    def _on_reset(self, _event: Any) -> None:
        self.state.clear()
        self.vis.clear()
        self.outcome = None
        self.call_log.clear()
        self.state_pane.object = state_html(self.state)
        self.status_pane.object = self._status_html("idle", "")
        self._render_readouts()

    async def _on_message(self, contents: str, user: str, instance: Any) -> Any:
        """Call the model off the event loop, and touch the interface on it.

        The split is not a nicety. Panel resolves which document a widget
        belongs to from the running context, and a worker thread has none, so
        a pane written from inside the thread reaches whichever session Panel
        guessed. With one session open that guess is right and nothing looks
        wrong; with two, the second one submits and never hears back.
        """
        self.status_pane.object = self._status_html("running", "")
        outcome = await asyncio.to_thread(self.extract, contents)
        # Folding goes off the loop as well, because deciding that two
        # entities are one may call a model, and a call made on the loop
        # freezes every session the server is holding.
        # Guarded, because folding may call a judge and every other provider
        # call in this file promises a failure is shown rather than raised.
        # Unguarded it left the status pane on "running" for good, with the
        # previous turn's read-outs beside a traceback and a turn counter that
        # had advanced for a turn that folded nothing.
        try:
            diff = await asyncio.to_thread(self.fold, outcome)
        except Exception as exc:
            import panel as pn

            self.render(None)
            return pn.chat.ChatMessage(
                f"Extracted, but the graph could not be updated: {type(exc).__name__}: {exc}",
                user="Playground",
                show_timestamp=False,
            )
        self.render(diff)
        return self._message_for(outcome, diff)

    def run(self, document: str) -> Outcome:
        """Extract and fold the result in, for a caller with no event loop."""
        outcome = self.extract(document)
        self.apply(outcome)
        return outcome

    def extract(self, document: str) -> Outcome:
        """Run one document under the current choices. Touches no widget.

        A provider failure is returned on the outcome rather than raised, so
        the interface can show it beside the graph that is already on screen.
        """
        if not document.strip() and self.loaded is None:
            return Outcome(cell=None, error="no document: type or paste something, or load a corpus task")
        task = self._task_for(document)
        options = self.effective_options(task)
        cell = build_cell(task, options)
        try:
            client = self.client_for(cell, task, self.client_choice.value)
            outcome = run_once(cell, client, attempts=options.attempts)
        except Exception as exc:
            return Outcome(cell=cell, error=f"{type(exc).__name__}: {exc}")

        # Logged here and not in `fold`, so a call is on the record even where
        # the document carried nothing to draw or the fold that follows fails.
        self.call_log.extend(session_call_rows(getattr(outcome.result, "calls", None), turn=self.state.turn + 1))

        if outcome.produced is not None:
            links, dangling = links_of(outcome.result)
            # Rebuilt rather than taken from the outcome, because `show_classes`
            # is a choice this interface owns and the session knows nothing
            # about. Every other input has to be passed again for that, the
            # mentions included: without them an entity no property named is
            # drawn under a stand-in even though the plan step read its name.
            outcome.graph = build_graph(
                outcome.produced,
                links=links,
                dangling=dangling,
                mentions=getattr(outcome.result, "mentions", None) or {},
                show_classes=bool(self.show_classes.value),
            )
        return outcome

    def judge(self) -> Judge | None:
        """Whatever decides the identities exact agreement did not, or ``None``.

        Built per answer, from the choice as it stands, so a reader can turn a
        judge on between two turns and see the next comparison change route.
        Offline replay drops it, decided in :meth:`effective_options` so that
        what is reported as the judge is what is asked. ``None`` is a real
        choice and not a placeholder: :func:`~oold_llm_bench.playground.identity.decide`
        reads it as no judge configured and defers through
        :class:`~oold_llm_bench.dedup.Resolver` the same way a graded run with
        no judge does.
        """
        name = self.effective_options().judge_model
        if not name:
            return None
        from oold_llm_bench.clients.azure import Credentials, build_client

        return ModelJudge(client=build_client(entry_for(name), Credentials.from_env()), name=name)

    def fold(self, outcome: Outcome) -> Diff | None:
        """Decide identity and fold one answer into the graph. No widget.

        Separate from :meth:`render` because this is the half that can call a
        model, and the other half is the one that has to run where Panel can
        resolve which document it belongs to.
        """
        self.outcome = outcome
        if outcome.graph is None:
            return None
        self.state.judge = self.judge()
        self.state.auto_merge_exact_match = bool(self.auto_merge_toggle.value)
        return self.state.update(outcome.graph)

    def render(self, diff: Diff | None) -> None:
        """Write the folded state into the interface, and nothing else.

        The status is written last, after everything it describes, so a reader
        waiting for it is waiting for a finished view and not for a running
        one.
        """
        outcome = self.outcome
        if diff is not None:
            if diff.actions:
                self.vis.execute_step({"actions": diff.actions})
            overlays = self.state.overlays()
            if overlays:
                self.vis.update_nodes(overlays)
            for overlay in self.state.edge_overlays():
                self.vis.update_edge(overlay)

        self._render_readouts()
        self.state_pane.object = state_html(self.state)
        if outcome is not None:
            self.status_pane.object = self._status_html("error" if outcome.error else "done", outcome.summary())

    def apply(self, outcome: Outcome) -> Diff | None:
        """Fold and render, for a caller with no event loop to split them on."""
        diff = self.fold(outcome)
        self.render(diff)
        return diff

    def _task_for(self, document: str) -> TaskRecord:
        """The loaded task when the text is still its document, a paste otherwise.

        Editing a loaded document makes it a pasted one. The expectation
        belongs to the words that were generated from it, and keeping it
        across an edit would score an answer against a document nobody sent.
        """
        text = document.strip()
        if self.loaded is not None and text == self.loaded.document.strip():
            return self.loaded
        # The example stands in where a task is built to *describe* the
        # condition and no document has been typed yet. Submitting an empty
        # box is refused in `extract`, because running the example and
        # drawing its graph as the user's answer is the one case where the
        # picture is about a document nobody submitted.
        return paste_task(self.corpus, text.strip() or EXAMPLE, catalogue_set=self.chosen_sets())

    def _message_for(self, outcome: Outcome, diff: Diff | None = None) -> Any:
        pn = self.pn
        if outcome.error or outcome.cell is None:
            # A refusal before a cell is built carries the reason and nothing
            # else, so it reports as a failure rather than as a blank answer.
            return pn.chat.ChatMessage(f"Failed: {outcome.error}", user="Playground", show_timestamp=False)
        lines = [f"**{outcome.cell.condition.key}** on `{outcome.cell.model.model}`", "", outcome.summary()]
        counts = outcome.graph.describe() if outcome.graph else {}
        if counts.get("dangling"):
            lines.append(f"{counts['dangling']} edge(s) point at an entity that was never reported.")
        if counts.get("self_loops"):
            lines.append(f"{counts['self_loops']} edge(s) point an entity at itself, which asserts nothing.")
        # This answer's decisions and not the session's. The ledger is only
        # cleared by `clear()`, so reading it here re-listed turn one's merges
        # on turn five and printed a closeMatch count that grows with the
        # square of the entities. `Diff.decisions` is written for exactly this
        # and was never read.
        # Read through `action_for` and not `d.merges`: the toggle can leave a
        # decision the resolver called `exactMatch` undrawn as a merge, and a
        # message that reported it merged regardless would describe a node
        # the graph never folded.
        decisions = diff.decisions if diff is not None else []
        actions = [(d, action_for(d, auto_merge_exact_match=self.state.auto_merge_exact_match)) for d in decisions]
        merged = [d for d, action in actions if action == MERGE]
        if merged:
            lines.append("Merged as one entity: " + "; ".join(f"{d.right} and {d.left} ({d.route})" for d in merged))
        deferred = [d for d, action in actions if action == DEFER]
        if deferred:
            lines.append(f"{len(deferred)} pair(s) left as skos:closeMatch for a person to resolve.")
        if diff is not None and diff.resolved_placeholders:
            lines.append(f"{len(diff.resolved_placeholders)} entity(ies) a link had only named are now reported.")
        if outcome.shortlist:
            lines.append(f"Shortlist pooled over the fill calls: {', '.join(outcome.shortlist)}")
        return pn.chat.ChatMessage("\n\n".join(lines), user="Playground", show_timestamp=False)

    def _render_readouts(self) -> None:
        outcome = self.outcome
        self.identity_pane.object = self._identity_html()
        self.calls_pane.object = _table(self.call_log, empty="No call has been made yet.")
        self.documents_pane.object = self._documents_html()
        if outcome is None or outcome.cell is None:
            empty = "<p class='pg-empty'>Nothing has been run yet.</p>"
            self.score_pane.object = empty
            self.cost_pane.object = empty
            self.schema_pane.object = empty
            self.validation_pane.object = empty
            return

        if outcome.score is not None:
            heading = f"<p class='pg-primary'>Primary value F1 <b>{outcome.score.primary:.3f}</b></p>"
            self.score_pane.object = heading + _table(score_rows(outcome.score), empty="No dimension was scored.")
        else:
            self.score_pane.object = (
                "<p class='pg-empty'>Pasted documents have no ground truth, so nothing is scored. "
                "Load a corpus task to see the score.</p>"
            )

        rows = cost_rows(getattr(outcome.result, "calls", None))
        note = ""
        if self.client_choice.value == REPLAY:
            note = "<p class='pg-empty'>Offline replay: token counts are a length estimate, not a provider count.</p>"
        self.cost_pane.object = (
            _table(rows, empty="No call was made.") + _table([cost_total(rows) | {"step": "total"}], empty="") + note
        )

        entry = entry_for(outcome.cell.model.model)
        self.schema_pane.object = _table(
            schema_rows(
                getattr(outcome.result, "degradation", None),
                declared=outcome.declared_schema,
                sent=outcome.sent_schema,
                profile=_profile(entry.provider_profile),
            ),
            empty="The arm sent no schema, so nothing was prepared.",
        )
        self.validation_pane.object = _table(
            validation_rows(outcome.result), empty="The condition did not validate the answer."
        )

    def _identity_html(self) -> str:
        """Which entities were compared, what was decided, and by whom.

        A merge that happens silently is indistinguishable from an extractor
        that reported one entity where there were two, so the decision and its
        evidence are on screen rather than in a log.

        The judge named here is the one that will be asked and not the one the
        widget holds. Offline replay drops it, and reporting the dropped
        choice as the judge would credit a deduplication nothing performed.
        """
        ledger = self.state.ledger
        options = self.effective_options()
        judge = options.judge_model
        chosen = self.current_options().judge_model
        notes = []
        if judge and judge == options.model:
            notes.append(
                f"<p class='pg-empty'>The judge and the extractor are both <b>{html.escape(judge)}</b>, "
                "so deduplication and extraction are not separable in this run.</p>"
            )
        if chosen and not judge:
            notes.append(
                f"<p class='pg-empty'>Offline replay makes no call, so <b>{html.escape(chosen)}</b> decides "
                "nothing here. Choose the provider client to put it to work.</p>"
            )
        if not chosen:
            notes.append(
                "<p class='pg-empty'>No judge is configured, so only exact agreement merges and "
                "everything else is left for a person as a closeMatch.</p>"
            )
        summary = ledger.describe()
        counts = summary["counts"]
        heading = (
            f"<p class='pg-primary'>{counts.get('agreement', 0)} decided by agreement, "
            f"{counts.get('judge', 0)} by judge, coverage <b>{summary['coverage']:.2f}</b></p>"
        )
        rows = [
            {
                "a": decision.left,
                "b": decision.right,
                "outcome": decision.outcome,
                "route": decision.route,
                "judge": decision.judge or "",
                "conflicts": ", ".join(decision.conflicts),
                "evidence": decision.evidence,
            }
            for decision in ledger.decisions
        ]
        return heading + "".join(notes) + _table(rows, empty="No two entities have been compared yet.")

    def _documents_html(self) -> str:
        """The entities drawn so far, one YAML document per entity.

        Read off the session's own state and not off the last outcome: the
        graph is the running fold of every turn, and a reader here wants what
        the graph currently holds, not what one answer added.
        """
        text = document_yaml(self.state.documents())
        if not text:
            return "<p class='pg-empty'>No entity has been extracted yet.</p>"
        return f'<pre data-testid="documents" style="{STATE_STYLE}">{html.escape(text)}</pre>'

    def _corpus_html(self) -> str:
        return _table(
            [{"measure": key, "value": value} for key, value in self.corpus.describe().items()],
            empty="No schema collection.",
        )

    def _publish_condition(self) -> None:
        self.condition_pane.object = self._condition_html()

    def _condition_html(self) -> str:
        """The condition the next submission runs under, as data and as words.

        The document decides part of it, so it is read out of the box as it
        stands: a corpus task is trimmed to its catalogue and a paste is not,
        and the pane said ``n25`` over a paste that ran untrimmed.

        Republished when a control moves and when a task is loaded, which is
        every way the condition changes except typing. A paste typed over a
        loaded document therefore still reads as that document until one of
        those happens. The submission is not caught out by it: it resolves the
        document again and runs what it resolves.
        """
        options = self.effective_options(self._task_for(self.typed()))
        sets = ",".join(sorted(options.catalogue_set))
        judge = options.judge_model or "none"
        auto_merge = "true" if self.auto_merge_toggle.value else "false"
        return (
            f'<div data-testid="condition" data-sets="{html.escape(sets)}" '
            f'data-judge="{html.escape(judge)}" data-model="{html.escape(options.model)}" '
            f'data-orchestration="{html.escape(options.orchestration)}" '
            f'data-auto-merge="{auto_merge}">'
            f"{html.escape(options.label())} on {html.escape(options.model)}<br>"
            f"catalogue {html.escape(sets)}, judge {html.escape(judge)}, "
            f"auto-merge exact match {auto_merge}</div>"
        )

    def _status_html(self, status: str, detail: str) -> str:
        return (
            f'<div data-testid="run-status" data-status="{html.escape(status)}">'
            f"<b>{html.escape(status)}</b> {html.escape(detail)}</div>"
        )

    def layout(self) -> Any:
        pn = self.pn
        controls = pn.Column(
            pn.pane.Markdown("### Condition"),
            self.orchestration,
            self.arm,
            self.model,
            self.catalogue_choice,
            self.shortlist_k,
            self.client_choice,
            self.judge_choice,
            self.show_classes,
            self.auto_merge_toggle,
            self.condition_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### Document"),
            self.corpus_choice,
            self.task_choice,
            pn.Row(self.load_button, self.reset_button),
            width=300,
        )
        left = pn.Column(self.chat, self.status_pane, sizing_mode="stretch_both", min_width=380)
        right = pn.Column(
            pn.pane.Markdown("### Extracted graph"),
            pn.Column(self.vis, css_classes=["pg-graph"], sizing_mode="stretch_both", min_height=420),
            # "Graph state" is a tab and not a band under the picture: it is
            # the graph as data, wanted when checking what the picture claims
            # and not while reading the picture itself.
            self.tabs,
            sizing_mode="stretch_both",
            min_width=460,
        )
        return pn.Row(controls, left, right, sizing_mode="stretch_both")

    def view(self) -> Any:
        """The whole interface, ready to be served.

        Returned unmarked rather than through ``Panelini.servable()``. That
        method marks the panel on the current document and returns it, so a
        server handed the return value adds a second copy, and every selector
        on the page then matches twice.
        """
        from panelini import Panelini

        app = Panelini(title="oold-llm-bench playground", sidebar_enabled=False)
        app.main_set(objects=[self.layout()])
        return app


def _profile(name: str) -> Any:
    from oold.agent.provider import profile_for

    return profile_for(name)


def default_client(cell: Any, task: TaskRecord, choice: str) -> ChatClient:
    """The client a choice names.

    Replay is the default, because a playground that spent a call on every
    keystroke would be a poor way to look at a condition and a good way to
    spend a grid's budget.
    """
    if choice == REPLAY:
        return ReplayClient(task)
    from oold_llm_bench.clients.azure import Credentials, build_client

    return build_client(entry_for(cell.model.model), Credentials.from_env(), temperature=cell.model.temperature)


def build_app(
    *,
    schemas: str | None = None,
    options: Options | None = None,
    client_for: Any = None,
    task_count: int = 12,
) -> Playground:
    """Read the schema collection and assemble the interface."""
    return Playground(
        load_schemaorg(schemas),
        options=options,
        client_for=client_for,
        task_count=task_count,
    )


def serve(
    *,
    schemas: str | None = None,
    port: int = 5011,
    show: bool = True,
    options: Options | None = None,
    address: str = "localhost",
    threaded: bool = False,
    task_count: int = 12,
) -> Any:
    """Run the interface on a local server.

    The schema collection is read once, before the server starts, so a missing
    one is reported at the command line instead of in a browser tab.

    ``task_count`` is how many documents the task selector offers, for every
    corpus it is pointed at. Raised past the default it reaches further into a
    corpus ordered by file position rather than balanced across classes, which
    is how a particular entity a caller wants on screen (a sequence's own
    steps, or a linked article whose target is also worth loading on its own)
    gets to be a choice in the dropdown rather than a day's draw away from it.
    """
    import panel as pn

    pn.extension(design="material")
    # Read once and shared. Every browser tab is its own session with its own
    # graph, but re-reading the whole module per tab would make opening one
    # slow for no gain: the collection is frozen and nothing writes to it.
    corpus = load_schemaorg(schemas)
    return pn.serve(
        lambda: Playground(corpus, options=options, task_count=task_count).view(),
        port=port,
        show=show,
        address=address,
        threaded=threaded,
        # Bokeh refuses a websocket whose Origin it was not told about, and a
        # page opened on the other spelling of the same host then renders an
        # empty shell with no error in it.
        websocket_origin=[f"{address}:{port}", f"localhost:{port}", f"127.0.0.1:{port}"],
    )
