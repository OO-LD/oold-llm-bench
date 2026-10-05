"""What an arm actually sends, and what comes back, as a document.

A pipeline described in prose is a pipeline nobody can check. The thing a
reader needs is the messages, the schema the request carried, and the reply,
for every step, on a real task.

Nothing in a result record can supply that. :class:`~oold.agent.client.Call`
keeps the step, the model, the usage and a hash of the schema, which is what a
cost table needs and not what a reader needs: the messages and the schema body
are never stored, because storing them for every cell of a grid would be
hundreds of megabytes of near-identical prompts.

So a chain is captured deliberately, on one task, by wrapping the client. The
protocol the agent calls through is small enough that a recorder is a
pass-through, and the agent cannot tell the difference:

    >>> recorder = Recorder(client)                       # doctest: +SKIP
    >>> result = agent(arm, recorder).run(request)        # doctest: +SKIP
    >>> print(render(chain_of(result, recorder, task)))   # doctest: +SKIP
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oold.agent.client import ChatResponse, Message

__all__ = ["Chain", "Exchange", "Recorder", "chain_of", "excerpt", "render"]

LIMIT = 1600
"""Characters of a message shown before it is cut.

A described catalogue is twelve thousand characters of the same shape, and a
printed union is forty-five thousand. Showing all of it hides the one thing
the page is for, which is what differs between two arms.
"""


@dataclass(frozen=True)
class Exchange:
    """One call, as it went out and as it came back."""

    index: int
    model: str
    system: str
    user: str
    response_format: dict[str, Any] | None
    strict: bool
    reply: str
    step: str = "?"
    """Filled in from the call log afterwards. The client is not told which
    step it is serving, and inventing a name here would be a guess."""


class Recorder:
    """A :class:`~oold.agent.client.ChatClient` that keeps what it was asked.

    Wraps rather than replaces, so the chain is a recording of a real call to
    a real provider and not a reconstruction of one.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self.model: str = client.model
        """An attribute and not a property, because the protocol the agent
        type-checks against declares one it may write to."""
        self.exchanges: list[Exchange] = []

    def invoke(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        strict: bool = False,
    ) -> ChatResponse:
        reply = self._client.invoke(messages, response_format=response_format, strict=strict)
        roles = {message.role: message.content for message in messages}
        self.exchanges.append(
            Exchange(
                index=len(self.exchanges),
                model=self.model,
                system=roles.get("system", ""),
                user=roles.get("user", ""),
                response_format=response_format,
                strict=strict,
                reply=reply.text,
            )
        )
        return reply


@dataclass
class Chain:
    """One task through one pipeline, step by step."""

    pipeline: str
    arm: str
    label: str
    model: str
    document: str
    expected: list[dict[str, Any]]
    exchanges: list[Exchange]
    produced: Any = None
    catalogue_size: int = 0
    """How many classes were on offer. Stated because a page excerpts its
    prompts, so the reader cannot count them and the number changes what the
    call costs more than anything else on the page."""
    scores: dict[str, float] = field(default_factory=dict)
    attribution: str = ""
    """Where the document comes from and under what licence, carried because
    a page quoting a Wikipedia sentence has to say so."""


def chain_of(
    result: Any,
    recorder: Recorder,
    task: Any,
    *,
    pipeline: str,
    arm: str,
    label: str,
    catalogue_size: int = 0,
) -> Chain:
    """Pair the recorded calls with the steps the agent logged them under.

    Zipped by position: the call log and the recorder see the same calls in
    the same order, and a mismatch in length means one of them dropped a call,
    which is worth knowing rather than papering over.
    """
    calls = list(getattr(result, "calls", []) or [])
    if calls and len(calls) != len(recorder.exchanges):
        raise ValueError(
            f"{len(calls)} calls logged against {len(recorder.exchanges)} recorded; "
            "one of them is not seeing every call"
        )
    exchanges = (
        [
            Exchange(**{**vars(exchange), "step": call.step})
            for exchange, call in zip(recorder.exchanges, calls, strict=True)
        ]
        if calls
        else recorder.exchanges
    )
    return Chain(
        pipeline=pipeline,
        arm=arm,
        label=label,
        model=recorder.model,
        document=task.document,
        expected=[
            {"key": i.key, "class": i.class_path, "fields": {k: str(v) for k, v in i.fields.items()}}
            for i in task.expected
        ],
        exchanges=exchanges,
        produced=getattr(result, "payload", None),
        catalogue_size=catalogue_size,
        attribution=str(task.notes or ""),
    )


def excerpt(text: str, limit: int = LIMIT) -> str:
    """Cut a long message and say what was cut.

    The count is in the document rather than an ellipsis, so a reader can tell
    a prompt that is mostly catalogue from one that is mostly instruction.
    """
    if len(text) <= limit:
        return text
    return (
        f"{text[:limit]}\n\n[... {len(text) - limit} of {len(text)} characters cut, the rest is more of the same ...]"
    )


def render(chain: Chain, *, limit: int = LIMIT) -> str:
    """One pipeline on one task, as a page.

    Every step shows what went out, what constrained it, and what came back.
    Nothing is reconstructed: the strings below are the ones the provider saw.
    """
    out = [
        f"## {chain.pipeline} under `{chain.label}`",
        "",
        f"`{chain.model}`, {len(chain.exchanges)} call(s), "
        f"{chain.catalogue_size} classes on offer. Arm `{chain.arm}` in the records.",
        "",
        "### Document",
        "",
        "```",
        excerpt(chain.document, limit),
        "```",
        "",
    ]
    if chain.attribution:
        out += [f"{chain.attribution}", ""]
    out += ["### Expected", "", "```json", json.dumps(chain.expected, indent=2, ensure_ascii=False), "```", ""]

    for exchange in chain.exchanges:
        out += [
            f"### Step {exchange.index + 1}: {exchange.step}",
            "",
            f"**system message** ({len(exchange.system)} chars)",
            "",
            "```",
            excerpt(exchange.system, limit),
            "```",
            "",
            "**user message**",
            "",
            "```",
            excerpt(exchange.user, limit),
            "```",
            "",
        ]
        if exchange.response_format is None:
            out += ["**enforced schema**: none", ""]
        else:
            body = json.dumps(exchange.response_format, indent=2, ensure_ascii=False)
            out += [
                f"**enforced schema** ({len(body)} chars, "
                f"{'as a grammar' if exchange.strict else 'as a request-body field'})",
                "",
                "```json",
                excerpt(body, limit),
                "```",
                "",
            ]
        out += ["**reply**", "", "```", excerpt(exchange.reply, limit), "```", ""]

    out += [
        "### Produced",
        "",
        "```json",
        excerpt(json.dumps(chain.produced, indent=2, ensure_ascii=False), limit),
        "```",
        "",
    ]
    if chain.scores:
        out += ["### Scored", "", "| dimension | F1 |", "|---|---|"]
        out += [f"| {name} | {value:.3f} |" for name, value in sorted(chain.scores.items())]
        out.append("")
        # One task. A page shows what a pipeline sends and what came back, and
        # a reader who takes one cell for a measurement has been misled by the
        # page rather than by the pipeline.
        out += [
            "One task, so this is what the pipeline did here and not what it scores.",
            "",
        ]
    return "\n".join(out)
