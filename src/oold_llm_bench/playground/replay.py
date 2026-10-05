"""A client that answers from a task's own ground truth, with no provider.

It exists so the interface can be driven, screenshotted and asserted against
without spending a call, and so a test of the view is a test of the view
rather than of whichever model answered that day.

What it is not is a model. It answers each orchestration as well as that
orchestration allows, which is the honest ceiling and not a prediction: under
single-shot there is no id slot, so a link has nothing to point at and is
omitted, exactly as a perfect single-shot answer would have to omit it. Under
a segmented run the plan hands out ids and the link is written as one. The
difference between the two pictures is therefore a property of the
orchestration and not of this file's opinion.

Token counts are a length estimate and are labelled as one wherever they are
shown. A replayed run reproduces the call structure, which is the part worth
looking at, and cannot reproduce a tokeniser.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from oold_llm_bench.grading.triples import Quantity, Reference
from oold_llm_bench.tasks.models import ExpectedInstance, TaskRecord

if TYPE_CHECKING:
    from oold.agent.client import ChatResponse

__all__ = ["CHARS_PER_TOKEN", "ReplayClient"]

CHARS_PER_TOKEN = 4
"""The estimate a replayed run reports tokens under.

Four characters is the usual English rule of thumb. It is stated here rather
than hidden in an expression so nobody reads a replayed cost as a measurement.
"""

_SELECT_MARKER = "which classes it could be"
_SEGMENTED_MARKER = "Give that id as the value of the id property"
_SEGMENTED_IDS = re.compile(r"id given here:\s*(.+?)\.\s", re.DOTALL)
_FILL_MARKER = "report only the entities identified as"
_PROSE_MARKER = "Answer in plain prose"


def _reply(prompt: str, answer: str) -> ChatResponse:
    """The reply a client returns, with tokens estimated from length.

    Built as the agent package's own response type rather than as a look-alike,
    so a replayed run travels the same path a real one does and the log it
    leaves behind has the same shape.
    """
    from oold.agent.client import ChatResponse, TokenUsage

    return ChatResponse(
        text=answer,
        parsed=None,
        usage=TokenUsage(
            input_tokens=max(1, len(prompt) // CHARS_PER_TOKEN),
            output_tokens=max(1, len(answer) // CHARS_PER_TOKEN),
        ),
    )


class ReplayClient:
    """Answers one task from its expectation, in whatever shape it was asked.

    ``omit`` names expected keys to leave out and ``point_at`` rewrites where a
    link goes, so a test can produce a missing entity or a dangling edge on
    purpose instead of waiting for a model to produce one.
    """

    model = "replay"
    fold_system = False

    def __init__(
        self,
        task: TaskRecord,
        *,
        omit: Sequence[str] = (),
        point_at: dict[str, str] | None = None,
    ) -> None:
        self.task = task
        self.omit = set(omit)
        self.point_at = dict(point_at or {})
        self.calls: list[str] = []
        """Which step each call was read as, in order. The record a test reads
        to check that an orchestration made the calls it claims."""
        self._mentions = {instance.key: _mention_of(instance) for instance in task.expected}

    @property
    def answered(self) -> list[ExpectedInstance]:
        return [instance for instance in self.task.expected if instance.key not in self.omit]

    def invoke(
        self,
        messages: Sequence[Any],
        *,
        response_format: dict[str, Any] | None = None,
        strict: bool = False,
    ) -> ChatResponse:
        """Answer from the task's own truth, ignoring ``strict``.

        Accepted and not used, because a replay has no decoder to constrain:
        the answer comes from the expectation rather than from a model, so
        there is nothing a grammar could rule out. It is in the signature
        because :class:`~oold.agent.client.ChatClient` carries it and the
        agent passes it on every call, and leaving it out took the whole
        offline path down with ``TypeError: unexpected keyword argument``.
        """
        prompt = "\n\n".join(getattr(message, "content", "") or "" for message in messages)
        if _SELECT_MARKER in prompt:
            self.calls.append("select")
            return self._json(prompt, self._plan())
        if _PROSE_MARKER in prompt:
            self.calls.append("prose")
            return _reply(prompt, self.task.document)
        if _SEGMENTED_MARKER in prompt:
            self.calls.append("fill")
            return self._json(prompt, self._entities(_ids_in(prompt, self._mentions), with_ids=True))
        if _FILL_MARKER in prompt:
            self.calls.append("fill")
            return self._json(prompt, self._entities(_mentioned_in(prompt, self._mentions), with_ids=False))
        self.calls.append("extract")
        return self._json(prompt, self._entities(None, with_ids=False))

    def _json(self, prompt: str, payload: dict[str, Any]) -> ChatResponse:
        return _reply(prompt, json.dumps(payload, ensure_ascii=False))

    def _plan(self) -> dict[str, Any]:
        """Every entity, each shortlisted to its own class.

        One class per entity, so the shortlist always holds the true class.
        A select step that sometimes missed would make every downstream
        difference ambiguous, which is the opposite of what a replay is for.
        """
        return {
            "entities": [
                {
                    "id": instance.key,
                    "candidates": [instance.class_path],
                    "mention": self._mentions[instance.key],
                }
                for instance in self.answered
            ]
        }

    def _entities(self, keys: set[str] | None, *, with_ids: bool) -> dict[str, Any]:
        wanted = [i for i in self.answered if keys is None or i.key in keys]
        return {"entities": [self._entity(instance, with_ids=with_ids) for instance in wanted]}

    def _entity(self, instance: ExpectedInstance, *, with_ids: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"type": instance.class_path}
        if with_ids:
            body["id"] = instance.key
        for prop, value in instance.fields.items():
            for item in value if isinstance(value, (list, tuple)) else [value]:
                if isinstance(item, Reference):
                    # No plan, no ids, and therefore nothing to point at. A
                    # single-shot answer that invented a target would score as
                    # an invented entity, which is worse than the omission.
                    if with_ids:
                        body[prop] = self.point_at.get(f"{instance.key}.{prop}", item.key)
                elif isinstance(item, Quantity):
                    body[prop] = item.magnitude
                    body["unit"] = item.unit
                else:
                    body[prop] = item
        return body


def _mention_of(instance: ExpectedInstance) -> str:
    """The words a plan would report this entity from.

    The first text value it carries, because that is what a reader would name
    it by, and the key when it carries none. Uniqueness matters more than
    fidelity: the fill step names entities by mention, so two entities sharing
    one would make a call ambiguous.
    """
    for value in instance.fields.values():
        if isinstance(value, str) and value.strip() and not value.strip().isdigit():
            return f"{value.strip()} ({instance.key})"
    return instance.key


def _ids_in(prompt: str, mentions: dict[str, str]) -> set[str]:
    """The planned ids one fill call was asked for."""
    found = _SEGMENTED_IDS.search(prompt)
    clause = found.group(1) if found else prompt
    return {key for key in mentions if re.search(rf"(?<![\w-]){re.escape(key)}(?![\w-])", clause)}


def _mentioned_in(prompt: str, mentions: dict[str, str]) -> set[str]:
    """The entities one select-then-fill call named, by their mentions."""
    return {key for key, mention in mentions.items() if mention in prompt}
