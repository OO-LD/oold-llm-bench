"""Deciding whether two extracted entities are one thing.

Extraction and deduplication are two capabilities and the playground needs
both: a graph that grows over turns has to know that the organisation named in
turn two is the one already on screen. The predecessor decided this three
different ways and measured none of them. Two of those are not worth porting.
``lookup_excact_matching_entity`` retrieves at a hardcoded ``k`` and falls back
to a similarity threshold its own comment calls arbitrary, and
``_merge_entity_data`` is ``{**existing, **new}``, which is not a merge policy
but the absence of one.

What does carry over is the question, and the answer is that identity is not
definable by rule. schema.org designates no identifying properties, and which
properties identify an entity is class- and context-dependent: two people share
a name, one company is written two ways. So judgement is needed, and the
judgement is part of the system under test rather than part of the grader.

Hence two routes to one decision, kept apart because the difference between
them is a cost:

* **agreement**, a fast path. Two entities of one class that agree on every
  value they both carry, and carry at least one in common, are the same thing
  and no call is spent confirming it.
* **judge**, a model asked in words. Named and configurable separately from the
  extractor, because a judge that is the model under test confounds
  deduplication with extraction, which is the fault that made every number the
  predecessor produced unreadable.

The judge answers three ways and not two. ``exactMatch`` and ``different``
decide; ``closeMatch`` defers, and defers to a human rather than to a default.
The names are SKOS's, because SKOS already names exactly these relations and
inventing a third spelling of ``exactMatch`` would help nobody.

Abstention is free unless it is measured, so a judge that always defers is
never wrong. :func:`coverage` reports the share of comparisons that were
decided at all, which is the number that makes a quiet collapse into
``closeMatch`` visible as a collapse.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "CLOSE_MATCH",
    "DIFFERENT",
    "EXACT_MATCH",
    "Comparable",
    "Decision",
    "Judge",
    "Ledger",
    "ModelJudge",
    "NoJudge",
    "counts",
    "coverage",
    "decide",
]

EXACT_MATCH = "skos:exactMatch"
"""Confidently one entity. Merges, with no human step."""

CLOSE_MATCH = "skos:closeMatch"
"""Similar, possibly interchangeable, not decided. Both entities stay and the
relation is drawn between them, waiting for a person to confirm or reject it."""

DIFFERENT = "different"
"""Two entities. No relation is asserted, because there is nothing to assert."""

AGREEMENT = "agreement"
"""The deterministic route: exact agreement, short-circuited before any call."""

JUDGE = "judge"
"""The route that spent a call."""


@dataclass(frozen=True)
class Comparable:
    """One entity as the identity step sees it: a class and its literals."""

    key: str
    class_path: str | None
    values: Mapping[str, str]

    def shared(self, other: Comparable) -> tuple[str, ...]:
        return tuple(sorted(set(self.values) & set(other.values)))

    def conflicts(self, other: Comparable) -> tuple[str, ...]:
        """Properties both carry and disagree on, compared case-insensitively."""
        return tuple(
            prop
            for prop in self.shared(other)
            if self.values[prop].strip().casefold() != other.values[prop].strip().casefold()
        )


@dataclass(frozen=True)
class Decision:
    """One comparison, its outcome, and how the outcome was reached."""

    left: str
    right: str
    outcome: str
    route: str
    evidence: str
    judge: str | None = None
    """The model that decided, or ``None`` where no call was made."""
    conflicts: tuple[str, ...] = ()
    """Properties the two disagree on.

    Carried through a merge rather than resolved by it. Deciding that two
    entities are one says nothing about which of two values for one property is
    right, and taking whichever arrived second is the thing this avoids."""

    @property
    def merges(self) -> bool:
        return self.outcome == EXACT_MATCH

    def describe(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "outcome": self.outcome,
            "route": self.route,
            "evidence": self.evidence,
            "judge": self.judge,
            "conflicts": list(self.conflicts),
        }


class Judge(Protocol):
    """Whatever decides the pairs exact agreement did not."""

    name: str

    def decide(self, left: Comparable, right: Comparable) -> tuple[str, str]:
        """The outcome and the evidence for it."""
        ...


@dataclass
class NoJudge:
    """No judgement available, so nothing is decided and nothing is merged.

    Returns ``closeMatch`` rather than ``different``, because two entities that
    agreed on nothing measurable are exactly the unclear case, and answering
    ``different`` would hide the absence of a judge as a finding about the
    entities.
    """

    name: str = "none"

    def decide(self, left: Comparable, right: Comparable) -> tuple[str, str]:
        return CLOSE_MATCH, "no judge is configured, so the pair is left for a person"


_PROMPT = (
    "Two entities were extracted from text. Decide whether they are the same real-world thing.\n"
    'Answer with JSON only: {{"outcome": "exactMatch" | "closeMatch" | "different", '
    '"reason": "<one sentence>"}}\n'
    "Use exactMatch when they are confidently the same, different when they are confidently not, "
    "and closeMatch when you are unsure and a person should look.\n\n"
    "A: {left}\nB: {right}\n"
)

_OUTCOMES = {"exactmatch": EXACT_MATCH, "closematch": CLOSE_MATCH, "different": DIFFERENT}


@dataclass
class ModelJudge:
    """A model asked, in words, whether two entities are one.

    Declared rather than implicit. The name is reported beside every decision
    it makes, so a reader can see when the judge and the extractor are the same
    model, which is the confound rather than a detail of it.
    """

    client: Any
    name: str

    def decide(self, left: Comparable, right: Comparable) -> tuple[str, str]:
        # Imported here and not at module scope, so the identity rules stay
        # testable with no agent package installed, the way the graph
        # conversion beside them is.
        from oold.agent.client import Message

        prompt = _PROMPT.format(left=_render(left), right=_render(right))
        reply = self.client.invoke([Message(role="user", content=prompt)])
        text = getattr(reply, "text", "") or ""
        parsed = _read(getattr(reply, "parsed", None) or text)
        outcome = _OUTCOMES.get(str(parsed.get("outcome", "")).strip().casefold())
        if outcome is None:
            return CLOSE_MATCH, f"the judge answered {text.strip()[:120]!r}, which names no outcome"
        return outcome, str(parsed.get("reason") or "").strip() or "the judge gave no reason"


def _compatible(left: str | None, right: str | None) -> bool:
    """Whether two claimed classes leave the pair open to being one thing."""
    return left is None or right is None or left == right


def _render(entity: Comparable) -> str:
    body = {"type": entity.class_path, **dict(sorted(entity.values.items()))}
    return json.dumps(body, ensure_ascii=False)


def _read(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def decide(left: Comparable, right: Comparable, judge: Judge) -> Decision:
    """Whether these two entities are one, by agreement first and judge after.

    A class mismatch is decided here and not asked about. The classes come from
    a catalogue the model was given, so two different ones are the model's own
    answer about what these things are, and spending a call to overrule it
    would put the judge in the extractor's seat.

    An entity that states no class contradicts nothing, so it is compatible
    with any. A nested object written into a link slot arrives that way: nano
    answered ``worksFor`` with ``{"name": "Example Corp"}`` beside an
    ``Organization`` of the same name, and reading the absence of a class as a
    class of its own drew the organisation twice.

    **Agreement means every property, not every shared one.** The earlier rule
    merged on one property in common that happened to agree, so a ``Person``
    with ``{name: "Alice Smith", jobTitle: "Engineer"}`` and a ``Person`` with
    ``{jobTitle: "Engineer", email: "bob@example.com"}`` were merged with no
    call and no prompt. A merge cannot be undone from the interface, so the
    cheap route has to be the safe one: it fires only where the two carry the
    same properties and agree on all of them, which is a restatement rather
    than a judgement. Everything else is what the judge is for, and a pair the
    judge cannot settle is a ``closeMatch`` for a person to resolve.

    Reference properties are not compared at all, and
    :func:`~oold_llm_bench.playground.graph._comparable` says why: a link's
    value is an id the pipeline invented, so agreement on it is not evidence
    about identity.
    """
    if not _compatible(left.class_path, right.class_path):
        return Decision(
            left=left.key,
            right=right.key,
            outcome=DIFFERENT,
            route=AGREEMENT,
            evidence=f"different classes: {left.class_path} and {right.class_path}",
        )

    shared = left.shared(right)
    conflicts = left.conflicts(right)
    if shared and not conflicts and set(left.values) == set(right.values):
        agreed = ", ".join(f"{prop}={left.values[prop]!r}" for prop in shared[:3])
        stated = left.class_path or right.class_path
        named = f"class {stated}" if stated else "no class either side"
        return Decision(
            left=left.key,
            right=right.key,
            outcome=EXACT_MATCH,
            route=AGREEMENT,
            evidence=f"same {named} and every property agrees: {agreed}",
        )

    outcome, evidence = judge.decide(left, right)
    # A judge that cannot be asked did not spend a call. JUDGE is documented
    # as the route that costs something and the Identity read-out reports the
    # two apart so the cost is on screen rather than in a bill, so counting a
    # no-op under it made a session that made no calls read as "12 by judge".
    asked = not isinstance(judge, NoJudge)
    return Decision(
        left=left.key,
        right=right.key,
        outcome=outcome,
        route=JUDGE if asked else AGREEMENT,
        evidence=evidence,
        judge=judge.name if asked else None,
        conflicts=conflicts,
    )


@dataclass
class Ledger:
    """Every comparison made, in the order it was made."""

    decisions: list[Decision] = field(default_factory=list)

    def record(self, decision: Decision) -> Decision:
        self.decisions.append(decision)
        return decision

    def describe(self) -> dict[str, Any]:
        return {
            "decisions": [decision.describe() for decision in self.decisions],
            "counts": counts(self.decisions),
            "coverage": coverage(self.decisions),
        }


def counts(decisions: Sequence[Decision]) -> dict[str, int]:
    """How many comparisons landed on each outcome and each route."""
    tally = {EXACT_MATCH: 0, CLOSE_MATCH: 0, DIFFERENT: 0, AGREEMENT: 0, JUDGE: 0}
    for decision in decisions:
        tally[decision.outcome] = tally.get(decision.outcome, 0) + 1
        tally[decision.route] = tally.get(decision.route, 0) + 1
    return tally


def coverage(decisions: Sequence[Decision]) -> float:
    """The share of comparisons that were decided rather than deferred.

    Reported because abstention is otherwise free: a judge that answers
    ``closeMatch`` every time is never wrong, and only this number shows it.
    Accuracy over the decided share is the other half and belongs to the
    deduplication benchmark, which has ground truth to measure it against.

    Nothing compared is 0.0 and not 1.0. The one number whose job is to make
    abstention visible should not open the panel claiming full coverage of a
    session in which nothing has been decided.
    """
    if not decisions:
        return 0.0
    decided = sum(1 for decision in decisions if decision.outcome != CLOSE_MATCH)
    return decided / len(decisions)
