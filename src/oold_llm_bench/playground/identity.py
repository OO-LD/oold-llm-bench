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

* **agreement**, a fast path. Two entities of one class with nothing left to
  decide between them, because every stated value is the same one or because
  an informative, shared name settles it, are the same thing and no call is
  spent confirming it. :func:`decide` says exactly which case is which.
* **judge**, a model asked in words. Named and configurable separately from the
  extractor, because a judge that is the model under test confounds
  deduplication with extraction, which is the fault that made every number the
  predecessor produced unreadable.

The judge answers three ways and not two. ``exactMatch`` and ``different``
decide; ``closeMatch`` defers, and defers to a human rather than to a default.
The names are SKOS's, because SKOS already names exactly these relations and
inventing a third spelling of ``exactMatch`` would help nobody. ``different``
is not a third SKOS relation beside them: it is the absence of a match, the
same outcome a comparison that settles nothing already defaults to, and
nothing here invents a name for it.

Abstention is free unless it is measured, so a judge that always defers is
never wrong. :func:`coverage` reports the share of comparisons that were
decided at all, which is the number that makes a quiet collapse into
``closeMatch`` visible as a collapse.

**Why this is** :class:`~oold_llm_bench.dedup.Resolver`, **now, and what that
changes.** It was not, on the grounds that reshaping a :class:`Comparable`
into an :class:`~oold_llm_bench.corpus.wikidata_identity.EntityState` would
mean inventing the labels and claims a sparse extraction never had, for a
similarity tier tuned to Wikidata's own multilingual-short-label noise rather
than to this corpus's prose. Both halves of that were true and neither is
undone here. :func:`_entity_state_of` invents nothing: a
:class:`Comparable`'s one name-bearing value becomes the single label, every
other stated value becomes a claim with no values added to it, and nothing is
synthesised to fill the fields a Wikidata item would have and a prose
extraction does not. What changes is the decision that mattered more than the
shape: a second, independent dedup implementation is a second place to get
dedup wrong, and a judge asked one question in the graded benchmark and a
different one in the interface a person tunes against is not one capability
measured twice, it is two capabilities that happen to share a name.

The entropy-gated similarity tier is accepted as it is tuned, not retuned for
this corpus, and that has a visible cost: it is a name-only tier, so two
entities sharing a near-identical name merge on that alone even where another
claim between them conflicts, the same way a Wikidata item with a stale
statement still merges on its name. The playground's older rule refused
exactly that case, on the grounds that a merge cannot be undone from the
interface. ``auto_merge_exact_match`` (see :func:`action_for`) is where that
caution now lives: off, an ``exactMatch`` the resolver is confident of is
still drawn rather than folded, for a person to confirm.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from oold_llm_bench.corpus.wikidata_identity import EntityState, IdentityClass, Pair, Truth
from oold_llm_bench.dedup import Resolver
from oold_llm_bench.grading.identity import JUDGE as _RESOLVER_JUDGE_ROUTE

__all__ = [
    "CLOSE_MATCH",
    "DEFER",
    "DIFFERENT",
    "EXACT_MATCH",
    "MERGE",
    "NO_ACTION",
    "Comparable",
    "Decision",
    "Judge",
    "Ledger",
    "ModelJudge",
    "action_for",
    "counts",
    "coverage",
    "decide",
]

EXACT_MATCH = "skos:exactMatch"
"""Confidently one entity. Merges by default; see :func:`action_for`."""

CLOSE_MATCH = "skos:closeMatch"
"""Similar, possibly interchangeable, not decided. Both entities stay and the
relation is drawn between them, waiting for a person to confirm or reject it."""

DIFFERENT = "different"
"""Two entities. No relation is asserted, because there is nothing to assert."""

AGREEMENT = "agreement"
"""The deterministic route: exact agreement, short-circuited before any call."""

JUDGE = "judge"
"""The route that spent a call."""

MERGE = "merge"
"""Fold the right node into the left one."""

DEFER = "defer"
"""Draw an edge between the two nodes and leave both standing."""

NO_ACTION = "none"
"""Draw nothing and fold nothing: the decision was ``different``."""


@dataclass(frozen=True)
class Comparable:
    """One entity as the identity step sees it: a class, its literals, and
    the one value among them that names it, where anything does."""

    key: str
    class_path: str | None
    values: Mapping[str, str]
    label: str | None = None
    """The name already computed for this entity, where one was.

    Read off the drawn node's own label by :func:`~oold_llm_bench.playground.graph._comparable`,
    which may have composed it from a split ``givenName``/``familyName`` that
    :attr:`values` still carries as two separate entries. Used as the
    designator when this entity is turned into an
    :class:`~oold_llm_bench.corpus.wikidata_identity.EntityState`; a
    :class:`Comparable` built with none falls back to whatever :attr:`values`
    states under ``name``, and failing that to :attr:`key`, so comparing two
    of these the way every test here already did before this field existed
    still compares the same way.
    """

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
        """Whether this decision is an ``exactMatch``.

        A classification of the decision alone, not of what a session did
        with it: :func:`action_for` is where ``auto_merge_exact_match`` is
        applied, so an ``exactMatch`` left undrawn by the toggle still reports
        ``merges`` true here, which is what it would merge to if asked.
        """
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
    """Whatever decides the pairs the cheap tiers did not."""

    name: str

    def decide(self, left: Comparable, right: Comparable) -> tuple[str, str]:
        """The outcome and the evidence for it."""
        ...


@dataclass
class ModelJudge:
    """A model asked, in words, whether two entities are one.

    A thin adapter over :func:`~oold_llm_bench.dedup.judge_of`, the one real
    judge implementation in the codebase: the two entities are turned into an
    :class:`~oold_llm_bench.corpus.wikidata_identity.EntityState` pair the same
    way :func:`decide` already does to reach :class:`~oold_llm_bench.dedup.Resolver`,
    the question is asked in :func:`~oold_llm_bench.dedup.judge_of`'s own
    words, and the identity class that comes back is translated into this
    module's outcome vocabulary. A session in the interface and a graded run
    of ``scripts/run_dedup_bench.py`` therefore ask one model the same
    question, not two.

    Declared rather than implicit. The name is reported beside every decision
    it makes, so a reader can see when the judge and the extractor are the
    same model, which is the confound rather than a detail of it.
    """

    client: Any
    name: str

    def decide(self, left: Comparable, right: Comparable) -> tuple[str, str]:
        from oold_llm_bench.dedup import judge_of

        outcome, evidence = judge_of(self.client)(_pair_of(left, right))
        return _IDENTITY_TO_OUTCOME[outcome], evidence


def _compatible(left: str | None, right: str | None) -> bool:
    """Whether two claimed classes leave the pair open to being one thing."""
    return left is None or right is None or left == right


_PLAYGROUND_TRUTH = Truth(
    kind="playground", detail="an in-session comparison, not a recorded judgement", recorded_at=None, recorded_by=None
)

_OUTCOME_TO_IDENTITY = {
    EXACT_MATCH: IdentityClass.SAME,
    CLOSE_MATCH: IdentityClass.UNCLEAR,
    DIFFERENT: IdentityClass.DIFFERENT,
}
_IDENTITY_TO_OUTCOME = {value: key for key, value in _OUTCOME_TO_IDENTITY.items()}


def _entity_state_of(entity: Comparable) -> EntityState:
    """One playground entity, in the shape :class:`~oold_llm_bench.dedup.Resolver` compares.

    The same conversion ``scripts/run_dedup_bench.py::_state_of`` applies to
    the generated sequence corpus, for the same reason: there is no Wikidata
    item behind an entity read off one document, so the designator is the one
    label and every other stated value becomes a claim, with nothing invented
    to stand in for the aliases, the description or the second language a
    Wikidata item would carry and a :class:`Comparable` never had.
    """
    designator = entity.label or entity.values.get("name") or entity.key
    claims: dict[str, tuple[str, ...]] = {prop: (value,) for prop, value in entity.values.items() if prop != "name"}
    return EntityState(
        qid=entity.key,
        labels={"en": designator},
        aliases=(),
        description=None,
        sitelinks={},
        claims=claims,
        statements=len(claims),
        revision=None,
        as_of="playground",
    )


def _pair_of(left: Comparable, right: Comparable) -> Pair:
    """Two entities, wrapped the way :class:`~oold_llm_bench.dedup.Resolver` expects a pair.

    The bookkeeping :class:`~oold_llm_bench.corpus.wikidata_identity.Pair`
    carries beyond ``left`` and ``right``, such as :attr:`Pair.identity`, is
    never read by :class:`~oold_llm_bench.dedup.Resolver`: it exists for
    :func:`~oold_llm_bench.grading.identity.score_identity`, which this pair is
    never handed to, so it is filled with a placeholder and marked unscored
    rather than left to mean something it does not.
    """
    left_state = _entity_state_of(left)
    right_state = _entity_state_of(right)
    return Pair(
        id=f"{left.key}|{right.key}",
        identity=IdentityClass.UNCLEAR,
        left=left_state,
        right=right_state,
        truth=_PLAYGROUND_TRUTH,
        era=None,
        shares_string=bool(left_state.strings & right_state.strings),
        shares_property=bool(set(left_state.claims) & set(right_state.claims)),
        scored=False,
    )


def decide(left: Comparable, right: Comparable, judge: Judge | None = None) -> Decision:
    """Whether these two entities are one, by :class:`~oold_llm_bench.dedup.Resolver`.

    A class mismatch is decided here and not asked about, and never reaches
    the resolver: the classes come from a catalogue the model was given, so
    two different ones are the model's own answer about what these things are,
    and spending a call to overrule it would put the judge in the extractor's
    seat.

    An entity that states no class contradicts nothing, so it is compatible
    with any. A nested object written into a link slot arrives that way: nano
    answered ``worksFor`` with ``{"name": "Example Corp"}`` beside an
    ``Organization`` of the same name, and reading the absence of a class as a
    class of its own drew the organisation twice.

    A literal restatement merges here too, before the resolver is asked at
    all, and for a reason the resolver's own corpus never has to cover: a
    Wikidata item is never compared against itself, so none of
    :class:`~oold_llm_bench.dedup.Resolver`'s tiers has to recognise an entity
    with no name at all, and this one does, because an orchestration
    resubmitting one unchanged document hands the same unnamed entity back
    under the same id. Where every stated property and every stated value
    agree, and at least one is stated, there is nothing left to decide, name
    or no name, so this is not a weaker rediscovery of what the resolver
    already answers; it is the one case that is not a question to begin with.
    Two records stating nothing agree on nothing either, which is why this
    requires a value and does not fire on two bare entities: a class and no
    properties is a real question and still the judge's.

    Everything else is :class:`~oold_llm_bench.dedup.Resolver`'s: exact
    agreement on an informative shared name with no conflicting claim merges
    free, a near-identical name merges free too wherever it is informative
    enough to trust (the module docstring says what that tier costs here), and
    a pair neither tier settles is the judge's, or ``closeMatch`` if none is
    configured. ``judge`` given here is never called directly: it is wrapped
    so :class:`~oold_llm_bench.dedup.Resolver` calls it only where its own
    cheap tiers could not decide, and the outcome it returns is translated
    back from :class:`~oold_llm_bench.corpus.wikidata_identity.IdentityClass`
    into this module's vocabulary.

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

    if left.values and set(left.values) == set(right.values) and not left.conflicts(right):
        return Decision(
            left=left.key,
            right=right.key,
            outcome=EXACT_MATCH,
            route=AGREEMENT,
            evidence="every stated property and value agree",
        )

    resolver_judge: Callable[[Pair], tuple[IdentityClass, str]] | None = None
    if judge is not None:

        def resolver_judge(_pair: Pair, _judge: Judge = judge) -> tuple[IdentityClass, str]:
            outcome, evidence = _judge.decide(left, right)
            return _OUTCOME_TO_IDENTITY[outcome], evidence

    judgement = Resolver(judge=resolver_judge)(_pair_of(left, right))
    asked = judgement.route == _RESOLVER_JUDGE_ROUTE
    return Decision(
        left=left.key,
        right=right.key,
        outcome=_IDENTITY_TO_OUTCOME[judgement.outcome],
        route=JUDGE if asked else AGREEMENT,
        evidence=judgement.evidence,
        judge=judge.name if asked and judge is not None else None,
        conflicts=left.conflicts(right),
    )


def action_for(decision: Decision, *, auto_merge_exact_match: bool = True) -> str:
    """What a session does with one decision, apart from the decision itself.

    Detecting that two entities are one and acting on that detection are
    different questions, and the first answering ``exactMatch`` does not have
    to settle the second: a merge cannot be undone from the interface, so
    ``auto_merge_exact_match`` set to false leaves an ``exactMatch`` drawn as
    a :data:`CLOSE_MATCH` edge already is, for a person to confirm, rather
    than folding it on sight. Left true, which is the default, an
    ``exactMatch`` folds the way it always did.

    ``closeMatch`` is unaffected either way: it was already deferred rather
    than applied, and the toggle only ever changes what happens to an answer
    the resolver was confident of, never an unsure one. ``different`` draws
    nothing to begin with.
    """
    if decision.outcome == EXACT_MATCH:
        return MERGE if auto_merge_exact_match else DEFER
    if decision.outcome == CLOSE_MATCH:
        return DEFER
    return NO_ACTION


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
