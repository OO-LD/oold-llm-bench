"""Entity pairs a person judged to be one thing, two things, or unresolved.

Deciding whether two mentions denote the same entity is a different capability
from reading a value out of a document, and the end use case needs both. They
are separate corpora here so that a failure is attributable to one of them:
an extraction benchmark that also measures deduplication reports a number that
cannot be acted on.

A judge is unavoidable in this task and is not the fault to avoid. Identity
properties are not definable: schema.org designates none, Wikidata designates
none, and which properties identify an entity is a question about the class
and the context rather than about the schema. What is refused is a judge in
the *grader*. The judge under test is a component measured against ground
truth that people recorded, one decision at a time, for their own reasons.

*Where the three classes come from.* Wikidata records all three, and records
each as a human act rather than as a derived score. ``same`` is an item merge:
``Q123`` now redirects to ``Q456`` because an editor concluded the two items
were one thing. ``different`` is ``P1889``, "different from", which the
property's own definition scopes to items "with which it may be confused".
``unclear`` is ``P460``, "said to be the same as", defined as "though this may
be uncertain or disputed", which is the class written down by the people who
could not decide.

*What the pair carries.* Both Q-ids, both entity states, the class, and the
provenance of the truth. For a merge the state is the state **before** the
merge, recovered for each side from its own history, because the post-merge
item already contains the answer. The two pre-merge revision ids, the merge
timestamp and the merge editor are carried with it, so the decision can be
read back. ``P1889`` and ``P460`` pairs carry the current state of both items
and the statement that asserts the class; neither the editor nor the date of
that statement is recorded, because finding the revision that introduced a
statement costs a search over the item's history and the class does not depend
on it.

*Why the states are trimmed the way they are.* An identifier decides more of
this corpus than a name does, so every claim property survives, with its
values. What does not survive is repetition: an item that carries one name in
forty languages says nothing in the fortieth that it did not say in the first,
so labels are kept one language per distinct string. The languages that differ
are the ones that matter, and they are the ones this keeps. ``P1889``,
``P460`` and ``P2959`` are
removed from every state on both sides, in all three classes, because they are
the truth this corpus is built from and an item that points at its own pair
answers the question in its own input.

*What the filter is for, and it is two things.* Both sides must carry at least
three statements and an English label. That keeps a pair substantive, and it
is also the filter that removes the errors: of 105 merge pairs hand-checked,
2 are wrong, and both have a source item with one sitelink and no statements
at all, so the editor who merged them had a page title to work from and
nothing else. Of the 43 judged pairs that pass this filter, 0 are wrong. The
measured wrong rate on a uniform draw is 1 of 60, upper bound 8.9%, which
excludes the 15% that would have made the corpus unreadable.

*The era column, and why merges before 2017 are left out.* Both wrong merges
and 13 of the 18 pairs judged undecidable fall before 2017. 2017 to 2020 is 30
correct and 1 undecidable of 31; 2021 to 2026 is 41 correct, 4 undecidable and
0 wrong of 45. The survey put the cut at about a tenth of the stock, and that
is its cost on the raw draw; taken after the statement floor it is a few
percent, because the pre-2017 merges are overwhelmingly the thin ones the
floor has already removed. The era each kept pair falls in stays on the
record, so a report can slice on it rather than take the cut on faith.

*The tier the corpus is for.* Not the pairs that share no property: those are
thin items rather than hard ones, and "shares no property" and "passes the
filter" are disjoint in a 296-pair sample, 0 pairs in both. The hard tier is
"passes the filter and shares no label or alias string", about a fifth of the
usable positives, and it is decided by identifiers rather than by names:
``Cilu Xiong`` against ``Xiong Cilv`` through one VIAF id,
``CRTS J103338.8-250002`` against ``HE 1031-2444`` through an identical
SIMBAD id and an identical right ascension.
:attr:`Pair.shares_string` marks it, and :func:`read_pairs` refuses a file in
which the tier has vanished.

*What the trivial rules score, and the prediction they refute.* An exact match
on any name either item carries gets 78% of the positives right and 42% of the
negatives, so 60% over the two classes a two-way rule can name at all, against
a coin's 50%. On English labels alone it is 71% and 62%, so 67%. The survey
this corpus was designed from predicted *below* chance, on the grounds that
real duplicates share an English label 27% of the time and ``P1889`` pairs
63%. Both figures are for a different population. The 27% was measured on the
uniform merge stock, whose median item carries four statements, and this
corpus keeps only pairs with at least three on both sides, where 78% share a
name; the kill test's own "17 of 81 filtered pairs share no string" says the
same thing and was never reconciled with the 27%. The 63% was measured before
disambiguation pages and name-string items were removed, and those collide by
construction. The claim that survives the correction is narrower and is still
the reason to build this: 58% of these negatives carry a name the other side
also carries, and a person declared them different anyway. A synthetic
negative set collides at a rate of about zero.

*One asymmetry that is not corrected.* A merge pair is two revisions from
before the merge; a statement pair is two items as they stand now. The
positives are therefore the thinner side: median 10 statements, 0 sitelinks
and 1 label language, against 17, 2 and 3 on the negatives. A system could
learn the shape of the record rather than the question asked of it. Matching
the distributions would mean selecting negatives by statement count, which is
a filter with no human decision behind it, so the built file carries the three
distributions in its header instead and an accuracy that does not control for
them is reading a confound.

*Why the negatives are not the hardest negatives available.* ``P1889`` has
about 114,000 pairs where two same-typed, identically-named things were
declared different, and drawing the class from those alone was the obvious
move. It is not made, because it would make the headline finding an artefact:
if every negative carries an identical label by selection, a name matcher
scores near zero on them because of how they were picked, not because of what
``P1889`` is. The class is drawn uniformly and the string-collision rate it
happens to carry is reported as a measurement.
:attr:`Pair.shares_string` recovers the hard subset for anyone who wants it.

*What is left out.* Every candidate is used or excluded under a stated reason,
per class, and :func:`read_pairs` refuses a file whose counts do not add up.
The failure being guarded against is the one
:mod:`oold_llm_bench.corpus.eln` names: a corpus that has quietly lost a third
of itself still loads, and nothing says so.

Licence: CC0 1.0. Everything here is main-namespace Wikidata structured data,
which Wikidata's own copyright page, the item page footer and
``meta=siteinfo&siprop=rightsinfo`` all put under CC0. Unlike
:mod:`oold_llm_bench.corpus.wiki_measurements` there is no second licence and
no third-party text: attribution is courtesy rather than obligation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "CORPUS_PATH",
    "DUMP_URL",
    "LICENCE",
    "MIN_STATEMENTS",
    "NOT_AN_ENTITY",
    "TRUTH_PROPERTIES",
    "Baseline",
    "EntityState",
    "IdentityClass",
    "IdentityCorpus",
    "Pair",
    "Truth",
    "agreeing_properties",
    "label_baseline",
    "property_baseline",
    "read_pairs",
    "score_baseline",
    "shared_strings",
]

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "wikidata_identity.json"

LICENCE = "CC0 1.0"
"""What every field of every record is under, and there is only one.

Confirmed from three places that agree: the footer of an item page, the API's
``meta=siteinfo&siprop=rightsinfo`` declaration, and the wikitext of
<https://www.wikidata.org/wiki/Wikidata:Copyright>. All three say that
structured data from the main, Property, Lexeme and EntitySchema namespaces is
CC0 and that the other namespaces are CC BY-SA. This corpus carries entity
JSON, Q-ids, revision ids and timestamps, all of which are main-namespace
structured data. Edit summaries are arguably not, so the merge comment is used
at build time to find the target and is not written out.
"""

DUMP_URL = "https://dumps.wikimedia.org/wikidatawiki/latest/wikidatawiki-latest-redirect.sql.gz"
"""Where the merge stock is enumerated.

A merge leaves no log entry. It is three ordinary edits, so ``letype=merge``
is empty and correct to be empty, and the population has to come either from
the redirect table or from a pass over ``mediawiki_history``. The redirect
table names the *final* target of a chain and the history names the immediate
one; this corpus takes the immediate one, out of the merge edit's own comment,
because that is the item the editor was looking at.
"""

MIN_STATEMENTS = 3
"""Statements required on both sides.

Substance and error removal, in one threshold. See the module docstring.
"""

TRUTH_PROPERTIES = ("P1889", "P460", "P2959")
"""Properties removed from every state, because they assert the answer.

``P1889`` and ``P460`` are two of the three class sources, so a ``different``
pair whose own state carries ``P1889`` pointing at the other side is not a
task. ``P2959``, "permanent duplicated item", is removed for the same reason
on the other side: it names a pair Wikidata holds to be one entity and cannot
merge. Removing all three from all three classes keeps the rule one rule.
"""

MATCH_BUCKETS = (5, 8, 12, 18, 26, 40)
"""Strata of ``min(statements)`` that :meth:`IdentityCorpus.matched` balances.

A positive is a pre-merge revision and a negative is a current state, so the
classes arrive at different sizes: median ``min(statements)`` is 7, 14 and 10
for same, different and unclear. Measured on the built corpus, a classifier
reading nothing but the two statement counts scores **70.6%** on same against
different, where the name matcher scores 60.0%. The shape is not a nuisance to
note in a header, it is the best single predictor in the corpus, and a judge
evaluated on the unmatched pairs can win by learning how full an item is.

Balancing over strata removes it: thinness falls to **51.0%**, which is
chance, and the medians close to 10, 9 and 10. The two content baselines move
little, 60.0% to 57.7% and 73.3% to 71.2%, which is the point. They were
reading the records; only the size rule was reading the artefact.

It costs 52 pairs per class of the 90, and 6 of the 20 hard-tier positives.
At 52 a paired test detects roughly +0.30 and not +0.20, so a matched result
is reported with that limit beside it rather than as though the sample were
still 90.

Edges are geometric rather than uniform because statement counts are, and they
are fixed here rather than derived from the data so that adding pairs does not
silently re-cut every earlier sample.
"""

NOT_AN_ENTITY = (
    "Q4167410",  # Wikimedia disambiguation page
    "Q22808320",  # Wikimedia human name disambiguation page
    "Q4167836",  # Wikimedia category
    "Q11266439",  # Wikimedia template
    "Q13406463",  # Wikimedia list article
    "Q17362920",  # Wikimedia duplicated page
    "Q15184295",  # Wikimedia module
    "Q101352",  # family name
    "Q12308941",  # male given name
    "Q11879590",  # female given name
    "Q3409032",  # unisex given name
    "Q202444",  # given name
)
"""``P31`` values that make an item a page or a string rather than a thing.

A fifth of the ``P1889`` stock is disambiguation pages and another eighth is
name-string items, and a deduplicator is never handed either. Merging two
categories across wikis is a real editorial act but it is an equivalence
between pages, which is a looser question than whether two entities are one
entity. Excluded with a count rather than argued about.
"""


class IdentityClass(str, Enum):
    """What a person decided about the pair.

    Three values and no fourth. ``unclear`` is not a hedge the corpus offers
    for convenience: it is ``P460``, a property whose definition is "said to
    be the same as that item, though this may be uncertain or disputed".
    """

    SAME = "same"
    DIFFERENT = "different"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class EntityState:
    """One item as it stood at one revision.

    For a ``same`` pair this is the state the day before the merge, which is
    what the editor saw. For the other two it is the current state.
    """

    qid: str
    labels: dict[str, str]
    """One language per distinct string, English first where there is one."""
    aliases: tuple[str, ...]
    description: str | None
    sitelinks: dict[str, str]
    claims: dict[str, tuple[str, ...]]
    """Property to values, with :data:`TRUTH_PROPERTIES` removed."""
    statements: int
    """How many statements the item carried, before any trimming."""
    revision: int | None
    """The revision this state was read from.

    ``None`` on a statement pair, where the state is the item as it stood on
    :attr:`as_of` and no revision was pinned. A merge pair always has one, and
    it is the revision before the merge.
    """
    as_of: str

    @property
    def strings(self) -> frozenset[str]:
        """Every name this item answers to, case folded."""
        found = {value.casefold() for value in self.labels.values()}
        found |= {alias.casefold() for alias in self.aliases}
        return frozenset(found)


@dataclass(frozen=True)
class Truth:
    """Who recorded the class, and where it can be read back."""

    kind: str
    """``merge``, ``P1889`` or ``P460``."""
    detail: str
    """The merge target, or the statement as a triple."""
    recorded_at: str | None
    recorded_by: str | None
    """The account that performed the merge. ``None`` for the two statement
    classes: finding the revision that introduced a statement costs a search
    over the item's history and is not carried."""
    final_redirect_target: str | None = None
    """Set when the redirect table names a later target than the merge did.

    The source is paired with the item named in its own merge comment, which
    is the item the editor was looking at, and not with wherever the redirect
    points now: those differ when the intermediate target was itself merged on
    and the redirect was retargeted. The kill test saw 1 source in 320 do
    this. This harvest saw 0 in 390, so the field exists for a case it has not
    yet had to record.
    """


@dataclass(frozen=True)
class Pair:
    """Two entity states and what a person concluded about them."""

    id: str
    identity: IdentityClass
    left: EntityState
    right: EntityState
    truth: Truth
    era: str | None
    """The period the merge falls in. ``None`` for the two statement classes."""
    shares_string: bool
    """Whether any label or alias string is common to both sides."""
    shares_property: bool
    scored: bool = True
    """Whether the pair belongs in an accuracy at all.

    Set false for the granularity cases: a heritage-register entry merged into
    the castle it registers, a statute section merged into the statute, two
    Motif-Index codes for one motif. Those are neither the same entity nor
    different entities, they are a question about what an item denotes, and
    counting them as positives would charge a system for the better answer.
    """
    note: str | None = None
    """Why :attr:`scored` is false. ``None`` on a pair that is scored.

    What the hand check reasoned is deliberately not here. It names the answer
    and it would be in the record a judge is handed; it is in
    ``docs/dedup_kill_test_judgements.tsv`` instead, and the header says how
    many pairs carry one.
    """


@dataclass(frozen=True)
class IdentityCorpus:
    """The built corpus and the arithmetic that says nothing was lost."""

    pairs: tuple[Pair, ...]
    candidates_in: dict[str, int]
    excluded: dict[str, dict[str, int]]
    """Per class, how many candidates each declared reason removed."""
    resolved: dict[str, int]
    """Per class, how many survived every reason, before the cap."""
    cap: int | None
    source: dict[str, Any]
    licence: dict[str, Any]
    built_at: str
    min_statements: int = MIN_STATEMENTS
    wikidata_base: str = "http://www.wikidata.org/entity/"

    @property
    def retrieved_at(self) -> datetime:
        stamp = str(self.source.get("retrieved_at") or self.built_at)
        return datetime.combine(date.fromisoformat(stamp), datetime.min.time(), tzinfo=timezone.utc)

    def of(self, identity: IdentityClass) -> tuple[Pair, ...]:
        return tuple(pair for pair in self.pairs if pair.identity is identity)

    @property
    def scored(self) -> tuple[Pair, ...]:
        return tuple(pair for pair in self.pairs if pair.scored)

    @property
    def hard_tier(self) -> tuple[Pair, ...]:
        """Positives that pass the filter and share no name with each other.

        The tier the benchmark is for, and the one a build is refused for
        losing.
        """
        return tuple(p for p in self.pairs if p.identity is IdentityClass.SAME and not p.shares_string)

    def matched(self) -> tuple[Pair, ...]:
        """The pairs with the state-size confound balanced out of them.

        Equal pairs per class in every stratum of :data:`MATCH_BUCKETS`, so a
        system cannot score by reading how full an item is. See that constant
        for what it costs and what it is worth.

        Deterministic and independent of draw order: a stratum is sorted by
        statement count and the kept pairs are spread evenly across that order,
        so the distribution inside a stratum is matched too and not only the
        count. Ties break on id, which is stable across builds.

        A selection, never a filter on the build. The excluded pairs are real
        decisions and stay in :attr:`pairs`; a result says which sample it was
        measured on.
        """
        strata: dict[int, dict[IdentityClass, list[Pair]]] = {}
        for pair in self.pairs:
            bucket = strata.setdefault(_bucket_of(pair), {})
            bucket.setdefault(pair.identity, []).append(pair)

        kept: list[Pair] = []
        for bucket in (strata[edge] for edge in sorted(strata)):
            if len(bucket) < len(IdentityClass):
                continue
            keep = min(len(rows) for rows in bucket.values())
            for rows in bucket.values():
                kept.extend(_spread(sorted(rows, key=lambda p: (_thinner(p), p.id)), keep))
        return tuple(sorted(kept, key=lambda p: p.id))


def _thinner(pair: Pair) -> int:
    """The smaller of the two statement counts.

    The smaller one, because a pair is only as informative as its poorer side
    and that is the side an extractor would struggle on.
    """
    return min(pair.left.statements, pair.right.statements)


def _bucket_of(pair: Pair) -> int:
    count = _thinner(pair)
    return next((edge for edge in MATCH_BUCKETS if count < edge), MATCH_BUCKETS[-1] + 1)


def _spread(rows: list[Pair], keep: int) -> list[Pair]:
    """``keep`` of ``rows``, evenly spaced across the order given."""
    if keep >= len(rows):
        return rows
    if keep == 1:
        return [rows[len(rows) // 2]]
    step = (len(rows) - 1) / (keep - 1)
    return [rows[round(index * step)] for index in range(keep)]


def _state(entry: dict[str, Any]) -> EntityState:
    return EntityState(
        qid=entry["qid"],
        labels=dict(entry["labels"]),
        aliases=tuple(entry["aliases"]),
        description=entry.get("description"),
        sitelinks=dict(entry["sitelinks"]),
        claims={prop: tuple(values) for prop, values in entry["claims"].items()},
        statements=int(entry["statements"]),
        revision=int(entry["revision"]) if entry.get("revision") else None,
        as_of=entry["as_of"],
    )


def _pair(entry: dict[str, Any]) -> Pair:
    truth = entry["truth"]
    return Pair(
        id=entry["id"],
        identity=IdentityClass(entry["class"]),
        left=_state(entry["left"]),
        right=_state(entry["right"]),
        truth=Truth(
            kind=truth["kind"],
            detail=truth["detail"],
            recorded_at=truth.get("recorded_at"),
            recorded_by=truth.get("recorded_by"),
            final_redirect_target=truth.get("final_redirect_target"),
        ),
        era=entry.get("era"),
        shares_string=bool(entry["shares_string"]),
        shares_property=bool(entry["shares_property"]),
        scored=bool(entry.get("scored", True)),
        note=entry.get("note"),
    )


def _closes(corpus: IdentityCorpus) -> list[str]:
    """Every way the file's own counts contradict each other."""
    problems: list[str] = []
    for name, went_in in corpus.candidates_in.items():
        excluded = sum(corpus.excluded.get(name, {}).values())
        accounted = corpus.resolved.get(name, 0) + excluded
        if accounted != went_in:
            problems.append(
                f"{name}: {went_in} candidates went in and {accounted} are accounted for, "
                f"so {went_in - accounted} left without a declared reason"
            )
    for name in corpus.excluded:
        if name not in corpus.candidates_in:
            problems.append(f"{name}: exclusions are counted for a class no candidate went into")
    written: dict[str, int] = {}
    for pair in corpus.pairs:
        written[pair.identity.value] = written.get(pair.identity.value, 0) + 1
    for name, count in written.items():
        if count > corpus.resolved.get(name, 0):
            problems.append(f"{name}: the file holds {count} pairs but only {corpus.resolved.get(name, 0)} resolved")
        if corpus.cap is not None and count > corpus.cap:
            problems.append(f"{name}: the file holds {count} pairs under a cap of {corpus.cap}")
    return problems


def _substantive(corpus: IdentityCorpus) -> list[str]:
    """Every way a written pair contradicts the filter the header declares."""
    problems: list[str] = []
    for pair in corpus.pairs:
        for side in (pair.left, pair.right):
            if side.statements < corpus.min_statements:
                problems.append(
                    f"{pair.id}: {side.qid} carries {side.statements} statements "
                    f"and the corpus declares a floor of {corpus.min_statements}"
                )
            if "en" not in side.labels:
                problems.append(f"{pair.id}: {side.qid} carries no English label")
            for prop in TRUTH_PROPERTIES:
                if prop in side.claims:
                    problems.append(f"{pair.id}: {side.qid} still carries {prop}, which asserts the answer")
        if pair.left.qid == pair.right.qid:
            problems.append(f"{pair.id}: both sides are {pair.left.qid}")
        if pair.shares_string != bool(pair.left.strings & pair.right.strings):
            problems.append(f"{pair.id}: shares_string={pair.shares_string} contradicts the two states")
        if pair.shares_property != bool(set(pair.left.claims) & set(pair.right.claims)):
            problems.append(f"{pair.id}: shares_property={pair.shares_property} contradicts the two states")
    return problems


def read_pairs(path: Path | None = None) -> IdentityCorpus:
    """Read the built corpus, refusing a file whose counts do not add up.

    Three kinds of refusal, and each one is a failure that would otherwise be
    silent. The per-class arithmetic has to close, so a candidate cannot
    disappear without a reason. Every written pair has to satisfy the filter
    the header declares, so a corpus cannot claim a floor of three statements
    and carry a stub. And the hard tier has to be non-empty, because it is the
    tier the corpus exists for and a filter change that empties it leaves a
    file that still loads and measures the wrong thing.
    """
    payload = json.loads((path or CORPUS_PATH).read_text(encoding="utf-8"))
    corpus = IdentityCorpus(
        pairs=tuple(_pair(entry) for entry in payload["pairs"]),
        candidates_in=dict(payload["candidates_in"]),
        excluded={name: dict(reasons) for name, reasons in payload["excluded"].items()},
        resolved=dict(payload["resolved"]),
        cap=payload.get("cap"),
        source=dict(payload["source"]),
        licence=dict(payload["licence"]),
        built_at=payload["built_at"],
        min_statements=int(payload.get("min_statements", MIN_STATEMENTS)),
        wikidata_base=payload.get("wikidata_base") or "http://www.wikidata.org/entity/",
    )
    problems = _closes(corpus) + _substantive(corpus)
    if not corpus.hard_tier and corpus.of(IdentityClass.SAME):
        problems.append("no positive pair shares no string, so the tier the corpus is for is not in the file")
    if problems:
        raise ValueError("the Wikidata identity corpus does not hold together: " + "; ".join(problems[:20]))
    return corpus


def shared_strings(left: EntityState, right: EntityState) -> frozenset[str]:
    """Names both sides answer to. Empty is the hard tier."""
    return left.strings & right.strings


def agreeing_properties(left: EntityState, right: EntityState) -> tuple[int, int]:
    """How many properties both sides carry, and how many of those conflict.

    A property conflicts when the two value sets are disjoint. No value is
    normalised and no unit is converted: this is the fast path a system would
    take before calling a judge, and it is measured as written.
    """
    shared = set(left.claims) & set(right.claims)
    conflicting = sum(1 for prop in shared if not set(left.claims[prop]) & set(right.claims[prop]))
    return len(shared), conflicting


def label_baseline(pair: Pair) -> IdentityClass:
    """Same if the two sides share a name string, different otherwise.

    The cheapest thing anyone would try. It beats a coin on this corpus and
    it is nowhere near usable: it calls 58% of the negatives the same entity,
    because the confusion that made a person write ``P1889`` down in the first
    place was usually the name.
    """
    return IdentityClass.SAME if shared_strings(pair.left, pair.right) else IdentityClass.DIFFERENT


def property_baseline(pair: Pair) -> IdentityClass:
    """Same if the sides share at least one property and none of them conflict.

    The other fast path, and it fails the other way round: nearly every pair
    here shares a property, so what decides the rule is whether any of them
    conflict, and two records of one entity conflict often enough that it
    rejects half the positives.
    """
    shared, conflicting = agreeing_properties(pair.left, pair.right)
    return IdentityClass.SAME if shared and not conflicting else IdentityClass.DIFFERENT


@dataclass(frozen=True)
class Baseline:
    """What a rule with no judge in it scores, and against what.

    Not a grader. A grader scores a system's answer against the corpus, and
    the corpus is what exists so far; this counts exact agreements between a
    stated rule and the recorded class, so that a report can say whether the
    corpus is hard before any model has been called.
    """

    name: str
    total: int
    correct: int
    decidable: int
    """Pairs in the two classes a two-way rule can name at all."""
    decidable_correct: int
    per_class: dict[str, tuple[int, int]]
    """Class to (correct, total)."""

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def decidable_accuracy(self) -> float:
        """Against a coin, which is the comparison the survey predicted."""
        return self.decidable_correct / self.decidable if self.decidable else 0.0


def score_baseline(pairs: Iterable[Pair], predict: Callable[[Pair], IdentityClass], name: str) -> Baseline:
    """Count where a two-way rule agrees with the recorded class."""
    per_class: dict[str, list[int]] = {}
    total = correct = decidable = decidable_correct = 0
    for pair in pairs:
        if not pair.scored:
            continue
        hit = int(predict(pair) is pair.identity)
        bucket = per_class.setdefault(pair.identity.value, [0, 0])
        bucket[0] += hit
        bucket[1] += 1
        total += 1
        correct += hit
        if pair.identity is not IdentityClass.UNCLEAR:
            decidable += 1
            decidable_correct += hit
    return Baseline(
        name=name,
        total=total,
        correct=correct,
        decidable=decidable,
        decidable_correct=decidable_correct,
        per_class={k: (v[0], v[1]) for k, v in sorted(per_class.items())},
    )
