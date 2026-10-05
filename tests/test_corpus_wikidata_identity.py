"""Wikidata identity pairs as a corpus.

Two things are being tested. That the file holds together: the per-class
arithmetic closes, every written pair satisfies the filter the header
declares, and the tier the corpus exists for is still in it. And that the
corpus is hard, which is not a property of the loader but of the data: a
matcher on names calls a quarter or more of the negatives the same entity,
because the confusion that made someone record ``P1889`` was the name.
"""

import collections
import json

import pytest

from oold_llm_bench.corpus.wikidata_identity import (
    CORPUS_PATH,
    LICENCE,
    MIN_STATEMENTS,
    TRUTH_PROPERTIES,
    IdentityClass,
    label_baseline,
    property_baseline,
    read_pairs,
    score_baseline,
    shared_strings,
)

CLASS_MINIMUM = 60
"""Pairs a class needs before a rate measured on it is worth quoting."""


@pytest.fixture(scope="module")
def corpus():
    return read_pairs()


def rewritten(tmp_path, change):
    """The committed corpus with one thing broken, on disk."""
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    change(payload)
    path = tmp_path / "wikidata_identity.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_every_class_is_present_and_large_enough(corpus):
    for identity in IdentityClass:
        assert len(corpus.of(identity)) >= CLASS_MINIMUM, identity


def test_the_arithmetic_closes_per_class(corpus):
    for name, went_in in corpus.candidates_in.items():
        assert corpus.resolved[name] + sum(corpus.excluded[name].values()) == went_in


def test_a_candidate_that_left_without_a_reason_refuses_to_load(tmp_path):
    def drop(payload):
        name = IdentityClass.SAME.value
        payload["excluded"][name] = dict(list(payload["excluded"][name].items())[1:])

    with pytest.raises(ValueError, match="left without a declared reason"):
        read_pairs(rewritten(tmp_path, drop))


def test_a_pair_below_the_declared_floor_refuses_to_load(tmp_path):
    def thin(payload):
        payload["pairs"][0]["left"]["statements"] = MIN_STATEMENTS - 1

    with pytest.raises(ValueError, match="statements"):
        read_pairs(rewritten(tmp_path, thin))


def test_a_state_still_asserting_the_answer_refuses_to_load(tmp_path):
    def leak(payload):
        payload["pairs"][0]["left"]["claims"][TRUTH_PROPERTIES[0]] = [payload["pairs"][0]["right"]["qid"]]

    with pytest.raises(ValueError, match="asserts the answer"):
        read_pairs(rewritten(tmp_path, leak))


def test_a_tier_flag_that_contradicts_the_states_refuses_to_load(tmp_path):
    def lie(payload):
        payload["pairs"][0]["shares_property"] = not payload["pairs"][0]["shares_property"]

    with pytest.raises(ValueError, match="contradicts the two states"):
        read_pairs(rewritten(tmp_path, lie))


def test_losing_the_hard_tier_refuses_to_load(tmp_path):
    def soften(payload):
        for pair in payload["pairs"]:
            if pair["class"] != IdentityClass.SAME.value:
                continue
            pair["left"]["labels"] = dict(pair["right"]["labels"])
            pair["left"]["aliases"] = []
            pair["shares_string"] = True

    with pytest.raises(ValueError, match="the tier the corpus is for"):
        read_pairs(rewritten(tmp_path, soften))


def test_the_hard_tier_survived_the_filter(corpus):
    """Positives that pass the filter and share no name with each other.

    The kill test found 17 of 81, so a fifth of the usable positives. They are
    the pairs decided by an identifier rather than by a name, and they are why
    this corpus exists rather than a string-similarity table.
    """
    hard = corpus.hard_tier
    assert len(hard) >= 10
    assert 0.10 <= len(hard) / len(corpus.of(IdentityClass.SAME)) <= 0.60
    for pair in hard:
        assert not shared_strings(pair.left, pair.right)
        assert pair.shares_property, pair.id


def test_a_merge_carries_who_decided_it_and_when(corpus):
    for pair in corpus.of(IdentityClass.SAME):
        assert pair.truth.kind == "merge"
        assert pair.truth.recorded_at and pair.truth.recorded_by
        assert int(pair.truth.recorded_at[:4]) >= corpus.source["same"].get("earliest_year", 2017)
        assert pair.era in ("2017-2020", "2021-2026")
        assert pair.left.revision and pair.right.revision
        assert pair.left.as_of < pair.truth.recorded_at
        assert pair.right.as_of < pair.truth.recorded_at


def test_a_statement_pair_names_the_statement_and_claims_no_editor(corpus):
    for identity, prop in ((IdentityClass.DIFFERENT, "P1889"), (IdentityClass.UNCLEAR, "P460")):
        pairs = corpus.of(identity)
        assert pairs
        for pair in pairs:
            assert pair.truth.kind == prop
            assert pair.truth.detail == f"{pair.left.qid} {prop} {pair.right.qid}"
            assert pair.truth.recorded_by is None
            assert pair.era is None


def test_the_file_states_cc0_and_names_the_dump_it_came_from(corpus):
    assert corpus.licence["name"] == LICENCE
    assert "creativecommons.org/publicdomain/zero" in corpus.licence["url"]
    assert corpus.source["same"]["dump_date"]
    assert corpus.source["same"]["dump"].endswith("redirect.sql.gz")
    assert corpus.source["same"]["count"] > 4_000_000


def test_a_perfect_answer_is_the_recorded_class(corpus):
    """There is no grader yet, so this is what one would have to score.

    A perfect answer names the class a person recorded, for every pair the
    corpus offers for scoring. The granularity pairs are not offered: a
    register entry against the thing it registers is a question about what an
    item denotes, and a system that answers "not the same" there is right in a
    way an accuracy cannot represent.
    """
    perfect = score_baseline(corpus.pairs, lambda pair: pair.identity, "perfect")
    assert perfect.correct == perfect.total == len(corpus.scored)
    assert perfect.accuracy == 1.0
    assert len(corpus.scored) <= len(corpus.pairs)


def test_a_granularity_pair_is_carried_and_not_scored(tmp_path):
    """The label the kill test asked for, exercised whether or not it fires.

    A register entry merged into the thing it registers is neither the same
    entity nor a different one, so it is carried with the reason and left out
    of every accuracy. Which pairs land there depends on the draw, and the
    behaviour must not depend on the draw containing one.
    """

    def mark(payload):
        payload["pairs"][0]["scored"] = False
        payload["pairs"][0]["note"] = "a hand check of this merge returned unclear"

    corpus = read_pairs(rewritten(tmp_path, mark))
    assert len(corpus.scored) == len(corpus.pairs) - 1
    assert corpus.pairs[0].note
    perfect = score_baseline(corpus.pairs, lambda pair: pair.identity, "perfect")
    assert perfect.total == len(corpus.pairs) - 1


def test_a_name_matcher_calls_a_large_share_of_the_negatives_the_same_thing(corpus):
    """The property no synthetic negative set has, measured here.

    The survey predicted a name matcher would score below chance, on rates
    taken from the unfiltered stock. On the corpus as specified it scores
    above chance and nowhere near usable, and the reason is the false
    positives: a person declared these pairs different while they carried the
    same name. A synthetic negative is two unrelated entities and collides at
    about zero, so this cell is the one a generated corpus cannot fill.
    """
    baseline = score_baseline(corpus.pairs, label_baseline, "exact name match")
    different_hits, different_total = baseline.per_class[IdentityClass.DIFFERENT.value]
    assert 1 - different_hits / different_total >= 0.25
    assert baseline.decidable_accuracy < 0.75
    assert baseline.per_class[IdentityClass.UNCLEAR.value][0] == 0


def test_a_property_agreement_rule_does_not_decide_it_either(corpus):
    """The other fast path, and it fails the other way round.

    Nearly every pair here shares a property, so the rule turns on whether any
    of them conflict, and two records of one entity conflict often enough that
    it rejects a large share of the positives. Carried so a report has a
    reference point with no judge in it, not because it works.
    """
    baseline = score_baseline(corpus.pairs, property_baseline, "property agreement")
    same_hits, same_total = baseline.per_class[IdentityClass.SAME.value]
    assert same_hits / same_total < 0.75
    assert baseline.decidable_accuracy < 0.85
    assert baseline.per_class[IdentityClass.UNCLEAR.value][0] == 0


def test_neither_rule_can_reach_the_unclear_class(corpus):
    """A two-way rule cannot name the third class, so it caps below the ceiling.

    Worth asserting rather than assuming: it is the reason an accuracy over
    the three classes and an accuracy over the two are different numbers, and
    a report that quotes one for the other is out by the size of this class.
    """
    unclear = [pair for pair in corpus.scored if pair.identity is IdentityClass.UNCLEAR]
    for rule in (label_baseline, property_baseline):
        baseline = score_baseline(corpus.pairs, rule, "two-way")
        assert baseline.accuracy < baseline.decidable_accuracy
        assert baseline.total - baseline.decidable == len(unclear)


def test_no_pair_and_no_item_is_carried_twice(corpus):
    """One decision per pair, and no item on both sides of one.

    A merge target absorbs more than one item 681,101 times over the whole
    stock, so drawing sources uniformly can hand the same target twice; that
    is two decisions and stays. The same source twice would be one decision
    counted twice, and an id against itself would be a pair with no question
    in it.
    """
    assert len({pair.id for pair in corpus.pairs}) == len(corpus.pairs)
    sides = [(pair.left.qid, pair.right.qid) for pair in corpus.pairs]
    assert len({frozenset(side) for side in sides}) == len(sides)
    assert all(left != right for left, right in sides)


def test_matched_sample_balances_the_state_size_confound(corpus):
    """The size shape stops predicting the class once the strata are balanced.

    The assertion that matters is the second one. A positive is a pre-merge
    revision and a negative a current state, so on the whole corpus a rule
    reading only the two statement counts beats the name matcher, and a judge
    could score well by learning how full an item is rather than whether two
    records are one thing. Balanced, that rule falls to chance and the
    corpus asks its own question again.
    """
    matched = corpus.matched()
    counts = collections.Counter(pair.identity for pair in matched)
    assert len(counts) == len(IdentityClass)
    assert len(set(counts.values())) == 1, counts

    def thinness_rule(pairs):
        binary = [p for p in pairs if p.identity is not IdentityClass.UNCLEAR]
        best = 0
        for cut in range(1, 60):
            hits = sum(
                1
                for p in binary
                if (min(p.left.statements, p.right.statements) < cut) == (p.identity is IdentityClass.SAME)
            )
            best = max(best, hits / len(binary))
        return best

    assert thinness_rule(corpus.pairs) > 0.65
    assert thinness_rule(matched) < 0.58


def test_matched_sample_is_a_selection_and_not_a_filter(corpus):
    """Every kept pair is a corpus pair, and the rest are still there.

    A sample that quietly became the corpus would drop human decisions on the
    floor and leave no way to report the unmatched number beside the matched
    one.
    """
    matched = corpus.matched()
    assert {pair.id for pair in matched} <= {pair.id for pair in corpus.pairs}
    assert len(matched) < len(corpus.pairs)
    assert corpus.matched() == matched
