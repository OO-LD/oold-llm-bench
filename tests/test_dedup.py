"""The identity grader and the tiers that run before a judge is called."""

from __future__ import annotations

import pytest

from oold_llm_bench.corpus.wikidata_identity import IdentityClass, read_pairs
from oold_llm_bench.dedup import Resolver, informative, jaccard, name_entropy, normalise, shingles
from oold_llm_bench.grading.identity import AGREEMENT, JUDGE, Judgement, score_identity


@pytest.fixture(scope="module")
def corpus():
    return read_pairs()


class TestTheEntropyGate:
    """A name has to carry information before agreeing on it means anything."""

    def test_a_repetitive_name_carries_nothing(self):
        assert name_entropy("aaaa") == pytest.approx(0.0)
        assert not informative("aaaa")

    def test_a_short_name_is_refused_whatever_its_entropy(self):
        """ "Acme" is well formed and still too short to decide an identity on."""
        assert name_entropy("Acme") > 1.5
        assert not informative("Acme")

    def test_a_real_name_passes(self):
        assert informative("Ottavio Panciroli")

    def test_normalisation_ignores_case_and_punctuation(self):
        assert normalise("  Acme, Inc.  ") == "acme inc"
        assert jaccard(shingles("Ottavio Panciroli"), shingles("Ottavio Panciroli.")) > 0.9


class TestTheGraderReportsBothNumbers:
    """Accuracy alone is gameable by abstaining, coverage alone by guessing."""

    def _pairs(self, corpus):
        return [p for p in corpus.pairs if p.scored][:30]

    def test_always_unclear_is_never_wrong_and_covers_nothing(self, corpus):
        def abstain(pair):
            return Judgement(pair.id, IdentityClass.UNCLEAR, AGREEMENT)

        score = score_identity(self._pairs(corpus), abstain, "abstain")
        assert score.coverage == 0.0
        assert score.decided == 0
        assert score.decided_accuracy == 0.0

    def test_a_perfect_resolver_scores_one_on_both(self, corpus):
        def oracle(pair):
            return Judgement(pair.id, pair.identity, JUDGE)

        score = score_identity(self._pairs(corpus), oracle, "oracle")
        assert score.accuracy == pytest.approx(1.0)
        assert score.wrong_merges == 0
        assert score.calls == score.total

    def test_a_wrong_merge_is_counted_apart_from_a_missed_one(self, corpus):
        """They are different failures and only one of them is unrecoverable."""

        def merge_everything(pair):
            return Judgement(pair.id, IdentityClass.SAME, AGREEMENT)

        score = score_identity(self._pairs(corpus), merge_everything, "merge all")
        expected = sum(1 for p in self._pairs(corpus) if p.identity is IdentityClass.DIFFERENT)
        assert score.wrong_merges == expected
        assert score.coverage == 1.0


class TestTheCheapTiersAlone:
    """What the deterministic part is worth, measured rather than assumed."""

    def test_it_spends_nothing(self, corpus):
        score = score_identity(corpus.pairs, Resolver(), "prefilter")
        assert score.calls == 0
        assert JUDGE not in score.routes

    def test_it_merges_pairs_a_person_called_different(self, corpus):
        """The reason a judge is needed, as a number rather than an argument.

        58 per cent of the negatives carry a name the other side also carries,
        because the confusion that made someone write ``P1889`` down was
        usually the name. A tier that trusts an agreeing name therefore merges
        a large share of them, and a merge cannot be undone.
        """
        score = score_identity(corpus.pairs, Resolver(), "prefilter")
        negatives = sum(1 for p in corpus.pairs if p.scored and p.identity is IdentityClass.DIFFERENT)
        assert score.wrong_merges > negatives * 0.3
        assert score.per_class[IdentityClass.DIFFERENT.value][0] == 0

    def test_a_judge_takes_everything_the_cheap_tiers_left(self, corpus):
        asked = []

        def judge(pair):
            asked.append(pair.id)
            return IdentityClass.DIFFERENT, "stub"

        score = score_identity(corpus.pairs, Resolver(judge=judge), "with judge")
        assert score.calls == len(asked)
        assert score.calls > 0
        assert score.coverage > score_identity(corpus.pairs, Resolver(), "bare").coverage
