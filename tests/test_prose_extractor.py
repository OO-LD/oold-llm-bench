"""The prose parser, and the error it is allowed to have.

``no-catalog-not-enforced-prose`` is scored through this parser, so the parser's own failures are
part of every number it produces. The last class of tests pins the ceiling it
reaches on the gold rendering, because a score without that ceiling beside
it says nothing.
"""

import pytest

from oold_llm_bench.corpus.quantities import QuantityKind, generate_task
from oold_llm_bench.extract import extract_prose, parse_loss
from oold_llm_bench.grading.score import score_task
from oold_llm_bench.grading.triples import Quantity
from oold_llm_bench.tasks.models import Difficulty

KINDS = [
    QuantityKind(name="Length", units=("meter",)),
    QuantityKind(name="Mass", units=("gram",)),
    QuantityKind(name="Duration", units=("second",)),
]


def quantities(text: str) -> list[Quantity]:
    return [t.value for t in extract_prose(text).triples if isinstance(t.value, Quantity)]


def units_of(text: str) -> list[str]:
    return sorted(q.unit for q in quantities(text))


def values_of(text: str) -> list[float]:
    return sorted(q.magnitude for q in quantities(text))


class TestReadingMeasurements:
    def test_a_measurement_beside_its_unit(self):
        assert values_of("The Length was 1.75 meter.") == [1.75]
        assert units_of("The Length was 1.75 meter.") == ["meter"]

    def test_a_multi_word_unit_is_joined_with_underscores(self):
        """Documents spell the corpus name out, so reading it back joins it."""
        assert units_of("Recorded 40.1 milli mole per kilo gram.") == ["milli_mole_per_kilo_gram"]

    def test_a_unit_containing_english_words_survives(self):
        """The and to are unit words in this corpus, so neither may stop a unit."""
        assert units_of("The instrument showed 74.4 coulomb to the fourth per joule cubed.") == [
            "coulomb_to_the_fourth_per_joule_cubed"
        ]

    def test_a_unit_containing_digits_survives(self):
        assert units_of("The angle was 15.6 to the 1000000000.") == ["to_the_1000000000"]

    def test_a_digit_inside_a_unit_is_not_a_second_measurement(self):
        assert values_of("The angle was 15.6 to the 1000000000.") == [15.6]

    def test_a_unit_does_not_run_into_the_next_clause(self):
        assert units_of("Settled at 65093.4 milli mole per kilo gram, which was fine.") == ["milli_mole_per_kilo_gram"]

    def test_a_unit_stops_at_a_conjunction(self):
        assert units_of("The Length was 1.75 meter and the Mass was 2.0 gram.") == [
            "gram",
            "meter",
        ]

    def test_several_measurements_in_a_list(self):
        text = "Measurements:\n- Molmasse: 79.7 gram per mole\n- Druck: 3.9 newton"
        assert units_of(text) == ["gram_per_mole", "newton"]
        assert values_of(text) == [3.9, 79.7]

    def test_a_unit_arriving_after_the_magnitude(self):
        """Ordinary English, so the parser handles it instead of losing it."""
        assert units_of("The Length came to 4.2, measured in newton.") == ["newton"]
        assert units_of("The reading came to 4.2, in newton.") == ["newton"]

    def test_a_label_is_kept_when_the_text_offers_one(self):
        produced = extract_prose("Molmasse: 79.7 gram per mole")
        assert list(produced.classes.values()) == ["Molmasse"]

    def test_a_thousands_separator_is_read(self):
        assert values_of("Recorded 65,093 meter.") == [65093.0]


class TestWhatItRefusesToGuess:
    def test_a_magnitude_with_no_unit_is_a_parse_error(self):
        """A number alone cannot answer a quantity, so it is loss, not a triple."""
        produced = extract_prose("The figure was 45.7. No further adjustment was made.")
        assert produced.triples == frozenset()
        assert produced.parse_errors == 1

    def test_a_unit_announced_in_a_separate_sentence_is_not_recovered(self):
        """Known loss. Recovering it would need a document-level default unit."""
        text = "Reported Mass was 45.7. The unit throughout is gram."
        assert extract_prose(text).triples == frozenset()

    def test_empty_text_produces_nothing_and_no_error(self):
        produced = extract_prose("")
        assert produced.triples == frozenset()
        assert produced.parse_errors == 0

    def test_text_with_no_measurement_at_all_is_one_error(self):
        assert extract_prose("Nothing was recorded on this sheet.").parse_errors == 1


class TestTheCeiling:
    """What the parser recovers from the gold rendering.

    No ``no-catalog-not-enforced-prose`` score can exceed these numbers, and reporting one without
    them hides the parser inside the score.
    """

    def tasks(self, difficulty: Difficulty, n: int = 1, count: int = 30):
        return [
            generate_task(KINDS, task_id=f"t{s}", seed=s, n_entities=n, difficulty=difficulty) for s in range(count)
        ]

    def test_the_parser_is_lossless_on_adjacent_renderings(self):
        for difficulty in (Difficulty.EASY, Difficulty.HARD):
            assert parse_loss(self.tasks(difficulty))["value_f1"] == 1.0

    def test_separated_renderings_cost_about_half(self):
        """Value and unit split across clauses. Half the templates are lost."""
        loss = parse_loss(self.tasks(Difficulty.MEDIUM))
        assert 0.4 <= loss["value_f1"] <= 0.7

    def test_the_ceiling_is_reported_with_its_task_count(self):
        loss = parse_loss(self.tasks(Difficulty.EASY))
        assert loss["tasks"] == 30
        assert set(loss) == {"tasks", "value_f1", "parse_errors"}

    def test_no_tasks_is_not_a_division_by_zero(self):
        assert parse_loss([])["tasks"] == 0

    def test_the_parser_still_scores_zero_on_a_different_document(self):
        """The control that matters. A parser that loses triples must not gain them
        on the wrong document.
        """
        task = self.tasks(Difficulty.EASY, count=1)[0]
        other = generate_task(KINDS, task_id="other", seed=9999, difficulty=Difficulty.EASY)
        assert score_task(task, extract_prose(other.document)).primary == pytest.approx(0.0)
