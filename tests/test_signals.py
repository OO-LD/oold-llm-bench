"""EMMO signals and the alternative vocabulary.

EMMO earns three things here that QUDT alone cannot give. An English synonym,
which a translated label is not. A second authored definition, independent of
QUDT's phrasing. And an opaque class identifier, which is the pretraining
contrast taken from a real standard and not from a scramble.
"""

import pytest

from oold_llm_bench.corpus.signals import Signal, SignalData, Vocabulary, load_signals


def data() -> SignalData:
    return SignalData(
        labels={"Length": {"de": "Länge"}},
        descriptions={"Length": "the extent of a thing"},
        symbols={},
        synonyms={"Length": ("linear extent",)},
        elucidations={"Length": "Extend of a spatial dimension."},
        emmo_iris={"Length": "EMMO_cd2cd0de_e0cc_4ef1_b27e_2e88db027bac"},
        qudt_version="3.1.5",
        emmo_version="1.0.4",
        source="test",
    )


class TestSupports:
    @pytest.mark.parametrize("signal", list(Signal))
    def test_a_kind_with_everything_supports_every_signal(self, signal):
        assert data().supports("Length", signal)

    @pytest.mark.parametrize(
        "signal",
        [Signal.TRANSLATED, Signal.DESCRIPTION, Signal.SYNONYM, Signal.ELUCIDATION],
    )
    def test_a_kind_with_nothing_supports_no_derived_signal(self, signal):
        assert not data().supports("Unknown", signal)

    @pytest.mark.parametrize("signal", [Signal.NAMED, Signal.UNIT])
    def test_named_and_unit_need_no_external_material(self, signal):
        assert data().supports("Unknown", signal)


class TestVocabulary:
    def test_consensus_answers_to_the_qudt_name(self):
        assert data().identifier("Length", Vocabulary.CONSENSUS) == "Length"

    def test_emmo_answers_to_an_opaque_identifier(self):
        identifier = data().identifier("Length", Vocabulary.EMMO)
        assert identifier == "EMMO_cd2cd0de_e0cc_4ef1_b27e_2e88db027bac"

    def test_the_emmo_identifier_carries_no_guessable_word(self):
        """That is the point: nothing here could have been memorised."""
        identifier = data().identifier("Length", Vocabulary.EMMO) or ""
        assert "length" not in identifier.lower()

    def test_a_kind_with_no_emmo_class_has_no_emmo_identifier(self):
        assert data().identifier("Unknown", Vocabulary.EMMO) is None


class TestCommittedData:
    """The shipped file, so a change to the build script is visible."""

    def test_it_loads_and_names_its_sources(self):
        described = load_signals().describe()
        assert described["qudt_version"] == "3.1.5"
        assert described["emmo_version"] == "1.0.4"

    def test_it_carries_every_kind_of_signal(self):
        described = load_signals().describe()
        for key in (
            "kinds_with_labels",
            "kinds_with_descriptions",
            "kinds_with_synonyms",
            "kinds_with_elucidations",
            "kinds_with_emmo_iri",
        ):
            assert described[key] > 0

    def test_no_label_or_description_names_its_own_kind(self):
        """485 of 807 QUDT descriptions did, and were filtered out."""
        signals = load_signals()
        flat = lambda text: "".join(c for c in text.lower() if c.isalnum())
        for kind, text in signals.descriptions.items():
            assert flat(kind) not in flat(text)
        for kind, text in signals.elucidations.items():
            assert flat(kind) not in flat(text)
        for kind, labels in signals.labels.items():
            for label in labels.values():
                assert flat(kind) not in flat(label)

    def test_english_is_never_a_translated_label(self):
        """An English label is the class name with spaces in it."""
        for labels in load_signals().labels.values():
            assert "en" not in labels

    def test_length_joins_qudt_to_emmo(self):
        signals = load_signals()
        assert signals.emmo_iris.get("Length", "").startswith("EMMO_")
        assert signals.elucidations.get("Length")


class TestOswVocabulary:
    """The corpus was generated from OSW pages and kept their identifiers.

    Every one of the 943 schemas carries an ``x-oold-uuid``, where EMMO
    reaches 357, so OSW is the opaque vocabulary that covers the whole corpus
    and not a third of it.
    """

    def kind(self, uuid: str = "ee9c7e5c-343e-542c-b5a8-b4648315902f"):
        from oold_llm_bench.corpus import QuantityKind

        return QuantityKind(name="Length", units=("meter",), uuid=uuid)

    def test_the_osw_identifier_is_the_page_id(self):
        from oold_llm_bench.corpus import identifier_for

        assert identifier_for(self.kind(), Vocabulary.OSW, data()) == "OSWee9c7e5c343e542cb5a8b4648315902f"

    def test_it_carries_no_guessable_word(self):
        from oold_llm_bench.corpus import identifier_for

        identifier = identifier_for(self.kind(), Vocabulary.OSW, data()) or ""
        assert "length" not in identifier.lower()

    def test_a_kind_with_no_uuid_has_no_osw_identifier(self):
        from oold_llm_bench.corpus import identifier_for

        assert identifier_for(self.kind(""), Vocabulary.OSW, data()) is None

    def test_consensus_still_answers_to_the_readable_name(self):
        from oold_llm_bench.corpus import identifier_for

        assert identifier_for(self.kind(), Vocabulary.CONSENSUS, data()) == "Length"

    def test_the_three_vocabularies_disagree(self):
        """Which is the point: each names the same concept differently."""
        from oold_llm_bench.corpus import identifier_for

        names = {identifier_for(self.kind(), vocabulary, data()) for vocabulary in Vocabulary}
        assert len(names) == 3
