"""What the prompt is allowed to say about each class.

Two failures are possible here and both are silent. Saying too little leaves a
model choosing between names it knows nothing about. Saying the sentence the
document was built from turns the task into a paraphrase match, which is the
predecessor's failure where four of six entities matched on class alone.
"""

from typing import ClassVar

import pytest

from oold_llm_bench.corpus.catalogue import (
    EXCLUDED_BY_SIGNAL,
    CatalogueEntry,
    render_catalogue,
)
from oold_llm_bench.corpus.signals import Signal

ENTRY = CatalogueEntry(
    identifier="PhononMeanFreePath",
    label="Phonon Mean Free Path",
    description='"Phonon Mean Free Path" is the mean free path of phonons.',
    parent="Length",
    units=("meter", "kilo_meter", "fermi"),
)
ENTRIES = {"PhononMeanFreePath": ENTRY}


class TestOneEntry:
    def test_the_identifier_leads_because_it_is_the_answer(self):
        assert ENTRY.render().startswith("- PhononMeanFreePath")

    def test_a_label_that_repeats_the_identifier_is_not_shown_twice(self):
        assert ENTRY.render().count("Phonon Mean Free Path") == 1

    def test_the_parent_is_shown(self):
        assert "< Length" in ENTRY.render()

    def test_units_are_shown_and_capped(self):
        rendered = ENTRY.render(max_units=2)
        assert "units: meter, kilo_meter, +1 more" in rendered

    def test_an_excluded_field_does_not_appear(self):
        assert "mean free path of phonons" not in ENTRY.render(frozenset({"description"}))

    def test_a_bare_entry_still_renders(self):
        assert CatalogueEntry(identifier="X").render() == "- X"


class TestWithholdingTheSignalSource:
    """The document and the catalogue must point at the class differently."""

    def test_an_elucidation_document_is_shown_no_description(self):
        assert "description" in EXCLUDED_BY_SIGNAL[Signal.ELUCIDATION]

    def test_a_description_document_is_shown_no_description(self):
        assert "description" in EXCLUDED_BY_SIGNAL[Signal.DESCRIPTION]

    def test_a_unit_document_is_shown_no_units(self):
        """Otherwise the answer is readable straight off the unit list."""
        assert "units" in EXCLUDED_BY_SIGNAL[Signal.UNIT]

    def test_a_translated_document_is_shown_no_label(self):
        assert "label" in EXCLUDED_BY_SIGNAL[Signal.TRANSLATED]

    def test_the_exclusion_is_applied(self):
        rendered = render_catalogue(("PhononMeanFreePath",), ENTRIES, signal=Signal.ELUCIDATION)
        assert "mean free path of phonons" not in rendered[0]

    def test_without_a_signal_nothing_is_withheld(self):
        rendered = render_catalogue(("PhononMeanFreePath",), ENTRIES)
        assert "mean free path of phonons" in rendered[0]


class TestAnOpaqueVocabulary:
    """Describing the catalogue is what makes an opaque identifier usable."""

    ALIAS: ClassVar[dict[str, str]] = {"PhononMeanFreePath": "EMMO_8dacb56f"}

    def test_the_opaque_identifier_leads_the_entry(self):
        rendered = render_catalogue(("EMMO_8dacb56f",), ENTRIES, alias=self.ALIAS)
        assert rendered[0].startswith("- EMMO_8dacb56f")

    def test_the_human_label_is_attached_to_it(self):
        rendered = render_catalogue(("EMMO_8dacb56f",), ENTRIES, alias=self.ALIAS)
        assert "(Phonon Mean Free Path)" in rendered[0]

    def test_an_identifier_with_no_entry_is_kept_not_dropped(self):
        """Dropping it would offer a smaller catalogue than declared."""
        rendered = render_catalogue(("EMMO_8dacb56f", "Unknown"), ENTRIES, alias=self.ALIAS)
        assert len(rendered) == 2
        assert rendered[1] == "- Unknown"

    def test_the_order_is_the_order_it_was_given(self):
        rendered = render_catalogue(("Unknown", "EMMO_8dacb56f"), ENTRIES, alias=self.ALIAS)
        assert rendered[0] == "- Unknown"


class TestAgainstTheEnforcement:
    def test_the_text_has_to_line_up_with_the_catalogue(self):
        """A mismatch would describe one class as another."""
        pytest.importorskip("oold.agent.enforcement")
        from oold.agent.enforcement import arm

        with pytest.raises(ValueError, match="one class would be described as another"):
            arm("schema-dump-catalog-flat-enforced", ("A", "B")).with_catalogue_text(("- A",))

    def test_matching_lengths_are_accepted(self):
        pytest.importorskip("oold.agent.enforcement")
        from oold.agent.enforcement import arm

        enforcement = arm("schema-dump-catalog-flat-enforced", ("A", "B")).with_catalogue_text(("- A", "- B"))
        assert enforcement.describe()["catalogue_described"] is True
