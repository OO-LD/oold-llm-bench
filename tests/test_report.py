"""What the condition key has to keep apart.

Every entry here is a pooling the plan forbids. A table that averaged across
one of them would report a single number for two different treatments, and
nothing downstream would show that it had happened.
"""

from __future__ import annotations

from typing import Any

import pytest

from oold_llm_bench.report import tabulate
from oold_llm_bench.report.table import Cell, _fidelity_band


def _record(**overrides: Any) -> dict[str, Any]:
    record = {
        "arm": "schema-dump-catalog-enforced",
        "model": {"model": "gpt-5-nano"},
        "enforcement": {"signal": "named", "vocabulary": "consensus", "catalogue_size": 25},
        "scores": [{"primary_f1": 1.0, "dimensions": {}}],
        "document_chars": 400,
        "calls": {"calls": []},
    }
    record.update(overrides)
    return record


def test_a_kept_union_and_a_flattened_one_are_separate_conditions() -> None:
    """The rule is about providers, and the provider is the model."""
    kept = _record(degradation={"fidelity": 1.09})
    flattened = _record(model={"model": "claude-haiku-4-5"}, degradation={"fidelity": 0.02})
    table = tabulate([kept, flattened])
    assert len(table.cells) == 2
    bands = sorted(cell.condition.rsplit("/", 1)[-1] for cell in table.cells.values())
    assert bands == ["fid<0.5", "fid>1.0"]


def test_one_model_spanning_two_bands_stays_one_row_and_says_so() -> None:
    """Both bands in the label, because that group should not have been one.

    Forking it instead was the fault: records that carried no fidelity at all
    got a row of their own and the rows they left behind read too high.
    """
    table = tabulate([_record(degradation={"fidelity": 1.09}), _record(degradation={"fidelity": 0.02})])
    assert len(table.conditions()) == 1
    assert table.conditions()[0].endswith("fid<0.5|>1.0")


def test_a_record_with_no_fidelity_joins_its_siblings() -> None:
    """A run that stopped before the provider saw the schema still counts.

    It is the same condition, so its score belongs in the same cell. Keying on
    what the record happens to hold put 179 answers of zero in rows of their
    own.
    """
    table = tabulate([_record(degradation={"fidelity": 1.09}), _record(degradation={})])
    assert len(table.conditions()) == 1
    cell = table.get(table.conditions()[0], "gpt-5-nano")
    assert cell is not None
    assert cell.n == 2


def test_nearby_fidelities_pool() -> None:
    """Two draws of the same treatment stay one row.

    Fidelity moves with the catalogue a task drew, so exact values would put
    every task in a row of its own and the table would carry no cell with
    more than one observation.
    """
    table = tabulate([_record(degradation={"fidelity": 1.06}), _record(degradation={"fidelity": 1.12})])
    assert len(table.conditions()) == 1
    cell = table.get(table.conditions()[0], "gpt-5-nano")
    assert cell is not None
    assert cell.n == 2


def test_orchestration_is_in_the_key() -> None:
    single = _record(enforcement={"orchestration": "single_shot"})
    two_step = _record(enforcement={"orchestration": "select_then_fill", "shortlist_k": 3})
    assert len(tabulate([single, two_step]).conditions()) == 2


def test_shortlist_k_is_in_the_key() -> None:
    """k=1 is the commit case and k=3 is not, so they never share a row."""
    one = _record(enforcement={"orchestration": "select_then_fill", "shortlist_k": 1})
    three = _record(enforcement={"orchestration": "select_then_fill", "shortlist_k": 3})
    assert len(tabulate([one, three]).conditions()) == 2


def test_models_are_never_pooled() -> None:
    table = tabulate([_record(), _record(model={"model": "claude-haiku-4-5"})])
    assert len(table.models()) == 2


def test_bands_cover_the_range() -> None:
    assert _fidelity_band(0.02) == "<0.5"
    assert _fidelity_band(0.49) == "<0.5"
    assert _fidelity_band(0.5) == "0.5-0.95"
    assert _fidelity_band(0.94) == "0.5-0.95"
    assert _fidelity_band(0.95) == "~1.0"
    assert _fidelity_band(1.0) == "~1.0"
    assert _fidelity_band(1.09) == ">1.0"


def _only_cell(record: dict[str, Any]) -> Cell:
    table = tabulate([record])
    cell = table.get(table.conditions()[0], "gpt-5-nano")
    assert cell is not None
    return cell


def test_entity_precision_and_recall_are_kept_apart() -> None:
    """One F1 cannot say whether an entity was missed or invented."""
    cell = _only_cell(
        _record(
            scores=[
                {
                    "primary_f1": 0.5,
                    "dimensions": {"entity": {"f1": 0.8889, "precision": 1.0, "recall": 0.8}},
                }
            ]
        )
    )
    assert cell.dimension("entity:precision") == pytest.approx(1.0)
    assert cell.dimension("entity:recall") == pytest.approx(0.8)
    assert cell.dimension("entity") == pytest.approx(0.8889)


def test_a_dimension_no_record_reported_prints_a_dash() -> None:
    """0.00 would read as a measured failure of something never measured."""
    table = tabulate([_record()])
    cell = _only_cell(_record())
    assert cell.dimension("duplicate") is None
    assert "duplicate" not in cell.describe()["dimensions"]
    row = table.render(dimensions=("duplicate",)).splitlines()[-1]
    assert row.split()[-5] == "-"


def test_the_default_shows_what_a_multi_entity_answer_can_get_wrong() -> None:
    """Missing, inventing and repeating an entity need three columns.

    A default that omitted one of them would leave the failure in the triple
    F1 with the other two, where nothing separates them."""
    header = tabulate([_record()]).render().splitlines()[0]
    for name in ("class", "value", "unit", "entity:precision", "entity:recall", "duplicate"):
        assert name in header


def test_dimension_columns_do_not_run_into_each_other() -> None:
    header = tabulate([_record()]).render().splitlines()[0]
    assert "entity:precisionentity:recall" not in header


def test_condition_column_fits_the_longest_key() -> None:
    """A fixed width silently misaligned every column to its right."""
    long_key = _record(
        enforcement={
            "signal": "named",
            "vocabulary": "consensus",
            "catalogue_size": 25,
            "describe_catalogue": True,
            "orchestration": "select_then_fill",
            "shortlist_k": 3,
        },
        degradation={"fidelity": 1.09},
    )
    rendered = tabulate([long_key]).render(dimensions=("class",)).splitlines()
    header, _, row = rendered
    assert row.index("gpt-5-nano") == header.index("model")


class TestTheRetryIsNotTheOrchestration:
    """Re-asking a failed plan call is a mitigation, not a pipeline.

    gpt-5-nano returns the selection schema instead of a plan on about a third
    of plan calls, 37 of 120 under segmented against 0 of 120 under single
    shot, and that costs the whole cell. A retry recovers it. Pooled with the
    plain arm, the gain would read as the orchestration's.
    """

    def _seg(self, **extra):
        return _record(enforcement={"orchestration": "segmented", "shortlist_k": 3, **extra})

    def test_a_retried_arm_is_its_own_row(self):
        table = tabulate([self._seg(), self._seg(plan_retry=True)])
        assert len(table.conditions()) == 2

    def test_the_retry_is_named_in_the_key(self):
        (condition,) = tabulate([self._seg(plan_retry=True)]).conditions()
        assert condition.endswith("retry")

    def test_an_arm_without_it_is_not_labelled(self):
        (condition,) = tabulate([self._seg()]).conditions()
        assert "retry" not in condition


def test_reasoning_is_in_the_key():
    """Added to Condition.key and missed in the report key, so for a day the
    report pooled a thinking-off cell with a thinking-on one."""
    from oold_llm_bench.report.table import _condition_key
    from oold_llm_bench.runner import Condition

    off = Condition(
        arm="schema-dump-catalog-flat-enforced", signal="unit", vocabulary="consensus", reasoning="off"
    ).describe()
    plain = Condition(arm="schema-dump-catalog-flat-enforced", signal="unit", vocabulary="consensus").describe()
    assert _condition_key(off, off) != _condition_key(plain, plain)


def test_every_arm_the_library_defines_has_a_published_name():
    """An arm whose name does not parse has no place in a result table.

    Labelling it anyway would put a constrained cell in the table as
    unconstrained, which is a condition somebody might have run, so an arm
    that names neither half raises instead.
    """
    import pytest

    from oold_llm_bench.report.axes import label_of

    oold = pytest.importorskip("oold.agent.enforcement")
    for arm in oold.ARMS:
        assert label_of(arm, {})
    with pytest.raises(KeyError):
        label_of("an-arm-nobody-declared", {})


def test_a_record_written_before_the_rename_reaches_the_same_label():
    """Published results carry the arm id they were written with.

    A result file is evidence and is never edited to match current
    vocabulary, so the reader translates and both sides of the rename land on
    one label.
    """
    from oold_llm_bench.report.axes import label_of

    assert label_of("A4-blind", {}) == label_of("no-catalog-enforced", {}) == "no-catalog-enforced"
    assert label_of("A2-enforced-only", {}) == "schema-prose-catalog-flat-enforced"
    assert label_of("A2-enforced-only", {"describe_catalogue": True}) == "schema-prose-full-catalog-flat-enforced"


def test_a_qualifier_stays_in_the_published_name():
    """Or a strict cell and a lenient one pool into a single row.

    The two send the same schema and differ in whether the provider enforces
    it, which is the whole contrast the qualifier exists to make.
    """
    from oold_llm_bench.report.axes import label_of

    assert label_of("A2", {}) == "schema-dump-catalog-flat-enforced"
    assert label_of("A2-strict", {}) == "schema-dump-catalog-flat-enforced-strict"
    assert label_of("A0-prose", {}) == "no-catalog-not-enforced-prose"
