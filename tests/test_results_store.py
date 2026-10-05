"""Keeping results, and reading them back as a table.

The store exists so a run that dies at cell four hundred keeps the first three
hundred and ninety-nine. The table exists so nobody reads a mean pooled over
things that were not held constant.
"""

import json

import pytest

from oold_llm_bench.report import tabulate
from oold_llm_bench.results import ResultStore, read_records
from oold_llm_bench.results.record import Environment, ModelSpec, RunRecord


def record(arm: str = "schema-dump-catalog-flat-enforced", primary: float = 1.0, run_id: str = "r1") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        arm=arm,
        model=ModelSpec(
            model="gpt-5-nano",
            provider_profile="openai",
            model_version="2025-08-07",
            deployment="a-deployment",
            endpoint="https://example.invalid",
        ),
        environment=Environment(benchmark_version="0.1.0", benchmark_sha="abc"),
        enforcement={"catalogue_size": 25, "pin_units": True, "signal": "unit"},
        corpus_hash="c" * 64,
        catalogue_hash="d" * 64,
        prompt_hash="e" * 64,
        split="dev",
        variant="native",
        repetition=1,
        scores=[
            {
                "primary_f1": primary,
                "dimensions": {"unit": {"f1": primary}, "class": {"f1": 1.0}},
            }
        ],
        calls={"calls": [{"input_tokens": 100, "output_tokens": 10}]},
    )


class TestWritingResults:
    def test_both_files_are_written_together(self, tmp_path):
        """A published file produced by a later step is one nobody runs."""
        store = ResultStore(tmp_path, "run-1")
        store.append(record())
        assert store.local_path.exists()
        assert store.published_path.exists()

    def test_the_published_file_carries_no_account_identifier(self, tmp_path):
        store = ResultStore(tmp_path, "run-1")
        store.append(record())
        text = store.published_path.read_text(encoding="utf-8")
        assert "a-deployment" not in text
        assert "example.invalid" not in text

    def test_the_local_file_keeps_the_deployment(self, tmp_path):
        """So a run can be repeated on the same infrastructure."""
        store = ResultStore(tmp_path, "run-1")
        store.append(record())
        assert "a-deployment" in store.local_path.read_text(encoding="utf-8")

    def test_publish_only_writes_no_local_copy(self, tmp_path):
        store = ResultStore(tmp_path, "run-1", publish_only=True)
        store.append(record())
        assert not store.local_path.exists()
        assert store.published_path.exists()

    def test_records_survive_a_run_that_stops_early(self, tmp_path):
        """Flushed per record, so a kill leaves what was already measured."""
        store = ResultStore(tmp_path, "run-1")
        store.append(record(run_id="a"))
        store.append(record(run_id="b"))
        assert len(list(read_records(store.local_path))) == 2

    def test_a_summary_lands_beside_the_records(self, tmp_path):
        store = ResultStore(tmp_path, "run-1")
        path = store.write_summary({"config_sha256": "x", "test_touches": 0})
        assert json.loads(path.read_text(encoding="utf-8"))["test_touches"] == 0


class TestReadingResults:
    def test_a_blank_line_is_skipped(self, tmp_path):
        path = tmp_path / "r.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
        assert len(list(read_records(path))) == 2

    def test_a_truncated_line_raises(self, tmp_path):
        """The normal state of a killed run. Reading it as whole would put a
        wrong number in a table."""
        path = tmp_path / "r.jsonl"
        path.write_text('{"a": 1}\n{"a": ', encoding="utf-8")
        with pytest.raises(ValueError, match="line 2 is not a whole record"):
            list(read_records(path))


class TestTheTable:
    def rows(self, *records):
        return tabulate([r.publish() for r in records])

    def test_a_row_is_keyed_on_condition_and_model(self):
        table = self.rows(record())
        assert table.conditions() == ["schema-dump-catalog-flat-enforced/unit/n25/units"]
        assert table.models() == ["gpt-5-nano"]

    def test_two_arms_are_not_pooled(self):
        """Pooling over arms is what made the predecessor unreadable."""
        table = self.rows(
            record(arm="schema-dump-catalog-not-enforced-gated"), record(arm="schema-dump-catalog-flat-enforced")
        )
        assert len(table.conditions()) == 2

    def test_repetitions_of_one_cell_are_pooled(self):
        table = self.rows(record(primary=1.0), record(primary=0.0))
        cell = table.get("schema-dump-catalog-flat-enforced/unit/n25/units", "gpt-5-nano")
        assert cell.n == 2
        assert cell.mean == pytest.approx(0.5)

    def test_dispersion_is_reported(self):
        table = self.rows(record(primary=1.0), record(primary=0.0))
        assert table.get("schema-dump-catalog-flat-enforced/unit/n25/units", "gpt-5-nano").stdev > 0

    def test_one_observation_reports_no_dispersion_and_says_so(self):
        table = self.rows(record())
        assert table.get("schema-dump-catalog-flat-enforced/unit/n25/units", "gpt-5-nano").stdev == 0.0
        assert table.single_observation() == [("schema-dump-catalog-flat-enforced/unit/n25/units", "gpt-5-nano")]

    def test_dimensions_come_through(self):
        table = self.rows(record(primary=0.5))
        assert table.get("schema-dump-catalog-flat-enforced/unit/n25/units", "gpt-5-nano").dimension("class") == 1.0

    def test_tokens_are_summed(self):
        table = self.rows(record(), record())
        assert table.get("schema-dump-catalog-flat-enforced/unit/n25/units", "gpt-5-nano").input_tokens == 200

    def test_a_record_with_no_score_counts_as_a_failure(self):
        broken = record()
        broken.scores = []
        table = self.rows(broken)
        assert table.get("schema-dump-catalog-flat-enforced/unit/n25/units", "gpt-5-nano").failures == 1

    def test_saturation_is_reported(self):
        assert self.rows(record(primary=1.0)).saturated() == [
            ("schema-dump-catalog-flat-enforced/unit/n25/units", "gpt-5-nano")
        ]

    def test_the_table_renders(self):
        rendered = self.rows(record()).render()
        assert "gpt-5-nano" in rendered
        assert "schema-dump-catalog-flat-enforced/unit/n25/units" in rendered

    def test_the_table_is_serialisable(self):
        assert json.dumps(self.rows(record()).describe())


class TestCostingAnOrchestration:
    """A two-step run sends the document twice, so the enum saving is not the
    cost story. The table reports the total per task."""

    def cell_of(self, table):
        cell = table.get("schema-dump-catalog-flat-enforced/unit/n25/units", "gpt-5-nano")
        assert cell is not None
        return cell

    def record_with(self, calls, doc_chars=400, run_id="r1"):
        made = record(run_id=run_id)
        made.calls = {"calls": calls}
        made.document_chars = doc_chars
        return made

    def test_tokens_are_reported_per_task_not_as_a_sum(self):
        table = tabulate([
            self.record_with([{"input_tokens": 100, "output_tokens": 10}]).publish(),
            self.record_with([{"input_tokens": 300, "output_tokens": 30}]).publish(),
        ])
        cell = self.cell_of(table)
        assert cell.input_per_task == pytest.approx(200.0)
        assert cell.output_per_task == pytest.approx(20.0)

    def test_every_step_counts_towards_the_task(self):
        """Counting only the last call would hide the select step entirely."""
        table = tabulate([
            self.record_with([
                {"input_tokens": 900, "output_tokens": 20},
                {"input_tokens": 600, "output_tokens": 40},
            ]).publish()
        ])
        cell = self.cell_of(table)
        assert cell.input_per_task == pytest.approx(1500.0)
        assert cell.calls_per_task == pytest.approx(2.0)

    def test_the_document_length_is_reported_beside_the_tokens(self):
        table = tabulate([self.record_with([{"input_tokens": 100}], doc_chars=2400).publish()])
        assert self.cell_of(table).document_size == pytest.approx(2400.0)

    def test_the_cost_fields_reach_the_description(self):
        table = tabulate([self.record_with([{"input_tokens": 100}]).publish()])
        described = self.cell_of(table).describe()
        assert {"input_per_task", "calls_per_task", "document_chars"} <= set(described)
