"""What kind of mistake a step-extract cell made, bucketed from stored
records.

Re-scores each cell's own ``answer_payload`` with the current grader, so a
grading fix made after a grid ran (the array-capable slot schema, the
vocabulary-aware VALUE/PROPERTY/CLASS dimensions) is reflected without
calling a model again. No network, no API cost.

Five buckets per missed value, in the order they are tried:

    right value, wrong slot name            -- VALUE_NEAR recovers it
    right value, forgivable wrong class     -- CLASS_NEAR recovers it
    value simply missing (recall)           -- the step never answered it
    value invented (precision)              -- produced, not expected
    value present but compares unequal      -- same slot, wrong content

    uv run python scripts/bucket_extract_faults.py results_local/step-extract-*.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

from oold_llm_bench.corpus.wikidata_schemaorg import load_entities
from oold_llm_bench.experiments.corpora import documents_for
from oold_llm_bench.extract.json_answer import extract_json
from oold_llm_bench.grading.score import score_task

QID = re.compile(r"qid=(Q\d+)")


def tasks_by_qid() -> dict[str, object]:
    found = ((QID.search(task.notes or ""), task) for task in load_entities(documents_for()))
    return {match.group(1): task for match, task in found if match}


def _rescored(records: list[pathlib.Path], tasks: dict):
    """Every cell's model and re-scored result, read from stored answers."""
    for path in records:
        if path.name.endswith(".published.jsonl") or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            found = QID.search(row.get("notes") or "")
            payload = row.get("answer_payload")
            if not found or payload is None or found.group(1) not in tasks:
                continue
            model = row.get("model", {}).get("model", "?")
            yield model, score_task(tasks[found.group(1)], extract_json(payload))


def _tally(scored, bucket: collections.Counter[str]) -> list[tuple[str, str | None]]:
    """Add one cell's faults to its model's bucket, return its class errors."""
    dims = scored.dimensions
    value_strict = dims["value"]
    value_near = dims.get("value_near", value_strict)
    prop_near = dims.get("property_near")
    class_near = dims.get("class_near")

    # A miss that VALUE_NEAR recovers named the value under a defensible
    # synonym; the rest is the property grader's own business and is read
    # off PROPERTY_NEAR, not duplicated here.
    bucket["right value, wrong slot name (VALUE_NEAR recovers it)"] += (
        value_near.true_positives - value_strict.true_positives
    )
    bucket["value simply missing (recall)"] += value_near.false_negatives
    bucket["value invented (precision)"] += value_strict.false_positives
    if prop_near is not None:
        bucket["slot named under a forgivable synonym (PROPERTY_NEAR)"] += (
            prop_near.true_positives - dims["property"].true_positives
        )
    key = (
        "wrong class, forgivable (CLASS_NEAR: ancestor/descendant)"
        if class_near is not None
        else "wrong class, no lineage to judge it against"
    )
    bucket[key] += len(scored.class_errors)
    return scored.class_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="+", type=pathlib.Path)
    args = parser.parse_args()

    tasks = tasks_by_qid()
    by_model: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    class_confusions: collections.Counter[tuple[str, str | None]] = collections.Counter()
    wrong_classes_by_model: dict[str, collections.Counter[str | None]] = collections.defaultdict(collections.Counter)

    for model, scored in _rescored(args.records, tasks):
        for expected_class, produced_class in _tally(scored, by_model[model]):
            wrong_classes_by_model[model][produced_class] += 1
            class_confusions[(expected_class, produced_class)] += 1

    print(f"{'model':22s} fault, as a share of its own total")
    for model, bucket in sorted(by_model.items()):
        total = sum(bucket.values()) or 1
        print(f"\n{model}")
        for label, count in bucket.most_common():
            print(f"  {count:6d}  ({count / total:.0%})  {label}")

    print("\nwhat a wrong class actually was, by model:")
    for model, counter in sorted(wrong_classes_by_model.items()):
        top = ", ".join(f"{cls or '(none)'} x{n}" for cls, n in counter.most_common(3))
        print(f"  {model:22s} {top}")

    print("\nmost confused class pairs (expected -> produced):")
    for (expected, produced), count in class_confusions.most_common(10):
        print(f"  {count:4d}  {expected} -> {produced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
