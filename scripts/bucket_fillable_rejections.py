"""Why a fillable answer was rejected, for every name a model offered that
the corpus did not expect.

Three causes, and only one of them is the step's: a name the vocabulary calls
a defensible synonym of an expected one; a name Wikidata does state for that
entity but whose value text never grounded against the lead, recorded in the
harvest as a distractor rather than dropped outright; and a name Wikidata
holds no statement of at all. The third is the one that is not a reading
error, because there was nothing in Wikidata's structured side to have read.

Run after a step-fillable grid has written its jsonl:

    uv run python scripts/bucket_fillable_rejections.py results_local/step-fillable-*.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

from oold_llm_bench.corpus.wikidata_schemaorg import load_entities, read_grounded_corpus
from oold_llm_bench.experiments.corpora import documents_for
from oold_llm_bench.grading.vocabulary import read_hierarchy
from oold_llm_bench.steps.oracle import fillable_of

QID = re.compile(r"qid=(Q\d+)")


def tasks_by_qid() -> dict[str, object]:
    found = ((QID.search(task.notes or ""), task) for task in load_entities(documents_for()))
    return {match.group(1): task for match, task in found if match}


def entities_by_qid() -> dict[str, object]:
    return {entity.qid: entity for entity in read_grounded_corpus().entities}


def _cause(name: str, want: set[str], entity: object, hierarchy) -> str:
    """Which of the three causes a rejected property name falls under."""
    if any(hierarchy.near(name, other) for other in want):
        return "vocabulary: a defensible synonym of a stated slot"
    if entity is not None and name in (getattr(entity, "distractors", None) or {}):
        return "corpus dropped it: Wikidata has it, the lead does not state it"
    return "corpus has no such statement for this entity"


def _rejections(records: list[pathlib.Path], tasks: dict, entities: dict, hierarchy):
    """Every rejected name across every record, with its cause."""
    for path in records:
        if path.name.endswith(".published.jsonl") or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            found = QID.search(row.get("notes") or "")
            payload = row.get("answer_payload")
            if not found or not isinstance(payload, dict) or found.group(1) not in tasks:
                continue
            wanted = fillable_of(tasks[found.group(1)])
            entity = entities.get(found.group(1))
            for key, names in payload.items():
                if not isinstance(names, list):
                    continue
                want = set(wanted.get(key, ()))
                for name in set(names) - want:
                    yield name, _cause(name, want, entity, hierarchy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="+", type=pathlib.Path)
    args = parser.parse_args()

    hierarchy = read_hierarchy()
    tasks = tasks_by_qid()
    entities = entities_by_qid()

    bucket: collections.Counter[str] = collections.Counter()
    by_name: collections.Counter[str] = collections.Counter()
    for name, cause in _rejections(args.records, tasks, entities, hierarchy):
        bucket[cause] += 1
        if cause == "corpus has no such statement for this entity":
            by_name[name] += 1

    total = sum(bucket.values())
    print(f"{total} rejected property names")
    for label, count in bucket.most_common():
        print(f"  {count:6d}  ({count / total:.0%})  {label}")
    print("\ntop names in the unexplained bucket:")
    for name, count in by_name.most_common(15):
        print(f"  {count:5d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
