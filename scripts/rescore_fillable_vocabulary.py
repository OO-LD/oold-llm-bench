"""How much of the fillable precision loss is the vocabulary rather than the step.

Re-scores finished step-fillable records against the property hierarchy. No
model is called: the names each cell chose are in its ``answer_payload``, and
the corpus holds what the document states, so the only thing that changes is
the comparison.

The strict number stays in the output beside the lenient one. A lenient
number quoted alone would say the vocabulary gap does not exist.

    uv run python scripts/rescore_fillable_vocabulary.py results_local/step-fillable-*.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

from oold_llm_bench.corpus.wikidata_schemaorg import load_entities
from oold_llm_bench.experiments.corpora import documents_for
from oold_llm_bench.grading.score import Score
from oold_llm_bench.grading.triples import Dimension
from oold_llm_bench.grading.vocabulary import Relation, read_hierarchy
from oold_llm_bench.steps.oracle import fillable_of
from oold_llm_bench.steps.score import score_fillable

QID = re.compile(r"qid=(Q\d+)")


def tasks_by_qid() -> dict[str, object]:
    found = ((QID.search(task.notes or ""), task) for task in load_entities(documents_for()))
    return {match.group(1): task for match, task in found if match}


def totals(rows: list[Score]) -> Score:
    return Score(
        true_positives=sum(r.true_positives for r in rows),
        false_positives=sum(r.false_positives for r in rows),
        false_negatives=sum(r.false_negatives for r in rows),
    )


def cells(paths: list[pathlib.Path], tasks: dict[str, object]):
    """Every finished fillable cell whose task is still in the corpus.

    A record naming a task the corpus no longer holds is skipped rather than
    scored against a neighbour: the corpus is rebuilt and ids do not survive
    a rebuild that dropped an entity.
    """
    for path in paths:
        if path.name.endswith(".published.jsonl") or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            found = QID.search(row.get("notes") or "")
            payload = row.get("answer_payload")
            if not found or not isinstance(payload, dict) or found.group(1) not in tasks:
                continue
            produced = {key: tuple(names) for key, names in payload.items() if isinstance(names, list)}
            yield row.get("model", {}).get("model", "?"), tasks[found.group(1)], produced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="+", type=pathlib.Path)
    args = parser.parse_args()

    hierarchy = read_hierarchy()
    tasks = tasks_by_qid()
    strict: dict[str, list[Score]] = collections.defaultdict(list)
    lenient: dict[str, list[Score]] = collections.defaultdict(list)
    forgiven: collections.Counter[tuple[str, str, str]] = collections.Counter()

    for model, task, produced in cells(args.records, tasks):
        scored = score_fillable(task, produced, vocabulary=hierarchy)
        strict[model].append(scored[Dimension.FILLABLE])
        lenient[model].append(scored[Dimension.FILLABLE_NEAR])
        wanted = fillable_of(task)
        for key, names in produced.items():
            want = set(wanted.get(key, ()))
            for name in set(names) - want:
                near = [(o, hierarchy.relation(name, o)) for o in sorted(want - set(names))]
                hit = next(((o, r) for o, r in near if r is not Relation.NONE), None)
                if hit:
                    forgiven[(name, hit[0], hit[1].value)] += 1

    print(f"{'model':34s} {'strict':>22s}   {'vocabulary-aware':>22s}")
    print(f"{'':34s} {'P':>6s} {'R':>6s} {'F1':>7s}   {'P':>6s} {'R':>6s} {'F1':>7s}   dF1")
    for model in sorted(strict):
        a, b = totals(strict[model]), totals(lenient[model])
        print(
            f"{model[:34]:34s} {a.precision:6.3f} {a.recall:6.3f} {a.f1:7.3f}   "
            f"{b.precision:6.3f} {b.recall:6.3f} {b.f1:7.3f}   {b.f1 - a.f1:+.3f}"
        )
    if strict:
        a = totals([s for rows in strict.values() for s in rows])
        b = totals([s for rows in lenient.values() for s in rows])
        print(
            f"{'pooled':34s} {a.precision:6.3f} {a.recall:6.3f} {a.f1:7.3f}   "
            f"{b.precision:6.3f} {b.recall:6.3f} {b.f1:7.3f}   {b.f1 - a.f1:+.3f}"
        )

    print(f"\nforgiven pairs, {sum(forgiven.values())} in all:")
    for name, other, relation in sorted(forgiven, key=lambda k: -forgiven[k])[:20]:
        print(
            f"  {forgiven[(name, other, relation)]:5d}  answered {name:20s} where {other:20s} was stated [{relation}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
