"""How often a flat grammar's licence to be wrong is actually taken.

A `flat-enforced` condition constrains the class and the unit as two
independent enumerations: a hundred class identifiers and six hundred and
five unit identifiers on the quantity pool. Nothing ties them, so
``{"type": "Altitude", "unit": "kilogram"}`` satisfies the grammar. An
`enforced` condition is an ``anyOf`` over one branch per class, each carrying
only the units that class admits, and the same answer cannot be produced.

The difference is only worth the tokens it costs if models take the licence.
This counts how often they do, over every answered entity in every
flat-enforced cell on disk: a produced unit that is not in the produced
class's own set.

The grader already marks those answers wrong, so the scores they appear in
are honest. What the count says is something else, about the arm rather than
the model: a condition named for enforcement was admitting an answer its
corpus forbids, at a rate worth stating beside any number measured under it.

Reads the local records rather than the published ones, because the produced
payload is local by design and the published allowlist omits it.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

from oold_llm_bench.corpus import QUANTITIES, resolve_module
from oold_llm_bench.report.axes import LEGACY_ARM_NAMES

FLAT_ARMS = {
    "schema-dump-catalog-flat-enforced",
    "schema-dump-catalog-flat-enforced-strict",
    "schema-dump-catalog-flat-enforced-grounded",
    "catalog-flat-enforced",
}
"""Every arm whose grammar pins the slots independently of each other."""

MIN_ANSWERS = 30
"""Below this a rate is one or two answers and reads as precision it has not
earned."""


def unit_sets(schemas: pathlib.Path) -> dict[str, set[str]]:
    from oold_llm_bench.corpus import load_kinds

    return {kind.name: set(kind.units) for kind in load_kinds(schemas) if kind.units}


def answered(record: dict) -> list[dict]:
    """The entities a cell produced, or nothing when it produced none."""
    payload = record.get("answer_payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, dict):
        return []
    found = payload.get("entities")
    return [e for e in found if isinstance(e, dict)] if isinstance(found, list) else []


def flat_records(results: pathlib.Path):
    """Every locally recorded cell that ran under a flat enumeration."""
    for path in sorted(results.glob("*.jsonl")):
        if "published" in path.name:
            continue
        for line in path.open(encoding="utf-8"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            arm = LEGACY_ARM_NAMES.get(record.get("arm", ""), record.get("arm"))
            if arm in FLAT_ARMS and (record.get("enforcement") or {}).get("pin_units"):
                yield record


def count(records, units: dict[str, set[str]]) -> dict[tuple[str, str], list[int]]:
    """Impossible pairs and answered entities, per model and arm."""
    tally: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    for record in records:
        arm = LEGACY_ARM_NAMES.get(record["arm"], record["arm"])
        key = (record["model"]["model"], arm)
        for entity in answered(record):
            name, unit = entity.get("type"), entity.get("unit")
            if not name or not isinstance(unit, str) or name not in units:
                continue
            tally[key][1] += 1
            if unit not in units[name]:
                tally[key][0] += 1
    return tally


def main() -> int:
    schemas = resolve_module(QUANTITIES, pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    results = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "results_local")
    units = unit_sets(schemas)

    tally = count(flat_records(results), units)

    print(f"{'model':26s} {'arm':42s} {'answers':>8s} {'impossible':>11s} {'rate':>7s}")
    total = [0, 0]
    for key, (bad, seen) in sorted(tally.items(), key=lambda item: -item[1][0] / max(item[1][1], 1)):
        total[0] += bad
        total[1] += seen
        if seen < MIN_ANSWERS:
            continue
        print(f"{key[0][-26:]:26s} {key[1]:42s} {seen:8d} {bad:11d} {bad / seen:7.1%}")
    if total[1]:
        print(f"\n{'ALL':26s} {'':18s} {total[1]:8d} {total[0]:11d} {total[0] / total[1]:7.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
