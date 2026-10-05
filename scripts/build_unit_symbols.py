"""Derive the published symbol for each unit the quantity corpus enumerates.

A document writes ``km``, not ``kilo_meter``. The symbol is not ours to invent:
the quantity schemas map every member of a unit enumeration to a QUDT unit in
their ``@context``, and QUDT publishes a symbol for that unit. This script
follows the one to the other and commits the result, so a run needs no network
and no second checkout.

Two inputs. The canonical quantity schemas supply the unit names the corpus
uses and the QUDT unit each one means. ``qudt_units.json`` supplies the symbol,
taken from the QUDT release the schemas were generated against.

The output is keyed by unit name, because 1,114 of the 1,125 names the corpus
draws mean one QUDT unit wherever they appear. The other 11 do not: ``meter``
is ``m`` under ``Length`` and ``m³/m²`` under ``VolumePerUnitArea``, so those
are written out per kind instead, with the inheritance already followed.

Run this when the schemas or the QUDT pin change.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

QUNIT = "qunit:"
"""The prefix the schemas use for a QUDT unit. The value behind it is the
unit's local name, ``KiloM``, which keys ``qudt_units.json``."""


def _unit_context(schema: dict) -> dict[str, str]:
    """The unit name to QUDT unit map one schema declares."""
    found: dict[str, str] = {}
    for entry in schema.get("@context") or []:
        if not isinstance(entry, dict):
            continue
        for name, target in entry.items():
            if isinstance(target, str) and target.startswith(QUNIT):
                found[name] = target[len(QUNIT) :]
    return found


def read_schemas(directory: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Each schema's declared unit map and its parent.

    The parent is kept because a subclass declares no units and no unit
    context, exactly as ``load_kinds`` finds it, so the chain has to be walked
    for the 505 subclasses to be reachable at all.
    """
    declared: dict[str, dict[str, str]] = {}
    parents: dict[str, str] = {}
    for path in sorted(directory.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        name = schema.get("title") or path.stem
        declared[name] = _unit_context(schema)
        for entry in schema.get("allOf") or []:
            ref = entry.get("$ref", "")
            if ref.endswith(".schema.json"):
                parents[name] = ref[: -len(".schema.json")]
    return declared, parents


def resolve(declared: dict[str, dict[str, str]], parents: dict[str, str]) -> dict[str, dict[str, str]]:
    """Every kind's effective unit map, a subclass inheriting its parent's."""

    def effective(name: str, seen: frozenset[str] = frozenset()) -> dict[str, str]:
        if declared.get(name) or name in seen:
            return declared.get(name, {})
        parent = parents.get(name)
        if not parent or parent not in declared:
            return {}
        return effective(parent, seen | {name})

    return {name: effective(name) for name in declared}


def build(schemas: Path, qudt_units: Path, out: Path) -> dict:
    units = json.loads(qudt_units.read_text(encoding="utf-8"))["units"]
    declared, parents = read_schemas(schemas)

    per_kind: dict[str, dict[str, str]] = {}
    for kind, mapping in resolve(declared, parents).items():
        symbols = {name: (units.get(target) or {}).get("symbol") for name, target in mapping.items()}
        per_kind[kind] = {name: symbol for name, symbol in symbols.items() if symbol}

    spellings: dict[str, set[str]] = defaultdict(set)
    for mapping in per_kind.values():
        for name, symbol in mapping.items():
            spellings[name].add(symbol)

    shared = {name for name, found in spellings.items() if len(found) > 1}
    payload = {
        # Names and not paths, because a committed artefact that records where
        # one machine kept its checkout is not provenance anyone can use.
        "source": schemas.name,
        "qudt_units": qudt_units.name,
        "built_at": datetime.now(UTC).date().isoformat(),
        "note": (
            "The symbol QUDT publishes for the unit each enumeration member "
            "denotes. 'by_kind' holds the names that denote different units "
            "in different kinds, which 'symbols' cannot express."
        ),
        "symbols": {name: next(iter(found)) for name, found in sorted(spellings.items()) if name not in shared},
        "by_kind": {
            kind: dict(sorted((n, s) for n, s in mapping.items() if n in shared))
            for kind, mapping in sorted(per_kind.items())
            if any(n in shared for n in mapping)
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemas", type=Path, required=True, help="the canonical quantity schemas")
    parser.add_argument("--qudt-units", type=Path, required=True, help="qudt_units.json from the schema generator")
    parser.add_argument("--out", type=Path, default=Path("src/oold_llm_bench/data/unit_symbols.json"))
    args = parser.parse_args()
    payload = build(args.schemas, args.qudt_units, args.out)
    print(f"wrote {args.out}")
    print(f"  {len(payload['symbols'])} unit names with one symbol")
    print(f"  {len(payload['by_kind'])} kinds carrying a name that means a different unit elsewhere")


if __name__ == "__main__":
    main()
