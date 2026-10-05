"""Turn the Wiki-Measurements archive into a gradeable corpus.

Four inputs, none of them ours. The Zenodo archive supplies the annotated
sentences and the Wikidata facts they were matched against. The canonical
quantity schemas supply the unit enumeration each quantity kind admits and,
through their ``@context``, the QUDT unit every member denotes. ``qudt_units``
supplies the symbol and the UCUM code QUDT publishes for that unit. Wikidata
supplies the QUDT unit id and the UCUM code of each unit item, which is the
only published route from a fact's unit to the enumeration.

The output is committed, so a run needs no network, no archive and no second
checkout. The archive stays wherever it was downloaded: it is 98 MB, the repo
is not a data store, and nothing in a run reads it. ``--cap`` is what keeps
the committed file small, and it is set where 250 examples per kind fit under
the large-file hook: 480 KiB against the hook's 500.

Two things this refuses to do silently. Every example that does not become a
task is counted under a stated reason, and the counts have to add up to the
examples that went in. And the two ``Area`` enumeration members whose name
contradicts the QUDT unit they denote are re-checked against the upstream
``@context`` here, so the declaration in
:mod:`oold_llm_bench.corpus.wiki_measurements` cannot outlive the defect.

Run this when the archive, the schemas or the QUDT pin changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oold_llm_bench.corpus.wiki_measurements import (
    DATASET_DOI,
    MISLABELLED_UNITS,
    PROPERTY_KIND,
    TEXT_LICENCE,
)
from oold_llm_bench.grading.compare import normalise_unit, unit_forms

DATASET_FILE = "Wiki-Measurements/raw/wiki-measurements_large_strict.json"
FACTS_FILE = "Wiki-Measurements/raw/additional_data.json"
"""The strict set, where the fact was aligned to the sentence more tightly.

The large set is 38,738 examples against 26,009 here. The strict one is taken
because the loose alignment shows up as a fact that disagrees with the
sentence, and this corpus keeps only the examples where the two agree.
"""

VALUE_MARK = "\U0001f350"
UNIT_MARK = "\U0001f353"
_SELECTOR = "\ufe0f"
_CODEPOINTS = (
    "\U0001f336",
    "\U0001f34a",
    "\U0001f34f",
    "\U0001f350",
    "\U0001f353",
    "\u260e",
    "\U0001f4c6",
    "\u23f1",
    "\u23f0",
    "\U0001f4cd",
    "\U0001f64b",
    "\U0001f52d",
    "\U0001f4cf",
    "\U0001f9b5",
    "\U0001f50e",
    "\U0001f6c1",
)
"""What the dataset delimits each annotated span with, by codepoint.

Written as escapes rather than as the characters, and each one is admitted
both with and without the variation selector: the dataset uses it on some
marks and not on others, and a mark matched one character short moves every
offset after it.
"""

ORDERED_MARKS = tuple(
    sorted({mark + suffix for mark in _CODEPOINTS for suffix in ("", _SELECTOR)}, key=len, reverse=True)
)

WIKIDATA_QUERY = "SELECT ?item ?qudt ?ucum WHERE { { ?item wdt:P2968 ?qudt } UNION { ?item wdt:P7825 ?ucum } }"
"""QUDT unit id and UCUM code for every Wikidata unit that publishes one.

P2968 alone is not enough. Square kilometre, the single commonest unit in this
dataset, carries no QUDT id and does carry the UCUM code ``km2``, which QUDT's
own table resolves. Both are published identifiers, so neither is a guess.
"""

USER_AGENT = "oold-llm-bench/1.0 (+https://github.com/OO-LD/oold-llm-bench)"

NUMBER = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")
READING = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)[ \u00a0]([A-Za-z\u00b5\u03bc\u00b0][A-Za-z0-9\u00b2\u00b3\u00b5\u03bc\u00b0/.-]*)"
)
"""A magnitude followed by a token that might be a unit.

Used to find a reading the annotation does not cover. The space is required:
without it "39th" is a reading of 39 in some unit whose symbol folds to "th",
and an ordinal is not a measurement.
"""

QUNIT = "qunit:"

CREATORS = "Jan Göpfert, Patrick Kuckertz, Jann M. Weinand, Detlef Stolten (FZJ ICE-2, 2025)"
"""Who to attribute. CC BY-SA names the dataset's authors; each record names
the Wikipedia article whose history is the author list for its sentence."""

_SIGNS = {0x2212: "-", 0x2013: "-"}
"""The minus a sentence writes, folded onto the one ``float`` reads."""


def _signed(text: str) -> str:
    return text.translate(_SIGNS)


def annotate(text: str) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    """The sentence without its marks, and where each span landed in it."""
    plain: list[str] = []
    spans: dict[str, list[tuple[int, int]]] = {}
    opened: dict[str, int] = {}
    index = 0
    while index < len(text):
        for mark in ORDERED_MARKS:
            if text.startswith(mark, index):
                if mark in opened:
                    spans.setdefault(mark, []).append((opened.pop(mark), len(plain)))
                else:
                    opened[mark] = len(plain)
                index += len(mark)
                break
        else:
            plain.append(text[index])
            index += 1
    return "".join(plain), spans


def unit_context(schema: dict) -> dict[str, str]:
    """The unit name to QUDT unit map one schema declares."""
    found: dict[str, str] = {}
    for entry in schema.get("@context") or []:
        if isinstance(entry, dict):
            for name, target in entry.items():
                if isinstance(target, str) and target.startswith(QUNIT):
                    found[name] = target[len(QUNIT) :]
    return found


def read_schemas(directory: Path) -> dict[str, dict[str, str]]:
    """Every kind's effective unit map, a subclass inheriting its parent's.

    The chain is walked for the reason ``load_kinds`` walks it: the subclasses
    declare no units of their own, and ``Altitude``, ``Diameter`` and
    ``Height`` are all subclasses of ``Length``.
    """
    declared: dict[str, dict[str, str]] = {}
    parents: dict[str, str] = {}
    for path in sorted(directory.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        name = schema.get("title") or path.stem.removesuffix(".schema")
        declared[name] = unit_context(schema)
        for entry in schema.get("allOf") or []:
            ref = entry.get("$ref", "")
            if ref.endswith(".schema.json"):
                parents[name] = ref[: -len(".schema.json")]

    def effective(name: str, seen: frozenset[str] = frozenset()) -> dict[str, str]:
        if declared.get(name) or name in seen:
            return declared.get(name, {})
        parent = parents.get(name)
        if not parent or parent not in declared:
            return {}
        return effective(parent, seen | {name})

    return {name: effective(name) for name in declared}


def check_mislabelled(kinds: dict[str, dict[str, str]]) -> None:
    """Refuse to build if the declared defect is no longer in the schemas.

    A declaration that outlived what it describes is worse than no
    declaration: it keeps excluding examples for a reason that stopped being
    true, and nothing would say so.
    """
    expected = {("Area", "year"): "ARE", ("Area", "deca_year"): "DecaARE"}
    wrong = [
        f"{kind}.{unit} now denotes {kinds.get(kind, {}).get(unit)!r}, not qunit:{target}"
        for (kind, unit), target in expected.items()
        if kinds.get(kind, {}).get(unit) != target
    ]
    declared = {(kind, unit) for kind, units in MISLABELLED_UNITS.items() for unit in units}
    if declared != set(expected):
        wrong.append(f"MISLABELLED_UNITS declares {sorted(declared)}, this check knows {sorted(expected)}")
    if wrong:
        raise ValueError("the mislabelled-unit declaration no longer matches the schemas: " + "; ".join(wrong))


class UnitTables:
    """Four ways into one kind's enumeration, and the one way back out.

    ``symbols`` is keyed on the published symbol with its case intact, because
    ``mW`` and ``MW`` are two units and folding them loses the answer.
    ``folded`` is keyed on :func:`~oold_llm_bench.grading.compare.unit_forms`,
    which is where a plural, a British spelling and a written-out exponent are
    resolved. ``plain`` is the same three spellings under case folding alone,
    and exists only so the fold can be reported as a number rather than
    asserted. ``by_qudt`` takes a fact's unit back to the enumeration.

    A key two units of one kind answer to is dropped rather than guessed at.
    """

    def __init__(self, kinds: dict[str, dict[str, str]], qudt_units: dict[str, dict]) -> None:
        self.symbols: dict[str, dict[str, str]] = {}
        self.folded: dict[str, dict[str, str]] = {}
        self.plain: dict[str, dict[str, str]] = {}
        self.by_qudt: dict[str, dict[str, str]] = {}
        for kind, mapping in kinds.items():
            exact: dict[str, set[str]] = {}
            keys: dict[str, set[str]] = {}
            literal: dict[str, set[str]] = {}
            inverse: dict[str, str] = {}
            for unit, target in mapping.items():
                if unit in MISLABELLED_UNITS.get(kind, ()):
                    continue
                inverse.setdefault(target, unit)
                spellings = [unit, unit.replace("_", " ")]
                symbol = (qudt_units.get(target) or {}).get("symbol")
                if symbol:
                    spellings.append(symbol)
                    exact.setdefault(normalise_unit(symbol), set()).add(unit)
                for spelling in spellings:
                    literal.setdefault(spelling.strip().casefold(), set()).add(unit)
                    for key in unit_forms(spelling):
                        keys.setdefault(key, set()).add(unit)
            self.symbols[kind] = {k: next(iter(v)) for k, v in exact.items() if len(v) == 1}
            self.folded[kind] = {k: next(iter(v)) for k, v in keys.items() if len(v) == 1}
            self.plain[kind] = {k: next(iter(v)) for k, v in literal.items() if len(v) == 1}
            self.by_qudt[kind] = inverse

    def every_form(self) -> set[str]:
        """Every key any kind answers to, for spotting a competing reading."""
        found: set[str] = set()
        for table in self.folded.values():
            found |= set(table)
        return found

    def resolve(self, kind: str, written: str) -> str | None:
        """The enumeration member a written unit names within one kind."""
        exact = self.symbols[kind].get(normalise_unit(written))
        if exact:
            return exact
        return next((self.folded[kind][key] for key in unit_forms(written) if key in self.folded[kind]), None)

    def resolve_plain(self, kind: str, written: str) -> str | None:
        """The same lookup under case folding alone. The baseline, not a path."""
        return self.plain[kind].get(written.strip().casefold())


def wikidata_units(cache: Path, refresh: bool = False) -> dict[str, dict[str, list[str]]]:
    """The QUDT id and UCUM code of every Wikidata unit, fetched once."""
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({
        "query": WIKIDATA_QUERY,
        "format": "json",
    })
    request = urllib.request.Request(  # noqa: S310 - the url is a constant
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 - same constant
        payload = json.loads(response.read().decode("utf-8"))

    found: dict[str, dict[str, set[str]]] = {"qudt": {}, "ucum": {}}
    for row in payload["results"]["bindings"]:
        item = row["item"]["value"].rsplit("/", 1)[-1]
        for field in ("qudt", "ucum"):
            if field in row:
                found[field].setdefault(item, set()).add(row[field]["value"])
    table = {field: {k: sorted(v) for k, v in items.items()} for field, items in found.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(table, indent=0, sort_keys=True), encoding="utf-8", newline="\n")
    return table


def fact_unit(qid: str, kind: str, table: dict, by_ucum: dict, by_qudt: dict) -> str | None:
    """The enumeration member a Wikidata unit item means, within one kind."""
    targets = list(table["qudt"].get(qid) or [])
    for code in table["ucum"].get(qid) or []:
        targets += by_ucum.get(code) or []
    for target in targets:
        unit = by_qudt.get(kind, {}).get(target)
        if unit:
            return unit
    return None


def build(  # noqa: C901 - one pass, one stated reason per branch
    archive: Path,
    schemas: Path,
    qudt_units_path: Path,
    units_cache: Path,
    retrieved_at: str,
    cap: int | None,
    refresh: bool,
) -> dict[str, Any]:
    kinds = read_schemas(schemas)
    check_mislabelled(kinds)
    qudt = json.loads(qudt_units_path.read_text(encoding="utf-8"))
    tables = UnitTables(kinds, qudt["units"])
    table = wikidata_units(units_cache, refresh)
    every_form = tables.every_form()

    with zipfile.ZipFile(archive) as bundle:
        facts = {entry["id"]: entry for entry in json.loads(bundle.read(FACTS_FILE).decode("utf-8"))["data"]}
        source = json.loads(bundle.read(DATASET_FILE).decode("utf-8"))
    examples_in = source["nbr_examples"]

    excluded: Counter[str] = Counter()
    per_kind: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    attempted = resolved_folded = resolved_plain = 0

    for entry in source["data"]:
        fact = facts[entry["id"]]["wikidata_fact"]
        if facts[entry["id"]]["target"] != entry["source"]:
            raise ValueError(f"example {entry['id']} and its fact name different articles")
        kind = PROPERTY_KIND.get(fact["property"].rsplit("/", 1)[-1])
        if kind is None or kind not in kinds:
            excluded["the Wikidata property names no quantity kind this corpus has"] += 1
            continue
        document, spans = annotate(entry["text"])
        values, units = spans.get(VALUE_MARK) or [], spans.get(UNIT_MARK) or []
        if len(values) != 1 or len(units) != 1:
            excluded["the annotation does not mark exactly one value and one unit"] += 1
            continue
        value_at = values[0]
        value_text = document[value_at[0] : value_at[1]].strip()
        unit_text = document[units[0][0] : units[0][1]].strip()
        if not NUMBER.match(_signed(value_text)):
            excluded["the value span is not a plain number"] += 1
            continue

        attempted += 1
        span_unit = tables.resolve(kind, unit_text)
        resolved_folded += span_unit is not None
        resolved_plain += tables.resolve_plain(kind, unit_text) is not None
        if span_unit is None:
            excluded["the unit the sentence writes is not a unit of the kind"] += 1
            continue

        qid = (fact.get("unit") or "").rsplit("/", 1)[-1]
        confirmed = fact_unit(qid, kind, table, qudt["by_ucum"], tables.by_qudt)
        if confirmed is None:
            excluded["the unit the Wikidata fact carries is not a unit of the kind"] += 1
            continue
        if confirmed != span_unit:
            excluded["the Wikidata fact states a different unit from the sentence"] += 1
            continue

        competing = [
            found
            for found in READING.finditer(document)
            if any(key in every_form for key in unit_forms(found.group(2)))
            and not (found.start(1) < value_at[1] and value_at[0] < found.end(1))
        ]
        if competing:
            excluded["the sentence states a reading the annotation does not cover"] += 1
            continue

        per_kind[kind] += 1
        kept.append({
            "id": f"wm-{entry['id']}",
            "text": document.strip(),
            "kind": kind,
            "unit": span_unit,
            "magnitude": float(_signed(value_text).replace(",", "")),
            "value_text": value_text,
            "unit_text": unit_text,
            "article": fact["article"],
            "qudt_unit": kinds[kind][span_unit],
            "wikidata": {
                "entity": fact["entity"].rsplit("/", 1)[-1],
                "property": fact["property"].rsplit("/", 1)[-1],
                "unit": qid,
            },
        })

    kept.sort(key=lambda record: (record["kind"], record["id"]))
    selected = kept
    if cap is not None:
        taken: Counter[str] = Counter()
        selected = []
        for record in kept:
            if taken[record["kind"]] < cap:
                taken[record["kind"]] += 1
                selected.append(record)

    return {
        "schema_version": "1",
        "name": "Wiki-Measurements",
        "dataset": {
            "title": source["name"],
            "doi": DATASET_DOI,
            "url": f"https://doi.org/{DATASET_DOI}",
            "file": DATASET_FILE,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "retrieved_at": retrieved_at,
            "creators": source.get("creators") or CREATORS,
        },
        "licence": {
            "text": {"name": TEXT_LICENCE, "url": "https://creativecommons.org/licenses/by-sa/4.0/"},
            "facts": {"name": "CC0 1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"},
            "note": (
                "The sentences are Wikipedia's under CC BY-SA 4.0 and anything derived from them "
                "stays under it, which includes this file and any task set built from it. Each "
                "record names the article it came from, and that article's history is the author "
                "list. The Wikidata facts are CC0 1.0. This repository is Apache-2.0; these terms "
                "are the file's own and are not changed by it."
            ),
        },
        "wikidata_base": "http://www.wikidata.org/entity/",
        "schemas": schemas.name,
        "qudt_units": qudt_units_path.name,
        "built_at": datetime.now(UTC).date().isoformat(),
        "examples_in": examples_in,
        "resolved": len(kept),
        "excluded": dict(excluded.most_common()),
        "per_kind": dict(per_kind.most_common()),
        "cap": cap,
        "units": {
            "attempted": attempted,
            "resolved": resolved_folded,
            "resolved_case_folded_only": resolved_plain,
            "rescued_by_the_fold": resolved_folded - resolved_plain,
        },
        "examples": selected,
    }


def dump(payload: dict[str, Any]) -> str:
    """The corpus as one line per example, under an indented header.

    Written this way for two readers. A person reads a diff, and one example
    per line makes a rebuild show which examples moved instead of reflowing
    the file. And the repository is not a data store: an indented example is
    around 560 bytes against 290 on one line, which is the difference between
    a file the large-file hook refuses and one it does not.
    """
    placeholder = "<<examples>>"
    header = json.dumps({**payload, "examples": placeholder}, indent=1, ensure_ascii=False)
    body = ",\n  ".join(json.dumps(e, ensure_ascii=False, separators=(",", ":")) for e in payload["examples"])
    return header.replace(f'"{placeholder}"', "[\n  " + body + "\n ]") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Wiki-Measurements.zip as downloaded")
    parser.add_argument("--schemas", type=Path, required=True, help="the canonical quantity schemas")
    parser.add_argument("--qudt-units", type=Path, required=True, help="qudt_units.json from the schema generator")
    parser.add_argument("--units-cache", type=Path, default=None, help="where the Wikidata unit table is kept")
    parser.add_argument("--retrieved-at", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--cap", type=int, default=400, help="examples kept per kind, 0 for all")
    parser.add_argument("--refresh", action="store_true", help="requery Wikidata instead of reading the cache")
    parser.add_argument("--out", type=Path, default=Path("src/oold_llm_bench/data/wiki_measurements.json"))
    args = parser.parse_args()

    cache = args.units_cache or args.archive.parent / "wikidata_units.json"
    payload = build(
        args.archive,
        args.schemas,
        args.qudt_units,
        cache,
        args.retrieved_at,
        None if args.cap == 0 else args.cap,
        args.refresh,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(dump(payload))

    units = payload["units"]
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")
    print(f"  examples in                : {payload['examples_in']}")
    for reason, count in payload["excluded"].items():
        print(f"  excluded {count:6}          : {reason}")
    print(f"  resolved                   : {payload['resolved']}")
    print(f"  written out                : {len(payload['examples'])} (cap {payload['cap']})")
    print(f"  written units attempted    : {units['attempted']}")
    print(f"  resolved with the fold     : {units['resolved']}")
    print(f"  resolved case folded only  : {units['resolved_case_folded_only']}")
    print(f"  rescued by the fold        : {units['rescued_by_the_fold']}")
    for kind, count in payload["per_kind"].items():
        print(f"  {kind:26} : {count}")


if __name__ == "__main__":
    main()
