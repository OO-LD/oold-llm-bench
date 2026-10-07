"""The Wikidata description of every property the corpus offers.

Ground truth here comes from Wikidata's statements, and the step is shown
schema.org's words for the slot. Where the two vocabularies disagree about
what a slot means, the step is being asked one question and graded on
another: schema.org calls `creator` "the creator/author of this CreativeWork",
Wikidata's mapped property may be narrower or wider.

This writes the Wikidata label and description for each mapped property, so
a condition can offer either vocabulary and the difference is measurable
rather than argued.

    uv run python scripts/wikidata_property_text.py          # dry
    uv run python scripts/wikidata_property_text.py --apply
"""

from __future__ import annotations

import json
import pathlib
import sys
import urllib.parse
import urllib.request

from oold_llm_bench.corpus.wikidata_schemaorg import read_grounded_corpus

MAPPINGS = pathlib.Path(".cache/wikidata_schemaorg/p1628.json")
OUT = pathlib.Path("src/oold_llm_bench/data/wikidata_property_text.json")
AGENT = "oold-llm-bench/0.1 (https://github.com/OO-LD/oold-llm-bench)"


def entities(ids: list[str]) -> dict:
    query = urllib.parse.urlencode({
        "action": "wbgetentities",
        "ids": "|".join(ids),
        "props": "labels|descriptions",
        "languages": "en",
        "format": "json",
        "formatversion": "2",
    })
    request = urllib.request.Request(f"https://www.wikidata.org/w/api.php?{query}", headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - same
        return json.load(response).get("entities") or {}


def main() -> int:
    apply = "--apply" in sys.argv
    offered = set(read_grounded_corpus().properties)
    rows = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    # One schema.org property may be the target of several Wikidata
    # properties. All of them are kept: which one a value came from is not
    # recorded, so narrowing to one would be choosing without knowing.
    by_target: dict[str, list[str]] = {}
    for row in rows:
        if row["target"] in offered:
            by_target.setdefault(row["target"], []).append(row["source"])

    wanted = sorted({pid for pids in by_target.values() for pid in pids})
    print(f"{len(by_target)} of {len(offered)} offered properties are mapped from {len(wanted)} Wikidata properties")
    described: dict[str, dict] = {}
    for start in range(0, len(wanted), 50):
        for pid, body in entities(wanted[start : start + 50]).items():
            described[pid] = {
                "label": ((body.get("labels") or {}).get("en") or {}).get("value", ""),
                "description": ((body.get("descriptions") or {}).get("en") or {}).get("value", ""),
            }

    out = {}
    for target, pids in sorted(by_target.items()):
        parts = [
            f"{described[pid]['label']}: {described[pid]['description']}"
            for pid in pids
            if described.get(pid, {}).get("description")
        ]
        if parts:
            out[target] = " | ".join(parts)
    print(f"{len(out)} described from Wikidata")
    for name in ("creator", "provider", "contentLocation", "award"):
        if name in out:
            print(f"   {name:16s} {out[name][:86]}")
    if not apply:
        print("\ndry run; pass --apply")
        return 0
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
