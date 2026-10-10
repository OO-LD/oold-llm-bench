"""Which offered properties are broader or narrower than which others.

The corpus files a fact under one property and the step is asked to choose a
name from a catalogue that also offers its parent. Wikidata records `author`
(P50) as a subproperty of `creator` (P170), so a document saying "written by
Jane Doe" licenses both names, and a step answering the parent has read the
text correctly and chosen a less specific slot. Scored strictly that is a
false positive on the parent and a false negative on the child, which is two
errors for one defensible answer.

The relation is taken from the vocabularies themselves and not authored here:
Wikidata's P1647 over the mapped properties, and schema.org's own
`rdfs:subPropertyOf`. They overlap barely. Of 53 Wikidata edges that project
onto schema.org names, schema.org declares 4 the same way and 1 the opposite
way, and asserts the rest about properties this corpus does not offer. So the
output names the vocabulary behind every edge, and names the contested pair
separately: an edge one vocabulary asserts and the other denies cannot be
read as the same kind of evidence as one nobody disputes.

Closure runs over Wikidata property ids and not over schema.org names, so a
chain through a property the corpus does not offer still connects its ends.
`birthDate` reaches `startDate` through `inception`, which is offered under
neither name.

    uv run python scripts/build_property_hierarchy.py          # dry
    uv run python scripts/build_property_hierarchy.py --apply
"""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any

from oold_llm_bench.corpus.wikidata_schemaorg import read_grounded_corpus

CACHE = pathlib.Path(".cache/wikidata_schemaorg")
MAPPINGS = CACHE / "p1628.json"
CLAIMS = CACHE / "property_claims.json"
VOCABULARY = CACHE / "schemaorg.jsonld"
OUT = pathlib.Path("src/oold_llm_bench/data/property_hierarchy.json")

AGENT = "oold-llm-bench/0.1 (https://github.com/OO-LD/oold-llm-bench)"
SCHEMAORG = "https://schema.org/version/latest/schemaorg-current-https.jsonld"
SUBPROPERTY = "P1647"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})  # noqa: S310 - https, fixed hosts
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - https, fixed hosts
        return response.read()


def claims(ids: list[str]) -> dict[str, Any]:
    """Wikidata claims for a batch of properties, cached whole.

    Cached because the closure is recomputed far more often than the
    vocabulary changes, and a rebuild that re-queries 168 properties to
    produce the same file is a rebuild nobody runs.
    """
    if CLAIMS.is_file():
        return json.loads(CLAIMS.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for start in range(0, len(ids), 50):
        query = urllib.parse.urlencode({
            "action": "wbgetentities",
            "ids": "|".join(ids[start : start + 50]),
            "props": "claims|labels",
            "languages": "en",
            "format": "json",
            "formatversion": "2",
        })
        out.update(json.loads(fetch(f"https://www.wikidata.org/w/api.php?{query}")).get("entities") or {})
    CLAIMS.write_text(json.dumps(out, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return out


def vocabulary() -> list[dict[str, Any]]:
    if not VOCABULARY.is_file():
        VOCABULARY.write_bytes(fetch(SCHEMAORG))
    return json.loads(VOCABULARY.read_text(encoding="utf-8"))["@graph"]


def _targets(body: dict[str, Any], prop: str) -> list[str]:
    out = []
    for claim in (body.get("claims") or {}).get(prop, []):
        value = (claim.get("mainsnak") or {}).get("datavalue") or {}
        if value.get("type") == "wikibase-entityid":
            out.append(value["value"]["id"])
    return sorted(out)


def _reaches(parents: dict[str, list[str]], start: str) -> list[str]:
    """Every ancestor of one property, by breadth.

    Iterative and seen-guarded because Wikidata's graph is user-maintained
    and a cycle in it is a data error that must not become a hang here.
    """
    seen: set[str] = set()
    queue = list(parents.get(start, ()))
    while queue:
        node = queue.pop(0)
        if node in seen or node == start:
            continue
        seen.add(node)
        queue.extend(parents.get(node, ()))
    return sorted(seen)


def build() -> dict[str, Any]:
    offered = set(read_grounded_corpus().properties)
    rows = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    # A property id and not an item id. The mapping accepts anything carrying
    # an equivalent-property statement, and Wikidata has items that do.
    name_of: dict[str, str] = {}
    for row in rows:
        if row["source"].startswith("P") and row["target"] in offered:
            name_of[row["source"]] = row["target"]

    bodies = claims(sorted(name_of))
    labels = {p: ((b.get("labels") or {}).get("en") or {}).get("value", "") for p, b in bodies.items()}
    parents = {p: _targets(b, SUBPROPERTY) for p, b in bodies.items()}

    edges: dict[tuple[str, str], list[str]] = {}
    for pid, child in sorted(name_of.items()):
        for ancestor in _reaches(parents, pid):
            parent = name_of.get(ancestor)
            if parent and parent != child:
                edges.setdefault((child, parent), []).append(
                    f"wikidata:{pid} ({labels.get(pid, '')}) -> {ancestor} ({labels.get(ancestor, '')})"
                )

    for node in vocabulary():
        declared = node.get("rdfs:subPropertyOf")
        if not declared:
            continue
        child = node["@id"].split(":")[-1]
        for item in [declared] if isinstance(declared, dict) else declared:
            parent = item["@id"].split(":")[-1]
            if child in offered and parent in offered and child != parent:
                edges.setdefault((child, parent), []).append(f"schemaorg:{child} rdfs:subPropertyOf {parent}")

    return {
        "schema_version": "1",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": {
            "wikidata": f"P1647 over {len(name_of)} properties the corpus maps from",
            "schemaorg": SCHEMAORG,
        },
        "contested": [
            # Each vocabulary makes the other's parent its child. Neither is
            # wrong enough to drop, so both edges stay and a consumer closing
            # over them makes the pair mutual, which is what two vocabularies
            # calling these near-synonyms amounts to.
            sorted(pair)
            for pair in sorted({tuple(sorted((child, parent))) for child, parent in edges if (parent, child) in edges})
        ],
        "edges": [
            {
                "child": child,
                "parent": parent,
                # Both vocabularies asserting one edge is the strongest
                # evidence available for it, and a reader deciding whether to
                # trust a forgiveness needs to see which asserted it.
                "vocabularies": sorted({via.split(":")[0] for via in vias}),
                "via": sorted(vias),
            }
            for (child, parent), vias in sorted(edges.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the file instead of printing what it would hold")
    args = parser.parse_args()

    built = build()
    both = [e for e in built["edges"] if len(e["vocabularies"]) > 1]
    print(f"{len(built['edges'])} edges between offered properties, {len(both)} asserted by both vocabularies")
    for edge in built["edges"]:
        mark = "both" if len(edge["vocabularies"]) > 1 else edge["vocabularies"][0]
        print(f"  {edge['child']:20s} is narrower than {edge['parent']:20s} [{mark}]")

    if args.apply:
        OUT.write_text(json.dumps(built, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {OUT}")
    else:
        print("dry run, pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
