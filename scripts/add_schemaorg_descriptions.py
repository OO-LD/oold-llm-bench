"""Give the corpus the schema.org comment for each property it offers.

The property step is asked which slots a document fills and is shown their
names alone. `contentLocation` against `provider` is a vocabulary question,
and a bare name does not answer it: measured over six models at n=100,
precision sits between 0.52 and 0.58 while recall reaches 0.92, so the values
are being found and filed under the wrong slot.

schema.org publishes a comment per property in its own JSON-LD. This reads it
once and writes it into the corpus beside the kind and the grounding rate.

    uv run python scripts/add_schemaorg_descriptions.py          # dry
    uv run python scripts/add_schemaorg_descriptions.py --apply
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request

from oold_llm_bench.corpus.wikidata_schemaorg import CORPUS_PATH

VOCABULARY = "https://schema.org/version/latest/schemaorg-current-https.jsonld"
AGENT = "oold-llm-bench/0.1 (https://github.com/OO-LD/oold-llm-bench)"
_TAGS = re.compile(r"<[^>]+>")


def comments() -> dict[str, str]:
    """Every property's comment, keyed by its bare name."""
    request = urllib.request.Request(VOCABULARY, headers={"User-Agent": AGENT})  # noqa: S310 - a fixed https url
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - same
        graph = json.load(response)["@graph"]
    found: dict[str, str] = {}
    for node in graph:
        types = node.get("@type")
        types = types if isinstance(types, list) else [types]
        if "rdf:Property" not in types:
            continue
        comment = node.get("rdfs:comment")
        if isinstance(comment, dict):
            comment = comment.get("@value")
        if not isinstance(comment, str):
            continue
        name = str(node.get("@id", "")).split(":")[-1]
        # The comments carry markup and a trailing newline. What the step
        # needs is the sentence, so the rest is dropped rather than shown.
        found[name] = " ".join(_TAGS.sub("", comment).split())
    return found


def main() -> int:
    apply = "--apply" in sys.argv
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    found = comments()
    print(f"{len(found)} properties described by schema.org")

    described, missing = 0, []
    for name, body in payload["properties"].items():
        comment = found.get(name)
        if comment is None:
            missing.append(name)
            continue
        body["description"] = comment
        described += 1
    print(f"{described} of {len(payload['properties'])} corpus properties described")
    if missing:
        print(f"  no comment for: {', '.join(sorted(missing))}")
    if not apply:
        print("\ndry run; pass --apply")
        return 0
    CORPUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written to {CORPUS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
