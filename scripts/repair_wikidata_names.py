"""Give `name` back to the entities whose document names them by title.

A lead names its subject under the title of the article it is the lead of,
and Wikidata's label is often longer. Where the two differ the build's
grounding check looked for the label, did not find it, and filed `name` as a
distractor: the document names the entity, the corpus says it does not, and a
step answering `name` is marked wrong for being right.

Measured before this ran: `name` was truth for 900 of 959 entities and a
distractor for 52.

The build takes the title as a form of the name now, so a rebuild needs none
of this. This repairs the corpus that is committed, whose harvest cache is
gone.

    uv run python scripts/repair_wikidata_names.py          # dry
    uv run python scripts/repair_wikidata_names.py --apply
"""

from __future__ import annotations

import json
import sys

from oold_llm_bench.corpus.wikidata_schemaorg import CORPUS_PATH, NAME, stated_in
from oold_llm_bench.experiments.corpora import documents_for


def main() -> int:
    apply = "--apply" in sys.argv
    documents = documents_for()
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    repaired, unnamed = 0, 0
    for record in payload["entities"]:
        if NAME in (record.get("facts") or {}):
            continue
        document = documents.get(f"wds-{record['qid']}")
        if document is None:
            continue
        title = record["title"]
        written = stated_in(document, tuple(dict.fromkeys([title, title.split(" (")[0].strip()])))
        if written is None:
            unnamed += 1
            continue
        record.setdefault("facts", {})[NAME] = [written]
        (record.get("distractors") or {}).pop(NAME, None)
        repaired += 1

    # The file publishes its own grounding arithmetic and a test recomputes
    # it from the entities. A repair that moves a value from distractor to
    # fact moves that arithmetic with it, or the file stops accounting for
    # itself.
    published = payload["grounding"]["published"]

    def counted(record: dict, key: str) -> int:
        return sum(len(values) for values in (record.get(key) or {}).values())

    published["stated"] = sum(counted(record, "facts") for record in payload["entities"])
    # A name that was neither fact nor distractor is a new candidate, not a
    # moved one: the label was never offered for that entity at all.
    published["candidates"] = sum(
        counted(record, "facts") + counted(record, "distractors") for record in payload["entities"]
    )
    print(f"{repaired} entities regain `name`, {unnamed} are named by neither label nor title")
    print(f"published: {published['stated']} stated of {published['candidates']} candidates")
    if not apply:
        print("\ndry run; pass --apply")
        return 0
    CORPUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written to {CORPUS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
