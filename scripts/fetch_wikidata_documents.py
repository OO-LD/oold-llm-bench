"""Fetch the article leads the Wikidata-schema.org corpus cites.

The leads are CC BY-SA 4.0 and this repository is Apache-2.0, so they are
never committed. The corpus commits the revision id and the sha256 of each
instead, which is what makes it reproducible without republishing anyone's
prose: this asks for those exact revisions and refuses a lead that does not
hash to what the corpus recorded.

    uv run python scripts/fetch_wikidata_documents.py

Writes `.cache/wikidata_schemaorg/documents.json`, which `.gitignore` already
excludes. Rerunning is cheap: what is on disk and still hashes correctly is
kept, so an interrupted fetch resumes.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from oold_llm_bench.corpus.wikidata_schemaorg import read_grounded_corpus

OUT = pathlib.Path(".cache/wikidata_schemaorg/documents.json")
AGENT = "oold-llm-bench/0.1 (https://github.com/OO-LD/oold-llm-bench) reproducing a cited corpus"
PAUSE = 1.0
"""Seconds between calls. The API asks for a serial client and an identifying
agent; this is slower than the limit rather than at it."""


def fetch(titles: list[str]) -> dict:
    """One call, by title and not by revision id.

    Extracts are not versioned: asking by ``revids`` returns the extract of
    the page as it stands now, and the page object carries no revision id
    unless the revisions property is asked for too. So this asks the way the
    harvest asked, and the sha256 decides whether the lead is still the one
    the corpus was measured against.
    """
    query = urllib.parse.urlencode({
        "action": "query",
        "prop": "extracts|revisions",
        "exintro": "1",
        "explaintext": "1",
        "exlimit": "max",
        "rvprop": "ids",
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
        "titles": "|".join(titles),
    })
    request = urllib.request.Request(f"https://en.wikipedia.org/w/api.php?{query}", headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - same
        return json.load(response)


def main() -> int:
    corpus = read_grounded_corpus()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    have: dict[str, str] = {}
    if OUT.is_file():
        have = {k: str(v) for k, v in json.loads(OUT.read_text(encoding="utf-8")).items() if v}
        print(f"{len(have)} already on disk")

    missing = [entity for entity in corpus.entities if entity.id not in have]
    if not missing:
        print("nothing to fetch")
        return 0
    batches = [missing[start : start + 20] for start in range(0, len(missing), 20)]
    print(f"fetching {len(missing)} leads in {len(batches)} calls, {PAUSE}s apart")

    drifted = 0
    for index, batch in enumerate(batches, start=1):
        by_title = {entity.title: entity for entity in batch}
        try:
            payload = fetch([entity.title for entity in batch])
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  call {index}: {type(exc).__name__}, skipped")
            continue
        query = payload.get("query") or {}
        # A redirect or a normalisation means the sitelink and the page title
        # disagree, which is ordinary. The response is keyed on the title the
        # API settled on, so the mapping has to be read back.
        moved = {entry["from"]: entry["to"] for entry in (query.get("redirects") or [])}
        moved |= {entry["from"]: entry["to"] for entry in (query.get("normalized") or [])}
        reverse = {moved.get(title, title): title for title in by_title}
        for page in query.get("pages") or []:
            entity = by_title.get(reverse.get(page.get("title"), ""))
            # Whitespace collapsed, because that is the form the build hashed
            # and therefore the form the ground truth was measured against.
            text = " ".join((page.get("extract") or "").split())
            if entity is None or not text:
                continue
            # The sha256 is what the ground truth was measured against. A lead
            # that has been edited since is a different document, and scoring
            # against it would score answers against prose nobody checked.
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != entity.sha256:
                drifted += 1
                continue
            have[entity.id] = text
        if index % 10 == 0:
            print(f"  {index}/{len(batches)} calls, {len(have)} kept, {drifted} drifted", flush=True)
        OUT.write_text(json.dumps(have, ensure_ascii=False), encoding="utf-8")
        time.sleep(PAUSE)

    print(f"{len(have)} of {len(corpus.entities)} leads kept, {drifted} edited since the corpus was built")
    print(f"written to {OUT}")
    return 0


def adopt_current(limit: int | None = None) -> int:
    """Re-pin the entities whose lead has been edited since the build.

    Extracts are not versioned: the API ignores ``revids`` and serves the
    current lead, so a drifted document cannot be recovered. The corpus can
    move to the new one instead, but only after its facts are checked against
    it. A fact is ground truth because the lead was found to state it, and a
    lead that no longer does makes the expected answer unanswerable.

    Each stored value is re-checked with the corpus's own
    :func:`~oold_llm_bench.corpus.wikidata_schemaorg.stated_in`, under the
    kind its slot declares. What survives has to clear the same floor the
    build applied, or the entity is left behind rather than published thin.
    """
    import json as _json

    from oold_llm_bench.corpus.schemaorg import Kind
    from oold_llm_bench.corpus.wikidata_schemaorg import (
        CORPUS_PATH,
        MIN_FACTS,
        MIN_SLOTS,
        read_documents,
        read_grounded_corpus,
        stated_in,
        written_forms,
    )

    corpus = read_grounded_corpus()
    documents = read_documents(OUT) if OUT.is_file() else {}
    drifted = [entity for entity in corpus.entities if entity.id not in documents]
    if limit:
        drifted = drifted[:limit]
    if not drifted:
        print("no drifted entity to re-pin")
        return 0
    print(f"re-pinning {len(drifted)} entities against their current lead")

    kinds = {cls: {slot[0]: slot[1] for slot in (body.get("slots") or ())} for cls, body in corpus.catalogue.items()}
    payload = _json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    records = {record["qid"]: record for record in payload["entities"]}

    adopted, dropped = 0, 0
    for batch in [drifted[start : start + 20] for start in range(0, len(drifted), 20)]:
        by_title = {entity.title: entity for entity in batch}
        payload_api = fetch([entity.title for entity in batch])
        query = payload_api.get("query") or {}
        moved = {entry["from"]: entry["to"] for entry in (query.get("redirects") or [])}
        moved |= {entry["from"]: entry["to"] for entry in (query.get("normalized") or [])}
        reverse = {moved.get(title, title): title for title in by_title}
        for page in query.get("pages") or []:
            entity = by_title.get(reverse.get(page.get("title"), ""))
            lead = " ".join((page.get("extract") or "").split())
            revision = (page.get("revisions") or [{}])[0].get("revid")
            if entity is None or not lead or revision is None:
                continue
            kept: dict[str, list] = {}
            for prop, values in entity.facts.items():
                kind = Kind(kinds.get(entity.cls, {}).get(prop, "text"))
                for value in values:
                    forms = written_forms(value, kind)
                    written = stated_in(lead, forms) if forms else None
                    if written is not None:
                        kept.setdefault(prop, []).append(written if kind is Kind.TEXT else value)
            total = sum(len(v) for v in kept.values())
            if len(kept) < MIN_SLOTS or total < MIN_FACTS:
                print(f"  {entity.id}: {total} values over {len(kept)} slots on the new lead, left behind")
                dropped += 1
                continue
            record = records[entity.qid]
            record["revision"] = int(revision)
            record["sha256"] = hashlib.sha256(lead.encode("utf-8")).hexdigest()
            record["facts"] = kept
            documents[entity.id] = lead
            adopted += 1
        time.sleep(PAUSE)

    payload["entities"] = list(records.values())
    CORPUS_PATH.write_text(_json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT.write_text(_json.dumps(documents, ensure_ascii=False), encoding="utf-8")
    print(f"{adopted} re-pinned to their current lead, {dropped} left behind")
    return 0


if __name__ == "__main__":
    sys.exit(adopt_current() if "--adopt" in sys.argv else main())
