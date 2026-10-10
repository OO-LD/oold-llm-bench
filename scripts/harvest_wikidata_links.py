"""Fetch the wikilinks inside the leads the Wikidata-schema.org corpus already
cites, for the linked-articles sequence variant.

The existing corpus's plain-text leads come from `prop=extracts`, which is
not versioned: asked by revision it ignores the id and renders the current
text, which is why the harvest hashes what it got rather than asking for a
revision back. Wikitext does not have that problem. `action=parse&oldid=`
returns the exact revision asked for, so this fetches each entity's lead
section as wikitext at the exact revision the corpus already pinned
(`GroundedEntity.revision`), rather than at whatever the article says now.

A link is kept only when its target resolves to a QID the corpus already
knows, because the ground truth this harvest exists to produce is "the anchor
in A resolves to the entity B is about", and that question is only answerable
within a graph of entities this corpus can already describe.

    uv run python scripts/harvest_wikidata_links.py          # resumable

Writes `.cache/wikidata_links/wikitext.json` (qid -> lead wikitext, so a
rerun does not re-fetch what it already has) and `.cache/wikidata_links/
links.json` (the resolved result: every kept anchor, its source and its
target). Both are gitignored, the way every other harvest cache is.
"""

from __future__ import annotations

import json
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from oold_llm_bench.corpus.wikidata_schemaorg import read_grounded_corpus

CACHE = pathlib.Path(".cache/wikidata_links")
WIKITEXT = CACHE / "wikitext.json"
LINKS = CACHE / "links.json"
AGENT = "oold-llm-bench/0.1 (https://github.com/OO-LD/oold-llm-bench) reproducing a cited corpus"
PAUSE = 1.0
"""Seconds between calls, politeness over throughput, the same pace every
other harvest in this repository uses against the same API."""

# `[[Target]]` or `[[Target|anchor]]`. A pipe inside a template or a table
# cell can still land inside this, which is why a kept link is also required
# to resolve to a known QID: a malformed capture resolves to nothing, rather
# than to a wrong one.
_LINK = re.compile(r"\[\[([^\[\]|#]+)(?:\|([^\[\]]*))?\]\]")
_SKIP_NAMESPACES = ("File:", "Image:", "Category:", "Wikipedia:", "Template:", "Help:", "Portal:", "wikt:")


def fetch(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})  # noqa: S310 - https, fixed host
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - https, fixed host
        return json.load(response)


def lead_wikitext(title: str, revision: int) -> str | None:
    """The lead section's wikitext at the exact revision the corpus pinned."""
    query = urllib.parse.urlencode({
        "action": "parse",
        "oldid": revision,
        "prop": "wikitext",
        "section": "0",
        "format": "json",
        "formatversion": "2",
    })
    try:
        payload = fetch(f"https://en.wikipedia.org/w/api.php?{query}")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  {title}: {type(exc).__name__}, skipped", flush=True)
        return None
    parse = payload.get("parse")
    if not parse:
        print(f"  {title}: {payload.get('error', {}).get('info', 'no parse result')}", flush=True)
        return None
    return parse.get("wikitext")


def anchors_in(wikitext: str) -> list[tuple[str, str]]:
    """Every ``(target title, anchor text)`` pair a lead's wikitext links to."""
    out = []
    for match in _LINK.finditer(wikitext):
        target, anchor = match.group(1).strip(), match.group(2)
        if any(target.startswith(ns) for ns in _SKIP_NAMESPACES):
            continue
        out.append((target, (anchor or target).strip()))
    return out


def resolve_titles(titles: list[str]) -> dict[str, str]:
    """Title to QID, through ``pageprops``, batched and redirect-aware."""
    resolved: dict[str, str] = {}
    for start in range(0, len(titles), 50):
        batch = titles[start : start + 50]
        query = urllib.parse.urlencode({
            "action": "query",
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "redirects": "1",
            "titles": "|".join(batch),
            "format": "json",
            "formatversion": "2",
        })
        try:
            payload = fetch(f"https://en.wikipedia.org/w/api.php?{query}")
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  resolve batch {start}: {type(exc).__name__}, skipped", flush=True)
            continue
        query_result = payload.get("query") or {}
        moved = {entry["from"]: entry["to"] for entry in (query_result.get("redirects") or [])}
        for page in query_result.get("pages") or []:
            title = page.get("title")
            qid = (page.get("pageprops") or {}).get("wikibase_item")
            if title and qid:
                resolved[title] = qid
        for source, target in moved.items():
            if target in resolved and source not in resolved:
                resolved[source] = resolved[target]
    return resolved


def main() -> int:
    corpus = read_grounded_corpus()
    CACHE.mkdir(parents=True, exist_ok=True)
    known_qids = {entity.qid for entity in corpus.entities}

    wikitext: dict[str, str] = {}
    if WIKITEXT.is_file():
        wikitext = json.loads(WIKITEXT.read_text(encoding="utf-8"))
        print(f"{len(wikitext)} leads already on disk")

    missing = [entity for entity in corpus.entities if entity.qid not in wikitext]
    print(f"fetching {len(missing)} of {len(corpus.entities)} leads, {PAUSE}s apart")
    for index, entity in enumerate(missing, start=1):
        text = lead_wikitext(entity.title, entity.revision)
        if text is not None:
            wikitext[entity.qid] = text
        if index % 25 == 0:
            WIKITEXT.write_text(json.dumps(wikitext, ensure_ascii=False), encoding="utf-8")
            print(f"  {index}/{len(missing)}, saved", flush=True)
        time.sleep(PAUSE)
    WIKITEXT.write_text(json.dumps(wikitext, ensure_ascii=False), encoding="utf-8")

    print("extracting anchors")
    by_source: dict[str, list[tuple[str, str]]] = {qid: anchors_in(text) for qid, text in wikitext.items()}
    every_title = sorted({target for anchors in by_source.values() for target, _ in anchors})
    print(f"resolving {len(every_title)} distinct link targets")
    resolved = resolve_titles(every_title)

    kept = [
        {"source": qid, "target": resolved[target], "target_title": target, "anchor": anchor}
        for qid, anchors in by_source.items()
        for target, anchor in anchors
        if resolved.get(target) in known_qids
    ]
    LINKS.write_text(json.dumps(kept, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"{len(kept)} links kept, between entities this corpus already knows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
