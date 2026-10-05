"""Build the Recipe and JobPosting corpus out of Common Crawl WARC records.

Three inputs, none of them ours and none of them redistributable. Web Data
Commons supplies the urls of pages that carried the markup in October 2024.
Common Crawl supplies the secondary index that turns a url into a WARC offset,
and the WARC record that holds the page as it was then. The page itself
supplies both halves of the ground truth, its markup and its text, out of one
snapshot.

What is committed is the third of that which may be: urls, WARC coordinates,
record digests, our own extraction, and the corpus statistics. The page text
is written to a local cache the repository ignores, and
:func:`~oold_llm_bench.corpus.wdc.load_tasks` reads it from there. Rerunning
this script with ``--pages-only`` rebuilds that cache from the committed
coordinates and nothing else, which is how a third party reproduces the corpus.

The network is used three ways and all three are cached on disk. The cluster
index is 121 MB and is fetched once per crawl. A subset part is 180 MB and only
its first few thousand hosts are wanted, so a prefix is fetched by ranged GET
and the truncated gzip stream is read as far as it goes. A page is one ranged
GET of one independently compressed WARC member, which is why this reaches
thousands of pages without downloading a terabyte.

Nothing is refetched live from the publisher. Measured over 115 urls from this
release, none of them still yields a usable entity: the domain is gone, the
mirror that answers disallows us, or the markup was removed. The archive is not
a convenience here, it is the only source that still has the pages.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oold_llm_bench.corpus.crawl import USER_AGENT
from oold_llm_bench.corpus.crawl import document_id as host_id
from oold_llm_bench.corpus.extract_markup import extract
from oold_llm_bench.corpus.wdc import (
    CC_DATA,
    CRAWL_ID,
    LANGUAGE,
    MAIN_CLASSES,
    MIN_GROUNDED_PROPERTIES,
    WDC_BASE,
    WDC_RELEASE,
    ClusterIndex,
    Corpus,
    PageDocument,
    WarcRef,
    cache_path,
    entities_of,
    measure_document,
    page_html,
    range_fetcher,
    read_corpus,
    resolve,
    shingle,
    subset_urls,
    usable_document,
    warc_member,
)

PARTS: dict[str, tuple[int, ...]] = {
    "Recipe": (0, 4, 8, 12, 16, 20),
    "JobPosting": (0, 3, 6, 9, 12),
}
"""Which part files of each subset a prefix is taken from.

The subsets are sorted by reversed pay-level domain, so one part covers a
contiguous slice of the alphabet and a prefix of one part covers a slice of
that slice. Spreading the prefixes across the parts is what keeps the corpus
from being every site whose name begins with the same two letters.
"""

PREFIX_BYTES = 25_000_000
"""How much of a part file is fetched.

Measured on the Recipe subset: 25 MB compressed decompresses to 1.5 million
quads across 420 hosts, which is more hosts than the per-domain cap will let
through. Fetching more would buy variety this corpus does not use.
"""

LICENCE = {
    "web_data_commons": {
        "statement": "We publish the corpora for research purposes only.",
        "url": "https://webdatacommons.org/structureddata/",
        "note": "No data licence is granted, so nothing from the subsets is redistributed here.",
    },
    "common_crawl": {
        "statement": "A limited, non-transferable licence to access and use the data.",
        "url": "https://commoncrawl.org/terms-of-use",
    },
    "page_text": {
        "statement": "Each page is the copyright of whoever published it.",
        "note": (
            "This file holds urls, WARC coordinates, record digests and our own extraction of the "
            "markup. It holds no page text. The documents are rebuilt locally by ranged GET against "
            "the coordinates, and the text hash says whether what came back is what was read."
        ),
    },
}


def _get(url: str, timeout: float = 300.0, byte_range: str | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if byte_range:
        headers["Range"] = byte_range
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - every caller passes an https literal
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - same
        return response.read()


def cluster_index(cache: Path, crawl: str) -> ClusterIndex:
    """The crawl's block boundaries, downloaded once and kept."""
    path = cache / f"cluster-{crawl}.idx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(f"{CC_DATA}/cc-index/collections/{crawl}/indexes/cluster.idx"))
    return ClusterIndex.read(path, crawl)


def subset_prefix(cache: Path, class_name: str, part: int) -> Path:
    """A prefix of one subset part, downloaded once and kept."""
    path = cache / f"{class_name}-part{part}.gz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(f"{WDC_BASE}/{class_name}/part_{part}.gz", byte_range=f"bytes=0-{PREFIX_BYTES}"))
    return path


def candidate_urls(cache: Path, class_name: str, per_host: int) -> list[tuple[str, str]]:
    """Urls for one class, at most ``per_host`` per host, parts interleaved.

    Interleaved rather than concatenated. Taking the parts in turn means a run
    that stops early still holds the spread across the alphabet that fetching
    six parts was for.
    """
    per_part = [list(subset_urls(subset_prefix(cache, class_name, part), per_host)) for part in PARTS[class_name]]
    seen: dict[str, int] = {}
    ordered: list[tuple[str, str]] = []
    for index in range(max((len(rows) for rows in per_part), default=0)):
        for rows in per_part:
            if index >= len(rows):
                continue
            url, host = rows[index]
            if seen.get(host, 0) >= per_host:
                continue
            seen[host] = seen.get(host, 0) + 1
            ordered.append((url, host))
    return ordered


def document_id(class_name: str, url: str) -> str:
    """A readable, stable name: the class, the host, and a digest of the url."""
    return f"wdc-{class_name.lower()}-{host_id(url)}"


def read_page(ref: WarcRef, fetch: Any) -> str:
    return page_html(warc_member(ref, fetch), ref.charset)


def build_one(
    url: str,
    host: str,
    class_name: str,
    index: ClusterIndex,
    fetch: Any,
) -> tuple[PageDocument | None, str, str, Counter]:
    """One url, or the stated reason it did not become a document.

    Returns the document, the reason it was refused, the text it was built
    from, and the grounding counts the page contributed. The counts are
    returned even for a refusal, because a page that failed the document filter
    still says something about how often markup matches text and leaving it out
    would report the rate over the survivors only.
    """
    counts: Counter = Counter()
    ref = resolve(url, index, fetch)
    if ref is None:
        return None, "the crawl index holds no successful capture of the url", "", counts
    if not ref.languages.startswith(LANGUAGE):
        return None, f"the index detects the page as something other than {LANGUAGE}", "", counts
    try:
        html = read_page(ref, fetch)
    except (urllib.error.URLError, gzip.BadGzipFile, OSError, ValueError) as error:
        return None, f"the WARC record could not be read ({type(error).__name__})", "", counts

    extraction = extract(html, base_url=url)
    if not extraction.typed:
        return None, "the archived page carries no typed markup", "", counts

    text = extraction.text
    # Only the class this subset was read for counts as the main entity. A
    # recipe page that also annotates a job would otherwise be recorded under
    # whichever of the two the page happened to put first.
    entities, grounding = entities_of(extraction, text, main_classes=(class_name,))
    counts["seen"] = grounding.seen
    counts["grounded"] = grounding.kept
    for prop, (seen, kept) in grounding.by_property.items():
        counts[f"prop:{prop}:seen"] = seen
        counts[f"prop:{prop}:grounded"] = kept

    if not entities:
        return None, "the page does not annotate exactly one main entity", text, counts
    if len(entities[0].fields) < MIN_GROUNDED_PROPERTIES:
        return (
            None,
            f"the page states fewer than {MIN_GROUNDED_PROPERTIES} of the properties it annotates",
            text,
            counts,
        )

    stats = measure_document(text)
    refusal = usable_document(stats)
    if refusal is not None:
        return None, refusal, text, counts

    document = PageDocument(
        id=document_id(class_name, url),
        url=url,
        host=host,
        main_class=class_name,
        warc=ref,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        entities=entities,
        stats=stats,
        properties_seen=grounding.seen,
        properties_grounded=grounding.kept,
    )
    return document, "", text, counts


def _describe(document: PageDocument) -> dict[str, Any]:
    """One document as the committed file holds it, which is without its text."""
    return {
        "id": document.id,
        "url": document.url,
        "host": document.host,
        "main_class": document.main_class,
        "warc_filename": document.warc.filename,
        "warc_offset": document.warc.offset,
        "warc_length": document.warc.length,
        "warc_digest": document.warc.digest,
        "warc_timestamp": document.warc.timestamp,
        "text_sha256": document.text_sha256,
        "properties_seen": document.properties_seen,
        "properties_grounded": document.properties_grounded,
        "stats": document.stats.describe(),
        "entities": [
            {
                "key": e.key,
                "class": e.class_name,
                "fields": e.fields,
                **({"links": {p: list(k) for p, k in e.links.items()}} if e.links else {}),
            }
            for e in document.entities
        ],
    }


def build(
    cache: Path,
    pages: Path,
    *,
    crawl: str,
    per_host: int,
    per_class: int,
    attempts: int,
    delay: float,
) -> dict[str, Any]:
    """Walk the candidate urls until each class has its quota, counting refusals."""
    index = cluster_index(cache, crawl)
    fetch = range_fetcher()
    pages.mkdir(parents=True, exist_ok=True)

    excluded: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    kept: list[PageDocument] = []
    urls_in = 0

    for class_name in MAIN_CLASSES:
        taken = 0
        tried = 0
        seen_copy: set[str] = set()
        for url, host in candidate_urls(cache, class_name, per_host):
            if taken >= per_class or tried >= attempts:
                break
            tried += 1
            urls_in += 1
            document, refusal, text, counts = build_one(url, host, class_name, index, fetch)
            totals.update(counts)
            time.sleep(delay)
            if document is None:
                excluded[refusal] += 1
                continue
            fingerprint = _copy_of(document)
            if fingerprint and fingerprint in seen_copy:
                excluded["the page repeats copy a document already in the corpus carries"] += 1
                continue
            seen_copy.add(fingerprint)
            cache_path(pages, document).write_text(text, encoding="utf-8", newline="\n")
            kept.append(document)
            taken += 1
            print(f"  {len(kept):4} {document.id} ({len(document.entities)} entities)", flush=True)

    kept.sort(key=lambda d: (d.main_class, d.id))
    counts_per_class = Counter(d.main_class for d in kept)
    return {
        "schema_version": "1",
        "name": "WDC schema.org via Common Crawl",
        "crawl": crawl,
        "release": WDC_RELEASE,
        "source": {
            "subsets": f"{WDC_BASE}/<class>/part_<n>.gz",
            "index": f"{CC_DATA}/cc-index/collections/{crawl}/indexes/cluster.idx",
            "warc": f"{CC_DATA}/<warc_filename>",
            "note": "Fetch bytes warc_offset..warc_offset+warc_length-1 and check warc_digest.",
        },
        "licence": LICENCE,
        "built_at": datetime.now(timezone.utc).date().isoformat(),
        "urls_in": urls_in,
        "resolved": len(kept),
        "excluded": dict(excluded.most_common()),
        "per_class": dict(counts_per_class.most_common()),
        "cap": per_class,
        "grounding": _grounding(totals, kept),
        "documents": [_describe(document) for document in kept],
    }


def _copy_of(document: PageDocument) -> str:
    """A fingerprint of the words a variant page would repeat verbatim.

    Name and description together. Either alone collides too readily: two
    unrelated job ads share a title, and a site that leaves the description
    empty would make every one of its pages the same document.
    """
    fields = document.entities[0].fields
    parts = [str(fields.get(name) or "") for name in ("name", "title", "description")]
    return shingle(" ".join(part for part in parts if part))


def _grounding(totals: Counter[str], kept: list[PageDocument]) -> dict[str, Any]:
    """The grounding rate as a corpus statistic, over every page that was read.

    Reported over everything attempted and not over the survivors. A rate
    measured after the document filter has run says how well the filter
    selects, which is a different and much less interesting number.
    """
    by_property: dict[str, dict[str, int]] = {}
    for key, value in totals.items():
        if not key.startswith("prop:"):
            continue
        _, prop, field_name = key.split(":", 2)
        by_property.setdefault(prop, {"seen": 0, "grounded": 0})[field_name] = value
    per_class = {
        name: {
            "seen": sum(d.properties_seen for d in kept if d.main_class == name),
            "grounded": sum(d.properties_grounded for d in kept if d.main_class == name),
        }
        for name in MAIN_CLASSES
    }
    return {
        "values_seen": totals.get("seen", 0),
        "values_grounded": totals.get("grounded", 0),
        "rate": round(totals.get("grounded", 0) / totals["seen"], 4) if totals.get("seen") else 0.0,
        "per_class": per_class,
        "per_property": dict(sorted(by_property.items())),
    }


def refill_pages(corpus: Corpus, pages: Path, delay: float) -> int:
    """Rebuild the local text cache from the committed coordinates alone.

    The reproduction path. Nothing here reads the subsets or the index: a
    WARC filename, an offset and a length are enough, and the text hash in the
    file says whether the page that came back is the page the corpus was built
    from.
    """
    fetch = range_fetcher()
    pages.mkdir(parents=True, exist_ok=True)
    rebuilt = 0
    for document in corpus.documents:
        target = cache_path(pages, document)
        if target.exists():
            continue
        html = read_page(document.warc, fetch)
        text = extract(html, base_url=document.url).text
        target.write_text(text, encoding="utf-8", newline="\n")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != document.text_sha256:
            print(f"  {document.id}: text hash differs, {digest[:12]} against {document.text_sha256[:12]}")
        rebuilt += 1
        time.sleep(delay)
    return rebuilt


def dump(payload: dict[str, Any]) -> str:
    """One document per line under an indented header.

    The layout :mod:`scripts.build_wiki_measurements` uses, for the same two
    readers: a diff shows which documents moved instead of reflowing the file,
    and the result stays under the large-file hook.
    """
    placeholder = "<<documents>>"
    header = json.dumps({**payload, "documents": placeholder}, indent=1, ensure_ascii=False)
    body = ",\n  ".join(json.dumps(d, ensure_ascii=False, separators=(",", ":")) for d in payload["documents"])
    return header.replace(f'"{placeholder}"', "[\n  " + body + "\n ]") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="where the index and the subset prefixes are kept")
    parser.add_argument("--pages", type=Path, required=True, help="where the page text is written, never committed")
    parser.add_argument("--crawl", default=CRAWL_ID)
    parser.add_argument("--per-host", type=int, default=1, help="urls tried per host, which caps one domain's share")
    parser.add_argument("--per-class", type=int, default=120, help="documents kept per main class")
    parser.add_argument("--attempts", type=int, default=2000, help="urls tried per class before giving up")
    parser.add_argument("--delay", type=float, default=0.2, help="seconds between two archive requests")
    parser.add_argument("--pages-only", action="store_true", help="rebuild the text cache from the committed file")
    parser.add_argument("--out", type=Path, default=Path("src/oold_llm_bench/data/wdc_schemaorg.json"))
    args = parser.parse_args()

    if args.pages_only:
        rebuilt = refill_pages(read_corpus(args.out), args.pages, args.delay)
        print(f"rebuilt {rebuilt} documents into {args.pages}")
        return

    payload = build(
        args.cache,
        args.pages,
        crawl=args.crawl,
        per_host=args.per_host,
        per_class=args.per_class,
        attempts=args.attempts,
        delay=args.delay,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(dump(payload))

    grounding = payload["grounding"]
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")
    print(f"  urls tried                 : {payload['urls_in']}")
    for reason, count in payload["excluded"].items():
        print(f"  excluded {count:6}          : {reason}")
    print(f"  kept                       : {payload['resolved']}")
    for name, count in payload["per_class"].items():
        print(f"  {name:26} : {count}")
    print(f"  values seen                : {grounding['values_seen']}")
    print(f"  grounded                   : {grounding['values_grounded']} ({grounding['rate']:.1%})")


if __name__ == "__main__":
    main()
