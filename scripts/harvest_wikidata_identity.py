"""Fetch the three raw harvests the identity corpus is built from.

Separate from ``build_wikidata_identity.py`` because this is the part that
needs the network and takes hours, and the build has to be rerunnable against
what it produced without fetching anything again. Every output is resumable:
rerunning skips what is already on disk and retries only what failed.

``merges`` draws uniformly from the namespace-0 redirect table of a dated
dump, resolves each ``rd_from`` page id to a Q-id through the API, and
recovers both pre-merge states, plus the target's post-merge state. A
Wikidata merge leaves no log entry, so the merge is found by edit comment:
``wbmergeitems-to`` on the source, or ``wbcreateredirect`` where the editor
redirected without merging. The source's pre-merge state is the revision
before that edit; the target's pre-merge state is its last revision before
the merge timestamp, which needs ``rvstart`` and ``rvdir=older`` rather than a
scan, because an active target may have had hundreds of edits since. The
target's post-merge state is the same call with ``rvdir=newer``, anchored on
the merge timestamp instead of just before it: the expected patch for a
sequence corpus built from the two pre-merge states, not re-derivable from
them, since what a merge actually kept, added or overwrote is a fact about
the edit and not about either side alone.

``statements`` draws from a harvest of one property's truthy statements,
deduplicated to undirected pairs. Both ``P1889`` and ``P460`` are reciprocated
on most pairs, so the directed count is nearly twice the undirected one and
the deduplication is not optional. ``P31`` is read through the query service
before any state is fetched: four fifths of the ``P460`` draw is given names
and disambiguation pages, and excluding those from a type costs nothing while
excluding them from a full state fetch costs an hour.

``pairs`` pages one property's statements out of the query service, which is
what produces the input ``statements`` reads.

The rate limit is one pause between calls in one process. Several processes
over disjoint index ranges go faster than one, up to a point: the cost here is
dominated by revisions that are not in Wikimedia's cache, where a single read
can take a minute, and those do not parallelise away.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from oold_llm_bench.corpus.wikidata_identity import NOT_AN_ENTITY

API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = "oold-llm-bench/1.0 (+https://github.com/OO-LD/oold-llm-bench)"

MERGE_MARKERS = (
    "wbmergeitems-to",
    "wbcreateredirect",
    "wbeditentity-override",
    "Clearing item",
    "wbsetsitelink-remove",
)
"""Edit comments that belong to the merge rather than to the item's own life.

The state wanted is the last one a person wrote for its own sake, so the walk
back through the history skips every edit the merge itself made. ``Clearing
item`` is the gadget emptying the source, and ``wbsetsitelink-remove`` is the
editor taking the sitelink off by hand before merging.
"""

REDIRECT_ROW = re.compile(rb"\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)',")
QID = re.compile(rb"Q\d+")
ITEM = re.compile(r"Q\d+")
"""What counts as an endpoint.

A statement whose object is ``somevalue`` comes back from the query service as
a blank node with a hash for a name, and two of those are not a pair. Rare
enough that neither draw here contains one, and excluded at the harvest rather
than left to fail as an unreadable item later.
"""
TARGET_IN_COMMENT = (
    re.compile(r"wbcreateredirect:0\|\|Q\d+\|(Q\d+)"),
    re.compile(r"wbmergeitems-to:0\|\|(Q\d+)"),
)

PAIRS_QUERY = "SELECT ?a ?b WHERE { ?a wdt:%s ?b } LIMIT %d OFFSET %d"
TYPES_QUERY = "SELECT ?item ?type WHERE { VALUES ?item { %s } OPTIONAL { ?item wdt:P31 ?type } }"

CLAIM_VALUES = 12
"""Values kept per property here, before the build trims further.

Generous on purpose: this file is the harvest and the build is what decides
what a record carries, so a change to the record shape must not mean fetching
everything again.
"""

_last = [0.0]


def get(params: dict[str, Any], pause: float = 0.2, timeout: int = 120) -> dict[str, Any]:
    """One API read, rate limited, retried, and gzipped.

    ``Accept-Encoding: gzip`` is not a micro-optimisation here: a large entity
    revision comes back in under a second with it and in sixteen without.
    """
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    for attempt in range(6):
        gap = time.time() - _last[0]
        if gap < pause:
            time.sleep(pause - gap)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                _last[0] = time.time()
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except urllib.error.HTTPError as error:
            _last[0] = time.time()
            if error.code not in (429, 503):
                raise
            time.sleep(5 * (attempt + 1))
        except (TimeoutError, OSError):
            _last[0] = time.time()
    raise RuntimeError(f"giving up on {url}")


def sparql(query: str, timeout: int = 180) -> list[dict[str, Any]]:
    request = urllib.request.Request(  # noqa: S310
        SPARQL,
        data=urllib.parse.urlencode({"query": query}).encode(),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.load(response)["results"]["bindings"]


def snak_value(snak: dict[str, Any]) -> Any:
    """One claim's main value, flattened to something comparable."""
    if snak.get("snaktype") != "value":
        return snak.get("snaktype")
    data = snak.get("datavalue") or {}
    value, kind = data.get("value"), data.get("type")
    if not isinstance(value, dict):
        return value
    if kind == "wikibase-entityid":
        return value.get("id")
    if kind == "time":
        return value.get("time", "")[:11].lstrip("+")
    if kind == "quantity":
        return value.get("amount")
    if kind == "globecoordinate":
        return "{:.4f},{:.4f}".format(value.get("latitude", 0), value.get("longitude", 0))
    if kind == "monolingualtext":
        return "{}@{}".format(value.get("text"), value.get("language"))
    return value


def trim(body: dict[str, Any]) -> dict[str, Any]:
    """One entity revision, with everything a judgement could turn on."""
    claims = body.get("claims") or {}
    return {
        "labels": {k: v["value"] for k, v in (body.get("labels") or {}).items()},
        "aliases": {k: [a["value"] for a in v] for k, v in (body.get("aliases") or {}).items()},
        "descriptions": {k: v["value"] for k, v in (body.get("descriptions") or {}).items()},
        "sitelinks": {k: v["title"] for k, v in (body.get("sitelinks") or {}).items()},
        "claims": {p: [snak_value(c.get("mainsnak", {})) for c in cs[:CLAIM_VALUES]] for p, cs in claims.items()},
        "nstmts": sum(len(v) for v in claims.values()),
    }


def history(title: str, limit: int = 40) -> list[dict[str, Any]]:
    payload = get({
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvlimit": limit,
        "rvprop": "ids|timestamp|comment|size|user",
    })
    return payload["query"]["pages"][0].get("revisions", [])


def content_at(title: str, **where: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    payload = get({
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvlimit": 1,
        "rvprop": "ids|timestamp|content|user",
        "rvslots": "main",
        **where,
    })
    revisions = payload["query"]["pages"][0].get("revisions", [])
    if not revisions:
        return None, None
    try:
        return revisions[0], json.loads(revisions[0]["slots"]["main"]["content"])
    except (KeyError, ValueError):
        return revisions[0], None


def probe(source: str, target_in_dump: str) -> dict[str, Any]:
    """Both states as they stood before one merge, or why not."""
    revisions = history(source)
    if not revisions:
        return {"src": source, "tgt": target_in_dump, "fail": "the source has no revisions"}
    at = next((i for i, r in enumerate(revisions) if "wbmergeitems-to" in (r.get("comment") or "")), None)
    if at is None:
        at = next((i for i, r in enumerate(revisions) if "wbcreateredirect" in (r.get("comment") or "")), None)
    if at is None:
        return {"src": source, "tgt": target_in_dump, "fail": "no merge marker in the last 40 revisions"}
    merge = revisions[at]

    target = target_in_dump
    for revision in revisions[: at + 2]:
        found = next(
            (
                p.search(revision.get("comment") or "")
                for p in TARGET_IN_COMMENT
                if p.search(revision.get("comment") or "")
            ),
            None,
        )
        if found:
            target = found.group(1)
            break

    before = next(
        (r for r in revisions[at + 1 :] if not any(m in (r.get("comment") or "") for m in MERGE_MARKERS)),
        None,
    )
    if before is None:
        return {"src": source, "tgt": target, "merge_ts": merge["timestamp"], "fail": "no pre-merge revision"}
    _, source_body = content_at(source, rvstartid=before["revid"])
    if not source_body:
        return {"src": source, "tgt": target, "fail": "the source's pre-merge content does not parse"}

    just_before = dt.datetime.strptime(merge["timestamp"], "%Y-%m-%dT%H:%M:%SZ") - dt.timedelta(seconds=2)
    record = {
        "src": source,
        "tgt": target,
        "tgt_dump": target_in_dump,
        "merge_ts": merge["timestamp"],
        "merge_user": merge.get("user"),
        "merge_comment": merge.get("comment"),
        "src_pre_revid": before["revid"],
        "src_pre_ts": before["timestamp"],
        "src_pre": trim(source_body),
        "tgt_pre": None,
    }
    revision, body = content_at(target, rvstart=just_before.strftime("%Y-%m-%dT%H:%M:%SZ"), rvdir="older")
    if revision and body:
        record["tgt_pre"] = trim(body)
        record["tgt_pre_revid"] = revision["revid"]
        record["tgt_pre_ts"] = revision["timestamp"]
    else:
        record["fail_tgt"] = "the target's pre-merge state is unrecoverable"

    # The expected patch: the target as the merge left it, not as it stands
    # now. `rvdir=newer` from the merge timestamp lands on the merge's own
    # revision or the one right after it, the same way `tgt_pre` is anchored
    # on the revision right before. Without this a sequence corpus built from
    # `tgt_pre`/`src_pre` would have two starting states and nothing to merge
    # them into.
    post_revision, post_body = content_at(target, rvstart=merge["timestamp"], rvdir="newer")
    if post_revision and post_body:
        record["tgt_post"] = trim(post_body)
        record["tgt_post_revid"] = post_revision["revid"]
        record["tgt_post_ts"] = post_revision["timestamp"]
    else:
        record["fail_tgt_post"] = "the target's post-merge state is unrecoverable"
    return record


def redirect_rows(dump: Path) -> list[tuple[int, str]]:
    """Every namespace-0 redirect with a Q-shaped target, from the dump."""
    rows: list[tuple[int, str]] = []
    with gzip.open(dump, "rb") as handle:
        for line in handle:
            if not (line.startswith(b"(") or line.startswith(b"INSERT")):
                continue
            for row in REDIRECT_ROW.finditer(line):
                if int(row.group(2)) == 0 and QID.fullmatch(row.group(3)):
                    rows.append((int(row.group(1)), row.group(3).decode()))
    return rows


def resolve(page_ids: list[int]) -> dict[int, str]:
    """Page ids to titles. Cheaper than the 3.6 GB page table by a lot."""
    found: dict[int, str] = {}
    for start in range(0, len(page_ids), 50):
        chunk = page_ids[start : start + 50]
        payload = get({"action": "query", "pageids": "|".join(str(p) for p in chunk), "prop": ""})
        for page in payload["query"]["pages"]:
            if "pageid" in page and re.fullmatch(r"Q\d+", page.get("title") or ""):
                found[page["pageid"]] = page["title"]
    return found


def harvest_merges(args: argparse.Namespace) -> None:
    rows = redirect_rows(args.dump)
    print(f"namespace-0 Q-target redirect rows: {len(rows)}")
    random.seed(args.seed)
    drawn = random.sample(rows, min(args.draw, len(rows)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs_path = args.out.parent / "merge_draw.json"
    pairs: list[dict[str, Any]]
    if pairs_path.exists():
        pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    else:
        titles = resolve([page_id for page_id, _ in drawn])
        pairs = [
            {"src": titles[page_id], "tgt": target, "pageid": page_id} for page_id, target in drawn if page_id in titles
        ]
        pairs_path.write_text(json.dumps(pairs), encoding="utf-8")
    print(f"drawn {len(drawn)} under seed {args.seed}, {len(pairs)} resolved to a Q-id")

    done = {
        record["src"]
        for path in [*args.out.parent.glob("*.jsonl")]
        for record in read_jsonl(path)
        if record.get("tgt_pre")
    }
    with args.out.open("a", encoding="utf-8", newline="\n") as handle:
        for index, pair in enumerate(pairs[args.start : args.stop], args.start):
            if pair["src"] in done:
                continue
            try:
                record = probe(pair["src"], pair["tgt"])
            except RuntimeError as error:
                record = {"src": pair["src"], "tgt": pair["tgt"], "fail": str(error)[:120]}
            record["draw_index"] = index
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def undirected(path: Path) -> list[tuple[str, str]]:
    """The statement harvest as distinct unordered pairs.

    Both properties are reciprocated on 88 to 95 percent of pairs, so this is
    where 1,256,324 ``P1889`` statements become 699,791 pairs.
    """
    seen: set[tuple[str, str]] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            left, _, right = line.rstrip("\n").partition("\t")
            if left == right or not (ITEM.fullmatch(left) and ITEM.fullmatch(right)):
                continue
            seen.add((left, right) if left < right else (right, left))
    return sorted(seen)


def types_of(entities: list[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {entity: [] for entity in entities}
    for start in range(0, len(entities), 400):
        chunk = entities[start : start + 400]
        for row in sparql(TYPES_QUERY % " ".join("wd:" + entity for entity in chunk)):
            if "type" in row:
                found[row["item"]["value"].rsplit("/", 1)[-1]].append(row["type"]["value"].rsplit("/", 1)[-1])
        print(f"  types {start + len(chunk)}/{len(entities)}", flush=True)
    return found


def harvest_statements(args: argparse.Namespace) -> None:
    pairs = undirected(args.pairs)
    print(f"{args.property} distinct undirected pairs: {len(pairs)}")
    random.seed(args.seed)
    drawn = random.sample(pairs, min(args.draw, len(pairs)))
    entities = sorted({entity for pair in drawn for entity in pair})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    types_path = args.out.parent / f"{args.property.lower()}_types.json"
    if types_path.exists():
        types = json.loads(types_path.read_text(encoding="utf-8"))
    else:
        types = types_of(entities)
        types_path.write_text(json.dumps(types), encoding="utf-8")

    def a_thing(entity: str) -> bool:
        return not set(types.get(entity) or ()) & set(NOT_AN_ENTITY)

    wanted = sorted({e for a, b in drawn if a_thing(a) and a_thing(b) for e in (a, b)})
    print(f"drawn {len(drawn)} pairs under seed {args.seed}, {len(wanted)} of {len(entities)} entities are things")

    states = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    todo = [entity for entity in wanted if entity not in states]
    for start in range(0, len(todo), 20):
        chunk = todo[start : start + 20]
        try:
            payload = get({
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "labels|aliases|descriptions|sitelinks|claims",
            })
        except RuntimeError as error:
            print(f"  {error}", flush=True)
            continue
        for qid, body in (payload.get("entities") or {}).items():
            if "claims" in body:
                states[qid] = trim(body)
        args.out.write_text(json.dumps(states, ensure_ascii=False), encoding="utf-8")
        print(f"  states {start + len(chunk)}/{len(todo)}", flush=True)

    draw_path = args.out.parent / f"{args.property.lower()}_draw.json"
    draw_path.write_text(
        json.dumps({
            "property": args.property,
            "seed": args.seed,
            "universe": len(pairs),
            "drawn": [list(pair) for pair in drawn],
            "types": types,
        }),
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({len(states)} states) and {draw_path}")


def harvest_pairs(args: argparse.Namespace) -> None:
    """One property's truthy statements, paged out of the query service."""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8", newline="\n") as handle:
        offset = 0
        while True:
            rows = sparql(PAIRS_QUERY % (args.property, args.page, offset))
            for row in rows:
                left = row["a"]["value"].rsplit("/", 1)[-1]
                right = row["b"]["value"].rsplit("/", 1)[-1]
                handle.write(f"{left}\t{right}\n")
            print(f"  {offset + len(rows)} statements", flush=True)
            if len(rows) < args.page:
                break
            offset += args.page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="what", required=True)

    merges = sub.add_parser("merges", help="draw from the redirect dump and recover both pre-merge states")
    merges.add_argument("--dump", type=Path, required=True, help="wikidatawiki-latest-redirect.sql.gz")
    merges.add_argument("--draw", type=int, default=1200)
    merges.add_argument("--seed", type=int, default=20260929)
    merges.add_argument("--start", type=int, default=0, help="first index of the draw this process takes")
    merges.add_argument("--stop", type=int, default=1 << 30)
    merges.add_argument("--out", type=Path, required=True, help="jsonl, appended to and resumable")
    merges.set_defaults(run=harvest_merges)

    statements = sub.add_parser("statements", help="draw from a statement harvest and fetch both states")
    statements.add_argument("--property", required=True, choices=("P1889", "P460"))
    statements.add_argument("--pairs", type=Path, required=True, help="the tsv.gz written by the pairs subcommand")
    statements.add_argument("--draw", type=int, required=True)
    statements.add_argument("--seed", type=int, default=20260930)
    statements.add_argument("--out", type=Path, required=True, help="json, one trimmed state per entity")
    statements.set_defaults(run=harvest_statements)

    pairs = sub.add_parser("pairs", help="page one property's statements out of the query service")
    pairs.add_argument("--property", required=True)
    pairs.add_argument("--page", type=int, default=50000)
    pairs.add_argument("--out", type=Path, required=True, help="tsv.gz, two Q-ids a line")
    pairs.set_defaults(run=harvest_pairs)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
