"""Fetch the five raw harvests the Wikidata-schema.org corpus is built from.

Separate from ``build_wikidata_schemaorg.py`` because this is the part that
needs the network and takes an hour, and the build has to be rerunnable
against what it produced without fetching anything again. Every output is
resumable: rerunning skips what is already on disk and retries only what is
missing.

``vocabulary`` takes schema.org's own JSON-LD, which is what says whether a
``P1709`` target is a class at all. 33 of the 471 mapped targets are
enumeration members or datatypes, Monday and True and Paperback and
StudioAlbum among them, and they are dropped by asking the vocabulary rather
than by listing them here.

``mappings`` takes ``P1709`` and ``P1628`` out of the query service. Together
they are the whole of the correspondence between the two vocabularies, and
neither is authored by us.

``draw`` runs :data:`~oold_llm_bench.corpus.wikidata_schemaorg.CLASS_QUERY`
once per class, raising the sitelink floor until the service answers inside
its budget. Three roots need the ladder: ``Q215627`` reaches ``Q5`` through
176,085 subclasses, and a query the service has to enumerate before it can
order or count times out on every one of them.

``entities`` reads the claims, the label and the English sitelink of each
drawn item in batches of 50, and keeps only the properties the mapping
reaches. Rank and qualifiers are kept at this stage, because the rule that
drops a time-scoped statement belongs to the build and must not need a refetch
to change.

``lexicon`` fetches the English label, the aliases and the demonym of every
item any kept statement points at. The demonym is ``P1549`` and it is the
cheapest thing in this pipeline: it is what lets "British-Austrian drama film"
count as stating two countries.

``articles`` fetches the lead of each article and the revision id it came
from, 20 at a time. It runs last and only for the entities that still have
enough candidate statements to be worth a document, which is what keeps it
from being the long pole.

The rate limit is one pause between calls in one process. The query service
drops to roughly one request a minute under load, so the draw retries with a
growing backoff rather than giving up.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from oold_llm_bench.corpus.wikidata_schemaorg import (
    CLASS_QUERY,
    EQUIVALENT_CLASS,
    EQUIVALENT_PROPERTY,
    MIN_FACTS,
)

SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
VOCABULARY = "https://schema.org/version/latest/schemaorg-current-https.jsonld"
USER_AGENT = "oold-llm-bench/1.0 (+https://github.com/OO-LD/oold-llm-bench)"

CANDIDATES = (
    "Person",
    "Movie",
    "MusicAlbum",
    "Organization",
    "Book",
    "City",
    "Event",
    "TVSeries",
    "VideoGame",
    "Painting",
    "Museum",
    "CollegeOrUniversity",
    "Airport",
    "Mountain",
    "Newspaper",
    "Periodical",
    "SoftwareApplication",
    "Country",
)
"""Which mapped classes are drawn at all, and why these.

A draw costs a query the service may take a minute over, so 438 mapped classes
cannot each have one. These are the mapped classes whose Wikidata root has
enough well-linked instances to fill a cell: measured at a sitelink floor of
25, from City 21,784 and Movie 6,582 down to Book 80. Which of them survives
grounding is not decided here: the build reports the rate per class and a
class that does not reach the minimum is carried with its count, never
dropped silently.
"""

FLOORS = (25, 60, 150)
"""Sitelink floors tried in order, lowest first.

A higher floor is a smaller candidate set and a faster query, and it is also a
stronger bias towards subjects many languages cover. So the lowest floor that
answers is the one used, and which one that was is recorded per class.
"""

LEXICON_QUERY = """SELECT ?item ?label ?alias ?demonym WHERE {
  VALUES ?item { %s }
  OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label) = "en") }
  OPTIONAL { ?item skos:altLabel ?alias . FILTER(LANG(?alias) = "en") }
  OPTIONAL { ?item wdt:P1549 ?demonym . FILTER(LANG(?demonym) = "en") }
}"""

MAPPING_QUERY = (
    'SELECT ?source ?target WHERE { ?source wdt:%s ?target . FILTER(CONTAINS(STR(?target), "schema.org/")) }'
)

_last = [0.0]


def _pause(gap: float) -> None:
    waited = time.time() - _last[0]
    if waited < gap:
        time.sleep(gap - waited)
    _last[0] = time.time()


def sparql(query: str, *, attempts: int = 6, timeout: int = 300) -> list[dict[str, Any]]:
    """One query, retried with a growing backoff.

    The service answers a cheap query in a second and refuses everything for a
    minute once it decides a client is busy, so a fixed pause is either too
    slow or too fast. The backoff is what makes a long harvest finish.
    """
    data = urllib.parse.urlencode({"query": query}).encode()
    for attempt in range(attempts):
        _pause(2.0)
        request = urllib.request.Request(  # noqa: S310 - one constant endpoint
            SPARQL,
            data=data,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/sparql-results+json",
                "Accept-Encoding": "gzip",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - same constant
                raw = response.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            # A Wikipedia title may hold a raw control character, and the
            # service passes it through unescaped. Strict parsing refuses the
            # whole page of results over one title nothing here reads.
            return json.loads(raw.decode("utf-8"), strict=False)["results"]["bindings"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            if attempt == attempts - 1:
                raise
            print(f"    retry {attempt + 1}: {type(error).__name__} {str(error)[:60]}", flush=True)
            time.sleep(20 * (attempt + 1))
    return []


def api(endpoint: str, params: dict[str, Any], *, gap: float = 1.0, attempts: int = 5) -> dict[str, Any]:
    """One MediaWiki read, rate limited, retried and gzipped.

    ``Accept-Encoding: gzip`` is not a micro-optimisation: a batch of 50
    entities with their full claims is several megabytes uncompressed.
    """
    url = endpoint + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    for attempt in range(attempts):
        _pause(gap)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})  # noqa: S310 - two constant endpoints
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 - same two
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            if attempt == attempts - 1:
                raise
            print(f"    retry {attempt + 1}: {type(error).__name__} {str(error)[:60]}", flush=True)
            time.sleep(5 * (attempt + 1))
    return {}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")


def qid(binding: dict[str, Any]) -> str:
    return str(binding["value"]).rsplit("/", 1)[-1]


def harvest_vocabulary(args: argparse.Namespace) -> None:
    path = args.cache / "schemaorg.jsonld"
    if path.exists() and not args.refresh:
        print(f"{path} is already here ({path.stat().st_size // 1024} KiB)")
        return
    request = urllib.request.Request(VOCABULARY, headers={"User-Agent": USER_AGENT})  # noqa: S310 - one constant
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 - same constant
        body = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    print(f"wrote {path} ({len(body) // 1024} KiB)")


def harvest_mappings(args: argparse.Namespace) -> None:
    for name, prop in (("p1709", EQUIVALENT_CLASS), ("p1628", EQUIVALENT_PROPERTY)):
        path = args.cache / f"{name}.json"
        if path.exists() and not args.refresh:
            print(f"{path} is already here ({len(read_json(path))} rows)")
            continue
        rows = sparql(MAPPING_QUERY % prop)
        write_json(path, [{"source": qid(r["source"]), "target": qid(r["target"])} for r in rows])
        print(f"wrote {path} ({len(rows)} rows)")


def _roots(cache: Path) -> dict[str, list[str]]:
    """Which Wikidata items each schema.org class is declared equivalent to."""
    found: dict[str, list[str]] = {}
    for row in read_json(cache / "p1709.json") or []:
        found.setdefault(row["target"], []).append(row["source"])
    return found


def harvest_draw(args: argparse.Namespace) -> None:
    roots = _roots(args.cache)
    path = args.cache / "draw.json"
    drawn: dict[str, Any] = read_json(path) or {}
    for name in args.classes:
        if name in drawn and not args.refresh:
            print(f"{name:22} already drawn ({len(drawn[name]['items'])})")
            continue
        for root in roots.get(name, []):
            for floor in FLOORS:
                query = CLASS_QUERY % {"root": root, "floor": floor, "limit": args.limit}
                started = time.time()
                try:
                    rows = sparql(query, attempts=2)
                except Exception as error:
                    print(f"{name:22} root {root} floor {floor}: {type(error).__name__}", flush=True)
                    continue
                drawn[name] = {
                    "root": root,
                    "floor": floor,
                    "limit": args.limit,
                    "seconds": round(time.time() - started, 1),
                    "items": [{"qid": qid(r["item"]), "article": r["article"]["value"]} for r in rows],
                }
                write_json(path, drawn)
                print(
                    f"{name:22} root {root} floor {floor}: {len(rows)} items in {drawn[name]['seconds']}s", flush=True
                )
                break
            if name in drawn:
                break
        if name not in drawn:
            print(f"{name:22} no floor answered, so the class is not drawn", flush=True)
    write_json(path, drawn)


def _mapped_properties(cache: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for row in read_json(cache / "p1628.json") or []:
        if row["source"].startswith("P"):
            found.setdefault(row["source"], []).append(row["target"])
    return found


def _snak(snak: dict[str, Any]) -> dict[str, Any] | None:
    """One claim's main value, flattened to what the build can read.

    An unknown or a missing value is dropped here rather than carried as a
    marker: schema.org has no way to write "some value", so there is nothing a
    document could state and nothing a model could answer.
    """
    if snak.get("snaktype") != "value":
        return None
    data = snak.get("datavalue") or {}
    value, kind = data.get("value"), data.get("type")
    if kind == "string":
        return {"kind": "string", "value": value}
    if kind == "monolingualtext" and isinstance(value, dict):
        return {"kind": "string", "value": value.get("text")} if value.get("language") == "en" else None
    if kind == "wikibase-entityid" and isinstance(value, dict):
        return {"kind": "item", "value": value.get("id")}
    if kind == "time" and isinstance(value, dict):
        return {"kind": "time", "value": value.get("time"), "precision": value.get("precision")}
    if kind == "quantity" and isinstance(value, dict):
        return {"kind": "quantity", "value": value.get("amount"), "unit": value.get("unit")}
    return None


def _claims(body: dict[str, Any], wanted: set[str]) -> dict[str, list[dict[str, Any]]]:
    """The mapped statements of one item, with rank and time qualifiers kept."""
    kept: dict[str, list[dict[str, Any]]] = {}
    for prop, claims in (body.get("claims") or {}).items():
        if prop not in wanted:
            continue
        for claim in claims:
            value = _snak(claim.get("mainsnak") or {})
            if value is None:
                continue
            qualifiers = sorted(claim.get("qualifiers") or {})
            kept.setdefault(prop, []).append({**value, "rank": claim.get("rank"), "qualifiers": qualifiers})
    return kept


def harvest_entities(args: argparse.Namespace) -> None:
    wanted = set(_mapped_properties(args.cache))
    drawn = read_json(args.cache / "draw.json") or {}
    path = args.cache / "entities.json"
    states: dict[str, Any] = read_json(path) or {}

    todo = [item["qid"] for entry in drawn.values() for item in entry["items"] if item["qid"] not in states]
    todo = list(dict.fromkeys(todo))
    print(f"{len(todo)} items to read, {len(states)} already here")
    for start in range(0, len(todo), 50):
        chunk = todo[start : start + 50]
        payload = api(
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "claims|labels|sitelinks",
                "languages": "en",
                "sitefilter": "enwiki",
            },
        )
        for item, body in (payload.get("entities") or {}).items():
            states[item] = {
                "label": ((body.get("labels") or {}).get("en") or {}).get("value"),
                "enwiki": ((body.get("sitelinks") or {}).get("enwiki") or {}).get("title"),
                "claims": _claims(body, wanted),
            }
        write_json(path, states)
        print(f"  entities {start + len(chunk)}/{len(todo)}", flush=True)


def _referenced(cache: Path) -> list[str]:
    """Every item any kept statement points at, which is the lexicon's scope."""
    states = read_json(cache / "entities.json") or {}
    found: set[str] = set()
    for body in states.values():
        for claims in (body.get("claims") or {}).values():
            for claim in claims:
                if claim.get("kind") == "item" and claim.get("value"):
                    found.add(claim["value"])
    return sorted(found)


def harvest_lexicon(args: argparse.Namespace) -> None:
    path = args.cache / "lexicon.json"
    lexicon: dict[str, Any] = read_json(path) or {}
    todo = [item for item in _referenced(args.cache) if item not in lexicon]
    print(f"{len(todo)} referenced items to name, {len(lexicon)} already here")
    for start in range(0, len(todo), 300):
        chunk = todo[start : start + 300]
        rows = sparql(LEXICON_QUERY % " ".join("wd:" + item for item in chunk))
        for item in chunk:
            lexicon.setdefault(item, {"label": None, "aliases": [], "demonyms": []})
        for row in rows:
            entry = lexicon[qid(row["item"])]
            if "label" in row:
                entry["label"] = row["label"]["value"]
            for field, key in (("alias", "aliases"), ("demonym", "demonyms")):
                if field in row and row[field]["value"] not in entry[key]:
                    entry[key].append(row[field]["value"])
        write_json(path, lexicon)
        print(f"  lexicon {start + len(chunk)}/{len(todo)}", flush=True)


def _article_titles(cache: Path) -> list[tuple[str, str]]:
    """The articles worth a fetch: an entity with too few statements to reach
    the minimum cannot reach it after grounding either, and a lead costs a
    call."""
    states = read_json(cache / "entities.json") or {}
    wanted: list[tuple[str, str]] = []
    for item, body in states.items():
        title = body.get("enwiki")
        if not title:
            continue
        # The label fills `name`, so one fewer statement is needed.
        statements = sum(len(v) for v in (body.get("claims") or {}).values())
        if statements + 1 >= MIN_FACTS:
            wanted.append((item, title))
    return sorted(wanted)


def harvest_articles(args: argparse.Namespace) -> None:
    path = args.cache / "articles.json"
    articles: dict[str, Any] = read_json(path) or {}
    todo = [(item, title) for item, title in _article_titles(args.cache) if item not in articles]
    print(f"{len(todo)} leads to read, {len(articles)} already here")
    by_title = {title: item for item, title in todo}
    titles = [title for _, title in todo]
    for start in range(0, len(titles), 20):
        chunk = titles[start : start + 20]
        payload = api(
            WIKIPEDIA_API,
            {
                "action": "query",
                "prop": "extracts|revisions",
                "exintro": 1,
                "explaintext": 1,
                "exlimit": "max",
                "rvprop": "ids",
                "redirects": 1,
                "titles": "|".join(chunk),
            },
        )
        query = payload.get("query") or {}
        # A redirect means the sitelink and the page title disagree, which is
        # ordinary; the normalised title is what the response is keyed on.
        aliases = {entry["from"]: entry["to"] for entry in (query.get("redirects") or [])}
        aliases |= {entry["from"]: entry["to"] for entry in (query.get("normalized") or [])}
        reverse = {aliases.get(title, title): title for title in chunk}
        for page in query.get("pages") or []:
            asked = reverse.get(page.get("title"))
            item = by_title.get(asked or "")
            if item is None or page.get("missing"):
                continue
            revisions = page.get("revisions") or [{}]
            articles[item] = {
                "title": page.get("title"),
                "revision": revisions[0].get("revid"),
                "extract": page.get("extract") or "",
            }
        write_json(path, articles)
        print(f"  leads {start + len(chunk)}/{len(titles)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/wikidata_schemaorg"))
    parser.add_argument("--refresh", action="store_true", help="refetch instead of reading what is on disk")
    sub = parser.add_subparsers(dest="what", required=True)

    sub.add_parser("vocabulary", help="schema.org's own JSON-LD").set_defaults(run=harvest_vocabulary)
    sub.add_parser("mappings", help="P1709 and P1628 out of the query service").set_defaults(run=harvest_mappings)

    draw = sub.add_parser("draw", help="one entity draw per class")
    draw.add_argument("--classes", nargs="*", default=list(CANDIDATES))
    draw.add_argument("--limit", type=int, default=1200)
    draw.set_defaults(run=harvest_draw)

    sub.add_parser("entities", help="claims, label and English sitelink of every drawn item").set_defaults(
        run=harvest_entities
    )
    sub.add_parser("lexicon", help="label, aliases and demonym of every referenced item").set_defaults(
        run=harvest_lexicon
    )
    sub.add_parser("articles", help="the lead and revision id of every article").set_defaults(run=harvest_articles)

    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.run(args)


if __name__ == "__main__":
    main()
