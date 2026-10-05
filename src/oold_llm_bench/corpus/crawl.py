"""Fetching pages that carry their own ground truth, under robots.txt and a delay.

Every number the benchmark has produced so far was measured on prose the
benchmark wrote itself. Real pages are the external check: someone else chose
the words, and the markup on the page says what they meant. What is bought
with that is validity, and the cost is that the documents are not ours.

So the corpus is split by what may be published. Web Data Commons states no
data license and Common Crawl grants access only, neither of which makes the
page text ours to pass on, and the text stays the copyright of whoever wrote
it. We keep the page locally, and we publish the url, the hash of what we
fetched and the truth we extracted. A third party reproduces the corpus by
fetching the same urls and checking the hashes, which
:func:`manifest` exists to make possible.

The politeness has a practical reason too. A benchmark that gets its collector
blocked cannot be reproduced by anyone, so robots.txt decides before anything
is fetched, one host is asked at most once every few seconds, and the user
agent names us and gives a contact address.

The network call is a parameter. Everything here is then testable without it,
including the retry path, which is the part that would otherwise only be
exercised when a site returns errors.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from oold_llm_bench.corpus.extract_markup import declared_licence, extract, to_jsonld
from oold_llm_bench.tasks.models import Source

__all__ = [
    "DEFAULT_DELAY",
    "MANIFEST_VERSION",
    "META_FILE",
    "MINIMUM_DELAY",
    "PAGE_FILE",
    "TEXT_FILE",
    "TRUTH_FILE",
    "USER_AGENT",
    "Crawler",
    "Document",
    "DocumentMeta",
    "FetchResult",
    "Response",
    "build_document",
    "document_id",
    "dump_meta",
    "load_meta",
    "manifest",
    "urlopen_fetcher",
    "write_document",
]

USER_AGENT = "oold-llm-bench/1.0 (+https://github.com/OO-LD/oold-llm-bench)"
"""Who is asking, and where to tell us to stop.

A product token and a contact url, never a browser string. A collector that
claims to be Chrome cannot be blocked selectively, which leaves a site
operator with nothing to do about us except block everything.
"""

DEFAULT_DELAY = 5.0
"""Seconds between two requests to one host when robots.txt states nothing.

Chosen high rather than low. The corpus is a few thousand pages spread over
many hosts, so the collection is bounded by breadth and not by this number,
and there is nothing to gain from being the fastest crawler a small site sees.
"""

MINIMUM_DELAY = 1.0
"""The floor, whatever the configuration says."""

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
"""Statuses that mean "not now" rather than "no".

429 and 503 are the two a rate limiter answers with, and retrying anything
else would be retrying a refusal.
"""

PAGE_FILE = "page.html"
TRUTH_FILE = "truth.jsonld"
TEXT_FILE = "input.txt"
META_FILE = "meta.yaml"

LOCAL_ONLY = (PAGE_FILE, TEXT_FILE)
"""The two files that never leave this machine.

Both are the page's own words. Everything else in a document directory is
either a pointer to the page or our own reading of its markup.
"""

MANIFEST_VERSION = "1"

_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Response:
    """What a fetcher hands back, whether the request went well or badly.

    An HTTP error is a response and not an exception here. A 404 and a 503 are
    data about the url that the record has to carry, and raising on them would
    make the common case of a dead link an exceptional path.
    """

    status: int
    body: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    url: str = ""
    """The url the response came from, which differs from the requested one
    after a redirect. Kept because a redirect can cross into a host whose
    robots.txt was never consulted."""


Fetcher = Callable[[str, Mapping[str, str]], Response]


@dataclass(frozen=True)
class FetchResult:
    """One attempt at one url, including the attempts that went nowhere.

    A refusal is recorded as fully as a success. Which urls robots.txt put out
    of reach, and how many were gone by the time we asked, is the difference
    between a corpus someone else can reproduce and a list of the pages that
    happened to work.
    """

    url: str
    fetched_at: datetime
    robots_allowed: bool
    status: int | None = None
    content_hash: str = ""
    licence: str | None = None
    body: str | None = None
    final_url: str | None = None
    attempts: int = 0

    @property
    def usable(self) -> bool:
        return self.robots_allowed and self.status == 200 and bool(self.body)


@dataclass(frozen=True)
class DocumentMeta:
    """Everything about a document except the document.

    This is the part a third party gets. It has to be enough to fetch the page
    again and to tell whether what came back is what we read, which is why the
    hashes are here and not derivable from anything else in the set.
    """

    document_id: str
    url: str
    content_hash: str
    """sha256 of the page as stored, so a refetch can be compared to it."""
    text_hash: str
    """sha256 of the stripped text, so a third party can tell a changed page
    from a changed extractor."""
    fetched_at: datetime
    robots_allowed: bool
    status: int | None = None
    entity_count: int = 0
    source: Source = Source.CRAWL
    final_url: str | None = None
    licence: str | None = None


@dataclass(frozen=True)
class Document:
    """One crawled page in the layout it is written to disk in."""

    meta: DocumentMeta
    page: str
    text: str
    truth: dict[str, Any]


def document_id(url: str) -> str:
    """A directory name that is stable across runs and readable in a listing.

    The name is host plus digest, not a counter. A counter renumbers the whole corpus when
    one url is dropped, and a bare digest leaves a directory nobody can place
    without opening it.
    """
    host = (urlsplit(url).hostname or "unknown").lower()
    if host.startswith("www."):
        host = host[4:]
    slug = _SLUG.sub("-", host).strip("-") or "unknown"
    digest = hashlib.blake2s(url.encode("utf-8"), digest_size=5).hexdigest()
    return f"{slug}-{digest}"


def urlopen_fetcher(timeout: float = 20.0) -> Fetcher:
    """The real network call, kept behind the same interface the tests use.

    Only http and https are accepted. A url list is data from outside, and a
    fetcher that follows whatever scheme it is handed will read local files for
    anyone who can put a line in that list.
    """

    def fetch(url: str, headers: Mapping[str, str]) -> Response:
        if urlsplit(url).scheme not in ("http", "https"):
            raise ValueError(f"refusing a non-http url: {url}")
        request = urllib.request.Request(url, headers=dict(headers))  # noqa: S310 - the scheme is checked above
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - same check
                charset = response.headers.get_content_charset() or "utf-8"
                return Response(
                    status=response.status,
                    body=response.read().decode(charset, errors="replace"),
                    headers=dict(response.headers.items()),
                    url=response.url,
                )
        except urllib.error.HTTPError as error:
            charset = error.headers.get_content_charset() or "utf-8" if error.headers else "utf-8"
            return Response(
                status=error.code,
                body=error.read().decode(charset, errors="replace"),
                headers=dict(error.headers.items()) if error.headers else {},
                url=error.url or url,
            )

    return fetch


class Crawler:
    """A fetcher that asks robots.txt first and waits between requests.

    The delay is per host and counts from the last request to that host,
    robots.txt included. Counting from the start of the request instead would
    let a slow host be hit as fast as a fast one, which is backwards.
    """

    def __init__(
        self,
        fetch: Fetcher,
        *,
        user_agent: str = USER_AGENT,
        delay: float = DEFAULT_DELAY,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self.user_agent = user_agent
        self.delay = delay
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._clock = clock
        self._robots: dict[str, RobotFileParser] = {}
        self._ready: dict[str, float] = {}

    @property
    def headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"}

    def robots(self, url: str) -> RobotFileParser:
        """The rules for a host, fetched once and kept for the run.

        Refetching per url would multiply the requests to every host by two,
        which is the opposite of what a robots.txt is for.
        """
        host = _host(url)
        cached = self._robots.get(host)
        if cached is not None:
            return cached
        parser = RobotFileParser()
        parser.set_url(_robots_url(url))
        response = self._request(_robots_url(url), self.delay)
        _apply_robots(parser, response)
        self._robots[host] = parser
        return parser

    def allowed(self, url: str) -> bool:
        return self.robots(url).can_fetch(self.user_agent, url)

    def delay_for(self, url: str) -> float:
        """How long to wait before the next request to this host.

        A stated crawl-delay is honored when it is longer than ours and
        ignored when it is shorter. A site asking to be crawled faster than we
        intend to does not change our delay.
        """
        stated = self.robots(url).crawl_delay(self.user_agent)
        wanted = max(self.delay, float(stated) if stated is not None else 0.0)
        return max(wanted, MINIMUM_DELAY)

    def fetch(self, url: str) -> FetchResult:
        """One url, with robots checked, the host delay kept, and 429s waited out."""
        if not self.allowed(url):
            return FetchResult(url=url, fetched_at=_now(), robots_allowed=False)

        delay = self.delay_for(url)
        response = Response(status=0)
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            response = self._request(url, delay)
            if response.status not in RETRY_STATUSES:
                break
            if attempts < self.max_attempts:
                self._sleep(self._backoff(response, attempts, delay))

        final = response.url or url
        if _host(final) != _host(url) and not self.allowed(final):
            return FetchResult(
                url=url,
                fetched_at=_now(),
                robots_allowed=False,
                status=response.status,
                final_url=final,
                attempts=attempts,
            )

        body = response.body if response.status == 200 else None
        return FetchResult(
            url=url,
            fetched_at=_now(),
            robots_allowed=True,
            status=response.status,
            content_hash=_digest(body) if body else "",
            licence=declared_licence(body) if body else None,
            body=body,
            final_url=final if final != url else None,
            attempts=attempts,
        )

    def collect(self, urls: Iterable[str]) -> Iterator[Document]:
        """Fetch a list of urls and yield the pages that produced markup.

        A page without markup is dropped here rather than written and filtered
        later, because it is a page we would be storing someone else's text
        for with nothing to show for it.
        """
        for url in urls:
            result = self.fetch(url)
            document = build_document(result)
            if document is not None:
                yield document

    def _request(self, url: str, delay: float) -> Response:
        """One request, after the delay for that host has passed.

        A transport failure is a response with status 0. A collection over
        thousands of urls that dies on the first refused connection is a
        collection that never finishes.
        """
        self._wait(_host(url), delay)
        try:
            return self._fetch(url, self.headers)
        except OSError:
            return Response(status=0, url=url)

    def _wait(self, host: str, delay: float) -> None:
        now = self._clock()
        ready = self._ready.get(host, 0.0)
        if now < ready:
            self._sleep(ready - now)
            now = ready
        self._ready[host] = now + delay

    def _backoff(self, response: Response, attempt: int, delay: float) -> float:
        """How long to wait after being told to slow down.

        A stated Retry-After wins outright, including when it is longer than
        anything we would have chosen: the server states what it needs.
        Otherwise the wait doubles per attempt, never dropping below the
        politeness delay we were already keeping.
        """
        stated = _header(response.headers, "retry-after")
        if stated is not None:
            try:
                return max(float(stated.strip()), MINIMUM_DELAY)
            except ValueError:
                pass
        return delay * (2**attempt)


def build_document(result: FetchResult) -> Document | None:
    """Turn a fetch into the four files, or return nothing if it is not worth it.

    Nothing is worth storing without at least one typed entity. The whole point
    of a crawled document is that the page brought its own answer, and a page
    with no markup is one we would have to annotate by hand, which is the cost
    this corpus exists to avoid.
    """
    if not result.usable or result.body is None:
        return None
    extraction = extract(result.body, base_url=result.final_url or result.url)
    if not extraction.typed:
        return None
    return Document(
        meta=DocumentMeta(
            document_id=document_id(result.url),
            url=result.url,
            content_hash=result.content_hash or _digest(result.body),
            text_hash=_digest(extraction.text),
            fetched_at=result.fetched_at,
            robots_allowed=result.robots_allowed,
            status=result.status,
            entity_count=len(extraction.typed),
            final_url=result.final_url,
            licence=result.licence or extraction.licence,
        ),
        page=result.body,
        text=extraction.text,
        truth=to_jsonld(extraction),
    )


def write_document(root: Path, document: Document) -> Path:
    """Write one document directory under ``root``.

    Newlines are forced to ``\\n``. A hash of a file that Windows rewrote on
    the way out would not match the same file written on Linux, and the hash is
    the whole of what a third party has to check against.
    """
    directory = root / document.meta.document_id
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / PAGE_FILE, document.page)
    _write(directory / TEXT_FILE, document.text)
    _write(directory / TRUTH_FILE, json.dumps(document.truth, indent=2, ensure_ascii=False) + "\n")
    _write(directory / META_FILE, dump_meta(document.meta))
    return directory


def manifest(root: Path) -> dict[str, Any]:
    """The corpus as it may be published: pointers, hashes and our extraction.

    Built by reading the directories instead of remembering what was
    written, so only what is on disk gets published and the omission of
    ``page.html`` and ``input.txt`` cannot be undone by a caller passing the
    wrong object.
    """
    documents: list[dict[str, Any]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        meta_path = directory / META_FILE
        truth_path = directory / TRUTH_FILE
        if not meta_path.exists() or not truth_path.exists():
            continue
        entry = _meta_fields(load_meta(meta_path.read_text(encoding="utf-8")))
        entry["truth"] = json.loads(truth_path.read_text(encoding="utf-8"))
        documents.append(entry)
    return {
        "schema_version": MANIFEST_VERSION,
        "user_agent": USER_AGENT,
        "note": "Page text is not redistributed. Fetch each url and check it against content_hash.",
        "documents": documents,
    }


def dump_meta(meta: DocumentMeta) -> str:
    """``meta.yaml`` as a flat mapping of JSON scalars.

    Every value is written the way JSON writes it, which YAML 1.2 reads
    unchanged. The file is therefore loadable by any YAML parser while this
    package needs none of its own: a real YAML writer buys nothing over a
    mapping of strings, numbers and nulls, and a dependency that the corpus
    tooling would carry into every environment that reads a manifest is worth
    more than the quoting it saves.
    """
    lines = [f"{key}: {json.dumps(value)}" for key, value in _meta_fields(meta).items()]
    return "\n".join(lines) + "\n"


def load_meta(text: str) -> DocumentMeta:
    """Read back what :func:`dump_meta` wrote.

    Deliberately narrow: this reads the file this package writes, not YAML in
    general. A meta.yaml a person has reformatted is a corpus that no longer
    round trips, and failing loudly here is better than half reading it.
    """
    fields: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, raw = stripped.partition(":")
        try:
            fields[key.strip()] = json.loads(raw.strip())
        except ValueError as error:
            raise ValueError(f"{key.strip()} is not a JSON scalar: {raw.strip()}") from error
    missing = {"document_id", "url", "content_hash", "text_hash", "fetched_at", "robots_allowed"} - set(fields)
    if missing:
        raise ValueError(f"meta is missing {sorted(missing)}")
    return DocumentMeta(
        document_id=str(fields["document_id"]),
        url=str(fields["url"]),
        content_hash=str(fields["content_hash"]),
        text_hash=str(fields["text_hash"]),
        fetched_at=datetime.fromisoformat(str(fields["fetched_at"])),
        robots_allowed=bool(fields["robots_allowed"]),
        status=fields.get("status"),
        entity_count=int(fields.get("entity_count") or 0),
        source=Source(fields.get("source") or Source.CRAWL.value),
        final_url=fields.get("final_url"),
        licence=fields.get("licence"),
    )


def _meta_fields(meta: DocumentMeta) -> dict[str, Any]:
    """The field order both the file and the manifest use.

    Fixed instead of derived from the dataclass, so a field added for local
    bookkeeping is not published by accident.
    """
    return {
        "document_id": meta.document_id,
        "source": meta.source.value,
        "url": meta.url,
        "final_url": meta.final_url,
        "fetched_at": meta.fetched_at.isoformat(),
        "robots_allowed": meta.robots_allowed,
        "status": meta.status,
        "licence": meta.licence,
        "content_hash": meta.content_hash,
        "text_hash": meta.text_hash,
        "entity_count": meta.entity_count,
    }


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


def _host(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


def _robots_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in dict(headers).items():
        if key.lower() == name:
            return value
    return None


_BLANKET_DISALLOW = ("User-agent: *", "Disallow: /")
"""What an unreadable robots.txt is treated as.

Written as rules rather than set as a flag on the parser, so the decision goes
through the same code path a stated rule does and cannot diverge from it.
"""


def _apply_robots(parser: RobotFileParser, response: Response) -> None:
    """Read one robots.txt, following RFC 9309 where it is silent about us.

    A 5xx or an unreachable server puts the whole host out of reach rather than
    meaning there are no rules: section 2.3.1.4 allows an unavailable
    robots.txt to be treated as a full disallow, and guessing the other way is
    how a collector gets itself banned. 401 and 403 are the same case. A 404
    means there are no rules, which is a yes.
    """
    if response.status == 200:
        parser.parse(response.body.splitlines())
    elif response.status in (401, 403) or response.status == 0 or response.status >= 500:
        parser.parse(_BLANKET_DISALLOW)
    else:
        parser.parse([])
