"""Real schema.org pages, read out of the Common Crawl archive they were seen in.

The generated schema.org corpus renders prose from a fixed set of frames, and
the prose names the slot it is filling: "Logged as person. ... That is its
additional name." A reader who knows no vocabulary can answer it by copying
the words next to the value, which is what
:class:`~oold_llm_bench.runner.controls.SpellingClient` exists to detect. This
module is the half of the replacement that Wikidata cannot supply, because a
knowledge base has no page: ``Recipe``, ``JobPosting``, and the ``Review``,
``Rating``, ``AggregateRating`` and ``Offer`` entities a recipe page nests
inside itself.

*Why the archive and not the live url.* Web Data Commons publishes the urls
and the quads it extracted from the October 2024 crawl. Refetching one of
those urls today fails for three separate reasons, and measured over 115 of
them it failed for every single one: the domain is gone, the mirror that
answers disallows us, or the markup has been removed. The archive fixes a
second problem that a successful refetch would not have fixed. A page's markup
from 2024 compared against its text from today measures two years of editing,
not whether the page states what it annotates. Reading both out of one WARC
record makes the comparison a statement about the page.

*What stops a page becoming a task.* Not grounding. Measured here over the
sample this corpus was built from, the large majority of content property
values do appear in the visible text, and the residue is collapsed nutrition
panels and SEO keyword lists rather than publishers annotating things they do
not say. What stops a page is the page: a product page is a specification
table reading ``Weight:`` then ``7.1oz``, which is the defect this corpus
exists to escape, and an event page from a ticketing platform is a form.
``Product`` and ``Event`` are therefore out of scope and
:data:`MAIN_CLASSES` names the two that are in it.

*Whitespace is the measurement, not a detail.* ``2 pounds (907 g) ground
beef`` reaches the page as seven DOM elements, and the extracted text puts a
newline between each. Matching the markup value against the text line by line
scores far below matching it against the text with every space removed, and
the gap is large enough that a naive implementation would report this corpus
as ungrounded. :func:`squeeze` is that decision.

*Slot leakage is concentrated, so the fix is a list and not a threshold.* A
nutrition panel labels its own fields: every ``calories`` value in the sample
is preceded by the word calories. The narrative properties leak nothing, which
is what makes them worth asking for. :data:`LABELLED_PROPERTIES` drops the
panel wholesale, and :func:`leaks_slot` drops anything else whose property
name is printed in front of its value.

*What may be published.* Web Data Commons states no data licence, only that
the corpora are for research purposes; Common Crawl grants a limited,
non-transferable right of access; and the page text belongs to whoever wrote
it. None of the three is a grant we can pass on. So the committed file holds
urls, WARC coordinates, record digests and our own extraction of the markup,
and never a line of page text. A third party reproduces the documents by
ranged GET against the coordinates and checks them against the digest, which
is what :func:`load_tasks` does locally.
"""

from __future__ import annotations

import bisect
import gzip
import hashlib
import html
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from oold_llm_bench.corpus.catalogue import CatalogueEntry, render_catalogue
from oold_llm_bench.corpus.extract_markup import FULL_TEXT_PROPERTIES, Extraction, normalise_term
from oold_llm_bench.grading.triples import Reference
from oold_llm_bench.tasks.models import (
    CorpusRef,
    Difficulty,
    ExpectedInstance,
    Source,
    Split,
    TaskRecord,
)

__all__ = [
    "CC_DATA",
    "CORPUS_PATH",
    "CRAWL_ID",
    "LABELLED_PROPERTIES",
    "LABEL_WINDOW",
    "LANGUAGE",
    "MAIN_CLASSES",
    "MAX_NESTED_PER_PROPERTY",
    "MAX_VALUE_CHARS",
    "MIN_GROUNDED_PROPERTIES",
    "NESTED_CLASSES",
    "NESTING_DEPTH",
    "PAGES_ENV",
    "STRUCTURAL_PROPERTIES",
    "SUPERSEDED_PROPERTIES",
    "WDC_BASE",
    "WDC_RELEASE",
    "ClusterIndex",
    "Corpus",
    "DocumentStats",
    "Entity",
    "MissingPages",
    "PageDocument",
    "WarcRef",
    "cache_path",
    "corpus_catalogue",
    "entities_of",
    "grounded",
    "leaks_slot",
    "load_tasks",
    "measure_document",
    "page_html",
    "pages_directory",
    "range_fetcher",
    "read_corpus",
    "resolve",
    "shingle",
    "squeeze",
    "subset_urls",
    "usable_document",
    "warc_member",
]

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "wdc_schemaorg.json"

PAGES_ENV = "OOLD_WDC_PAGES"
"""Where a caller says the rebuilt page text is, when it does not pass a path."""

CRAWL_ID = "CC-MAIN-2024-42"
"""The Common Crawl release the Web Data Commons October 2024 corpus is drawn from.

Pinned rather than discovered. A WARC offset means nothing without the crawl
it indexes, and the index for one crawl does not resolve a url seen in another.
"""

WDC_RELEASE = "2024-12"
"""The Web Data Commons release directory holding the class-specific subsets.

Named for its publication month and not for the crawl, which is a trap worth
stating: ``2024-12`` is the October 2024 crawl.
"""

WDC_BASE = f"https://data.dws.informatik.uni-mannheim.de/structureddata/{WDC_RELEASE}/quads/classspecific"
CC_DATA = "https://data.commoncrawl.org"

MAIN_CLASSES = ("Recipe", "JobPosting")
"""The two classes whose pages are documents a person wrote for a reader.

``Product`` and ``Event`` were tried and are out. A product page repeats a
specification block once per colour variant, which breaks one main entity per
document, and its properties reach the reader as a two-column table, which is
the register this corpus exists to leave behind. An event page from a ticketing
platform carries a form and a venue address and no prose at all.
"""

NESTED_CLASSES = ("AggregateRating", "Offer", "Rating", "Review")
"""Classes kept when a page nests them inside a main entity.

The Web Data Commons subsets are page-level, so everything a recipe page
annotates arrives with the recipe. Most of it is markup about the markup:
``ListItem`` is a breadcrumb, ``ImageObject`` is a file, ``HowToStep`` restates
``recipeInstructions`` one step at a time. These four are the ones that carry
values a reader can see, and they are what makes a document hold more than one
entity.
"""

STRUCTURAL_PROPERTIES = frozenset({
    "additionalType",
    "availability",
    "bestRating",
    "datePosted",
    "dateModified",
    "datePublished",
    "directApply",
    "gtin",
    "gtin8",
    "gtin12",
    "gtin13",
    "gtin14",
    "id",
    "identifier",
    "image",
    "inLanguage",
    "itemCondition",
    "mainEntityOfPage",
    "mpn",
    "potentialAction",
    "priceCurrency",
    "priceValidUntil",
    "productID",
    "salaryCurrency",
    "sameAs",
    "sku",
    "subjectOf",
    "thumbnailUrl",
    "type",
    "url",
    "validThrough",
    "worstRating",
})
"""Properties excluded before grounding is counted.

Five groups, and each would push the rate the same way. Urls, identifiers and
image references are addresses rather than claims, and a page never prints
them. ISO dates are stored as ``2024-03-07`` and printed as "March 7, 2024", so
counting them would measure a date formatter. A currency code and a stock
status are vocabulary terms the markup picks from a list while the page shows a
symbol and a button. ``directApply`` is a flag addressed to the crawler, which
nothing on the page corresponds to at all.

``bestRating`` and ``worstRating`` are the fifth and the least obvious. They
are the scale a rating is on and not a fact about the thing, they are 5 and 1
on nearly every page that states them, and a one-digit value is found in any
document by accident. Measured here they were grounded 100% of the time, which
is what a property that cannot fail looks like.
"""

LABELLED_PROPERTIES = frozenset({
    "calories",
    "carbohydrateContent",
    "cholesterolContent",
    "fatContent",
    "fiberContent",
    "proteinContent",
    "recipeYield",
    "saturatedFatContent",
    "servingSize",
    "sodiumContent",
    "sugarContent",
    "transFatContent",
    "unsaturatedFatContent",
})
"""Properties dropped wholesale because their value is printed under its own name.

Slot leakage over the sample is low in aggregate and almost entirely inside
this set: every ``calories`` and every ``sodiumContent`` value is printed
beside the word, ``recipeYield`` and ``servingSize`` nearly always.
:func:`leaks_slot` would remove most of them one at a time, and removing the
set outright says why rather than leaving a pattern of absences a reader has to
infer. The narrative properties, ``name`` and ``description`` and
``recipeIngredient`` and ``recipeInstructions``, leak nothing, and they are the
reason this corpus is worth having.
"""

SUPERSEDED_PROPERTIES = frozenset({"ingredient", "ingredients"})
"""Properties schema.org has replaced, which a page states twice.

``recipeIngredient`` succeeded both of these and older templates still emit one
of them beside it with identical values. Two property names for one list is two
sets of triples an answer has to produce, and a model that gives the current
name once is marked down for not also giving the retired one.
"""

LANGUAGE = "eng"
"""The language a page has to be in, as the crawl index detects it.

An exclusion made once rather than ten times. The duration words a page can
print, the labels :func:`leaks_slot` looks for, the lexicon
:data:`BOILERPLATE_MARKERS` carries and the sentence-ending punctuation
:func:`measure_document` counts are all English, and each would silently
under-report on a page that is not. Restricting the corpus is honest; applying
English heuristics to a French page and reporting the result is not.
"""

LABEL_WINDOW = 70
"""How far in front of a value :func:`leaks_slot` looks for its own name.

Wide enough to catch a table cell and its header, narrow enough that an
unrelated mention of the word elsewhere in a paragraph does not condemn a
value. A nutrition panel puts the two within about twenty characters; this
leaves room for the markup that sits between them.
"""

NESTING_DEPTH = 2
"""How far below the main entity an entity is still part of this document.

Two, because a recipe page's useful structure is exactly two: the recipe
carries a review and the review carries the rating it gave. One level would
leave the rating attached to nothing, and three reaches the breadcrumb and
publisher markup, which describes the site rather than the thing on the page.
"""

MAX_NESTED_PER_PROPERTY = 4
"""Entities one property may point at before the property is dropped.

A page showing twenty reviews is displaying a list, and asking for twenty
near-identical entities measures bookkeeping. Below the cap the entities are
kept together, because the page asserts all of them and an answer giving all of
them is the right answer.
"""

MAX_VALUE_CHARS = 400
"""How long one property value may be before it stops being an answer.

The rule :data:`~oold_llm_bench.corpus.extract_markup.FULL_TEXT_PROPERTIES`
states by name, applied by length. No arm is asked to reproduce an article, and
a job description of three thousand characters is an article: an answer that
differs by one word is marked wrong, so the property would measure
transcription rather than extraction. It is also the line this corpus must not
cross on redistribution, because the committed truth is published and a
three-thousand-character value is the page.
"""

MIN_GROUNDED_PROPERTIES = 3
"""Grounded content properties a document needs before it is a task.

Below three a task is answerable by naming the class and the title, which is a
classification task with one value attached and not an extraction.
"""

MIN_SENTENCE_LINES = 10
"""Lines of running prose a document needs.

The filter that actually binds. Grounding passes on a product page, because the
specification table really does state its values; what disqualifies it is that
nothing on it is a sentence. Ten is where recipe and job pages sit comfortably
above and specification pages sit below.
"""

SENTENCE_LINE_CHARS = 60
"""How long a line has to be before ending in a full stop makes it a sentence.

A short line ending in a period is usually an abbreviation in a nav label or a
file name.
"""

MAX_BOILERPLATE = 0.15
"""Share of the document that may be cookie notices, menus and legal text."""

_SPEC_LINE_CHARS = 60
"""How long a line may be and still be a cell of a table rather than prose."""

MAX_SPEC_RATIO = 1.5
"""Label-and-value lines per sentence line, above which the page is a table.

Stated as a ratio rather than a count because a long recipe page legitimately
carries an ingredient list that looks like label-value pairs. What separates a
recipe from a specification sheet is that the recipe also has prose.
"""

SHINGLE_WORDS = 8
"""Words per shingle when two descriptions are compared.

Variant pages repeat marketing copy verbatim, and a shingle long enough to be
specific is the cheapest way to see it without comparing every pair of
documents.
"""

_WARC_FIELD = re.compile(r"^(?P<name>[A-Za-z-]+):\s*(?P<value>.*)$")
_ISO_DURATION = re.compile(
    r"^P(?:(?P<days>\d+(?:\.\d+)?)D)?(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$")
_NUMERIC = re.compile(r"^-?\d+(?:[.,]\d+)?$")
_WORDS = re.compile(r"[a-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPEC_PAIR = re.compile(r"^[^:]{1,40}:\s*\S")
_TAG = re.compile(r"<[^>]{0,400}>")
_GRAPH_NAME = re.compile(r"<(https?://[^>]+)>\s+\.\s*$")
_SENTENCE_END = (".", "!", "?")

_SECONDS = {"days": 86400.0, "hours": 3600.0, "minutes": 60.0, "seconds": 1.0}

_DURATION_WORDS: dict[str, tuple[str, ...]] = {
    "days": ("day", "days", "d"),
    "hours": ("hour", "hours", "hr", "hrs", "h"),
    "minutes": ("minute", "minutes", "min", "mins", "m"),
    "seconds": ("second", "seconds", "sec", "secs", "s"),
}

_DURATION_SPAN = 40
"""Characters a duration's unit word may sit behind its number.

``PT25M`` is printed as "25 to 30 mins" often enough that requiring the number
and the word to be adjacent loses the property. Forty characters reaches across
a range and the markup between its halves.
"""

BOILERPLATE_MARKERS = (
    "all rights reserved",
    "cookie",
    "privacy policy",
    "terms of service",
    "terms and conditions",
    "subscribe",
    "newsletter",
    "sign in",
    "log in",
    "create an account",
    "skip to content",
    "skip to main",
    "follow us",
    "share this",
    "copyright",
    "©",
)
"""Phrases that mark a line as furniture rather than content.

A lexicon and not a classifier, because the decision has to be reproducible
from this file alone by anyone checking the corpus statistics. It is a floor on
the boilerplate a page carries and never an estimate of all of it, which is why
the threshold it feeds is low.
"""


@dataclass(frozen=True)
class WarcRef:
    """Where one page sits in the archive, exactly enough to fetch it again."""

    url: str
    crawl: str
    filename: str
    offset: int
    length: int
    digest: str
    """The WARC record's own payload digest, base32 as the index publishes it."""
    timestamp: str = ""
    status: str = "200"
    charset: str = ""
    languages: str = ""
    """What the index detected the page is written in, as ISO 639-3 codes."""

    def describe(self) -> dict[str, Any]:
        return {
            "crawl": self.crawl,
            "warc_filename": self.filename,
            "warc_offset": self.offset,
            "warc_length": self.length,
            "warc_digest": self.digest,
            "warc_timestamp": self.timestamp,
        }

    @property
    def fetched_at(self) -> datetime:
        """When the crawler saw the page, from the index timestamp."""
        stamp = self.timestamp or ""
        if len(stamp) >= 8:
            return datetime(
                int(stamp[0:4]),
                int(stamp[4:6]),
                int(stamp[6:8]),
                tzinfo=UTC,
            )
        return datetime(2024, 10, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Entity:
    """One entity a document asserts, after every filter has run."""

    key: str
    class_name: str
    fields: dict[str, Any]
    """Property name to the value the markup gives it, links excluded."""
    links: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Property name to the keys of the entities in this document it reaches.

    Several targets and not one, because a page states three reviews under one
    property. An edge to a list is how the page says so, and keeping only the
    first would make the complete answer wrong on precision.
    """

    def expected(self) -> ExpectedInstance:
        values: dict[str, Any] = dict(self.fields)
        for name, targets in self.links.items():
            found = [Reference(key=key) for key in targets]
            values[name] = found[0] if len(found) == 1 else found
        return ExpectedInstance(key=self.key, class_path=self.class_name, fields=values)


@dataclass(frozen=True)
class DocumentStats:
    """What the document filter measured, kept so a rejection can be explained."""

    lines: int
    sentence_lines: int
    spec_lines: int
    boilerplate_lines: int
    characters: int

    @property
    def boilerplate_ratio(self) -> float:
        return self.boilerplate_lines / self.lines if self.lines else 1.0

    @property
    def spec_ratio(self) -> float:
        """Label-value lines per sentence line, capped when there are no sentences."""
        if not self.sentence_lines:
            return float("inf") if self.spec_lines else 0.0
        return self.spec_lines / self.sentence_lines

    def describe(self) -> dict[str, Any]:
        return {
            "lines": self.lines,
            "sentence_lines": self.sentence_lines,
            "spec_lines": self.spec_lines,
            "boilerplate_lines": self.boilerplate_lines,
            "characters": self.characters,
        }


@dataclass(frozen=True)
class PageDocument:
    """One page as the committed corpus carries it, which is without its text."""

    id: str
    url: str
    host: str
    main_class: str
    warc: WarcRef
    text_sha256: str
    """sha256 of the visible text the build extracted, so a cache can be checked.

    This is what stands in for the document. A third party fetches the WARC
    record, runs the same extraction and compares; a mismatch says the
    extractor changed, not the page, because the page is frozen in the archive.
    """
    entities: tuple[Entity, ...]
    stats: DocumentStats
    properties_seen: int = 0
    properties_grounded: int = 0

    @property
    def grounding_rate(self) -> float:
        return self.properties_grounded / self.properties_seen if self.properties_seen else 0.0


@dataclass(frozen=True)
class Corpus:
    """The built corpus and the arithmetic that says nothing left unexplained."""

    documents: tuple[PageDocument, ...]
    excluded: dict[str, int]
    urls_in: int
    resolved: int
    per_class: dict[str, int]
    cap: int | None
    crawl: str
    release: str
    licence: dict[str, Any]
    grounding: dict[str, Any]
    built_at: str

    @property
    def dropped_by_cap(self) -> int:
        """How many documents the per-class cap held back.

        Reported rather than hidden. A cap that bound is a corpus that could
        have been larger, and a reader comparing two builds needs to know
        whether a class ran out of pages or out of quota.
        """
        return self.resolved - len(self.documents)


Ranged = Callable[[str, int, int], bytes]
"""A ranged GET. The network is a parameter here for the reason it is one in
:mod:`oold_llm_bench.corpus.crawl`: every path through this module has to be
reachable from a fixture."""


def range_fetcher(
    timeout: float = 60.0,
    user_agent: str | None = None,
    attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> Ranged:
    """The real ranged GET, behind the interface the tests use.

    Only https is accepted. The coordinates come from a published index, which
    is data from outside, and a fetcher that follows whatever scheme it is
    handed reads local files for anyone who can edit that index.

    The retry is not optional here. A corpus is a few thousand ranged GETs
    against one host, and that host answers 503 when it wants the rate lowered;
    a build without a backoff dies partway through and leaves no record of
    where. A stated ``Retry-After`` wins outright, because the server is saying
    what it needs.
    """
    from oold_llm_bench.corpus.crawl import RETRY_STATUSES, USER_AGENT

    agent = user_agent or USER_AGENT

    def fetch(url: str, offset: int, length: int) -> bytes:
        if urlsplit(url).scheme != "https":
            raise ValueError(f"refusing a non-https url: {url}")
        request = urllib.request.Request(  # noqa: S310 - the scheme is checked above
            url,
            headers={"User-Agent": agent, "Range": f"bytes={offset}-{offset + length - 1}"},
        )
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - same check
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code not in RETRY_STATUSES or attempt == attempts:
                    raise
                sleep(_retry_after(error) or 2.0**attempt)
            except urllib.error.URLError:
                if attempt == attempts:
                    raise
                sleep(2.0**attempt)
        raise RuntimeError(f"{url} was not answered in {attempts} attempts")

    return fetch


def _retry_after(error: urllib.error.HTTPError) -> float | None:
    stated = error.headers.get("Retry-After") if error.headers else None
    try:
        return max(float(str(stated).strip()), 1.0) if stated else None
    except ValueError:
        return None


def surt(url: str) -> str:
    """The key the Common Crawl index sorts on.

    Host labels reversed so one site is one contiguous range, ``www`` dropped,
    path lowercased, and the trailing slash removed from anything but the root.
    Each of those is the index's own canonicalisation: a key that keeps the
    trailing slash sorts next to the record it was meant to find and matches
    nothing, which is a silent miss rather than an error.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (parts.path or "/").lower()
    if len(path) > 1:
        path = path.rstrip("/")
    key = ",".join(reversed(host.split("."))) + ")" + path
    return key + ("?" + parts.query.lower() if parts.query else "")


class ClusterIndex:
    """The Common Crawl secondary index, read once and searched locally.

    The public lookup API is a convenience that answers one url per request and
    was returning gateway errors throughout this build. The index it serves is
    published as files, so this reads the 121 MB of block boundaries once and
    then resolves a url with a single ranged GET for the block that holds it.
    That is both faster and less of an imposition than a request per url.
    """

    def __init__(self, blocks: Iterable[tuple[str, str, int, int]], crawl: str = CRAWL_ID) -> None:
        self.blocks = sorted(blocks)
        self.keys = [block[0] for block in self.blocks]
        self.crawl = crawl

    @classmethod
    def read(cls, path: Path, crawl: str = CRAWL_ID) -> ClusterIndex:
        blocks: list[tuple[str, str, int, int]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(" ")
            fields = rest.split("\t")
            if len(fields) < 4:
                continue
            blocks.append((key, fields[1], int(fields[2]), int(fields[3])))
        if not blocks:
            raise ValueError(f"{path} holds no index blocks, so it is not a cluster.idx")
        return cls(blocks, crawl)

    def block_for(self, key: str) -> tuple[str, int, int]:
        """The index block whose key range covers this one."""
        position = max(bisect.bisect_right(self.keys, key) - 1, 0)
        _, name, offset, length = self.blocks[position]
        return name, offset, length

    def url(self, name: str) -> str:
        return f"{CC_DATA}/cc-index/collections/{self.crawl}/indexes/{name}"


def resolve(url: str, index: ClusterIndex, fetch: Ranged) -> WarcRef | None:
    """Where this url's page sits in the archive, or nothing if it is not there.

    The first successful capture wins. A url crawled twice in one release
    differs between the captures by minutes, and taking the first keeps the
    choice independent of how the index happens to order a tie.
    """
    key = surt(url)
    name, offset, length = index.block_for(key)
    block = gzip.decompress(fetch(index.url(name), offset, length)).decode("utf-8", "replace")
    for line in block.splitlines():
        if not line.startswith(key + " "):
            continue
        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue
        record = json.loads(parts[2])
        if record.get("status") != "200" or "warc" not in record.get("filename", ""):
            continue
        return WarcRef(
            url=record.get("url") or url,
            crawl=index.crawl,
            filename=record["filename"],
            offset=int(record["offset"]),
            length=int(record["length"]),
            digest=record.get("digest", ""),
            timestamp=parts[1],
            status=record.get("status", ""),
            charset=record.get("charset", ""),
            languages=record.get("languages", ""),
        )
    return None


def warc_member(ref: WarcRef, fetch: Ranged) -> bytes:
    """The one gzip member holding this record.

    A WARC file concatenates independently compressed members, so the bytes a
    record's offset and length name decompress on their own. That is what makes
    one page one request instead of a 1 GB download.
    """
    return gzip.decompress(fetch(f"{CC_DATA}/{ref.filename}", ref.offset, ref.length))


def page_html(member: bytes, charset: str = "") -> str:
    """The response body out of one WARC member.

    Two headers come off, the WARC record's and the HTTP response's, and the
    encoding is looked for in four places in decreasing order of authority:
    the HTTP header, the page's own ``meta`` declaration, the charset the crawl
    index detected, and then cp1252. The last is not a guess, it is the
    commonest lie: a page that declares UTF-8 and contains a Windows smart
    quote decodes to a replacement character under its own declaration and to
    the right character under cp1252. A document whose apostrophes are
    replacement characters fails the grounding check wherever the markup kept
    the real one, so this is a correctness decision and not a tidiness one.
    """
    _, _, rest = member.partition(b"\r\n\r\n")
    http_head, _, body = rest.partition(b"\r\n\r\n")
    declared = _charset_of(http_head.decode("latin-1", "replace"))
    for candidate in (declared, _meta_charset(body), charset, "utf-8", "cp1252"):
        if not candidate:
            continue
        try:
            return body.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", "replace")


def _charset_of(headers: str) -> str:
    for line in headers.splitlines():
        found = _WARC_FIELD.match(line)
        if found and found.group("name").lower() == "content-type":
            return _charset_in(found.group("value"))
    return ""


def _charset_in(value: str) -> str:
    lowered = value.lower()
    return lowered.split("charset=", 1)[1].strip().strip("\"'; ") if "charset=" in lowered else ""


def _meta_charset(body: bytes) -> str:
    """What the page says about its own encoding, from the head of the file."""
    head = body[:4096].decode("latin-1", "replace")
    found = re.search(r"<meta[^>]+charset\s*=\s*[\"']?([A-Za-z0-9_-]+)", head, re.IGNORECASE)
    return found.group(1).lower() if found else ""


def subset_urls(path: Path, per_host: int = 1) -> Iterator[tuple[str, str]]:
    """Urls out of a Web Data Commons class subset, capped per host.

    The subsets are clustered by pay-level domain, so the first thousand lines
    of a part file are one site. Yielding at most ``per_host`` urls per host as
    the file is read is what keeps a sample from being a single domain, and it
    is done here rather than afterwards so the whole file never has to be held.

    The file is read as lines rather than parsed as N-Quads. The graph name is
    the last angle-bracketed term on a line and is the page the quad came from,
    which is the only field this needs; a quad parser would also have to survive
    every escaping accident in two billion lines of extracted markup.
    """
    seen: dict[str, int] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:  # type: ignore[operator]
        for line in _tolerant(handle):
            found = _GRAPH_NAME.search(line.rstrip())
            if not found:
                continue
            url = found.group(1)
            host = (urlsplit(url).hostname or "").lower()
            if not host or seen.get(host, 0) >= per_host:
                continue
            seen[host] = seen.get(host, 0) + 1
            yield url, host


def _tolerant(handle: Iterable[str]) -> Iterator[str]:
    """Read a gzip stream that stops in the middle of a member.

    The subsets are 180 MB per part and this corpus needs the first few
    thousand hosts out of each, so the build fetches a prefix. A truncated
    gzip stream raises at the end, and the lines before it are the ones that
    were wanted.
    """
    try:
        yield from handle
    except (EOFError, OSError, gzip.BadGzipFile):
        return


_TYPOGRAPHY = (
    dict.fromkeys(range(0x2010, 0x2016), "")
    | dict.fromkeys((0x2018, 0x2019, 0x201B, 0x2032), "'")
    | dict.fromkeys((0x201C, 0x201D, 0x201F, 0x2033), '"')
    | dict.fromkeys((0x00AD, 0x200B, 0x200C, 0x200D, 0xFEFF, 0x2212), "")
    | {ord("-"): "", ord("_"): ""}
)
"""Characters a template and a theme spell differently for the same words.

A CMS stores ``Alyssa's`` in the markup and renders it through a smart-quote
filter, so the page shows a right single quotation mark where the annotation
holds an apostrophe. Measured on the sample, that one substitution is the
commonest reason a title and a description look ungrounded. Soft hyphens and
zero-width joiners come from the same place and are invisible to the reader
this check stands in for.

The hyphen and the underscore are removed rather than folded, and that is the
one entry here with a cost. schema.org enumerations are written ``FULL_TIME``
and the page prints "Full-time", so without it every enumerated job property
reads as ungrounded and the one slot where a closed vocabulary could help is
the one slot the corpus would not contain. What it costs is a minus sign: a
negative number squeezes to its magnitude. No property this corpus keeps takes
a negative value, and a count, a price and a rating cannot.
"""


def squeeze(text: str) -> str:
    """One string with every space removed, folded for case and typography.

    This is the methodological decision the grounding rate turns on. A recipe
    ingredient reaches the page as ``2`` ``pounds`` ``(`` ``907`` ``g`` ``)``
    ``ground beef`` in seven elements, and the extracted text puts a line break
    between each of them. Comparing the markup value against that line by line
    reports the page as not stating what it annotates; comparing it with the
    whitespace gone reports what a reader sees.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY)
    return "".join(folded.split()).casefold()


def _number_forms(value: str) -> tuple[str, ...]:
    """How a page might print one number the markup stores plainly.

    ``129.0`` is printed ``$129``, ``1200`` is printed ``1,200`` and ``1.200``
    depending on the locale. Only separators and a redundant decimal part are
    added, never a rounding: a page that prints 129 for 128.6 is a page stating
    something else.
    """
    plain = value.strip().replace(",", ".")
    try:
        number = float(plain)
    except ValueError:
        return (value,)
    forms = {value.strip(), plain}
    if number == int(number):
        whole = str(int(number))
        forms |= {whole, f"{int(number):,}", f"{int(number):,}".replace(",", "."), f"{int(number):,}".replace(",", " ")}
    else:
        forms |= {plain, plain.replace(".", ",")}
    return tuple(sorted(forms))


def _duration_parts(value: str) -> tuple[tuple[str, float], ...]:
    found = _ISO_DURATION.match(value.strip())
    if not found:
        return ()
    parts = [(name, float(raw)) for name, raw in found.groupdict().items() if raw]
    return tuple(parts)


def _renderings(parts: tuple[tuple[str, float], ...]) -> list[tuple[tuple[str, float], ...]]:
    """The ways a page can write one duration, as sets of components to find.

    Three, and each is a real convention. As the markup writes it, because most
    pages agree with their own annotation. Carried down into days, hours and
    minutes, because ``PT380M`` is printed "6 hr 20 min". And as a single unit,
    because ``PT1H`` is printed "60 minutes" as often as "1 hour".
    """
    seconds = sum(amount * _SECONDS[name] for name, amount in parts)
    found: list[tuple[tuple[str, float], ...]] = [parts]
    carried: list[tuple[str, float]] = []
    left = seconds
    for name in ("days", "hours", "minutes", "seconds"):
        whole, left = divmod(left, _SECONDS[name])
        if whole:
            carried.append((name, whole))
    if carried:
        found.append(tuple(carried))
    for name, factor in _SECONDS.items():
        amount = seconds / factor
        if amount and amount == int(amount):
            found.append(((name, amount),))
    return found


def _duration_grounded(value: str, text: str) -> bool:
    """Whether the page states this duration, however it chose to write it.

    ``PT25M`` is "25 mins", "25 minutes" and "25 to 30 mins", and a corpus that
    accepted only the first would report cooking times as ungrounded on most
    pages that state them. Within one rendering every component has to be
    found: a page that says "1 hour" has not stated ``PT1H30M``.

    A component is found when its number appears in the text with a word for
    its unit within :data:`_DURATION_SPAN` characters after it. The window is
    what lets a range be matched; requiring adjacency would not.
    """
    parts = _duration_parts(value)
    if not parts:
        return False
    lowered = text.casefold()
    return any(
        all(_component_in(name, amount, lowered) for name, amount in rendering) for rendering in _renderings(parts)
    )


def _component_in(name: str, amount: float, lowered: str) -> bool:
    """One component of a duration, found by its number and a word for its unit.

    The unit word needs no word boundary in front of it, only the absence of a
    letter behind it. "90m" and "1h30" are how a recipe card writes a time, and
    a boundary before the word would reject both because a digit and a letter
    are both word characters.
    """
    rendered = str(int(amount)) if amount == int(amount) else str(amount)
    words = "|".join(_DURATION_WORDS[name])
    pattern = re.compile(rf"(?<!\d){re.escape(rendered)}(?!\d).{{0,{_DURATION_SPAN}}}?(?:{words})(?![a-z])", re.DOTALL)
    return bool(pattern.search(lowered))


def grounded(value: Any, text: str, squeezed: str | None = None) -> bool:
    """Whether the document states this value.

    Three shapes, three tests, and the shape decides. A duration is compared
    through :func:`_duration_grounded`, because the markup stores ``PT25M`` and
    no page prints that. A number is compared through its separator forms,
    because the markup stores ``129.0`` and the page prints ``$129``. Anything
    else is a substring test with the whitespace gone, which is what
    :func:`squeeze` exists for.

    A date is never grounded here and is not asked to be: dates sit in
    :data:`STRUCTURAL_PROPERTIES` and never reach this function from a corpus
    build.
    """
    if value is None or isinstance(value, bool):
        return False
    text_value = str(value).strip()
    if not text_value:
        return False
    haystack = squeezed if squeezed is not None else squeeze(text)
    if _ISO_DURATION.match(text_value) and text_value.startswith("P"):
        return _duration_grounded(text_value, text)
    if _NUMERIC.match(text_value):
        return _number_grounded(text_value, text)
    return squeeze(text_value) in haystack


def _number_grounded(value: str, text: str) -> bool:
    """Whether the page prints this number, and not a longer one containing it.

    The boundary is the point. A substring test says the page states 129
    because it somewhere prints 1129, and on a short value it says the page
    states 5 because the word "5" occurs. Requiring no digit on either side
    removes the first error outright; the second is why a one-digit scale
    belongs in :data:`STRUCTURAL_PROPERTIES` rather than in an answer.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY)
    return any(re.search(rf"(?<!\d){re.escape(form)}(?!\d)", folded) is not None for form in _number_forms(value))


def leaks_slot(prop: str, value: Any, text: str) -> bool:
    """Whether the page prints the property's own name in front of its value.

    This is the defect the generated corpus has by construction and the one
    thing a real page can reproduce. "Calories: 350" lets a reader fill the
    ``calories`` slot without knowing the vocabulary, and a benchmark built on
    such values measures a label matcher.

    Only the head noun of the property is looked for: ``recipeIngredient`` is
    caught by "ingredient" and not by "recipe", which is printed on every
    recipe page and would condemn the whole class. The search runs on the
    squeezed text, so the window is about a dozen words rather than seventy
    characters of rendering.
    """
    head = _CAMEL.split(prop)[-1].casefold()
    if len(head) < 4:
        return False
    needle = squeeze(str(value))
    if not needle:
        return False
    lowered = squeeze(text)
    position = lowered.find(needle)
    if position < 0:
        return False
    return head in lowered[max(0, position - LABEL_WINDOW) : position]


def measure_document(text: str) -> DocumentStats:
    """Count the three things the document filter decides on.

    A sentence line is long and ends in a full stop. A spec line is a short
    label, a colon and a value, which is how a specification table renders once
    the markup is gone. A boilerplate line carries one of
    :data:`BOILERPLATE_MARKERS`. All three are counted over the same lines, so
    the ratios between them are ratios and not estimates.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = sum(1 for line in lines if len(line) >= SENTENCE_LINE_CHARS and line.endswith(_SENTENCE_END))
    specs = sum(1 for line in lines if _is_spec(line))
    noise = sum(1 for line in lines if any(marker in line.casefold() for marker in BOILERPLATE_MARKERS))
    return DocumentStats(
        lines=len(lines),
        sentence_lines=sentences,
        spec_lines=specs,
        boilerplate_lines=noise,
        characters=len(text),
    )


def _is_spec(line: str) -> bool:
    """Whether one line is half of a label-and-value pair.

    Both halves count, and the second is the one that matters. A specification
    table puts its label in one cell and its value in the next, so the markup
    stripper writes "Weight:" and "7.1oz" on separate lines, and a pattern
    needing both on one line fires on none of the pages it was written for.
    """
    if len(line) > _SPEC_LINE_CHARS:
        return False
    return line.endswith(":") or bool(_SPEC_PAIR.match(line))


def usable_document(stats: DocumentStats) -> str | None:
    """Why this page is not a document, or ``None`` if it is one.

    The reason is a fixed phrase and never carries the measurement that
    produced it. A build counts refusals by reason, and a reason that reads
    "7 sentence lines" makes eight separate reasons out of one filter, which
    is a table nobody can read and an accounting check that proves less.
    """
    if stats.sentence_lines < MIN_SENTENCE_LINES:
        return f"the page carries fewer than {MIN_SENTENCE_LINES} lines of running prose"
    if stats.boilerplate_ratio >= MAX_BOILERPLATE:
        return f"menus, cookie notices and legal text are {MAX_BOILERPLATE:.0%} or more of the lines"
    if stats.spec_ratio >= MAX_SPEC_RATIO:
        return "the page holds more label-value lines than sentences, so it is a table"
    return None


def shingle(text: str, words: int = SHINGLE_WORDS) -> str:
    """A fingerprint of the first few words, for spotting repeated copy.

    Variant pages differ in a size or a colour and repeat the description word
    for word, and one page per piece of copy is what "one document per thing"
    means here. Text shorter than one shingle has no fingerprint rather than a
    short one: two pages that both carry three words are not thereby the same
    page.
    """
    tokens = _WORDS.findall(unicodedata.normalize("NFKC", text).casefold())
    return " ".join(tokens[:words]) if len(tokens) >= words else ""


def _scalar(value: Any) -> Any:
    """One markup value reduced to something gradeable, or ``None``.

    A wrapper object is unwrapped to the one string it carries. Half the recipe
    pages write ``recipeInstructions`` as a list of ``HowToStep`` objects and
    half as a list of strings, and an author reaches the markup as a ``Person``
    with a name about as often as it reaches it as a name. Reading only the
    strings would lose the property on whichever half writes it the other way,
    which is a measurement of CMS conventions rather than of pages.

    An object that is kept as an entity in its own right never arrives here;
    :func:`entities_of` takes those first.
    """
    carried = None
    if isinstance(value, dict):
        for key in ("name", "text"):
            found = value.get(key)
            if isinstance(found, str) and found.strip():
                carried = found
                break
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        carried = str(value)
    if carried is None:
        return None
    collapsed = _detag(carried)
    return collapsed if collapsed and len(collapsed) <= MAX_VALUE_CHARS else None


def _detag(value: str) -> str:
    """One markup value as the page renders it, with its own HTML taken out.

    A job description reaches the markup as ``<p>We are looking for...</p>``
    and reaches the reader as the sentence. Some templates escape it once and
    some twice, so the unescaping runs until it stops changing. Without this
    the property reads as ungrounded on most job pages, which would be a
    measurement of how many publishers paste HTML into a JSON string.
    """
    text = value
    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return " ".join(_TAG.sub(" ", text).split())


def _values(value: Any) -> list[Any] | None:
    """Every value of one property, or nothing if one of them is not a value.

    All or none, like the grounding rule it feeds. A list of twelve ingredients
    of which one is too long to be an answer is not a list of eleven
    ingredients: the page states twelve, and a corpus asking for eleven marks
    the right answer down for the one it removed.
    """
    if isinstance(value, list):
        reduced = [_scalar(entry) for entry in value]
        return None if any(entry is None for entry in reduced) else reduced
    single = _scalar(value)
    return [single] if single is not None else None


def _nested(value: Any) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else [value]
    return [item for item in items if isinstance(item, dict)]


def _class_of(node: dict[str, Any], wanted: Iterable[str]) -> str | None:
    declared = node.get("@type", node.get("type"))
    names = [declared] if isinstance(declared, str) else list(declared or [])
    allowed = set(wanted)
    for name in names:
        if isinstance(name, str) and normalise_term(name) in allowed:
            return normalise_term(name)
    return None


@dataclass
class _Counter:
    """Grounding counts accumulated while a page is being read."""

    seen: int = 0
    kept: int = 0
    by_property: dict[str, list[int]] = field(default_factory=dict)

    def record(self, prop: str, ok: bool) -> None:
        self.seen += 1
        self.kept += int(ok)
        row = self.by_property.setdefault(prop, [0, 0])
        row[0] += 1
        row[1] += int(ok)


def _fields_of(node: dict[str, Any], text: str, squeezed: str, counts: _Counter, allowed: set[str]) -> dict[str, Any]:
    """One node's literal properties, keeping only what the page states whole.

    The order of the refusals matters and is the order they are written in. A
    structural property is never counted, because counting an image url as
    ungrounded would put the grounding rate below what the page actually does.
    A labelled property is dropped before grounding, because it would pass and
    should not be asked. A property that reaches an entity is left to
    :func:`_tree`, or the same property would be both a value here and a link
    there. What is left is counted and then judged.

    A property is kept only when **every** one of its values is grounded and
    none of them leaks. Keeping the grounded nine of twelve ingredients would
    build a corpus where the correct answer, the twelve the page lists, is
    marked down on precision for the three the filter removed. A property is
    all of what the page says it is or it is not asked.
    """
    kept: dict[str, Any] = {}
    for raw, value in node.items():
        prop = normalise_term(raw)
        if prop.startswith("@") or prop in STRUCTURAL_PROPERTIES or prop in FULL_TEXT_PROPERTIES:
            continue
        if prop in LABELLED_PROPERTIES or prop in SUPERSEDED_PROPERTIES:
            continue
        if any(_class_of(child, allowed) for child in _nested(value)):
            continue
        values = _values(value)
        if not values:
            continue
        whole = True
        for item in values:
            ok = grounded(item, text, squeezed)
            counts.record(prop, ok)
            whole = whole and ok and not leaks_slot(prop, item, text)
        if whole:
            kept[prop] = values[0] if len(values) == 1 else values
    return kept


def entities_of(
    extraction: Extraction,
    text: str,
    main_classes: Iterable[str] = MAIN_CLASSES,
    nested_classes: Iterable[str] = NESTED_CLASSES,
) -> tuple[tuple[Entity, ...], _Counter]:
    """The entities one page asserts, with the values the page also states.

    One main entity per document. A page that annotates two recipes is a listing
    page, and the question this benchmark asks, which thing does this document
    describe, has no single answer on one. The nested entities come with it,
    because the subsets are page-level and a rating is only interesting while it
    is still attached to the thing it rates.
    """
    squeezed = squeeze(text)
    counts = _Counter()
    mains = [
        entity for entity in extraction.typed if _class_of(entity.node, main_classes) and not _is_listing(entity.node)
    ]
    if len(mains) != 1:
        return (), counts
    node = mains[0].node
    tree = _tree(node, _class_of(node, main_classes) or "", text, squeezed, counts, set(nested_classes), 0)
    return _flatten(tree), counts


@dataclass
class _Node:
    """One entity while the page is still being read, before keys are assigned.

    Keys come last because a child that turns out to state nothing is not an
    entity, and numbering as the tree is walked would leave a gap where it was.
    """

    class_name: str
    fields: dict[str, Any]
    children: dict[str, list[_Node]]


def _tree(
    node: dict[str, Any],
    class_name: str,
    text: str,
    squeezed: str,
    counts: _Counter,
    allowed: set[str],
    depth: int,
) -> _Node:
    """One entity and the entities hanging off it, to a fixed depth.

    Two levels below the main entity, which is what a recipe page needs: the
    recipe carries a review and the review carries the rating it gave. Stopping
    at one would turn the rating into a property of nothing, and going deeper
    reaches the breadcrumb markup that describes the site rather than the thing.

    A property's children are kept together or not at all, for the reason a
    property's values are: a page that states three reviews has three reviews
    as its answer, and keeping the two that yielded fields would mark the
    complete answer down. More than :data:`MAX_NESTED_PER_PROPERTY` under one
    property is a list the page is displaying rather than a description, and
    the property goes.
    """
    fields = _fields_of(node, text, squeezed, counts, allowed)
    children: dict[str, list[_Node]] = {}
    if depth >= NESTING_DEPTH:
        return _Node(class_name=class_name, fields=fields, children=children)
    for raw, value in node.items():
        prop = normalise_term(raw)
        if prop in STRUCTURAL_PROPERTIES:
            continue
        found = [(child, _class_of(child, allowed)) for child in _nested(value)]
        named = [(child, name) for child, name in found if name]
        if not named or len(named) > MAX_NESTED_PER_PROPERTY:
            continue
        built = [_tree(child, name or "", text, squeezed, counts, allowed, depth + 1) for child, name in named]
        if all(entry.fields for entry in built):
            children[prop] = built
    return _Node(class_name=class_name, fields=fields, children=children)


def _flatten(root: _Node) -> tuple[Entity, ...]:
    """The tree as a flat list of entities whose links name each other by key."""
    entities: list[Entity | None] = []

    def walk(node: _Node) -> str:
        position = len(entities)
        entities.append(None)
        key = f"e{position + 1}"
        links = {prop: tuple(walk(child) for child in kids) for prop, kids in node.children.items()}
        entities[position] = Entity(key=key, class_name=node.class_name, fields=node.fields, links=links)
        return key

    walk(root)
    return tuple(entity for entity in entities if entity is not None)


def _is_listing(node: dict[str, Any]) -> bool:
    """Whether this node is an index of other things rather than one of them.

    A ``mainEntity`` or ``itemListElement`` holding objects is a listing page's
    own markup, and its properties describe the list.
    """
    return any(
        normalise_term(key) in ("itemListElement", "mainEntity") and _nested(value) for key, value in node.items()
    )


class MissingPages(RuntimeError):
    """The documents are not on this machine, and they are not in the repository.

    Its own exception rather than a ``FileNotFoundError``, because a caller
    that has every other corpus available needs to tell "this one has to be
    rebuilt" from "something is wrong with the installation".
    """


def pages_directory(explicit: str | Path | None = None) -> Path:
    """Where the rebuilt page text lives, or a message saying how to get it."""
    raw = explicit or os.environ.get(PAGES_ENV)
    if not raw:
        raise MissingPages(
            f"no page cache: pass a directory or set {PAGES_ENV}. The page text is the site "
            f"owner's and is never committed, so it is rebuilt from the WARC coordinates with "
            f"scripts/build_wdc_schemaorg.py --pages-only"
        )
    directory = Path(raw).expanduser()
    if not directory.is_dir():
        raise MissingPages(f"{directory} is not a directory, so there is no page cache to read")
    return directory


def cache_path(root: Path, document: PageDocument) -> Path:
    """Where the locally held text of one document lives.

    Outside the repository by default and never committed. The text is the
    site owner's, and this package publishes the coordinates that reach it.
    """
    return root / f"{document.id}.txt"


def read_corpus(path: Path | None = None) -> Corpus:
    """Read the committed corpus, refusing a file whose counts do not add up.

    The guard is the one :func:`~oold_llm_bench.corpus.wiki_measurements.read_corpus`
    carries and it is here for the same reason: a corpus that has quietly lost
    half of itself still loads, and a number measured on it is not comparable
    to a number measured before the loss.
    """
    payload = json.loads((path or CORPUS_PATH).read_text(encoding="utf-8"))
    documents = tuple(_document_from(entry, payload["crawl"]) for entry in payload["documents"])
    corpus = Corpus(
        documents=documents,
        excluded=dict(payload["excluded"]),
        urls_in=int(payload["urls_in"]),
        resolved=int(payload["resolved"]),
        per_class=dict(payload["per_class"]),
        cap=payload.get("cap"),
        crawl=payload["crawl"],
        release=payload["release"],
        licence=dict(payload["licence"]),
        grounding=dict(payload["grounding"]),
        built_at=payload["built_at"],
    )
    accounted = corpus.resolved + sum(corpus.excluded.values())
    if accounted != corpus.urls_in:
        raise ValueError(
            f"{corpus.urls_in} urls went in and {accounted} are accounted for, "
            f"so {corpus.urls_in - accounted} left without a declared reason"
        )
    if sum(corpus.per_class.values()) != corpus.resolved:
        raise ValueError(
            f"{corpus.resolved} documents resolved but the per-class counts sum to {sum(corpus.per_class.values())}"
        )
    if len(corpus.documents) > corpus.resolved:
        raise ValueError(f"the file holds {len(corpus.documents)} documents but only {corpus.resolved} resolved")
    return corpus


def _document_from(entry: dict[str, Any], crawl: str) -> PageDocument:
    return PageDocument(
        id=entry["id"],
        url=entry["url"],
        host=entry["host"],
        main_class=entry["main_class"],
        warc=WarcRef(
            url=entry["url"],
            crawl=entry.get("crawl") or crawl,
            filename=entry["warc_filename"],
            offset=int(entry["warc_offset"]),
            length=int(entry["warc_length"]),
            digest=entry["warc_digest"],
            timestamp=entry.get("warc_timestamp", ""),
        ),
        text_sha256=entry["text_sha256"],
        entities=tuple(
            Entity(
                key=item["key"],
                class_name=item["class"],
                fields=dict(item["fields"]),
                links={prop: tuple(keys) for prop, keys in (item.get("links") or {}).items()},
            )
            for item in entry["entities"]
        ),
        stats=DocumentStats(**entry["stats"]),
        properties_seen=int(entry.get("properties_seen") or 0),
        properties_grounded=int(entry.get("properties_grounded") or 0),
    )


def _document_text(document: PageDocument, cache: Path) -> str:
    path = cache_path(cache, document)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The page text is the site owner's and is never committed, so the "
            f"documents are rebuilt locally: run scripts/build_wdc_schemaorg.py --pages-only"
        )
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != document.text_sha256:
        raise ValueError(
            f"{path} hashes to {digest[:12]} and the corpus records {document.text_sha256[:12]}, "
            f"so the cached text is not the text this corpus was built from"
        )
    return text


def load_tasks(
    cache: Path | None = None,
    *,
    path: Path | None = None,
    entries: dict[str, CatalogueEntry] | None = None,
    catalogue: tuple[str, ...] | None = None,
    split: Split = Split.DEV,
    difficulty: Difficulty = Difficulty.HARD,
    limit: int | None = None,
) -> list[TaskRecord]:
    """The corpus as tasks, reading each document out of the local cache.

    ``cache`` is where the page text was written by the build, defaulting to
    :data:`PAGES_ENV`. It is not in the repository and it is not
    redistributable, which is the whole reason this signature has an argument
    :func:`~oold_llm_bench.corpus.wiki_measurements.load_examples` does not.

    ``limit`` takes the first ``n`` per class in file order, which the build
    fixed. A prefix rather than a sample, so two callers asking for 60 tasks
    look at the same 60.

    ``difficulty`` is :attr:`~oold_llm_bench.tasks.models.Difficulty.HARD`
    because nothing on these pages was written for an extractor.
    """
    corpus = read_corpus(path)
    root = pages_directory(cache)
    offered = tuple(catalogue) if catalogue else corpus_catalogue(corpus)
    chosen = _chosen(corpus, limit)
    described = dict(zip(offered, render_catalogue(offered, entries), strict=True)) if entries else None
    notes = f"corpus=wdc-schemaorg,crawl={corpus.crawl},release={corpus.release},licence=none-granted"

    tasks: list[TaskRecord] = []
    for document in chosen:
        text = _document_text(document, root)
        missing = sorted({entity.class_name for entity in document.entities} - set(offered))
        if missing:
            raise ValueError(f"{document.id} needs {missing}, which the catalogue does not offer")
        tasks.append(
            TaskRecord(
                id=document.id,
                document=text,
                expected=[entity.expected() for entity in document.entities],
                corpus=CorpusRef(
                    name="wdc",
                    source=Source.CRAWL,
                    document_id=document.id,
                    content_hash=document.text_sha256,
                    url=document.url,
                    retrieved_at=document.warc.fetched_at,
                    licence=None,
                ),
                split=split,
                difficulty=difficulty,
                notes=notes,
                catalogue=list(offered),
                catalogue_text=described,
            )
        )
    return tasks


def corpus_catalogue(corpus: Corpus) -> tuple[str, ...]:
    """Every class the corpus answers with, in a fixed order.

    Taken from the corpus rather than declared, so a class that stopped
    appearing stops being offered and a task is never offered a class no
    document in the set uses.
    """
    return tuple(sorted({entity.class_name for document in corpus.documents for entity in document.entities}))


def _chosen(corpus: Corpus, limit: int | None) -> list[PageDocument]:
    if limit is None:
        return list(corpus.documents)
    taken: dict[str, int] = {}
    chosen: list[PageDocument] = []
    for document in corpus.documents:
        if taken.get(document.main_class, 0) < limit:
            taken[document.main_class] = taken.get(document.main_class, 0) + 1
            chosen.append(document)
    return chosen
