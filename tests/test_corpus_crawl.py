"""Reading markup off a page, and fetching pages without being rude.

No test here opens a socket. The fetcher is a parameter, so the retry path,
the robots decisions and the per-host delay are all driven from a stub, and
the extraction tests run against fixture HTML written for the mess they are
about: several JSON-LD blocks with one broken, a ``@graph``, an array at the
top level, a ``@type`` that is a list.
"""

import json
from collections.abc import Mapping
from datetime import datetime, timezone

import pytest

from oold_llm_bench.corpus.crawl import (
    DEFAULT_DELAY,
    META_FILE,
    MINIMUM_DELAY,
    PAGE_FILE,
    TEXT_FILE,
    TRUTH_FILE,
    USER_AGENT,
    Crawler,
    DocumentMeta,
    Response,
    build_document,
    document_id,
    dump_meta,
    load_meta,
    manifest,
    urlopen_fetcher,
    write_document,
)
from oold_llm_bench.corpus.extract_markup import (
    FULL_TEXT_PROPERTIES,
    Syntax,
    declared_licence,
    extract,
    normalise_term,
    to_jsonld,
)
from oold_llm_bench.tasks.models import Source

BOOK_PAGE = """
<!doctype html>
<html>
<head>
  <title>Shop</title>
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "Book",
    "name": "Calder Wynn",
    "isbn": "QT-41822",
    "numberOfPages": 318
  }
  </script>
  <style>body { color: red; }</style>
</head>
<body>
  <h1>Calder Wynn</h1>
  <p>A novel of 318 pages, catalogued under QT-41822.</p>
  <script>var tracker = 1;</script>
</body>
</html>
"""

GRAPH_PAGE = """
<html><body>
<script type="application/ld+json">
{"@context": "https://schema.org", "@graph": [
  {"@type": "Organization", "name": "Nervin Press"},
  {"@type": ["Product", "Book"], "name": "Hesvin Ast", "sku": "BR-20114"}
]}
</script>
<p>Nervin Press publishes Hesvin Ast.</p>
</body></html>
"""

ARRAY_PAGE = """
<html><body>
<script type="application/ld+json">
[{"@type": "Person", "name": "Volric Mardros"}, {"@type": "Place", "name": "Selquel"}]
</script>
</body></html>
"""

MIXED_PAGE = """
<html><body>
<script type="application/ld+json">{"@type": "WebPage", "name": "first"}</script>
<script type="application/ld+json">{"@type": "Event", "name": "broken",</script>
<script type="application/ld+json">{"@type": "Offer", "price": "12.50"}</script>
</body></html>
"""

MICRODATA_PAGE = """
<html><body>
<div itemscope itemtype="https://schema.org/Product">
  <span itemprop="name">Torlin Quelric</span>
  <meta itemprop="sku" content="DV-77310">
  <img itemprop="image" src="/photo.png">
  <time itemprop="releaseDate" datetime="2021-04-09">April 2021</time>
  <div itemprop="offers" itemscope itemtype="https://schema.org/Offer">
    <span itemprop="price">12.50</span>
    <span itemprop="priceCurrency">EUR</span>
  </div>
</div>
</body></html>
"""

LICENSED_PAGE = """
<html><head>
<link rel="license" href="https://creativecommons.org/licenses/by/4.0/">
<meta name="DC.rights" content="All rights reserved">
</head><body><p>Text.</p></body></html>
"""

ROBOTS_ALLOW = "User-agent: *\nDisallow: /private/\n"
ROBOTS_DENY = "User-agent: *\nDisallow: /\n"
ROBOTS_DELAY = "User-agent: *\nCrawl-delay: 20\nDisallow: /private/\n"


class Stub:
    """A fetcher with a canned answer per url, and a record of the calls.

    Keyed on the url rather than sequenced, because the crawler decides for
    itself when to ask for robots.txt and a sequence would encode that
    decision into every test that does not care about it.
    """

    def __init__(self, pages: dict[str, Response]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> Response:
        self.calls.append((url, headers))
        response = self.pages.get(url, Response(status=404, url=url))
        return Response(
            status=response.status,
            body=response.body,
            headers=response.headers,
            url=response.url or url,
        )

    @property
    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]


class Queue:
    """A fetcher that answers one url differently on each attempt."""

    def __init__(self, url: str, responses: list[Response], robots: str = ROBOTS_ALLOW) -> None:
        self.url = url
        self.responses = responses
        self.robots = robots
        self.calls: list[str] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> Response:
        self.calls.append(url)
        if url.endswith("/robots.txt"):
            return Response(status=200, body=self.robots, url=url)
        if self.responses:
            return self.responses.pop(0)
        return Response(status=500, url=url)


class Clock:
    """A clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.time = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.time += seconds


def page(body: str, status: int = 200, url: str = "") -> Response:
    return Response(status=status, body=body, url=url)


def crawler(fetch, clock: Clock, delay: float = DEFAULT_DELAY, max_attempts: int = 3) -> Crawler:
    return Crawler(
        fetch,
        delay=delay,
        max_attempts=max_attempts,
        sleep=clock.sleep,
        clock=clock.now,
    )


class TestReadingJsonLd:
    def test_one_block_yields_its_entity(self):
        result = extract(BOOK_PAGE)
        assert [entity.types for entity in result.entities] == [("Book",)]
        assert result.entities[0].node["isbn"] == "QT-41822"
        assert result.entities[0].syntax is Syntax.JSON_LD

    def test_a_graph_wrapper_is_unpacked(self):
        result = extract(GRAPH_PAGE)
        assert [entity.types for entity in result.entities] == [("Organization",), ("Product", "Book")]

    def test_an_array_at_the_top_level_is_unpacked(self):
        result = extract(ARRAY_PAGE)
        assert [entity.types for entity in result.entities] == [("Person",), ("Place",)]

    def test_a_broken_block_does_not_take_the_others_with_it(self):
        result = extract(MIXED_PAGE)
        assert [entity.types for entity in result.entities] == [("WebPage",), ("Offer",)]
        assert len(result.errors) == 1
        assert result.errors[0].startswith("json-ld block 1")

    def test_a_prefixed_type_reads_as_the_bare_class(self):
        document = '<script type="application/ld+json">{"@type": "http://schema.org/Recipe"}</script>'
        assert extract(document).entities[0].types == ("Recipe",)

    def test_an_html_escaped_block_is_recovered(self):
        document = '<script type="application/ld+json">{&quot;@type&quot;: &quot;Person&quot;}</script>'
        assert extract(document).entities[0].types == ("Person",)

    def test_a_cdata_wrapped_block_is_recovered(self):
        document = '<script type="application/ld+json">//<![CDATA[\n{"@type": "Person"}\n//]]></script>'
        assert extract(document).entities[0].types == ("Person",)

    def test_a_trailing_comma_is_repaired_as_a_last_resort(self):
        document = '<script type="application/ld+json">{"@type": "Person", "name": "Fen Lin",}</script>'
        entity = extract(document).entities[0]
        assert entity.node["name"] == "Fen Lin"

    def test_a_literal_newline_inside_a_string_is_tolerated(self):
        document = '<script type="application/ld+json">{"@type": "Person", "name": "Fen\nLin"}</script>'
        assert extract(document).entities[0].node["name"] == "Fen\nLin"

    def test_a_block_that_is_not_json_ld_is_ignored(self):
        document = '<script type="application/json">{"@type": "Person"}</script>'
        assert extract(document).entities == ()

    def test_an_untyped_node_is_kept_but_not_counted_as_truth(self):
        document = '<script type="application/ld+json">{"name": "Nothing"}</script>'
        result = extract(document)
        assert len(result.entities) == 1
        assert result.typed == ()


class TestReadingMicrodata:
    def test_properties_come_from_text_and_from_attributes(self):
        node = extract(MICRODATA_PAGE).entities[0].node
        assert node["@type"] == "Product"
        assert node["name"] == "Torlin Quelric"
        assert node["sku"] == "DV-77310"
        assert node["releaseDate"] == "2021-04-09"
        assert node["image"] == "/photo.png"

    def test_a_nested_item_stays_attached_to_its_subject(self):
        entities = extract(MICRODATA_PAGE).entities
        assert len(entities) == 1
        assert entities[0].node["offers"] == {"@type": "Offer", "price": "12.50", "priceCurrency": "EUR"}

    def test_a_relative_url_is_resolved_against_the_page(self):
        node = extract(MICRODATA_PAGE, base_url="https://example.org/shop/item").entities[0].node
        assert node["image"] == "https://example.org/photo.png"

    def test_a_repeated_property_becomes_a_list(self):
        document = """
        <div itemscope itemtype="https://schema.org/Recipe">
          <span itemprop="recipeIngredient">salt</span>
          <span itemprop="recipeIngredient">water</span>
        </div>
        """
        assert extract(document).entities[0].node["recipeIngredient"] == ["salt", "water"]

    def test_one_element_can_carry_two_property_names(self):
        document = '<div itemscope itemtype="https://schema.org/Person"><b itemprop="name givenName">Oss</b></div>'
        node = extract(document).entities[0].node
        assert node["name"] == "Oss" and node["givenName"] == "Oss"

    def test_an_unclosed_element_does_not_lose_the_item(self):
        document = """
        <div itemscope itemtype="https://schema.org/Person">
          <p itemprop="name">Rho Mickbec
          <p itemprop="jobTitle">Clerk
        </div>
        """
        node = extract(document).entities[0].node
        assert node["name"].startswith("Rho Mickbec")
        assert "jobTitle" in node

    def test_itemid_becomes_the_node_identifier(self):
        document = '<div itemscope itemid="urn:x:1" itemtype="https://schema.org/Book"><b itemprop="isbn">9</b></div>'
        assert extract(document).entities[0].node["@id"] == "urn:x:1"

    def test_two_itemtypes_become_a_list(self):
        document = '<div itemscope itemtype="https://schema.org/Product https://schema.org/Book">'
        document += '<b itemprop="name">Both</b></div>'
        assert extract(document).entities[0].node["@type"] == ["Product", "Book"]

    def test_a_base_element_resolves_the_urls(self):
        document = (
            '<html><head><base href="https://example.org/shop/"></head><body>'
            '<div itemscope itemtype="https://schema.org/Product"><img itemprop="image" src="photo.png"/></div>'
            "</body></html>"
        )
        assert extract(document).entities[0].node["image"] == "https://example.org/shop/photo.png"

    def test_the_keywords_are_not_part_of_the_properties(self):
        entity = extract(MICRODATA_PAGE).entities[0]
        assert "@type" not in entity.properties
        assert entity.properties["name"] == "Torlin Quelric"

    def test_two_sibling_items_are_two_entities(self):
        document = (
            '<div itemscope itemtype="https://schema.org/Person"><b itemprop="name">A</b></div>'
            '<div itemscope itemtype="https://schema.org/Person"><b itemprop="name">B</b></div>'
        )
        assert len(extract(document).entities) == 2


class TestStrippedText:
    def test_script_and_style_never_reach_the_document(self):
        text = extract(BOOK_PAGE).text
        assert "tracker" not in text
        assert "color: red" not in text
        assert "@type" not in text

    def test_the_visible_prose_survives(self):
        text = extract(BOOK_PAGE).text
        assert "A novel of 318 pages, catalogued under QT-41822." in text

    def test_the_page_title_is_not_part_of_the_document(self):
        assert "Shop" not in extract(BOOK_PAGE).text

    def test_block_elements_keep_their_sentences_apart(self):
        text = extract("<p>One sentence.</p><p>Another one.</p>").text
        assert text == "One sentence.\n\nAnother one."

    def test_a_line_break_is_one_line_and_not_a_paragraph(self):
        assert extract("<p>One line<br>and the next</p>").text == "One line\nand the next"

    def test_entities_are_decoded(self):
        assert extract("<p>caf&eacute; &amp; bar</p>").text == "café & bar"

    def test_inline_markup_leaves_no_seam(self):
        assert extract("<p>a <b>bold</b> claim</p>").text == "a bold claim"


class TestDeclaredLicence:
    def test_a_rel_license_link_is_found(self):
        assert declared_licence(LICENSED_PAGE) == "https://creativecommons.org/licenses/by/4.0/"

    def test_markup_outranks_the_link(self):
        document = (
            '<script type="application/ld+json">{"@type": "Article", "license": "https://example.org/terms"}</script>'
            + LICENSED_PAGE
        )
        assert declared_licence(document) == "https://example.org/terms"

    def test_dublin_core_is_the_last_resort(self):
        assert declared_licence('<meta name="DC.rights" content="All rights reserved">') == "All rights reserved"

    def test_a_page_that_says_nothing_declares_nothing(self):
        assert declared_licence("<p>Text.</p>") is None

    def test_a_rel_license_anchor_counts(self):
        document = '<a rel="license nofollow" href="https://example.org/terms">terms</a>'
        assert declared_licence(document) == "https://example.org/terms"

    def test_the_first_of_several_stated_licences_is_taken(self):
        document = (
            '<script type="application/ld+json">'
            '{"@type": "Article", "license": ["https://example.org/a", "https://example.org/b"]}'
            "</script>"
        )
        assert declared_licence(document) == "https://example.org/a"

    def test_a_licence_node_is_read_through_its_identifier(self):
        document = '<script type="application/ld+json">{"@type": "Article", "license": {"@id": "urn:licence"}}</script>'
        assert declared_licence(document) == "urn:licence"


class TestRedistributableTruth:
    def test_the_graph_carries_every_entity(self):
        truth = to_jsonld(extract(GRAPH_PAGE))
        assert truth["@context"] == "https://schema.org"
        assert [node["name"] for node in truth["@graph"]] == ["Nervin Press", "Hesvin Ast"]

    def test_the_page_body_is_dropped_from_the_truth(self):
        document = (
            '<script type="application/ld+json">'
            '{"@type": "Article", "headline": "Kept", "articleBody": "Someone else wrote this."}'
            "</script>"
        )
        node = to_jsonld(extract(document))["@graph"][0]
        assert node["headline"] == "Kept"
        assert "articleBody" not in node

    def test_a_nested_body_is_dropped_too(self):
        document = (
            '<script type="application/ld+json">'
            '{"@type": "WebPage", "mainEntity": {"@type": "Article", "articleBody": "Body."}}'
            "</script>"
        )
        assert "articleBody" not in to_jsonld(extract(document))["@graph"][0]["mainEntity"]

    def test_the_drop_list_names_the_properties_that_are_the_page(self):
        assert "articleBody" in FULL_TEXT_PROPERTIES

    def test_a_caller_can_keep_everything(self):
        document = '<script type="application/ld+json">{"@type": "Article", "articleBody": "Body."}</script>'
        assert to_jsonld(extract(document), drop_properties=())["@graph"][0]["articleBody"] == "Body."


class TestNormalisingTerms:
    @pytest.mark.parametrize(
        "written",
        ["Book", "schema:Book", "http://schema.org/Book", "https://schema.org/Book", "https://www.schema.org/Book"],
    )
    def test_every_spelling_of_a_class_reads_the_same(self, written: str):
        assert normalise_term(written) == "Book"


class TestRobots:
    def test_a_disallowed_url_is_not_fetched(self):
        clock = Clock()
        stub = Stub({"https://example.org/robots.txt": page(ROBOTS_DENY)})
        result = crawler(stub, clock).fetch("https://example.org/a")
        assert result.robots_allowed is False
        assert result.status is None
        assert stub.urls == ["https://example.org/robots.txt"]

    def test_an_allowed_url_is_fetched(self):
        clock = Clock()
        stub = Stub({
            "https://example.org/robots.txt": page(ROBOTS_ALLOW),
            "https://example.org/a": page(BOOK_PAGE),
        })
        result = crawler(stub, clock).fetch("https://example.org/a")
        assert result.robots_allowed is True
        assert result.status == 200
        assert result.content_hash

    def test_robots_is_read_once_per_host(self):
        clock = Clock()
        stub = Stub({
            "https://example.org/robots.txt": page(ROBOTS_ALLOW),
            "https://example.org/a": page("<p>a</p>"),
            "https://example.org/b": page("<p>b</p>"),
        })
        collector = crawler(stub, clock)
        collector.fetch("https://example.org/a")
        collector.fetch("https://example.org/b")
        assert stub.urls.count("https://example.org/robots.txt") == 1

    def test_a_missing_robots_allows_the_host(self):
        clock = Clock()
        stub = Stub({"https://example.org/a": page("<p>a</p>")})
        assert crawler(stub, clock).allowed("https://example.org/a") is True

    def test_an_unreachable_robots_disallows_the_host(self):
        clock = Clock()
        stub = Stub({"https://example.org/robots.txt": Response(status=503)})
        assert crawler(stub, clock).allowed("https://example.org/a") is False

    def test_a_forbidden_robots_disallows_the_host(self):
        clock = Clock()
        stub = Stub({"https://example.org/robots.txt": Response(status=403)})
        assert crawler(stub, clock).allowed("https://example.org/a") is False

    def test_a_transport_failure_on_robots_disallows_the_host(self):
        def refuse(url: str, headers: Mapping[str, str]) -> Response:
            raise OSError("connection refused")

        assert crawler(refuse, Clock()).allowed("https://example.org/a") is False

    def test_the_request_identifies_the_project(self):
        clock = Clock()
        stub = Stub({"https://example.org/robots.txt": page(ROBOTS_ALLOW)})
        crawler(stub, clock).fetch("https://example.org/a")
        agent = stub.calls[0][1]["User-Agent"]
        assert agent == USER_AGENT
        assert "github.com" in agent
        assert "Mozilla" not in agent


class TestRateLimit:
    def test_two_requests_to_one_host_are_spaced(self):
        clock = Clock()
        stub = Stub({
            "https://example.org/robots.txt": page(ROBOTS_ALLOW),
            "https://example.org/a": page("<p>a</p>"),
            "https://example.org/b": page("<p>b</p>"),
        })
        collector = crawler(stub, clock)
        collector.fetch("https://example.org/a")
        collector.fetch("https://example.org/b")
        assert clock.slept == [DEFAULT_DELAY, DEFAULT_DELAY]

    def test_a_second_host_does_not_wait_on_the_first(self):
        clock = Clock()
        stub = Stub({
            "https://a.example/robots.txt": page(ROBOTS_ALLOW),
            "https://a.example/x": page("<p>x</p>"),
            "https://a.example/y": page("<p>y</p>"),
            "https://b.example/robots.txt": page(ROBOTS_ALLOW),
            "https://b.example/x": page("<p>x</p>"),
        })
        collector = crawler(stub, clock)
        collector.fetch("https://a.example/x")
        collector.fetch("https://a.example/y")
        collector.fetch("https://b.example/x")
        assert clock.slept == [DEFAULT_DELAY] * 3
        assert clock.time == 3 * DEFAULT_DELAY

    def test_a_longer_stated_delay_is_honoured(self):
        clock = Clock()
        stub = Stub({"https://example.org/robots.txt": page(ROBOTS_DELAY)})
        assert crawler(stub, clock).delay_for("https://example.org/a") == 20.0

    def test_a_shorter_stated_delay_does_not_speed_us_up(self):
        clock = Clock()
        stub = Stub({"https://example.org/robots.txt": page("User-agent: *\nCrawl-delay: 0.2\n")})
        assert crawler(stub, clock).delay_for("https://example.org/a") == DEFAULT_DELAY

    def test_the_configured_delay_has_a_floor(self):
        clock = Clock()
        stub = Stub({"https://example.org/robots.txt": page(ROBOTS_ALLOW)})
        assert crawler(stub, clock, delay=0.01).delay_for("https://example.org/a") == MINIMUM_DELAY


class TestBackoff:
    def test_a_429_is_waited_out_and_retried(self):
        clock = Clock()
        queue = Queue(
            "https://example.org/a",
            [Response(status=429), page(BOOK_PAGE)],
        )
        result = crawler(queue, clock).fetch("https://example.org/a")
        assert result.status == 200
        assert result.attempts == 2

    def test_a_stated_retry_after_decides_the_wait(self):
        clock = Clock()
        queue = Queue(
            "https://example.org/a",
            [Response(status=429, headers={"Retry-After": "42"}), page(BOOK_PAGE)],
        )
        crawler(queue, clock).fetch("https://example.org/a")
        assert 42.0 in clock.slept

    def test_a_server_error_is_retried_up_to_the_limit(self):
        clock = Clock()
        queue = Queue("https://example.org/a", [Response(status=500), Response(status=502), Response(status=503)])
        result = crawler(queue, clock, max_attempts=3).fetch("https://example.org/a")
        assert result.attempts == 3
        assert result.status == 503
        assert result.body is None

    def test_a_retry_after_date_falls_back_to_the_doubling_wait(self):
        clock = Clock()
        queue = Queue(
            "https://example.org/a",
            [Response(status=503, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), page(BOOK_PAGE)],
        )
        crawler(queue, clock).fetch("https://example.org/a")
        assert clock.slept == [DEFAULT_DELAY, 2 * DEFAULT_DELAY]

    def test_a_404_is_not_retried(self):
        clock = Clock()
        queue = Queue("https://example.org/a", [Response(status=404)])
        result = crawler(queue, clock).fetch("https://example.org/a")
        assert result.attempts == 1
        assert result.status == 404

    def test_a_redirect_off_the_host_is_checked_against_the_new_robots(self):
        clock = Clock()
        stub = Stub({
            "https://a.example/robots.txt": page(ROBOTS_ALLOW),
            "https://a.example/x": page(BOOK_PAGE, url="https://b.example/y"),
            "https://b.example/robots.txt": page(ROBOTS_DENY),
        })
        result = crawler(stub, clock).fetch("https://a.example/x")
        assert result.robots_allowed is False
        assert result.final_url == "https://b.example/y"


class TestNetworkFetcher:
    def test_a_url_that_is_not_http_is_refused_before_anything_opens(self):
        with pytest.raises(ValueError, match="non-http"):
            urlopen_fetcher()("file:///etc/passwd", {})


class TestFetchRecord:
    def test_every_fetch_records_what_it_did(self):
        clock = Clock()
        stub = Stub({
            "https://example.org/robots.txt": page(ROBOTS_ALLOW),
            "https://example.org/a": page(LICENSED_PAGE),
        })
        result = crawler(stub, clock).fetch("https://example.org/a")
        assert result.url == "https://example.org/a"
        assert result.fetched_at.tzinfo is timezone.utc
        assert result.robots_allowed is True
        assert result.status == 200
        assert len(result.content_hash) == 64
        assert result.licence == "https://creativecommons.org/licenses/by/4.0/"

    def test_the_hash_covers_what_was_stored(self):
        import hashlib

        clock = Clock()
        stub = Stub({
            "https://example.org/robots.txt": page(ROBOTS_ALLOW),
            "https://example.org/a": page(BOOK_PAGE),
        })
        result = crawler(stub, clock).fetch("https://example.org/a")
        assert result.content_hash == hashlib.sha256(BOOK_PAGE.encode("utf-8")).hexdigest()


class TestDocumentLayout:
    def fetched(self, body: str = BOOK_PAGE, url: str = "https://example.org/a"):
        clock = Clock()
        stub = Stub({"https://example.org/robots.txt": page(ROBOTS_ALLOW), url: page(body)})
        return crawler(stub, clock).fetch(url)

    def test_a_page_with_markup_becomes_four_files(self, tmp_path):
        document = build_document(self.fetched())
        assert document is not None
        directory = write_document(tmp_path, document)
        assert sorted(path.name for path in directory.iterdir()) == sorted([
            PAGE_FILE,
            TEXT_FILE,
            TRUTH_FILE,
            META_FILE,
        ])

    def test_the_input_is_the_page_without_its_markup(self, tmp_path):
        document = build_document(self.fetched())
        assert document is not None
        directory = write_document(tmp_path, document)
        text = (directory / TEXT_FILE).read_text(encoding="utf-8")
        assert "318 pages" in text
        assert "application/ld+json" not in text

    def test_the_truth_is_the_markup(self, tmp_path):
        document = build_document(self.fetched())
        assert document is not None
        directory = write_document(tmp_path, document)
        truth = json.loads((directory / TRUTH_FILE).read_text(encoding="utf-8"))
        assert truth["@graph"][0]["isbn"] == "QT-41822"

    def test_a_page_without_markup_is_not_stored(self):
        assert build_document(self.fetched(body="<p>Nothing annotated here.</p>")) is None

    def test_a_refused_fetch_is_not_stored(self):
        clock = Clock()
        stub = Stub({"https://example.org/robots.txt": page(ROBOTS_DENY)})
        assert build_document(crawler(stub, clock).fetch("https://example.org/a")) is None

    def test_the_directory_name_is_stable_and_placeable(self):
        first = document_id("https://www.example.org/shop/item?id=1")
        assert first == document_id("https://www.example.org/shop/item?id=1")
        assert first.startswith("example-org-")
        assert document_id("https://example.org/other") != first

    def test_collect_yields_only_the_pages_that_carried_truth(self):
        clock = Clock()
        stub = Stub({
            "https://example.org/robots.txt": page(ROBOTS_ALLOW),
            "https://example.org/a": page(BOOK_PAGE),
            "https://example.org/b": page("<p>nothing</p>"),
            "https://example.org/c": page(GRAPH_PAGE),
        })
        collected = list(
            crawler(stub, clock).collect([
                "https://example.org/a",
                "https://example.org/b",
                "https://example.org/c",
            ])
        )
        assert [document.meta.url for document in collected] == [
            "https://example.org/a",
            "https://example.org/c",
        ]


class TestMeta:
    def sample(self) -> DocumentMeta:
        return DocumentMeta(
            document_id="example-org-0123456789",
            url="https://example.org/a: b",
            content_hash="a" * 64,
            text_hash="b" * 64,
            fetched_at=datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc),
            robots_allowed=True,
            status=200,
            entity_count=2,
            licence="https://creativecommons.org/licenses/by/4.0/",
        )

    def test_a_meta_file_round_trips(self):
        assert load_meta(dump_meta(self.sample())) == self.sample()

    def test_a_missing_field_is_refused(self):
        with pytest.raises(ValueError, match="missing"):
            load_meta('url: "https://example.org/a"\n')

    def test_a_value_that_is_not_a_scalar_is_refused(self):
        with pytest.raises(ValueError, match="not a JSON scalar"):
            load_meta("url: https://example.org/a\n")

    def test_a_comment_or_a_blank_line_is_ignored(self):
        text = "# written by oold-llm-bench\n\n" + dump_meta(self.sample())
        assert load_meta(text) == self.sample()

    def test_the_source_is_recorded_as_a_crawl(self):
        assert load_meta(dump_meta(self.sample())).source is Source.CRAWL

    def test_the_file_is_yaml(self):
        yaml = pytest.importorskip("yaml")
        loaded = yaml.safe_load(dump_meta(self.sample()))
        assert loaded["url"] == "https://example.org/a: b"
        assert loaded["status"] == 200
        assert loaded["robots_allowed"] is True


class TestManifest:
    def corpus(self, tmp_path):
        clock = Clock()
        stub = Stub({
            "https://example.org/robots.txt": page(ROBOTS_ALLOW),
            "https://example.org/a": page(BOOK_PAGE),
        })
        for document in crawler(stub, clock).collect(["https://example.org/a"]):
            write_document(tmp_path, document)
        return manifest(tmp_path)

    def test_the_manifest_points_at_the_page_instead_of_carrying_it(self, tmp_path):
        published = json.dumps(self.corpus(tmp_path))
        assert "A novel of 318 pages" not in published
        assert "<html>" not in published
        assert "https://example.org/a" in published

    def test_every_entry_carries_what_a_refetch_is_checked_against(self, tmp_path):
        entry = self.corpus(tmp_path)["documents"][0]
        assert entry["url"] == "https://example.org/a"
        assert len(entry["content_hash"]) == 64
        assert len(entry["text_hash"]) == 64
        assert entry["fetched_at"].startswith("2")
        assert entry["robots_allowed"] is True

    def test_every_entry_carries_our_extraction(self, tmp_path):
        entry = self.corpus(tmp_path)["documents"][0]
        assert entry["truth"]["@graph"][0]["@type"] == "Book"

    def test_a_directory_that_is_not_a_document_is_skipped(self, tmp_path):
        (tmp_path / "notes").mkdir()
        assert len(self.corpus(tmp_path)["documents"]) == 1

    def test_the_manifest_says_the_text_is_not_included(self, tmp_path):
        assert "not redistributed" in self.corpus(tmp_path)["note"]
