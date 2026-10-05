"""Pages out of the Common Crawl archive, as tasks.

The failures to test for are the ones neither a generated corpus nor a
published dataset can have. A url that resolves to the wrong archive record
silently, because the index key is canonicalised one way and the lookup
another. A page that decodes under the wrong charset, which looks like a page
that does not state what it annotates. Ground truth the document does not
contain, which has no answer that scores. And a committed file that holds page
text, which is the one thing the licences do not allow.

Nothing here reaches the network. The archive is a fixture: a gzip member with
a WARC header, an HTTP header and a body is a few lines to build, and building
it is what makes the retrieval path testable at all.
"""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from oold_llm_bench.corpus.extract_markup import extract
from oold_llm_bench.corpus.wdc import (
    CRAWL_ID,
    LABELLED_PROPERTIES,
    MAX_NESTED_PER_PROPERTY,
    MAX_VALUE_CHARS,
    MIN_GROUNDED_PROPERTIES,
    STRUCTURAL_PROPERTIES,
    ClusterIndex,
    MissingPages,
    entities_of,
    grounded,
    leaks_slot,
    load_tasks,
    measure_document,
    page_html,
    read_corpus,
    resolve,
    shingle,
    squeeze,
    subset_urls,
    surt,
    usable_document,
    warc_member,
)
from oold_llm_bench.grading import Dimension, TripleSet, make_triples
from oold_llm_bench.grading.score import score_task
from oold_llm_bench.results.record import Environment, ModelSpec
from oold_llm_bench.runner import Cell, Condition, ExperimentConfig
from oold_llm_bench.runner.preflight import CONTROL_CEILING, SPELLING_CEILING, control_scores, preflight
from oold_llm_bench.tasks.models import Source, Split

RECIPE_TEXT = """Grandmother's Apple Cake

This is the cake my grandmother made every autumn, and the one I still make
when the first cooking apples come in from the garden up the lane.
It takes about twenty minutes to put together and the house smells of
cinnamon for the rest of the afternoon, which is most of the reason to make it.
The batter is wetter than you expect and that is deliberate, because the apples
give up a great deal of water while the cake is in the oven.
Use a tin with a loose base if you have one, since the cake is fragile while it
is still warm and will not survive being turned out of a solid tin.
I have made this with pears instead of apples and it works, though the result
is sweeter and needs rather less sugar than the recipe asks for.
Leftovers keep for three days in a tin and are very good indeed toasted.
Preheat the oven to 180C and butter a twenty centimetre tin thoroughly.
Peel and slice the apples, then toss them with the lemon juice and set aside.
Beat the butter and the sugar together until the mixture is pale and fluffy.
Fold in the flour, then the apples, and turn the whole lot into the tin.
Bake for 45 mins, until a skewer pushed into the middle comes out clean.

Rated 4.6 out of 5 by 118 people.
"""
"""A document that passes every filter, written for the tests.

Real text could not be used: the page is the site owner's, and a test fixture
that reproduced one would be the one thing the licence posture forbids.
"""

SOUP_TEXT = """Leek and Potato Soup for a Cold Evening

My father made this on the evenings when the weather turned and there was
nothing else in the house, and it is still what I cook when I cannot think.
The whole business takes an hour and a quarter from start to finish, most of
which is the pot sitting on the stove doing the work without any help from me.
Everything depends on softening the leeks slowly, so give them a quarter of an
hour on their own before anything else goes anywhere near the pan.
If you rush that step you get a soup that tastes of boiled water and leeks, and
there is no rescuing it afterwards with cream or with pepper or with anything.
Floury potatoes break down and thicken the soup, waxy ones stay in pieces, and
which of the two you want is a matter of taste rather than of correctness.
I blend about half of it and leave the rest as it is, which gives a soup that
is thick without being a puree and still has something to chew on.
It freezes well and is better on the second day, as most soups of this kind are.
Serve it with a great deal of bread and do not bother with a starter.
"""
"""A second document that shares neither a value nor a shape with the first.

Both apart on purpose. Two near-identical fixtures let the wrong-document
control answer one with the other and score 0.83, which reads as the grader
rewarding shape when it is the fixture repeating itself. The shape matters as
much as the values: when every document is one class pointing at one other,
the alignment puts a wrong document's entities on the right ones and the edge
between them is correct without anything having been read. That is the defect
``TestTheControlsOnALinkedGrid`` in ``test_preflight`` demonstrates, and this
one carries no edge so the pair cannot share it.
"""

SPEC_TEXT = """Technical Specification

Weight:
7.1oz
Country of Origin:
Vietnam
Material:
Nylon
Closure:
Zip
Warranty:
2 years
Cookie policy and privacy policy apply.
All rights reserved.
"""

RECIPE_NODE = {
    "@type": "Recipe",
    "name": "Grandmother's Apple Cake",
    "url": "https://example.invalid/cake",
    "image": "https://example.invalid/cake.jpg",
    "cookTime": "PT45M",
    "calories": "320 calories",
    "recipeInstructions": [
        {"@type": "HowToStep", "text": "Preheat the oven to 180C"},
        {"@type": "HowToStep", "text": "Peel and slice the apples"},
    ],
    "aggregateRating": {"@type": "AggregateRating", "ratingValue": "4.6", "ratingCount": "118"},
}


def page(html: str) -> str:
    return f"<html><body>{html}</body></html>"


def warc_record(body: str, header: str = "Content-Type: text/html; charset=utf-8") -> bytes:
    """One archived response, with the two headers a WARC record carries."""
    return (
        b"WARC/1.0\r\nWARC-Type: response\r\n\r\n"
        + f"HTTP/1.1 200 OK\r\n{header}\r\n\r\n".encode()
        + body.encode("utf-8")
    )


def cdx_block(*lines: str) -> bytes:
    return gzip.compress("\n".join(lines).encode("utf-8"))


def cdx_line(key: str, stamp: str, **fields) -> str:
    record = {
        "url": "https://example.invalid/cake",
        "status": "200",
        "digest": "AAAA",
        "length": "100",
        "offset": "500",
        "filename": "crawl-data/CC-MAIN-2024-42/segments/1/warc/one.warc.gz",
        "languages": "eng",
    }
    record.update(fields)
    return f"{key} {stamp} {json.dumps(record)}"


def fetcher(blocks: dict[str, bytes]):
    """A ranged GET that answers from a mapping instead of from the network."""

    def fetch(url: str, offset: int, length: int) -> bytes:
        for name, payload in blocks.items():
            if name in url:
                return payload
        raise KeyError(url)

    return fetch


def index() -> ClusterIndex:
    return ClusterIndex([("com,a)/", "cdx-00000.gz", 0, 10), ("invalid,example)/", "cdx-00001.gz", 0, 10)], CRAWL_ID)


class TestTheIndexKey:
    """A key canonicalised one way and looked up another finds nothing, and
    finding nothing looks exactly like a page that was never crawled."""

    def test_the_host_labels_are_reversed_so_one_site_is_one_range(self):
        assert surt("https://example.invalid/cake").startswith("invalid,example)")

    def test_www_is_dropped_because_the_index_drops_it(self):
        assert surt("https://www.example.invalid/cake") == surt("https://example.invalid/cake")

    def test_the_trailing_slash_comes_off_everything_but_the_root(self):
        assert surt("https://example.invalid/cake/") == "invalid,example)/cake"
        assert surt("https://example.invalid/") == "invalid,example)/"

    def test_the_path_is_lowercased(self):
        assert surt("https://example.invalid/Cake") == "invalid,example)/cake"

    def test_a_query_is_kept_because_it_names_a_different_page(self):
        assert surt("https://example.invalid/p?id=2") == "invalid,example)/p?id=2"

    def test_the_block_holding_a_key_is_the_last_one_at_or_below_it(self):
        assert index().block_for("invalid,example)/cake")[0] == "cdx-00001.gz"
        assert index().block_for("com,a)/z")[0] == "cdx-00000.gz"

    def test_a_key_below_every_block_still_lands_somewhere(self):
        assert index().block_for("aaa,aaa)/")[0] == "cdx-00000.gz"

    def test_a_file_with_no_blocks_is_refused(self, tmp_path):
        path = tmp_path / "cluster.idx"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no index blocks"):
            ClusterIndex.read(path)


class TestResolvingAUrlToTheArchive:
    """The record a url resolves to is the whole of the provenance, so a wrong
    one is a corpus whose coordinates point at someone else's page."""

    def blocks(self, *lines: str) -> dict[str, bytes]:
        return {"cdx-00001.gz": cdx_block(*lines)}

    def test_a_captured_url_yields_its_warc_coordinates(self):
        lines = self.blocks(cdx_line("invalid,example)/cake", "20241012210331"))
        ref = resolve("https://example.invalid/cake", index(), fetcher(lines))
        assert ref is not None
        assert (ref.filename, ref.offset, ref.length, ref.digest) == (
            "crawl-data/CC-MAIN-2024-42/segments/1/warc/one.warc.gz",
            500,
            100,
            "AAAA",
        )

    def test_the_crawl_the_index_names_travels_with_the_reference(self):
        lines = self.blocks(cdx_line("invalid,example)/cake", "20241012210331"))
        ref = resolve("https://example.invalid/cake", index(), fetcher(lines))
        assert ref is not None
        assert ref.crawl == CRAWL_ID

    def test_the_capture_time_becomes_the_retrieval_time(self):
        lines = self.blocks(cdx_line("invalid,example)/cake", "20241012210331"))
        ref = resolve("https://example.invalid/cake", index(), fetcher(lines))
        assert ref is not None
        assert ref.fetched_at.date().isoformat() == "2024-10-12"

    def test_a_redirect_is_not_a_capture(self):
        lines = self.blocks(cdx_line("invalid,example)/cake", "20241012210331", status="301"))
        assert resolve("https://example.invalid/cake", index(), fetcher(lines)) is None

    def test_a_robots_record_is_not_a_page(self):
        lines = self.blocks(cdx_line("invalid,example)/cake", "2024", filename="crawl-data/x/robotstxt/a.gz"))
        assert resolve("https://example.invalid/cake", index(), fetcher(lines)) is None

    def test_a_url_the_block_does_not_hold_resolves_to_nothing(self):
        lines = self.blocks(cdx_line("invalid,example)/other", "20241012210331"))
        assert resolve("https://example.invalid/cake", index(), fetcher(lines)) is None

    def test_the_first_capture_wins_so_a_tie_is_not_decided_by_the_file_order(self):
        lines = self.blocks(
            cdx_line("invalid,example)/cake", "20241012210331", offset="1"),
            cdx_line("invalid,example)/cake", "20241012215959", offset="2"),
        )
        ref = resolve("https://example.invalid/cake", index(), fetcher(lines))
        assert ref is not None
        assert ref.offset == 1


class TestReadingOneArchivedPage:
    """A page decoded under the wrong charset looks like a page that does not
    state what it annotates, and the grounding rate would carry the error."""

    def ref(self, **fields):
        lines = {"cdx-00001.gz": cdx_block(cdx_line("invalid,example)/cake", "20241012210331", **fields))}
        return resolve("https://example.invalid/cake", index(), fetcher(lines))

    def test_the_body_comes_out_without_either_header(self):
        assert page_html(warc_record("<html>hi</html>")) == "<html>hi</html>"

    def test_one_ranged_get_is_one_independently_compressed_member(self):
        record = warc_record("<html>hi</html>")
        member = gzip.compress(record)
        assert warc_member(self.ref(), fetcher({"one.warc.gz": member})) == record

    def test_the_http_header_decides_the_encoding(self):
        body = "<html>café</html>".encode("cp1252")
        record = b"WARC/1.0\r\n\r\nHTTP/1.1 200 OK\r\nContent-Type: text/html; charset=windows-1252\r\n\r\n" + body
        assert "café" in page_html(record)

    def test_the_page_declares_its_own_encoding_when_the_header_does_not(self):
        body = '<html><meta charset="windows-1252">café</html>'.encode("cp1252")
        record = b"WARC/1.0\r\n\r\nHTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n" + body
        assert "café" in page_html(record)

    def test_a_page_that_declares_utf8_and_is_not_falls_back_rather_than_mangling(self):
        body = "<html>café</html>".encode("cp1252")
        record = b"WARC/1.0\r\n\r\nHTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n" + body
        assert page_html(record) == "<html>café</html>"


class TestReadingTheSubsetWithoutLettingOneDomainFillIt:
    """A 1000-quad sample of one subset was 28 pages from one spam domain. The
    subsets are clustered by domain, so a cap while reading is the only cap
    that works on a file this size."""

    def quads(self, tmp_path, *urls) -> Path:
        lines = [f'_:b <http://schema.org/name> "x" <{url}>   .' for url in urls]
        path = tmp_path / "part.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_the_graph_name_is_the_page_the_quad_came_from(self, tmp_path):
        path = self.quads(tmp_path, "https://a.invalid/one")
        assert list(subset_urls(path)) == [("https://a.invalid/one", "a.invalid")]

    def test_one_host_contributes_at_most_the_cap(self, tmp_path):
        path = self.quads(tmp_path, "https://a.invalid/one", "https://a.invalid/two", "https://b.invalid/three")
        assert [host for _, host in subset_urls(path, per_host=1)] == ["a.invalid", "b.invalid"]

    def test_the_cap_can_be_raised(self, tmp_path):
        path = self.quads(tmp_path, "https://a.invalid/one", "https://a.invalid/two")
        assert len(list(subset_urls(path, per_host=2))) == 2

    def test_a_gzip_stream_that_stops_partway_yields_what_it_held(self, tmp_path):
        full = gzip.compress(b'_:b <http://schema.org/name> "x" <https://a.invalid/one>   .\n' * 200)
        path = tmp_path / "part.gz"
        path.write_bytes(full[: len(full) // 2])
        assert list(subset_urls(path, per_host=5)) == []


class TestWhitespaceIsTheMeasurement:
    """Naive line-joined matching and whitespace-insensitive matching differ by
    seventeen points on the same pages, because one ingredient reaches the page
    as seven elements."""

    SPLIT = "2\npounds\n(\n907\ng\n)\nground beef"

    def test_a_value_split_across_elements_is_still_stated_by_the_page(self):
        assert grounded("2 pounds (907 g) ground beef", self.SPLIT)

    def test_the_naive_reading_is_the_one_that_fails(self):
        assert "2 pounds (907 g) ground beef" not in self.SPLIT

    def test_a_smart_quote_in_the_rendering_is_the_apostrophe_in_the_markup(self):
        rendered = "Alyssa" + chr(0x2019) + "s Bites are here"
        assert grounded("Alyssa's Bites", rendered)

    def test_a_hyphen_and_an_underscore_are_the_same_gap(self):
        assert grounded("FULL_TIME", "This is a Full-time position")

    def test_a_soft_hyphen_in_the_rendering_is_not_a_difference(self):
        assert grounded("breakfast", "break­fast")

    def test_squeeze_keeps_the_words_and_drops_only_the_spacing(self):
        assert squeeze(" Two  Words\n") == "twowords"

    def test_a_value_the_page_does_not_hold_is_not_grounded(self):
        assert not grounded("chicken thighs", self.SPLIT)

    def test_an_empty_value_is_never_grounded(self):
        assert not grounded("", "anything")
        assert not grounded(None, "anything")


class TestDurationsAndNumbersAreNormalisedBeforeTheyAreJudged:
    """The markup stores PT25M and $129.0 and no page prints either, so a
    literal comparison reports the two commonest properties as ungrounded."""

    def test_a_duration_is_found_through_the_words_a_page_uses(self):
        assert grounded("PT25M", "Ready in 25 mins")
        assert grounded("PT25M", "Ready in 25 minutes")

    def test_a_range_still_states_its_lower_bound(self):
        assert grounded("PT25M", "Bake for 25 to 30 mins")

    def test_a_duration_written_in_another_unit_is_the_same_duration(self):
        assert grounded("PT1H", "Takes 60 minutes")
        assert grounded("PT90M", "Takes 1 hour 30 minutes")

    def test_a_symbol_against_a_digit_is_reached_without_a_word_boundary(self):
        assert grounded("PT90M", "Total 1h 30m")

    def test_a_page_stating_only_part_of_a_duration_has_not_stated_it(self):
        assert not grounded("PT1H30M", "Takes 1 hour")

    def test_a_number_is_found_under_the_separators_a_locale_writes(self):
        assert grounded("129.0", "Only $129 today")
        assert grounded("1200", "1,200 reviews")
        assert grounded("1200", "1.200 Bewertungen")

    def test_a_different_number_is_not_the_same_number(self):
        assert not grounded("129.0", "Only $130 today")

    def test_a_number_inside_a_longer_number_is_not_that_number(self):
        assert not grounded("129", "Order number 1129 shipped")
        assert not grounded("4.6", "Scored 4.65 overall")

    def test_a_number_inside_a_longer_one_is_not_a_statement_of_it(self):
        assert not grounded("PT25M", "Bake for 125 minutes")


class TestSlotLeakage:
    """Every calories value in the sample is printed beside the word. A corpus
    that kept them would measure a label matcher and call it extraction."""

    def test_a_labelled_value_is_caught(self):
        assert leaks_slot("calories", "320", "Nutrition Calories: 320 kcal")

    def test_a_value_the_page_does_not_label_is_not(self):
        assert not leaks_slot("name", "Apple Cake", "A slice of Apple Cake on a plate")

    def test_only_the_head_noun_is_looked_for(self):
        """``recipe`` is printed on every recipe page, so matching on it would
        condemn every property whose name starts with it."""
        assert not leaks_slot("recipeIngredient", "two eggs", "Recipe by Jane. We used two eggs")
        assert leaks_slot("recipeIngredient", "two eggs", "Ingredients: two eggs")

    def test_a_label_far_from_its_value_is_not_a_label(self):
        text = "Calories are a thing. " + "x" * 200 + " 320"
        assert not leaks_slot("calories", "320", text)

    def test_a_value_the_page_does_not_hold_cannot_leak(self):
        assert not leaks_slot("calories", "320", "no numbers here")

    def test_the_nutrition_panel_is_dropped_by_name_and_not_one_value_at_a_time(self):
        assert {"calories", "sodiumContent", "recipeYield", "servingSize"} <= LABELLED_PROPERTIES


class TestTheDocumentFilter:
    """The filter that actually binds. Grounding passes on a product page,
    because the specification table does state its values; what disqualifies it
    is that nothing on it is a sentence."""

    def test_a_page_of_prose_passes(self):
        assert usable_document(measure_document(RECIPE_TEXT)) is None

    def test_a_specification_table_is_refused(self):
        assert "prose" in (usable_document(measure_document(SPEC_TEXT)) or "")

    def test_a_short_line_ending_in_a_period_is_not_a_sentence(self):
        assert measure_document("Home.\nAbout.\nContact.\n").sentence_lines == 0

    def test_label_and_value_lines_are_counted(self):
        assert measure_document(SPEC_TEXT).spec_lines >= 5

    def test_boilerplate_is_counted_from_a_stated_lexicon(self):
        assert measure_document(SPEC_TEXT).boilerplate_lines >= 2

    def test_a_page_of_furniture_is_refused_for_being_furniture(self):
        text = "\n".join([line for line in RECIPE_TEXT.splitlines() if line] + ["Cookie policy."] * 12)
        assert "cookie" in (usable_document(measure_document(text)) or "")

    def test_a_page_with_no_lines_at_all_is_refused(self):
        assert usable_document(measure_document("")) is not None


class TestWhatOnePageAsserts:
    """One main entity per document, the entities it nests, and nothing the
    page did not say."""

    def entities(self, node, text=RECIPE_TEXT):
        html = page(f'<script type="application/ld+json">{json.dumps(node)}</script>')
        return entities_of(extract(html), text)[0]

    def test_the_main_entity_comes_first_and_carries_the_key_the_links_use(self):
        found = self.entities(RECIPE_NODE)
        assert found[0].key == "e1"
        assert found[0].class_name == "Recipe"

    def test_a_nested_entity_is_reached_by_a_link_and_not_inlined(self):
        found = self.entities(RECIPE_NODE)
        assert found[0].links["aggregateRating"] == ("e2",)
        assert found[1].class_name == "AggregateRating"
        assert found[1].fields == {"ratingValue": "4.6", "ratingCount": "118"}

    def test_the_link_survives_into_the_expected_instance_as_a_reference(self):
        found = self.entities(RECIPE_NODE)
        assert found[0].expected().fields["aggregateRating"].key == "e2"

    def test_a_wrapper_object_contributes_the_string_it_carries(self):
        found = self.entities(RECIPE_NODE)
        assert found[0].fields["recipeInstructions"] == [
            "Preheat the oven to 180C",
            "Peel and slice the apples",
        ]

    def test_a_structural_property_never_becomes_an_answer(self):
        found = self.entities(RECIPE_NODE)
        assert not {"url", "image"} & set(found[0].fields)

    def test_the_nutrition_panel_never_becomes_an_answer(self):
        found = self.entities(RECIPE_NODE)
        assert "calories" not in found[0].fields

    def test_a_property_the_page_only_half_states_is_dropped_whole(self):
        """Keeping the grounded nine of twelve ingredients would mark the
        complete answer down on precision for the three that were removed."""
        node = dict(RECIPE_NODE, recipeIngredient=["the lemon juice", "a thing the page never mentions"])
        assert "recipeIngredient" not in self.entities(node)[0].fields

    def test_a_property_the_page_states_whole_is_kept_whole(self):
        node = dict(RECIPE_NODE, recipeIngredient=["the lemon juice", "the apples"])
        assert self.entities(node)[0].fields["recipeIngredient"] == ["the lemon juice", "the apples"]

    def test_a_value_longer_than_an_answer_is_not_an_answer(self):
        node = dict(RECIPE_NODE, description="x" * (MAX_VALUE_CHARS + 1))
        assert "description" not in self.entities(node)[0].fields

    def test_html_pasted_into_a_json_string_is_read_as_the_text_it_renders(self):
        node = dict(RECIPE_NODE, description="<p>Peel and slice the <b>apples</b></p>")
        assert self.entities(node)[0].fields["description"] == "Peel and slice the apples"

    def test_a_page_annotating_two_recipes_has_no_single_answer(self):
        html = page(
            f'<script type="application/ld+json">{json.dumps(RECIPE_NODE)}</script>'
            f'<script type="application/ld+json">{json.dumps(RECIPE_NODE)}</script>'
        )
        assert entities_of(extract(html), RECIPE_TEXT)[0] == ()

    def test_a_listing_page_is_not_a_description_of_one_thing(self):
        node = dict(RECIPE_NODE, itemListElement=[{"@type": "ListItem"}])
        assert self.entities(node) == ()

    def test_a_property_pointing_at_many_near_identical_children_is_a_list(self):
        node = dict(
            RECIPE_NODE,
            review=[{"@type": "Review", "name": "the apples"} for _ in range(MAX_NESTED_PER_PROPERTY + 1)],
        )
        assert "review" not in self.entities(node)[0].links

    def test_a_child_that_states_nothing_takes_its_siblings_with_it(self):
        node = dict(
            RECIPE_NODE,
            review=[{"@type": "Review", "name": "the apples"}, {"@type": "Review", "name": "never said"}],
        )
        assert "review" not in self.entities(node)[0].links

    def test_an_entity_two_levels_down_is_still_part_of_this_document(self):
        node = dict(
            RECIPE_NODE,
            review=[
                {"@type": "Review", "name": "the apples", "reviewRating": {"@type": "Rating", "ratingValue": "4.6"}}
            ],
        )
        found = self.entities(node)
        assert [entity.class_name for entity in found] == ["Recipe", "AggregateRating", "Review", "Rating"]

    def test_a_page_with_no_markup_at_all_asserts_nothing(self):
        assert entities_of(extract(page("<p>words</p>")), RECIPE_TEXT)[0] == ()

    def test_the_grounding_counts_cover_what_was_read_and_not_what_was_kept(self):
        node = dict(RECIPE_NODE, recipeIngredient=["the apples", "a thing the page never mentions"])
        html = page(f'<script type="application/ld+json">{json.dumps(node)}</script>')
        counts = entities_of(extract(html), RECIPE_TEXT)[1]
        assert counts.by_property["recipeIngredient"] == [2, 1]


class TestRepeatedCopy:
    """Variant pages repeat marketing copy verbatim, and one page per piece of
    copy is what one document per thing means."""

    def test_two_pages_with_the_same_opening_share_a_fingerprint(self):
        assert shingle("The best cake you will ever bake, honestly now") == shingle(
            "The best cake you will ever bake, honestly later"
        )

    def test_a_different_opening_is_a_different_fingerprint(self):
        assert shingle("One two three four five six seven eight") != shingle(
            "Nine ten eleven twelve thirteen fourteen fifteen sixteen"
        )

    def test_text_shorter_than_a_shingle_has_no_fingerprint(self):
        assert shingle("two words") == ""


def corpus_file(tmp_path, documents, **changes) -> Path:
    payload = {
        "schema_version": "1",
        "name": "WDC schema.org via Common Crawl",
        "crawl": CRAWL_ID,
        "release": "2024-12",
        "licence": {"page_text": {"statement": "Each page is the copyright of whoever published it."}},
        "built_at": "2026-10-03",
        "urls_in": len(documents),
        "resolved": len(documents),
        "excluded": {},
        "per_class": {"Recipe": len(documents)},
        "cap": 150,
        "grounding": {"values_seen": 10, "values_grounded": 8, "rate": 0.8, "per_class": {}, "per_property": {}},
        "documents": documents,
    }
    payload.update(changes)
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def described(index_: int, text: str, entities=None) -> dict:
    return {
        "id": f"wdc-recipe-example-{index_}",
        "url": f"https://example-{index_}.invalid/dish",
        "host": f"example-{index_}.invalid",
        "main_class": "Recipe",
        "warc_filename": "crawl-data/CC-MAIN-2024-42/segments/1/warc/one.warc.gz",
        "warc_offset": 500 + index_,
        "warc_length": 100,
        "warc_digest": "AAAA",
        "warc_timestamp": "20241012210331",
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "properties_seen": 5,
        "properties_grounded": 4,
        "stats": {"lines": 20, "sentence_lines": 12, "spec_lines": 1, "boilerplate_lines": 0, "characters": len(text)},
        "entities": entities
        or [
            {
                "key": "e1",
                "class": "Recipe",
                "fields": {"name": "Grandmother's Apple Cake", "cookTime": "PT45M", "recipeCategory": "cinnamon"},
                "links": {"aggregateRating": ["e2"]},
            },
            {"key": "e2", "class": "AggregateRating", "fields": {"ratingValue": "4.6", "ratingCount": "118"}},
        ],
    }


SOUP_ENTITIES = [
    {
        "key": "e1",
        "class": "Recipe",
        "fields": {
            "name": "Leek and Potato Soup for a Cold Evening",
            "cookTime": "PT75M",
            "recipeCategory": "bread",
        },
    }
]


@pytest.fixture
def built(tmp_path):
    """A two-document corpus with its page cache, which is the pair a run needs."""
    pages = tmp_path / "pages"
    pages.mkdir()
    documents = []
    for position, (text, entities) in enumerate([(RECIPE_TEXT, None), (SOUP_TEXT, SOUP_ENTITIES)]):
        entry = described(position, text, entities)
        documents.append(entry)
        (pages / f"{entry['id']}.txt").write_text(text, encoding="utf-8")
    return corpus_file(tmp_path, documents, urls_in=2, resolved=2, per_class={"Recipe": 2}), pages


class TestNothingIsLostBetweenTheArchiveAndTheTasks:
    """A corpus that has quietly lost half of itself still loads, and a number
    measured on it is not comparable to a number measured before the loss."""

    def test_a_url_that_left_without_a_reason_is_refused(self, tmp_path):
        path = corpus_file(tmp_path, [described(0, RECIPE_TEXT)], urls_in=9, resolved=1, per_class={"Recipe": 1})
        with pytest.raises(ValueError, match="without a declared reason"):
            read_corpus(path)

    def test_per_class_counts_that_do_not_add_up_are_refused(self, tmp_path):
        path = corpus_file(tmp_path, [described(0, RECIPE_TEXT)], urls_in=1, resolved=1, per_class={"Recipe": 2})
        with pytest.raises(ValueError, match="per-class counts sum to"):
            read_corpus(path)

    def test_a_file_holding_more_than_it_resolved_is_refused(self, tmp_path):
        documents = [described(0, RECIPE_TEXT), described(1, RECIPE_TEXT)]
        path = corpus_file(
            tmp_path, documents, urls_in=3, resolved=1, excluded={"a reason": 2}, per_class={"Recipe": 1}
        )
        with pytest.raises(ValueError, match="but only 1 resolved"):
            read_corpus(path)

    def test_every_refusal_is_counted_under_a_reason_that_reads_as_one(self, tmp_path):
        path = corpus_file(
            tmp_path,
            [described(0, RECIPE_TEXT)],
            urls_in=3,
            resolved=1,
            excluded={"the page carries fewer than 10 lines of running prose": 2},
            per_class={"Recipe": 1},
        )
        assert sum(read_corpus(path).excluded.values()) == 2


class TestTheTextIsNotInTheRepository:
    """Web Data Commons grants no data licence and Common Crawl grants access
    only, so the documents are rebuilt and never published."""

    def test_the_tasks_are_built_from_the_locally_held_text(self, built):
        path, pages = built
        tasks = load_tasks(pages, path=path)
        assert tasks[0].document == RECIPE_TEXT

    def test_a_missing_cache_says_how_to_rebuild_it(self, built, tmp_path):
        path, _ = built
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="pages-only"):
            load_tasks(empty, path=path)

    def test_no_cache_at_all_is_told_apart_from_a_broken_installation(self, built, monkeypatch):
        path, _ = built
        monkeypatch.delenv("OOLD_WDC_PAGES", raising=False)
        with pytest.raises(MissingPages, match="OOLD_WDC_PAGES"):
            load_tasks(path=path)

    def test_text_that_is_not_what_the_corpus_was_built_from_is_refused(self, built):
        path, pages = built
        next(pages.glob("*.txt")).write_text("something else entirely", encoding="utf-8")
        with pytest.raises(ValueError, match="not the text this corpus was built from"):
            load_tasks(pages, path=path)

    def test_the_committed_file_carries_no_page_text(self, built):
        path, _ = built
        assert "grandmother made every autumn" not in path.read_text(encoding="utf-8")


class TestProvenance:
    """A third party reproduces the corpus by ranged GET and a digest check, so
    the coordinates are the corpus and have to be complete."""

    def test_every_document_names_the_record_it_came_from(self, built):
        path, _ = built
        for document in read_corpus(path).documents:
            assert document.warc.filename.endswith(".warc.gz")
            assert document.warc.offset > 0
            assert document.warc.length > 0
            assert document.warc.digest

    def test_every_document_names_the_crawl_its_offsets_belong_to(self, built):
        path, _ = built
        assert {d.warc.crawl for d in read_corpus(path).documents} == {CRAWL_ID}

    def test_every_record_is_marked_as_crawled_and_not_synthetic(self, built):
        path, pages = built
        assert {t.corpus.source for t in load_tasks(pages, path=path)} == {Source.CRAWL}

    def test_every_record_names_the_page_and_when_it_was_captured(self, built):
        path, pages = built
        for task in load_tasks(pages, path=path):
            assert (task.corpus.url or "").startswith("https://")
            assert task.corpus.retrieved_at is not None

    def test_the_content_hash_covers_the_document_the_model_is_given(self, built):
        path, pages = built
        for task in load_tasks(pages, path=path):
            assert task.corpus.content_hash == hashlib.sha256(task.document.encode("utf-8")).hexdigest()

    def test_no_licence_is_claimed_because_none_was_granted(self, built):
        path, pages = built
        assert {t.corpus.licence for t in load_tasks(pages, path=path)} == {None}

    def test_the_notes_say_which_crawl_and_which_release(self, built):
        path, pages = built
        notes = load_tasks(pages, path=path)[0].notes or ""
        assert "corpus=wdc-schemaorg" in notes
        assert CRAWL_ID in notes

    def test_the_grounding_rate_is_in_the_record_and_not_in_a_comment(self, built):
        path, _ = built
        corpus = read_corpus(path)
        assert corpus.grounding["rate"] > 0
        assert corpus.documents[0].grounding_rate > 0


class TestTheCatalogue:
    """A task offered a catalogue that does not hold its answer has no answer."""

    def test_every_class_the_corpus_answers_with_is_offered(self, built):
        path, pages = built
        assert set(load_tasks(pages, path=path)[0].catalogue or ()) == {"Recipe", "AggregateRating"}

    def test_a_caller_may_choose_the_catalogue(self, built):
        path, pages = built
        offered = ("Recipe", "AggregateRating", "JobPosting")
        assert load_tasks(pages, path=path, catalogue=offered)[0].catalogue == list(offered)

    def test_a_catalogue_without_an_expected_class_stops_the_load(self, built):
        path, pages = built
        with pytest.raises(ValueError, match="which the catalogue does not offer"):
            load_tasks(pages, path=path, catalogue=("Recipe",))

    def test_bare_identifiers_are_offered_when_no_schemas_are_given(self, built):
        path, pages = built
        assert load_tasks(pages, path=path)[0].catalogue_text is None

    def test_a_caller_may_take_a_fixed_number_per_class(self, built):
        path, pages = built
        assert len(load_tasks(pages, path=path, limit=1)) == 1

    def test_the_split_can_be_chosen(self, built):
        path, pages = built
        assert load_tasks(pages, path=path, split=Split.TEST)[0].split is Split.TEST


def perfect_answer(task) -> TripleSet:
    """What a model that read the page correctly would produce."""
    triples = frozenset()
    classes = {}
    for instance in task.expected:
        triples |= make_triples(instance.key, instance.fields)
        classes[instance.key] = instance.class_path
    return TripleSet(triples=triples, classes=classes, provenance={})


def grid_of(tasks) -> ExperimentConfig:
    return ExperimentConfig(
        name="wdc-schemaorg",
        conditions=[Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=8)],
        models=[ModelSpec(model="m", provider_profile="openai", model_version="1")],
        tasks=tasks,
        runs_per_cell=1,
    )


class TestTheSeamCloses:
    """Corpus to task to score, with no model involved. A corpus that does not
    pass this is not a corpus."""

    def test_a_perfect_answer_scores_one(self, built):
        path, pages = built
        for task in load_tasks(pages, path=path):
            result = score_task(task, perfect_answer(task))
            assert result.primary == pytest.approx(1.0)
            assert result.dimensions[Dimension.CLASS].f1 == pytest.approx(1.0)

    def test_an_empty_answer_scores_zero(self, built):
        path, pages = built
        empty = TripleSet(triples=frozenset(), classes={}, provenance={})
        assert score_task(load_tasks(pages, path=path)[0], empty).primary == 0.0

    def test_every_negative_control_stays_at_its_floor(self, built):
        path, pages = built
        scores = control_scores(
            grid_of(load_tasks(pages, path=path)), Environment(benchmark_version="0.1.0", benchmark_sha="abc")
        )
        assert scores["spelling"] <= SPELLING_CEILING
        assert all(score <= CONTROL_CEILING for name, score in scores.items() if name != "spelling")

    def test_the_preflight_clears_on_the_dev_split(self, built):
        path, pages = built
        check = preflight(
            grid_of(load_tasks(pages, path=path)), Environment(benchmark_version="0.1.0", benchmark_sha="abc")
        )
        assert check.clear, check.blocked

    def test_the_adapter_can_read_the_records(self, built):
        pytest.importorskip("oold.agent.prompts")
        from oold_llm_bench.runner.adapter import build_request

        path, pages = built
        tasks = load_tasks(pages, path=path)
        cell = Cell(
            condition=Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=8),
            model=ModelSpec(model="m", provider_profile="openai", model_version="1"),
            task=tasks[0],
            repetition=1,
        )
        assert build_request(cell).document == tasks[0].document


class TestTheCommittedCorpus:
    """The file in the repository, checked for the things only it can be wrong
    about. The documents themselves need the page cache, which is local."""

    @pytest.fixture(scope="class")
    @classmethod
    def corpus(cls):
        return read_corpus()

    def test_it_is_built_from_one_named_crawl(self, corpus):
        assert corpus.crawl == CRAWL_ID

    def test_every_document_can_be_fetched_again_from_what_it_records(self, corpus):
        for document in corpus.documents:
            assert document.warc.filename.startswith("crawl-data/")
            assert document.warc.length > 0
            assert len(document.warc.digest) > 10

    def test_every_document_states_at_least_what_a_task_needs(self, corpus):
        for document in corpus.documents:
            assert len(document.entities[0].fields) >= MIN_GROUNDED_PROPERTIES

    def test_no_document_answers_with_a_structural_property(self, corpus):
        for document in corpus.documents:
            for entity in document.entities:
                assert not set(entity.fields) & STRUCTURAL_PROPERTIES

    def test_no_document_answers_with_a_labelled_property(self, corpus):
        for document in corpus.documents:
            for entity in document.entities:
                assert not set(entity.fields) & LABELLED_PROPERTIES

    def test_no_value_is_long_enough_to_be_the_page(self, corpus):
        for document in corpus.documents:
            for entity in document.entities:
                for value in entity.fields.values():
                    for item in value if isinstance(value, list) else [value]:
                        assert len(str(item)) <= MAX_VALUE_CHARS

    def test_every_link_reaches_an_entity_of_the_same_document(self, corpus):
        for document in corpus.documents:
            keys = {entity.key for entity in document.entities}
            for entity in document.entities:
                for targets in entity.links.values():
                    assert set(targets) <= keys

    def test_the_class_a_document_is_filed_under_is_the_class_it_asserts(self, corpus):
        """The subset a url came from decides nothing on its own: what the page
        annotates is what the archived snapshot says it annotates."""
        for document in corpus.documents:
            assert document.entities[0].class_name == document.main_class

    def test_the_corpus_does_not_repeat_one_entity_shape(self, corpus):
        """One shape everywhere makes the edge answerable without reading.

        Where every document is one class pointing at one other, alignment puts
        a wrong document's entities on the right ones and the link between them
        is correct by construction. ``TestTheControlsOnALinkedGrid`` in
        ``test_preflight`` is that failure, measured.
        """
        shapes = Counter(tuple(entity.class_name for entity in document.entities) for document in corpus.documents)
        assert len(shapes) >= 3
        assert max(shapes.values()) < len(corpus.documents) * 0.9

    def test_one_domain_does_not_fill_the_corpus(self, corpus):
        """A 1000-quad sample of one subset was 28 pages from one spam domain."""
        hosts = {document.host for document in corpus.documents}
        assert len(hosts) == len(corpus.documents)

    def test_the_licences_that_were_not_granted_are_recorded(self, corpus):
        assert "research purposes only" in corpus.licence["web_data_commons"]["statement"]
        assert "copyright" in corpus.licence["page_text"]["statement"]

    def test_the_grounding_rate_is_reported_per_class_and_per_property(self, corpus):
        assert corpus.grounding["per_class"]
        assert corpus.grounding["per_property"]
        assert 0.0 < corpus.grounding["rate"] <= 1.0

    def test_the_documents_are_not_in_the_file(self):
        from oold_llm_bench.corpus.wdc import CORPUS_PATH

        assert "text_sha256" in CORPUS_PATH.read_text(encoding="utf-8")
        assert '"document"' not in CORPUS_PATH.read_text(encoding="utf-8")
