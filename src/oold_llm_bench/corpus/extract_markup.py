"""Schema.org markup lifted out of a page, and the page text without it.

A crawled document is worth a task only if its ground truth came from the page
rather than from someone reading the page. Embedded markup is that truth: the
site owner wrote the prose and the annotation together, so the pairing is
authored rather than annotated, and the benchmark inherits it at no labeling
cost. This is the external validity check on the round-trip corpora, which
measure sixteen sentence frames however many documents they emit.

Nothing here touches the network. A string goes in and data comes out, so
every mess this has to survive is reproducible from a fixture. There are
plenty of them in the wild: four JSON-LD blocks of which one is truncated, a
``@graph`` wrapper, an array at the top level, a ``@type`` that is a list, a
CMS that HTML-escaped the whole block. A parser that gives up on the page
because one block is broken throws away the larger share of the real corpus,
so each block is decoded on its own and a failure is recorded rather than
raised.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

__all__ = [
    "FULL_TEXT_PROPERTIES",
    "Extraction",
    "MarkupEntity",
    "Syntax",
    "declared_licence",
    "extract",
    "normalise_term",
    "to_jsonld",
]


class Syntax(str, Enum):
    """Which serialization an entity was read from.

    Recorded per entity, not per page. The two syntaxes fail
    differently, JSON-LD on malformed JSON and microdata on unclosed tags, so
    a corpus statistic that mixed them would hide which of the two is costing
    documents.
    """

    JSON_LD = "json-ld"
    MICRODATA = "microdata"


FULL_TEXT_PROPERTIES = frozenset({"articleBody", "reviewBody", "text", "transcript"})
"""Properties whose value is the page body rather than a fact about it.

Dropped from the redistributable truth. We publish urls, hashes and our own
extraction, never page text, and ``articleBody`` is the page text under
another name. Nothing is lost for extraction: no arm is asked to reproduce a
whole article, so these carry no gradeable answer.
"""

_URL_ATTRIBUTE = {
    "a": "href",
    "area": "href",
    "audio": "src",
    "embed": "src",
    "iframe": "src",
    "img": "src",
    "link": "href",
    "object": "data",
    "source": "src",
    "track": "src",
    "video": "src",
}
"""Where a microdata value lives when the element is not read for its text.

From the microdata specification. Without this an ``<img itemprop="image">``
contributes the empty string, and a ``<time itemprop="datePublished">`` yields
whatever the page happened to display instead of the machine-readable date it
carries in an attribute.
"""

_PLAIN_ATTRIBUTE = {"meta": "content", "data": "value", "meter": "value", "time": "datetime"}

_VOID = frozenset({
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
})

_TEXTLESS = frozenset({"script", "style", "noscript", "template", "svg", "title"})
"""Elements whose content is not what a reader sees.

``title`` is in the list because a task document is the page body. Keeping it
would put the site name and a slogan in front of every model, which is text
that looks like content and answers nothing.
"""

_BLOCK = frozenset({
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "dd",
    "dt",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
})

_LICENCE_META = frozenset({"dcterms.license", "dc.rights", "dc.rights.license", "license", "copyright"})

_SCHEMA_PREFIX = re.compile(r"^(?:https?://(?:www\.)?schema\.org/|schema:)")
_CDATA_OPEN = re.compile(r"^\s*(?://\s*)?(?:<!--\s*)?<!\[CDATA\[")
_CDATA_CLOSE = re.compile(r"(?://\s*)?\]\]>\s*(?:-->)?\s*$")
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")

_DECODER = json.JSONDecoder(strict=False)
"""Tolerates raw control characters inside strings.

A CMS that pastes a description with a literal newline in it produces JSON
that no strict parser accepts, and that is a formatting accident rather than a
statement the page did not mean to make.
"""


def normalise_term(value: str) -> str:
    """A type or property name as the vocabulary spells it.

    The same class arrives as ``Book``, ``schema:Book`` and
    ``https://schema.org/Book`` depending on who generated the page. Three
    spellings of one class would look like three classes to anything counting
    the corpus, and would need the consumer to know all three to match on any.
    """
    return _SCHEMA_PREFIX.sub("", value.strip())


@dataclass(frozen=True)
class MarkupEntity:
    """One annotated thing found on a page.

    ``node`` is the entity as it will be redistributed, so nested objects stay
    nested: an ``Offer`` inside a ``Product`` is part of what the page asserts
    about the product, and flattening it here would discard the attachment
    that makes the document interesting to extract from.
    """

    types: tuple[str, ...]
    node: dict[str, Any]
    syntax: Syntax

    @property
    def properties(self) -> dict[str, Any]:
        """The node without the JSON-LD keywords."""
        return {key: value for key, value in self.node.items() if not key.startswith("@")}


@dataclass(frozen=True)
class Extraction:
    """What one page yielded.

    ``errors`` is part of the result and not an exception, because a page with
    one broken block among four is a usable page and the broken block is a
    corpus statistic worth keeping.
    """

    entities: tuple[MarkupEntity, ...]
    text: str
    licence: str | None = None
    errors: tuple[str, ...] = ()

    @property
    def typed(self) -> tuple[MarkupEntity, ...]:
        """Entities that name a class.

        An untyped node asserts properties of nothing, so it cannot be the
        answer to "what does this document describe" and is not counted when
        deciding whether a page carries usable truth.
        """
        return tuple(entity for entity in self.entities if entity.types)


def extract(document: str, *, base_url: str | None = None) -> Extraction:
    """Read the markup and the visible text out of one HTML document.

    JSON-LD is read before microdata because it is the larger share of the
    web and because it survives a mangled DOM, which microdata does not. Both
    are kept: a page that carries only microdata is still a page with authored
    truth on it, and dropping those would bias the corpus towards whichever
    CMS families emit JSON-LD.
    """
    reader = _Reader(base_url)
    reader.feed(document)
    reader.close()

    entities: list[MarkupEntity] = []
    errors: list[str] = []
    for index, block in enumerate(reader.scripts):
        value, error = _decode(block)
        if error is not None:
            errors.append(f"json-ld block {index}: {error}")
            continue
        for node in _nodes(value):
            entities.append(MarkupEntity(types=_types(node), node=node, syntax=Syntax.JSON_LD))
    for item in reader.items:
        node = item.node()
        entities.append(MarkupEntity(types=_types(node), node=node, syntax=Syntax.MICRODATA))

    return Extraction(
        entities=tuple(entities),
        text=reader.text(),
        licence=_licence(entities, reader),
        errors=tuple(errors),
    )


def declared_licence(document: str) -> str | None:
    """The licence the page states, if it states one."""
    return extract(document).licence


def to_jsonld(
    extraction: Extraction,
    *,
    drop_properties: Collection[str] = FULL_TEXT_PROPERTIES,
) -> dict[str, Any]:
    """The extracted entities as the one JSON-LD document we may republish.

    A single ``@graph`` instead of one file per entity, because what the page
    asserts is the set of them together, and an entity separated from its
    siblings loses the cross references between them.
    """
    drop = frozenset(drop_properties)
    return {
        "@context": "https://schema.org",
        "@graph": [_pruned(entity.node, drop) for entity in extraction.entities],
    }


def _pruned(node: Any, drop: frozenset[str]) -> Any:
    if isinstance(node, dict):
        return {key: _pruned(value, drop) for key, value in node.items() if normalise_term(key) not in drop}
    if isinstance(node, list):
        return [_pruned(item, drop) for item in node]
    return node


def _unwrap(block: str) -> str:
    """Strip the wrappers pages put around an inline script.

    Comment and CDATA fencing is left over from XHTML era templates and is
    still emitted by older CMS themes. It is not JSON, so a strict parser
    rejects the block, and the block is otherwise fine.
    """
    text = block.strip()
    if text.startswith("<!--"):
        text = text[4:]
    if text.endswith("-->"):
        text = text[:-3]
    text = _CDATA_OPEN.sub("", text)
    text = _CDATA_CLOSE.sub("", text)
    return text.strip()


def _candidates(block: str) -> Iterator[str]:
    """Progressively less faithful readings of one block.

    Order is the point. The block as written is tried first, so a page that is
    correct is never altered. A repair is reached only once strict decoding has
    already failed, where the alternative is losing the block outright.
    """
    text = _unwrap(block)
    yield text
    unescaped = html.unescape(text)
    if unescaped != text:
        yield unescaped
    # A trailing comma before a closing brace is the most common hand-edited
    # JSON-LD error. The substitution can also reach inside a string literal
    # that happens to contain ", }", which is why it is last.
    for candidate in (text, unescaped):
        repaired = _TRAILING_COMMA.sub(r"\1", candidate)
        if repaired != candidate:
            yield repaired


def _decode(block: str) -> tuple[Any, str | None]:
    """One JSON-LD block, or why it could not be read."""
    message = "empty block"
    for candidate in _candidates(block):
        if not candidate:
            continue
        try:
            return _DECODER.decode(candidate), None
        except ValueError as error:
            message = str(error)
    return None, message


def _nodes(value: Any) -> Iterator[dict[str, Any]]:
    """Flatten whatever shape the block has.

    Three shapes are all common: a bare object, an array of objects, and an
    object whose real content sits under ``@graph``. Handling only the first
    would drop most pages that describe more than one thing, which includes
    every breadcrumb trail and every product listing.
    """
    if isinstance(value, list):
        for item in value:
            yield from _nodes(item)
    elif isinstance(value, dict):
        graph = value.get("@graph")
        if graph is not None:
            yield from _nodes(graph)
        elif value:
            yield value


def _types(node: dict[str, Any]) -> tuple[str, ...]:
    """The classes a node claims.

    A list is as valid as a string, and a page writes one when one thing is
    both a ``Product`` and a ``Book``, so both spellings resolve to the same
    tuple and a consumer never has to test which it got.
    """
    declared = node.get("@type", node.get("type"))
    if isinstance(declared, str):
        return (normalise_term(declared),)
    if isinstance(declared, list):
        return tuple(normalise_term(entry) for entry in declared if isinstance(entry, str))
    return ()


def _licence(entities: Iterable[MarkupEntity], reader: _Reader) -> str | None:
    """What the page says about reuse, in decreasing order of authority.

    The markup is read first: a ``license`` property is the site owner stating
    the terms for the content itself. ``rel="license"`` comes next, then the
    Dublin Core metadata, both of which describe the page. None of the three is
    a permission to redistribute the text, and nothing in this package treats
    them as one. They are recorded so a later decision has the evidence.
    """
    for entity in entities:
        for key, value in entity.node.items():
            if normalise_term(key) == "license":
                stated = _licence_value(value)
                if stated:
                    return stated
    for href in reader.licence_links:
        if href:
            return href
    for name, content in reader.metas:
        if name.lower() in _LICENCE_META and content:
            return content
    return None


def _licence_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        for entry in value:
            stated = _licence_value(entry)
            if stated:
                return stated
        return None
    if isinstance(value, dict):
        for key in ("@id", "url", "name"):
            stated = value.get(key)
            if isinstance(stated, str) and stated.strip():
                return stated.strip()
    return None


def _collapse(text: str) -> str:
    return " ".join(text.split())


@dataclass
class _Item:
    """A microdata item while it is still being read."""

    types: tuple[str, ...]
    item_id: str | None = None
    properties: dict[str, list[Any]] = field(default_factory=dict)

    def add(self, name: str, value: Any) -> None:
        self.properties.setdefault(name, []).append(value)

    def node(self) -> dict[str, Any]:
        """The item in the JSON-LD shape the rest of the package reads.

        Microdata repeats a property name to express a list and has no way to
        say "exactly one", so a single occurrence is written as a scalar and
        several as an array. That is the convention JSON-LD pages already
        follow, and it keeps both syntaxes gradeable by one consumer.
        """
        node: dict[str, Any] = {}
        if len(self.types) == 1:
            node["@type"] = self.types[0]
        elif self.types:
            node["@type"] = list(self.types)
        if self.item_id:
            node["@id"] = self.item_id
        for name, values in self.properties.items():
            unpacked = [value.node() if isinstance(value, _Item) else value for value in values]
            node[name] = unpacked[0] if len(unpacked) == 1 else unpacked
        return node


@dataclass
class _Frame:
    """One open element, kept so its end tag knows what to do."""

    tag: str
    item: _Item | None
    properties: tuple[str, ...]
    value: str | None
    text: list[str] = field(default_factory=list)


class _Reader(HTMLParser):
    """One pass over the document for markup, license hints and text.

    One pass instead of three, because the three readings have to agree on what
    counts as content. Stripping the text separately from lifting the microdata
    would let a ``<script>`` body end up in the document a model reads while
    the same block was also parsed as truth.
    """

    def __init__(self, base_url: str | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[str] = []
        self.items: list[_Item] = []
        self.licence_links: list[str] = []
        self.metas: list[tuple[str, str]] = []
        self.base = base_url
        self._frames: list[_Frame] = []
        self._open: list[_Item] = []
        self._chunks: list[str] = []
        self._textless = 0
        self._script: list[str] | None = None

    def text(self) -> str:
        """The visible text, without the whitespace markup leaves behind.

        Block boundaries become line breaks rather than spaces. Without them a
        table of specifications collapses into one sentence and the model is
        asked to read prose the page never showed anyone. A block that opens
        and closes leaves two breaks, so paragraphs end up separated by a blank
        line and a ``<br>`` only by one, which is the distinction the page was
        making.
        """
        lines = [_collapse(line) for line in "".join(self._chunks).split("\n")]
        kept: list[str] = []
        for line in lines:
            if line or (kept and kept[-1]):
                kept.append(line)
        return "\n".join(kept).strip()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.lower(): (value or "") for name, value in attrs}
        if tag == "base" and attributes.get("href"):
            self.base = attributes["href"]
        elif tag in ("link", "a") and "license" in attributes.get("rel", "").lower().split():
            self.licence_links.append(attributes.get("href", ""))
        elif tag == "meta" and attributes.get("name"):
            self.metas.append((attributes["name"], attributes.get("content", "")))

        if tag == "script" and _is_jsonld(attributes.get("type", "")):
            self._script = []
        if tag in _TEXTLESS:
            self._textless += 1
        if tag in _BLOCK:
            self._chunks.append("\n")

        self._push(tag, attributes)
        if tag in _VOID:
            self.handle_endtag(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script is not None:
            self.scripts.append("".join(self._script))
            self._script = None
        if tag in _TEXTLESS:
            self._textless = max(0, self._textless - 1)
        if tag in _BLOCK and tag not in _VOID:
            self._chunks.append("\n")
        self._pop(tag)

    def handle_data(self, data: str) -> None:
        if self._script is not None:
            self._script.append(data)
            return
        if self._textless:
            return
        self._chunks.append(data)
        for frame in self._frames:
            frame.text.append(data)

    def _push(self, tag: str, attributes: dict[str, str]) -> None:
        item: _Item | None = None
        if "itemscope" in attributes:
            types = tuple(normalise_term(entry) for entry in attributes.get("itemtype", "").split() if entry)
            item = _Item(types=types, item_id=attributes.get("itemid") or None)
            self._open.append(item)
        properties = tuple(normalise_term(name) for name in attributes.get("itemprop", "").split())
        value = self._attribute_value(tag, attributes) if properties and item is None else None
        self._frames.append(_Frame(tag=tag, item=item, properties=properties, value=value))

    def _pop(self, tag: str) -> None:
        """Close the element, tolerating the tags a page forgot to close.

        Real pages leave ``<li>`` and ``<p>`` open, and a strict stack would
        either lose every item after the first mistake or attach a property to
        the wrong subject. Unwinding to the matching tag keeps the damage
        inside the element that was malformed.
        """
        for index in range(len(self._frames) - 1, -1, -1):
            if self._frames[index].tag != tag:
                continue
            for frame in reversed(self._frames[index:]):
                self._finish(frame)
            del self._frames[index:]
            return

    def _finish(self, frame: _Frame) -> None:
        if frame.item is not None:
            if self._open and self._open[-1] is frame.item:
                self._open.pop()
            parent = self._open[-1] if self._open else None
            if frame.properties and parent is not None:
                for name in frame.properties:
                    parent.add(name, frame.item)
            else:
                self.items.append(frame.item)
            return
        if not frame.properties or not self._open:
            return
        value = frame.value if frame.value is not None else _collapse("".join(frame.text))
        for name in frame.properties:
            self._open[-1].add(name, value)

    def _attribute_value(self, tag: str, attributes: dict[str, str]) -> str | None:
        name = _PLAIN_ATTRIBUTE.get(tag)
        if name is not None:
            return attributes.get(name, "")
        name = _URL_ATTRIBUTE.get(tag)
        if name is None:
            return None
        value = attributes.get(name, "")
        return urljoin(self.base, value) if self.base and value else value


def _is_jsonld(declared: str) -> bool:
    return declared.strip().lower().split(";")[0] in {"application/ld+json", "application/json+ld"}
