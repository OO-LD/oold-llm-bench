"""A directional link between two entities this corpus already knows.

The Wikidata-schema.org corpus grounds an entity's own facts against its own
lead. This corpus grounds something the lead does not state about itself at
all: that the lead's text names another entity, under a given anchor, and that
anchor resolves to a Wikidata item the corpus already has a record for. The
question this asks of a step is not "what does this article say about its
subject" but "which of the entities it mentions does it actually mean."

*The harvest.* :mod:`scripts.harvest_wikidata_links` fetched each article's
lead section as wikitext at the exact revision
:attr:`~oold_llm_bench.corpus.wikidata_schemaorg.GroundedEntity.revision`
already pins, extracted every ``[[target|anchor]]`` wikilink, and kept the ones
whose target resolves to a QID this corpus's own entity set contains. That is
``.cache/wikidata_links/links.json``, 513 rows over 959 articles, read-only
input to this module.

*Grounding, the same discipline as the fact it extends.* The wikitext a link
was parsed from is not the plain-text lead a model is shown; `prop=extracts`
strips markup, folds whitespace and can drop a sentence the infobox alone
carried. An anchor is kept only once
:func:`~oold_llm_bench.corpus.wikidata_schemaorg.stated_in` finds it, on word
boundaries, in the exact document :func:`~oold_llm_bench.experiments.corpora.documents_for`
returns for the source. Checked against the harvest as cached here: 380 of the
513 rows ground this way, not the 513 a sidebar field would suggest and not
quite the 385 an earlier spot check reported either, which is why this module
recomputes the count from the real matcher rather than carrying it as a
constant. The other 133 are infobox or sidebar wikilinks, such as an
affiliation field, that the lead's prose never reaches.

*Self-links, found while building this.* 31 of the 380 grounded rows have the
same QID on both ends: a title resolves, through a redirect or a merged
Wikidata item, back to the very article that links it ("Akai" linking "Akai
Professional", which Wikidata holds as one item; "University of Ottawa"
bolding its own name as a wikilink to itself). A link task's two ends have to
be two entities, or :class:`~oold_llm_bench.tasks.models.ExpectedInstance`'s
own key would collide with itself, so these are dropped here and counted, not
silently absorbed into the grounding rate above.

*One task per source, several targets where the lead names several.* 41 of the
208 remaining sources link to more than one distinct target (Amherst
College's lead names both Smith College and UMass Amherst), so
:data:`LINK_PROPERTY` holds a list of
:class:`~oold_llm_bench.grading.triples.Reference` and never a single bare
one: the schema a model is offered has to admit as many links as the ground
truth can hold, and declaring it singular would make the second link
unanswerable by construction. Five pairs also carry more than one distinct
anchor for the same edge ("Dow" and "Dow Chemical" both naming the same
target); those collapse into one edge with every anchor kept as a mention,
because repeating the edge once per anchor would let a step score the same
link twice.

*What the link is called.* schema.org pairs such as ``alumniOf`` between a
``Person`` and a ``CollegeOrUniversity`` would be a closer fit for some of
these edges than a generic name, but deciding which schema.org property
applies needs the two classes' domain and range to be checked against each
other, and no mapping of that kind exists in this codebase to reuse; writing
one here would be asserting semantics for 208 class pairs that nothing
verifies. :data:`LINK_PROPERTY` is therefore one property, offered on every
class the same way ``name`` is, and the question this corpus asks is only ever
"does the link resolve," never "is this the schema.org term for it."

*The target's own record.* :class:`~oold_llm_bench.tasks.models.TaskRecord`
refuses a :class:`~oold_llm_bench.grading.triples.Reference` that names no
expected instance of the same task, so the target has to appear in
``expected`` beside the source, even though its own facts are not stated by
this document and are not asked for here: :func:`load_entities` already turns
it into a task of its own, grounded in its own lead, elsewhere in this corpus.
The stub carried here asserts exactly one thing, the target's own name, using
the identifier scheme :func:`~oold_llm_bench.corpus.wikidata_schemaorg.GroundedEntity.id`
and :attr:`~oold_llm_bench.corpus.wikidata_schemaorg.GroundedEntity.cls`
already give it, so a reader comparing this task's target against its
standalone one is comparing the same key and the same class and not a second
naming scheme invented for this corpus alone.

*Why there is no build script.* Turning this into a persisted file under
``data/`` the way :mod:`scripts.build_wikidata_schemaorg` persists the facts
corpus would mean either committing the anchors, which are short verbatim
quotes of CC BY-SA 4.0 prose and a licensing call this module should not make
unilaterally, or committing only the grounding decision and still needing
:func:`~oold_llm_bench.experiments.corpora.documents_for` at build time to
make it, which is the one expensive step and is not avoided. The filter over
513 cached rows is a few hundred word-boundary string searches; it is cheap
enough to redo on every load, and redoing it means the kept set can never
drift from the documents it was computed against.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oold_llm_bench.corpus.schemaorg import CATALOGUE_SLOTS, answer_schema, branches_for, render_class
from oold_llm_bench.corpus.wikidata_schemaorg import (
    NAME,
    TEXT_LICENCE,
    GroundedEntity,
    _mentions_of,
    classes_of,
    read_grounded_corpus,
    stated_in,
)
from oold_llm_bench.grading.triples import Reference
from oold_llm_bench.tasks.models import (
    CorpusRef,
    Difficulty,
    ExpectedInstance,
    Source,
    Split,
    TaskRecord,
    Variant,
)

__all__ = [
    "LINKS_CACHE",
    "LINK_PROPERTY",
    "LINK_PROPERTY_DESCRIPTION",
    "GroundedLink",
    "ground_links",
    "load_linked_articles",
    "read_links",
]

LINKS_CACHE = Path(".cache/wikidata_links/links.json")
"""Where :mod:`scripts.harvest_wikidata_links` left the harvested rows.

Gitignored, like every other harvest cache in this repository. A caller
without it gets a plain ``FileNotFoundError`` pointing at the script that
produces it, the same posture :func:`~oold_llm_bench.experiments.corpora.documents_for`
takes towards its own cache.
"""

LINK_PROPERTY = "mentionsEntity"
"""The one property every offered class carries this edge under.

Chosen over a class-pair-specific schema.org property such as ``alumniOf``
because verifying the second requires checking a property's domain and range
against the two classes an edge actually connects, and nothing in this
codebase does that checking; see the module docstring. A single name that
means "this article's text names that entity" is true of every pair this
harvest produced, and is not a claim about which schema.org relation, if any,
would fit better.
"""

LINK_PROPERTY_DESCRIPTION = (
    "Another entity this article's text names, resolved to that entity's own record. "
    "The value points at an entity, not at a string: grade it by which record it reaches."
)

_LINK_SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "object"}}
"""The JSON shape :data:`LINK_PROPERTY` takes in an answer schema.

Always an array, for the reason :func:`~oold_llm_bench.corpus.schemaorg.slot_schema`
is: 41 of the 208 kept sources name more than one distinct target, so a schema
that let the property be a single object would make the second link
unanswerable by construction. Not :func:`~oold_llm_bench.corpus.schemaorg.link_schema`,
which is exactly this shape minus the array: that function serves the
generated corpus's edges, which :func:`~oold_llm_bench.corpus.schemaorg.draw_linked`
never draws more than one of per entity, so the gap was never exercised there.
"""


@dataclass(frozen=True)
class GroundedLink:
    """One directional edge, kept because the lead was found to state it.

    ``anchors`` holds every distinct anchor the source document names the
    target by, not only the first: five kept pairs are named more than one
    way in their own lead ("Dow" and "Dow Chemical"), and a step answering
    either has read the edge correctly.
    """

    source: str
    target: str
    anchors: tuple[str, ...]


def read_links(path: Path | None = None) -> list[dict[str, Any]]:
    """The harvested rows, exactly as the harvest script wrote them."""
    return json.loads((path or LINKS_CACHE).read_text(encoding="utf-8"))


def ground_links(
    rows: list[dict[str, Any]],
    by_qid: Mapping[str, GroundedEntity],
    documents: Mapping[str, str],
) -> list[GroundedLink]:
    """Harvested rows, reduced to the edges a model's own document supports.

    A row is kept only when its anchor is found, on word boundaries, in the
    exact plain-text lead :func:`~oold_llm_bench.experiments.corpora.documents_for`
    returns for the source: the harvest parsed wikitext, which the lead a
    model reads has already been rendered away from.

    A row whose source and target are the same QID is dropped before anything
    else, because that is not a link between two entities, and keeping it
    would collide the one expected instance a key can name with itself.

    Rows that share a ``(source, target)`` collapse into one
    :class:`GroundedLink`, anchors unioned in the order first seen: the
    harvest can state one edge under two spellings, and that is one edge
    stated twice, not two edges.
    """
    anchors_by_pair: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        source, target, anchor = row["source"], row["target"], row["anchor"]
        if source == target:
            continue
        entity = by_qid.get(source)
        if entity is None or target not in by_qid:
            # The harvest resolved this against a corpus snapshot; a link
            # whose source or target has since left the corpus names nothing
            # this build can score, and is left out rather than guessed at.
            continue
        document = documents.get(entity.id)
        if document is None:
            raise ValueError(f"{entity.id} is linked from but {entity.id} has no document to ground its anchors in")
        if hashlib.sha256(document.encode("utf-8")).hexdigest() != entity.sha256:
            raise ValueError(f"{entity.id} was measured on revision {entity.revision} and the document given is not it")
        if stated_in(document, (anchor,)) is None:
            continue
        anchors_by_pair.setdefault((source, target), []).append(anchor)
    return [
        GroundedLink(source=source, target=target, anchors=tuple(dict.fromkeys(anchors)))
        for (source, target), anchors in sorted(anchors_by_pair.items())
    ]


def _fields_of(entity: GroundedEntity) -> dict[str, Any]:
    """An entity's own facts, one value bare and several as a list.

    The same reduction :func:`~oold_llm_bench.corpus.wikidata_schemaorg.load_entities`
    applies to the identical ``facts`` dict, repeated here rather than
    imported because it is a one-line reduction and not a decision this module
    would otherwise have to keep in step with by calling back into a private
    part of that one.
    """
    return {prop: values[0] if len(values) == 1 else list(values) for prop, values in entity.facts.items()}


def load_linked_articles(
    documents: Mapping[str, str],
    *,
    links_path: Path | None = None,
    corpus_path: Path | None = None,
    rows: list[dict[str, Any]] | None = None,
    split: Split = Split.DEV,
    difficulty: Difficulty = Difficulty.HARD,
    describe_catalogue: bool = True,
) -> list[TaskRecord]:
    """Grounded wikilinks, as one task per source article.

    ``documents`` is the same mapping :func:`~oold_llm_bench.corpus.wikidata_schemaorg.load_entities`
    takes: a record id to the lead of the revision the corpus was measured on.
    Every source document a kept link needs is checked against its recorded
    sha256 here too, for the reason that function already states: the
    grounding decision in :func:`ground_links` was taken against those exact
    bytes.

    ``rows`` bypasses :func:`read_links`, mainly so a test can hand this a
    handful of rows without a file on disk; a caller outside a test leaves it
    ``None`` and gets the cached harvest.

    The catalogue offered is every class this corpus answers, the same set
    :func:`~oold_llm_bench.corpus.wikidata_schemaorg.load_entities` offers by
    default, with :data:`LINK_PROPERTY` added to every branch and to the
    answer schema: the property is not declared on any one
    :class:`~oold_llm_bench.corpus.schemaorg.SchemaClass` the way a real
    schema.org link would be, because it is not one, so it is added here by
    hand rather than appearing through :func:`~oold_llm_bench.corpus.schemaorg.branches_for`'s
    own ``with_links``.
    """
    corpus = read_grounded_corpus(corpus_path)
    by_qid = {entity.qid: entity for entity in corpus.entities}
    kept = ground_links(rows if rows is not None else read_links(links_path), by_qid, documents)

    by_source: dict[str, dict[str, tuple[str, ...]]] = {}
    for link in kept:
        by_source.setdefault(link.source, {})[link.target] = link.anchors

    classes = classes_of(corpus)
    offered = [cls.name for cls in classes]
    shape = answer_schema(classes, Variant.NATIVE)
    narrowed = branches_for(classes, Variant.NATIVE)
    shape["properties"]["entities"]["items"]["properties"][LINK_PROPERTY] = _LINK_SCHEMA
    for entry in narrowed.values():
        entry[LINK_PROPERTY] = _LINK_SCHEMA
    lineage = {cls.name: [p for p in cls.parents if p in set(offered)] for cls in classes}
    described = (
        {cls.name: render_class(cls, Variant.NATIVE, CATALOGUE_SLOTS) for cls in classes}
        if describe_catalogue
        else None
    )
    described_properties = {
        name: body["description"]
        for name, body in (corpus.properties or {}).items()
        if isinstance(body, dict) and body.get("description")
    }
    described_properties[LINK_PROPERTY] = LINK_PROPERTY_DESCRIPTION

    tasks: list[TaskRecord] = []
    for source in sorted(by_source):
        entity = by_qid[source]
        document = documents[entity.id]
        targets = by_source[source]
        target_ids = sorted(targets)

        fields = _fields_of(entity)
        fields[LINK_PROPERTY] = [Reference(key=by_qid[target].id) for target in target_ids]
        source_instance = ExpectedInstance(
            key=entity.id,
            class_path=entity.cls,
            fields=fields,
            mentions=_mentions_of(entity, document),
        )
        target_instances = []
        for target in target_ids:
            target_entity = by_qid[target]
            name = (target_entity.facts.get(NAME) or [target_entity.title])[0]
            target_instances.append(
                ExpectedInstance(
                    key=target_entity.id,
                    class_path=target_entity.cls,
                    fields={NAME: name},
                    mentions=targets[target],
                )
            )

        tasks.append(
            TaskRecord(
                id=f"wdl-{source}",
                document=document,
                expected=[source_instance, *target_instances],
                corpus=CorpusRef(
                    name="wikidata-links",
                    source=Source.BULK,
                    # The source's own facts are the subset load_entities also
                    # publishes, and the targets carry one field each by
                    # construction, so neither end is the whole of what either
                    # entity's own article states.
                    exhaustive=False,
                    document_id=source,
                    content_hash=entity.sha256,
                    url=entity.url,
                    retrieved_at=corpus.retrieved_at,
                    licence=TEXT_LICENCE,
                ),
                split=split,
                difficulty=difficulty,
                catalogue=list(offered),
                catalogue_text=described,
                property_text=described_properties,
                answer_schema=shape,
                branches=narrowed,
                class_parents=lineage,
                notes=(
                    f"corpus=wikidata-links,source={source},targets={len(target_ids)},"
                    f"revision={entity.revision},licence={TEXT_LICENCE}"
                ),
            )
        )
    return tasks
