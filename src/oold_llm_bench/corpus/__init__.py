"""Turning schemas and documents into tasks.

Round-trip generation first, because it needs no network and its ground truth
is correct by construction. Acquired documents arrive later as the check that
synthetic prose is not the only thing being measured.

:mod:`~oold_llm_bench.corpus.wiki_measurements` is the first of those that can
be published rather than pointed at, and it carries its own licence terms:
CC BY-SA 4.0 on the sentences, which reaches anything derived from them.

:mod:`~oold_llm_bench.corpus.wikidata_identity` is the second, and the first
under a single licence with no conditions. It measures a different capability:
whether two mentions denote one entity. It is kept separate from the
extraction corpora on purpose, because a system needs both and a failure has
to be attributable to one of them.

:mod:`~oold_llm_bench.corpus.wikidata_schemaorg` is the third, and it is what
:mod:`~oold_llm_bench.corpus.schemaorg` cannot be: a schema.org extraction
task whose document names neither the class nor the property. It publishes no
page text, so its documents are supplied by the caller and checked against the
sha256 of the revision the truth was measured on.
"""

from oold_llm_bench.corpus.crawl import (
    Crawler,
    Document,
    DocumentMeta,
    FetchResult,
    Response,
    build_document,
    document_id,
    manifest,
    urlopen_fetcher,
    write_document,
)
from oold_llm_bench.corpus.eln import Note, NoteFile, Reading, load_notes, read_notes
from oold_llm_bench.corpus.extract_markup import (
    Extraction,
    MarkupEntity,
    Syntax,
    declared_licence,
    extract,
    to_jsonld,
)
from oold_llm_bench.corpus.provenance import (
    QUANTITIES,
    SCHEMAORG,
    CorpusDigest,
    Module,
    corpus_digest,
    fetch_module,
    resolve_module,
)
from oold_llm_bench.corpus.quantities import (
    Notation,
    QuantityKind,
    UnitSymbols,
    candidates,
    describable,
    generate_task,
    identifier_for,
    load_kinds,
    load_unit_symbols,
    render,
    translatable,
    unit_identifiable,
    writable,
)
from oold_llm_bench.corpus.schemaorg import (
    Edge,
    Form,
    Kind,
    Labelling,
    Link,
    SchemaClass,
    Slot,
    canonical_value,
    describable_classes,
    designating_slots,
    generate_pair,
    implied_classes,
    legible_slots,
    linked_classes,
    load_classes,
    resolvable_links,
    top_level_sets,
    value_object_classes,
    written_value,
)
from oold_llm_bench.corpus.schemaorg import generate_task as generate_schemaorg_task
from oold_llm_bench.corpus.signals import Signal, SignalData, Vocabulary, load_signals
from oold_llm_bench.corpus.wdc import (
    ClusterIndex,
    MissingPages,
    PageDocument,
    WarcRef,
)
from oold_llm_bench.corpus.wdc import Corpus as WdcCorpus
from oold_llm_bench.corpus.wdc import Entity as WdcEntity
from oold_llm_bench.corpus.wdc import load_tasks as load_wdc_tasks
from oold_llm_bench.corpus.wdc import read_corpus as read_wdc_corpus
from oold_llm_bench.corpus.wiki_measurements import (
    Corpus,
    Example,
    load_examples,
    read_corpus,
)
from oold_llm_bench.corpus.wikidata_identity import (
    Baseline,
    EntityState,
    IdentityClass,
    IdentityCorpus,
    Pair,
    Truth,
    agreeing_properties,
    label_baseline,
    property_baseline,
    read_pairs,
    score_baseline,
    shared_strings,
)
from oold_llm_bench.corpus.wikidata_schemaorg import (
    GroundedCorpus,
    GroundedEntity,
    classes_of,
    document_request,
    load_entities,
    read_documents,
    read_grounded_corpus,
    spelled_date,
    stated_in,
    truthy,
    written_forms,
)

__all__ = [
    "QUANTITIES",
    "SCHEMAORG",
    "Baseline",
    "ClusterIndex",
    "Corpus",
    "CorpusDigest",
    "Crawler",
    "Document",
    "DocumentMeta",
    "Edge",
    "EntityState",
    "Example",
    "Extraction",
    "FetchResult",
    "Form",
    "GroundedCorpus",
    "GroundedEntity",
    "IdentityClass",
    "IdentityCorpus",
    "Kind",
    "Labelling",
    "Link",
    "MarkupEntity",
    "MissingPages",
    "Module",
    "Notation",
    "Note",
    "NoteFile",
    "PageDocument",
    "Pair",
    "QuantityKind",
    "Reading",
    "Response",
    "SchemaClass",
    "Signal",
    "SignalData",
    "Slot",
    "Syntax",
    "Truth",
    "UnitSymbols",
    "Vocabulary",
    "WarcRef",
    "WdcCorpus",
    "WdcEntity",
    "agreeing_properties",
    "build_document",
    "candidates",
    "canonical_value",
    "classes_of",
    "corpus_digest",
    "declared_licence",
    "describable",
    "describable_classes",
    "designating_slots",
    "document_id",
    "document_request",
    "extract",
    "fetch_module",
    "generate_pair",
    "generate_schemaorg_task",
    "generate_task",
    "identifier_for",
    "implied_classes",
    "label_baseline",
    "legible_slots",
    "linked_classes",
    "load_classes",
    "load_entities",
    "load_examples",
    "load_kinds",
    "load_notes",
    "load_signals",
    "load_unit_symbols",
    "load_wdc_tasks",
    "manifest",
    "property_baseline",
    "read_corpus",
    "read_documents",
    "read_grounded_corpus",
    "read_notes",
    "read_pairs",
    "read_wdc_corpus",
    "render",
    "resolvable_links",
    "resolve_module",
    "score_baseline",
    "shared_strings",
    "spelled_date",
    "stated_in",
    "to_jsonld",
    "top_level_sets",
    "translatable",
    "truthy",
    "unit_identifiable",
    "urlopen_fetcher",
    "value_object_classes",
    "writable",
    "write_document",
    "written_forms",
    "written_value",
]
