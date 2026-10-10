"""The rules that turn Wikidata into a schema module, tested without the network.

The generator is a script and the harvest it runs takes an hour, so the part
worth testing is the part that decides: which parents survive, what a datatype
becomes, which item ranges are embedded, and whether two runs produce the same
bytes. All of that is pure, and the fixtures below are what a harvest would
have handed it.

The last test is the one that costs the most and is worth the most: the
emitted module is put in front of the OO-LD validator, offline and strict,
and has to come back with nothing failing. A schema that is well-formed JSON
and not a valid OO-LD document would pass every other test here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

GENERATOR = Path(__file__).resolve().parent.parent / "scripts" / "generate_wikidata_oold.py"


def _load():
    """The script as a module.

    By path, because ``scripts/`` is not a package and never will be: a build
    step that has to be importable as a library is a library.
    """
    spec = importlib.util.spec_from_file_location("generate_wikidata_oold", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{GENERATOR} cannot be imported")
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs, because `dataclass` looks its own module up
    # in `sys.modules` while deciding whether an annotation is `KW_ONLY`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wd = _load()


def _example(qid, **claims):
    return wd.WikidataExample(qid=qid, label=qid, claims=claims)


def _string(value):
    return [{"kind": "string", "value": value}]


def _item(value):
    return [{"kind": "item", "value": value}]


FILM = wd.WikidataClass(
    qid="Q11424",
    label="film",
    description="a work of moving images",
    parents=("Q2431196", "Q4502142"),
    declared=("P57", "P345", "P577", "P2047", "P21"),
    sampled=4,
    revision=1001,
    example=_example(
        "Q2875",
        P57=_item("Q62503"),
        P345=_string("tt0031381"),
        P577=[{"kind": "time", "value": "+1939-12-15T00:00:00Z", "precision": 11}],
        P2047=[{"kind": "quantity", "value": "+238"}],
        P21=_item("Q6581072"),
    ),
)

WORK = wd.WikidataClass(qid="Q2431196", label="audiovisual work", description="a work", revision=1002)

ARTWORK = wd.WikidataClass(qid="Q4502142", label="visual artwork", description="an artwork", revision=1003)

HUMAN = wd.WikidataClass(
    qid="Q5",
    label="human",
    observed=(("P345", 18), ("P569", 12), ("P1559", 2)),
    sampled=20,
    revision=1004,
    example=_example(
        "Q62503", P345=_string("nm0004306"), P569=[{"kind": "time", "value": "+1886-04-02T00:00:00Z", "precision": 11}]
    ),
)

PROPERTIES = {
    "P21": wd.WikidataProperty(pid="P21", label="sex or gender", datatype="wikibase-item", ranges=("Q48264",)),
    "P57": wd.WikidataProperty(pid="P57", label="director", datatype="wikibase-item", ranges=("Q5",)),
    "P345": wd.WikidataProperty(pid="P345", label="IMDb ID", datatype="external-id", description="an identifier"),
    "P569": wd.WikidataProperty(pid="P569", label="date of birth", datatype="time"),
    "P577": wd.WikidataProperty(pid="P577", label="publication date", datatype="time"),
    "P1559": wd.WikidataProperty(pid="P1559", label="name in native language", datatype="monolingualtext"),
    "P2047": wd.WikidataProperty(pid="P2047", label="duration", datatype="quantity"),
}

CLASSES = [FILM, WORK, ARTWORK, HUMAN]


def _module(classes=None, properties=None, **kwargs):
    return wd.build_module(classes or CLASSES, properties or PROPERTIES, **kwargs)


def _meta(module, classes=None, properties=None):
    return wd.module_meta(
        module,
        classes or CLASSES,
        properties or PROPERTIES,
        roots=["Q11424"],
        depth=1,
        retrieved="2026-10-07",
        sample=4,
        cap=wd.DEFAULT_CAP,
        usage=wd.DEFAULT_USAGE,
    )


class TestInheritance:
    """``film`` has three parents in Wikidata and the schema.org module keeps
    one. That is the defect this generator exists not to repeat."""

    def test_every_parent_in_the_module_is_kept(self):
        schema = _module().schemas["Film.schema.json"]
        assert [entry["$ref"] for entry in schema["allOf"]] == [
            "AudiovisualWork.schema.json",
            "VisualArtwork.schema.json",
        ]

    def test_the_context_reflects_the_parents_in_the_same_order(self):
        """The validator refuses a schema whose ``@context`` and ``allOf``
        disagree, because it is read as a remote context as it stands."""
        schema = _module().schemas["Film.schema.json"]
        assert schema["@context"][:2] == ["AudiovisualWork.schema.json", "VisualArtwork.schema.json"]

    def test_a_parent_outside_the_module_is_recorded_and_not_dropped(self):
        """No ``$ref`` can name a class the walk did not reach, so the only
        honest place for it is the record."""
        narrowed = _module(classes=[FILM, WORK, HUMAN])
        assert narrowed.outside["Q11424"] == ["Q4502142"]
        assert _meta(narrowed, classes=[FILM, WORK, HUMAN])["counts"]["parents_outside"] == 1

    def test_a_subclass_declares_only_what_it_adds(self):
        """``IMDb ID`` is declared by ``film`` and observed on ``human``, and
        neither inherits from the other, so both declare it."""
        module = _module()
        assert "imdb_id" in module.schemas["Film.schema.json"]["properties"]
        assert "imdb_id" in module.schemas["Human.schema.json"]["properties"]


class TestDatatypes:
    def test_a_quantity_is_a_number(self):
        slots = _module().schemas["Film.schema.json"]["properties"]
        assert slots["duration"]["type"] == "number"

    def test_a_time_is_a_date(self):
        slots = _module().schemas["Film.schema.json"]["properties"]
        assert slots["publication_date"] == {
            "type": "string",
            "format": "date",
            "title": "publication date",
        }

    def test_an_external_id_is_a_plain_string(self):
        """And not an IRI reference. An IMDb id looks like a reference and is
        not one, and coercing it would publish it as an IRI."""
        slots = _module().schemas["Film.schema.json"]["properties"]
        assert slots["imdb_id"]["type"] == "string"
        assert "format" not in slots["imdb_id"]
        assert _module().schemas["Film.schema.json"]["@context"][-1]["imdb_id"] == "wdt:P345"

    def test_a_datatype_with_no_scalar_form_costs_the_slot(self):
        """A globe coordinate is two numbers and flattening it to a string
        would invent a notation nothing reads back."""
        located = wd.WikidataProperty(pid="P625", label="coordinate location", datatype="globe-coordinate")
        module = _module(
            classes=[wd.WikidataClass(qid="Q5", label="human", declared=("P625",))],
            properties={"P625": located},
        )
        assert module.dropped == {"globe-coordinate": ["P625"]}
        assert "coordinate_location" not in module.schemas["Human.schema.json"]["properties"]


class TestItemRanges:
    def test_an_item_range_inside_the_module_becomes_a_ref(self):
        schema = _module().schemas["Film.schema.json"]
        assert schema["properties"]["director"]["$ref"] == "Human.schema.json"

    def test_the_embedding_term_carries_a_scoped_context(self):
        """Without one the embedded schema's terms resolve against the root
        and fall out of RDF entirely."""
        schema = _module().schemas["Film.schema.json"]
        assert schema["@context"][-1]["director"] == {"@id": "wdt:P57", "@context": "Human.schema.json"}

    def test_an_item_range_outside_the_module_becomes_an_iri_reference(self):
        schema = _module().schemas["Film.schema.json"]
        assert schema["properties"]["sex_or_gender"]["format"] == "iri-reference"
        assert schema["@context"][-1]["sex_or_gender"] == {"@id": "wdt:P21", "@type": "@id"}

    def test_an_embed_that_would_close_a_cycle_becomes_a_reference(self):
        """Neither JSON-LD processor resolves a cyclic scoped context, and a
        class embedding its own type cannot be framed apart from its value."""
        spouse = wd.WikidataProperty(pid="P26", label="spouse", datatype="wikibase-item", ranges=("Q5",))
        module = _module(
            classes=[wd.WikidataClass(qid="Q5", label="human", declared=("P26",))],
            properties={"P26": spouse},
        )
        assert module.schemas["Human.schema.json"]["properties"]["spouse"]["format"] == "iri-reference"
        assert module.demoted == {"Human.schema.json": ["Human.schema.json"]}


class TestPropertyRules:
    def test_a_class_declaring_p1963_is_described_by_what_it_declares(self):
        module = _module()
        assert module.rules["Q11424"] == wd.PROPERTIES_FOR_TYPE

    def test_a_class_without_p1963_falls_back_to_what_its_instances_state(self):
        module = _module()
        assert module.rules["Q5"] == "instances"

    def test_a_property_too_few_instances_state_is_not_kept(self):
        """One entity's peculiarity is not what the class is about: at the
        default floor a property needs a quarter of the sample."""
        slots = _module().schemas["Human.schema.json"]["properties"]
        assert "date_of_birth" in slots
        assert "name_in_native_language" not in slots

    def test_the_cap_is_spent_on_what_the_sample_uses_most(self):
        module = _module(cap=1)
        assert sorted(module.schemas["Human.schema.json"]["properties"]) == ["imdb_id", "type"]


class TestInstances:
    def test_an_instance_is_built_from_real_statements(self):
        instance = _module().instances["Film.instance.json"]
        assert instance["$schema"] == "Film.schema.json"
        assert instance["id"] == "wd:Q2875"
        assert instance["imdb_id"] == "tt0031381"
        assert instance["publication_date"] == "1939-12-15"

    def test_a_coarse_date_is_left_out_rather_than_written_as_a_date(self):
        coarse = wd.WikidataClass(
            qid="Q5",
            label="human",
            declared=("P569",),
            sampled=1,
            example=_example("Q1", P569=[{"kind": "time", "value": "+1886-00-00T00:00:00Z", "precision": 9}]),
        )
        module = _module(classes=[coarse], properties=PROPERTIES)
        assert "date_of_birth" not in module.instances["Human.instance.json"]

    def test_a_class_with_no_sampled_instance_gets_no_instance_file(self):
        """Nothing is invented to fill the gap. An abstract class has no
        instance, and a plausible one would be the only thing in the module
        that is not a Wikidata statement."""
        module = _module()
        assert "AudiovisualWork.instance.json" not in module.instances
        assert _meta(module)["counts"]["classes_without_instance"] == 2


class TestDegenerateInput:
    def test_a_class_with_no_description_still_emits_a_schema(self):
        """``human`` carries none here. A missing description is a missing
        field, not a reason to drop a class."""
        schema = _module().schemas["Human.schema.json"]
        assert "description" not in schema
        assert schema["title"] == "Human"
        assert schema["x-oold-iri"] == "wd:Q5"

    def test_a_class_with_no_english_label_is_filed_under_its_qid(self):
        module = _module(classes=[wd.WikidataClass(qid="Q999", label="")], properties={})
        assert "Q999.schema.json" in module.schemas

    def test_two_classes_with_one_label_both_keep_their_qid(self):
        """All of them, not the second onwards: which one keeps the bare name
        would otherwise depend on the order the module was built in."""
        twins = [wd.WikidataClass(qid="Q1", label="mercury"), wd.WikidataClass(qid="Q2", label="mercury")]
        assert sorted(_module(classes=twins, properties={}).schemas) == [
            "Mercury_Q1.schema.json",
            "Mercury_Q2.schema.json",
        ]

    def test_an_accented_label_keeps_its_letters(self):
        assert wd.property_name("AlloCiné film ID", "P1265") == "allocine_film_id"

    def test_a_label_that_would_take_a_reserved_term_keeps_its_identifier(self):
        assert wd.property_name("type", "P31") == "p31"


class TestDeterminism:
    def test_two_runs_over_one_cache_are_byte_identical(self, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        for out in (first, second):
            module = _module()
            wd.write_module(module, out, _meta(module))
        for path in sorted(first.iterdir()):
            assert path.read_bytes() == (second / path.name).read_bytes(), path.name

    def test_the_order_the_classes_arrive_in_does_not_reach_the_output(self, tmp_path):
        for index, order in enumerate(([FILM, WORK, ARTWORK, HUMAN], [HUMAN, ARTWORK, WORK, FILM])):
            module = _module(classes=order)
            wd.write_module(module, tmp_path / str(index), _meta(module))
        for path in sorted((tmp_path / "0").iterdir()):
            assert path.read_bytes() == (tmp_path / "1" / path.name).read_bytes(), path.name

    def test_the_module_is_digested_the_way_the_published_ones_are(self, tmp_path):
        """Over the schemas and not over every file, because ``_meta.json``
        carries the digest and cannot be part of it."""
        from oold_llm_bench.corpus.provenance import corpus_digest

        module = _module()
        recorded = wd.write_module(module, tmp_path, _meta(module))
        assert recorded["digest"] == str(corpus_digest(tmp_path, "*.schema.json"))
        assert recorded["counts"]["digested"] == len(module.schemas)


class TestAgainstTheValidator:
    """A module that is well-formed JSON and not a valid OO-LD document would
    pass every other test in this file."""

    def _validated(self, out):
        pipeline = pytest.importorskip("oold.validation.pipeline")
        module = _module()
        wd.write_module(module, out, _meta(module))
        return pipeline.validate_directory(out, pipeline.Options(offline=True, strict=True))

    def test_nothing_fails(self, tmp_path):
        report = self._validated(tmp_path)
        failed = [f"{check.id} on {check.target}: {check.message}" for check in report.checks if check.status == "fail"]
        assert report.fatal_error is None
        assert failed == []

    def test_nothing_faults(self, tmp_path):
        """A fault is a defect in the validator and says the document was
        neither condemned nor cleared, so it is not a pass."""
        report = self._validated(tmp_path)
        assert [check.id for check in report.checks if check.status == "fault"] == []

    def test_every_schema_and_every_instance_was_looked_at(self, tmp_path):
        """A report with nothing failing is only worth something if it ran."""
        report = self._validated(tmp_path)
        looked = {check.target for check in report.checks}
        assert "Film.schema.json" in looked
        assert "Film.instance.json" in looked

    def test_the_instances_round_trip_through_rdf(self, tmp_path):
        """The check that makes the instance half real: a property with no
        working ``@context`` term produces no triples and disappears."""
        report = self._validated(tmp_path)
        trips = [check for check in report.checks if check.id == "roundtrip.instance"]
        assert trips
        assert all(check.status == "ok" for check in trips)


def test_the_meta_records_what_the_module_cannot_say_about_itself(tmp_path):
    module = _module()
    recorded = wd.write_module(module, tmp_path, _meta(module))
    assert recorded["roots"] == ["Q11424"]
    assert recorded["depth"] == 1
    assert recorded["retrieved_at"] == "2026-10-07"
    assert recorded["revisions"]["Q11424"] == 1001
    assert recorded["property_rule"]["Q5"] == "instances"
    assert json.loads((tmp_path / "_meta.json").read_text(encoding="utf-8")) == recorded
