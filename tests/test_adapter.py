"""The join between a cell and a configured agent.

Two packages meet here, so these tests run a real
:class:`~oold.agent.extraction.ExtractionAgent` against a scripted client
instead of a stand-in. A stand-in would test the adapter against the shape
the adapter assumes, which proves nothing about the shape the agent has.
"""

import json

import pytest

from oold_llm_bench.corpus.quantities import QuantityKind, generate_task
from oold_llm_bench.results.record import Environment, ModelSpec
from oold_llm_bench.runner import (
    ANSWER_SCHEMA,
    Cell,
    Condition,
    ExperimentConfig,
    agent_factory,
    build_agent,
    build_request,
    run_experiment,
)

oold_agent = pytest.importorskip("oold.agent.extraction")

KINDS = [
    QuantityKind(name="Length", units=("meter",)),
    QuantityKind(name="Mass", units=("gram",)),
    QuantityKind(name="Duration", units=("second",)),
]
CATALOGUE = tuple(k.name for k in KINDS)


class ScriptedClient:
    """Replies with whatever it was given, and remembers what it was asked."""

    model = "scripted"

    def __init__(self, reply: str = "{}") -> None:
        self.reply = reply
        self.messages = None
        self.response_format = None

    def invoke(self, messages, *, response_format=None, strict=False):
        self.messages = messages
        self.response_format = response_format
        return type(
            "Reply",
            (),
            {"text": self.reply, "parsed": None, "usage": None, "raw": None},
        )()


def task(n: int = 1, catalogue: tuple[str, ...] | None = None):
    return generate_task(
        KINDS,
        task_id="t0",
        seed=1,
        n_entities=n,
        catalogue=CATALOGUE if catalogue is None else catalogue,
    )


def cell(arm: str = "schema-dump-catalog-flat-enforced", catalogue_size: int | None = None, n: int = 1) -> Cell:
    return Cell(
        condition=Condition(arm=arm, catalogue_size=catalogue_size),
        model=ModelSpec(model="m", provider_profile="openai", model_version="1"),
        task=task(n),
        repetition=1,
    )


def answer_for(record) -> str:
    return json.dumps({
        "entities": [
            {
                "type": i.class_path,
                "value": i.fields["value"].magnitude,
                "unit": i.fields["value"].unit,
            }
            for i in record.expected
        ]
    })


class TestWhatTheCellDecides:
    def test_the_arm_decides_the_enforcement(self):
        assert (
            build_agent(
                cell("schema-dump-catalog-not-enforced-gated"), ScriptedClient()
            ).enforcement.decode_constraint.value
            == "none"
        )
        assert (
            build_agent(cell("schema-dump-catalog-flat-enforced"), ScriptedClient()).enforcement.decode_constraint.value
            == "json_schema+enum"
        )

    def test_the_model_spec_decides_the_provider_profile(self):
        agent = build_agent(cell(), ScriptedClient())
        assert agent.profile.name == "openai"

    def test_an_unknown_arm_is_refused(self):
        with pytest.raises(KeyError, match="unknown arm"):
            build_agent(cell("A9"), ScriptedClient())

    def test_an_unknown_provider_profile_is_refused(self):
        broken = Cell(
            condition=Condition(arm="schema-dump-catalog-flat-enforced"),
            model=ModelSpec(model="m", provider_profile="nonesuch", model_version="1"),
            task=task(),
            repetition=1,
        )
        with pytest.raises(KeyError, match="unknown provider profile"):
            build_agent(broken, ScriptedClient())

    def test_nothing_is_read_from_the_environment(self, monkeypatch):
        """The predecessor configured a model by writing os.environ."""
        monkeypatch.setenv("CATALOGUE_MODE", "paths")
        monkeypatch.setenv("DECODE_CONSTRAINT", "none")
        agent = build_agent(cell("schema-dump-catalog-flat-enforced"), ScriptedClient())
        assert agent.enforcement.decode_constraint.value == "json_schema+enum"


class TestTheCatalogue:
    def test_the_whole_catalogue_is_offered_by_default(self):
        agent = build_agent(cell(), ScriptedClient())
        assert set(agent.enforcement.catalogue or ()) == set(CATALOGUE)

    def test_a_declared_size_trims_it(self):
        agent = build_agent(cell(catalogue_size=2), ScriptedClient())
        assert len(agent.enforcement.catalogue or ()) == 2

    def test_trimming_keeps_the_classes_the_task_needs(self):
        """Otherwise the sweep changes the question, not the catalogue size."""
        target = cell(catalogue_size=1)
        agent = build_agent(target, ScriptedClient())
        needed = {i.class_path for i in target.task.expected}
        assert needed <= set(agent.enforcement.catalogue or ())

    def test_a_catalogue_too_small_for_the_task_is_refused(self):
        with pytest.raises(ValueError, match="needs 2 classes"):
            build_agent(cell(catalogue_size=1, n=2), ScriptedClient())

    def test_an_arm_with_no_catalogue_ignores_the_size(self):
        agent = build_agent(cell("no-catalog-not-enforced", catalogue_size=2), ScriptedClient())
        assert agent.enforcement.catalogue is None


class TestTheUnitSlot:
    """A quantity corpus closes the unit slot as firmly as the class slot."""

    def units(self, size=None, pin=True):
        target = Cell(
            condition=Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=size, pin_units=pin),
            model=ModelSpec(model="m", provider_profile="openai", model_version="1"),
            task=task(),
            repetition=1,
        )
        return build_agent(target, ScriptedClient()).enforcement.unit_catalogue or ()

    def test_a_constrained_arm_closes_the_unit_slot(self):
        assert set(self.units()) == {"meter", "gram", "second"}

    def test_trimming_the_catalogue_trims_the_units(self):
        """Otherwise a small catalogue reports a closed slot it has not closed."""
        assert len(self.units(size=1)) < len(self.units())

    def test_the_units_belong_to_the_offered_classes(self):
        target = Cell(
            condition=Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=1),
            model=ModelSpec(model="m", provider_profile="openai", model_version="1"),
            task=task(),
            repetition=1,
        )
        agent = build_agent(target, ScriptedClient())
        offered = set(agent.enforcement.catalogue or ())
        per_class = target.task.unit_catalogue or {}
        allowed = {u for name in offered for u in per_class.get(name, ())}
        assert set(agent.enforcement.unit_catalogue or ()) == allowed

    def test_the_slot_can_be_left_open_as_a_condition(self):
        assert self.units(pin=False) == ()

    def test_an_arm_with_no_catalogue_closes_nothing(self):
        target = Cell(
            condition=Condition(arm="no-catalog-not-enforced"),
            model=ModelSpec(model="m", provider_profile="openai", model_version="1"),
            task=task(),
            repetition=1,
        )
        assert build_agent(target, ScriptedClient()).enforcement.unit_catalogue is None


class TestTheRequest:
    def test_the_document_is_carried_over(self):
        target = cell()
        assert build_request(target).document == target.task.document

    def test_the_schema_is_offered_to_every_arm(self):
        """Which arm shows it is the enforcement's decision, not the adapter's."""
        assert build_request(cell("no-catalog-not-enforced")).schema is ANSWER_SCHEMA

    def test_the_answer_schema_has_a_class_slot_to_pin(self):
        properties = ANSWER_SCHEMA["properties"]["entities"]["items"]["properties"]
        assert "type" in properties


class TestThroughTheWholePath:
    """A cell, a real agent, a scripted reply, a score."""

    def run(self, arm: str, reply_for):
        grid = ExperimentConfig(
            name="adapter",
            conditions=[Condition(arm=arm)],
            models=[ModelSpec(model="m", provider_profile="openai", model_version="1")],
            tasks=[task()],
            runs_per_cell=1,
        )
        clients: list[ScriptedClient] = []

        def client_for(c: Cell) -> ScriptedClient:
            client = ScriptedClient(reply_for(c.task))
            clients.append(client)
            return client

        run = run_experiment(
            grid,
            agent_factory(client_for),
            Environment(benchmark_version="0.1.0", benchmark_sha="abc"),
            make_request=build_request,
        )
        return run, clients[0]

    def test_a_correct_answer_scores_one(self):
        run, _ = self.run("schema-dump-catalog-flat-enforced", answer_for)
        assert run.mean_primary("schema-dump-catalog-flat-enforced/unit/consensus") == pytest.approx(1.0)

    def test_a_constrained_arm_sends_the_schema_to_the_provider(self):
        _, client = self.run("schema-dump-catalog-flat-enforced", answer_for)
        assert client.response_format is not None

    def test_an_unconstrained_arm_sends_no_schema_to_the_provider(self):
        _, client = self.run("schema-dump-catalog-not-enforced-gated", answer_for)
        assert client.response_format is None

    def test_a_constrained_arm_pins_the_class_slot_to_the_catalogue(self):
        _, client = self.run("schema-dump-catalog-flat-enforced", answer_for)
        pinned = client.response_format["properties"]["entities"]["items"]["properties"]["type"]
        assert set(pinned["enum"]) == set(CATALOGUE)

    def test_a_constrained_arm_pins_the_unit_slot_too(self):
        """The corpus closes it, so A2 closes it."""
        _, client = self.run("schema-dump-catalog-flat-enforced", answer_for)
        pinned = client.response_format["properties"]["entities"]["items"]["properties"]["unit"]
        assert set(pinned["enum"]) == {"meter", "gram", "second"}

    def test_the_prompt_actually_sent_is_hashed_into_the_record(self):
        """Not the document. Two arms send different instructions."""
        a1, _ = self.run("schema-dump-catalog-not-enforced-gated", answer_for)
        a2, _ = self.run("schema-dump-catalog-flat-enforced", answer_for)
        assert a1.outcomes[0].record.prompt_hash != a2.outcomes[0].record.prompt_hash

    def test_the_schema_actually_sent_is_hashed_into_the_record(self):
        run, _ = self.run("schema-dump-catalog-flat-enforced", answer_for)
        assert run.outcomes[0].record.schema_hash is not None

    def test_a0_json_is_shown_no_schema_at_all(self):
        _, client = self.run("no-catalog-not-enforced", answer_for)
        system = client.messages[0].content
        assert "properties" not in system

    def test_a1_is_shown_the_schema_in_the_prompt(self):
        _, client = self.run("schema-dump-catalog-not-enforced-gated", answer_for)
        system = client.messages[0].content
        assert "QuantityValues" in system

    def test_a_refusal_scores_zero_without_crashing(self):
        run, _ = self.run("schema-dump-catalog-flat-enforced", lambda t: "I cannot help with that.")
        assert run.outcomes[0].ok
        assert run.mean_primary("schema-dump-catalog-flat-enforced/unit/consensus") == 0.0


class TestTheCatalogueOrder:
    """The ordering must not carry the answer."""

    def cells(self, n=12):
        out = []
        for i in range(n):
            record = generate_task(KINDS, task_id=f"o{i}", seed=i, n_entities=1, catalogue=CATALOGUE)
            out.append(
                Cell(
                    condition=Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=3),
                    model=ModelSpec(model="m", provider_profile="openai", model_version="1"),
                    task=record,
                    repetition=1,
                )
            )
        return out

    def test_the_answer_is_not_always_first(self):
        """Keeping the needed class at the front puts the answer at index 0 of
        every catalogue and every enum."""
        positions = set()
        for cell in self.cells():
            offered = build_agent(cell, ScriptedClient()).enforcement.catalogue or ()
            positions.add(offered.index(cell.task.expected[0].class_path))
        assert positions != {0}

    def test_the_order_is_stable_for_one_task(self):
        """Two runs of one config have to be comparable."""
        cell = self.cells(1)[0]
        first = build_agent(cell, ScriptedClient()).enforcement.catalogue
        second = build_agent(cell, ScriptedClient()).enforcement.catalogue
        assert first == second

    def test_the_needed_class_is_still_offered(self):
        for cell in self.cells():
            offered = build_agent(cell, ScriptedClient()).enforcement.catalogue or ()
            assert cell.task.expected[0].class_path in offered

    def test_an_untrimmed_catalogue_is_shuffled_too(self):
        cell = Cell(
            condition=Condition(arm="schema-dump-catalog-flat-enforced"),
            model=ModelSpec(model="m", provider_profile="openai", model_version="1"),
            task=task(),
            repetition=1,
        )
        offered = build_agent(cell, ScriptedClient()).enforcement.catalogue or ()
        assert set(offered) == set(CATALOGUE)


class TestTheOrchestrationCondition:
    """Orchestration is a declared axis, not a code path."""

    def cell_for(self, orchestration="single_shot", k=None):
        return Cell(
            condition=Condition(
                arm="schema-dump-catalog-enforced", catalogue_size=2, orchestration=orchestration, shortlist_k=k
            ),
            model=ModelSpec(model="m", provider_profile="openai", model_version="1"),
            task=task(),
            repetition=1,
        )

    def test_single_shot_is_the_default(self):
        assert Condition(arm="schema-dump-catalog-flat-enforced").orchestration == "single_shot"

    def test_the_orchestration_reaches_the_agent(self):
        agent = build_agent(self.cell_for("select_then_fill", 3), ScriptedClient())
        assert agent.orchestration.value == "select_then_fill"
        assert agent.shortlist_k == 3

    def test_select_then_fill_without_a_k_is_refused(self):
        """A shortlist the size of the catalogue gains the second step nothing."""
        with pytest.raises(ValueError, match="needs a shortlist_k"):
            Condition(arm="schema-dump-catalog-enforced", orchestration="select_then_fill")

    def test_a_shortlist_of_nothing_is_refused(self):
        with pytest.raises(ValueError, match="no class to fill"):
            Condition(arm="schema-dump-catalog-enforced", orchestration="select_then_fill", shortlist_k=0)

    def test_an_unknown_orchestration_is_refused(self):
        with pytest.raises(ValueError, match="unknown orchestration"):
            Condition(arm="schema-dump-catalog-enforced", orchestration="telepathy")

    def test_the_condition_key_names_the_orchestration(self):
        """Or the table pools two orchestrations into one row."""
        key = Condition(arm="schema-dump-catalog-enforced", orchestration="select_then_fill", shortlist_k=3).key
        assert "select_then_fill(k3)" in key

    def test_branches_are_built_from_the_units_each_class_admits(self):
        request = build_request(self.cell_for())
        assert request.branches is not None
        assert set(request.branches) <= set(CATALOGUE)
        assert request.branches["Length"]["unit"]["enum"] == ["meter"]


class TestACorpusThatBringsItsOwnShape:
    """A quantity is a class, a magnitude and a unit. A schema.org entity is a
    class and whatever properties that class defines. One hardcoded shape
    fits the first and silently misdescribes the second."""

    def task_with_shape(self):
        base = task()
        return base.model_copy(
            update={
                "answer_schema": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "onlyLength": {"type": "string"},
                                    "onlyMass": {"type": "string"},
                                    "onlyElsewhere": {"type": "string"},
                                    "shared": {"type": "number"},
                                },
                                "required": ["type"],
                            },
                        }
                    },
                },
                "branches": {
                    "Length": {"onlyLength": {"type": "string"}},
                    "Mass": {"onlyMass": {"type": "string"}},
                    "Duration": {"onlyElsewhere": {"type": "string"}},
                },
            }
        )

    def cell_for(self, arm, size):
        return Cell(
            condition=Condition(arm=arm, catalogue_size=size, pin_units=False),
            model=ModelSpec(model="m", provider_profile="openai", model_version="1"),
            task=self.task_with_shape(),
            repetition=1,
        )

    def entity_properties(self, cell):
        schema = build_request(cell).schema
        assert schema is not None
        return schema["properties"]["entities"]["items"]["properties"]

    def test_the_task_schema_is_used_instead_of_the_quantity_one(self):
        assert "onlyLength" in self.entity_properties(self.cell_for("schema-dump-catalog-flat-enforced", 3))

    def test_the_task_branches_are_used_instead_of_the_units(self):
        request = build_request(self.cell_for("schema-dump-catalog-enforced", 3))
        assert set(request.branches or {}) == {"Length", "Mass", "Duration"}

    def test_a_property_no_offered_class_defines_is_trimmed_away(self):
        """Otherwise an unconstrained arm may answer with a property that
        belongs to a class it was never offered."""
        cell = self.cell_for("schema-dump-catalog-flat-enforced", 1)
        offered = set(build_agent(cell, ScriptedClient()).enforcement.catalogue or ())
        properties = self.entity_properties(cell)
        for name, prop in (("Length", "onlyLength"), ("Mass", "onlyMass"), ("Duration", "onlyElsewhere")):
            assert (prop in properties) is (name in offered)

    def test_a_property_no_branch_defines_survives_as_structural(self):
        """The class slot, and anything every class shares."""
        properties = self.entity_properties(self.cell_for("schema-dump-catalog-flat-enforced", 1))
        assert "type" in properties
        assert "shared" in properties

    def test_the_union_does_not_copy_every_property_into_every_branch(self):
        """A union that did came to 1.5 MB on a 60 kB schema."""
        client = ScriptedClient()
        build_agent(self.cell_for("schema-dump-catalog-enforced", 3), client).run(
            build_request(self.cell_for("schema-dump-catalog-enforced", 3))
        )
        sent = client.response_format
        assert sent is not None
        items = sent["properties"]["entities"]["items"]
        for branch in items["anyOf"]:
            assert len(branch["properties"]) <= 3

    def test_a_task_without_its_own_shape_still_gets_the_quantity_one(self):
        cell = Cell(
            condition=Condition(arm="schema-dump-catalog-flat-enforced", catalogue_size=2),
            model=ModelSpec(model="m", provider_profile="openai", model_version="1"),
            task=task(),
            repetition=1,
        )
        assert build_request(cell).schema is ANSWER_SCHEMA
