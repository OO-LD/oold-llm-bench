"""The tuning corpus, checked without a provider.

Everything a wasted job would come from is offline: the split, the classes an
example is allowed to draw, the prompt the example carries, and whether the
answer survives the extractor and the grader. None of it needs an account, so
none of it is a reason to find out after the bill.
"""

import json
import random
from typing import ClassVar, cast

import pytest

from oold_llm_bench.corpus.schemaorg import Kind, SchemaClass, Slot
from oold_llm_bench.finetune import (
    LEAN_ARM,
    ClassSplit,
    DataPlane,
    JobRequest,
    answer_text,
    build_corpus,
    lean_condition,
    load_split,
    offered_catalogue,
    partition,
    round_trip_score,
    write_jsonl,
)
from oold_llm_bench.finetune.azure import data_plane_endpoint
from oold_llm_bench.finetune.dataset import CATALOGUE_SIZE, TRAIN_SEED_BASE
from oold_llm_bench.finetune.split import QUANTITIES_SPLIT_PATH, SPLIT_PATH
from oold_llm_bench.runner import Cell, build_enforcement, build_request

pytest.importorskip("oold.agent.prompts")


def _class(name: str, parent: str = "Thing") -> SchemaClass:
    """A class with enough distinct declared slots to describe."""
    return SchemaClass(
        name=name,
        parents=(parent,),
        label=name,
        description=f"A {name}.",
        slots=(
            Slot(name=f"{name}Code", kind=Kind.TEXT),
            Slot(name=f"{name}Count", kind=Kind.INTEGER),
            Slot(name=f"{name}Date", kind=Kind.DATE),
            Slot(name=f"{name}Flag", kind=Kind.BOOLEAN),
            Slot(name=f"{name}Grade", kind=Kind.ENUM, choices=("High", "Low")),
            Slot(name="name", kind=Kind.TEXT, inherited=True),
        ),
    )


POOL = [_class(f"Kind{index:02d}") for index in range(20)]
SPLIT = partition((cls.name for cls in POOL), seed=7)


def test_partition_is_deterministic_and_disjoint():
    again = partition((cls.name for cls in POOL), seed=7)
    assert again.train == SPLIT.train
    assert again.heldout == SPLIT.heldout
    assert not set(SPLIT.train) & set(SPLIT.heldout)
    assert set(SPLIT.names) == {cls.name for cls in POOL}
    assert abs(len(SPLIT.train) - len(SPLIT.heldout)) <= 1


def test_partition_does_not_depend_on_input_order():
    shuffled = partition(reversed([cls.name for cls in POOL]), seed=7)
    assert shuffled.train == SPLIT.train


def test_a_different_seed_gives_a_different_split():
    assert partition((cls.name for cls in POOL), seed=8).train != SPLIT.train


def test_both_halves_must_be_populated():
    with pytest.raises(ValueError, match="non-empty"):
        ClassSplit(train=("A",), heldout=())
    with pytest.raises(ValueError, match="both halves"):
        ClassSplit(train=("A", "B"), heldout=("B",))


def test_the_committed_split_round_trips():
    found = load_split(QUANTITIES_SPLIT_PATH, expect_corpus="quantities")
    assert not set(found.train) & set(found.heldout)
    assert len(found.train) + len(found.heldout) == len(found.names)
    assert found.train and found.heldout


def test_a_split_drawn_on_another_corpus_is_refused():
    """The check that was missing when it mattered.

    ``schemaorg_split.json`` was byte-identical to ``quantities_split.json``
    and held out ``AbsoluteActivity`` where a reader expected ``Person``.
    Nothing crashed, so the only way to find it was to read the file.
    """
    with pytest.raises(ValueError, match="not 'schemaorg'"):
        load_split(QUANTITIES_SPLIT_PATH, expect_corpus="schemaorg")


def test_the_schemaorg_split_is_absent_until_it_is_built():
    """Absent on purpose, and the test says so rather than the filesystem.

    Deleted 2026-10-03 because the file that occupied this path was a copy of
    the quantity split. A reader finding nothing here asks where it went; a
    reader finding the wrong one asks nothing at all.
    """
    assert not SPLIT_PATH.exists(), (
        "a schema.org split exists again: check its corpus field names schemaorg before anything trains against it"
    )


def test_the_offered_catalogue_carries_both_halves():
    offered = offered_catalogue([SPLIT.train[0]], SPLIT, random.Random(1), size=10)  # noqa: S311
    assert len(offered) == 10
    assert SPLIT.train[0] in offered
    assert set(offered) & set(SPLIT.heldout)
    assert set(offered) & set(SPLIT.train)


def test_a_catalogue_smaller_than_the_answer_is_refused():
    with pytest.raises(ValueError, match="cannot be offered"):
        offered_catalogue(["A", "B", "C"], SPLIT, random.Random(1), size=2)  # noqa: S311


def _draw(half=None, catalogue_size: int = 8):
    """A schema.org source over one half of the split."""
    from oold_llm_bench.finetune.evaluate import Half, answer_names
    from oold_llm_bench.finetune.sources import schemaorg_source

    wanted = answer_names(SPLIT, half or Half.TRAIN)
    return schemaorg_source(POOL, [c for c in POOL if c.name in wanted], SPLIT, catalogue_size=catalogue_size)


def _corpus(count: int = 6, draw=None, **kwargs):
    return build_corpus(
        SPLIT,
        draw or _draw(),
        name="t",
        count=count,
        seed_base=TRAIN_SEED_BASE,
        **kwargs,
    )


def test_training_answers_are_never_held_out():
    corpus = _corpus()
    assert len(corpus.examples) == 6
    assert set(corpus.classes) <= set(SPLIT.train)
    assert not set(corpus.classes) & set(SPLIT.heldout)


def test_every_assistant_turn_scores_one():
    """The check the whole job rests on.

    An answer the grader does not score is an answer the model is trained to
    give and then marked down for.
    """
    corpus = _corpus(count=12)
    for task, example in zip(corpus.tasks, corpus.examples, strict=True):
        turn = example["messages"][-1]
        assert turn["role"] == "assistant"
        assert round_trip_score(task, turn["content"]) == 1.0
    assert all(turn["role"] in {"system", "user", "assistant"} for e in corpus.examples for turn in e["messages"])


def test_a_wrong_answer_does_not_score_one():
    """The round trip has to be able to fail, or it checks nothing."""
    corpus = _corpus(count=1)
    task = corpus.tasks[0]
    payload = json.loads(corpus.examples[0]["messages"][-1]["content"])
    payload["entities"][0] = {"type": "NotAClass", "name": "wrong"}
    assert round_trip_score(task, json.dumps(payload)) < 1.0


def test_unparseable_output_scores_nothing():
    corpus = _corpus(count=1)
    assert round_trip_score(corpus.tasks[0], "not json at all") == 0.0


@pytest.mark.parametrize("described", [True, False])
def test_the_prompt_is_the_one_an_evaluation_sends(described):
    """Character for character, from the same two functions a run calls.

    Both arms, because the described and the bare catalogue are two training
    sets and a guarantee that holds for one of them is not the guarantee.
    """
    from oold.agent.prompts import build_messages

    from oold_llm_bench.finetune.dataset import _PROMPT_MODEL

    condition = lean_condition(describe_catalogue=described)
    corpus = _corpus(count=1, condition=condition)
    task, example = corpus.tasks[0], corpus.examples[0]
    cell = Cell(condition=condition, model=_PROMPT_MODEL, task=task, repetition=1)
    expected = build_messages(build_request(cell), build_enforcement(cell))

    assert [turn["role"] for turn in example["messages"]] == ["system", "user", "assistant"]
    assert example["messages"][0]["content"] == expected[0].content
    assert example["messages"][1]["content"] == expected[1].content
    assert example["messages"][1]["content"] == task.document


def test_the_lean_prompt_shows_the_catalogue_and_withholds_the_schema():
    corpus = _corpus(count=1)
    task, example = corpus.tasks[0], corpus.examples[0]
    system = example["messages"][0]["content"]
    assert "Choose the class of each entity from this list" in system
    assert "Each entity must conform to this schema:" not in system
    for name in task.catalogue:
        assert name in system


def test_the_bare_prompt_lists_identifiers_and_nothing_else():
    """The second arm: the names are offered, what they mean is not."""
    described = _corpus(count=1)
    bare = _corpus(count=1, condition=lean_condition(describe_catalogue=False))
    task = bare.tasks[0]
    system = bare.examples[0]["messages"][0]["content"]

    for name in task.catalogue:
        assert f"- {name}" in system
    assert task.catalogue_text
    for rendered in task.catalogue_text.values():
        for line in rendered.splitlines()[1:]:
            assert line not in system
    assert len(system) < len(described.examples[0]["messages"][0]["content"])


def test_the_two_arms_differ_only_in_the_system_message():
    """The paired comparison rests on this: same document, same answer."""
    described = _corpus(count=4)
    bare = _corpus(count=4, condition=lean_condition(describe_catalogue=False))
    assert [t.id for t in described.tasks] == [t.id for t in bare.tasks]
    for left, right in zip(described.examples, bare.examples, strict=True):
        assert left["messages"][1] == right["messages"][1]
        assert left["messages"][2] == right["messages"][2]
        assert left["messages"][0] != right["messages"][0]


def test_the_lean_arm_differs_from_its_baseline_only_in_the_schema():
    from dataclasses import replace

    from oold.agent.enforcement import ARMS

    from oold_llm_bench.finetune import BASELINE_ARM, register_lean_arm

    lean = register_lean_arm()
    assert lean is ARMS[LEAN_ARM]
    assert lean.schema_in_prompt is False
    assert replace(lean, schema_in_prompt=True) == ARMS[BASELINE_ARM]


def test_the_answer_is_the_ground_truth_in_the_extractor_shape():
    task = _corpus(count=1).tasks[0]
    payload = json.loads(answer_text(task))
    assert list(payload) == ["entities"]
    assert payload["entities"][0]["type"] == task.expected[0].class_path
    assert round_trip_score(task, answer_text(task)) == 1.0


def test_training_and_validation_documents_do_not_overlap():
    seen: set[str] = set()
    first = _corpus(count=5, seen=seen)
    second = build_corpus(SPLIT, _draw(), name="v", count=5, seed_base=900000, seen=seen)
    documents = {e["messages"][1]["content"] for e in first.examples}
    assert not documents & {e["messages"][1]["content"] for e in second.examples}


def test_written_examples_are_one_json_object_per_line(tmp_path):
    corpus = _corpus(count=3)
    path = write_jsonl(tmp_path / "train.jsonl", corpus.examples)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == 3
    assert all(list(json.loads(line)) == ["messages"] for line in lines)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_the_corpus_reports_what_it_dropped():
    corpus = _corpus(count=4)
    described = corpus.describe()
    assert described["n_examples"] == 4
    assert described["n_rejected"] == len(corpus.rejected)
    assert described["last_seed"] >= TRAIN_SEED_BASE


def test_a_half_that_names_nothing_in_the_pool_is_refused():
    from oold_llm_bench.finetune.evaluate import Half, answer_pool

    with pytest.raises(ValueError, match="names nothing the pool has"):
        answer_pool([], SPLIT, Half.TRAIN, key=lambda c: c.name)


def test_the_job_body_names_the_developer_tier():
    body = JobRequest(
        model="gpt-4.1-mini-2025-04-14",
        training_file="file-a",
        validation_file="file-b",
        suffix="oold",
        n_epochs=2,
        seed=11,
    ).payload()
    assert body["trainingType"] == "developerTier"
    assert body["hyperparameters"] == {"n_epochs": 2}
    assert body["model"] == "gpt-4.1-mini-2025-04-14"
    assert body["training_file"] == "file-a"
    assert body["validation_file"] == "file-b"


def test_the_job_body_omits_what_it_was_not_given():
    body = JobRequest(model="m", training_file="file-a", tier=None).payload()
    assert set(body) == {"model", "training_file"}


def test_a_job_summary_says_which_tier_it_ran_on():
    """The tier is the whole cost difference, so a poll that omits it hides one."""
    from oold_llm_bench.finetune import summarise

    found = summarise({"id": "ftjob-1", "status": "running", "trainingType": "developerTier"})
    assert found["trainingType"] == "developerTier"
    assert found["fine_tuned_model"] is None


class _RecordingPlane(DataPlane):
    """A data plane that records the call instead of sending it."""

    sent: ClassVar[dict] = {}

    def post(self, path, payload=None):
        type(self).sent = {"path": path, "payload": payload}
        return {"id": "ftjob-1", "status": "cancelled"}


def test_cancel_posts_to_the_job_it_was_given():
    """No default job id: the one irreversible call names its target."""
    plane = _RecordingPlane(endpoint="https://a.openai.azure.com", api_key="k")
    assert plane.cancel("ftjob-1")["status"] == "cancelled"
    assert _RecordingPlane.sent == {"path": "fine_tuning/jobs/ftjob-1/cancel", "payload": None}


def test_the_tuning_endpoint_is_derived_from_the_inference_one():
    environ = {"OOLD_BENCH_ENDPOINT": "https://account-name.services.ai.azure.com/models"}
    assert data_plane_endpoint(environ) == "https://account-name.openai.azure.com"


def test_an_explicit_tuning_endpoint_wins():
    environ = {
        "OOLD_BENCH_ENDPOINT": "https://a.services.ai.azure.com/models",
        "OOLD_BENCH_FINETUNE_ENDPOINT": "https://b.openai.azure.com/",
    }
    assert data_plane_endpoint(environ) == "https://b.openai.azure.com"


def test_the_data_plane_never_describes_its_key():
    plane = DataPlane(endpoint="https://a.openai.azure.com", api_key="secret")
    assert "secret" not in json.dumps(plane.describe())
    assert plane.describe()["host"] == "a.openai.azure.com"


def test_catalogue_size_matches_the_grid():
    assert CATALOGUE_SIZE == 25
    assert lean_condition().arm == LEAN_ARM


def test_evaluation_seeds_are_clear_of_tuning():
    """The one collision that would quietly contaminate the result."""
    from oold_llm_bench.finetune.dataset import VALIDATION_SEED_BASE
    from oold_llm_bench.finetune.evaluate import EVAL_SEED_BASE

    assert EVAL_SEED_BASE > VALIDATION_SEED_BASE
    assert EVAL_SEED_BASE > TRAIN_SEED_BASE
    assert EVAL_SEED_BASE > 7000 + 120


def test_each_half_answers_only_its_own_classes():
    from oold_llm_bench.finetune.evaluate import Half, evaluation_tasks

    for half, wanted in ((Half.TRAIN, SPLIT.train), (Half.HELDOUT, SPLIT.heldout)):
        tasks = evaluation_tasks(_draw(half), half, count=8)
        assert len(tasks) == 8
        answered = {i.class_path for t in tasks for i in t.expected}
        assert answered <= set(wanted)


def test_every_evaluation_catalogue_offers_both_halves():
    """Without this the held-out score is zero by construction."""
    from oold_llm_bench.finetune.evaluate import Half, evaluation_tasks

    for half in Half:
        for task in evaluation_tasks(_draw(half), half, count=6):
            offered = set(task.catalogue or ())
            assert offered & set(SPLIT.train)
            assert offered & set(SPLIT.heldout)
            assert {i.class_path for i in task.expected} <= offered


def test_the_two_halves_draw_different_documents():
    from oold_llm_bench.finetune.evaluate import Half, evaluation_tasks

    train = {t.document for t in evaluation_tasks(_draw(Half.TRAIN), Half.TRAIN, count=8)}
    held = {t.document for t in evaluation_tasks(_draw(Half.HELDOUT), Half.HELDOUT, count=8)}
    assert not train & held


def test_the_three_prompts_do_not_share_a_condition_key():
    """Two prompts under one key would be averaged into one column."""
    from oold_llm_bench.finetune.evaluate import pairings

    keys = {p.prompt: p.condition.key for p in pairings("b", "l", "r")}
    assert len({p.prompt: p.condition.key for p in pairings("b", "l", "r")}.values()) == 3
    assert keys["lean"] != keys["bare"]
    assert keys["full"] != keys["lean"]


def test_the_base_column_is_optional_and_the_tuned_ones_are_not():
    from oold_llm_bench.finetune.evaluate import pairings

    assert [p.label for p in pairings(None, "l", "r")] == ["tuned-lean+lean", "tuned-bare+bare"]
    assert len(pairings("b", "l", "r")) == 5


def test_each_tuned_model_is_read_at_the_prompt_it_was_tuned_on():
    from oold_llm_bench.finetune.evaluate import pairings

    tuned = {p.label: p for p in pairings("b", "l", "r") if p.tuned}
    assert tuned["tuned-lean+lean"].condition.describe_catalogue is True
    assert tuned["tuned-bare+bare"].condition.describe_catalogue is False


def test_a_tuned_entry_names_no_account_of_its_own():
    from oold_llm_bench.finetune.evaluate import tuned_entry

    entry = tuned_entry("some-deployment")
    assert entry.model == "some-deployment"
    assert entry.transport == "openai"
    assert entry.system_role is True


def test_the_ceiling_reports_what_the_grader_can_score():
    from oold_llm_bench.finetune.evaluate import Half, ceiling, evaluation_tasks

    tasks = evaluation_tasks(_draw(Half.TRAIN), Half.TRAIN, count=6)
    assert ceiling(tasks) == 1.0
    assert ceiling([]) == 0.0


QUDT_SPLIT = ClassSplit(train=("Length", "Mass"), heldout=("Force", "Power"), corpus="quantities")


def test_a_split_records_which_corpus_it_covers():
    """Two corpora, two files. One read against the other holds out nothing."""
    from oold_llm_bench.finetune.split import QUANTITIES_SPLIT_PATH

    assert QUDT_SPLIT.corpus == "quantities"
    assert partition(("a", "b", "c", "d"), corpus="quantities").corpus == "quantities"
    assert partition(("a", "b", "c", "d")).corpus == "schemaorg"
    assert load_split(QUANTITIES_SPLIT_PATH).corpus == "quantities"


def test_the_committed_quantities_split_is_disjoint_and_covers_the_pool():
    from oold_llm_bench.finetune.split import QUANTITIES_SPLIT_PATH

    found = load_split(QUANTITIES_SPLIT_PATH)
    assert not set(found.train) & set(found.heldout)
    assert len(found.train) + len(found.heldout) == len(found.names)
    assert abs(len(found.train) - len(found.heldout)) <= 1


def test_a_quantity_answer_is_written_as_a_magnitude_beside_a_unit():
    """The shape the extractor folds back into one quantity."""
    from oold_llm_bench.grading.triples import Quantity
    from oold_llm_bench.tasks.models import CorpusRef, ExpectedInstance, Source, Split, TaskRecord

    task = TaskRecord(
        id="q",
        document="7 bar",
        expected=[
            ExpectedInstance(key="q1", class_path="Pressure", fields={"value": Quantity(magnitude=7.0, unit="bar")})
        ],
        corpus=CorpusRef(source=Source.SYNTHETIC, document_id="q", content_hash="0" * 64),
        split=Split.DEV,
    )
    payload = json.loads(answer_text(task))
    assert payload["entities"] == [{"type": "Pressure", "value": 7.0, "unit": "bar"}]
    assert round_trip_score(task, answer_text(task)) == 1.0


def test_the_quantities_signal_never_names_the_class():
    """The defect that made schema.org unusable, checked rather than assumed."""
    from oold_llm_bench.corpus.signals import Signal, Vocabulary
    from oold_llm_bench.finetune.sources import QUDT_SIGNAL, QUDT_VOCABULARY

    assert QUDT_SIGNAL is Signal.ELUCIDATION
    assert QUDT_SIGNAL is not Signal.NAMED
    assert QUDT_VOCABULARY is Vocabulary.CONSENSUS


def test_a_run_counts_how_its_failed_cells_failed():
    from oold_llm_bench.finetune.dataset import _PROMPT_MODEL
    from oold_llm_bench.runner import ExperimentConfig
    from oold_llm_bench.runner.execute import CellOutcome, ExperimentRun

    tasks = _corpus(count=2).tasks
    condition = lean_condition()
    config = ExperimentConfig(name="t", conditions=[condition], models=[_PROMPT_MODEL], tasks=tasks)
    cell = Cell(condition=condition, model=_PROMPT_MODEL, task=tasks[0], repetition=1)
    run = ExperimentRun(config=config)
    run.outcomes = [
        CellOutcome(cell=cell, error="OpenAIRateLimitError: 429"),
        CellOutcome(cell=cell, error="OpenAIRateLimitError: 429"),
        CellOutcome(cell=cell, error="TimeoutError: slow"),
    ]
    assert run.error_kinds() == {"OpenAIRateLimitError": 2, "TimeoutError": 1}
    assert run.describe()["error_kinds"] == {"OpenAIRateLimitError": 2, "TimeoutError": 1}


def test_a_written_document_narrows_the_pool_on_both_sides():
    """A kind whose every unit has no symbol cannot be written down."""
    from oold_llm_bench.corpus.quantities import Notation, QuantityKind
    from oold_llm_bench.corpus.signals import SignalData
    from oold_llm_bench.finetune.sources import quantities_pool

    class _Table:
        def written_units(self, name, units):
            return [u for u in units if u != "unitless"]

    class _Data:
        def identifier(self, name, vocabulary):
            return name

        def supports(self, name, signal):
            return True

        elucidations: ClassVar[dict] = {"Length": "how long", "Ratio": "a ratio"}
        descriptions: ClassVar[dict] = {}
        synonyms: ClassVar[dict] = {}
        labels: ClassVar[dict] = {}

    kinds = [
        QuantityKind(name="Length", units=("meter",)),
        QuantityKind(name="Ratio", units=("unitless",)),
    ]
    data = _Data()
    signals = cast("SignalData", data)
    canonical = {k.name for k in quantities_pool(kinds, signals)}
    written = {k.name for k in quantities_pool(kinds, signals, Notation.WRITTEN)}
    assert "Ratio" in canonical
    assert written == {"Length"}


def test_the_committed_split_still_covers_the_written_pool_on_both_halves():
    """Reusing the split needs both halves to survive the narrowing."""
    found = load_split(__import__("oold_llm_bench.finetune.split", fromlist=["x"]).QUANTITIES_SPLIT_PATH)
    assert len(found.train) + len(found.heldout) == 323
    assert abs(len(found.train) - len(found.heldout)) <= 1


class TestTheTrainingConditionIsCompared:
    """``training_match`` existed with no caller and no test, and its one
    live axis could not fire: ``Condition`` has no ``corpus`` attribute, so
    the key that matters was skipped rather than reported."""

    @staticmethod
    def _condition(pin_units: bool = True, describe_catalogue: bool = True):
        from oold_llm_bench.runner import Condition

        return Condition(
            arm="schema-dump-catalog-flat-enforced",
            signal="named",
            vocabulary="consensus",
            pin_units=pin_units,
            describe_catalogue=describe_catalogue,
        )

    def test_a_base_model_is_always_comparable(self):
        from oold_llm_bench.runner.training_match import match_of

        found = match_of(None, self._condition())
        assert found.status == "base"
        assert found.comparable

    def test_a_matching_condition_is_matched(self):
        from oold_llm_bench.runner.training_match import match_of

        trained = {"signal": "named", "pin_units": False, "describe_catalogue": False}
        found = match_of(trained, self._condition(pin_units=False, describe_catalogue=False))
        assert found.status == "matched"
        assert found.comparable

    def test_a_differing_key_is_named_and_refuses_pooling(self):
        from oold_llm_bench.runner.training_match import match_of

        trained = {"signal": "named", "pin_units": False, "describe_catalogue": False}
        found = match_of(trained, self._condition(pin_units=True, describe_catalogue=True))
        assert found.status == "mismatched"
        assert found.differing == ("describe_catalogue", "pin_units")
        assert not found.comparable

    def test_an_axis_the_condition_cannot_express_is_reported_not_skipped(self):
        """``corpus`` is the axis the module was written for and the one it
        silently ignored, because ``Condition`` has no such attribute."""
        from oold_llm_bench.runner.training_match import match_of

        found = match_of({"corpus": "quantities"}, self._condition())
        assert found.status == "undeclared"
        assert found.differing == ("corpus",)
        assert not found.comparable


class TestTheCorpusIsCompared:
    """The axis that had no field, so nothing could check it.

    An adapter tuned on generated quantity prose was scored on human
    encyclopedic prose, and the gap between it and one tuned on the register
    it was tested on was reported as a fact about the two models. The corpus
    now has a name on every task and the comparison reads it.
    """

    def test_a_corpus_the_model_was_not_tuned_on_is_a_mismatch(self):
        from oold_llm_bench.tasks.models import CorpusRef, Source

        ref = CorpusRef(name="wiki-measurements", source=Source.SYNTHETIC, document_id="d", content_hash="h")
        assert ref.name == "wiki-measurements"

    def test_every_corpus_names_itself(self):
        """A corpus with no name cannot be compared against a training record."""
        import pathlib
        import re

        missing = []
        for path in sorted(pathlib.Path("src/oold_llm_bench/corpus").glob("*.py")):
            body = path.read_text(encoding="utf-8")
            if "CorpusRef(" not in body:
                continue
            if not re.search(r'CorpusRef\(\s*name="[a-z-]+"', body):
                missing.append(path.name)
        assert not missing, f"these build a CorpusRef with no name: {missing}"
