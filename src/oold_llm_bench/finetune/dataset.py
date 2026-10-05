"""Turning the corpus into the chat JSONL a tuning job reads.

Three things decide whether the job is worth running, and all three are
checked here rather than assumed.

The prompt is the evaluation prompt. It is assembled by
:func:`~oold_llm_bench.runner.build_enforcement` and
:func:`~oold_llm_bench.runner.build_request`, the same two functions a run
calls, and handed to ``build_messages`` unchanged. Tuning on one wording and
evaluating on another measures the wording.

The answer is the task's own ground truth, serialised in the shape the JSON
extractor reads. Every example is fed back through that extractor and the
grader before it is written, and one that does not score 1.00 is dropped.
A training set the grader disagrees with teaches the model to lose marks.

The classes are the training half of :mod:`~oold_llm_bench.finetune.split`.
The catalogue offered in the prompt is drawn from both halves, so the prompt
looks at tuning time exactly as it looks at evaluation time, and a held-out
class is a name the model has seen offered and never seen chosen.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oold_llm_bench.extract import extract_json
from oold_llm_bench.finetune.arm import LEAN_ARM, register_lean_arm
from oold_llm_bench.finetune.split import ClassSplit
from oold_llm_bench.grading.score import score_task
from oold_llm_bench.results.record import ModelSpec
from oold_llm_bench.runner import Cell, Condition, build_enforcement, build_request
from oold_llm_bench.tasks.models import TaskRecord

__all__ = [
    "CATALOGUE_SIZE",
    "TRAIN_SEED_BASE",
    "VALIDATION_SEED_BASE",
    "Corpus",
    "answer_text",
    "build_corpus",
    "build_example",
    "lean_condition",
    "offered_catalogue",
    "round_trip_score",
    "write_jsonl",
]

TRAIN_SEED_BASE = 100000
VALIDATION_SEED_BASE = 200000
"""Seeds for the two tuning files.

The evaluation grid draws ``7000 + i``. Starting here leaves that range alone,
so no document the model was tuned on can turn up as a test item. Two bases
rather than one keeps the validation file off the training documents as well.
"""

CATALOGUE_SIZE = 25
"""How many classes a prompt offers.

The same number the schema.org grid runs at, because the catalogue is part of
the prompt and a tuned model trained against a different one would be
evaluated on a prompt it has not seen.
"""

_PROMPT_MODEL = ModelSpec(model="prompt-only", provider_profile="openai")
"""A cell needs a model, and prompt assembly never reads one.

Named so that a record carrying it would be obviously wrong. Nothing here
writes a record.
"""


def lean_condition(catalogue_size: int | None = None, *, describe_catalogue: bool = True) -> Condition:
    """The condition the tuning corpus and the tuned arm both run under.

    ``catalogue_size`` stays ``None`` by default because the catalogue is
    already sized when the task is generated. Trimming it again in the adapter
    would keep the classes a task needs and then fill from the head of the
    list, so every prompt would offer the same distractors.

    ``describe_catalogue`` is the second tuning arm. Described, the prompt
    carries each class with its properties and enumerations, and most of the
    prompt is that rendering rather than the schema H4 is about. Bare, the
    prompt lists identifiers only, so what a class means has to come from the
    weights. Only the condition changes: the task, the document and the answer
    are the same, so the two tuned models are a paired comparison.
    """
    register_lean_arm()
    return Condition(
        arm=LEAN_ARM,
        catalogue_size=catalogue_size,
        signal="named",
        vocabulary="consensus",
        pin_units=False,
        describe_catalogue=describe_catalogue,
        orchestration="single_shot",
    )


def offered_catalogue(
    needed: Sequence[str],
    split: ClassSplit,
    rng: random.Random,
    size: int = CATALOGUE_SIZE,
) -> tuple[str, ...]:
    """The classes one prompt offers: the answer, plus distractors from both halves.

    Held-out classes are offered at tuning time and never answered. That is
    what makes the evaluation contrast readable: if the offered list held only
    training classes, the tuned model would have learned that the held-out
    names are never the answer, and the drop on them would measure that rather
    than the missing schema.
    """
    kept = list(dict.fromkeys(needed))
    if len(kept) > size:
        raise ValueError(f"a task needing {len(kept)} classes cannot be offered a catalogue of {size}")
    rest = size - len(kept)
    from_heldout = [name for name in split.heldout if name not in kept]
    from_train = [name for name in split.train if name not in kept]
    n_heldout = min(rest // 2, len(from_heldout))
    n_train = min(rest - n_heldout, len(from_train))
    drawn = rng.sample(from_heldout, n_heldout) + rng.sample(from_train, n_train)
    return tuple(sorted(kept) + sorted(drawn))


def answer_text(task: TaskRecord) -> str:
    """The task's own answer, in the shape the JSON extractor reads.

    A quantity is written out as a magnitude beside a unit, because that is
    what the extractor folds back into one, and a corpus whose values are
    quantities would otherwise not serialise at all. Everything else is
    written as it stands.

    Compact, because a tuned model emits what it was tuned on and indentation
    is output tokens that carry nothing.
    """
    entities = []
    for instance in task.expected:
        entity: dict[str, Any] = {"type": instance.class_path}
        for prop, value in instance.fields.items():
            unit = getattr(value, "unit", None)
            if unit is None:
                entity[prop] = value
                continue
            entity[prop] = value.magnitude
            entity["unit"] = unit
        entities.append(entity)
    return json.dumps({"entities": entities}, ensure_ascii=False)


def round_trip_score(task: TaskRecord, text: str) -> float:
    """What the grader gives this answer, read back the way a reply is read.

    Through the agent's own parser and the shared extractor, so a mismatch
    between the written shape and the read shape shows up here and not in a
    tuned model that learned to write something the grader discards.
    """
    from oold.agent.extraction import parse_json_answer

    payload = parse_json_answer(text)
    if payload is None:
        return 0.0
    return score_task(task, extract_json(payload)).primary


def build_example(task: TaskRecord, condition: Condition) -> dict[str, Any]:
    """One chat example: the evaluation prompt, and the correct answer."""
    from oold.agent.prompts import build_messages

    cell = Cell(condition=condition, model=_PROMPT_MODEL, task=task, repetition=1)
    messages = build_messages(build_request(cell), build_enforcement(cell))
    turns = [{"role": message.role, "content": message.content} for message in messages]
    turns.append({"role": "assistant", "content": answer_text(task)})
    return {"messages": turns}


@dataclass
class Corpus:
    """One tuning file, and what was thrown away building it."""

    name: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[TaskRecord] = field(default_factory=list)
    """The task each example was written from, kept so an example can be
    graded again by whoever reads the corpus and not only while it is built."""
    classes: dict[str, int] = field(default_factory=dict)
    rejected: list[tuple[int, str]] = field(default_factory=list)
    """Seed and reason for every draw that did not make it in.

    Kept because a silent drop is the one failure this module exists to
    prevent: a rejection rate that climbs says the answer shape and the
    extractor have parted company.
    """
    last_seed: int = 0

    def __len__(self) -> int:
        return len(self.examples)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_examples": len(self.examples),
            "n_classes": len(self.classes),
            "n_rejected": len(self.rejected),
            "last_seed": self.last_seed,
            "reasons": sorted({reason for _, reason in self.rejected}),
        }


SEED_BUDGET = 40
"""Seeds allowed per wanted example before a build gives up.

Generous, because a draw is cheap and a legitimate rejection rate of nine in
ten is possible when half the classes are held out. It exists to end a walk
that cannot finish, not to tune one that can."""


def build_corpus(
    split: ClassSplit,
    draw: Callable[[str, int], TaskRecord],
    *,
    name: str,
    count: int,
    seed_base: int,
    condition: Condition | None = None,
    seen: set[str] | None = None,
) -> Corpus:
    """Draw ``count`` examples whose answers are all training classes.

    ``draw`` is the corpus. Everything that differs between schema.org and
    QUDT is closed over before it arrives here, so the three checks below run
    the same way on both: the answer is never a held-out class, the document
    is new, and the answer survives the grader.

    Seeds are walked upward from ``seed_base`` and a draw that fails is
    skipped, so the set is always the requested size and the seeds it used are
    recorded. ``seen`` carries document hashes across calls, which keeps a
    validation document off the training file.
    """
    condition = condition or lean_condition()
    heldout = set(split.heldout)

    corpus = Corpus(name=name)
    seen = set() if seen is None else seen
    seed = seed_base
    # A bound, because the two corpora exhaust differently. schema.org draws a
    # fresh task per seed and never runs out; Wiki-Measurements is 1,254 fixed
    # examples and the split holds out half the kinds, so a request for more
    # than the pool can answer walked seeds for ever and appended a rejection
    # for each one. That is how a build for 1,000 wiki examples ended in
    # MemoryError rather than in a message saying the corpus is too small.
    budget = seed_base + max(count * SEED_BUDGET, SEED_BUDGET)
    while len(corpus.examples) < count:
        if seed > budget:
            raise ValueError(
                f"{name}: asked for {count} examples and the corpus yielded "
                f"{len(corpus.examples)} in {seed - seed_base} draws. "
                f"A finite corpus minus the held-out half may simply be smaller "
                f"than the request; ask for fewer."
            )
        corpus.last_seed = seed
        try:
            task = draw(f"{name}{seed}", seed)
        except (ValueError, KeyError) as exc:
            corpus.rejected.append((seed, f"generation: {exc}"))
            seed += 1
            continue

        answered = {instance.class_path for instance in task.expected}
        if answered & heldout:
            corpus.rejected.append((seed, "answer is a held-out class"))
            seed += 1
            continue
        if task.corpus.content_hash in seen:
            corpus.rejected.append((seed, "duplicate document"))
            seed += 1
            continue

        example = build_example(task, condition)
        graded = round_trip_score(task, example["messages"][-1]["content"])
        if graded < 1.0:
            corpus.rejected.append((seed, f"round trip scored {graded:.2f}"))
            seed += 1
            continue

        seen.add(task.corpus.content_hash)
        corpus.examples.append(example)
        corpus.tasks.append(task)
        for answered_name in answered:
            corpus.classes[answered_name] = corpus.classes.get(answered_name, 0) + 1
        seed += 1
    return corpus


def write_jsonl(path: Path, examples: Iterable[dict[str, Any]]) -> Path:
    """Write one example per line, as the tuning API wants it.

    With a byte-order mark, which is what the service documents for a training
    file. Reading one back needs ``utf-8-sig``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return path
