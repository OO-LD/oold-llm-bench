"""Reading the two tuned models against the base one, on both halves.

The grid is five prompt-and-model pairings run twice, once on tasks whose
answer is a class tuning saw and once on tasks whose answer is a class it did
not. Ten cells, and the pair of them is the whole result: a tuned model that
scores well on the training half proves nothing on its own, because memorising
the corpus produces the same number.

Two things decide whether any of it measures what it claims to.

The offered catalogue carries both halves in every cell. A held-out class that
is not on the list cannot be chosen, so the held-out score would be zero by
construction and the drop would be the experiment's own doing.

Each tuned model is read at the prompt it was tuned on. A model tuned on a
described catalogue and evaluated on a bare one is a distribution-shift
measurement wearing an H4 label.

Nothing here filters the task set by what the grader can score. The extractor
folds a ``value`` and a ``unit`` on one entity into a single quantity, which
caps a handful of classes below 1.00, and :func:`ceiling` reports how much of
each half that costs. Dropping those tasks would tilt the set toward what the
instrument likes, and the cap falls on every cell equally, so it moves no
comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import Any

from oold_llm_bench.config.models import ModelEntry
from oold_llm_bench.finetune.arm import LEAN_ARM, register_lean_arm
from oold_llm_bench.finetune.dataset import answer_text, round_trip_score
from oold_llm_bench.finetune.split import ClassSplit
from oold_llm_bench.runner import Condition
from oold_llm_bench.tasks.models import TaskRecord

__all__ = [
    "EVAL_SEED_BASE",
    "FULL_ARM",
    "PROMPTS",
    "QUDT_EVAL_SEED_BASE",
    "Half",
    "Pairing",
    "answer_names",
    "answer_pool",
    "ceiling",
    "evaluation_tasks",
    "pairings",
    "tuned_entry",
]

EVAL_SEED_BASE = 300000
"""Where evaluation seeds start.

Not 200000, which is where validation starts. The validation file is read
during tuning, so a task drawn at a validation seed is not a clean test even
though it was never trained on. Training sits at 100000 and the schema.org
grid at 7000, so this range is clear of all three.
"""

QUDT_EVAL_SEED_BASE = 400000
"""Where QUDT evaluation seeds start.

Clear of QUDT tuning at 100000 and validation at 200000, and of the schema.org
evaluation at 300000, so no two task sets in this study can collide.
"""

FULL_ARM = "schema-dump-catalog-not-enforced-gated"
"""The baseline prompt: the schema shown, no decode-time constraint.

Unenforced rather than flat-enforced, because the lean arm is this arm with
the schema section removed. A flat-enforced baseline would put a decode-time
constraint on one side of the contrast and nothing on the other, and the
difference would have two causes.
"""


class Half(str, Enum):
    """Which side of the split a task's answer comes from."""

    TRAIN = "train"
    """Classes tuning was allowed to answer. The claim."""

    HELDOUT = "heldout"
    """Classes tuning never answered, though it always offered them. The
    control. A tuned model that does as well here learned the task format and
    not the schemas."""


PROMPTS: dict[bool, dict[str, Condition]] = {}
"""The three prompts per pinning choice, filled on first use."""


def _register_prompts(pin_units: bool = False) -> dict[str, Condition]:
    """The three prompts, built once.

    ``full`` and ``lean`` differ only in the schema section, ``lean`` and
    ``bare`` only in whether the catalogue entries are rendered. One variable
    each, so a gap has one candidate cause.

    ``pin_units`` is the argument and not a constant. It was ``False`` here
    with no way to say otherwise, which made every H4 number a half-enforced
    one without saying so: the unit slot stayed open while the arm was read
    as the enforcement baseline. Both settings are defensible and they ask
    different questions. Open, the tune has to supply the vocabulary from its
    weights, which is what H4 is about. Closed, the grammar supplies it and
    the question becomes whether the tune still buys anything on top. Neither
    is the default, so a caller states which.
    """
    key = bool(pin_units)
    if key in PROMPTS:
        return PROMPTS[key]
    register_lean_arm()
    shared: dict[str, Any] = {
        "catalogue_size": None,
        "signal": "named",
        "vocabulary": "consensus",
        "pin_units": key,
        "orchestration": "single_shot",
    }
    PROMPTS[key] = {
        "full": Condition(arm=FULL_ARM, describe_catalogue=True, **shared),
        "lean": Condition(arm=LEAN_ARM, describe_catalogue=True, **shared),
        "bare": Condition(arm=LEAN_ARM, describe_catalogue=False, **shared),
    }
    return PROMPTS[key]


@dataclass(frozen=True)
class Pairing:
    """One model read at one prompt, which is one column of the result."""

    label: str
    prompt: str
    deployment: str
    tuned: bool
    pin_units: bool = False
    """Whether the unit slot is closed by the grammar. Carried on the pairing
    so it reaches the record, rather than sitting as a constant nobody saw."""

    @property
    def condition(self) -> Condition:
        return _register_prompts(self.pin_units)[self.prompt]


def pairings(
    base: str | None,
    tuned_lean: str | None,
    tuned_bare: str | None,
    *,
    pin_units: bool = False,
) -> list[Pairing]:
    """The five columns, named by what they are and not by an account.

    The base model is read at all three prompts, which is what turns the two
    tuned columns into a comparison rather than two isolated numbers.

    Any of the three may be absent, because they are deployed at different
    times and a tuning job takes hours. A missing column is left out rather
    than filled in from somewhere else: the held-out drop inside one column is
    readable on its own, and nothing that compares two columns is.
    """
    found = []
    if base:
        found += [
            Pairing("base+full", "full", base, tuned=False, pin_units=pin_units),
            Pairing("base+lean", "lean", base, tuned=False, pin_units=pin_units),
            Pairing("base+bare", "bare", base, tuned=False, pin_units=pin_units),
        ]
    if tuned_lean:
        found.append(Pairing("tuned-lean+lean", "lean", tuned_lean, tuned=True, pin_units=pin_units))
    if tuned_bare:
        found.append(Pairing("tuned-bare+bare", "bare", tuned_bare, tuned=True, pin_units=pin_units))
    if not found:
        raise ValueError("name at least one deployment, or there is nothing to read")
    return found


def tuned_entry(deployment: str, *, model_version: str = "2025-04-14") -> ModelEntry:
    """A model entry for a deployment this repo must not name.

    The deployment name is account-scoped, so it arrives as an argument and is
    never written down here. Everything else is the base model's: a tuned
    ``gpt-4.1-mini`` answers on the same transport, takes the same schema
    subset and accepts the same parameters.
    """
    return ModelEntry(
        model=deployment,
        model_version=model_version,
        provider_profile="openai",
        tier="mid",
        family="gpt-4.1",
    )


def answer_names(split: ClassSplit, half: Half) -> set[str]:
    """The class names a task in this half may be answered with."""
    return set(split.train if half is Half.TRAIN else split.heldout)


def answer_pool(items: list[Any], split: ClassSplit, half: Half, key: Callable[[Any], str]) -> list[Any]:
    """Whichever half of a corpus pool this half of the split names.

    ``key`` reads the name off a pool member, because a schema.org class and a
    QUDT quantity kind carry it under different attributes and neither is this
    module's business.
    """
    wanted = answer_names(split, half)
    found = [item for item in items if key(item) in wanted]
    if not found:
        raise ValueError(f"the {half.value} half names nothing the pool has")
    return found


def evaluation_tasks(
    draw: Callable[[str, int], TaskRecord],
    half: Half,
    *,
    count: int = 120,
    seed_base: int = EVAL_SEED_BASE,
) -> list[TaskRecord]:
    """``count`` tasks answered from one half and offered both.

    ``draw`` carries the corpus and which half may be the answer, so this is
    the same walk for schema.org and for QUDT. Both halves start from the same
    seed base: the generator is handed a different list of candidate classes
    for each, so the documents differ, while the values behind them come from
    the same stream and the halves are as alike as two different answers can
    be.
    """
    tasks: list[TaskRecord] = []
    seed = seed_base
    while len(tasks) < count:
        # A class offering fewer distinct slots than a task asks for raises,
        # and the seed is skipped. Anything else is a real fault and stops.
        with suppress(ValueError):
            tasks.append(draw(f"e{half.value[0]}{seed}", seed))
        seed += 1
    return tasks


def ceiling(tasks: list[TaskRecord]) -> float:
    """What a perfect answer scores on this set, which is not always 1.00.

    Reported beside every cell of the half it belongs to. A score read against
    1.00 when the instrument tops out lower understates every arm by the same
    amount, and someone will eventually read the gap as a model failure.
    """
    if not tasks:
        return 0.0
    return sum(round_trip_score(task, answer_text(task)) for task in tasks) / len(tasks)


def capped(tasks: list[TaskRecord]) -> Iterator[TaskRecord]:
    """The tasks a perfect answer cannot score 1.00 on."""
    return (task for task in tasks if round_trip_score(task, answer_text(task)) < 1.0)
