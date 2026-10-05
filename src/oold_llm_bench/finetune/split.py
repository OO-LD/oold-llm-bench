"""Which classes tuning is allowed to see.

H4 asks whether tuning on a schema set substitutes for sending the schema in
the prompt. A tuned model scoring well on the classes it was tuned on answers
nothing: it is consistent with the model having learned the schemas and with
the model having learned the corpus. The claim only becomes testable if half
the classes were never in the training data, so a drop on that half is the
evidence and the other half is the control.

The partition is therefore corpus metadata and not a runtime decision. It is
computed once, written to :data:`SPLIT_PATH`, and read from there afterwards.
Recomputing it at evaluation time would make the contrast depend on a seed
nobody checked, and a partition that silently moved between tuning and
evaluation would report the held-out drop of a different split.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "QUANTITIES_SPLIT_PATH",
    "SPLIT_PATH",
    "SPLIT_SEED",
    "SPLIT_VERSION",
    "ClassSplit",
    "load_split",
    "partition",
]

SPLIT_VERSION = "1"
"""Bumped when the partition changes. A result produced under one version
cannot be compared with a result produced under another, so the number is
recorded beside the split rather than left to a file date."""

SPLIT_SEED = 41011
"""The seed the committed partition was drawn with."""

SPLIT_PATH = Path(__file__).resolve().parent.parent / "data" / "schemaorg_split.json"
"""Where the committed schema.org partition lives. Corpus metadata, not a secret."""

QUANTITIES_SPLIT_PATH = Path(__file__).resolve().parent.parent / "data" / "quantities_split.json"
"""The same for QUDT, which is the corpus H4 can actually be tested on."""


@dataclass(frozen=True)
class ClassSplit:
    """The two halves of the class pool, and how they were drawn."""

    train: tuple[str, ...]
    heldout: tuple[str, ...]
    seed: int = SPLIT_SEED
    min_own_slots: int = 3
    """The threshold the pool was filtered at. A split over a pool drawn at a
    different threshold covers different classes, so the number belongs to the
    split. schema.org only: QUDT filters on signal support instead."""
    corpus: str = "schemaorg"
    """Which corpus the names belong to. Two corpora, two files, and a split
    read against the wrong one would silently hold out nothing."""
    version: str = SPLIT_VERSION

    def __post_init__(self) -> None:
        overlap = set(self.train) & set(self.heldout)
        if overlap:
            raise ValueError(f"a class cannot be in both halves: {sorted(overlap)[:5]}")
        if not self.train or not self.heldout:
            raise ValueError("both halves must be non-empty, or there is no contrast to measure")

    @property
    def names(self) -> tuple[str, ...]:
        """Every class the split covers, sorted."""
        return tuple(sorted(set(self.train) | set(self.heldout)))

    def describe(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "corpus": self.corpus,
            "min_own_slots": self.min_own_slots,
            "n_train": len(self.train),
            "n_heldout": len(self.heldout),
            "train": list(self.train),
            "heldout": list(self.heldout),
        }

    def write(self, path: Path = SPLIT_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.describe(), indent=2) + "\n", encoding="utf-8")
        return path


def partition(
    names: Iterable[str],
    *,
    seed: int = SPLIT_SEED,
    min_own_slots: int = 3,
    corpus: str = "schemaorg",
) -> ClassSplit:
    """Split the class names in half, deterministically.

    Sorted before shuffling, so the result depends on the set of names and the
    seed and on nothing else. Taking the names in the order a directory listing
    or a dict happened to yield them would give two machines two splits from
    one seed, and neither would know.
    """
    ordered = sorted(set(names))
    if len(ordered) < 2:
        raise ValueError(f"a pool of {len(ordered)} classes cannot be split in two")
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
    half = len(shuffled) // 2
    return ClassSplit(
        train=tuple(sorted(shuffled[:half])),
        heldout=tuple(sorted(shuffled[half:])),
        seed=seed,
        min_own_slots=min_own_slots,
        corpus=corpus,
    )


def load_split(path: Path = SPLIT_PATH, *, expect_corpus: str | None = None) -> ClassSplit:
    """Read the committed partition.

    Read, not recomputed. The file is what tuning ran against.

    ``expect_corpus`` is checked rather than trusted, because a split read
    against the wrong corpus holds out nothing and nothing crashes. A
    quantities split read over a schema.org pool names ``AbsoluteActivity``
    where a reader expects ``Person``, holds out no class the pool contains,
    and reports a held-out score that is a train-half score under another
    name.

    The default stays ``None`` so an exploratory read still works, and every
    caller that knows which corpus it wants says so.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    found = str(payload.get("corpus", "schemaorg"))
    if expect_corpus is not None and found != expect_corpus:
        raise ValueError(
            f"{path.name} is a {found!r} split, not {expect_corpus!r}. "
            "Reading it would hold out classes the pool does not contain."
        )
    return ClassSplit(
        train=tuple(payload["train"]),
        heldout=tuple(payload["heldout"]),
        seed=int(payload["seed"]),
        min_own_slots=int(payload["min_own_slots"]),
        corpus=found,
        version=str(payload["version"]),
    )
