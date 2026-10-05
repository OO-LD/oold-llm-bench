"""Whether a tuned model is being asked the question it was trained on.

A fine-tuned model answers about the prompt it saw in training. Shown a
different one it is not the tuned model under test, it is a model out of
distribution, and its score is about the distance between the two prompts
rather than about the tune.

This is not a new rule. The Azure H4 already followed it and put the pairing
in the cell name, ``tuned-lean+lean``, where a mismatch cannot be written down
by accident. The local run reused the general grid, which carries its own
condition axis and no knowledge that a tune happened, and so trained on a
2,292 character catalogue of bare class names and evaluated against a 38,654
character described one. The first sign was a score moving the wrong way,
which is the worst way to find out, because a plausible wrong number is
reportable and a crash is not.

So the pairing is computed, recorded on every cell, and carried into the
report key the way fidelity already is. A matched and a mismatched cell are
different treatments and are never averaged together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["TrainingMatch", "match_of"]


@dataclass(frozen=True)
class TrainingMatch:
    """How the evaluation condition relates to the training one."""

    status: str
    """``base`` where no tune happened, ``matched`` where every declared key
    agrees, ``mismatched`` where one disagrees, ``undeclared`` where the
    condition cannot express a key the training record claims.

    ``undeclared`` exists because skipping such a key leaves the axis this
    module protects unguarded. The adapters declare ``corpus``,
    :class:`~oold_llm_bench.runner.config.Condition` has no such field, and a
    quantities-trained adapter evaluated on Wiki-Measurements would otherwise
    report ``matched``."""

    differing: tuple[str, ...] = ()
    """The keys that disagree, in a stable order, so two records naming the
    same mismatch read the same."""

    @property
    def comparable(self) -> bool:
        """Whether this cell may be pooled with a base cell of the condition.

        A base model is always comparable: it was not trained on anything, so
        no condition is privileged for it. ``undeclared`` is not comparable
        either, because an unchecked axis is not a checked one.
        """
        return self.status in ("base", "matched")

    def describe(self) -> dict[str, Any]:
        return {"status": self.status, "differing": list(self.differing)}

    @property
    def label(self) -> str:
        """A short form for a report key."""
        if self.status == "base":
            return ""
        if self.status == "matched":
            return "trained-for"
        if self.status == "undeclared":
            return "unchecked:" + ",".join(self.differing)
        return "off-train:" + ",".join(self.differing)


def match_of(trained_on: dict[str, Any] | None, condition: Any) -> TrainingMatch:
    """Compare what a model was trained under against what it is being asked.

    Only keys the entry actually declares are compared. An axis the training
    record is silent about is not a claim, so adding one later does not
    retroactively invalidate a record that predates it. The cost of that
    choice is that an undeclared axis cannot raise a mismatch, which is why
    :attr:`~oold_llm_bench.config.models.ModelEntry.trained_on` is worth
    filling in completely rather than partly.
    """
    if not trained_on:
        return TrainingMatch(status="base")

    differing: list[str] = []
    undeclared: list[str] = []
    for key in sorted(trained_on):
        if not hasattr(condition, key):
            undeclared.append(key)
        elif getattr(condition, key) != trained_on[key]:
            differing.append(key)
    if differing:
        return TrainingMatch(status="mismatched", differing=tuple(differing))
    # Reported rather than skipped. A key the condition cannot express is
    # not evidence of agreement, and treating it as one is how a
    # quantities-trained adapter read as matched on another corpus.
    if undeclared:
        return TrainingMatch(status="undeclared", differing=tuple(undeclared))
    return TrainingMatch(status="matched")
