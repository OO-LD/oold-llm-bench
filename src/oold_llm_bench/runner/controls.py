"""Agents that are meant to score badly.

A grader that cannot fail is not measuring anything. These exist so that
claim can be tested instead of assumed, and the plan is explicit that no cell
gets reported until the negative control has been run and has scored near
zero.

Each one implements the same :class:`~oold.agent.client.ChatClient` protocol a
real model does, so it travels the whole path: the same prompt, the same
extractor, the same alignment, the same score. A control that bypassed any of
those would test less than the thing it is vouching for.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from typing import Any

__all__ = ["EmptyClient", "RandomClient", "SpellingClient", "WrongDocumentClient"]


class _Reply:
    """The shape a ChatClient returns, without importing the agent package.

    The benchmark depends on oold as one consumer among several, and a control
    is useful even where that dependency is not installed.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.parsed: dict[str, Any] | None = None
        self.usage = _Usage()
        self.raw = None


class _Usage:
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0
    reasoning_tokens = 0

    @property
    def total(self) -> int:
        return 0

    def describe(self) -> dict[str, int]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }


class EmptyClient:
    """Answers nothing at all. The floor any real arm must clear."""

    model = "control-empty"

    def invoke(self, messages: Sequence[Any], *, response_format=None) -> _Reply:
        return _Reply("{}")


class RandomClient:
    """Picks a class at random and invents a value.

    Scores above zero only if the grader is rewarding plausibility. With a
    catalogue of any size the chance of naming the right class is small, and
    the chance of also naming the right value is negligible.
    """

    model = "control-random"

    def __init__(
        self,
        catalogue: Sequence[str],
        seed: int = 0,
        units: Sequence[str] = ("meter", "second", "gram"),
    ) -> None:
        self.catalogue = list(catalogue)
        self.units = list(units)
        self._rng = random.Random(seed)  # noqa: S311 - reproducible, not secure

    def invoke(self, messages: Sequence[Any], *, response_format=None) -> _Reply:
        payload = {
            "entities": [
                {
                    "type": self._rng.choice(self.catalogue) if self.catalogue else "Unknown",
                    "value": round(self._rng.uniform(1, 1000), 3),
                    "unit": self._rng.choice(self.units),
                }
            ]
        }
        return _Reply(json.dumps(payload))


class WrongDocumentClient:
    """Answers a different document correctly.

    The sharpest control of the three. Every value it gives is a real value
    with a real unit and a real class, so anything it scores is the grader
    rewarding shape instead of content.
    """

    model = "control-wrong-document"

    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.answers = answers

    def invoke(self, messages: Sequence[Any], *, response_format=None) -> _Reply:
        return _Reply(json.dumps({"entities": self.answers}))


class SpellingClient:
    """Reads nothing. Copies the number and the words beside it.

    The sharpest control there is, and the one that was missing. It knows no
    vocabulary, no catalogue and no schema. It finds a number in the document,
    takes the words following it, replaces the spaces with underscores, and
    submits that as the unit.

    Measured on the quantity corpus as it was rendered until 2026-09-28: a
    perfect primary score on 115 of 120 tasks. The other three controls sat at
    0.00 and the grid was declared interpretable on that basis, while this one
    would have said the primary metric was a spelling convention and not an
    extraction.

    Under a corpus that writes units the way a document writes them it scores
    1 of 120, which is what a control is supposed to do.

    It is given the expected class, deliberately. Class is a separate
    dimension, and withholding it would let a failure on class disguise how
    much of the value dimension is recoverable without reading anything.

    On real text it will not reach zero and should not be read as a defect
    there. A document writing `bar` or `Hz` has a surface form that happens to
    equal the identifier, and no rendering choice of ours caused that. What
    this control reports is the share of a corpus answerable without reading
    any vocabulary, and on a generated corpus that share is a decision we made.
    """

    model = "control-spelling"

    _READING = re.compile(r"(\d[\d,]*\.?\d*)\s+([^.,;]+?)(?=[.,;]|\s+which\b|\s+and\b|$)")

    def __init__(self, document: str, class_path: str) -> None:
        self.document = document
        self.class_path = class_path

    def invoke(self, messages: Sequence[Any], *, response_format=None) -> _Reply:
        found = self._READING.search(self.document)
        if not found:
            return _Reply("{}")
        magnitude = float(found.group(1).replace(",", ""))
        unit = "_".join(found.group(2).split())
        entity = {"type": self.class_path, "value": magnitude, "unit": unit}
        return _Reply(json.dumps({"entities": [entity]}))
