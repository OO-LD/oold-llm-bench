"""Turning an arm's prose answer into triples.

The parser makes its own errors, and those errors are part of every
``no-catalog-not-enforced-prose`` score. :func:`parse_loss` runs it over a rendering of the gold
answer, the best input it can get, and returns what fraction it recovers. A
low ``no-catalog-not-enforced-prose`` score has to be read against that ceiling, because it could
be the model or it could be this file.

``no-catalog-not-enforced-prose`` exists to answer the objection that JSON is already a form of
structure, so asking a model for JSON already decides half the question.

The parser is frozen and kept simple on purpose. Anything cleverer would be
tuned against the arm it scores, which is the failure this refactor removes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from oold_llm_bench.grading.triples import Quantity, Triple, TripleSet, make_triple

if TYPE_CHECKING:
    from oold_llm_bench.tasks.models import TaskRecord

__all__ = ["extract_prose", "parse_loss", "unit_phrase"]

_NUMBER = re.compile(r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)(?![\w.])")

_UNIT_TOKEN = re.compile(r"[A-Za-zÀ-ÿ]+|\d+")
"""Digits count, because seven of the corpus units carry one, as in
``cal_to_the_15_degree_celsius``. Letting the unit take that 15 also
stops it being read as a second measurement."""

_STOP_WORDS = frozenset({
    "was",
    "were",
    "is",
    "are",
    "an",
    "and",
    "or",
    "but",
    "which",
    "that",
    "with",
    "within",
    "in",
    "on",
    "of",
    "for",
    "from",
    "by",
    "then",
    "also",
    "recorded",
    "reported",
    "measured",
    "showed",
    "settled",
    "came",
    "noted",
    "wrote",
    "moved",
    "found",
    "gave",
    "read",
    "reading",
    "value",
    "values",
    "instrument",
    "figure",
    "sheet",
    "everything",
    "throughout",
    "band",
    "expected",
    "further",
    "adjustment",
    "operator",
    "attempt",
    "step",
    "next",
    "comment",
    "run",
    "made",
    "no",
    "not",
    "it",
    "down",
})
"""Words that end a unit phrase.

Narrow on purpose. Over the 1,291 distinct unit names in the corpus the only
words that also read as ordinary English are ``a``, ``the``, ``to`` and
``per``, so all four are absent here: without ``the`` and ``to`` in the
list, ``coulomb to the fourth meter to the fourth per joule cubed`` comes
back complete. Everything listed is a clause connective or a word out of the
rendering templates, and none of them occurs inside a real unit.
"""


def unit_phrase(words: list[str]) -> str:
    """The unit spelled with underscores, as the corpus writes it.

    Documents render ``milli_mole_per_kilo_gram`` as "milli mole per kilo
    gram", so reading it back means joining the words again. A unit that the
    corpus does not use will simply not match, which is correct.
    """
    return "_".join(w.lower() for w in words)


def _unit_after(text: str, position: int) -> tuple[str, int]:
    """The unit words following a number, and where they end.

    Stops at punctuation, at a clause connective, or at any gap wider than a
    single space. The returned offset keeps a digit inside a unit from
    being picked up again as a magnitude.
    """
    words: list[str] = []
    cursor = position + 1 if text[position : position + 1] == " " else position
    while cursor < len(text):
        match = _UNIT_TOKEN.match(text, cursor)
        if match is None:
            break
        word = match.group(0)
        if word.lower() in _STOP_WORDS:
            break
        words.append(word)
        cursor = match.end()
        if cursor >= len(text) or text[cursor] != " ":
            break
        cursor += 1

    if words and words[0].isdigit():
        return "", position
    return unit_phrase(words), cursor


_TRAILING_IN = re.compile(r"\s*,?\s*(?:measured|expressed|reported|given|stated)?\s*in\s+")
"""A unit that arrives after the magnitude instead of beside it, as in
"came to 4.2, measured in newton". Ordinary English, not a quirk of this
corpus, so the parser handles it. A unit announced in a separate
sentence is not handled, and shows up as parse loss."""

_LABEL_TAIL = re.compile(
    r"([A-Za-zÀ-ÿ][\w\-À-ÿ ]{0,60}?)\s*(?::|was|is|of|showed|reported|came to|settled at)\s*$",
    re.IGNORECASE,
)


def _label_before(text: str, position: int) -> str | None:
    """A label immediately preceding a measurement, if there is one."""
    window = text[max(0, position - 90) : position]
    window = window.rsplit("\n", 1)[-1].lstrip("- ").strip()
    match = _LABEL_TAIL.search(window)
    if match is None:
        return None
    label = match.group(1).strip()
    return label or None


def extract_prose(text: str) -> TripleSet:
    """Read measurements out of prose.

    One entity per measurement found, keyed in the order they appear, which
    is enough because a list of quantity values attaches nothing to a
    subject. A number with no readable unit is counted as a parse error,
    since a magnitude alone cannot answer a quantity.
    """
    triples: set[Triple] = set()
    classes: dict[str, str] = {}
    parse_errors = 0
    index = 0
    consumed = 0

    for match in _NUMBER.finditer(text or ""):
        if match.start() < consumed:
            continue
        magnitude = float(match.group(1).replace(",", ""))
        unit, consumed = _unit_after(text, match.end())
        if not unit:
            trailing = _TRAILING_IN.match(text, match.end())
            if trailing is not None:
                unit, consumed = _unit_after(text, trailing.end())
        if not unit:
            parse_errors += 1
            continue
        index += 1
        key = f"p{index}"
        triples.add(make_triple(key, "value", Quantity(magnitude=magnitude, unit=unit)))
        label = _label_before(text, match.start())
        if label:
            classes[key] = label

    if not triples and (text or "").strip():
        parse_errors += 1

    return TripleSet(
        triples=frozenset(triples),
        classes=classes,
        provenance={},
        parse_errors=parse_errors,
    )


def parse_loss(tasks: list[TaskRecord]) -> dict[str, float]:
    """What this parser recovers from the documents themselves.

    The document was rendered from the answer, so it is the best prose the
    parser will get. Whatever it fails to recover here is the parser's own
    error, and no ``no-catalog-not-enforced-prose`` score can exceed this ceiling.
    """
    from oold_llm_bench.grading.score import score_task

    if not tasks:
        return {"tasks": 0, "value_f1": 0.0, "parse_errors": 0}

    total = 0.0
    errors = 0
    for task in tasks:
        produced = extract_prose(task.document)
        total += score_task(task, produced).primary
        errors += produced.parse_errors
    return {
        "tasks": len(tasks),
        "value_f1": round(total / len(tasks), 4),
        "parse_errors": errors,
    }
