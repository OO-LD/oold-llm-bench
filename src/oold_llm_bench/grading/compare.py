"""Deciding whether two values are the same answer.

The predecessor compared strings with a case-insensitive substring test, which
accepts "Doe" for "Doe-Smith" and accepts any long hallucination that happens
to contain the expected phrase. Equality after normalisation is the default
here, and anything looser has to be asked for per field.
"""

from __future__ import annotations

import math
import re
import unicodedata
from enum import Enum
from typing import Any

from oold_llm_bench.grading.triples import Quantity, Scalar

__all__ = [
    "NUMERIC_TOLERANCE",
    "MatchMode",
    "UnitMatch",
    "normalise_text",
    "normalise_unit",
    "same_quantity",
    "same_value",
    "unit_forms",
]

NUMERIC_TOLERANCE = 1e-6
"""Relative tolerance for numbers. Tight, because a value read out of a
document should be the value in the document, not near it."""

_WHITESPACE = re.compile(r"\s+")


class MatchMode(str, Enum):
    """How strictly one expected value is compared.

    ``EXACT`` is the default and the only one a task gets without asking.
    """

    EXACT = "exact"
    CONTAINS = "contains"
    """The expected value appears inside the produced one. For fields whose
    gold value is a fragment of a longer rendering, and never as a default."""


def normalise_text(value: object) -> str:
    """Fold a value to the form two spellings of one answer share.

    Unicode normalisation, case folding and whitespace collapsing only. No
    stemming, no punctuation stripping, because removing punctuation would
    make "1,000" and "1000" and "1.000" indistinguishable across locales.
    """
    text = unicodedata.normalize("NFKC", str(value))
    return _WHITESPACE.sub(" ", text).strip().casefold()


_DASHES = dict.fromkeys(range(0x2010, 0x2016), "-") | {0x2212: "-"}
"""Every dash a document writes, folded onto the hyphen.

Real text writes an exponent with U+2212, the mathematical minus, and a
table keyed on the hyphen never sees it. Folding them together costs nothing:
no unit spelling distinguishes one dash from another.
"""

_BRACKETS = re.compile(r"[\[\](){}]")

_LETTERS = {"\u03bc": "u", "\u03a9": "ohm", "\u00b0": "deg"}
"""Symbol characters a spelled-out name writes as letters.

Read after compatibility normalisation, which already folds the micro sign
onto Greek mu and the ohm sign onto capital omega, so only one spelling of
each has to appear here.
"""

_NOT_ALPHANUMERIC = re.compile(r"[^a-z0-9]")

_SPELLINGS = (("metre", "meter"), ("litre", "liter"), ("gramme", "gram"), ("tonne", "ton"))
"""British spellings folded onto the one QUDT publishes.

Wikipedia writes both, often in one article. QUDT writes ``meter``, so that is
the side the fold lands on, and nothing here changes which name is the answer.
"""

_POWERS = (
    (re.compile(r"^(?:square|sq|sq\.) (.+)$"), "2"),
    (re.compile(r"^(.+?) (?:squared|square)$"), "2"),
    (re.compile(r"^(?:cubic|cu|cu\.) (.+)$"), "3"),
    (re.compile(r"^(.+?) (?:cubed|cube)$"), "3"),
)
"""An exponent written as a word, rewritten as the digit.

"square kilometres" and ``kilo_meter_squared`` are the same unit and share no
substring longer than the stem, so the exponent has to move to one side before
either can be looked up.
"""


def normalise_unit(value: object) -> str:
    """Fold a written unit to the form two spellings of it share, keeping case.

    Case survives on purpose. ``mW`` and ``MW`` are milliwatt and megawatt, so
    a comparison that folds them has lost the answer rather than found it, and
    the published symbol is the one place the distinction is still reliable.
    :func:`unit_forms` is where case goes, once the symbol has had its chance.

    Compatibility normalisation does the superscripts: NFKC maps a
    superscript two onto a plain two, so both spellings of ``km2`` arrive as
    one string without a table of exponent characters.
    """
    text = unicodedata.normalize("NFKC", str(value)).translate(_DASHES)
    text = _BRACKETS.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip(" .,;:")


def _fold(text: str) -> str:
    """One spelling reduced to letters and digits, case and accents gone."""
    text = text.casefold()
    for character, word in _LETTERS.items():
        text = text.replace(character, word)
    for british, published in _SPELLINGS:
        text = text.replace(british, published)
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return _NOT_ALPHANUMERIC.sub("", text)


def unit_forms(unit: object) -> tuple[str, ...]:
    """The keys a written unit may be looked up under, best first.

    Real text writes ``km2``, ``km`` with a superscript two, ``square
    kilometres`` and ``square kilometers`` for the unit the quantity corpus
    calls ``kilo_meter_squared``, and a lookup on the literal string finds
    none of them. Each rule folds one difference a person would not notice:
    the exponent, the plural, the British spelling, the separators.

    Measured over the 17,661 Wiki-Measurements sentences whose Wikidata
    property names a quantity kind the corpus has: resolution against that
    kind's own enumeration reaches 9,451 of the written units, against 2,778
    under case folding alone, so the fold rescues 6,673.
    """
    base = normalise_unit(unit).casefold()
    # The corpus spells one unit ``kilo_meter_squared`` and a sentence spells
    # it "square kilometres", so the separator is folded before the exponent
    # is looked for: otherwise the written-out exponent is only reachable from
    # the side that already wrote it as a word.
    stems = [(spelling, "") for spelling in dict.fromkeys((base, base.replace("_", " ")))]
    for spelling, _ in list(stems):
        for pattern, power in _POWERS:
            found = pattern.match(spelling)
            if found:
                stems.append((found.group(1), power))

    keys: list[str] = []
    for stem, power in stems:
        singular = stem[:-1] if stem.endswith("s") and len(stem) > 2 else stem
        for spelling in dict.fromkeys((stem, singular)):
            key = _fold(spelling) + power
            if key and key != power and key not in keys:
                keys.append(key)
    return tuple(keys)


def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", ""))
        except ValueError:
            return None
    return None


def same_number(expected: float, produced: float) -> bool:
    return math.isclose(expected, produced, rel_tol=NUMERIC_TOLERANCE, abs_tol=NUMERIC_TOLERANCE)


SI_PREFIXES = (
    "quetta",
    "ronna",
    "yotta",
    "zetta",
    "exa",
    "peta",
    "tera",
    "giga",
    "mega",
    "kilo",
    "hecto",
    "deca",
    "deci",
    "centi",
    "milli",
    "micro",
    "nano",
    "pico",
    "femto",
    "atto",
    "zepto",
    "yocto",
    "ronto",
    "quecto",
)


def pint_name(unit: str) -> str:
    """Rewrite a QUDT-style unit name into one pint parses.

    Underscores become spaces, then only an SI prefix is rejoined to the word
    after it, so ``kilo_meter`` becomes ``kilometer`` while
    ``meter_per_second`` becomes ``meter per second``. A leading ``per``
    becomes ``1 /``, and ``number`` means dimensionless.

    Taken from ``QuantityValue.get_pint_ureg_compatible_str`` in
    ``opensemantic.characteristics.quantitative``, which generates the
    corpus these names come from. Measured over the 1,291 distinct unit names
    in that corpus it resolves 89.2%, against 50.4% for a naive
    underscore-stripping rule.
    """
    name = unit.replace("_", " ")
    for prefix in SI_PREFIXES:
        name = name.replace(prefix + " ", prefix)
    name = name.strip(" ")
    if name.split(" ")[0] == "per":
        name = name.replace("per", "1 /", 1)
    name = name.lower()
    return "dimensionless" if name == "number" else name


def unit_candidates(unit: str) -> list[str]:
    """Spellings of a unit name to try against pint, best first.

    The rewrite above resolves most of the corpus, and the literal spelling
    resolves 30 names it does not, mostly ones pint knows with their
    underscores such as ``astronomical_unit`` and ``electron_volt``. Together
    they reach 91.5%. The rest are QUDT units pint does not define, and those
    fall back to comparing the unit as a string, which is the honest answer.
    """
    seen: list[str] = []
    for candidate in (
        pint_name(unit),
        unit,
        unit.replace("_", ""),
        unit.replace("_", " "),
    ):
        if candidate and candidate not in seen:
            seen.append(candidate)
    return seen


def _as_pint(registry: Any, magnitude: float, unit: str) -> Any:
    for candidate in unit_candidates(unit):
        try:
            return registry.Quantity(magnitude, candidate)
        except Exception:  # noqa: S112 - pint raises several unrelated types
            continue
    return None


class UnitMatch(str, Enum):
    """How strictly a unit has to be spelled.

    ``EXACT`` accepts only the name the corpus uses. When a quantity corpus
    closes the unit slot, that name is the answer, and anything else has not
    conformed to the schema however physically right it is.

    ``PHYSICAL`` accepts any spelling pint can convert, so ``MeV`` passes for
    ``mega_electron_volt``. That is the wrong default here: an arm with no
    enum emits variants and an arm with an enum cannot, so the leniency lands
    on one side of the comparison and flatters it.
    """

    EXACT = "exact"
    PHYSICAL = "physical"


def same_quantity(expected: Quantity, produced: Quantity, *, unit_match: UnitMatch = UnitMatch.EXACT) -> bool:
    """Whether a produced quantity is the expected answer.

    Under ``EXACT`` the unit name has to match. Under ``PHYSICAL`` pint
    converts, and a unit pint cannot parse falls back to name equality, which
    is the honest answer for the QUDT units pint does not define.
    """
    if normalise_text(expected.unit) == normalise_text(produced.unit):
        return same_number(expected.magnitude, produced.magnitude)
    if unit_match is UnitMatch.EXACT:
        return False
    try:
        import pint
    except ImportError:
        return False
    registry = pint.get_application_registry()
    left = _as_pint(registry, expected.magnitude, expected.unit)
    right = _as_pint(registry, produced.magnitude, produced.unit)
    if left is None or right is None:
        return False
    try:
        converted = right.to(left.units)
    except Exception:
        return False
    return same_number(float(left.magnitude), float(converted.magnitude))


def same_unit(expected: Scalar, produced: Scalar, *, unit_match: UnitMatch = UnitMatch.EXACT) -> bool | None:
    """Whether the unit is right, scored apart from the magnitude.

    Returns ``None`` when neither side is a quantity, so unit accuracy is
    reported over the fields that have units, not over all of them.
    """
    if not isinstance(expected, Quantity):
        return None
    if not isinstance(produced, Quantity):
        return False
    if normalise_text(expected.unit) == normalise_text(produced.unit):
        return True
    if unit_match is UnitMatch.EXACT:
        return False
    return same_quantity(
        Quantity(magnitude=expected.magnitude, unit=expected.unit),
        Quantity(magnitude=expected.magnitude, unit=produced.unit),
        unit_match=UnitMatch.PHYSICAL,
    )


_ISO_DURATION = re.compile(
    r"(?i)^P(?!$)(?:(\d+(?:\.\d+)?)Y)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)W)?(?:(\d+(?:\.\d+)?)D)?"
    r"(?:T(?!$)(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?)?$"
)
"""ISO 8601, the form markup stores a duration in."""

_WRITTEN_DURATION = re.compile(
    r"(?i)(\d+(?:[.,]\d+)?)\s*(years?|yrs?|months?|weeks?|wks?|days?|hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b"
)
"""A duration as a page writes it for a reader, which is never ISO 8601.

Measured over the Web Data Commons corpus: of 324 expected ``prepTime``,
``cookTime`` and ``totalTime`` values, 0 appear in the page verbatim. The
markup says ``PT5M`` and the page says "5 minutes", so comparing them as text
scores every one of them wrong whatever the model answered.
"""

_SECONDS = {
    "y": 31_536_000.0,
    "mo": 2_592_000.0,
    "w": 604_800.0,
    "d": 86_400.0,
    "h": 3600.0,
    "mi": 60.0,
    "s": 1.0,
}
"""Seconds per unit, with a year 365 days and a month 30.

Both are conventions and neither is right for every calendar. They are here
because a duration property states an interval and not a span between two
dates: a recipe's ``PT1H`` is an hour wherever it is cooked. A value whose
answer turns on which convention was used is a value this should not be
deciding, and no corpus here states one.
"""

_WRITTEN_UNITS = {
    "year": "y", "yr": "y", "month": "mo", "week": "w", "wk": "w", "day": "d",
    "hour": "h", "hr": "h", "h": "h", "minute": "mi", "min": "mi", "m": "mi",
    "second": "s", "sec": "s", "s": "s",
}  # fmt: skip


def _as_duration(value: Scalar) -> float | None:
    """One duration in seconds, written either way, or ``None``.

    ``None`` where the text is not a duration at all, so a caller can fall
    through to comparing it as text. A bare number is not one either: "5" is a
    count until something says what of.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    iso = _ISO_DURATION.match(text)
    if iso:
        years, months, weeks, days, hours, minutes, seconds = (float(part or 0) for part in iso.groups())
        return (
            years * _SECONDS["y"]
            + months * _SECONDS["mo"]
            + weeks * _SECONDS["w"]
            + days * _SECONDS["d"]
            + hours * _SECONDS["h"]
            + minutes * _SECONDS["mi"]
            + seconds * _SECONDS["s"]
        )
    found = _WRITTEN_DURATION.findall(text)
    if not found:
        return None
    # The whole text has to be duration parts. "1 hour 30 minutes" is a
    # duration; "ready in 5 minutes, serves 4" is prose that holds one, and
    # reading it as a duration would match it against any answer stating five
    # minutes of anything.
    if re.search(r"[A-Za-z0-9]", _WRITTEN_DURATION.sub(" ", text).replace("and", " ")):
        return None
    total = 0.0
    for amount, unit in found:
        key = _WRITTEN_UNITS.get(unit.lower()) or _WRITTEN_UNITS.get(unit.lower().rstrip("s"))
        if key is None:
            return None
        total += float(amount.replace(",", ".")) * _SECONDS[key]
    return total


def same_value(
    expected: Scalar,
    produced: Scalar,
    mode: MatchMode = MatchMode.EXACT,
    *,
    unit_match: UnitMatch = UnitMatch.EXACT,
) -> bool:
    """Whether a produced value is the expected answer."""
    if isinstance(expected, Quantity) or isinstance(produced, Quantity):
        if isinstance(expected, Quantity) and isinstance(produced, Quantity):
            return same_quantity(expected, produced, unit_match=unit_match)
        return False

    if expected is None or produced is None:
        return expected is None and produced is None

    if isinstance(expected, bool) or isinstance(produced, bool):
        return expected is produced

    expected_number = _as_number(expected)
    produced_number = _as_number(produced)
    if expected_number is not None and produced_number is not None:
        return same_number(expected_number, produced_number)

    expected_span = _as_duration(expected)
    produced_span = _as_duration(produced)
    if expected_span is not None and produced_span is not None:
        return expected_span == produced_span

    left = normalise_text(expected)
    right = normalise_text(produced)
    if mode is MatchMode.CONTAINS:
        return bool(left) and left in right
    return left == right
