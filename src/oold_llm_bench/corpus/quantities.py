"""Tasks generated from the QUDT quantity corpus by round trip.

Generate a typed instance against a schema, render it to prose, and keep the
instance as the answer. Ground truth is then correct by construction rather
than by annotation, difficulty is a parameter and not a property of
whatever documents happened to be available, and the supply is unlimited.

The unit enumeration makes this corpus worth leading with. A closed
list of units per quantity kind means a wrong unit is detectable, not merely
implausible, so unit correctness becomes a dimension the grader can report
instead of an error class it has to guess at.

Notation is the second axis, and here it is not a refinement at the margin.
The only field a quantity instance carries is the reading, so the value
dimension is the primary metric by definition and the unit is most of what
that dimension is: measured on haiku with no decode constraint, 0.897 primary, 0.897 value and
0.900 unit. Under :attr:`Notation.CANONICAL` the unit reaches the page as the
identifier with its underscores replaced by spaces, so the headline number of
every quantity result so far is substantially a number about de-underscoring.
Under :attr:`Notation.WRITTEN` the document writes ``km`` and ``J/K`` while
the answer stays ``kilo_meter`` and ``joule_per_kelvin``, which is what
extraction is: a surface form resolved to a canonical identifier.

A quantity document writes two things, and these are the decisions each took.

*Where the unit's surface form comes from.* QUDT publishes it. Each schema's
``@context`` says which QUDT unit an enumeration member denotes, ``kilo_meter``
is ``qunit:KiloM``, and QUDT gives ``KiloM`` the symbol ``km``. Nothing is
authored here, and that is the whole reason this is worth doing. UCUM is a
second published form and is declined: ``A.h`` and ``m.s-1`` are a machine
code, so writing them would swap one canonical identifier for another and
leave the fault in place; they are absent for 97 of the units where a symbol
is absent for 1; and a unit carrying two of them, ``A.h.kg-1`` and ``A.h/kg``,
would need a choice this module has no grounds to make.

*Coverage, and what happens to a unit with no symbol.* Of the 15,358 unit
slots the 942 kinds with units carry, 13,843 have a published symbol and
12,629 of those are writable, 82.2%, reaching 1,104 of the 1,291 distinct unit
names. A unit that cannot be written is dropped from the draw rather than
written canonically, because a mode that falls back silently in a fifth of its
documents measures the fallback. A kind keeps the units it can write: 397 keep
all of them, 466 keep some, 79 keep none and leave the pool, and
:func:`writable` is where that is counted. Against the pools the signals
already form, 942 named kinds become 863, the 112 unit-identifiable ones
become 108, and the 323 elucidated become 281.

*Ambiguity.* None, measured rather than assumed. The 1,104 writable names
carry 1,116 distinct symbols and no symbol is claimed by two names, inside a
kind's enumeration or across the corpus, so a symbol maps back to exactly one
canonical identifier. :func:`load_unit_symbols` builds that inverse and refuses
a table where it does not hold, because a shared symbol would break the primary
metric rather than a dimension reported beside it. Three classes of published
symbol are refused as unwritable instead, and :data:`UNWRITABLE` is that list.

*The magnitude.* Digits grouped in threes, ``76,037.1179``, and nothing else.
Grouping is typography, so deleting the separators returns the value exactly.
Rounding does not return it at all and a decimal comma means a different number
under a convention no prompt states, so both are declined on injectivity and
not on taste. ``babel`` would supply the locale conventions and is not a
dependency; it is not needed either, because ``format(value, ",")`` is the
whole of the injective part, while ``babel.numbers.format_decimal`` rounds to
three fraction digits by default and that is the part that cannot be used.

*The grader.* Unchanged, and that is a finding rather than an omission. The
expected unit is still the canonical name, :attr:`UnitMatch.EXACT` still
compares it to what the model produced, and neither side of that comparison is
what the document now spells, so the existing machinery accepts a correct
answer without a spelling-variant list and without ``pint``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from oold_llm_bench.corpus.signals import Signal, SignalData, Vocabulary, load_signals
from oold_llm_bench.grading.triples import Quantity
from oold_llm_bench.tasks.models import (
    CorpusRef,
    Difficulty,
    ExpectedInstance,
    Source,
    Split,
    TaskRecord,
)

__all__ = [
    "UNWRITABLE",
    "Notation",
    "QuantityKind",
    "Signal",
    "UnitSymbols",
    "Vocabulary",
    "canonical_number",
    "describable",
    "generate_task",
    "load_kinds",
    "load_unit_symbols",
    "render",
    "translatable",
    "unambiguous",
    "unit_identifiable",
    "writable",
    "written_number",
]


@dataclass(frozen=True)
class QuantityKind:
    """One quantity kind from the corpus, with the units it allows."""

    name: str
    units: tuple[str, ...]
    """The units this kind may carry, inherited from its parent when it does
    not declare its own."""
    own_units: bool = True
    """Whether the enumeration is declared here or inherited. A kind
    that inherits cannot be identified by its unit, because its parent and
    every sibling share it."""
    parent: str | None = None
    description: str = ""
    uuid: str = ""

    @property
    def has_units(self) -> bool:
        return bool(self.units)


def load_kinds(directory: Path) -> list[QuantityKind]:
    """Read the corpus, resolving each kind's effective unit enumeration.

    Of 943 schemas, 437 declare their own units and 505 are subclasses that
    inherit a parent's. Reading only the declared enumeration would drop every
    subclass, which is 505 of the corpus and includes ``Diameter``,
    ``Radius`` and ``Wavelength``, so the chain is followed instead.

    A subclass therefore has units, and its units are its parent's. That is
    exactly why no unit can identify it, and why a subclass can only be
    pointed at by a translated label or a description.
    """
    declared: dict[str, tuple[str, ...]] = {}
    parents: dict[str, str] = {}
    raw: dict[str, dict] = {}

    for path in sorted(directory.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        name = schema.get("title") or path.stem
        raw[name] = schema
        units = ((schema.get("properties") or {}).get("unit") or {}).get("enum") or []
        declared[name] = tuple(u for u in units if isinstance(u, str))
        for entry in schema.get("allOf") or []:
            ref = entry.get("$ref", "")
            if ref.endswith(".schema.json"):
                parents[name] = ref[: -len(".schema.json")]

    def effective(name: str, seen: frozenset[str] = frozenset()) -> tuple[str, ...]:
        if declared.get(name) or name in seen:
            return declared.get(name, ())
        parent = parents.get(name)
        if not parent or parent not in declared:
            return ()
        return effective(parent, seen | {name})

    return [
        QuantityKind(
            name=name,
            units=effective(name),
            own_units=bool(declared[name]),
            parent=parents.get(name),
            description=(schema.get("description") or "").strip(),
            uuid=schema.get("x-oold-uuid", ""),
        )
        for name, schema in raw.items()
    ]


def _readable(unit: str) -> str:
    """The unit as a sentence would spell it."""
    return unit.replace("_", " ")


class Notation(str, Enum):
    """Whether a document spells a value as the schema does or as a note does.

    ``CANONICAL`` is what every measured result rests on, so it is the default
    and the frames behind it are never touched. ``WRITTEN`` writes the unit
    with the symbol QUDT publishes and groups the magnitude's digits, so the
    answer stays the canonical identifier and reaching it takes a
    normalisation rather than a substring.
    """

    CANONICAL = "canonical"
    WRITTEN = "written"


UNWRITABLE: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("a UCUM annotation rather than part of the symbol", lambda symbol: "{" in symbol),
    ("a scaling factor, which runs into the magnitude", lambda symbol: symbol.strip().isdigit()),
    ("a mark for the absence of a unit", lambda symbol: symbol.strip() in {"一", "#"}),
)
"""Why a published symbol is still not what a document writes.

Measured over the 1,137 symbols the corpus draws: 13 are annotated, ``cal{IT}``
and ``a{tropical}``; 6 are numeric, ``10`` through ``1000000000000``, and
"76,037.1179 1000" leaves no reader able to say where the number ends; 2 stand
for no unit at all, QUDT's ``一`` for the dimensionless unit and ``#`` for a
count. The remaining 1,116 are written exactly as published, and that is the
point of taking them from a vocabulary instead of authoring them.
"""

_SYMBOLS = Path(__file__).resolve().parent.parent / "data" / "unit_symbols.json"


@dataclass(frozen=True)
class UnitSymbols:
    """The symbol each unit of the corpus is published under."""

    symbols: dict[str, str]
    """Unit name to symbol, for the names that mean one QUDT unit everywhere."""
    by_kind: dict[str, dict[str, str]]
    """Kind to unit name to symbol, for the names that do not. ``meter`` is
    ``m`` under ``Length`` and ``m²/m`` under ``AreaPerLength``, so the kind
    has to be known before the symbol can be. 11 names of the 1,125, over the
    138 kinds that carry one."""
    names: dict[str, str]
    """Symbol back to unit name, the inverse the answer is recovered by."""
    source: str

    def published(self, kind: str, unit: str) -> str | None:
        """The symbol QUDT publishes for this unit within this kind."""
        return (self.by_kind.get(kind) or {}).get(unit) or self.symbols.get(unit)

    def written(self, kind: str, unit: str) -> str | None:
        """The symbol a document may write, ``None`` where none may be."""
        symbol = self.published(kind, unit)
        if symbol is None or any(rejects(symbol) for _, rejects in UNWRITABLE):
            return None
        return symbol

    def canonical(self, symbol: str) -> str | None:
        """The unit name a written symbol means."""
        return self.names.get(symbol)

    def written_units(self, kind: str, units: tuple[str, ...]) -> tuple[str, ...]:
        """The members of one enumeration a document can write."""
        return tuple(unit for unit in units if self.written(kind, unit))


@lru_cache(maxsize=1)
def load_unit_symbols(path: Path | None = None) -> UnitSymbols:
    """Read the committed symbol table. Cached, because it never changes.

    The inverse is built here and refused if two unit names share a symbol.
    A shared symbol would leave a document with no answer a grader could
    accept, and on this corpus the unit is the primary metric, so the check
    belongs where the table is read rather than in a report nobody runs.
    """
    payload = json.loads((path or _SYMBOLS).read_text(encoding="utf-8"))
    symbols: dict[str, str] = payload["symbols"]
    by_kind: dict[str, dict[str, str]] = payload["by_kind"]

    names: dict[str, str] = {}
    pairs = list(symbols.items()) + [pair for mapping in by_kind.values() for pair in mapping.items()]
    for unit, symbol in pairs:
        claimed = names.setdefault(symbol, unit)
        if claimed != unit:
            raise ValueError(f"symbol {symbol!r} is claimed by both {claimed} and {unit}")
    return UnitSymbols(symbols=symbols, by_kind=by_kind, names=names, source=payload["source"])


def written_number(value: float) -> str:
    """The magnitude as a document writes it: digits grouped in threes.

    The one number convention that is typography and not precision, so it
    stays injective: deleting the separators gives the value back exactly.
    Rounding does not, a decimal comma needs a locale the prompt never states,
    and both would leave tasks whose gold value is unreachable from the page.
    """
    return f"{value:,}"


def canonical_number(text: str) -> float:
    """The magnitude behind a written number. The inverse of the above."""
    return float(text.replace(",", ""))


def writable(kinds: list[QuantityKind], symbols: UnitSymbols | None = None) -> list[QuantityKind]:
    """The kinds a written document can be built from.

    The analogue of :func:`~oold_llm_bench.corpus.schemaorg.implied_classes`:
    the price of the mode, paid in the pool where it can be counted rather
    than in a ceiling that moves per document. A kind keeps every unit it can
    write and loses the rest, and only a kind left with none drops out.
    """
    table = symbols if symbols is not None else load_unit_symbols()
    return [kind for kind in kinds if table.written_units(kind.name, kind.units)]


def readable_kind(name: str) -> str:
    """A class name as prose, so ``IonicStrength`` reads "ionic strength".

    Lowercasing the class name whole produces "ionicstrength", which no
    document contains, and which would hand the model a token it can only
    match by guessing the corpus convention instead of by reading.
    """
    words: list[str] = []
    current = ""
    for index, char in enumerate(name):
        previous = name[index - 1] if index else ""
        starts_word = char.isupper() and (previous.islower() or previous.isdigit())
        ends_acronym = char.isupper() and previous.isupper() and index + 1 < len(name) and name[index + 1].islower()
        if current and (starts_word or ends_acronym):
            words.append(current)
            current = char
        else:
            current += char
    if current:
        words.append(current)
    return " ".join(words).lower()


_PLAIN = (
    "The {kind} was {value} {unit}.",
    "A {kind} of {value} {unit} was recorded.",
    "{kind}: {value} {unit}.",
)

_SEPARATED = (
    "The {kind} came to {value}, measured in {unit}.",
    "Reported {kind} was {value}. The unit throughout is {unit}.",
)

_LABELLED_PLAIN = (
    "The {kind} was {value} {unit}.",
    "{kind}: {value} {unit}.",
)
"""Templates for a foreign-language label. The indefinite article is left out
because "a elektrische Leitfaehigkeit" reads as a generation artefact and
invites a model to treat the phrase as noise."""

_EMBEDDED = (
    "During the run the {kind} settled at {value} {unit}, which was within "
    "the expected band, and no further adjustment was made.",
    "After the third attempt the operator noted a {kind} of {value} {unit} "
    "and moved on to the next step without comment.",
)

_IMPLIED_PLAIN = (
    "The reading was {value} {unit}.",
    "Recorded {value} {unit}.",
    "The instrument showed {value} {unit}.",
)

_IMPLIED_SEPARATED = (
    "The reading came to {value}, in {unit}.",
    "The figure was {value}. Everything on this sheet is in {unit}.",
)

_IMPLIED_EMBEDDED = (
    "During the run the instrument settled at {value} {unit}, which was "
    "within the expected band, and no further adjustment was made.",
    "After the third attempt the operator wrote down {value} {unit} and moved on to the next step without comment.",
)


def unit_identifiable(kinds: list[QuantityKind]) -> list[QuantityKind]:
    """The kinds every one of whose units names no other kind.

    These are the ones an implied task can be built from without the answer
    becoming arguable. Measured over the corpus: 124 of 437, because a unit
    like ``per_meter`` is shared by ten kinds and ``unitless`` by seventy-two.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for kind in kinds:
        for unit in kind.units:
            owners[unit].add(kind.name)
    unique = {unit for unit, names in owners.items() if len(names) == 1}
    return [kind for kind in kinds if kind.own_units and kind.has_units and all(u in unique for u in kind.units)]


def translatable(kinds: list[QuantityKind], data: SignalData) -> list[QuantityKind]:
    """Kinds QUDT gives a non-English label. Reaches subclasses."""
    return [k for k in kinds if data.supports(k.name, Signal.TRANSLATED)]


def describable(kinds: list[QuantityKind], data: SignalData) -> list[QuantityKind]:
    """Kinds QUDT describes without naming them."""
    return [k for k in kinds if data.supports(k.name, Signal.DESCRIPTION)]


def identifier_for(kind: QuantityKind, vocabulary: Vocabulary, data: SignalData) -> str | None:
    """The name a kind answers to under one vocabulary.

    ``OSW`` comes from the schema's own identifier, which every schema has,
    so it is the opaque vocabulary that covers the whole corpus.
    """
    if vocabulary is Vocabulary.OSW:
        return f"OSW{kind.uuid.replace('-', '')}" if kind.uuid else None
    return data.identifier(kind.name, vocabulary)


def candidates(
    kinds: list[QuantityKind],
    signal: Signal,
    data: SignalData,
    vocabulary: Vocabulary = Vocabulary.CONSENSUS,
) -> list[QuantityKind]:
    """The kinds a task can be built from under one signal and vocabulary.

    An EMMO vocabulary also needs the kind to carry an EMMO identifier, since
    without one there is nothing for the answer to be graded against.
    """
    kinds = [k for k in kinds if identifier_for(k, vocabulary, data)]
    if signal is Signal.UNIT:
        return unit_identifiable(kinds)
    with_units = [k for k in kinds if k.has_units]
    if signal is Signal.NAMED:
        return with_units
    if signal is Signal.TRANSLATED:
        return unambiguous(translatable(with_units, data), signal, data)
    usable = [k for k in with_units if data.supports(k.name, signal)]
    return unambiguous(usable, signal, data)


def _signal_text(kind: str, signal: Signal, data: SignalData) -> str | None:
    """The words a document uses to point at this kind under one signal."""
    if signal is Signal.ELUCIDATION:
        return data.elucidations.get(kind)
    if signal is Signal.DESCRIPTION:
        return data.descriptions.get(kind)
    if signal is Signal.SYNONYM:
        return " ".join(data.synonyms.get(kind) or ())
    if signal is Signal.TRANSLATED:
        return " ".join(sorted((data.labels.get(kind) or {}).values()))
    return None


def unambiguous(kinds: list[QuantityKind], signal: Signal, data: SignalData) -> list[QuantityKind]:
    """Drop kinds that share their signal text with another kind.

    Two kinds described by the same sentence cannot both be the answer, so no
    arm can be right on them more than half the time and the loss is the
    corpus, not the model. Measured over the elucidation pool: 2 texts shared
    by 2 kinds each, so 4 of 347 go.
    """
    seen: dict[str, list[str]] = {}
    for kind in kinds:
        text = (_signal_text(kind.name, signal, data) or "").strip().casefold()
        if text:
            seen.setdefault(text, []).append(kind.name)
    shared = {name for names in seen.values() if len(names) > 1 for name in names}
    return [k for k in kinds if k.name not in shared]


def spell_reading(
    kind: str,
    value: float,
    unit: str,
    notation: Notation = Notation.CANONICAL,
    symbols: UnitSymbols | None = None,
) -> tuple[float | str, str]:
    """The magnitude and the unit as the document will carry them.

    Under :attr:`Notation.CANONICAL` the magnitude is interpolated as the
    float it is, not as a string of it, so the frames see exactly what they
    always saw.
    """
    if notation is Notation.CANONICAL:
        return value, _readable(unit)
    table = symbols if symbols is not None else load_unit_symbols()
    symbol = table.written(kind, unit)
    if symbol is None:
        raise ValueError(f"no document could write {unit} for {kind}")
    return written_number(value), symbol


def render(
    kind: str,
    value: float,
    unit: str,
    difficulty: Difficulty,
    rng: random.Random,
    signal: Signal = Signal.UNIT,
    data: SignalData | None = None,
    language: str | None = None,
    notation: Notation = Notation.CANONICAL,
    symbols: UnitSymbols | None = None,
) -> str:
    """Render one measurement as the prose a model will read.

    Difficulty moves two things that matter and nothing that does not. Harder
    prose puts distance between the number and its unit, and surrounds both
    with text that carries no answer. It never makes the answer ambiguous,
    because a task whose gold value is arguable measures the annotator.

    ``signal`` decides what points at the quantity kind, ``notation`` how the
    reading itself is spelled.
    """
    written, spelled = spell_reading(kind, value, unit, notation, symbols)
    if difficulty is Difficulty.EASY:
        anonymous, labelled = _IMPLIED_PLAIN, _PLAIN
    elif difficulty is Difficulty.MEDIUM:
        anonymous, labelled = _IMPLIED_SEPARATED, _SEPARATED
    else:
        anonymous, labelled = _IMPLIED_EMBEDDED, _EMBEDDED

    pointer = pointer_text(kind, signal, data, language)
    if pointer is None:
        return rng.choice(anonymous).format(value=written, unit=spelled)

    # A definition is a sentence of its own, so it precedes the measurement
    # instead of standing where a short label would.
    if signal in (Signal.DESCRIPTION, Signal.ELUCIDATION):
        measurement = rng.choice(anonymous).format(value=written, unit=spelled)
        return f"{pointer} {measurement}"

    pool = labelled
    if signal is not Signal.NAMED and difficulty is Difficulty.EASY:
        pool = _LABELLED_PLAIN
    return rng.choice(pool).format(kind=pointer, value=written, unit=spelled)


def pointer_text(
    kind: str,
    signal: Signal,
    data: SignalData | None = None,
    language: str | None = None,
) -> str | None:
    """The words that point at a quantity kind under one signal.

    ``None`` for :attr:`Signal.UNIT`, which points with the unit alone and so
    has nothing to say here.
    """
    if signal is Signal.UNIT:
        return None
    if signal is Signal.NAMED:
        return readable_kind(kind)
    if data is None:
        raise ValueError(f"a {signal.value} signal needs its source data")

    if signal is Signal.TRANSLATED:
        labels = data.labels.get(kind) or {}
        tag = language or (sorted(labels)[0] if labels else None)
        if not tag or tag not in labels:
            raise ValueError(f"no label for {kind} in {language!r}")
        return labels[tag]

    sources = {
        Signal.DESCRIPTION: lambda: data.descriptions.get(kind),
        Signal.SYNONYM: lambda: next(iter(data.synonyms.get(kind) or ()), None),
        Signal.ELUCIDATION: lambda: data.elucidations.get(kind),
    }
    text = sources[signal]()
    if not text:
        raise ValueError(f"no {signal.value} for {kind}")
    return text


def render_entry(
    kind: str,
    value: float,
    unit: str,
    signal: Signal,
    data: SignalData | None = None,
    language: str | None = None,
    notation: Notation = Notation.CANONICAL,
    symbols: UnitSymbols | None = None,
) -> str:
    """One line of a list of quantity values.

    Several measurements are a list, not a run of sentences, because
    repeating "the reading was" once per entity reads as a generation
    artefact, and because a record of measurements is a list.

    Nothing here attaches a reading to a subject. That is deliberate for now:
    the grader aligns each reading by its own value and unit, so with no
    subject there is nothing to confuse, and order stops mattering. Attaching
    a measurement to a sample comes later, and the schema.org corpus already
    carries that shape, a price belonging to a product.
    """
    written, spelled = spell_reading(kind, value, unit, notation, symbols)
    pointer = pointer_text(kind, signal, data, language)
    return f"{pointer}: {written} {spelled}" if pointer else f"{written} {spelled}"


def _magnitude(rng: random.Random, difficulty: Difficulty) -> float:
    if difficulty is Difficulty.EASY:
        return round(rng.uniform(1, 100), 1)
    if difficulty is Difficulty.MEDIUM:
        return round(rng.uniform(0.01, 1000), 3)
    return round(rng.uniform(1e-4, 1e5), 4)


def generate_task(
    kinds: list[QuantityKind],
    *,
    task_id: str,
    seed: int,
    entries: dict[str, Any] | None = None,
    difficulty: Difficulty = Difficulty.EASY,
    n_entities: int = 1,
    split: Split = Split.DEV,
    catalogue: tuple[str, ...] | None = None,
    signal: Signal = Signal.UNIT,
    vocabulary: Vocabulary = Vocabulary.CONSENSUS,
    language: str | None = None,
    data: SignalData | None = None,
    draw_from: list[QuantityKind] | None = None,
    notation: Notation = Notation.CANONICAL,
    symbols: UnitSymbols | None = None,
) -> TaskRecord:
    """One task: some measurements rendered to prose, and what they were.

    The seed is the whole of the randomness, so a task id and a seed
    reproduce the document byte for byte.

    ``signal`` defaults to :attr:`Signal.UNIT`, because a document that
    names its own classes makes class selection a substring match and no arm
    comparison can see past that. ``notation`` defaults to
    :attr:`Notation.CANONICAL`, because every measured result was taken under
    it and a corpus that moves underneath a run makes the runs incomparable.
    """
    data = data if data is not None else load_signals()
    table = (symbols if symbols is not None else load_unit_symbols()) if notation is Notation.WRITTEN else None
    # ``draw_from`` narrows what the answer may be while ``kinds`` still
    # supplies the catalogue, its units and its rendered entries. The two
    # differ whenever half the corpus is held out of something: the offered
    # list has to keep both halves or a held-out kind cannot be chosen.
    usable = candidates(draw_from if draw_from is not None else kinds, signal, data, vocabulary)
    if language and signal is Signal.TRANSLATED:
        usable = [k for k in usable if language in data.labels.get(k.name, {})]
    # Applied after the signal has chosen its pool, never before, so the
    # conditions the existing pools rest on are the same conditions and the
    # cost of the mode is one number: the kinds that leave.
    if table is not None:
        usable = writable(usable, table)
    if not usable:
        raise ValueError(
            f"no quantity kind is usable for signal {signal.value}" + (f" in {language}" if language else "")
        )
    if n_entities > len(usable):
        raise ValueError(f"asked for {n_entities} entities, corpus offers {len(usable)}")

    rng = random.Random(seed)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one
    chosen = rng.sample(usable, n_entities)

    parts: list[str] = []
    expected: list[ExpectedInstance] = []
    for index, kind in enumerate(chosen, start=1):
        unit = rng.choice(kind.units if table is None else table.written_units(kind.name, kind.units))
        value = _magnitude(rng, difficulty)
        parts.append(
            render(kind.name, value, unit, difficulty, rng, signal, data, language, notation, table)
            if n_entities == 1
            else render_entry(kind.name, value, unit, signal, data, language, notation, table)
        )
        expected.append(
            ExpectedInstance(
                key=f"q{index}",
                class_path=identifier_for(kind, vocabulary, data) or kind.name,
                fields={"value": Quantity(magnitude=value, unit=unit)},
            )
        )

    document = parts[0] if n_entities == 1 else "Measurements:\n" + "\n".join(f"- {line}" for line in parts)
    # Units per offered class, so an arm that constrains decoding can close
    # the unit slot too. Keyed on the identifiers the catalogue uses, because
    # that is what a catalogue trim works on.
    units: dict[str, list[str]] | None = None
    if catalogue:
        offered = set(catalogue)
        units = {}
        for kind in kinds:
            identifier = identifier_for(kind, vocabulary, data) or kind.name
            if identifier in offered and kind.units:
                units.setdefault(identifier, sorted(set(kind.units)))
    # What the prompt may say about each offered class, with whatever the
    # document's signal was drawn from withheld. Showing the model the same
    # sentence the document was built from would make the task a paraphrase
    # match instead of a lookup.
    described: dict[str, str] | None = None
    enums: dict[str, str] | None = None
    if catalogue and entries:
        from oold_llm_bench.corpus.catalogue import render_catalogue

        alias = {k.name: identifier_for(k, vocabulary, data) or k.name for k in kinds}
        rendered = render_catalogue(tuple(catalogue), entries, signal=signal, alias=alias)
        described = dict(zip(catalogue, rendered, strict=True))
        bare = render_catalogue(tuple(catalogue), entries, signal=signal, alias=alias, annotations=False)
        enums = dict(zip(catalogue, bare, strict=True))

    return TaskRecord(
        id=task_id,
        document=document,
        expected=expected,
        corpus=CorpusRef(
            name="quantities",
            source=Source.SYNTHETIC,
            document_id=task_id,
            content_hash=hashlib.sha256(document.encode("utf-8")).hexdigest(),
        ),
        split=split,
        difficulty=difficulty,
        notes=f"signal={signal.value},vocabulary={vocabulary.value}"
        + (f",language={language}" if language else "")
        + (f",notation={notation.value}" if notation is not Notation.CANONICAL else ""),
        catalogue=list(catalogue) if catalogue else None,
        unit_catalogue=units,
        catalogue_text=described,
        catalogue_enums=enums,
    )
