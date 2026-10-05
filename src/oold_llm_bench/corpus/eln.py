"""Tasks read from the hand-authored ELN notes.

Every number this benchmark has produced was measured on sixteen sentence
frames, which means an enforcement effect and a phrasing effect are not yet
separable. These notes are the register check: the same quantity kinds and the
same unit enumeration, written the way a notebook writes them, with the
reading sitting inside a sentence that carries no answer.

Twenty notes is a check and not a corpus, and nothing here pretends otherwise.
Every record carries :attr:`~oold_llm_bench.tasks.models.Source.MANUAL`, so a
report can keep these out of any pooled synthetic number from the record
alone.

Ground truth is annotated here and not constructed, which is the one worry the
generated corpora do not have. A class or a unit this corpus cannot name is an
error in the annotation, so the loader raises instead of dropping the note:
twenty notes quietly becoming twelve tasks is the kind of thing that later
gets reported as a result. The only way a note is left out is the annotation
declaring it unscorable and saying why, which :func:`read_notes` reports.

Trimming the catalogue and ordering it stay where they already are, in
:mod:`oold_llm_bench.runner.adapter`, so a note is offered exactly what a
synthetic task under the same condition is offered and the two read side by
side.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from oold_llm_bench.corpus.catalogue import CatalogueEntry, render_catalogue
from oold_llm_bench.corpus.quantities import QuantityKind
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
    "NOTES_PATH",
    "Note",
    "NoteFile",
    "Reading",
    "load_notes",
    "read_notes",
    "resolve_unit",
]

NOTES_PATH = Path(__file__).resolve().parent.parent / "data" / "eln_notes.json"

_DEGREE = "degree"


@dataclass(frozen=True)
class Reading:
    """One measurement a note states, spelled as the annotation spelled it."""

    class_name: str
    magnitude: float
    unit: str


@dataclass(frozen=True)
class Note:
    """One note, before anything has been resolved against the corpus."""

    id: str
    text: str
    readings: tuple[Reading, ...]
    skip: str = ""
    """Why the annotation declares this note unscorable, empty when it does
    not. A note excluded for any other reason is a bug, not a decision."""

    @property
    def scorable(self) -> bool:
        return not self.skip


@dataclass(frozen=True)
class NoteFile:
    """The notes and what the person who wrote them recorded about them."""

    source: str
    written: str
    note: str
    notes: tuple[Note, ...]

    @property
    def declared_skips(self) -> tuple[Note, ...]:
        return tuple(note for note in self.notes if note.skip)


def read_notes(path: Path | None = None) -> NoteFile:
    """Read the notes file without consulting the corpus.

    Separate from :func:`load_notes` so the count that was written down and
    the count that became tasks can be compared, which is the whole of the
    guard against a silent loss.
    """
    payload = json.loads((path or NOTES_PATH).read_text(encoding="utf-8"))
    notes = tuple(
        Note(
            id=entry["id"],
            text=entry["text"],
            readings=tuple(
                Reading(class_name=r["class"], magnitude=float(r["value"]), unit=r["unit"]) for r in entry["expected"]
            ),
            skip=(entry.get("skip") or "").strip(),
        )
        for entry in payload["notes"]
    )
    return NoteFile(
        source=payload.get("source", ""),
        written=payload.get("written", ""),
        note=payload.get("note", ""),
        notes=notes,
    )


def _flat(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _unit_key(unit: str) -> str:
    """A unit reduced to what two spellings of it have in common.

    The leading "degree" goes because QUDT writes "degree Celsius" where the
    schemas write ``Celsius``. Over the 1291 units in the corpus this folds no
    two units of one kind together, so a match stays a match and not a guess.
    """
    flat = _flat(unit)
    return flat[len(_DEGREE) :] if flat.startswith(_DEGREE) and flat != _DEGREE else flat


def resolve_unit(kind: QuantityKind, unit: str) -> str | None:
    """The identifier a written unit answers to within one kind's enumeration.

    ``None`` when nothing in the enumeration answers to it, which is a unit
    the grader would compare against and the catalogue would never offer.
    """
    if unit in kind.units:
        return unit
    key = _unit_key(unit)
    matches = {candidate for candidate in kind.units if _unit_key(candidate) == key}
    return matches.pop() if len(matches) == 1 else None


def _spellings(
    kinds: list[QuantityKind],
    entries: dict[str, CatalogueEntry] | None,
) -> tuple[dict[str, QuantityKind], dict[str, str]]:
    """The names a note may use for a kind, mapped to the kind it names.

    A note is written by a person, and a person writes a label. ``Pressure``
    happens to be the identifier and ``Volume Flow Rate`` is not, so the label
    each schema carries is admitted as a second spelling. A spelling two kinds
    answer to is dropped rather than guessed at, because the wrong class
    quietly assigned is worse than a note that refuses to load.
    """
    by_name = {kind.name: kind for kind in kinds}
    spellings: dict[str, set[str]] = {}
    for kind in kinds:
        spellings.setdefault(_flat(kind.name), set()).add(kind.name)
    for name, entry in (entries or {}).items():
        if entry.label and name in by_name:
            spellings.setdefault(_flat(entry.label), set()).add(name)
    return by_name, {key: next(iter(names)) for key, names in spellings.items() if len(names) == 1}


def _resolve(
    note: Note,
    by_name: dict[str, QuantityKind],
    by_spelling: dict[str, str],
    offered: tuple[str, ...],
) -> tuple[list[ExpectedInstance], list[str]]:
    """What one note expects, or what stopped it resolving."""
    found: list[ExpectedInstance] = []
    failures: list[str] = []
    for index, reading in enumerate(note.readings, start=1):
        kind = by_name.get(reading.class_name) or by_name.get(by_spelling.get(_flat(reading.class_name), ""))
        if kind is None:
            failures.append(f"{note.id} names the class {reading.class_name!r}, which the corpus does not have")
            continue
        unit = resolve_unit(kind, reading.unit)
        if unit is None:
            failures.append(f"{note.id} gives {reading.unit!r}, which is not a unit of {kind.name}")
            continue
        if kind.name not in offered:
            failures.append(f"{note.id} needs {kind.name}, which the catalogue does not offer")
            continue
        found.append(
            ExpectedInstance(
                key=f"q{index}",
                class_path=kind.name,
                fields={"value": Quantity(magnitude=reading.magnitude, unit=unit)},
            )
        )
    return found, failures


def load_notes(
    kinds: list[QuantityKind],
    *,
    path: Path | None = None,
    entries: dict[str, CatalogueEntry] | None = None,
    catalogue: tuple[str, ...] | None = None,
    split: Split = Split.DEV,
    difficulty: Difficulty = Difficulty.HARD,
) -> list[TaskRecord]:
    """The notes as tasks, refusing to load rather than dropping one.

    Every class and every unit is resolved through the same corpus the
    catalogue is built from and the grader compares against, so a note that
    resolves is a note an arm can answer and be scored on. Anything that does
    not resolve is reported together with everything else that did not, since
    fixing an annotation one raised error at a time is how a corpus loses
    notes.

    ``difficulty`` is :attr:`~oold_llm_bench.tasks.models.Difficulty.HARD`
    because that is what the register is. The synthetic corpus calls a
    document hard when the reading sits inside a sentence that carries no
    answer, and a notebook writes nothing else.
    """
    source = read_notes(path)
    by_name, by_spelling = _spellings(kinds, entries)
    offered = tuple(catalogue) if catalogue else tuple(kind.name for kind in kinds if kind.has_units)

    resolved: list[tuple[Note, list[ExpectedInstance]]] = []
    problems: list[str] = []
    for note in source.notes:
        found, failures = _resolve(note, by_name, by_spelling, offered)
        if note.skip:
            # A marker that hides a working note is a silent loss with a
            # sentence in front of it, so it is refused the same way.
            if not failures:
                problems.append(f"{note.id} is declared unscorable as {note.skip!r} but resolves")
            continue
        problems.extend(failures)
        if not failures:
            resolved.append((note, found))
    if problems:
        raise ValueError("the ELN notes do not resolve against the quantity corpus: " + "; ".join(problems))

    units = {kind.name: sorted(set(kind.units)) for kind in kinds if kind.name in set(offered) and kind.units}
    # Nothing is withheld from the catalogue. The synthetic corpus hides
    # whatever its document was generated from, and these documents were
    # generated from nothing, so there is no sentence to match a copy of.
    described = dict(zip(offered, render_catalogue(offered, entries), strict=True)) if entries else None

    return [
        TaskRecord(
            id=note.id,
            document=note.text,
            expected=expected,
            corpus=CorpusRef(
                name="eln",
                source=Source.MANUAL,
                document_id=note.id,
                content_hash=hashlib.sha256(note.text.encode("utf-8")).hexdigest(),
            ),
            split=split,
            difficulty=difficulty,
            notes=f"corpus=eln,source={source.source},written={source.written}",
            catalogue=list(offered),
            unit_catalogue=units,
            catalogue_text=described,
        )
        for note, expected in resolved
    ]
