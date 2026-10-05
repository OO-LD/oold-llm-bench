"""Rendering the class catalogue the model is shown.

A catalogue of bare identifiers asks a model to choose between names it has
been told nothing about. Under a readable vocabulary that measures whether the
names happen to be self-describing. Under an opaque one it measures nothing at
all, because the prompt carries no information to choose on. Neither is the
question the study is asking.

So the catalogue carries what the schema carries: the identifier, the human
label, the description, the parent, and the units the class admits. That is
the minimum a person would need, and the predecessor put it in context.

One thing is left out on purpose. Whatever annotation the document's signal was
drawn from is excluded, or the task collapses into matching one copy of a
sentence against another. That exclusion is declared per signal and recorded,
never decided here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from oold_llm_bench.corpus.quantities import QuantityKind
from oold_llm_bench.corpus.signals import Signal

__all__ = ["EXCLUDED_BY_SIGNAL", "CatalogueEntry", "load_entries", "render_catalogue"]

EXCLUDED_BY_SIGNAL: dict[Signal, frozenset[str]] = {
    Signal.NAMED: frozenset({"label"}),
    Signal.DESCRIPTION: frozenset({"description"}),
    Signal.ELUCIDATION: frozenset({"description"}),
    Signal.SYNONYM: frozenset({"label"}),
    Signal.TRANSLATED: frozenset({"label"}),
    Signal.UNIT: frozenset({"units"}),
}
"""Which catalogue field to withhold for each signal.

``ELUCIDATION`` withholds the description as well. The elucidation and the
QUDT description are different texts from different standards, but they define
the same concept closely enough that showing one while asking about the other
is closer to a paraphrase match than to a lookup.

``UNIT`` withholds the units, because a document that points at its class
through the unit would otherwise be answerable by reading the unit list.
"""


@dataclass(frozen=True)
class CatalogueEntry:
    """One class, as the prompt can show it."""

    identifier: str
    label: str = ""
    description: str = ""
    parent: str = ""
    units: tuple[str, ...] = ()

    def render(self, exclude: frozenset[str] = frozenset(), max_units: int = 0) -> str:
        """One entry, compact, carrying its own identifier.

        The identifier leads every entry, because it stays the answer however
        much else is shown.

        **What a schema enforces is always shown; what it annotates is
        optional.** A presented schema has to carry the class names, the
        property names, the enumerations, the patterns, because those are the
        answers the grammar will accept and a model asked to produce one it
        was never shown is not being tested, it is being guessed at. Labels,
        parents and descriptions carry nothing the grammar can check, so they
        exist only here and are the part a condition may withhold.

        The asymmetry this rule removes was real and it ran through every
        quantity number: the hundred class identifiers were listed in full
        while no unit identifier appeared at all, so class accuracy read
        0.88 to 0.99 and unit accuracy read 0.02. That is not a fact about
        models. It is a fact about what the prompt was given.

        ``max_units`` therefore defaults to 0, meaning all of them. It was 8,
        which printed "units: a, b, ... +11 more" while the grammar enforced
        the full enumeration, so even the described catalogue showed less
        than it would accept.
        """
        head = self.identifier
        if "label" not in exclude and self.label and self.label.replace(" ", "") != self.identifier:
            head += f" ({self.label})"
        if "parent" not in exclude and self.parent:
            head += f" < {self.parent}"
        lines = [f"- {head}"]
        if "description" not in exclude and self.description:
            lines.append(f"  {self.description}")
        if "units" not in exclude and self.units:
            shown = list(self.units if max_units <= 0 else self.units[:max_units])
            more = "" if max_units <= 0 or len(self.units) <= max_units else f", +{len(self.units) - max_units} more"
            lines.append(f"  units: {', '.join(shown)}{more}")
        return "\n".join(lines)


def load_entries(directory: Path, kinds: list[QuantityKind]) -> dict[str, CatalogueEntry]:
    """Read the label, description and parent each schema carries.

    Units come from the kinds, which resolve the parent chain: 505 of the 943
    schemas declare none of their own and inherit their parent's.
    """
    by_name = {kind.name: kind for kind in kinds}
    entries: dict[str, CatalogueEntry] = {}
    for path in sorted(directory.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        name = schema.get("title") or path.stem
        kind = by_name.get(name)
        parent = kind.parent if kind else None
        entries[name] = CatalogueEntry(
            identifier=name,
            label=((schema.get("x-oold-multilang-title") or {}).get("en") or "").strip(),
            description=(schema.get("description") or "").strip(),
            # The root carries every kind, so naming it says nothing.
            parent=parent if parent and parent != "QuantityValue" else "",
            units=tuple(kind.units) if kind else (),
        )
    return entries


def render_catalogue(
    identifiers: tuple[str, ...],
    entries: dict[str, CatalogueEntry],
    *,
    signal: Signal | None = None,
    alias: dict[str, str] | None = None,
    max_units: int = 0,
    annotations: bool = True,
) -> tuple[str, ...]:
    """The catalogue as the prompt shows it, one rendered entry per class.

    ``alias`` maps the corpus name to the identifier this condition answers
    to, so an opaque vocabulary is described by the same text a readable one
    gets. That is the whole point of describing the catalogue: the identifier
    stops being the only thing the model has to go on.

    ``annotations`` off keeps the identifier and the unit enumeration and
    drops the label, the parent and the description. That is the line the
    presented schema has to hold: an enumeration is what the grammar will
    accept, so a condition that enforces it must show it, while a
    description is checkable by nobody and is the part a condition may
    withhold. Before this the two moved together and the bare catalogue
    hid the units the grammar was enforcing.

    An identifier with no entry is rendered bare instead of being dropped,
    because dropping it would offer a smaller catalogue than the condition
    declares.
    """
    exclude = EXCLUDED_BY_SIGNAL.get(signal, frozenset()) if signal else frozenset()
    if not annotations:
        exclude = exclude | {"label", "parent", "description"}
    reverse = {v: k for k, v in (alias or {}).items()}
    rendered: list[str] = []
    for identifier in identifiers:
        source = reverse.get(identifier, identifier)
        entry = entries.get(source)
        if entry is None:
            rendered.append(f"- {identifier}")
            continue
        shown = CatalogueEntry(
            identifier=identifier,
            label=entry.label,
            description=entry.description,
            parent=entry.parent,
            units=entry.units,
        )
        rendered.append(shown.render(exclude, max_units=max_units))
    return tuple(rendered)
