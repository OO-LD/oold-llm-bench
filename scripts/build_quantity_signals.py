"""Derive the identifying signals for each quantity kind from QUDT.

A task has to point at a quantity kind without naming it, or class selection
collapses into a substring match. The unit does that for 124 of the 437 kinds
that declare their own units, and for none of the 505 subclasses, which
inherit their parent's units by construction.

So the other signals come from QUDT itself: a label in another language, and
the authored description. Both are human-written and citable, which is why
this is preferred over generating text with a model that might later be
evaluated on it.

Run this when the QUDT pin changes. The output is committed, so a run needs
no network.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

QUDT_VERSION = "3.1.5"
QUDT_QUANTITYKIND = f"https://qudt.org/{QUDT_VERSION}/vocab/quantitykind"
QUDT_SCHEMA = "http://qudt.org/schema/qudt/"

EMMO_VERSION = "1.0.4"
EMMO_COMMIT = "12492654851d51d427d33de9f569abf3d76490d3"
EMMO_BASE = f"https://raw.githubusercontent.com/emmo-repo/EMMO/{EMMO_COMMIT}/"
EMMO_FILES = ("disciplines/isq.ttl", "disciplines/metrology.ttl")
"""The same commit the OO-LD reference schemas pin, so the two agree.

EMMO earns its place three ways. Its ``elucidation`` is an ISO 80000 derived
definition, a second authored description independent of QUDT's. Its
``altLabel`` is an English synonym, which no translated label can be. And its
class IRIs are opaque, so ``EMMO_cd2cd0de_e0cc_4ef1_b27e_2e88db027bac`` names
Length without a token any model could have memorised, which is a real
alternative vocabulary, not a scrambled one.
"""

EMMO_NS = "https://w3id.org/emmo#"

WANTED_LANGUAGES = ("de", "fr", "it", "es", "pt", "pl", "ja", "zh")
"""Languages other than English. English is excluded on purpose, because an
English label is the class name with spaces in it and leaks the answer."""


def _flat(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def leaks(name: str, text: str) -> bool:
    """Whether a text gives the class name away.

    Checked on the flattened form, so "Absorbed Dose Rate" is caught for
    ``AbsorbedDoseRate``. Measured over QUDT: 485 of 807 descriptions do.
    """
    return _flat(name) in _flat(text)


def build_emmo() -> dict[str, dict]:
    """EMMO material for each quantity kind, keyed by its QUDT local name.

    The join is EMMO's own ``qudtReference``, so nothing is matched by string
    similarity. Both http and https forms appear in the source, hence the
    split on the last path segment instead of a prefix test.
    """
    import rdflib

    graph = rdflib.Graph()
    for name in EMMO_FILES:
        graph.parse(EMMO_BASE + name, format="turtle")

    emmo = rdflib.Namespace(EMMO_NS)
    skos = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
    out: dict[str, dict] = {}

    for subject, _, reference in graph.triples((None, emmo.qudtReference, None)):
        kind = str(reference).rstrip("/").rsplit("/", 1)[-1]
        if not kind:
            continue
        iri = str(subject).rsplit("#", 1)[-1].rsplit("/", 1)[-1]

        def texts(predicate, subject=subject):
            return [str(o) for o in graph.objects(subject, predicate) if isinstance(o, rdflib.Literal)]

        pref = next(iter(texts(skos.prefLabel) or texts(emmo.prefLabel)), None)
        entry = {
            "iri": iri,
            "pref_label": pref,
            "alt_labels": sorted({
                text for text in texts(skos.altLabel) + texts(emmo.altLabel) if not leaks(kind, text)
            }),
            "elucidation": next(
                (text for text in texts(emmo.elucidation) if not leaks(kind, text)),
                None,
            ),
            "iso": next(iter(texts(emmo.ISO80000Reference)), None),
        }
        existing = out.get(kind)
        if existing is None or (not existing["elucidation"] and entry["elucidation"]):
            out[kind] = entry
    return out


def build(out: Path) -> dict:
    import rdflib

    graph = rdflib.Graph()
    graph.parse(QUDT_QUANTITYKIND, format="turtle")
    schema = rdflib.Namespace(QUDT_SCHEMA)
    emmo_by_kind = build_emmo()

    kinds: dict[str, dict] = {}
    subjects = {s for s in graph.subjects() if "quantitykind" in str(s)}
    for subject in sorted(subjects, key=str):
        name = str(subject).rsplit("/", 1)[-1]
        labels: dict[str, str] = {}
        for label in graph.objects(subject, rdflib.RDFS.label):
            if isinstance(label, rdflib.Literal) and label.language:
                tag = label.language.split("-")[0]
                if tag in WANTED_LANGUAGES and tag not in labels:
                    labels[tag] = str(label)

        description = next(
            (str(o) for o in graph.objects(subject, schema.plainTextDescription) if isinstance(o, rdflib.Literal)),
            None,
        )
        symbol = next(
            (str(o) for o in graph.objects(subject, schema.symbol) if isinstance(o, rdflib.Literal)),
            None,
        )
        entry = {
            "labels": {tag: text for tag, text in sorted(labels.items()) if not leaks(name, text)},
            "description": description if description and not leaks(name, description) else None,
            "symbol": symbol,
            "emmo": emmo_by_kind.get(name),
        }
        if entry["labels"] or entry["description"] or entry["emmo"]:
            kinds[name] = entry

    payload = {
        "source": QUDT_QUANTITYKIND,
        "qudt_version": QUDT_VERSION,
        "emmo_version": EMMO_VERSION,
        "emmo_commit": EMMO_COMMIT,
        "built_at": datetime.now(UTC).date().isoformat(),
        "languages": list(WANTED_LANGUAGES),
        "note": (
            "English labels and any text containing the class name are "
            "excluded, because they would hand the answer to the model."
        ),
        "kinds": kinds,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/oold_llm_bench/data/quantity_signals.json"),
    )
    args = parser.parse_args()
    payload = build(args.out)
    kinds = payload["kinds"]
    with_label = sum(1 for k in kinds.values() if k["labels"])
    with_description = sum(1 for k in kinds.values() if k["description"])
    with_emmo = sum(1 for k in kinds.values() if k.get("emmo"))
    with_alt = sum(1 for k in kinds.values() if (k.get("emmo") or {}).get("alt_labels"))
    with_eluc = sum(1 for k in kinds.values() if (k.get("emmo") or {}).get("elucidation"))
    print(f"wrote {args.out}")
    print(f"  kinds with a usable signal : {len(kinds)}")
    print(f"  with a translated label    : {with_label}")
    print(f"  with a QUDT description    : {with_description}")
    print(f"  with an EMMO class         : {with_emmo}")
    print(f"  with an EMMO synonym       : {with_alt}")
    print(f"  with an EMMO elucidation   : {with_eluc}")


if __name__ == "__main__":
    main()
