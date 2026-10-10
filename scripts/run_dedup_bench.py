"""Run the dedup resolver with a real judge, over two of the three sequence
sources: the Wikidata merge-pair corpus, and a generated sequence.

Neither has been run with a judge before today. The cheap tiers were already
measured judge-free (131/270 correct, 72/146 of what they actually decided).
:func:`~oold_llm_bench.dedup.judge_of` is the one real-model judge in the
codebase: the playground's :class:`~oold_llm_bench.playground.identity.ModelJudge`
calls it too, so a session in the interface and a run of this script ask a
model the same question in the same words.

Variant 1, the generated sequence, has no corpus of hard negatives the way
the Wikidata corpus does: a draw gives one entity, not a pair. So one is
built here: for each drawn entity, two of its own per-document sightings are
the positive pair (same by construction), and its first sighting against a
freshly drawn entity of the same class is the negative (different by
construction, since the two draws share no identity and the generator's
nonsense-word values make a chance string collision negligible).

    uv run python scripts/run_dedup_bench.py --judge claude-haiku-4-5
    uv run python scripts/run_dedup_bench.py --judge claude-haiku-4-5 --variant wikidata
    uv run python scripts/run_dedup_bench.py --judge claude-haiku-4-5 --variant sequence --n 30
"""

from __future__ import annotations

import argparse
import random
import sys

from oold_llm_bench.corpus.wikidata_identity import EntityState, IdentityClass, Pair, Truth, read_pairs
from oold_llm_bench.dedup import Resolver, judge_of
from oold_llm_bench.grading.identity import score_identity


def _state_of(key: str, fields: dict[str, object], designator: str) -> EntityState:
    """One sequence document's fields, in the shape the resolver compares.

    There is no Wikidata item behind a generated entity, so the fields that
    exist only for one (statements, sitelinks, a real revision) take the
    values that make the rest of the resolver's logic true rather than
    merely present: the designator is the one label, every other stated
    field becomes a claim, and ``statements`` counts what was actually
    stated so the resolver's own entropy gate reads it correctly.
    """
    claims: dict[str, tuple[str, ...]] = {name: (str(value),) for name, value in fields.items() if name != "name"}
    return EntityState(
        qid=key,
        labels={"en": designator},
        aliases=(),
        description=None,
        sitelinks={},
        claims=claims,
        statements=len(claims),
        revision=None,
        as_of="generated",
    )


def sequence_pairs(n: int, seed: int) -> list[Pair]:
    from oold_llm_bench.corpus import SCHEMAORG, designating_slots, load_classes, resolve_module
    from oold_llm_bench.corpus.sequence import generate_sequence

    classes = load_classes(resolve_module(SCHEMAORG))
    # Needs a nameable slot as well as enough of them: draw_sequence refuses a
    # class offering none, since nothing would let a later document say it is
    # about the entity the first one introduced.
    usable = [cls for cls in classes if len(cls.slots) >= 6 and designating_slots(cls)]
    rng = random.Random(seed)  # noqa: S311 - a reproducible PRNG is the requirement, not a secure one

    pairs: list[Pair] = []
    for index in range(n):
        cls = rng.choice(usable)
        item = generate_sequence([cls], seed=seed * 1000 + index, n_slots=6, n_documents=2, overlap=1)
        left = _state_of(f"{item.key}-a{index}", item.per_document[0], item.expected_merge.mentions[0])
        right_same = _state_of(f"{item.key}-b{index}", item.per_document[1], item.expected_merge.mentions[0])
        pairs.append(
            Pair(
                id=f"seq-same-{index}",
                identity=IdentityClass.SAME,
                left=left,
                right=right_same,
                truth=Truth(
                    kind="generated", detail="two sightings of one drawn entity", recorded_at=None, recorded_by=None
                ),
                era=None,
                shares_string=True,
                shares_property=bool(item.overlapping),
                scored=True,
            )
        )
        decoy = generate_sequence([cls], seed=seed * 1000 + index + 500_000, n_slots=6, n_documents=1, overlap=0)
        right_diff = _state_of(f"{decoy.key}-c{index}", decoy.per_document[0], decoy.expected_merge.mentions[0])
        pairs.append(
            Pair(
                id=f"seq-diff-{index}",
                identity=IdentityClass.DIFFERENT,
                left=left,
                right=right_diff,
                truth=Truth(
                    kind="generated",
                    detail="two independently drawn entities of one class",
                    recorded_at=None,
                    recorded_by=None,
                ),
                era=None,
                shares_string=bool(left.strings & right_diff.strings),
                shares_property=bool(set(left.claims) & set(right_diff.claims)),
                scored=True,
            )
        )
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge", required=True, help="a model id from the catalogue")
    parser.add_argument("--variant", choices=["wikidata", "sequence", "both"], default="both")
    parser.add_argument("--n", type=int, default=20, help="sequence draws, each giving one same and one different pair")
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    from oold_llm_bench.clients import Credentials, build_client
    from oold_llm_bench.config import entry_for
    from oold_llm_bench.finetune.cli import load_env_file

    env_path = args.env or ".env"
    import pathlib

    if pathlib.Path(env_path).is_file():
        load_env_file(pathlib.Path(env_path))

    entry = entry_for(args.judge)
    credentials = Credentials.from_env(transports=[entry.transport])
    client = build_client(entry, credentials)
    resolver = Resolver(judge=judge_of(client))

    if args.variant in ("wikidata", "both"):
        corpus = read_pairs()
        sample = corpus.matched()
        print(f"wikidata merge-pair corpus: {len(sample)} pairs (balanced sample of {len(corpus.pairs)})")
        score = score_identity(sample, resolver, f"{args.judge} over wikidata identity")
        print(score)
        print()

    if args.variant in ("sequence", "both"):
        pairs = sequence_pairs(args.n, seed=1)
        print(f"generated sequence corpus: {len(pairs)} pairs ({args.n} same, {args.n} different)")
        score = score_identity(pairs, resolver, f"{args.judge} over generated sequences")
        print(score)

    return 0


if __name__ == "__main__":
    sys.exit(main())
