# What the grammar does, what the prompt does, and what a tune buys

Measured 2026-10-04 on Wiki-Measurements and on schema.org.

Every condition is named by what it carried and what it enforced, so a table
can be read without a legend:

| name | Context | Enforced |
|---|---|---|
| `no-catalog-not-enforced` | the document alone | nothing |
| `no-catalog-enforced` | the document alone | `anyOf`, one branch per class with its units |
| `schema-prose-catalog-enforced` | the quantities, each with its unit list | same |
| `schema-prose-full-catalog-enforced` | the same plus descriptions | same |
| `schema-dump-catalog-enforced` | the JSON Schema printed out | same |
| `..-flat-enforced` | as above | class and unit as **independent** enumerations |

## The headline

Qwen3.5-9B, 120 tasks, the per-class grammar identical in every enforced row.

| name | context | base | tuned |
|---|---|---|---|
| `no-catalog-not-enforced` | 81 | **0.000** | 0.218 |
| `no-catalog-enforced` | **81** | **0.281** | **0.675** |
| `schema-prose-catalog-enforced` | 9,247 | **0.873** | 0.842 |
| `schema-prose-full-catalog-enforced` | 14,384 | 0.714 | 0.834 |

**A grammar carries the task with no catalogue at all.** Told nothing, the
base model scores zero; given a decoder that admits one branch per class, it
scores 0.281 on the same 81 tokens. It is never shown a class name.

**A tune more than doubles that.** 0.281 to 0.675, still on 81 tokens. This is
the fine-tuning claim in its strongest form: the adapter pays for a catalogue
that is not in the prompt.

**Neither replaces the catalogue.** 0.675 tuned on 81 tokens against 0.873
untuned on 9,247. The catalogue is worth about 0.2 and buying it with a tune
does not close the gap.

**Above the floor the tune inverts.** Where the catalogue is present, tuning
costs 0.03; on the 4B it costs more. An adapter is worth having only where the
prompt is withheld.

**More prose is worse.** `schema-prose-full` adds 5,000 tokens of descriptions
and the base drops from 0.873 to 0.714.

## Size predicts neither the floor nor the gain

Five models at the context floor, 81 tokens, union enforced, the same r=16
`all-linear` recipe and the same thousand examples throughout.

| model | params | base | tuned | gain |
|---|---|---|---|---|
| Qwen3.5-4B | 4B | 0.340 | 0.644 | +0.304 |
| **Qwen3.5-9B** | 9B | 0.283 | **0.792** | **+0.509** |
| Qwen3.5-27B | 27B | **0.227** | 0.583 | +0.356 |
| Phi-4 | 14B | **0.388** | 0.733 | +0.345 |
| DeepSeek-R1-Distill-Qwen-32B | 32B | 0.338 | 0.776 | +0.438 |

**Within one family the base score falls as the model grows**, 0.340 to 0.283
to 0.227 across 4B, 9B and 27B. The tuned score peaks in the middle, at the 9B.
So at this task neither half of the comparison is ordered by size, and the 27B
is the worst base and the worst tune of the three.

That bears on the `gpt-4.1-mini` gap below, which was left attributed to
"adapter type and model size". Size is now measured across a factor of eight
in two families and does not order the result, so what is left is the adapter
and the training, not the parameter count.

Read with the n beside it: these cells were served on rented and local
endpoints that time out, and a cell that timed out is absent rather than
zero. The Qwen3.5-27B rows lost 52 of 240 calls that way.

## It is not one model: a 32B of another family does the same

DeepSeek-R1-Distill-Qwen-32B, the same r=16 all-linear recipe and the same
thousand examples.

| name | context | base | tuned |
|---|---|---|---|
| `no-catalog-not-enforced` | 68 | **0.000** | **0.000** |
| `no-catalog-enforced` | **68** | **0.338** | **0.776** |
| `schema-prose-catalog-enforced` | 9,674 | **0.888** | 0.844 |

Same shape as the 9B and sharper: +0.438 at the context floor against the
9B's +0.394, and the same 0.04 cost once the catalogue is present.

The unenforced row is cleaner here. Both base and tuned score exactly zero
with no catalogue and no grammar, where the tuned 9B managed 0.218. So what
the adapter holds is not standalone knowledge, it is knowledge a grammar can
cash in: without the decoder to spend it against, the tune is worth nothing
at all.

Two families agreeing settles one thing the gpt-4.1-mini result could not.
Learning the vocabulary from a thousand examples is not a property of that
model or that provider; an open 32B does it too, at the same recipe.

## Which modules are adapted matters more than rank

Qwen3.5-9B at the context floor, 81 tokens, one axis moved at a time.

| adapter | F1 | over base |
|---|---|---|
| none | 0.284 | |
| r=16, Qwen module list | 0.675 | +0.391 |
| **r=16, `all-linear`** | **0.792** | **+0.508** |
| r=64, `all-linear` | 0.710 | +0.426 |

Changing `target_modules` from an explicit Qwen list to `all-linear` is worth
**+0.117**, larger than anything rank or exposure moved. Raising the rank
fourfold **costs** 0.082, the same direction the 4B showed, so capacity is not
what limits these adapters.

The switch was made for portability rather than for score: Phi-4 fuses
attention into `qkv_proj`, so a list naming Qwen's projections matches nothing
there and one recipe across architectures needed a rule that resolves on each.
It turned out to be the larger lever, which was luck.

So the floor figures elsewhere in this file are a floor twice over. The 9B's
true gain is **+0.508**, and the 4B, Phi-4 and DeepSeek rows already use
`all-linear` and are therefore the better ones.

## What hyperparameters were not swept

One rank, one step count, one learning rate, one adapter family. r=16,
1,000 steps, lr 2e-4, `all-linear`, chosen to be identical across
architectures rather than good for any of them.

No hyperparameter has been swept at the context floor, which is the only
condition where a tune changes the score. A sweep run where the prompt
withholds the unit vocabulary measures how fast an adapter memorises
identifiers the prompt is hiding, which is a different question.

Rank was swept at the floor and does not help. `target_modules` was changed
for portability and helped more than rank ever did. Exposure is untested
there: a 4,000 step run on the 9B is about seven hours and eighteen dollars
of rented A100, and about an hour on a local box.

## Enforcement against presentation, on the 4B

Same corpus, the unit enumerations present in every row.

| name | context | F1 |
|---|---|---|
| `schema-prose-catalog-not-enforced` | 9,515 | 0.32 |
| `schema-prose-catalog-flat-enforced` | 9,249 | **0.81** |
| `schema-dump-catalog-flat-enforced` | 15,569 | 0.77 |

Enforcement is worth **+0.49** at the same prompt length. Printing the schema
on top costs 6,300 tokens and 0.04: the catalogue already states the
enumerations in prose, and the JSON block restates them.

## Tuning, once the vocabulary is visible

Qwen3.5-4B at `schema-prose-catalog-flat-enforced`, 9,249 tokens throughout.

| adapter | F1 |
|---|---|
| none | **0.81** |
| 20,000 examples | 0.74 |
| r=16, 4,000 steps | 0.74 |
| r=64, 1,000 steps | 0.59 |
| r=16, 1,000 steps | 0.44 |

Every adapter is below the untuned model. The earlier conclusion that rank
and exposure were the axes was drawn when the prompt withheld the unit
vocabulary that the grammar was enforcing; with the vocabulary visible, the
tune has nothing left to supply and only costs generality.

## Nobody knows the identifiers cold

Unit accuracy with the identifier required from weights, no enumeration shown
and none enforced:

| model | unit |
|---|---|
| Qwen3.5-4B / 9B / 27B | 0.03 |
| Qwen3.8-27B | 0.02 |
| Llama-3.3-70B | 0.03 |
| Mistral-Small-2603 | 0.03 |
| gpt-oss-120b | 0.02 |
| Gemma-4-12B | **0.000** |
| gpt-4.1-mini | 0.012 |

Seven models, six families, five vendors, 4B to 120B. Class accuracy is
0.80-0.99 on the same prompt, so it is specifically the 2,864 QUDT
identifiers. Gemma names the right class 98.8% of the time and the right unit
never.

This is a fact about the prompt as much as the models: the catalogue listed a
hundred class names and no unit name. Asking for an identifier never shown
and not constrained is not a test of knowledge.

## gpt-4.1-mini, and the gap that is still unexplained

Tuned by Azure on the same generated quantity corpus, scored on
Wiki-Measurements, held-out class half, bare prompt both sides.

| | F1 | context |
|---|---|---|
| base | 0.006 | ~220 |
| **tuned** | **0.846** | ~220 |
| base, full catalogue | 0.850 | ~2,300 |

**+0.840 against our 4B's +0.081 at the same thousand examples.** Ruled out by
reading the training files: not the corpus, 1,000 rows each and 1,424,278
against ~1,430,792 trained tokens; and not the arm, since Azure ran with no
decode constraint while the Qwen runs had a shape grammar, so the open model
had more help and still did worse. What remains is adapter type and model
size, which the Phi-4, DeepSeek-32B and Qwen3.5-27B tunes exist to separate.

## schema.org: enforcement does not transfer

Gemma-4-12B, 400 tasks over 10 classes of human prose.

| condition | F1 | class | property |
|---|---|---|---|
| shown, not enforced | **0.26** | 0.87 | 0.50 |
| shown and flat-enforced | 0.23 | 0.88 | 0.46 |
| flat-enforced, not shown | **0.11** | 0.85 | 0.26 |

**Class selection works and property filling does not**: 0.85-0.88 against
0.26-0.50. The model knows the document describes a `Movie` and cannot fill
`Movie`'s slots.

**Enforcement hurts here**, which is the opposite of quantities. A quantity's
unit comes from a closed enumeration the grammar can supply; a schema.org
property value is open text, so the grammar constrains shape without
supplying anything, and withholding the schema costs 0.15. A corpus whose
values are open text does not hand enforcement the same win.

## Flat against union on real text: unresolved, and the reason is worth stating

Gemma-4-12B, 400 Wikidata-grounded schema.org entities, catalogue of 25
classes described in both arms. The only difference is whether the decoder
accepts a flat enumeration or one branch per class.

| condition | n | F1 | class | property |
|---|---|---|---|---|
| `schema-prose-full-catalog-flat-enforced` | 400 | 0.108 | **0.86** | 0.26 |
| `schema-prose-full-catalog-enforced` | **224** | 0.123 | 0.41 | **0.41** |

**The union arm lost 176 of its 400 calls**, 107 connection errors and 69
timeouts, where the flat arm lost none. So the two columns are not the same
tasks and the 0.015 between them is not a comparison. What the run does
establish is operational: a 25-branch discriminated union over schema.org
classes is at the edge of what this server will compile, and a grammar the
provider cannot build is a condition that cannot be measured rather than one
that scores badly.

The trade inside the completed cells is real and points the way the union was
meant to: property accuracy rises from 0.26 to 0.41, because a branch offers
only the properties its class declares. Class accuracy falls from 0.86 to
0.41, which is the opposite of what a discriminator should do and is the
thing to explain before this is rerun.

## What this does not say

One corpus family at a time. The quantity result rests on a closed unit
vocabulary; the schema.org result rests on open text. Neither generalises to
the other, which is the point of running both.

## The flat grammar's licence is taken one answer in seven

Every `-flat-enforced` number inherits an unsound decoder: two independent
enumerations, a hundred class names and six hundred and five units, so
`Altitude` with `kilogram` validates. Only the `enforced` rows use `anyOf`
over per-class branches, where that answer cannot be produced at all.

Counted over every answered entity in every flat-enforced cell,
**2,060 of 13,572, or 15.2%**, name a unit the produced class does not admit.

| model | arm | answers | impossible | rate |
|---|---|---|---|---|
| llama-3.3-70b | flat, schema shown | 697 | 95 | 13.6% |
| Qwen3.8-27B | flat, schema shown | 146 | 19 | 13.0% |
| mistral-small-2603 | flat, not shown | 160 | 19 | 11.9% |
| gpt-5-nano | flat, schema shown | 693 | 32 | 4.6% |

The grader marks those wrong, so the scores are honest. What the rate says is
about the arm: a condition named for enforcement was admitting an answer its
corpus forbids, at a rate worth stating beside any number measured under it.

`scripts/measure_flat_licence.py` recomputes it from the local records.

Pending: why a union halves class accuracy on schema.org, a rerun of that arm
on a server that compiles the grammar, and exposure at the context floor.
