# Method

## No judge

An answer is reduced to triples of (entity, property, value) and compared
with the triples the task was built from. The primary metric is F1 over
those. Nothing anywhere asks a model whether an answer is good.

Ten dimensions are scored separately, because a single number cannot say
which half of a task failed:

| dimension | what it counts |
|---|---|
| `entity` | precision and recall over entities, reported apart |
| `class` | the class chosen for each entity |
| `property` | the slots filled |
| `value` | the literal, compared after parsing |
| `unit` | the unit identifier, exactly |
| `unit_physical` | the unit after conversion, so 1.0 m and 100 cm agree |
| `shortlist` | whether a select step offered the true class at all |
| `duplicate` | one entity emitted twice, distinct from one invented |
| `provenance` | whether a value is traceable to the document |
| `grounded` | whether a value appears in the document |

Exact unit matching is the default and physical equality is reported beside
it. Making leniency the default would land the benefit on the arm with no
enumeration, which is the one it must not help.

Entities are aligned between produced and expected by bipartite matching, not
by order, so a correct answer in a different order is a correct answer.

## Four controls, and what each would hide

Preflight runs them before any budget is spent and refuses the grid if one
scores above its ceiling. Each catches a different way for a grader to be
rewarding shape.

| control | answers | ceiling |
|---|---|---|
| empty | nothing | 0.01 |
| random | a class at random, with an invented value | 0.01 |
| wrong document | a different document, correctly | 0.01 |
| spelling | the number and the words beside it, underscored | **0.25** |

**The spelling control has its own ceiling, and that is the point.** The
other three answer something wrong, so any score at all is a defect. This one
answers the document without reading a vocabulary, and on real text it cannot
reach zero: the hand-authored ELN notes score 0.06 on a single note reading
"4.7 ohm", where the identifier is `ohm`. That is language, not a defect.

What the 0.25 separates is a corpus where a few readings happen to spell
their own identifier from one where that is how the corpus is written. A
corpus that writes a unit as its identifier scores **0.96** on this control,
and no other control detects it.

A control that cannot fail is worse than no control. Two shapes of that are
guarded against in `_answers_for`: a control reading only `value` degenerates
to `{"value": null}` on a corpus whose instances carry no value, and a
control that returns nothing scores zero for the right-looking reason.

## Statistical discipline

The design is paired: the same tasks across arms, temperature 0.

**Repetitions buy nothing on a deterministic model and are required on a
reasoning one.** Measured over two runs of the same cell forty-five minutes
apart: claude-haiku 120 of 120 identical, llama-3.3-70b 118 of 120,
gpt-5-nano **101 of 120**, where 19 tasks flipped and the cell moved 0.092.
So no gpt-5-nano comparison below 0.09 is claimed from one run.

McNemar, exact, two-sided, alpha 0.05:

| effect | tasks for 80% power |
|---|---|
| +0.30 | 50 |
| +0.20 | 110 |
| +0.15 | 180 |
| +0.10 | 380 |

**Minimum 120 tasks per cell** for any comparison reported, and no comparison
below +0.20 without 180.

A cell that times out leaves no record. It is absent, not zero, and the `n`
beside a number says how many survived.

## Splits

Held out by class, not by document. Roughly half the describable pool is
excluded from training entirely, and at evaluation the offered catalogue
still carries both halves, or a held-out class could not be chosen and the
contrast could not be measured. The split is committed corpus metadata, not
recomputed per run.

Training seeds are disjoint from evaluation seeds, so no scored document was
trained on.

The register is held out too: adapters train on generated prose and are
scored on Wikipedia sentences. `training_match` records how a cell's
condition relates to what the model was trained on, including the case where
the condition has no field for a key the adapter declares, so a
quantities-trained adapter scored on another corpus cannot report `matched`.

## What the corpus states about itself

A document that names its own class and property names makes selection a
substring match, and an arm comparison cannot see past it. The schema.org
corpus has an `IMPLIED` labelling mode for that reason: 0 of 120 class names
and 0 of 480 property names leak under it, against 120 and 480 without. It
costs the describable pool, 122 classes down to 19.

The same discipline applies to surface form. A corpus that writes a unit as
its QUDT identifier is testing a spacing transformation and not a
normalisation, which is what the spelling control at 0.96 was detecting.
`Notation.WRITTEN` writes units as QUDT publishes them, with both forms
injective and the round trip asserted over the whole pool.
