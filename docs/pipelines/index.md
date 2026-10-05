# Pipelines

What each orchestration sends, step by step, on one real task. Every prompt,
every enforced schema and every reply below is verbatim from a live call,
captured by `oold-bench chains` and regenerable:

```
oold-bench chains --model gpt-5-nano --arm catalog-enforced \
  --orchestration select_then_fill --out docs/pipelines/select_then_fill.md
```

Long messages are cut and say how much was cut. A described catalogue is
tens of thousands of characters of the same shape, and printing it whole
hides the thing a page is for, which is what differs between two pipelines.

| page | steps | what it is for |
|---|---|---|
| [single shot](single_shot.md) | extract | One call. The baseline the others have to beat. |
| [select then fill](select_then_fill.md) | select, fill | Shortlist the class, then fill the schema that shortlist implies. The general answer where subclasses add properties. |
| [segmented](segmented.md) | plan, fill | Plan the whole document, then fill the plan. The ids the plan hands out are what makes an edge expressible. |
| [multi step](multi_step.md) | detect, properties, extract | Segmented with a property step in the middle, which closes which slots exist at all. |

`recursive` has no page. It is declared in the enforcement axis and is not
implemented in the runner, so there is nothing to record.

## Two of these pages show a pipeline failing

Kept rather than recaptured, because both failures are properties of the
pipeline and not accidents of the day.

**The planning step can answer with the schema.** On the first capture of
[segmented](segmented.md), gpt-5-nano replied with the selection schema
itself, with the value spliced into it, and the cell produced nothing. The
page now runs with `--plan-retry`, and the second `plan` call in it is the
retry doing its work. A single-shot arm has no call whose answer is a
structure, so it cannot fail this way.

**The property step can drop the slot the task is about.** In
[multi step](multi_step.md) the detect step names `Altitude` and the property
step does not choose `value`, so the extract step has nowhere to put 1750.
The answer carries the right class and the right unit and no magnitude, and
scores zero. That is the cost of a step that narrows before it reads: it
makes an entity unanswerable downstream, which is why a two-step failure has
to be attributable to a step.

Each page is one task. What these pipelines score is in the
[findings](../findings.md).
