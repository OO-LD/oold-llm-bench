# Two arms on one document

What changes between arms, shown rather than described. Captured by wrapping
`client.invoke`, so every message and every reply below is verbatim from a
live run and nothing is reconstructed. The same treatment per pipeline is in
[pipelines](pipelines/index.md).

## 1. Real text, llama-3.3-70b, presented against enforced

Wiki-Measurements, one Wikipedia sentence, CC BY-SA 4.0. One call each. The
two arms differ in one thing: the enforced one sends a `response_format` that
pins the unit slot to the QUDT enumeration, the other does not.

### Document

```
Slættaratindur (English: "Flat peak") is the highest mountain in the Faroe Islands, at an elevation of 880 metres.
```

### Expected

```json
[
  {
    "key": "q1",
    "class": "Altitude",
    "fields": {
      "value": "880.0 meter"
    }
  }
]
```

### Presented, not enforced (`schema-prose-catalog-not-enforced-gated`)

**system message** (12701 chars)

```
Read the document and report every entity it describes, with the values stated for each.

Answer with JSON only. No explanation, no markdown fence, no commentary.

Choose the class of each entity from this list, using one of these exactly:
- AmountOfSubstanceConcentration (Amount of Substance of Concentration) < Concentration
  "Amount of Substance of Concentration of B" is defined as the amount of a constituent divided by the volume of the mixture.
  units: mole_per_meter_cubed, femto_mole_per_liter, pico_mole_per_meter_cubed, pico_mole_per_liter, nano_mole_per_liter, milli_mole_per_meter_cubed, micro_mole_per_liter, milli_mole_per_liter, +4 more
- AbsorbedDoseRate
  "Absorbed Dose Rate" is the absorbed dose of ionizing radiation imparted at a given location per unit of time (second, minute, hour, or day).
  units: gray_per_second, nano_sievert_per_hour, micro_sievert_per_hour, nano_gray_per_second, milli_sievert_per_hour, micro_gray_per_second, sievert_per_hour, milli_gray_per_second, +3 more
- AbsoluteActivity < InverseVolume
  The "Absolute Activity" is the exponential of the ratio of the chemical potential to $RT$ where $R$ is the gas constant and $T$ the thermodynamic temperature.
  units: per_meter_cubed, becquerel_second_per_meter_cubed, per_liter, per_centi_meter_cubed, per_milli_liter, per_milli_meter_cubed
- Absorptance
  Absorptance is the ratio of the radiation abs

[... 11301 of 12701 chars of system message cut, the rest is more catalogue entries ...]
```

**user message**

```
Slættaratindur (English: "Flat peak") is the highest mountain in the Faroe Islands, at an elevation of 880 metres.
```

**response_format**: none

**reply**

```json
{
  "entities": [
    {
      "type": "Altitude",
      "value": 880,
      "unit": "metres"
    }
  ]
}
```

### Presented and enforced (`schema-dump-catalog-flat-enforced`)

**system message** (24048 chars)

```
Read the document and report every entity it describes, with the values stated for each.

Answer with JSON only. No explanation, no markdown fence, no commentary.

Choose the class of each entity from this list, using one of these exactly:
- AmountOfSubstanceConcentration (Amount of Substance of Concentration) < Concentration
  "Amount of Substance of Concentration of B" is defined as the amount of a constituent divided by the volume of the mixture.
  units: mole_per_meter_cubed, femto_mole_per_liter, pico_mole_per_meter_cubed, pico_mole_per_liter, nano_mole_per_liter, milli_mole_per_meter_cubed, micro_mole_per_liter, milli_mole_per_liter, +4 more
- AbsorbedDoseRate
  "Absorbed Dose Rate" is the absorbed dose of ionizing radiation imparted at a given location per unit of time (second, minute, hour, or day).
  units: gray_per_second, nano_sievert_per_hour, micro_sievert_per_hour, nano_gray_per_second, milli_sievert_per_hour, micro_gray_per_second, sievert_per_hour, milli_gray_per_second, +3 more
- AbsoluteActivity < InverseVolume
  The "Absolute Activity" is the exponential of the ratio of the chemical potential to $RT$ where $R$ is the gas constant and $T$ the thermodynamic temperature.
  units: per_meter_cubed, becquerel_second_per_meter_cubed, per_liter, per_centi_meter_cubed, per_milli_liter, per_milli_meter_cubed
- Absorptance
  Absorptance is the ratio of the radiation abs

[... 22648 of 24048 chars of system message cut, the rest is more catalogue entries ...]
```

**user message**

```
Slættaratindur (English: "Flat peak") is the highest mountain in the Faroe Islands, at an elevation of 880 metres.
```

**response_format**: 7428 chars, unit slot shown below

```json
n the document."
          },
          "unit": {
            "type": "string",
            "description": "The unit the magnitude is given in.",
            "enum": [
              "Ha",
              "a_sidereal",
              "astronomical_unit",
              "atto_joule",
              "atto_joule_second",
              "atto_second",
              "bar",
              "bar_absolute",
              "bar_liter_per_second",
              "becquerel",
              "becquerel_per_liter",
              "becquerel_per_meter_cubed",
              "becquerel_second_per_meter_cubed",
              "cal_to_the_15_degree_celsius",
              "calorie",
              "centi_bar",
              "centi_gray",
              "centi_meter",
              "centi_meter_per_second_squared",
              "centi_mole",
              "centi_mole_per_liter",
              "day",
              "day_sidereal",
              "deca_meter",

```

**reply**

```json
{
  "entities": [
    {
      "type": "Altitude",
      "value": 880,
      "unit": "meter"
    }
  ]
}
```

**Unenforced answered `metres`, enforced answered `meter`.** Only the second
is a QUDT identifier, so the first scores 0 on this task and the second 1.
That single pair is the
+0.79 llama gain in miniature.

## 2. A linked document from the grid, two models, two orchestrations

Task `e29` of the 120 linked schema.org tasks that `e3`, `e4` and `e6` all ran.
Chosen because both classes are ones a reader already knows: a book with an
isbn, a page count and an edition, illustrated by a person. 22 of the 120
tasks have two familiar classes and this is the clearest of them.

### Document

```
The entry below was filed as book. torvale-marton appears near the top. It belongs to the book edition field. The isbn on file is QG-54546. The rest of the page was left blank. The value 5977 was entered. Its label, further down, is number of pages. illustrator: see the entry for Ossquel Tavdros.

Logged as person. The form has nervale-marrith written on it, and the heading above it is address. As for the family name, it is Ossquel Tavdros. The file was closed the same afternoon. Copied across: Female. The heading for it reads gender.
```

### Expected

```json
[
  {
    "key": "e1",
    "class": "Book",
    "fields": {
      "bookEdition": "torvale-marton",
      "isbn": "QG-54546",
      "numberOfPages": 5977,
      "illustrator": "-> e2"
    }
  },
  {
    "key": "e2",
    "class": "Person",
    "fields": {
      "address": "nervale-marrith",
      "familyName": "Ossquel Tavdros",
      "gender": "Female"
    }
  }
]
```

### What each run scored

| model | orchestration | F1 | the `illustrator` link |
|---|---|---|---|
| claude-haiku-4-5 | single shot | 0.923 | absent |
| claude-haiku-4-5 | multi_step | 0.857 | `e2` |
| **Qwen3.8-27B** | single shot | **0.280** | absent |
| **Qwen3.8-27B** | multi_step | **1.000** | `person-entry` |

### multi_step on claude-haiku-4-5, four calls

**call 1, detect** (20477 chars of system message)

```json
{
  "entities": [
    {
      "id": "e1",
      "mention": "torvale-marton",
      "candidates": [
        "Book"
      ]
    },
    {
      "id": "e2",
      "mention": "Ossquel Tavdros",
      "candidates": [
        "Person"
      ]
    }
  ]
}
```

**call 2, choose properties** (2114 chars of system message)

```json
{
  "fillable": {
    "e1": [
      "name",
      "isbn",
      "numberOfPages",
      "illustrator"
    ],
    "e2": [
      "address",
      "familyName",
      "gender"
    ]
  }
}
```

**call 3, extract e1** (21678 chars of system message)

```json
{
  "entities": [
    {
      "type": "Book",
      "id": "e1",
      "name": "torvale-marton",
      "isbn": "QG-54546",
      "numberOfPages": 5977,
      "illustrator": "e2"
    },
    {
      "type": "Person",
      "id": "e2",
      "name": "Ossquel Tavdros"
    }
  ]
}
```

**call 4, extract e2** (21553 chars of system message)

```json
{
  "entities": [
    {
      "type": "Person",
      "id": "e2",
      "familyName": "Ossquel Tavdros",
      "address": "nervale-marrith",
      "gender": "Female"
    }
  ]
}
```

**assembled**

```json
{
 "entities": [
  {
   "type": "Book",
   "id": "e1",
   "name": "torvale-marton",
   "isbn": "QG-54546",
   "numberOfPages": 5977,
   "illustrator": "e2"
  },
  {
   "type": "Person",
   "id": "e2",
   "familyName": "Ossquel Tavdros",
   "address": "nervale-marrith",
   "gender": "Female"
  }
 ]
}
```

### multi_step on Qwen3.8-27B, four calls

**call 1, detect** (20477 chars of system message)

```json
{
  "entities": [
    {
      "id": "book-entry",
      "candidates": ["Book"],
      "mention": "filed as book"
    },
    {
      "id": "person-entry",
      "candidates": ["Person"],
      "mention": "Logged as person"
    }
  ]
}
```

**call 2, choose properties** (2132 chars of system message)

```json
{
  "fillable": {
    "book-entry": [
      "bookEdition",
      "isbn",
      "numberOfPages",
      "illustrator"
    ],
    "person-entry": [
      "address",
      "familyName",
      "gender"
    ]
  }
}
```

**call 3, extract e1** (22051 chars of system message)

```json
{
  "entities": [
    {
      "type": "Book",
      "bookEdition": "torvale-marton",
      "isbn": "QG-54546",
      "numberOfPages": 5977,
      "illustrator": "person-entry",
      "id": "book-entry"
    }
  ]
}
```

**call 4, extract e2** (21802 chars of system message)

```json
{
  "entities": [
    {
      "type": "Person",
      "address": "nervale-marrith",
      "familyName": "Ossquel Tavdros",
      "gender": "Female",
      "id": "person-entry"
    }
  ]
}
```

**assembled**

```json
{
 "entities": [
  {
   "type": "Book",
   "bookEdition": "torvale-marton",
   "isbn": "QG-54546",
   "numberOfPages": 5977,
   "illustrator": "person-entry",
   "id": "book-entry"
  },
  {
   "type": "Person",
   "address": "nervale-marrith",
   "familyName": "Ossquel Tavdros",
   "gender": "Female",
   "id": "person-entry"
  }
 ]
}
```

### single shot, the one call

**claude-haiku-4-5**

```json
{
 "entities": [
  {
   "type": "Book",
   "bookEdition": "torvale-marton",
   "isbn": "QG-54546",
   "numberOfPages": 5977
  },
  {
   "type": "Person",
   "address": "nervale-marrith",
   "familyName": "Ossquel Tavdros",
   "gender": "Female"
  }
 ]
}
```

**Qwen3.8-27B**

```json
{
 "entities": [
  {
   "type": "Book",
   "alternateName": null,
   "name": null,
   "url": null,
   "address": null,
   "faxNumber": null,
   "isAccessibleForFree": null,
   "logo": null,
   "aggregateRating": null,
   "hasCertification": null,
   "keywords": null,
   "review": null,
   "speakable": null,
   "abstract": null,
   "accessibilityAPI": null,
   "accessibilityControl": null,
   "accessibilityFeature": null,
   "accessibilityHazard": null,
   "accessibilitySummary": null,
   "accessMode": null,
   "actionPlatform": "GenericWebPlatform",
   "contentType": null,
   "encodingType": null,
   "httpMethod": null,
   "urlTemplate": null,
   "actionApplication": null,
   "doorTime": null,
   "duration": null,
   "endDate": null,
   "eventAttendanceMode": "OfflineEventAttendanceMode",
   "eventStatus": "EventScheduled",
   "maximumAttendeeCapacity": null,
   "maximumPhysicalAttendeeC
```

### What this shows

**Qwen single shot emits 157 properties and fills almost none of them.** The
flat schema admits every property any offered class declares, and
`provider.prepare` makes them all required, so a grammar forces a value for
each one. The model spends its output on nulls and the content is lost: 0.280.
Narrowed to the four properties the document actually states, the same model
scores 1.000.

That is the clearest case in this work for narrowing before filling, and it is
the architecture the orchestration axis exists to test: select the class,
select the fillable properties, construct the narrowed schema, fill it.

**It also inverts on haiku**, which wins at single shot and loses at
multi_step. Both are the same rule: an orchestration helps the model whose
plan is better than its one-shot answer, and costs the model whose is not.
haiku's multi_step files the edition under `name` and pays a property and a
value for it; its single shot drops the link instead.
