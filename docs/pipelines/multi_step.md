## multi_step under `schema-prose-catalog-enforced`

`gpt-5-nano`, 3 call(s), 25 classes on offer. Arm `catalog-enforced` in the records.

### Document

```
Arak is surrounded by mountains in the south, west, and east, and its average altitude is about 1750m above sea level.
```

corpus=wiki-measurements,dataset=Wiki-Measurements/raw/wiki-measurements_large_strict.json,doi=10.5281/zenodo.14858280,notation=written,licence=CC BY-SA 4.0

### Expected

```json
[
  {
    "key": "q1",
    "class": "Altitude",
    "fields": {
      "value": "1750.0 meter"
    }
  }
]
```

### Step 1: detect

**system message** (8938 chars)

```
Read the document and say, for every entity it describes, which classes it could be. An entity is a thing the document describes, not a statement about one. A document that gives four properties of one thing describes one entity, not four, so list a thing once however many times the document mentions it, and do not list a property, a value or a field name as an entity of its own.

Answer with JSON only. For each entity give a short id, the classes it could belong to, most likely first, and the words that identify it in the document.

List at most 3 classes per entity. Fewer is better when you are sure. A class you leave out cannot be chosen later.

Choose the class of each entity from this list, using one of these exactly:
- ActivityCoefficient
  units: unitless
- AlphaDisintegrationEnergy
  units: joule, electron_volt, atto_joule, Ha, kilo_electron_volt, femto_joule, mega_electron_volt, pico_joule, giga_electron_volt, nano_joule, micro_joule, milli_joule, second_watt, calorie, cal_to_the_15_degree_celsius, international_calorie, mean_calorie, kilo_joule, hour_volt_ampere, hour_watt, kilo_calorie, kilo_international_calorie, mega_joule, hour_kilo_volt_ampere, hour_kilo_watt, giga_joule, hour_mega_volt_ampere, hour_mega_watt, metric_ton_per_force_pound, tera_joule, giga_watt_hour, peta_joule, hour_tera_watt, exa_joule, quad
- AbsoluteHumidity
  units: gram_per_liter, femto_gram_per_liter, nano_gram_per_meter_cubed, pico_gram_per_liter, nano_gram_per_liter, pico_gram_per_milli_liter, micro_gram_per_meter_cubed, nano_gram_per_deci_liter, milli_gram_per_meter_cubed, nano_gram_p

[... 7338 of 8938 characters cut, the rest is more of the same ...]
```

**user message**

```
Arak is surrounded by mountains in the south, west, and east, and its average altitude is about 1750m above sea level.
```

**enforced schema** (2075 chars, as a request-body field)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "EntityCandidates",
  "description": "Which classes each entity in the document could be.",
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string",
            "description": "A short identifier for this entity, e.g. e1."
          },
          "candidates": {
            "type": "array",
            "maxItems": 3,
            "items": {
              "type": "string",
              "enum": [
                "ActivityCoefficient",
                "AlphaDisintegrationEnergy",
                "AbsoluteHumidity",
                "AmbientPressure",
                "Admittance",
                "AccelerationOfGravity",
                "Action",
                "Acceleration",
                "AmountOfSubstance",
                "Adaptation",
                "Activity",
                "AbsorbedDose",
                "ActionTime",
                "AcousticImpedance",
                "AmountOfSubstanceConcentration",
                "Altitude",
                "AbsoluteTypographicMeasurement",
                "AbsoluteActivity",
                "AbsorbedDoseRate",
                "ActivePower",
                "ActivityConcentration",
                "Acidity",
                "ActiveEnergy",
                "Absorptance",
                "AcceptorIonizationEnergy"
              ]
            },
            "description": "Classes this entity could be, most likely first."


[... 475 of 2075 characters cut, the rest is more of the same ...]
```

**reply**

```
{"entities":[{"id":"e1","candidates":["Altitude"],"mention":"Arak is surrounded by mountains in the south, west, and east, and its average altitude is about 1750m above sea level."}]}
```

### Step 2: properties

**system message** (569 chars)

```
Read the document and say, for each entity listed below, which of its own properties the document gives a value for.

Answer with JSON only. Give one list of property names per entity, under that entity's id, using the names exactly as they are written here.

Leave out a property whose value you would have to invent. A property you leave out will not be asked for again.

The entities, with the properties each one may carry:
e1 ("Arak is surrounded by mountains in the south, west, and east, and its average altitude is about 1750m above sea level."), Altitude: unit
```

**user message**

```
Arak is surrounded by mountains in the south, west, and east, and its average altitude is about 1750m above sea level.
```

**enforced schema** (728 chars, as a request-body field)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "FillableProperties",
  "description": "Which properties of each entity the document gives a value for.",
  "type": "object",
  "properties": {
    "fillable": {
      "type": "object",
      "properties": {
        "e1": {
          "type": "array",
          "maxItems": 1,
          "items": {
            "type": "string",
            "enum": [
              "unit"
            ]
          },
          "description": "The properties of e1 the document gives a value for."
        }
      },
      "required": [
        "e1"
      ],
      "additionalProperties": false
    }
  },
  "required": [
    "fillable"
  ],
  "additionalProperties": false
}
```

**reply**

```
{"fillable":{"e1":["unit"]}}
```

### Step 3: extract

**system message** (8750 chars)

```
Read the document and report these entities, each under the id given here: e1 ("Arak is surrounded by mountains in the south, west, and east, and its average altitude is about 1750m above sea level."). Give that id as the value of the id property. Report only these properties for them: unit.

Answer with JSON only. No explanation, no markdown fence, no commentary.

Choose the class of each entity from this list, using one of these exactly:
- ActivityCoefficient
  units: unitless
- AlphaDisintegrationEnergy
  units: joule, electron_volt, atto_joule, Ha, kilo_electron_volt, femto_joule, mega_electron_volt, pico_joule, giga_electron_volt, nano_joule, micro_joule, milli_joule, second_watt, calorie, cal_to_the_15_degree_celsius, international_calorie, mean_calorie, kilo_joule, hour_volt_ampere, hour_watt, kilo_calorie, kilo_international_calorie, mega_joule, hour_kilo_volt_ampere, hour_kilo_watt, giga_joule, hour_mega_volt_ampere, hour_mega_watt, metric_ton_per_force_pound, tera_joule, giga_watt_hour, peta_joule, hour_tera_watt, exa_joule, quad
- AbsoluteHumidity
  units: gram_per_liter, femto_gram_per_liter, nano_gram_per_meter_cubed, pico_gram_per_liter, nano_gram_per_liter, pico_gram_per_milli_liter, micro_gram_per_meter_cubed, nano_gram_per_deci_liter, milli_gram_per_meter_cubed, nano_gram_per_milli_liter, micro_gram_per_liter, micro_gram_per_deci_liter, grain_per_meter_cubed, gram_per_meter_cubed, milli_gram_per_liter, nano_gram_per_micro_liter, micro_gram_per_milli_liter, milli_gram_per_deci_liter, gram_per_deci_meter_cubed, kilo_gram_per_meter_cubed, milli_gram_per_milli_

[... 7150 of 8750 characters cut, the rest is more of the same ...]
```

**user message**

```
Arak is surrounded by mountains in the south, west, and east, and its average altitude is about 1750m above sea level.
```

**enforced schema** (1465 chars, as a request-body field)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "QuantityValues",
  "description": "Every measurement stated in the document.",
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "anyOf": [
          {
            "type": "object",
            "properties": {
              "type": {
                "const": "Altitude"
              },
              "unit": {
                "type": [
                  "string",
                  "null"
                ],
                "enum": [
                  "astronomical_unit",
                  "centi_meter",
                  "deca_meter",
                  "deci_meter",
                  "fermi",
                  "hecto_meter",
                  "kilo_meter",
                  "meter",
                  "micro_meter",
                  "milli_meter",
                  "nano_meter",
                  "pico_meter"
                ]
              },
              "id": {
                "type": "string",
                "description": "The id this entity was planned under.",
                "enum": [
                  "e1"
                ]
              }
            },
            "required": [
              "type",
              "unit",
              "id"
            ],
            "additionalProperties": false
          }
        ]
      }
    }
  },
  "required": [
    "entities"
  ],
  "additionalProperties": false
}
```

**reply**

```
{"entities":[{"type":"Altitude","unit":"meter","id":"e1"}]}
```

### Produced

```json
{
  "entities": [
    {
      "type": "Altitude",
      "unit": "meter",
      "id": "e1"
    }
  ]
}
```

### Scored

| dimension | F1 |
|---|---|
| primary | 0.000 |
