## single_shot under `schema-prose-catalog-enforced`

`unsloth/gemma-4-12B-it-qat-GGUF`, 1 call(s), 100 classes on offer. Arm `catalog-enforced` in the records.

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

### Step 1: extract

**system message** (28818 chars)

```
Read the document and report every entity it describes, with the values stated for each.

Answer with JSON only. No explanation, no markdown fence, no commentary.

Choose the class of each entity from this list, using one of these exactly:
- AcousticImpedance
  units: pascal_second_per_meter, gram_per_hectare_per_year, micro_gram_per_day_per_meter_squared, nano_gram_per_centi_meter_squared_per_day, kilo_gram_per_hectare_per_year, milli_gram_per_day_per_meter_squared, micro_g_per_cm_squared_wk, gram_per_meter_squared_per_year, milli_gram_per_hour_per_meter_squared, gram_per_meter_squared_per_week, mega_gram_per_hectare_per_year, metric_ton_per_hectare_per_year, gram_per_day_per_meter_squared, gram_per_hour_per_meter_squared, gram_per_centi_meter_squared_per_year, milli_gram_per_meter_squared_per_second, kilo_gram_per_day_per_meter_squared, gram_per_meter_squared_per_second, kilo_gram_per_meter_squared_per_second
- Basicity
  units: pH_value
- Acidity
  units: pH_value
- BevelGearPitchAngle
  units: dimensionless, neper, PSU, dec, grain, octave, ppq, parts_per_trillion, vacuum_permittivity, ppb, PPTM, ppm, permille, percent, Sh, barn, COUNT, E, point, flight, heartbeat, nano_technical_atmosphere, one, unknown, unitless, ban, Hart, bel, to_the_10, to_the_100, kilo_barn, kibi_barn, to_the_1000, kilo_byte, kibi_byte, mega_barn, mebi_barn, to_the_1000000, mega_byte, mebi_byte, gigabit, gibi_barn, to_the_1000000000, giga_point, giga_byte, gibi_byte, tera_barn, tebi_barn, to_the_1000000000000, tera_byte, tebi_byte, pebi_barn, peta_barn, pebi_byte, peta_byte, exbi_barn, exa_barn, ex

[... 27218 of 28818 characters cut, the rest is more of the same ...]
```

**user message**

```
Arak is surrounded by mountains in the south, west, and east, and its average altitude is about 1750m above sea level.
```

**enforced schema** (127264 chars, as a request-body field)

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
                "const": "AcousticImpedance"
              },
              "value": {
                "type": "number",
                "description": "The magnitude, as written in the document."
              },
              "unit": {
                "type": [
                  "string",
                  "null"
                ],
                "enum": [
                  "gram_per_centi_meter_squared_per_year",
                  "gram_per_day_per_meter_squared",
                  "gram_per_hectare_per_year",
                  "gram_per_hour_per_meter_squared",
                  "gram_per_meter_squared_per_second",
                  "gram_per_meter_squared_per_week",
                  "gram_per_meter_squared_per_year",
                  "kilo_gram_per_day_per_meter_squared",
                  "kilo_gram_per_hectare_per_year",
                  "kilo_gram_per_meter_squared_per_second",
                  "mega_gram_per_hectare_per_year",
                  "metric_ton_per_hectare_per_year",
                  "micro_g_per_cm_squared_wk",
                  "micro_gram_per_day_per_meter_squared",
                  "milli_gram_per_day_per_meter_squared",
                  "milli_gram_per_hour_per_meter_squared",


[... 125664 of 127264 characters cut, the rest is more of the same ...]
```

**reply**

```
{
  "entities": [
    {
      "type": "Altitude",
      "value": 1750,
      "unit": "meter"
    }
  ]
}
```

### Produced

```json
{
  "entities": [
    {
      "type": "Altitude",
      "value": 1750,
      "unit": "meter"
    }
  ]
}
```

### Scored

| dimension | F1 |
|---|---|
| primary | 1.000 |

One task, so this is what the pipeline did here and not what it scores.
