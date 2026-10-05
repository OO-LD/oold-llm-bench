# Corpora

Four corpora, and the rule that decides what may be redistributed from each.

| corpus | documents | what it tests |
|---|---|---|
| generated quantities | unlimited, from the schema module | enforcement against a closed unit vocabulary |
| Wiki-Measurements | Wikipedia sentences, CC BY-SA 4.0 | the same on human prose |
| generated schema.org | unlimited, 75 frames | orchestration where subclasses add properties |
| Wikidata-grounded schema.org | real leads, real markup | class selection the document does not give away |

## The schema modules

Both generated corpora read a directory of OO-LD schemas held outside this
repository and fetched by digest:

```bash
oold-bench fetch-corpus quantities --dest ./schemas
oold-bench fetch-corpus schemaorg --dest ./schemas-sorg
```

**The digest is the pin, not a version number**, and the quantity module is
why. A second published generation of it case-folds the unit identifiers of
234 of its 942 kinds. `celsius_per_hour` against `Celsius_per_hour`, `ha`
against `Ha`, `ph_value` against `pH_value`. The fold is injective, 1,291
distinct identifiers on both sides, and no kind loses a unit, so neither
vocabulary is wrong. They are different, and a model answering `Ha` is right
against one and wrong against the other. A version number cannot tell them
apart. `resolve_module` refuses a directory that does not hash to the pin,
before a grid is run and billed.

## What generated prose cannot test

Generated schema.org prose is rendered from 75 templates and names its own
class and property slots. A document reading "Logged as person. ... That is
its additional name" tells the model what to extract, so the selection step
the benchmark exists to measure is not exercised. `Labelling.IMPLIED` removes
the leakage and costs the describable pool, 122 classes down to 19. That is
why real text is on the critical path rather than beside it.

## Real text: the two sources that work

| source | classes | tasks | what it gives |
|---|---|---|---|
| Wikidata + Wikipedia | 6-12 | 20-50k | Person, Movie, MusicAlbum, Organization |
| Web Data Commons via Common Crawl WARC | 3-5 | tens of thousands | Recipe, JobPosting, HowTo, plus Rating, Review and Offer nested in recipe pages |

Together roughly 10-15 classes of human prose with no slot leakage, against
the 19 the synthetic implied mode admits with leakage removed and the 122 it
admits with leakage intact.

**Measured, and this is the point of the exercise: real prose does not name
property slots.** Wikidata/Wikipedia: 2.9% incidental matches over 50,000
documents, all ordinary English. WDC: 2.4% over 831 grounded values. Neither
resembles the synthetic corpus's behaviour, where the rate is by construction
100%.

## Transactional classes are out of scope

`Order`, `ParcelDelivery`, `Flight`, `TrainTrip`,
`UnitPriceSpecification`, `Demand`, `SizeSpecification`, `BroadcastService`
and `MerchantReturnPolicySeasonalOverride` are not in the real-text corpus and
will not be.

They are absent from Wikidata, absent from the Web Data Commons extraction,
and absent from any web crawl, because the pages that carry them sit behind
authentication. A benchmark cannot crawl someone's order history.

The tempting alternative is to keep them synthetic beside the real ones. That
is refused on the same grounds every other pooling decision in this benchmark
is refused: a corpus that silently mixes generated prose with human prose
reports one number for two treatments, and the generated half is the half
whose defects this document exists to escape. Synthetic results stay
reportable on their own and are never averaged with crawled ones.

## Why the commerce classes fail even where the markup is abundant

This is the finding worth keeping, because volume suggests the opposite.
`Offer`, `PriceSpecification`, `AggregateRating` and the rest appear in the
Product subset at 10^6 to 10^8 instances. Grounding is not the problem either:
85.3% of marked-up content values appear in the visible text, and the residue
is collapsed nutrition panels and SEO keywords rather than publishers marking
up what is not on the page.

**The problem is that a product page is not a document.** Measured on a live
page carrying `Product` markup:

```
Weight:
7.1oz (W7), 8.9oz (M9)
Country of Origin:
Vietnam
```

That is the synthetic corpus's defect reproduced by a real publisher. The same
page repeats its specification block three times, once per colour variant, so
one main entity per document fails as well, and 22 per cent of its visible
text is inlined terms of service.

A recipe page from the same crawl, by contrast, opens with prose and the
narrative properties carry no slot labels at all. So the split is not between
markup and no markup, it is between pages written to be read and pages
assembled from a database.

There is a third defect, subtler than leakage and worth naming separately:
the product page labels `Weight` and `Country of Origin`, and **neither is in
the markup**, which carries `name`, `brand`, `price` and `sku`. The visible
table names slots the ground truth does not use and leaves unlabelled the ones
it does. A model reading that page sees an obvious table and is scored on
something else.

## Source the page from the archive, not from the web

**Do not refetch a Web Data Commons source URL.** Measured on 115 of them from
the October 2024 release: **zero** still yield a usable entity. Dead domains,
robots-disallowed mirrors, markup since removed.

Resolve through the Common Crawl index and range-read the WARC record
instead:

```
https://index.commoncrawl.org/CC-MAIN-2024-42-index?url=<url>&output=json
  -> {"filename": ..., "offset": ..., "length": ..., "digest": ...}
curl -r <offset>-<offset+length-1> https://data.commoncrawl.org/<filename>
```

Each record is an independently gzipped member, so one ranged GET is one page.
This removes link rot, and it fixes a validity problem that matters more: the
markup and the visible text then come from the same snapshot. Comparing
today's text against markup extracted two years ago measures drift, not
grounding.

## Licensing

Wikidata is CC0. Wikipedia text is CC BY-SA 4.0 dual with GFDL. Web Data
Commons states no data licence at all, only "We publish the corpora for
research purposes only", and Common Crawl grants a limited non-transferable
access licence with page content remaining the site owner's.

So the corpus publishes **URLs, WARC coordinates, content digests and our own
extracted truth, never page text**. A third party reproduces it by fetching
and validating against the digest. Facts are not copyrightable and our
labelling is our own.

One consequence to state rather than discover: "research purposes only" is not
a grant that can be relicensed. Our annotations can carry our licence with the
source cited as provenance. A redistributable "WDC subset" cannot exist.

## What was measured and rejected

**T-REx.** Measured on 50,000 documents. It already aligns Wikipedia abstracts
to Wikidata triples and already applied a grounding filter, which is why it
was worth checking first. It loses to pulling Wikidata ourselves: roughly
13,000 balanced tasks over 7 classes against 20-50k over 6-12, with 89% of
usable documents being Person, zero Book and effectively zero Event.

Its grounding claim does not survive inspection either. Hand-judging 45
aligned triples gave 80% genuinely stated, 73% strict, against the published
97.8%. The error is not spread evenly: dates are clean at 28/28, while
`deathPlace` is 7/14 and `knowsLanguage` is **1/14**. The last is systematic
rather than noisy, because Wikidata infers the property from nationality and
T-REx then grounds it on the nationality adjective, so "Italo Calvino was an
Italian journalist" becomes `knowsLanguage = Italian`. It appears in 11.7% of
usable documents.

It is also a 2017 snapshot: 8% of its document entities no longer carry a
`P31`. And its figshare record declares CC BY 4.0 over text whose provenance
is CC BY-SA, which is a more permissive licence than the source allows.

**`NoeFlandre/schema-dot-org`.** CC BY 4.0, 12.4M records, 2,093 type labels.
Unusable here: every record is geo-anchored, carries at most three text
properties, and ships no page text, so grounding cannot be checked at all.

**Amazon ESCI.** Apache-2.0 and genuinely redistributable, 1.8M products with
title, description, bullet points, brand and colour. The best `Product` source
available and the only one whose text we could publish. It carries no `Offer`,
no price and no `PriceSpecification`, which is exactly the gap that sent us
looking, and its bullet points have the same table-shaped problem.
