# oold-llm-bench

Hypothesis-driven benchmark for schema-constrained LLM extraction.

It measures where structure has to be enforced for an LLM to produce data that
validates. The candidates are the prompt, decode time, commit time, and nowhere
at all. Arms range from free text with no schema to a fully grounded OO-LD
schema, and one deterministic grader scores all of them.

## Scope

| | |
| --- | --- |
| Question | Does schema enforcement change extraction quality, and where does it have to happen |
| Arms | `no-catalog-not-enforced-prose`, `no-catalog-not-enforced`, `schema-dump-catalog-not-enforced-gated` (schema in prompt), `schema-dump-catalog-flat-enforced` (decode-time constraint), `schema-dump-catalog-flat-enforced-grounded` (ontology grounding) |
| Metric | Triple-level F1 over (entity, property, value), with class, unit and provenance reported separately |
| Corpora | schema.org from real and synthetic documents; Recipe and JobPosting pages read out of the Common Crawl archive; QUDT quantity kinds with closed unit enums; Wiki-Measurements sentences with annotated spans |
| Deduplication | A separate corpus, `wikidata_identity`, because the end use case needs extraction and identity and a failure has to be attributable to one of them. Wikidata item merges, `P1889` and `P460` supply `same`, `different` and `unclear`, each decided by a person. |

## Playground

A browser interface for looking at one extraction at a time, with what the
benchmark measured beside it: the score against ground truth, what each step of
the orchestration cost, how much of the schema reached the model, and why an
answer that parsed still fails to validate. Entities are drawn as nodes, links
as edges, and a link whose target was never reported as a dashed edge into a red
node.

Orchestration, arm and model are chosen in the interface, and every agent is
built through `runner.adapter.build_agent`, so a run here is the run a grid cell
would make.

A pasted document is offered the classes of named sets rather than a count of
them. The sets are the ontology's own, one per child of `Thing`, and several
may be ticked: a sentence about employment needs `Person` and `Organization`,
which are two branches.

The graph grows across turns, so a second document adds to the first. Two
entities are decided to be one by exact agreement where every value they both
carry agrees, and by a judge otherwise. The judge is a model chosen apart from
the extractor, it answers `skos:exactMatch`, `skos:closeMatch` or different, and
only an exact match merges. A close match leaves both nodes and draws the
relation between them for a person to resolve. Which pairs were compared, what
was decided, by which route and on what evidence are under the Identity tab,
together with the share of comparisons that were decided at all.

```bash
uv sync --extra playground
OOLD_BENCH_SCHEMAS=/path/to/schemaorg/generated uv run python -m oold_llm_bench.playground
```

`OOLD_BENCH_SCHEMAS` points at the generated schema.org module, which is not in
this repository. The interface reports its digest and file count under the
Corpus tab, and refuses to start with a message naming the variable when it is
absent.

| | |
| --- | --- |
| Extra | `playground`, which pulls `agent` and `providers` with it. Never installed by `make ci`. |
| Input | A corpus task, whose ground truth gives a score, or a pasted document, which has none. |
| Client | Offline replay by default, answering from the task's own ground truth. Switch to the provider to spend a call. |
| Catalogue | Named class sets, several at a time. A pasted document is offered their union. |
| Identity | Exact agreement merges for free; anything else goes to a judge chosen apart from the extractor, which may defer as `skos:closeMatch`. |
| End-to-end test | `uv run playwright install chromium`, then `uv run pytest tests/test_playground_ui.py -m ui`. Deselected by default. The pasted-document tests need `OOLD_BENCH_UI_LIVE=1`, because replay answers from ground truth and a paste has none. |

## Data licences

The code is Apache-2.0. Three data files are not, and each states its own terms.

| File | Licence | Terms |
| --- | --- | --- |
| `src/oold_llm_bench/data/wiki_measurements.json` | CC BY-SA 4.0 (sentences), CC0 1.0 (Wikidata facts) | Attribution and share-alike. Every record names the Wikipedia article it came from, and anything derived from the sentences, including a task set built from them, stays CC BY-SA 4.0. |
| `src/oold_llm_bench/data/wikidata_identity.json` | CC0 1.0 | None. Every field is main-namespace Wikidata structured data, which [Wikidata:Copyright](https://www.wikidata.org/wiki/Wikidata:Copyright) puts under CC0. Attribution is courtesy. |
| `src/oold_llm_bench/data/wdc_schemaorg.json` | No licence is granted, so none is claimed | The file holds urls, Common Crawl WARC coordinates, record digests and our own extraction of each page's markup. It holds no page text. The documents are rebuilt locally from the coordinates and stay on the machine that rebuilt them. |

Built from [Wiki-Quantities and Wiki-Measurements](https://doi.org/10.5281/zenodo.14858280), Göpfert, Kuckertz, Weinand and Stolten, FZJ ICE-2, 2025. The 98 MB archive is not in this repository: `scripts/build_wiki_measurements.py` reads it wherever it was downloaded and writes the committed corpus.

The Recipe and JobPosting documents come from the
[Web Data Commons October 2024 subsets](https://webdatacommons.org/structureddata/2024-12/stats/schema_org_subsets.html),
which supply the urls, and from
[Common Crawl](https://data.commoncrawl.org) `CC-MAIN-2024-42`, which supplies the pages. Reading
both the markup and the visible text out of one archived snapshot is what makes the pairing a
statement about the page rather than a measurement of two years of editing. Web Data Commons states
no data licence and Common Crawl grants access only, so neither makes a page ours to pass on:
`scripts/build_wdc_schemaorg.py` writes the coordinates here and the page text into a local
directory, and `--pages-only` rebuilds that directory from the coordinates alone.

The identity pairs are drawn from the namespace-0
[redirect table](https://dumps.wikimedia.org/wikidatawiki/latest/wikidatawiki-latest-redirect.sql.gz)
and from the `P1889` and `P460` statement harvests. `scripts/build_wikidata_identity.py` writes the
capped corpus here and the full resolved set into the user cache, because the repository is not a
data store.

The benchmark depends on [oold-python](https://github.com/OO-LD/oold-python) as
one consumer among several, and must be able to run an arm that depends on
nothing at all. That is why it does not live inside the library it evaluates.

## Development

One command mirrors CI. Run it before every push:

```bash
make ci
```

It runs these in order, and each is also usable on its own:

| Command | Does |
| --- | --- |
| `make check` | Lock file consistency, `pre-commit` on tracked and untracked files, `ty`, `deptry` |
| `make test` | `pytest` with coverage |
| `make docs-test` | Builds the documentation strictly, failing on any warning |

Everything else, or `make help` for the full list:

| Command | Does |
| --- | --- |
| `make install` | Creates the environment and installs the git hooks |
| `make docs` | Serves the documentation locally |
| `make build` | Builds a wheel into `dist/` |
`make check` lints untracked files on purpose: `pre-commit run -a` skips them, which is how a new file passes locally and then fails in CI the moment it is committed.

Commits follow [Conventional Commits](https://www.conventionalcommits.org/); releases, the changelog and versioned documentation are automated on merge to main. Until a release App is configured, the release workflow skips itself instead of failing.

## Updates from the template

This repository was generated from [coregraft](https://github.com/OO-LD/coregraft) and records which version in `.copier-answers.yml`. A weekly workflow checks whether the template has moved on and, if so, replays its changes here. Your own edits are preserved: copier re-applies them on top of the new template version rather than over it, and anything it cannot merge arrives as `<<<<<<<` conflict markers for a human to decide.

By default the update arrives as an **issue** naming the new version and the command to run:

```bash
uvx copier update --skip-answered --trust --conflict inline
```

To get it as a **pull request** instead, this repository needs a GitHub App, because GitHub refuses to let the built-in token create or update anything under `.github/workflows/`, and template updates routinely do. One-time setup:

1. Use or create a GitHub App with **Contents: read and write**, **Pull requests: read and write** and **Workflows: read and write**. The last one is the whole point; without it the push is rejected. Organisations often already have a release App, which may need the Workflows permission added and the updated permission accepted on each installation.
2. Install the App on this repository.
3. Provide `RELEASE_APP_ID` and `RELEASE_APP_PRIVATE_KEY` as repository secrets, or share the organisation secrets with this repository. Installing the App and sharing the secret are two separate settings.

The workflow detects the secret at runtime, so it opens a pull request once the App is available and falls back to an issue when it is not. Either way the merge is the same; only the automation differs.

---

Generated from [coregraft](https://github.com/OO-LD/coregraft).
