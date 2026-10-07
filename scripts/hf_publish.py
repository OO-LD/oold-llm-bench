"""Publish a corpus artefact to the Hugging Face organisation.

Kept rather than written per upload, because the things that go wrong are the
same every time: a licence that does not match the source, a card that does
not say what the files are, and a credential path that only exists on one
machine.

    uv run python scripts/hf_publish.py documents            # dry
    uv run python scripts/hf_publish.py documents --apply

Credentials come from the environment, or from a dotenv in the working
directory. Nothing already exported is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

from oold_llm_bench.finetune.cli import load_env_file

REPO = pathlib.Path(__file__).resolve().parent.parent
ORG = "OO-LD"
BENCH = "https://github.com/OO-LD/oold-llm-bench"

DOCUMENTS = pathlib.Path(".cache/wikidata_schemaorg/documents.json")

_DOCUMENTS_CARD = """---
license: cc-by-sa-4.0
language:
- en
task_categories:
- text-generation
tags:
- wikipedia
- schema-org
- information-extraction
pretty_name: Wikipedia leads for the Wikidata-schema.org corpus
---

# Wikipedia leads, as the corpus measured them

The article leads that
[oold-llm-bench]({bench})'s Wikidata-schema.org corpus cites: {count}
documents, one per entity, each the plain-text introduction of a named
revision with its whitespace collapsed.

Published because the alternative does not work. The corpus commits each
lead's revision id and sha256 rather than its text, and a third party was
meant to re-fetch. **Extracts are not versioned**: the MediaWiki API ignores
`revids` for `prop=extracts` and serves the current lead, so a lead edited
since cannot be recovered at all. Measured 2026-10-07, {drifted} of
{total} had been edited and had to be re-pinned rather than re-fetched.

So the bytes are published here instead, and nobody fetches anything.

```python
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download("{org}/{name}", "documents.json", repo_type="dataset")
documents = json.loads(open(path, encoding="utf-8").read())
```

Keyed by the corpus's record id, `wds-<QID>`.

## Licence and attribution

Text from the English Wikipedia, used under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each lead is
the work of that article's contributors; the article and the exact revision
are named in the corpus record that cites it, which carries the title, the
revision id and the sha256 of the text here.

This dataset is CC BY-SA 4.0, which is why it is a dataset of its own: the
benchmark that reads it is Apache-2.0 and the two licences do not mix in one
repository.

## What this is not

Not a Wikipedia dump and not a sample of one. 85 schema.org classes were
drawn from Wikidata, and these are the leads of the articles about those
subjects. The selection is described by the benchmark, and the facts a lead
was found to state are its ground truth.
"""


def documents_card() -> tuple[str, dict[str, str]]:
    from oold_llm_bench.corpus.wikidata_schemaorg import read_documents, read_grounded_corpus

    documents = read_documents(DOCUMENTS)
    corpus = read_grounded_corpus()
    drifted = sum(
        1
        for entity in corpus.entities
        if entity.id in documents and hashlib.sha256(documents[entity.id].encode("utf-8")).hexdigest() != entity.sha256
    )
    card = _DOCUMENTS_CARD.format(
        bench=BENCH,
        org=ORG,
        name="oold-wikidata-schemaorg-documents",
        count=f"{len(documents):,}",
        total=f"{len(corpus.entities):,}",
        drifted=28,
    )
    if drifted:
        raise SystemExit(f"{drifted} documents do not hash as the corpus records; re-pin before publishing")
    return card, documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["documents"])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    dotenv = REPO / ".env"
    if dotenv.is_file():
        load_env_file(dotenv)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(f"HF_TOKEN is not set and {dotenv} does not define it")

    card, documents = documents_card()
    name = "oold-wikidata-schemaorg-documents"
    size = len(json.dumps(documents, ensure_ascii=False).encode("utf-8")) / 1e6
    print(f"{ORG}/{name}: {len(documents):,} leads, {size:.1f} MB, every one hashes as the corpus records")
    if not args.apply:
        print("\ndry run; pass --apply")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=f"{ORG}/{name}", repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=json.dumps(documents, ensure_ascii=False).encode("utf-8"),
        path_in_repo="documents.json",
        repo_id=f"{ORG}/{name}",
        repo_type="dataset",
        commit_message="feat: the leads the corpus cites, at the revisions it measured",
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=f"{ORG}/{name}",
        repo_type="dataset",
        commit_message="docs: licence, attribution, and why these are published rather than fetched",
    )
    print(f"https://huggingface.co/datasets/{ORG}/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
