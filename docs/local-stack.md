# Training and serving an open-weight model on the local box

What runs where, which API calls drive it, and what is worth knowing before
starting. Against Unsloth Studio 2026.9.14 and vLLM 0.30.0 on two NVIDIA
A40s.

## Why a local box at all

Azure fine-tuning works and produced the H4 result, but it is closed-weight.
Two things need weights we control:

**A grammar that accepts arbitrary JSON Schema.** Azure's `strict` mode
refuses 48 per cent of real-world schemas, measured on 300 of
JSONSchemaBench's. vLLM applies a grammar to whatever schema it is given, so
the union arm and the grounded arm can run as declared instead of as whatever
survived a provider subset.

**A tuned model whose prompt is a few hundred tokens instead of ten
thousand.** Prefill dominates everything here: a 15,829-token call answers in
39 seconds idle. If the vocabulary moves into the weights, the prompt and the
wall clock both collapse.

## The shape

```
card 0                              card 1
  vllm-9b   Qwen3.5-9B  util 0.60     unsloth   training
  vllm-4b   Qwen3.5-4B  util 0.28
```

Both vLLM services sit behind Caddy on their own hostnames. Unsloth is pinned
to card 1 with `device_ids: ['1']` so a tune and a grid can run at once.

## Training, as API calls

```
POST /api/datasets/upload              multipart, one file field
POST /api/train/start                  -> job_id
GET  /api/train/status                 poll until phase == completed
GET  /api/train/runs/{job_id}          -> run.preview_ref
POST /api/export/load-checkpoint       /opt/unsloth-studio/outputs/{preview_ref}
POST /api/export/export/lora           push_to_hub + repo_id + hf_token
```

`results_local/ft_pipeline.py` is this loop end to end, one model per call,
pushing each adapter to the Hub before starting the next.

### The path convention, which no endpoint reports

This is the step that blocks everything else. The API hides filesystem paths
by design: `models-folder` returns `{"path": ""}`, `cache_path` is `""`,
`run.config.output_dir` is `None`, and a run's `output_dir` is an opaque
`ref:` handle that `load-checkpoint` rejects. Artifacts are nevertheless at:

```
/opt/unsloth-studio/outputs/{preview_ref}     the adapter a run wrote
/opt/unsloth-studio/exports/{ref}             what an export produced
```

and `preview_ref` *is* on the run record. So the path is reconstructable even
though nothing will tell you it.

## Serving, as API calls

```
POST /v1/load_lora_adapter   {"lora_name": ..., "lora_path": "/adapters/{ref}"}
POST /v1/unload_lora_adapter
```

Needs `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` and `--enable-lora`. Mount
Unsloth's output directory into vLLM read-only and the adapter it just wrote
is the one vLLM loads: no merge, no GGUF conversion, no Hub round trip. The
Hub is then publication rather than plumbing.

`lora_path` is documented only with local paths, and that reading was too
cautious. Measured 2026-10-03 against vLLM 0.30.0: posting
`{"lora_name": "probe-hub", "lora_path": "OO-LD/oold-lean-qwen4b-r16-s1000"}`
returns 200, the adapter appears in `/v1/models`, and no plugin was loaded.
So a tune reaches the benchmark with no compose change at all: train, push,
POST, run.

The `lora_hf_hub_resolver` plugin solves a different problem, loading an
adapter because a request named a model nobody has loaded yet, and it is the
one carrying the remote-download warning. Explicit `lora_path` does not need
it.

`load_inplace: true` replaces an adapter without interrupting inference, which
is what makes a second tune a swap rather than a restart.

## The enforcement probe, which runs before every grid

A backend that cannot express a constraint proceeds unconstrained and still
answers 200. That is worse than a refusal, because it records.

Send a schema whose `unit` is `{"enum": ["meter"]}` against a document that
says "metres". Measured 2026-10-03 on `qwen3.5-9b`:

| | answer |
|---|---|
| free decoding | `"quantity": 880, "unit": "metres"` |
| `response_format: json_schema` | `"type": "Length", "value": 880, "unit": "meter"` |

Same model, same prompt. The second is enforced. Any cell from a backend that
fails this probe is a cell about nothing.

## Operating notes

**1. Studio runs two inference backends.** `LlamaCppBackend` for GGUF has a
grammar engine. The Transformers path has neither a grammar engine nor paged
attention. A GGUF model took 15,829-token prompts and enforced schemas, while
a merged export OOMed at 10,000 tokens with 30 GB free and refused
`response_format` outright. Not configurable, architectural.

**2. `load-checkpoint` is the exporter, not the server.** It answers
`/v1/chat/completions`, so a short prompt returns 200 and looks like serving.
It is holding the model so it can be merged.

**3. An OOM may be another model, not the input.** `export/cleanup` frees
nothing. `POST /v1/unload` with an explicit `model_path` frees everything.
The message blames the prompt either way.

**4. `download-progress` lies.** It reported `on_disk=True, progress=1.0` for
four GGUFs of which `gguf-variants` showed zero downloaded. Trust
`cached-gguf` and `gguf-variants`, which report per file.

**5. `/api/hub/download` with a variant is a no-op.** Accepted, `state:
running`, no active downloads, nothing lands. Both request shapes, twice.
Whatever put the 27B in the cache was not this endpoint.

**6. `num_train_epochs` is silently ignored.** Only `max_steps` takes effect.
A run asking for 2 epochs ran 500 steps, which at batch 2 over 1,000 examples
is one. The comparable unit against an Azure job is epochs: 1,000 examples x
2 epochs / batch 2 = 1,000 steps.

**7. Studio passes arguments its own TRL has removed.**
`SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`
blocked every run until the image was pulled. It is sent whether or not you
set it, so there is nothing to work around in the request.

**8. `/opt/unsloth-studio` is not persisted by default.** `studio.db` is, so
run records survive a container recreate while the weights they name do not.
Four adapters were lost this way, and the one that survived was already on
the Hub. Mount `outputs`, `exports` and `assets/datasets`, and push each adapter
as soon as its run finishes.

**9. QLoRA is not recommended on Qwen3.5.** Unsloth's own page: "not
recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE
or dense", for larger than usual quantization error. Its code comments
contradict this for dense models. bf16 LoRA needs 10 GB at 4B and 22 GB at
9B, so the argument is cheap to avoid.

**10. Qwen3.5 reserves a Mamba block per concurrent sequence.** It is a
Gated DeltaNet architecture, so `max_num_seqs` costs state cache as well as
KV: `max_num_seqs (256) exceeds available Mamba cache blocks (166)`. Lower it
rather than raise `gpu-memory-utilization`, which takes memory from whatever
else shares the card.

**11. Two models on one card starves one of them, whichever is smaller.**
Qwen3.5 reserves a Mamba state block per sequence, so a share sized by weights
is not a share sized by work. At 0.28/0.60 the 4B preempted 485 times and
recomputed every prefill; at 0.36/0.52 the 9B preempted instead and fell to
2.7 requests a minute. There is no split that leaves both comfortable. One
model per card removed the question, and the 4B went from 15 to 45 cells a
minute.

**12. `max_tokens` defaulting to `None` hands over the whole context window.**
A 9B that began restating its own schema generated sixteen thousand tokens a
call and never stopped, and because the runner consumes outcomes in declared
order three such calls shut the record stream for forty minutes while the card
stayed busy. Bound it. A truncated answer records as a parse error, which is
honest and cheap.

**13. A LoRA timeout is a floor, not a ceiling, until retries are set.**
LangChain retries twice by default, so a 900-second timeout was three quarters
of an hour before a hung call gave up.

**14. `gpu-memory-utilization` is a fraction of the whole card, measured
against free memory at startup.** Two models on one card is therefore order
dependent, and `restart: unless-stopped` on both races them. Use `depends_on`
with a healthcheck and start the larger one first.

## What this does not solve

A vLLM process serves one base model. An adapter loads at runtime without a
restart, a different base does not. For a short base list this is two compose
services. Past that, llama-swap puts the declaration in a per-model file
instead, at the cost of a Docker socket mount and undocumented passthrough of
the flags this benchmark depends on.

**Pin the image tag.** vLLM has moved these flags before, `guided_json` was
removed in 0.12.0, and `latest` means a `docker compose pull` can change what
`schema-dump-catalog-flat-enforced-strict` denotes with no trace in the results.
