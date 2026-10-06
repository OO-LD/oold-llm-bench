# Serving a tuned model on Hugging Face

The hosted half of the stack. `local-stack.md` covers the box; this covers
what runs the same model on rented hardware, and why that is worth doing at
all. Against vLLM 0.30.0 and `huggingface_hub` 1.33.

## Why rent anything when there is a box

Not cost. The whole benchmark is one to two GPU-hours, so every option is
within a few dollars of every other one.

**Reproducibility.** A reviewer cannot run our A40s. They can run a pinned
image on a rented L4, and a result whose central claim is about the decoder
has to name the decoder. The endpoint pins `vllm/vllm-openai:v0.30.0`, which
is the version the local box runs, so a number from one is comparable to a
number from the other and the grid does not have to say which.

**A second opinion on the instrument.** Enforcement that holds on two
independently configured servers is enforcement. Holding on one is a
configuration.

## The deployment, as it stands

```python
create_inference_endpoint(
    "qwen35-4b-oold",
    repository="unsloth/Qwen3.5-4B",
    framework="pytorch", task="text-generation",
    accelerator="gpu", vendor="aws", region="us-east-1",
    instance_type="nvidia-l4", instance_size="x1",
    type="protected",
    min_replica=0, max_replica=1, scale_to_zero_timeout=15,
    custom_image={"vLLM": {"url": "vllm/vllm-openai:v0.30.0", "port": 8000}},
    container_args=[
        "--enable-lora",
        "--lora-modules", "oold=OO-LD/oold-quantities-lean-qwen4b-r16-s1000",
        "--max-lora-rank", "16",
        "--max-model-len", "24576",
        "--structured-outputs-config.backend", "guidance",
        "--reasoning-parser", "qwen3",
    ],
    secrets={"HF_TOKEN": token},
)
```

`container_args` are appended to the managed image's entrypoint, so the
`--model` the engine already carries stays and everything here is additional.
That is the whole reason the managed vLLM variant is worth using over a bare
custom container: no entrypoint to write, every engine flag still reachable.

The endpoint serves two models, `unsloth/Qwen3.5-4B` and `oold`. Asking for
the second is what selects the adapter.

**No merge step is needed.** `--lora-modules` takes a Hub repository id and
vLLM fetches it at startup. The adapter stays an adapter, the base stays the
published base, and the thing under test is the pair rather than a third
artefact nobody can reproduce.

### The adapter loads despite a regex `target_modules`

Worth recording because it was the one thing expected to fail. Unsloth writes
`target_modules` as a **regex string** rather than the list PEFT usually
carries, and vLLM's LoRA loader has historically wanted a list. vLLM 0.30.0
accepts the regex. Checked on both adapters, r=16.

## Cost, measured rather than quoted

L4 24GB at **$0.80/h**, Qwen3.5-4B, 14.5k-token prompts, reasoning off,
`response_format` enforced, eight concurrent requests:

| prompts | input tok/s | s/request | per 1M input tokens | per 1000 requests |
|---|---|---|---|---|
| sharing a prefix | 31,524 | 0.46 | **$0.007** | **$0.10** |
| all distinct | 7,589 | 1.92 | **$0.029** | **$0.43** |

Both rows are real and the honest number depends on the workload. A grid
sends one catalogue to many documents, so the prefix is shared by
construction and the first row is what a grid pays. The second row is the
floor under any claim, and it is the one to quote where the workload is not
known.

For comparison, Together's serverless rate for the 9B is $0.17 per million
input tokens. Renting the whole GPU and keeping it busy is between six and
twenty-four times cheaper per token, which is what full utilisation buys and
is also the catch: the rate above assumes the card never idles.

## Scale-to-zero fires late, and the gap is billed

`scale_to_zero_timeout=15` is not honoured at fifteen minutes. Measured: last
request 07:06Z, `targetReplica` still 1 at 07:50Z, so forty-four idle minutes
were billed. It does eventually scale down, and a later check found the
endpoint at `scaledToZero` with no replicas, so the setting is a floor on the
wait rather than a value.

**Pause explicitly after every session anyway.** `ep.pause()` sets
`targetReplica` to 0 immediately and `resume()` takes about six minutes to
come back. The difference between "fires late" and "does not fire" does not
change what to do, because an unknown amount of idle time at $1.80/h is still
the largest avoidable line in the bill, and the marketing page's "no cold
starts" claim contradicts the autoscaling guide's own cold-start section.

## A runtime adapter does not survive scale-to-zero

`POST /v1/load_lora_adapter` puts the adapter in the running server's memory
and nowhere else. The endpoint sleeps, comes back with whatever its creation
arguments named, and the adapter is gone.

Measured 2026-10-04: four adapters loaded, a grid ran, the endpoint idled out,
the next grid asked for one of them and 240 cells returned
`OpenAIModelNotFoundError`. Recorded as failures rather than silently, which
is the only mercy in it.

Two ways out, and they differ in what they cost. Naming each adapter in
`--lora-modules` at creation makes it part of the startup and it is rebuilt on
every wake, at the price of a new endpoint whenever the set changes. Reloading
after each wake keeps the set mutable and costs three seconds, at the price of
remembering. A long `scale_to_zero_timeout` only widens the window.

## A cold endpoint loses a whole grid in ninety seconds

The proxy answers `503 Service Unavailable` while a replica starts, and a
runner reads that as a failed cell rather than as a reason to wait. The first
`tuned+none` attempt lost all 480 cells in 93 seconds, which is faster than a
successful run and therefore does not look wrong.

`X-Scale-Up-Timeout: 600` makes the proxy hold the request instead, but the
benchmark's client does not send it. So grids that share an endpoint run
back-to-back in one process, which keeps it warm between them, and a wake
request goes first when one has been idle.

## Training is also possible, through Jobs and not AutoTrain

**AutoTrain is no longer maintained.** Its own documentation says so and points
at Axolotl, TRL or `transformers.Trainer` instead.

What replaced it for our purpose is **HF Jobs**: pay-as-you-go compute billed
by the second, a Docker image and a command, with TRL's `sft.py` documented as
a UV job. So the Unsloth loop has a hosted equivalent that needs no box.

| flavor | accelerator | $/h |
|---|---|---|
| `l4x1` | 1x L4 24GB | 0.80 |
| `a10g-large` | 1x A10G 24GB | 1.50 |
| `l40sx1` | 1x L40S 48GB | 1.80 |
| `a100-large` | 1x A100 80GB | 2.50 |
| `h200` | 1x H200 141GB | 5.00 |

Same hardware and same rates as the endpoints, so a tune and a serve cost the
same per hour.

**Jobs default to a 30-minute timeout** and training runs are not 30 minutes.
Pass `timeout="2h"` or the run is killed with its output half written, which
is the hosted version of the unmounted-outputs fault in `local-stack.md`.

Volumes mount a Hub repo or a storage bucket, so checkpoints can be written
somewhere that outlives the job. That is the part the local stack had to learn
the hard way.

## What it does not solve

One vLLM process still serves one base model, exactly as on the box. Two base
models are two endpoints and two hourly bills, and the adapter-at-runtime
trick only moves within a base.

The 180-second ceiling that applies on Azure ML does not apply here, but a
cold start does: six minutes from `resume()` to the first answer, during which
the proxy returns 503 unless the request carries `X-Scale-Up-Timeout`.
