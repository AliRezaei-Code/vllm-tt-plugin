# CONTEXT — glossary for the vLLM TT Plugin atlas

Nouns only, one line each. For what each thing *does*, see
[`SYSTEM.md`](./SYSTEM.md) (generated) or `atlas/atlas.html`. For why a call was
made, see the decisions table in `SYSTEM.md` and `AGENTS.md`.

## The hardware

- **TT device** — one Tenstorrent accelerator, Wormhole or Blackhole.
- **Mesh device** — the set of devices one process drives, and the unit tt-metal addresses.
- **Mesh grid** — the `(rows, columns)` shape of a mesh, parsed from `MESH_DEVICE`.
- **Fabric** — the links between devices inside a mesh; set per model via `fabric_config`.
- **Canvas** — the physical output tile a block-output model commits to.
- **Token tile** — the device's granularity for prompt lengths; `_TT_TOKEN_TILE_SIZE`.

## The three execution modes

- **Single-process non-DP** — one engine, one mesh, no data parallelism.
- **Lane-DP** — one process, several independent model and KV replicas, batched together.
- **Standard DP** — several processes, one mesh per rank, device ownership agreed in advance.

## Configuration

- **Entry point** — the `vllm.platform_plugins` / `vllm.general_plugins` hook that activates the plugin.
- **TT config** — the operator's `"tt"` sub-dict of `additional_config`.
- **additional_config** — the serializable config field that carries state across processes.
- **model_capabilities** — the dict a model class declares; the only legitimate gate for new behaviour.
- **Block output** — a model committing more than one token per step.
- **Adaptive block output** — block output that falls back to width-1 whenever a batch forms.
- **Block contract** — the `(output_size, max_model_len)` pair a block-output model is admitted under.
- **MRV1 / MRV2** — Model Runner V1 and V2; the plugin pins V1.

## Scheduling

- **Token budget** — upstream's per-step token allowance, which TT cannot always fill.
- **Step plan** — the merged per-lane work list handed to the runner.
- **Skipped waiting** — upstream's queue for requests blocked by a dependency.
- **Prefix cache** — reuse of KV blocks across requests with a shared prompt prefix.
- **Placeholder block** — the K-token block reserved for a block-output model on a solo decode.
- **Admission** — the decision that a request may run this step at all.

## Execution

- **Worker** — the per-process handle on the mesh.
- **Model runner** — turns one scheduler step into one device forward pass.
- **Input batch** — the pre-baked, persistent view of the batch the device will see.
- **Cached request state** — upstream's per-request snapshot the plugin reuses rather than reimplementing.
- **MultiGroupBlockTable** — upstream's block table, spanning several KV groups.
- **Deferred decode** — an async readback drained exactly once, by whichever caller arrives first.
- **Device sampling** — choosing the token on the accelerator rather than the host.
- **Compat sampling** — falling back to host sampling when a request needs more than the device offers.

## Optional features

- **Structured output** — a grammar bitmask that makes schema-invalid tokens unsamplable.
- **Logprobs** — the log probabilities behind sampled tokens.
- **Prompt logprobs** — the same, at prompt positions; needs prefill logits from tt-metal.
- **Speculative decoding** — drafting several tokens, then accepting the ones the target agrees with.
- **Drafter** — the model that proposes speculative tokens; the missing runtime half.

## Upstream coupling

- **Platform** — vLLM's view of the hardware; `TTPlatform` subclasses it.
- **WorkerBase** — vLLM's worker interface; `TTWorker` subclasses it.
- **SchedulerInterface** — the scheduler contract; `TTScheduler` and `TTLaneCoordinator` implement it.
- **Monkeypatch** — a runtime replacement of an upstream method, installed beside the `_install_*` helpers.
- **Gap** — an upstream contract member the plugin does not implement.
- **Refusal** — a loud `ValueError` or `NotImplementedError` naming the offending value and its flag.
- **Unreachable** — a contract member with no call site on a TT path, so overriding it would be dead code.
