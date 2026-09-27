# vLLM TT Plugin — System Definition

**This file is the living source of truth for the plugin's architecture.** The interactive atlas, <code>SYSTEM.md</code>, and every claim in the research report rebuild from it.

_Question status: **27 open · 5 resolved**._

## One paragraph

This atlas covers **both repositories**: the <code>vllm-tt-plugin</code> that Tenstorrent maintains, and the stock vLLM 0.26.0 it plugs into. The plugin teaches stock vLLM how to run on Tenstorrent hardware without forking vLLM. It registers itself through two entry points, substitutes a Platform, a Worker, a Scheduler and a ModelRunner for the ones vLLM would pick, and routes every model through a tt-metal generator that owns the actual device kernels. It ships three execution modes, owns no model code of its own, and refuses loudly — rather than silently degrading — on every upstream vLLM feature the TT backend does not serve.

## Decisions locked

| Axis | Decision | ADR |
|---|---|---|
| Integration form | Plugin-only integration. No TT feature is ever landed in vLLM core; where an entry point cannot express the behaviour, a runtime monkeypatch lives beside the other _install_* helpers in platform.py. | [AGENTS.md §1](../AGENTS.md) |
| Gating contract | New behaviour gates on the model class's model_capabilities dict, never on a model-type allowlist. Absent keys default via .get; a present value that contradicts another resolved property raises. | [AGENTS.md §6](../AGENTS.md) |
| Dependency set | docs/install-vllm-tt.sh owns the whole dependency set, because vLLM's PyPI metadata is generated on a CUDA machine and uv would otherwise resolve requirements/cuda.txt regardless of VLLM_TARGET_DEVICE. | [install-vllm-tt.sh](../docs/install-vllm-tt.sh) |
| Execution modes | Three live modes: single-process non-DP, single-process lane-DP, and standard multi-process DP with one device mesh per rank. launcher.py is retained MPI/tt-run and is not hooked by vLLM. | [SCHEDULING.md](../docs/SCHEDULING.md) |
| Patch placement | Runtime monkeypatches live in platform.py next to the six _install_* helpers at lines 636, 696, 809, 875, 896, 956, plus _pin_v1_model_runner at 742, never in entrypoints.py. | [platform.py:636](../src/vllm_tt_plugin/platform.py) |
| Speculative decoding | The contract types and the host accept walk landed before the runtime wiring. The modules are unreachable at runtime until the drafter work merges; the config-time refusal is what keeps that honest. | issue #110 |
| Unsupported features | Fail loudly at config time with the offending value and the CLI flag, rather than degrade silently. A refusal is the shipped behaviour for every upstream feature TT does not serve. | [feature_support.py](../src/vllm_tt_plugin/feature_support.py) |
| Model Runner | Model Runner V1 is pinned at config time. vLLM 0.29.0 made MRV2 the default and deprecated MRV1 for removal in v0.32, so the pin is a tracked removal clock, not a permanent choice. | issue #50 |

## Cost model

The plugin adds no per-token compute of its own. Its cost is configuration-time: model registration, mesh construction, and KV-pool sizing happen once per process.
`model_runner.py` (2,899 lines) and `platform.py` (2,290 lines) are the two files a reviewer watches for growth; AGENTS.md §7 asks new branches there to move to a named neighbour instead.
## Reading order (the atlas chapters)

1. **Entry and configuration** — Two hooks make vLLM notice Tenstorrent; one class answers every question about the hardware. _(adds EP, PF)_
2. **Configuration and model registration** — Settings are worked out once and carried across process boundaries; models are translated from names to generators. _(adds CF, MR)_
3. **Scheduling and batching** — Upstream fills a token budget; TT can only fill what the device can actually take in one step. _(adds SC, IB)_
4. **Worker and runner execution** — The engine room: open the mesh, turn one step into one device forward pass, and never wait on the device. _(adds WK, RN, AD)_
5. **Parallelism: lane-DP, standard DP, and block output** — Three different ways to use more than one device, each with its own admission rules. _(adds LC, DP, BO)_
6. **Optional features** — Structured output, logprobs, and speculative decoding: two shipped, one not yet reachable. _(adds SO, LP, SD)_
7. **Upstream coupling and the refusal surface** — Where the plugin binds to vLLM internals, and how it handles what it does not serve. _(adds UC, FS)_
8. **The upstream v1 engine** — The engine TT plugs into: who validates a request, who decides what runs, and who owns the memory. _(adds vPlat, vInput, vSamp)_
9. **The upstream scheduler and KV cache** — Where the step is planned and where the blocks live — the two places TT had to change. _(adds vEngine, vSched, vKV)_
10. **The upstream worker and runner** — The executor reaches the worker by method name, and that detail explains the whole failure mode. _(adds vExec, vWorker, vRunner)_
11. **The whole system** — Everything at once, for free exploration.

## Structures

### Entry and configuration

#### EP · Entry points

**In one line.** The two hooks that make vLLM notice Tenstorrent at all.

**What it does.** When you start vLLM, these two functions are the first thing it runs. If they cannot load the Tenstorrent software, they step aside and vLLM behaves exactly as if the plugin were not installed.

**How it's built.** Two entry points in `pyproject.toml:36-40`: `vllm.platform_plugins:tt` → `entrypoints:platform_plugin`, and `vllm.general_plugins:tt_model_registry` → `entrypoints:register`. Both gate on `import ttnn` in `src/vllm_tt_plugin/entrypoints.py`. `register()` calls `register_tt_models_from_plugin()` so models are known in every process, including worker subprocesses that start before the platform hook runs.

**Steps in execution.**

1. **Import guard** — Decline to activate when ttnn is absent.
2. **Register models** — register_tt_models_from_plugin() in every process.
3. **Claim platform** — Return the TTPlatform class by qualified name.

**Questions.**

- **Q-EP1** Both hooks run in the API server and in every engine subprocess; is a failure there reported clearly enough to diagnose a partial install?

#### PF · TTPlatform

**In one line.** The class vLLM asks about the hardware, and the one place every unsupported feature is refused.

**What it does.** Everything vLLM needs to know about Tenstorrent as a platform goes through this one class: the device name, how many devices are visible, which worker class to build, and whether a feature you asked for can actually run here.

**How it's built.** class `TTPlatform(Platform)` at `src/vllm_tt_plugin/platform.py:1337`. `_enum = PlatformEnum.OOT` (line 1327), `device_name = device_type = "tt"` (1328-1329), `device_control_env_var = "TT_VISIBLE_DEVICES"` (1330), `simple_compile_backend = "eager"` (1351) because tt-metal's Triton is incompatible with inductor. `check_and_update_config` (1462) calls `_pin_v1_model_runner()` before anything reads `use_v2_model_runner`, then delegates to `_apply_check_and_update_config` (1560) which is the single gate site for every refusal.

**Steps in execution.**

1. **Pin MRV1** — _pin_v1_model_runner() before VllmConfig.__post_init__ reads use_v2_model_runner.
2. **Read capabilities** — Resolve model_capabilities off the registered TT model class.
3. **Refuse** — Call the feature_support validators in a fixed sequence.
4. **Resolve keys** — Store lane count and output width into additional_config.

**Questions.**

- **Q-PF1** The MRV1 pin is a removal clock: vLLM 0.29.0 deprecated MRV1 for removal in v0.32. When does that pin get revisited?
- **Q-PF2** Every refusal message must name the CLI flag the operator typed. Is there a test that proves each one does?

#### CF · Config namespace

**In one line.** The bridge between config-time decisions and the worker process that needs them later.

**What it does.** Some settings are worked out once, in the API server, but needed later in a worker process on another machine. This module carries them across, and keeps operator input separate from values the plugin derived itself.

**How it's built.** `src/vllm_tt_plugin/config.py` (369 lines). Operator input arrives as `get_tt_config()`, reading the `"tt"` key of `VllmConfig.additional_config`. Platform-derived state is stored under a leading-underscore top-level key — `_tt_resolved_lane_count`, `_tt_output_tokens_per_step`, `_tt_adaptive_block_output` — via `store_tt_lane_count()` and read by `get_tt_data_parallel_size()` / `get_tt_output_tokens_per_step()`. `additional_config` is a declared config field, so it survives the copy and pickle into the worker subprocess.

**Steps in execution.**

1. **Read operator config** — get_tt_config() returns the "tt" sub-dict.
2. **Derive** — Platform resolves capabilities and calls a store_tt_* writer.
3. **Hand off** — additional_config is pickled into the worker subprocess.
4. **Read back** — A scheduler or worker calls the matching get_tt_* accessor.

**Questions.**

- **Q-CF1** AGENTS.md §8 requires any new TTPlatform class attribute to be added to _TT_PLATFORM_CONFIG_ATTRS in tests/conftest.py, or test state leaks. A new attribute here would trip that.

#### MR · Model registry

**In one line.** Where a HuggingFace architecture name becomes a tt-metal generator that can actually run it.

**What it does.** vLLM knows model names, not Tenstorrent models. This is the translation layer: give it a model name from a config file, and it hands back the class that knows how to execute it on the device.

**How it's built.** `src/vllm_tt_plugin/model_registry.py` (36 lines) orchestrates `register_tt_models()` at `platform.py:1100`. Operator overrides come first and win: `TT_MODEL_CLASS_OVERRIDES`, parsed at `platform.py:580-623` from the environment variable of the same name. Then tt-metal's own generators, then `_iter_extra_model_bundles()` (`platform.py:1006-1051`) which walks `EXTRA_MODELS_DIR` (env var, read at line 1006) and picks up any self-contained bundle folder carrying a `vllm_metadata.json` — so a distribution tool can add a model with no source edit. `check_and_update_config` also prepends `"TT"` to the architectures list, which is why `TTWorker` resolves the prefixed entry rather than the cached `architecture` property.

**Steps in execution.**

1. **Operator overrides** — TT_MODEL_CLASS_OVERRIDES, most specific intent, registered first.
2. **tt-metal generators** — models.tt_transformers.tt.generator_vllm and per-demo generators.
3. **Extra bundles** — EXTRA_MODELS_DIR folders with a vllm_metadata.json.
4. **Resolve** — ModelRegistry resolves the TT-prefixed architecture at load time.

**Questions.**

- **Q-MR1** Capabilities are declared by tt-metal model classes, not here. A plugin change that reads a new capability needs a matching tt-metal change in the same release.

### Scheduling and batching

#### SC · TTScheduler

**In one line.** Decides which requests run this step, under constraints vLLM's token budget never had to consider.

**What it does.** Upstream vLLM fills a token budget each step. TT cannot always fill it, because the device takes work in fixed shapes. This scheduler refuses to promise more than the hardware can do in one step, and keeps a record of what it gave up.

**How it's built.** class `TTScheduler(AsyncScheduler)` at `src/vllm_tt_plugin/scheduler.py:170`. It inherits upstream's priority `SchedulingPolicy` handling and its `skipped_waiting` queue, and adds the TT admission rules on top — including the async-decode clean-up preservation added in `scheduler: preserve cleanup across prefill decode fallback (#135)`. Block-output models reserve a K-token placeholder block only on solo decode steps, and `_update_after_schedule` owns that predicate. `TTSchedulingMode` (line 76) is DEFAULT / DECODE_ONLY / PREFILL_ONLY. `_TT_TOKEN_TILE_SIZE` is imported from platform.py at line 359 to round prompt lengths up to a tile.

**Steps in execution.**

1. **Admit** — Pop waiting requests while the step fits the device shape.
2. **Reserve** — Block-output models reserve the K-token placeholder only on solo decodes.
3. **Emit** — Build a SchedulerOutput in the same shape upstream expects.
4. **Account** — get_tt_forced_reset_discard_counts() tracks forced resets.

**Questions.**

- **Q-SC1** Upstream reserves capacity for in-flight prefills (_inflight_prefill_reserved_blocks) and throttles on prefill_capacity_bound. Neither is implemented; open issue #133 tracks KV-pressure scheduling.
- **Q-SC2** scheduler_reserve_full_isl is read upstream and never read here.

#### IB · InputBatch

**In one line.** The flat, pre-baked view of the batch the device actually wants to see.

**What it does.** vLLM normally assembles this per step. TT builds it once and keeps it, because re-packing host tensors every step would cost more than the model forward does. Per-lane batches let several independent replicas share one process.

**How it's built.** The plugin reimplements `InputBatch` (vLLM's `gpu_input_batch.InputBatch` is _not_ subclassed, so upstream churn there is decoupled) at `src/vllm_tt_plugin/input_batch.py:218`, with `TTLaneInputBatch(InputBatch)` at line 801 for lane-DP. It still imports upstream's shared types: `CachedRequestState` from `vllm.v1.worker.gpu_input_batch` (line 23) and `MultiGroupBlockTable` from `vllm.v1.worker.block_table` (line 22) — those two are the live coupling. `SEED_NONE_SENTINEL` (line 42) distinguishes "no seed" from seed `-1`, which vLLM treats as equivalent. Both runtime guards here raise rather than assert, because `python -O` strips asserts: `build_cached_request_state` refuses a pooling request at lines 61-69, and `add_request` bounds the caller-supplied `req_index` at lines 315-322.

**Steps in execution.**

1. **Add request** — Allocate a slot, copy prompt state via build_cached_request_state.
2. **Rebuild cache state** — apply_cached_req_state_update refreshes CachedRequestState.
3. **Slice** — slice_tt_sampling_params produces the per-step view.
4. **Gather** — _gather_multi_modal_inputs assembles image features.

**Questions.**

- **Q-IB1** build_cached_request_state / apply_cached_req_state_update are the extension seam for prompt_logprobs. Only the builder is tested directly; the updater is not.
- ~~**Q-IB2** The two pooling asserts at lines 61-63 and 338 should become ValueError, or be made unreachable by a config-time runner_type refusal.~~ ✓ Both resolved differently, on 2026-09-27. The one in `add_request` at 338 was deleted rather than converted: it is unreachable, because the scheduler replaces an embeds-only prompt with placeholder ids, `build_cached_request_state` raises NotImplementedError for one first, and `CachedRequestState.__post_init__` refuses a state whose token ids and embeds are both None. The `req_index` bound next to it became an IndexError, because that one is a real invariant on a caller-supplied integer. `build_cached_request_state` already raised. Covered by tests/test_pooling_refusal.py, including under `python -O`.

### Execution

#### WK · TTWorker

**In one line.** The per-process handle on the mesh: opens devices, sizes the KV pool, loads the model.

**What it does.** One of these exists per engine process. It opens the device mesh, decides how much KV cache fits, loads the model, and then runs steps on request. It is also the object vLLM pokes with utility RPCs.

**How it's built.** class `TTWorker(WorkerBase)` at `src/vllm_tt_plugin/worker.py:184`. `init_device()` (220) binds `TT_VISIBLE_DEVICES` _before_ calling `check_and_update_config`, because tt-metal latches the visible set at first cluster construction and never re-reads the variable. `get_kv_cache_spec()` (283) prefers a `get_kv_cache_spec` classmethod on the model class for hybrid models and otherwise builds one homogeneous spec under the layer name `"foo"` (363). `determine_available_memory()` (401) does not profile: it returns a synthetic byte budget, `page_size * num_blocks * group_size`, computed by `_available_kv_cache_memory_bytes_for_num_blocks` (154) from `get_kv_cache_groups` and `get_uniform_page_size`, so the upstream planner in a different process reconstructs the same count. `initialize_from_config` (427) and `compile_or_warm_up_model` (446) follow. `update_max_model_len` (435) exists because WorkerBase has no such hook and the engine calls it by RPC. The four members upstream reaches but WorkerBase does not declare are also answered here, each by name: `get_model` (284), which fixes `get_model_inspection` and `apply_model` transitively, the LoRA quartet `add_lora`/`remove_lora`/`pin_lora`/`list_loras` (307-333), and `execute_dummy_batch` (335). The LoRA four and the dummy batch raise by name rather than staying silent, because a config-time refusal that something bypassed should still fail where the operator can see it.

**Steps in execution.**

1. **Bind devices** — Bind TT_VISIBLE_DEVICES before any mesh construction.
2. **Open mesh** — open_mesh_device() honours model_capabilities["fabric_config"].
3. **Size KV** — get_num_available_blocks_tt() then _fit_block_output_max_model_len().
4. **Build runner** — TTModelRunner is constructed here, once.

**Questions.**

- ~~**Q-WK1** get_model is absent, so get_model_inspection and apply_model raise NotImplementedError.~~ ✓ Fixed 2026-09-27. `get_model` at worker.py:284 answers it, which fixes both members that build on it transitively.
- ~~**Q-WK2** The four LoRA methods and get_punica_wrapper are absent.~~ ✓ Added 2026-09-27. `get_punica_wrapper` (platform.py:1429) and the LoRA quartet (worker.py:307-333) each raise by name. They stay unreachable while validate_lora refuses --enable-lora at config time, but a bypassed refusal should fail where the operator can see it rather than as a bare NotImplementedError.
- ~~**Q-WK3** get_cache_block_size_bytes is absent.~~ ✓ Left absent, decided 2026-09-27. Its docstring says "used in speculative decoding" but the only occurrence in the whole 0.26.0 tree is the definition itself, so an override would be dead code.

#### RN · TTModelRunner

**In one line.** Turns one scheduler step into one device forward pass, and the result back into something vLLM understands.

**What it does.** This is the engine room. It takes the step the scheduler produced, builds the tensors the device expects, runs the model, reads back tokens, and assembles the output structure vLLM asked for.

**How it's built.** class `TTModelRunner` at `src/vllm_tt_plugin/model_runner.py:181` — the largest file in the plugin at 2,899 lines. It owns `get_supported_tasks`, the prefill and decode paths, the sampling handoff (`Sampler` from `vllm.v1.sample.sampler`, line 33), and the output types. `_gather_multi_modal_inputs` (834-903) assembles image features and now has host coverage in `tests/test_multimodal_gather.py`. `prompt_logprobs_dict` routes through `_compute_prompt_logprobs_dict` at both output sites (1954, 2613) rather than being hard-coded to `None`; the packing lives in `logprobs.build_prompt_logprobs` and the platform still refuses `prompt_logprobs` at `platform.py:2207-2219`, because the seam is inert in situ — see the Logprobs structure. The runner also holds the speculative wiring that step 6.1 makes reachable.

**Steps in execution.**

1. **Step in** — Take SchedulerOutput, resolve the step plan via get_tt_step_plan().
2. **Build tensors** — Slice TTModelInput / TTSamplingParams for the device.
3. **Forward** — Call the tt-metal generator on the mesh.
4. **Sample** — On-device or host sampling per compat_sampling_required().
5. **Step out** — Assemble ModelRunnerOutput, or defer it for async decode.

**Questions.**

- **Q-RN1** prompt_logprobs still needs three pieces: a tt-metal generator returning one logits row per prompt position, the two output sites passing that output into the seam, and a per-chunk accumulation for chunked prefill. The refusal stands until all three land.
- **Q-RN2** This file is a growth risk: AGENTS.md §7 asks new branches here to move to a named neighbour.

#### AD · Async decode

**In one line.** Overlaps the device readback with the next step, so the host is never the bottleneck.

**What it does.** Waiting for the device is dead time. This hands the token readback off as a future, lets the next step start, and makes sure the deferred result is read back exactly once no matter who asks first.

**How it's built.** `src/vllm_tt_plugin/async_decode.py` (945 lines). `DeferredDecodeOutput(AsyncModelRunnerOutput)` at line 85 runs the deferred device readback exactly once from whichever caller reaches it first; `AsyncTTModelRunnerOutput(DeferredDecodeOutput)` at 148 adds the non-blocking single-process submission and covers both plain single-process decode and lane-DP decode. `SEED_NONE_SENTINEL` and `get_tt_forced_reset_discard_counts()` are imported for per-step seed and reset accounting. Device logprobs are only available on multi-device setups, and only for the sampled token — tracked upstream as tt-metal#34077.

**Steps in execution.**

1. **Submit** — Post the decode without blocking on the readback.
2. **Defer** — Wrap the pending readback in DeferredDecodeOutput.
3. **Continue** — Let the next step build while the device works.
4. **Resolve once** — The first caller drains the deferred readback.

### Parallelism

#### LC · TTLaneCoordinator

**In one line.** Several independent KV replicas in one process, batched together on a gathered device batch.

**What it does.** Lane-DP gives you data parallelism without several processes: one engine, several independent copies of the model and its KV cache, all driven from a single scheduler that merges their work into one device call.

**How it's built.** class `TTLaneCoordinator(SchedulerInterface)` at `src/vllm_tt_plugin/lane_scheduler.py:242`. It owns one fully independent `TTScheduler` per lane and implements the whole abstract interface (`interface.py:37-254`). `get_tt_step_plan()` produces a `TTStepPlan` that the runner consumes. `get_kv_connector()` (815) returns `None` — which is byte-identical to the `SchedulerInterface` default at `interface.py:248-249`, so the override is redundant. Lane folding is a Galaxy generator-version check plus an identity test on `model_class.__name__ == "GptOssForCausalLM"`; AGENTS.md §6 records that as leftover identity gating not to be copied.

**Steps in execution.**

1. **Own lanes** — One independent TTScheduler per lane.
2. **Merge** — Fold lane step plans into a gathered batch.
3. **Execute** — Runner runs the merged batch once.
4. **Split out** — Per-lane outputs route back to each scheduler.

**Questions.**

- **Q-LC1** get_kv_connector is redundant with the base default (interface.py:248-249). Pre-existing; not removed here because it is an unrelated cleanup.

#### DP · DP discovery

**In one line.** Works out which devices this rank owns before anyone builds a mesh.

**What it does.** In multi-process data parallelism every rank needs to agree on which devices are its own, without any of them guessing. This is the job that settles it, and it has a timeout because a rank that never arrives would otherwise hang the launch.

**How it's built.** `src/vllm_tt_plugin/utils/dp_discovery.py` (270 lines): `run_standard_dp_visible_device_group_discovery()`, `split_standard_dp_discovery_result()`, `parse_mesh_grid()` and `format_tt_visible_devices()`. Results are cached per `TT_VISIBLE_DEVICES` string in `TTPlatform._standard_dp_visible_device_groups` and `_standard_dp_mesh_grids` — both are in `_TT_PLATFORM_CONFIG_ATTRS` in `tests/conftest.py`, so they are process-global state that tests must reset. `_resolve_mesh_grid()` in worker.py (127) reads that cache. A discovery-join timeout was added with the variable-block-output DP work (#118).

**Steps in execution.**

1. **Gather** — Each rank publishes the device set it can see.
2. **Join** — Ranks agree on a partition; the join has a timeout.
3. **Split** — split_standard_dp_discovery_result() hands each rank its group.
4. **Cache** — Store the group string and its mesh grid for the process.

**Questions.**

- ~~**Q-DP1** execute_dummy_batch is the data-parallelism plugin surface, and TT ships standard DP. Whether TT's DP path can reach it is open.~~ ✓ Yes, reachable, and answered 2026-09-27. The chain is llm_engine.py:202 → :297-299 → core.py:2055 → core.py:921 → executor/abstract.py:250. It is reached by string dispatch, so a missing method was an AttributeError from the worker loop. `TTWorker.execute_dummy_batch` (worker.py:335) now exists and refuses loudly; doing a real dummy forward needs a validated device forward on the mesh, which is a paired tt-metal change.

#### BO · Block-output gate

**In one line.** Admission rules for models that commit several tokens at once.

**What it does.** Most models emit one token per step. A few — diffusion-style models — emit a whole block, and only when they are the only request running. This is the checkpoint that decides who is allowed to have that speedup, and how many output tokens anyone may ask for.

**How it's built.** The gate reads `output_tokens_per_step` from `model_capabilities` and relaxes five upstream rules when it is greater than 1: `max_num_seqs 1`, data parallelism, the distributed-executor backend allow-list (which gains `mp`), the async-scheduling refusal, and the block-output sampling mode (which then accepts `decode_only` as well as `all`). The optional `tt_adaptive_block_output` capability commits the block only on solo decode steps and falls back to a plain width-1 baseline whenever two or more requests batch — never worse, just not faster. `tt_adaptive_block_max_prompt_tokens` (default 0 = no limit) serves over-long prompts as baseline for their whole lifetime. Canvas arithmetic is `_resolve_block_output_max_tokens()` (2121); `_fit_block_output_max_model_len()` shrinks `--max-model-len -1` to what the KV pool can hold. The separate pause guard at `platform.py:956-1003` refuses `pause_generation(mode="keep", clear_cache=True)` with live requests — that is unrelated to sleep mode.

**Steps in execution.**

1. **Read capability** — output_tokens_per_step and the adaptive flags off model_capabilities.
2. **Relax or hold** — Block width > 1 relaxes five gates; width 1 holds them.
3. **Reserve** — Only solo decode steps get the K-token placeholder block.
4. **Fit** — max_model_len is fitted down to the KV pool capacity.

**Questions.**

- **Q-BO1** A present capability value that contradicts another resolved property must raise — e.g. block output plus prefix caching, or DiffusionGemma without output_tokens_per_step > 1.

### Optional features

#### SO · Structured output

**In one line.** Turns a JSON schema into a bitmask that makes invalid tokens impossible to sample.

**What it does.** Ask for JSON and get JSON, every time. The schema is compiled into a mask of which tokens are currently legal, and the sampler never considers the rest.

**How it's built.** `src/vllm_tt_plugin/structured_output.py` (100 lines) — `has_structured_outputs()` and `reorder_grammar_bitmask_for_tt_batch()`. The reorder exists because the device batches differently from the order the scheduler emitted grammars in. `_apply_check_and_update_config` forces `structured_outputs_config.disable_any_whitespace = True` (platform.py:1592) for every model, not just block ones: xgrammar and guidance allow arbitrary inter-field whitespace, and under greedy decoding the model can pick a whitespace token as the argmax indefinitely, exhausting the token budget before emitting a property name and returning truncated, unparseable JSON. Masking whitespace makes that loop structurally impossible. The backend stays "auto" so schemas xgrammar cannot compile still fall back to guidance.

**Steps in execution.**

1. **Compile** — Schema becomes a grammar per request.
2. **Collect** — Scheduler emits GrammarOutput for scheduled requests.
3. **Reorder** — Reindex the bitmask rows into device batch order.
4. **Mask** — Illegal tokens are excluded from sampling.

#### LP · Logprobs

**In one line.** Returns the probabilities behind the tokens, from the device where they were computed.

**What it does.** Logprobs tell you how confident the model was. On Tenstorrent they are produced on the device during sampling, which is fast, but has limits on how many you can get at once.

**How it's built.** `src/vllm_tt_plugin/logprobs.py` (190 lines) — `build_logprobs_from_topk()` (line 10) and `build_device_logprobs()` (line 54) pack _sampled_ tokens off the device top-32; `build_prompt_logprobs()` and `shift_prompt_target_token_ids()` (lines 84, 158) are the separate **prompt** path, because a prompt position names an arbitrary token rather than the sampled one, so it needs a full log_softmax gather instead of a match against the top-32. That is also why the first two must not be reused here: feeding a prompt token outside the top-32 into `build_logprobs_from_topk` would silently report the rank-0 token's logprob. Upstream tensors `LogprobsTensors` and `LogprobsLists` are reused. The constraint is recorded in `TTPlatform.compat_sampling_required` (`platform.py:2233-2260`): device logprobs need a multi-device setup and return only the sampled token's logprob, so any `logprobs > 1`, or any logprobs at all on a single device, forces host sampling. Tracked upstream as tt-metal#34077. The device computes top-32; the OpenAI API caps at 20, so `max_logprobs` is clamped at `platform.py:1602-1609` with a warning naming both the requested and the clamped value.

**Steps in execution.**

1. **Decide** — compat_sampling_required() picks device or host sampling.
2. **Sample** — Device returns the sampled token plus, where allowed, top-k.
3. **Pack** — build_device_logprobs() or build_logprobs_from_topk().
4. **Clamp** — max_logprobs is capped at 20 before the request is admitted.

**Questions.**

- **Q-LP1** prompt_logprobs is a different code path and is still not served: the packing is written and tested, but both call sites pass no per-position logits into the seam, so the platform refusal is the correct answer. The seam raises rather than returning an empty result.

#### SD · Spec-decode contract

**In one line.** The types and the accept walk for speculative decoding — written, tested, and not yet reachable.

**What it does.** Speculative decoding guesses several tokens at once and checks them in a batch. The rules for which guesses are accepted already exist here as data types and as a host-side walk. What is missing is the runtime that drafts them.

**How it's built.** `src/vllm_tt_plugin/spec_decode.py` (385 lines) holds the contract types including `PLACEHOLDER_TOKEN_ID`; `spec_accept.py` (512 lines) implements the accept walk and imports `apply_top_k_top_p_pytorch` from `vllm.v1.sample.ops.topk_topp_sampler` at line 312 to apply the model's own sampling distribution to the candidates. Neither module is imported by any runtime module — only by each other and by `tests/spec/` — and `feature_support.validate_speculative_decoding` (line 53) still refuses `speculative_config`. The contract landed before the runtime on purpose: see the decisions table. The runtime half exists as open PRs #125, #127, #130, #131 and #142, with #131 the model-drafter branch that #142 builds on.

**Steps in execution.**

1. **Contract** — spec_decode.py declares the types and the placeholder token.
2. **Draft** — Not yet wired: the model-drafter branch is #131.
3. **Accept walk** — spec_accept.py checks candidates against the target distribution.
4. **Reach it** — A host test asserts these modules are in the runner's import graph.

**Questions.**

- **Q-SD1** On main today the speculative contract is unreachable at runtime. That is the exact condition tests/test_spec_runtime_wiring.py must fail on.
- **Q-SD2** The drafter branch, #142, and the int64-seed fix #141 all predate the current branch and must have this work's changes re-applied on top.

### Upstream coupling

#### UC · Upstream coupling

**In one line.** Every place the plugin binds to vLLM internals, and therefore every place a vLLM bump can break it.

**What it does.** vLLM does not promise stability on its internals, and this plugin depends on a lot of them. This is the honest inventory of the bindings, so a version bump is a checklist rather than a surprise.

**How it's built.** Derived with `grep -rnE "^\s*(from|import)\s+vllm" src/` per the repo's own playbook (`.github/prompts/vllm-upgrade-assessment.prompt.md:53-115`). Subclassed bases: `TTPlatform(Platform)`, `TTWorker(WorkerBase)`, `TTScheduler(AsyncScheduler)`, `TTLaneCoordinator(SchedulerInterface)`, `TTModelLoader(BaseModelLoader)`, `DeferredDecodeOutput`/`AsyncTTModelRunnerOutput(AsyncModelRunnerOutput)`, `TTLaunchPlan(EngineLaunchPlan)`, `TTCoreEngineLauncher(CoreEngineLauncher)`. Constructed upstream dataclasses: `ModelRunnerOutput`, `SchedulerOutput`, `SamplingMetadata`, `CachedRequestState`, `LogprobsTensors`, `KVCacheConfig` and specs, `EngineCoreOutputs`, `GrammarOutput`, `SchedulerStats`, `MultiGroupBlockTable(...)`. Seven runtime monkeypatches in `platform.py` at lines 636, 696, 809, 875, 896, 956. Seven internal `additional_config` keys. Ten `model_capabilities` keys — note that `tt_block_kv_extent_tokens`, `supports_device_penalties` and `fabric_config` are consumed in code but are missing from the AGENTS.md §6 table. The per-member matrix for every coupled symbol — plugin site, base behaviour, and whether it is called unconditionally — is not committed here; it is re-derived per version bump using the playbook above, which is the point of the playbook.

**Steps in execution.**

1. **Grep imports** — The four playbook categories, re-derived every time.
2. **Diff the tag** — BASELINE (the 0.26.0 pin) against the target tag.
3. **Classify** — BREAKING-IMPORT / -CONSTRUCT / -OVERRIDE / (silent) / BEHAVIORAL / NONE.
4. **Plan** — One required edit per coupled symbol, with a file:line.

**Questions.**

- **Q-UC1** AGENTS.md §6 lists ten capability keys but three more are consumed in code. The table is incomplete.
- **Q-UC2** launcher.py runs its ImportError fallback: upstream CoreEngineLauncher / EngineLaunchPlan are absent at 0.26.0, so it is dormant and unhooked.

#### FS · Feature support

**In one line.** One loud refusal per upstream feature the TT backend does not serve.

**What it does.** Ask for something Tenstorrent cannot do and you get a clear error naming the flag you typed, at startup, instead of a wrong answer later or an AttributeError from deep inside vLLM.

**How it's built.** `src/vllm_tt_plugin/feature_support.py`, one `validate_<feature>(cls, vllm_config)` per feature, called in a fixed sequence from `_apply_check_and_update_config` (`platform.py:1590-1596`) at the point the three `assert`s used to sit (1561-1568). It is a separate module because `platform.py` is already 2,290 lines and AGENTS.md §7 says a new conditional branch there is a prompt to split into a neighbour; `config.py` is the accessor layer, not a validator, so a purpose-named module is the smaller change. Every function raises `ValueError` printing the offending value and naming the CLI flag — never `assert`, which `python -O` removes (AGENTS.md §2.4).

**Steps in execution.**

1. **Read config** — Each validator reads only the fields upstream actually defines.
2. **Compare** — Against the TT-supported set for that feature.
3. **Raise** — ValueError naming the offending value and the flag.
4. **No override** — Features the base class already refuses correctly get no validator at all.

**Questions.**

- **Q-FS1** A validator must never read a field that does not exist on the pinned vLLM: three rows in the original plan did exactly that and would have raised AttributeError instead of the intended clean refusal.
- **Q-FS2** request_validation refusals for n > 1, thinking_token_budget, repetition_detection and extra_args reuse the existing block-model message builder at platform.py:2257-2261 rather than writing a second one.

### Upstream vLLM 0.26.0

#### vP · Platform resolution

**In one line.** How upstream decides which platform it is running on.

**What it does.** Five built-in hardware backends compete with any out-of-tree plugin. Exactly one wins, and the loser is told so immediately rather than at the first feature that needs it.

**How it's built.** `vllm/platforms/__init__.py` at tag v0.26.0. `resolve_current_platform_cls_qualname()` walks the builtins {tpu, cuda, rocm, xpu, cpu} plus every `vllm.platform_plugins` callable, each inside a bare `try/except Exception: pass`. Two or more out-of-tree plugins raises `RuntimeError`; exactly one OOT wins outright over any builtin; zero matches yields `UnspecifiedPlatform` (`interface.py`), which sets `_enum` and `device_type = ""` and omits `device_name` entirely — so the inherited bare annotation has no value to read. `current_platform` is resolved through a module `__getattr__` so an OOT plugin can import `Platform` without re-entering resolution. `TTPlatform` wins by being the only OOT plugin, and `entrypoints.platform_plugin()` returns `None` when `import ttnn` fails so nothing is claimed. **Substituted by** `TTPlatform` (`platform.py:1337`), which is claimed through the `tt` platform-plugin entry point and declines by returning `None` when `import ttnn` fails.

**Steps in execution.**

1. **List plugins** — Entry points from both groups, filtered by VLLM_PLUGINS.
2. **Run detectors** — Each callable in try/except; a non-None return claims the platform.
3. **Prefer OOT** — One OOT beats any builtin; two OOT is a RuntimeError.
4. **Resolve lazily** — current_platform resolved on first access, not at import.

#### vI · InputProcessor

**In one line.** Turns a client request into validated, per-request state — and is where platform validation runs.

**What it does.** The front desk. It takes the prompt, the sampling parameters and the platform, and it is the last point where a request can be refused before it becomes work for the device.

**How it's built.** `vllm/v1/engine/input_processor.py` at v0.26.0. It calls `current_platform.validate_request(processed_inputs, params)` at line 296 — unconditionally, once per request, with the **parent** parameters. That ordering is why `n > 1` is not a gap on TT: this validator sees the parent `n`, and the fan-out to `n == 1` children happens later and downstream. **Consumed by** `TTPlatform.validate_request` (`platform.py:2193`), which sees the parent params, including `n`, before the fan-out below.

**Steps in execution.**

1. **Build inputs** — Tokenise, apply the chat template, gather multimodal features.
2. **Validate** — current_platform.validate_request — per request, on the parent params.
3. **Block hashes** — Compute prefix hashes when prefix caching is on.

**Questions.**

- **Q-vI1** Platform validation runs here, before the request becomes a Request, so a refusal costs nothing.

#### vS · Sampler and parallel sampling

**In one line.** Upstream turns one request with n>1 into n independent single-sample children.

**What it does.** Multi-completion is bookkeeping, not a sampling mode. The parent request is expanded into separate child requests, each carrying its own parameters, and the outputs are re-joined under one id.

**How it's built.** `vllm/v1/engine/parallel_sampling.py` (150 lines) and `vllm/v1/sample/sampler.py` (436 lines). `child_sampling_params.n = 1` at `parallel_sampling.py:74`; `get_child_info(index)` returns the child id and those params at `:83`; `AsyncLLM` expands at `async_llm.py:388`; `OutputProcessor` re-joins at `output_processor.py:326`. The consequence for a plugin is that the model runner only ever sees `n == 1`, whatever the client asked for. **Reached by TT only through the child-request shape.** Because the expansion happens here, the model runner never sees `n != 1`.

**Steps in execution.**

1. **Expand** — ParentRequest.get_child_info per index, each child n = 1.
2. **Schedule** — Children are independent requests to the scheduler.
3. **Re-join** — OutputProcessor aggregates children under the parent external id.

**Questions.**

- **Q-vS1** This is why refusing n > 1 in a platform validator would break a request that already works.

#### vE · EngineCore

**In one line.** The engine loop: take requests, step the scheduler, run the model, hand back outputs.

**What it does.** The process that owns the model. It polls for work, asks the scheduler what to run this step, sends that to the workers, and puts the results on the output queue.

**How it's built.** `vllm/v1/engine/core.py` at v0.26.0, 2407 lines. `VllmConfig.__post_init__` calls the platform hook here, so configuration is settled before the loop starts. `run_busy_loop` is defined twice: `EngineCoreProc.run_busy_loop` at line 1358 for the single-process case, and `DPEngineCoreProc.run_busy_loop` at line 2024 for data parallel. Only the DP one fires `execute_dummy_batch` when a step executed nothing (lines 2049-2055) — the call chain that makes a TT DP rank that idles while a peer runs reach that RPC. `TTWorker.execute_dummy_batch` (worker.py:335) now answers it by name, because `Executor.collective_rpc` dispatches by string and a missing method was an `AttributeError` from the worker loop. It refuses loudly rather than doing a dummy forward: the TT equivalent needs a validated device forward on the mesh, which is a paired tt-metal change. **Substituted only where the loop is DP-specific.**

**Steps in execution.**

1. **Poll** — _process_input_queue takes new and aborted requests.
2. **Step** — scheduler.schedule() then the worker executor.
3. **Publish** — Outputs go to the output queue; stats are drained.

**Questions.**

- **Q-vE1** run_busy_loop is the DP-only loop; the single-process path never reaches execute_dummy_batch.

#### vS · Upstream scheduler

**In one line.** Fills a token budget each step, with no notion of a prefill or decode phase.

**What it does.** Upstream asks one question every step: how many tokens can I afford? Whatever fits, fits — prefill and decode are just tokens. That simplicity is what the TT scheduler has to give up.

**How it's built.** `vllm/v1/core/sched/scheduler.py` at v0.26.0, 2823 lines. `schedule()` opens with `token_budget = self.max_num_scheduled_tokens`, spends it in a running pass, then admits waiting requests — under woosuk's explicit NOTE that "There's no 'decoding phase' nor 'prefill phase' in the scheduler." `TTScheduler` inverts this: it classifies the step first and hides the ineligible half of the state from the base class, which is why `throttle_prefills` is accepted and ignored on TT and the DP prefill-balancing cadence is unreachable there. **Substituted by** `TTScheduler` (`scheduler.py:170`): it classifies the step first, then hides the ineligible half of the state from this base class.

**Steps in execution.**

1. **Budget** — One scalar for the whole step.
2. **Running pass** — Decode and prefill chunks already resident, decrementing the budget.
3. **Waiting pass** — Admit new work while tokens remain and KV is available.
4. **Preempt** — PRIORITY: max(running, key=(priority, arrival_time)) at scheduler.py:576-581. FCFS: running.pop() at 601.

**Questions.**

- **Q-vS1** TT gives up the single-budget model because the device can only take work in fixed shapes.

#### vK · KV cache manager

**In one line.** Owns the block pool, the prefix cache, and the decision about what fits.

**What it does.** The memory bookkeeper. It hands out blocks, remembers which blocks hold which prompt prefixes, and is the reason a second request with the same prompt is cheap.

**How it's built.** `vllm/v1/core/kv_cache_manager.py` at v0.26.0, 760 lines. `get_computed_blocks(request)` (line 207) is what the scheduler consults before prefilling; `allocate_slots` (283) hands out the rest; `cache_blocks` (689) records the full blocks of a finished request. The pool itself is built by the coordinator, not here: `BlockPool(num_gpu_blocks=kv_cache_config.num_blocks, ...)` at `kv_cache_coordinator.py:90-97`. So the pool is sized by whatever the engine KV planner derived from the `determine_available_memory` result — which is why a synthetic byte budget returned by the TT worker is enough to control the count from another process. **Substituted by** `TTWorker.determine_available_memory` (`worker.py:401`), which returns a synthetic byte budget instead of profiling.

**Steps in execution.**

1. **Allocate** — get_computed_blocks for the prefix hit, then allocate_slots for the rest.
2. **Hash** — Block hashes come from cache_config.prefix_caching_hash_algo.
3. **Cache on free** — A finished request's full blocks are recorded, not discarded.

**Questions.**

- **Q-vK1** The TT worker does not profile: it returns a synthetic byte budget so the planner in another process reconstructs the same block count.

#### vX · Executor and collective RPC

**In one line.** Reaches the worker by method NAME, which is why a missing override is an AttributeError.

**What it does.** The layer between the engine and the worker process. It does not import the worker class; it looks methods up by string and calls whatever it finds. That is the whole reason an unimplemented optional feature fails so unhelpfully.

**How it's built.** `vllm/v1/executor/abstract.py` at v0.26.0. Every optional worker feature is dispatched by name: `initialize_from_config`, `compile_or_warm_up_model`, `determine_available_memory`, `get_kv_cache_spec`, `execute_dummy_batch`, `take_draft_token_ids`, `sleep`, `wake_up`. None of those names is declared on `WorkerBase`; the lookup lands on `WorkerWrapperBase.__getattr__` (`worker_base.py:333`). So a plugin that omits one gets `AttributeError` from inside a worker RPC loop, and only when the feature is actually used. **Kept as-is.** The executor is upstream code; a plugin only has to answer the names it wants, which is why a missing one is an AttributeError rather than a clean refusal.

**Steps in execution.**

1. **Select** — Worker class resolved from parallel_config.worker_cls.
2. **RPC** — collective_rpc("method") dispatches by string to each worker.
3. **Fold** — Results reduce into compilation_config, KV counts and stats.

**Questions.**

- **Q-vX1** This string dispatch is why an unimplemented optional feature must fail loudly at config time, not be left to surface as an AttributeError.

#### vW · Reference Worker

**In one line.** What a complete WorkerBase implementation looks like, and the bar TTWorker is measured against.

**What it does.** The GPU implementation. It shows which optional methods exist in practice and what they do — LoRA, sleep, dummy batches — so you can see which ones a plugin is declining rather than missing.

**How it's built.** `vllm/v1/worker/gpu_worker.py` at v0.26.0, 1453 lines. Implements `get_model` (929), the four LoRA methods (add_lora 1237, remove_lora 1240, list_loras 1243, pin_lora 1246, one-line delegations to the model runner), `execute_dummy_batch` (1233, sized by `model_runner.uniform_decode_query_len`), `sleep(level)` (192) and `wake_up(tags)` (227) — the last three absent from `WorkerBase` entirely. `TTWorker` mirrors the first group and omits only `sleep` and `wake_up`, which cannot be reached on TT because `ModelConfig.__post_init__` already refuses `--enable-sleep-mode` for any non-CUDA/ROCm/XPU platform; the LoRA quartet and `execute_dummy_batch` are present and refuse by name. **Substituted by** `TTWorker` (`worker.py:184`): a mesh device instead of a GPU, a tt-metal generator instead of vLLM layers, and no profiling run.

**Steps in execution.**

1. **init_device** — Set the device up, then run the platform hook again per subprocess.
2. **load_model** — Weights onto the device; the runner holds the model.
3. **execute_model** — One step; returns None to defer sampling.

**Questions.**

- **Q-vW1** TT init_device re-runs TTPlatform.check_and_update_config because multiprocessing means platform class state is per process.

#### vR · GPU model runner

**In one line.** The upstream engine room: step in as tensors, step out as ModelRunnerOutput.

**What it does.** The counterpart TTModelRunner replaces. It builds the batch tensors, runs the model, samples, and assembles the output structure the engine expects.

**How it's built.** `vllm/v1/worker/gpu_model_runner.py` at v0.26.0, 7846 lines. Owns `get_supported_generation_tasks` and `get_supported_pooling_tasks` (3309, 3327) — the members that decide what a model can be asked to do, and where `runner_type` becomes visible. It is also the file whose churn the plugin insulates itself from by reimplementing `InputBatch` rather than subclassing `gpu_input_batch.InputBatch`. **Substituted by** `TTModelRunner` (`model_runner.py:181`): a tt-metal generator instead of vLLM model layers, over a persistent `InputBatch` rather than a per-step rebuild.

**Steps in execution.**

1. **Build** — InputBatch and the persistent state from a SchedulerOutput.
2. **Forward** — Attention, MLP, sampling.
3. **Emit** — ModelRunnerOutput with sampled tokens and logprobs.

**Questions.**

- **Q-vR1** The plugin reimplements its own InputBatch, so churn in gpu_input_batch is decoupled; the shared types it does import are not.

## Flows (representative packets)

Payload shapes are what the design implies, not measured traffic.

### Lifecycle

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | EP → MR | activate | `{"group":"vllm.general_plugins","name":"tt_model_registry"}` |
| 2 | MR → PF | platform claimed | `{"cls":"vllm_tt_plugin.platform.TTPlatform"}` |
| 3 | PF → CF | check_and_update_config | `{"keys_written":["_tt_output_tokens_per_step","_tt_resolved_lane_count"]}` |
| 4 | CF → SC | scheduler built | `{"mode":"DEFAULT","max_num_seqs":1}` |
| 5 | SC → IB | scheduled | `{"num_scheduled_tokens":{"req-7":128}}` |
| 6 | IB → RN | step input | `{"batch_size":1,"block_size":64}` |
| 7 | RN → RN | forward + sample | `{"mesh":"1,1","num_tokens_out":1}` |
| 8 | RN → SC | update_from_output | `{"finished":[],"token_ids":[4821]}` |

### Config

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | PF → MR | resolve class | `{"arch":"TTGptOssForCausalLM","source":"tt-metal generator"}` |
| 2 | MR → PF | capabilities | `{"output_tokens_per_step":1,"supports_prefix_caching":true}` |
| 3 | PF → FS | refuse unserved | `{"flag":"--kv-cache-dtype","value":"fp8"}` |
| 4 | PF → BO | block gate | `{"output_tokens_per_step":1,"block_contract":null}` |
| 5 | BO → CF | fit max_model_len | `{"max_model_len":8192,"kv_blocks":4096}` |
| 6 | CF → WK | config pickled | `{"additional_config":{"_tt_output_tokens_per_step":1}}` |

### Lane-DP fold

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | SC → LC | lane plan | `{"lanes":2,"per_lane_waiting":[3,1]}` |
| 2 | LC → IB | per-lane batch | `{"lane_0":3,"lane_1":1}` |
| 3 | IB → RN | gathered batch | `{"total_seqs":4,"gathered":true}` |
| 4 | RN → LC | split outputs | `{"lane_0":3,"lane_1":1}` |
| 5 | LC → SC | merged schedule | `{"num_scheduled_tokens":512}` |

### Standard-DP discovery

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | DP → DP | publish visible set | `{"TT_VISIBLE_DEVICES":"0,1,2,3","ranks":4}` |
| 2 | DP → DP | join with timeout | `{"groups":["0","1","2,3"],"timeout_s":30}` |
| 3 | DP → WK | split result | `{"this_rank_group":"2,3","mesh_grid":[1,2]}` |
| 4 | WK → DP | cache group + grid | `{"group":"2,3","grid":[1,2]}` |

### Block output

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | PF → BO | capability read | `{"output_tokens_per_step":4,"adaptive":true}` |
| 2 | BO → FS | relax five gates | `{"max_num_seqs":1,"dp":false,"backend":"mp","async":false}` |
| 3 | BO → SC | reserve placeholder | `{"batch_size":1,"k_tokens":4}` |
| 4 | SC → BO | fallback decision | `{"batch_size":3,"served_as":"baseline width-1"}` |
| 5 | BO → RN | canvas arithmetic | `{"prompt_len":1300,"output_tokens":8,"tile":32}` |

### Optional

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | SC → SO | GrammarOutput | `{"req_ids":["req-7"],"masks":1}` |
| 2 | SO → IB | reorder for device batch | `{"from":[0],"to":[0]}` |
| 3 | IB → RN | bitmask | `{"illegal_masked":true,"disable_any_whitespace":true}` |
| 4 | RN → SC | valid token only | `{"token_id":1274,"text":"{\""}` |
| 5 | RN → AD | deferred readback | `{"pending":true,"token_id":4821}` |
| 6 | SC → SD | draft request | `{"num_draft":4,"lookahead":4}` |
| 7 | SD → SC | accept walk | `{"accepted":3,"rejected_at":3}` |

### TT vs upstream

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | UC → PF | Platform members | `{"overridden":14,"refused":11}` |
| 2 | UC → WK | WorkerBase members | `{"overridden":9,"missing_loud":6}` |
| 3 | UC → SC | scheduler overrides | `{"subclass":"AsyncScheduler"}` |
| 4 | UC → RN | runner types | `{"subclass":"AsyncModelRunnerOutput"}` |
| 5 | UC → FS | refusal surface | `{"config_time":11,"request_time":4}` |
| 6 | SC → vSched | replaces | `{"change":"classify the step, then hide the ineligible half"}` |
| 7 | WK → vWorker | replaces | `{"change":"mesh device, no profiling"}` |
| 8 | RN → vRunner | replaces | `{"change":"tt-metal generator, persistent batch"}` |

## Questions — index

Reference by ID. ✓ resolved (with date) · otherwise open.

- **Q-EP1** (EP) Both hooks run in the API server and in every engine subprocess; is a failure there reported clearly enough to diagnose a partial install?
- **Q-PF1** (PF) The MRV1 pin is a removal clock: vLLM 0.29.0 deprecated MRV1 for removal in v0.32. When does that pin get revisited?
- **Q-PF2** (PF) Every refusal message must name the CLI flag the operator typed. Is there a test that proves each one does?
- **Q-CF1** (CF) AGENTS.md §8 requires any new TTPlatform class attribute to be added to _TT_PLATFORM_CONFIG_ATTRS in tests/conftest.py, or test state leaks. A new attribute here would trip that.
- **Q-MR1** (MR) Capabilities are declared by tt-metal model classes, not here. A plugin change that reads a new capability needs a matching tt-metal change in the same release.
- **Q-SC1** (SC) Upstream reserves capacity for in-flight prefills (_inflight_prefill_reserved_blocks) and throttles on prefill_capacity_bound. Neither is implemented; open issue #133 tracks KV-pressure scheduling.
- **Q-SC2** (SC) scheduler_reserve_full_isl is read upstream and never read here.
- **Q-IB1** (IB) build_cached_request_state / apply_cached_req_state_update are the extension seam for prompt_logprobs. Only the builder is tested directly; the updater is not.
- ~~**Q-IB2**~~ (IB) ✓ Both resolved differently, on 2026-09-27. The one in `add_request` at 338 was deleted rather than converted: it is unreachable, because the scheduler replaces an embeds-only prompt with placeholder ids, `build_cached_request_state` raises NotImplementedError for one first, and `CachedRequestState.__post_init__` refuses a state whose token ids and embeds are both None. The `req_index` bound next to it became an IndexError, because that one is a real invariant on a caller-supplied integer. `build_cached_request_state` already raised. Covered by tests/test_pooling_refusal.py, including under `python -O`.
- ~~**Q-WK1**~~ (WK) ✓ Fixed 2026-09-27. `get_model` at worker.py:284 answers it, which fixes both members that build on it transitively.
- ~~**Q-WK2**~~ (WK) ✓ Added 2026-09-27. `get_punica_wrapper` (platform.py:1429) and the LoRA quartet (worker.py:307-333) each raise by name. They stay unreachable while validate_lora refuses --enable-lora at config time, but a bypassed refusal should fail where the operator can see it rather than as a bare NotImplementedError.
- ~~**Q-WK3**~~ (WK) ✓ Left absent, decided 2026-09-27. Its docstring says "used in speculative decoding" but the only occurrence in the whole 0.26.0 tree is the definition itself, so an override would be dead code.
- **Q-RN1** (RN) prompt_logprobs still needs three pieces: a tt-metal generator returning one logits row per prompt position, the two output sites passing that output into the seam, and a per-chunk accumulation for chunked prefill. The refusal stands until all three land.
- **Q-RN2** (RN) This file is a growth risk: AGENTS.md §7 asks new branches here to move to a named neighbour.
- **Q-LC1** (LC) get_kv_connector is redundant with the base default (interface.py:248-249). Pre-existing; not removed here because it is an unrelated cleanup.
- ~~**Q-DP1**~~ (DP) ✓ Yes, reachable, and answered 2026-09-27. The chain is llm_engine.py:202 → :297-299 → core.py:2055 → core.py:921 → executor/abstract.py:250. It is reached by string dispatch, so a missing method was an AttributeError from the worker loop. `TTWorker.execute_dummy_batch` (worker.py:335) now exists and refuses loudly; doing a real dummy forward needs a validated device forward on the mesh, which is a paired tt-metal change.
- **Q-BO1** (BO) A present capability value that contradicts another resolved property must raise — e.g. block output plus prefix caching, or DiffusionGemma without output_tokens_per_step > 1.
- **Q-LP1** (LP) prompt_logprobs is a different code path and is still not served: the packing is written and tested, but both call sites pass no per-position logits into the seam, so the platform refusal is the correct answer. The seam raises rather than returning an empty result.
- **Q-SD1** (SD) On main today the speculative contract is unreachable at runtime. That is the exact condition tests/test_spec_runtime_wiring.py must fail on.
- **Q-SD2** (SD) The drafter branch, #142, and the int64-seed fix #141 all predate the current branch and must have this work's changes re-applied on top.
- **Q-UC1** (UC) AGENTS.md §6 lists ten capability keys but three more are consumed in code. The table is incomplete.
- **Q-UC2** (UC) launcher.py runs its ImportError fallback: upstream CoreEngineLauncher / EngineLaunchPlan are absent at 0.26.0, so it is dormant and unhooked.
- **Q-FS1** (FS) A validator must never read a field that does not exist on the pinned vLLM: three rows in the original plan did exactly that and would have raised AttributeError instead of the intended clean refusal.
- **Q-FS2** (FS) request_validation refusals for n > 1, thinking_token_budget, repetition_detection and extra_args reuse the existing block-model message builder at platform.py:2257-2261 rather than writing a second one.
- **Q-vI1** (vI) Platform validation runs here, before the request becomes a Request, so a refusal costs nothing.
- **Q-vS1** (vS) This is why refusing n > 1 in a platform validator would break a request that already works.
- **Q-vE1** (vE) run_busy_loop is the DP-only loop; the single-process path never reaches execute_dummy_batch.
- **Q-vS1** (vS) TT gives up the single-budget model because the device can only take work in fixed shapes.
- **Q-vK1** (vK) The TT worker does not profile: it returns a synthetic byte budget so the planner in another process reconstructs the same block count.
- **Q-vX1** (vX) This string dispatch is why an unimplemented optional feature must fail loudly at config time, not be left to surface as an AttributeError.
- **Q-vW1** (vW) TT init_device re-runs TTPlatform.check_and_update_config because multiprocessing means platform class state is per process.
- **Q-vR1** (vR) The plugin reimplements its own InputBatch, so churn in gpu_input_batch is decoupled; the shared types it does import are not.

## What the platform gives vs what we own

**Platform gives:** Entry-point discovery (<code>vllm.platform_plugins</code> / <code>vllm.general_plugins</code>), the <code>Platform</code> / <code>WorkerBase</code> / <code>SchedulerInterface</code> / <code>BaseModelLoader</code> base classes, the token-budget scheduler, the KV-cache planner, the executor and collective-RPC layer, the sampling stack, and the block-table primitives. The plugin consumes all of it and monkeypatches seven of its methods at runtime.

**We own:** Device abstraction (mesh, fabric, kernels, KV allocation), the model registry, three execution modes, block-output admission and canvas arithmetic, lane-DP folding, standard-DP discovery, the async-decode fast path, and the whole refusal surface for unsupported upstream features.

## Planned filesystem

```
src/vllm_tt_plugin/
  entrypoints.py · model_registry.py · loader.py      entry and model binding
  platform.py · config.py · logger.py                configuration and the Platform
  scheduler.py · lane_scheduler.py · input_batch.py  scheduling and batching
  worker.py · model_runner.py · model_input.py       execution
  async_decode.py · structured_output.py · logprobs.py
  spec_decode.py · spec_accept.py                    speculative contract
  feature_support.py                                 refusals for unserved features
  utils/dp_discovery.py
(upstream vLLM v0.26.0)                      read at that tag, not vendored here
system-atlas/atlas/                                  this atlas
```

## How this file is maintained

Generated from `system-atlas/atlas/data.mjs` by `bun system-atlas/atlas/build.mjs`, which also builds the interactive atlas (`atlas.html`). Edit the data file, rebuild, republish — never edit this file by hand.
