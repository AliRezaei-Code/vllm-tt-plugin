// Single source of truth for the vLLM TT Plugin architecture atlas.
// Build: bun system-atlas/atlas/build.mjs  → writes SYSTEM.md and atlas.html
//
// Baseline: vLLM 0.26.0 (tag v0.26.0 = 568afb3a13806beb53bb2e6bd518269357b237c0),
// pinned by docs/install-vllm-tt.sh:36. Every "how" below cites a file:line in
// this repo, or an upstream file:line readable at that tag. Upstream paths are
// given as <upstream-path>@v0.26.0 rather than as a local checkout, because the
// nested reference tree is gitignored and is not part of this repository.

export const META = {
  title: 'vLLM TT Plugin',
  artifactUrl: '',
  sourcePath: 'system-atlas/atlas/data.mjs',
  buildCmd: 'bun system-atlas/atlas/build.mjs',
  stats: [
    { k: 'System', v: 'vllm-tt-plugin' },
    { k: 'Roles', v: '26' },
    { k: 'Plugin', v: '12.7k lines' },
    { k: 'Upstream', v: '0.26.0' },
  ],
  intro: `**This file is the living source of truth for the plugin's architecture.** The interactive atlas, <code>SYSTEM.md</code>, and every claim in the research report rebuild from it.`,
  onePara: `This atlas covers **both repositories**: the <code>vllm-tt-plugin</code> that Tenstorrent maintains, and the stock vLLM 0.26.0 it plugs into. The plugin teaches stock vLLM how to run on Tenstorrent hardware without forking vLLM. It registers itself through two entry points, substitutes a Platform, a Worker, a Scheduler and a ModelRunner for the ones vLLM would pick, and routes every model through a tt-metal generator that owns the actual device kernels. It ships three execution modes, owns no model code of its own, and refuses loudly — rather than silently degrading — on every upstream vLLM feature the TT backend does not serve.`,
  costModel: [
    'The plugin adds no per-token compute of its own. Its cost is configuration-time: model registration, mesh construction, and KV-pool sizing happen once per process.',
    '`model_runner.py` (2,753 lines) and `platform.py` (2,260 lines) are the two files a reviewer watches for growth; AGENTS.md §7 asks new branches there to move to a named neighbour instead.',
  ],
  deepDive: '',
  platformGives: `Entry-point discovery (<code>vllm.platform_plugins</code> / <code>vllm.general_plugins</code>), the <code>Platform</code> / <code>WorkerBase</code> / <code>SchedulerInterface</code> / <code>BaseModelLoader</code> base classes, the token-budget scheduler, the KV-cache planner, the executor and collective-RPC layer, the sampling stack, and the block-table primitives. The plugin consumes all of it and monkeypatches seven of its methods at runtime.`,
  weOwn: `Device abstraction (mesh, fabric, kernels, KV allocation), the model registry, three execution modes, block-output admission and canvas arithmetic, lane-DP folding, standard-DP discovery, the async-decode fast path, and the whole refusal surface for unsupported upstream features.`,
  filesystem: `src/vllm_tt_plugin/
  entrypoints.py · model_registry.py · loader.py      entry and model binding
  platform.py · config.py · logger.py                configuration and the Platform
  scheduler.py · lane_scheduler.py · input_batch.py  scheduling and batching
  worker.py · model_runner.py · model_input.py       execution
  async_decode.py · structured_output.py · logprobs.py
  spec_decode.py · spec_accept.py                    speculative contract
  feature_support.py                                 refusals for unserved features
  utils/dp_discovery.py
(upstream vLLM v0.26.0)                      read at that tag, not vendored here
system-atlas/atlas/                                  this atlas`,
};

export const DECISIONS = [
  { axis: 'Integration form', decision: 'Plugin-only integration. No TT feature is ever landed in vLLM core; where an entry point cannot express the behaviour, a runtime monkeypatch lives beside the other _install_* helpers in platform.py.', adr: '[AGENTS.md §1](../AGENTS.md)' },
  { axis: 'Gating contract', decision: 'New behaviour gates on the model class\'s model_capabilities dict, never on a model-type allowlist. Absent keys default via .get; a present value that contradicts another resolved property raises.', adr: '[AGENTS.md §6](../AGENTS.md)' },
  { axis: 'Dependency set', decision: 'docs/install-vllm-tt.sh owns the whole dependency set, because vLLM\'s PyPI metadata is generated on a CUDA machine and uv would otherwise resolve requirements/cuda.txt regardless of VLLM_TARGET_DEVICE.', adr: '[install-vllm-tt.sh](../docs/install-vllm-tt.sh)' },
  { axis: 'Execution modes', decision: 'Three live modes: single-process non-DP, single-process lane-DP, and standard multi-process DP with one device mesh per rank. launcher.py is retained MPI/tt-run and is not hooked by vLLM.', adr: '[SCHEDULING.md](../docs/SCHEDULING.md)' },
  { axis: 'Patch placement', decision: 'Runtime monkeypatches live in platform.py next to the seven _install_* helpers at lines 625, 685, 731, 798, 864, 885, 945, never in entrypoints.py.', adr: '[platform.py:625](../src/vllm_tt_plugin/platform.py)' },
  { axis: 'Speculative decoding', decision: 'The contract types and the host accept walk landed before the runtime wiring. The modules are unreachable at runtime until the drafter work merges; the config-time refusal is what keeps that honest.', adr: 'issue #110' },
  { axis: 'Unsupported features', decision: 'Fail loudly at config time with the offending value and the CLI flag, rather than degrade silently. A refusal is the shipped behaviour for every upstream feature TT does not serve.', adr: '[feature_support.py](../src/vllm_tt_plugin/feature_support.py)' },
  { axis: 'Model Runner', decision: 'Model Runner V1 is pinned at config time. vLLM 0.29.0 made MRV2 the default and deprecated MRV1 for removal in v0.32, so the pin is a tracked removal clock, not a permanent choice.', adr: 'issue #50' },
];

export const GROUPS = [
  { id: 'entry', title: 'Entry and configuration' },
  { id: 'sched', title: 'Scheduling and batching' },
  { id: 'exec', title: 'Execution' },
  { id: 'par', title: 'Parallelism' },
  { id: 'opt', title: 'Optional features' },
  { id: 'up', title: 'Upstream coupling' },
  { id: 'v', title: 'Upstream vLLM 0.26.0' },
];

export const NODES = [
  { id: 'EP', code: 'EP', name: 'Entry points', short: 'ENTRY', group: 'entry', gx: 0, gy: 0, w: 2, d: 2, h: 40, kind: 'gate',
    one: 'The two hooks that make vLLM notice Tenstorrent at all.',
    what: 'When you start vLLM, these two functions are the first thing it runs. If they cannot load the Tenstorrent software, they step aside and vLLM behaves exactly as if the plugin were not installed.',
    how: 'Two entry points in <code>pyproject.toml:36-40</code>: <code>vllm.platform_plugins:tt</code> → <code>entrypoints:platform_plugin</code>, and <code>vllm.general_plugins:tt_model_registry</code> → <code>entrypoints:register</code>. Both gate on <code>import ttnn</code> in <code>src/vllm_tt_plugin/entrypoints.py</code>. <code>register()</code> calls <code>register_tt_models_from_plugin()</code> so models are known in every process, including worker subprocesses that start before the platform hook runs.',
    steps: [['Import guard', 'Decline to activate when ttnn is absent.'], ['Register models', 'register_tt_models_from_plugin() in every process.'], ['Claim platform', 'Return the TTPlatform class by qualified name.']],
    cond: ['Both hooks run in the API server and in every engine subprocess; is a failure there reported clearly enough to diagnose a partial install?'] },

  { id: 'PF', code: 'PF', name: 'TTPlatform', short: 'PLATFORM', group: 'entry', gx: 4, gy: 0, w: 3, d: 3, h: 78, kind: 'tall',
    one: 'The class vLLM asks about the hardware, and the one place every unsupported feature is refused.',
    what: 'Everything vLLM needs to know about Tenstorrent as a platform goes through this one class: the device name, how many devices are visible, which worker class to build, and whether a feature you asked for can actually run here.',
    how: 'class <code>TTPlatform(Platform)</code> at <code>src/vllm_tt_plugin/platform.py:1326</code>. <code>_enum = PlatformEnum.OOT</code> (line 1327), <code>device_name = device_type = "tt"</code> (1328-1329), <code>device_control_env_var = "TT_VISIBLE_DEVICES"</code> (1330), <code>simple_compile_backend = "eager"</code> (1351) because tt-metal\'s Triton is incompatible with inductor. <code>check_and_update_config</code> (1462) calls <code>_pin_v1_model_runner()</code> before anything reads <code>use_v2_model_runner</code>, then delegates to <code>_apply_check_and_update_config</code> (1560) which is the single gate site for every refusal.',
    steps: [['Pin MRV1', '_pin_v1_model_runner() before VllmConfig.__post_init__ reads use_v2_model_runner.'], ['Read capabilities', 'Resolve model_capabilities off the registered TT model class.'], ['Refuse', 'Call the feature_support validators in a fixed sequence.'], ['Resolve keys', 'Store lane count and output width into additional_config.']],
    cond: ['The MRV1 pin is a removal clock: vLLM 0.29.0 deprecated MRV1 for removal in v0.32. When does that pin get revisited?', 'Every refusal message must name the CLI flag the operator typed. Is there a test that proves each one does?'] },

  { id: 'CF', code: 'CF', name: 'Config namespace', short: 'CONFIG', group: 'entry', gx: 8, gy: 0, w: 2, d: 2, h: 26, kind: 'store',
    one: 'The bridge between config-time decisions and the worker process that needs them later.',
    what: 'Some settings are worked out once, in the API server, but needed later in a worker process on another machine. This module carries them across, and keeps operator input separate from values the plugin derived itself.',
    how: '<code>src/vllm_tt_plugin/config.py</code> (369 lines). Operator input arrives as <code>get_tt_config()</code>, reading the <code>"tt"</code> key of <code>VllmConfig.additional_config</code>. Platform-derived state is stored under a leading-underscore top-level key — <code>_tt_resolved_lane_count</code>, <code>_tt_output_tokens_per_step</code>, <code>_tt_adaptive_block_output</code> — via <code>store_tt_lane_count()</code> and read by <code>get_tt_data_parallel_size()</code> / <code>get_tt_output_tokens_per_step()</code>. <code>additional_config</code> is a declared config field, so it survives the copy and pickle into the worker subprocess.',
    steps: [['Read operator config', 'get_tt_config() returns the "tt" sub-dict.'], ['Derive', 'Platform resolves capabilities and calls a store_tt_* writer.'], ['Hand off', 'additional_config is pickled into the worker subprocess.'], ['Read back', 'A scheduler or worker calls the matching get_tt_* accessor.']],
    cond: ['AGENTS.md §8 requires any new TTPlatform class attribute to be added to _TT_PLATFORM_CONFIG_ATTRS in tests/conftest.py, or test state leaks. A new attribute here would trip that.'] },

  { id: 'MR', code: 'MR', name: 'Model registry', short: 'REGISTRY', group: 'entry', gx: 12, gy: 0, w: 3, d: 2, h: 34, kind: 'cards',
    one: 'Where a HuggingFace architecture name becomes a tt-metal generator that can actually run it.',
    what: 'vLLM knows model names, not Tenstorrent models. This is the translation layer: give it a model name from a config file, and it hands back the class that knows how to execute it on the device.',
    how: '<code>src/vllm_tt_plugin/model_registry.py</code> (36 lines) orchestrates <code>register_tt_models()</code> at <code>platform.py:1089</code>. Operator overrides come first and win: <code>TT_MODEL_CLASS_OVERRIDES</code>, parsed at <code>platform.py:580-623</code> from the environment variable of the same name. Then tt-metal\'s own generators, then <code>_iter_extra_model_bundles()</code> (<code>platform.py:995-1040</code>) which walks <code>EXTRA_MODELS_DIR</code> (env var, read at line 1006) and picks up any self-contained bundle folder carrying a <code>vllm_metadata.json</code> — so a distribution tool can add a model with no source edit. <code>check_and_update_config</code> also prepends <code>"TT"</code> to the architectures list, which is why <code>TTWorker</code> resolves the prefixed entry rather than the cached <code>architecture</code> property.',
    steps: [['Operator overrides', 'TT_MODEL_CLASS_OVERRIDES, most specific intent, registered first.'], ['tt-metal generators', 'models.tt_transformers.tt.generator_vllm and per-demo generators.'], ['Extra bundles', 'EXTRA_MODELS_DIR folders with a vllm_metadata.json.'], ['Resolve', 'ModelRegistry resolves the TT-prefixed architecture at load time.']],
    cond: ['Capabilities are declared by tt-metal model classes, not here. A plugin change that reads a new capability needs a matching tt-metal change in the same release.'] },

  { id: 'SC', code: 'SC', name: 'TTScheduler', short: 'SCHEDULER', group: 'sched', gx: 1, gy: 4.5, w: 3, d: 3, h: 46, kind: 'tall',
    one: 'Decides which requests run this step, under constraints vLLM\'s token budget never had to consider.',
    what: 'Upstream vLLM fills a token budget each step. TT cannot always fill it, because the device takes work in fixed shapes. This scheduler refuses to promise more than the hardware can do in one step, and keeps a record of what it gave up.',
    how: 'class <code>TTScheduler(AsyncScheduler)</code> at <code>src/vllm_tt_plugin/scheduler.py:170</code>. It inherits upstream\'s priority <code>SchedulingPolicy</code> handling and its <code>skipped_waiting</code> queue, and adds the TT admission rules on top — including the async-decode clean-up preservation added in <code>scheduler: preserve cleanup across prefill decode fallback (#135)</code>. Block-output models reserve a K-token placeholder block only on solo decode steps, and <code>_update_after_schedule</code> owns that predicate. <code>TTSchedulingMode</code> (line 76) is DEFAULT / DECODE_ONLY / PREFILL_ONLY. <code>_TT_TOKEN_TILE_SIZE</code> is imported from platform.py at line 359 to round prompt lengths up to a tile.',
    steps: [['Admit', 'Pop waiting requests while the step fits the device shape.'], ['Reserve', 'Block-output models reserve the K-token placeholder only on solo decodes.'], ['Emit', 'Build a SchedulerOutput in the same shape upstream expects.'], ['Account', 'get_tt_forced_reset_discard_counts() tracks forced resets.']],
    cond: ['Upstream reserves capacity for in-flight prefills (_inflight_prefill_reserved_blocks) and throttles on prefill_capacity_bound. Neither is implemented; open issue #133 tracks KV-pressure scheduling.', 'scheduler_reserve_full_isl is read upstream and never read here.'] },

  { id: 'IB', code: 'IB', name: 'InputBatch', short: 'INPUTBATCH', group: 'sched', gx: 5, gy: 4.5, w: 3, d: 2, h: 30, kind: 'slab',
    one: 'The flat, pre-baked view of the batch the device actually wants to see.',
    what: 'vLLM normally assembles this per step. TT builds it once and keeps it, because re-packing host tensors every step would cost more than the model forward does. Per-lane batches let several independent replicas share one process.',
    how: 'The plugin reimplements <code>InputBatch</code> (vLLM\'s <code>gpu_input_batch.InputBatch</code> is <em>not</em> subclassed, so upstream churn there is decoupled) at <code>src/vllm_tt_plugin/input_batch.py:200</code>, with <code>TTLaneInputBatch(InputBatch)</code> at line 726 for lane-DP. It still imports upstream\'s shared types: <code>CachedRequestState</code> from <code>vllm.v1.worker.gpu_input_batch</code> (line 23) and <code>MultiGroupBlockTable</code> from <code>vllm.v1.worker.block_table</code> (line 22) — those two are the live coupling. <code>SEED_NONE_SENTINEL</code> (line 42) distinguishes "no seed" from seed <code>-1</code>, which vLLM treats as equivalent. Both runtime guards here raise rather than assert, because <code>python -O</code> strips asserts: <code>build_cached_request_state</code> refuses a pooling request at lines 61-69, and <code>add_request</code> bounds the caller-supplied <code>req_index</code> at lines 315-322.',
    steps: [['Add request', 'Allocate a slot, copy prompt state via build_cached_request_state.'], ['Rebuild cache state', 'apply_cached_req_state_update refreshes CachedRequestState.'], ['Slice', 'slice_tt_sampling_params produces the per-step view.'], ['Gather', '_gather_multi_modal_inputs assembles image features.']],
    cond: ['build_cached_request_state / apply_cached_req_state_update are the extension seam for prompt_logprobs. Only the builder is tested directly; the updater is not.', { q: 'The two pooling asserts at lines 61-63 and 338 should become ValueError, or be made unreachable by a config-time runner_type refusal.', r: 'Both resolved differently, on 2026-09-27. The one in <code>add_request</code> at 338 was deleted rather than converted: it is unreachable, because the scheduler replaces an embeds-only prompt with placeholder ids, <code>build_cached_request_state</code> raises NotImplementedError for one first, and <code>CachedRequestState.__post_init__</code> refuses a state whose token ids and embeds are both None. The <code>req_index</code> bound next to it became an IndexError, because that one is a real invariant on a caller-supplied integer. <code>build_cached_request_state</code> already raised. Covered by tests/test_pooling_refusal.py, including under <code>python -O</code>.' }] },

  { id: 'WK', code: 'WK', name: 'TTWorker', short: 'WORKER', group: 'exec', gx: 1, gy: 9, w: 3, d: 3, h: 40, kind: 'box',
    one: 'The per-process handle on the mesh: opens devices, sizes the KV pool, loads the model.',
    what: 'One of these exists per engine process. It opens the device mesh, decides how much KV cache fits, loads the model, and then runs steps on request. It is also the object vLLM pokes with utility RPCs.',
    how: 'class <code>TTWorker(WorkerBase)</code> at <code>src/vllm_tt_plugin/worker.py:183</code>. <code>init_device()</code> (220) binds <code>TT_VISIBLE_DEVICES</code> <em>before</em> calling <code>check_and_update_config</code>, because tt-metal latches the visible set at first cluster construction and never re-reads the variable. <code>get_kv_cache_spec()</code> (283) prefers a <code>get_kv_cache_spec</code> classmethod on the model class for hybrid models and otherwise builds one homogeneous spec under the layer name <code>"foo"</code> (363). <code>determine_available_memory()</code> (401) does not profile: it returns a synthetic byte budget, <code>page_size * num_blocks * group_size</code>, computed by <code>_available_kv_cache_memory_bytes_for_num_blocks</code> (154) from <code>get_kv_cache_groups</code> and <code>get_uniform_page_size</code>, so the upstream planner in a different process reconstructs the same count. <code>initialize_from_config</code> (427) and <code>compile_or_warm_up_model</code> (446) follow. <code>update_max_model_len</code> (435) exists because WorkerBase has no such hook and the engine calls it by RPC. The four members upstream reaches but WorkerBase does not declare are also answered here, each by name: <code>get_model</code> (284), which fixes <code>get_model_inspection</code> and <code>apply_model</code> transitively, the LoRA quartet <code>add_lora</code>/<code>remove_lora</code>/<code>pin_lora</code>/<code>list_loras</code> (307-333), and <code>execute_dummy_batch</code> (335). The LoRA four and the dummy batch raise by name rather than staying silent, because a config-time refusal that something bypassed should still fail where the operator can see it.',
    steps: [['Bind devices', 'Bind TT_VISIBLE_DEVICES before any mesh construction.'], ['Open mesh', 'open_mesh_device() honours model_capabilities["fabric_config"].'], ['Size KV', 'get_num_available_blocks_tt() then _fit_block_output_max_model_len().'], ['Build runner', 'TTModelRunner is constructed here, once.']],
    cond: [{ q: 'get_model is absent, so get_model_inspection and apply_model raise NotImplementedError.', r: 'Fixed 2026-09-27. <code>get_model</code> at worker.py:284 answers it, which fixes both members that build on it transitively.' }, { q: 'The four LoRA methods and get_punica_wrapper are absent.', r: 'Added 2026-09-27. <code>get_punica_wrapper</code> (platform.py:1429) and the LoRA quartet (worker.py:307-333) each raise by name. They stay unreachable while validate_lora refuses --enable-lora at config time, but a bypassed refusal should fail where the operator can see it rather than as a bare NotImplementedError.' }, { q: 'get_cache_block_size_bytes is absent.', r: 'Left absent, decided 2026-09-27. Its docstring says "used in speculative decoding" but the only occurrence in the whole 0.26.0 tree is the definition itself, so an override would be dead code.' }] },

  { id: 'RN', code: 'RN', name: 'TTModelRunner', short: 'RUNNER', group: 'exec', gx: 5, gy: 9, w: 4, d: 3, h: 70, kind: 'tall',
    one: 'Turns one scheduler step into one device forward pass, and the result back into something vLLM understands.',
    what: 'This is the engine room. It takes the step the scheduler produced, builds the tensors the device expects, runs the model, reads back tokens, and assembles the output structure vLLM asked for.',
    how: 'class <code>TTModelRunner</code> at <code>src/vllm_tt_plugin/model_runner.py:177</code> — the largest file in the plugin at 2,753 lines. It owns <code>get_supported_tasks</code>, the prefill and decode paths, the sampling handoff (<code>Sampler</code> from <code>vllm.v1.sample.sampler</code>, line 33), and the output types. <code>_gather_multi_modal_inputs</code> (834-903) assembles image features and has no host coverage today. <code>prompt_logprobs_dict</code> is hard-coded to <code>dict.fromkeys(req_ids, None)</code> at lines 1950 and 2477-2489, which is why the platform refuses <code>prompt_logprobs</code> at <code>platform.py:2179-2180</code>. The runner also holds the speculative wiring that step 6.1 makes reachable.',
    steps: [['Step in', 'Take SchedulerOutput, resolve the step plan via get_tt_step_plan().'], ['Build tensors', 'Slice TTModelInput / TTSamplingParams for the device.'], ['Forward', 'Call the tt-metal generator on the mesh.'], ['Sample', 'On-device or host sampling per compat_sampling_required().'], ['Step out', 'Assemble ModelRunnerOutput, or defer it for async decode.']],
    cond: ['prompt_logprobs is refused and hard-coded to None. Implementing it needs a tt-metal generator that exposes prompt-position logits — a paired change.', '_gather_multi_modal_inputs has no host test, so the image-only contract is unproven.', 'This file is a growth risk: AGENTS.md §7 asks new branches here to move to a named neighbour.'] },

  { id: 'AD', code: 'AD', name: 'Async decode', short: 'ASYNCDEC', group: 'exec', gx: 10, gy: 9, w: 3, d: 2, h: 30, kind: 'cards',
    one: 'Overlaps the device readback with the next step, so the host is never the bottleneck.',
    what: 'Waiting for the device is dead time. This hands the token readback off as a future, lets the next step start, and makes sure the deferred result is read back exactly once no matter who asks first.',
    how: '<code>src/vllm_tt_plugin/async_decode.py</code> (945 lines). <code>DeferredDecodeOutput(AsyncModelRunnerOutput)</code> at line 85 runs the deferred device readback exactly once from whichever caller reaches it first; <code>AsyncTTModelRunnerOutput(DeferredDecodeOutput)</code> at 148 adds the non-blocking single-process submission and covers both plain single-process decode and lane-DP decode. <code>SEED_NONE_SENTINEL</code> and <code>get_tt_forced_reset_discard_counts()</code> are imported for per-step seed and reset accounting. Device logprobs are only available on multi-device setups, and only for the sampled token — tracked upstream as tt-metal#34077.',
    steps: [['Submit', 'Post the decode without blocking on the readback.'], ['Defer', 'Wrap the pending readback in DeferredDecodeOutput.'], ['Continue', 'Let the next step build while the device works.'], ['Resolve once', 'The first caller drains the deferred readback.']],
    cond: [] },

  { id: 'LC', code: 'LC', name: 'TTLaneCoordinator', short: 'LANES', group: 'par', gx: 1, gy: 13.5, w: 3, d: 3, h: 40, kind: 'tall',
    one: 'Several independent KV replicas in one process, batched together on a gathered device batch.',
    what: 'Lane-DP gives you data parallelism without several processes: one engine, several independent copies of the model and its KV cache, all driven from a single scheduler that merges their work into one device call.',
    how: 'class <code>TTLaneCoordinator(SchedulerInterface)</code> at <code>src/vllm_tt_plugin/lane_scheduler.py:242</code>. It owns one fully independent <code>TTScheduler</code> per lane and implements the whole abstract interface (<code>interface.py:35-252</code>). <code>get_tt_step_plan()</code> produces a <code>TTStepPlan</code> that the runner consumes. <code>get_kv_connector()</code> (815) returns <code>None</code> — which is byte-identical to the <code>SchedulerInterface</code> default at <code>interface.py:248-249</code>, so the override is redundant. Lane folding is a Galaxy generator-version check plus an identity test on <code>model_class.__name__ == "GptOssForCausalLM"</code>; AGENTS.md §6 records that as leftover identity gating not to be copied.',
    steps: [['Own lanes', 'One independent TTScheduler per lane.'], ['Merge', 'Fold lane step plans into a gathered batch.'], ['Execute', 'Runner runs the merged batch once.'], ['Split out', 'Per-lane outputs route back to each scheduler.']],
    cond: ['get_kv_connector is redundant with the base default (interface.py:248-249). Pre-existing; not removed here because it is an unrelated cleanup.'] },

  { id: 'DP', code: 'DP', name: 'DP discovery', short: 'DP-DISCOVERY', group: 'par', gx: 5, gy: 13.5, w: 3, d: 2, h: 24, kind: 'job',
    one: 'Works out which devices this rank owns before anyone builds a mesh.',
    what: 'In multi-process data parallelism every rank needs to agree on which devices are its own, without any of them guessing. This is the job that settles it, and it has a timeout because a rank that never arrives would otherwise hang the launch.',
    how: '<code>src/vllm_tt_plugin/utils/dp_discovery.py</code> (270 lines): <code>run_standard_dp_visible_device_group_discovery()</code>, <code>split_standard_dp_discovery_result()</code>, <code>parse_mesh_grid()</code> and <code>format_tt_visible_devices()</code>. Results are cached per <code>TT_VISIBLE_DEVICES</code> string in <code>TTPlatform._standard_dp_visible_device_groups</code> and <code>_standard_dp_mesh_grids</code> — both are in <code>_TT_PLATFORM_CONFIG_ATTRS</code> in <code>tests/conftest.py</code>, so they are process-global state that tests must reset. <code>_resolve_mesh_grid()</code> in worker.py (127) reads that cache. A discovery-join timeout was added with the variable-block-output DP work (#118).',
    steps: [['Gather', 'Each rank publishes the device set it can see.'], ['Join', 'Ranks agree on a partition; the join has a timeout.'], ['Split', 'split_standard_dp_discovery_result() hands each rank its group.'], ['Cache', 'Store the group string and its mesh grid for the process.']],
    cond: [{ q: 'execute_dummy_batch is the data-parallelism plugin surface, and TT ships standard DP. Whether TT\'s DP path can reach it is open.', r: 'Yes, reachable, and answered 2026-09-27. The chain is llm_engine.py:202 → :297-299 → core.py:2055 → core.py:921 → executor/abstract.py:250. It is reached by string dispatch, so a missing method was an AttributeError from the worker loop. <code>TTWorker.execute_dummy_batch</code> (worker.py:335) now exists and refuses loudly; doing a real dummy forward needs a validated device forward on the mesh, which is a paired tt-metal change.' }] },

  { id: 'BO', code: 'BO', name: 'Block-output gate', short: 'BLOCKOUT', group: 'par', gx: 10, gy: 13.5, w: 3, d: 3, h: 52, kind: 'gate',
    one: 'Admission rules for models that commit several tokens at once.',
    what: 'Most models emit one token per step. A few — diffusion-style models — emit a whole block, and only when they are the only request running. This is the checkpoint that decides who is allowed to have that speedup, and how many output tokens anyone may ask for.',
    how: 'The gate reads <code>output_tokens_per_step</code> from <code>model_capabilities</code> and relaxes five upstream rules when it is greater than 1: <code>max_num_seqs 1</code>, data parallelism, the distributed-executor backend allow-list (which gains <code>mp</code>), the async-scheduling refusal, and the block-output sampling mode (which then accepts <code>decode_only</code> as well as <code>all</code>). The optional <code>tt_adaptive_block_output</code> capability commits the block only on solo decode steps and falls back to a plain width-1 baseline whenever two or more requests batch — never worse, just not faster. <code>tt_adaptive_block_max_prompt_tokens</code> (default 0 = no limit) serves over-long prompts as baseline for their whole lifetime. Canvas arithmetic is <code>_resolve_block_output_max_tokens()</code> (2121); <code>_fit_block_output_max_model_len()</code> shrinks <code>--max-model-len -1</code> to what the KV pool can hold. The separate pause guard at <code>platform.py:945-992</code> refuses <code>pause_generation(mode="keep", clear_cache=True)</code> with live requests — that is unrelated to sleep mode.',
    steps: [['Read capability', 'output_tokens_per_step and the adaptive flags off model_capabilities.'], ['Relax or hold', 'Block width > 1 relaxes five gates; width 1 holds them.'], ['Reserve', 'Only solo decode steps get the K-token placeholder block.'], ['Fit', 'max_model_len is fitted down to the KV pool capacity.']],
    cond: ['A present capability value that contradicts another resolved property must raise — e.g. block output plus prefix caching, or DiffusionGemma without output_tokens_per_step > 1.'] },

  { id: 'SO', code: 'SO', name: 'Structured output', short: 'STRUCTURED', group: 'opt', gx: 1, gy: 18, w: 2, d: 2, h: 22, kind: 'cards',
    one: 'Turns a JSON schema into a bitmask that makes invalid tokens impossible to sample.',
    what: 'Ask for JSON and get JSON, every time. The schema is compiled into a mask of which tokens are currently legal, and the sampler never considers the rest.',
    how: '<code>src/vllm_tt_plugin/structured_output.py</code> (100 lines) — <code>has_structured_outputs()</code> and <code>reorder_grammar_bitmask_for_tt_batch()</code>. The reorder exists because the device batches differently from the order the scheduler emitted grammars in. <code>_apply_check_and_update_config</code> forces <code>structured_outputs_config.disable_any_whitespace = True</code> (platform.py:1592) for every model, not just block ones: xgrammar and guidance allow arbitrary inter-field whitespace, and under greedy decoding the model can pick a whitespace token as the argmax indefinitely, exhausting the token budget before emitting a property name and returning truncated, unparseable JSON. Masking whitespace makes that loop structurally impossible. The backend stays "auto" so schemas xgrammar cannot compile still fall back to guidance.',
    steps: [['Compile', 'Schema becomes a grammar per request.'], ['Collect', 'Scheduler emits GrammarOutput for scheduled requests.'], ['Reorder', 'Reindex the bitmask rows into device batch order.'], ['Mask', 'Illegal tokens are excluded from sampling.']],
    cond: [] },

  { id: 'LP', code: 'LP', name: 'Logprobs', short: 'LOGPROBS', group: 'opt', gx: 4, gy: 18, w: 2, d: 2, h: 22, kind: 'cards',
    one: 'Returns the probabilities behind the tokens, from the device where they were computed.',
    what: 'Logprobs tell you how confident the model was. On Tenstorrent they are produced on the device during sampling, which is fast, but has limits on how many you can get at once.',
    how: '<code>src/vllm_tt_plugin/logprobs.py</code> (81 lines) — <code>build_logprobs_from_topk()</code> (line 10) and <code>build_device_logprobs()</code> (line 54), the only two packing paths; new logprob features must extend these rather than add a third. Upstream tensors <code>LogprobsTensors</code> and <code>LogprobsLists</code> are reused. The constraint is recorded in <code>TTPlatform.compat_sampling_required</code> (<code>platform.py:2233-2260</code>): device logprobs need a multi-device setup and return only the sampled token\'s logprob, so any <code>logprobs &gt; 1</code>, or any logprobs at all on a single device, forces host sampling. Tracked upstream as tt-metal#34077. The device computes top-32; the OpenAI API caps at 20, so <code>max_logprobs</code> is clamped at <code>platform.py:1574-1581</code> with a warning naming both values.',
    steps: [['Decide', 'compat_sampling_required() picks device or host sampling.'], ['Sample', 'Device returns the sampled token plus, where allowed, top-k.'], ['Pack', 'build_device_logprobs() or build_logprobs_from_topk().'], ['Clamp', 'max_logprobs is capped at 20 before the request is admitted.']],
    cond: ['prompt_logprobs is a different code path entirely: it is refused and hard-coded to None. It needs the prefill-position logits from tt-metal.'] },

  { id: 'SD', code: 'SD', name: 'Spec-decode contract', short: 'SPECDECODE', group: 'opt', gx: 7, gy: 18, w: 3, d: 2, h: 32, kind: 'slab',
    one: 'The types and the accept walk for speculative decoding — written, tested, and not yet reachable.',
    what: 'Speculative decoding guesses several tokens at once and checks them in a batch. The rules for which guesses are accepted already exist here as data types and as a host-side walk. What is missing is the runtime that drafts them.',
    how: '<code>src/vllm_tt_plugin/spec_decode.py</code> (385 lines) holds the contract types including <code>PLACEHOLDER_TOKEN_ID</code>; <code>spec_accept.py</code> (512 lines) implements the accept walk and imports <code>apply_top_k_top_p_pytorch</code> from <code>vllm.v1.sample.ops.topk_topp_sampler</code> at line 312 to apply the model\'s own sampling distribution to the candidates. <strong>Neither module is imported by any runtime module</strong> — only by each other and by <code>tests/spec/</code> — and <code>feature_support.validate_speculative_decoding</code> (line 53) still refuses <code>speculative_config</code>. The contract landed before the runtime on purpose: see the decisions table. The runtime half exists as open PRs #125, #127, #130, #131 and #142, with #131 the model-drafter branch that #142 builds on.',
    steps: [['Contract', 'spec_decode.py declares the types and the placeholder token.'], ['Draft', 'Not yet wired: the model-drafter branch is #131.'], ['Accept walk', 'spec_accept.py checks candidates against the target distribution.'], ['Reach it', 'A host test asserts these modules are in the runner\'s import graph.']],
    cond: ['On main today the speculative contract is unreachable at runtime. That is the exact condition tests/test_spec_runtime_wiring.py must fail on.', 'The drafter branch, #142, and the int64-seed fix #141 all predate the current branch and must have this work\'s changes re-applied on top.'] },

  { id: 'UC', code: 'UC', name: 'Upstream coupling', short: 'COUPLING', group: 'up', gx: 11, gy: 18, w: 3, d: 3, h: 36, kind: 'slab',
    one: 'Every place the plugin binds to vLLM internals, and therefore every place a vLLM bump can break it.',
    what: 'vLLM does not promise stability on its internals, and this plugin depends on a lot of them. This is the honest inventory of the bindings, so a version bump is a checklist rather than a surprise.',
    how: 'Derived with <code>grep -rnE "^\\s*(from|import)\\s+vllm" src/</code> per the repo\'s own playbook (<code>.github/prompts/vllm-upgrade-assessment.prompt.md:53-115</code>). Subclassed bases: <code>TTPlatform(Platform)</code>, <code>TTWorker(WorkerBase)</code>, <code>TTScheduler(AsyncScheduler)</code>, <code>TTLaneCoordinator(SchedulerInterface)</code>, <code>TTModelLoader(BaseModelLoader)</code>, <code>DeferredDecodeOutput</code>/<code>AsyncTTModelRunnerOutput(AsyncModelRunnerOutput)</code>, <code>TTLaunchPlan(EngineLaunchPlan)</code>, <code>TTCoreEngineLauncher(CoreEngineLauncher)</code>. Constructed upstream dataclasses: <code>ModelRunnerOutput</code>, <code>SchedulerOutput</code>, <code>SamplingMetadata</code>, <code>CachedRequestState</code>, <code>MultiGroupBlockTable</code> and the KV cache specs. The <code>model_capabilities</code> keys the plugin actually reads go beyond the AGENTS.md §6 table: that table lists ten, and <code>tt_block_kv_extent_tokens</code>, <code>supports_device_penalties</code> and <code>fabric_config</code> are consumed in code but absent from it. The per-member matrix for every coupled symbol — plugin site, base behaviour, and whether it is called unconditionally — is not committed here; it is re-derived per version bump using the playbook above, which is the point of the playbook.',
    steps: [['Grep imports', 'The four playbook categories, re-derived every time.'], ['Diff the tag', 'BASELINE (the 0.26.0 pin) against the target tag.'], ['Classify', 'BREAKING-IMPORT / -CONSTRUCT / -OVERRIDE / (silent) / BEHAVIORAL / NONE.'], ['Plan', 'One required edit per coupled symbol, with a file:line.']],
    cond: ['AGENTS.md §6 lists ten capability keys but three more are consumed in code. The table is incomplete.', 'launcher.py runs its ImportError fallback: upstream CoreEngineLauncher / EngineLaunchPlan are absent at 0.26.0, so it is dormant and unhooked.'] },

  { id: 'FS', code: 'FS', name: 'Feature support', short: 'FEATURESUP', group: 'up', gx: 15, gy: 13.5, w: 3, d: 3, h: 44, kind: 'gate',
    one: 'One loud refusal per upstream feature the TT backend does not serve.',
    what: 'Ask for something Tenstorrent cannot do and you get a clear error naming the flag you typed, at startup, instead of a wrong answer later or an AttributeError from deep inside vLLM.',
    how: '<code>src/vllm_tt_plugin/feature_support.py</code>, one <code>validate_&lt;feature&gt;(cls, vllm_config)</code> per feature, called in a fixed sequence from <code>_apply_check_and_update_config</code> (<code>platform.py:1560</code>) at the point the three <code>assert</code>s used to sit (1561-1568). It is a separate module because <code>platform.py</code> is already 2,260 lines and AGENTS.md §7 says a new conditional branch there is a prompt to split into a neighbour; <code>config.py</code> is the accessor layer, not a validator, so a purpose-named module is the smaller change. Every function raises <code>ValueError</code> printing the offending value and naming the CLI flag — never <code>assert</code>, which <code>python -O</code> removes (AGENTS.md §2.4).',
    steps: [['Read config', 'Each validator reads only the fields upstream actually defines.'], ['Compare', 'Against the TT-supported set for that feature.'], ['Raise', 'ValueError naming the offending value and the flag.'], ['No override', 'Features the base class already refuses correctly get no validator at all.']],
    cond: ['A validator must never read a field that does not exist on the pinned vLLM: three rows in the original plan did exactly that and would have raised AttributeError instead of the intended clean refusal.', 'request_validation refusals for n > 1, thinking_token_budget, repetition_detection and extra_args reuse the existing block-model message builder at platform.py:2203-2231 rather than writing a second one.'] },

  // ---- Upstream vLLM 0.26.0: the structures the plugin substitutes for. ----
  { id: 'vPlat', code: 'vP', name: 'Platform resolution', short: 'V-PLATFORM', group: 'v', gx: 0, gy: 22.5, w: 3, d: 3, h: 40, kind: 'gate',
    one: 'How upstream decides which platform it is running on.',
    what: 'Five built-in hardware backends compete with any out-of-tree plugin. Exactly one wins, and the loser is told so immediately rather than at the first feature that needs it.',
    how: '`vllm/platforms/__init__.py` at tag v0.26.0. `resolve_current_platform_cls_qualname()` walks the builtins {tpu, cuda, rocm, xpu, cpu} plus every `vllm.platform_plugins` callable, each inside a bare `try/except Exception: pass`. Two or more out-of-tree plugins raises `RuntimeError`; exactly one OOT wins outright over any builtin; zero matches yields `UnspecifiedPlatform` (`interface.py`), which sets `_enum` and `device_type = ""` and omits `device_name` entirely — so the inherited bare annotation has no value to read. `current_platform` is resolved through a module `__getattr__` so an OOT plugin can import `Platform` without re-entering resolution. `TTPlatform` wins by being the only OOT plugin, and `entrypoints.platform_plugin()` returns `None` when `import ttnn` fails so nothing is claimed. **Substituted by** `TTPlatform` (`platform.py:1326`), which is claimed through the `tt` platform-plugin entry point and declines by returning `None` when `import ttnn` fails.',
    steps: [['List plugins', 'Entry points from both groups, filtered by VLLM_PLUGINS.'], ['Run detectors', 'Each callable in try/except; a non-None return claims the platform.'], ['Prefer OOT', 'One OOT beats any builtin; two OOT is a RuntimeError.'], ['Resolve lazily', 'current_platform resolved on first access, not at import.']],
    cond: [] },

  { id: 'vInput', code: 'vI', name: 'InputProcessor', short: 'V-INPUT', group: 'v', gx: 4, gy: 22.5, w: 3, d: 2, h: 34, kind: 'screen',
    one: 'Turns a client request into validated, per-request state — and is where platform validation runs.',
    what: 'The front desk. It takes the prompt, the sampling parameters and the platform, and it is the last point where a request can be refused before it becomes work for the device.',
    how: '`vllm/v1/engine/input_processor.py` at v0.26.0. It calls `current_platform.validate_request(processed_inputs, params)` at line 296 — unconditionally, once per request, with the **parent** parameters. That ordering is why `n > 1` is not a gap on TT: this validator sees the parent `n`, and the fan-out to `n == 1` children happens later and downstream. **Consumed by** `TTPlatform.validate_request` (`platform.py:2193`), which sees the parent params, including `n`, before the fan-out below.',
    steps: [['Build inputs', 'Tokenise, apply the chat template, gather multimodal features.'], ['Validate', 'current_platform.validate_request — per request, on the parent params.'], ['Block hashes', 'Compute prefix hashes when prefix caching is on.']],
    cond: ['Platform validation runs here, before the request becomes a Request, so a refusal costs nothing.'] },

  { id: 'vSamp', code: 'vS', name: 'Sampler and parallel sampling', short: 'V-SAMPLER', group: 'v', gx: 8, gy: 22.5, w: 3, d: 2, h: 30, kind: 'cards',
    one: 'Upstream turns one request with n>1 into n independent single-sample children.',
    what: 'Multi-completion is bookkeeping, not a sampling mode. The parent request is expanded into separate child requests, each carrying its own parameters, and the outputs are re-joined under one id.',
    how: '`vllm/v1/engine/parallel_sampling.py` (150 lines) and `vllm/v1/sample/sampler.py` (436 lines). `child_sampling_params.n = 1` at `parallel_sampling.py:74`; `get_child_info(index)` returns the child id and those params at `:83`; `AsyncLLM` expands at `async_llm.py:388`; `OutputProcessor` re-joins at `output_processor.py:326`. The consequence for a plugin is that the model runner only ever sees `n == 1`, whatever the client asked for. **Reached by TT only through the child-request shape.** Because the expansion happens here, the model runner never sees `n != 1`.',
    steps: [['Expand', 'ParentRequest.get_child_info per index, each child n = 1.'], ['Schedule', 'Children are independent requests to the scheduler.'], ['Re-join', 'OutputProcessor aggregates children under the parent external id.']],
    cond: ['This is why refusing n > 1 in a platform validator would break a request that already works.'] },

  { id: 'vEngine', code: 'vE', name: 'EngineCore', short: 'V-ENGINECORE', group: 'v', gx: 0, gy: 27, w: 3, d: 3, h: 58, kind: 'tall',
    one: 'The engine loop: take requests, step the scheduler, run the model, hand back outputs.',
    what: 'The process that owns the model. It polls for work, asks the scheduler what to run this step, sends that to the workers, and puts the results on the output queue.',
    how: '`vllm/v1/engine/core.py` at v0.26.0, 2407 lines. `VllmConfig.__post_init__` calls the platform hook here, so configuration is settled before the loop starts. `run_busy_loop` is defined twice: `EngineCoreProc.run_busy_loop` at line 1358 for the single-process case, and `DPEngineCoreProc.run_busy_loop` at line 2024 for data parallel. Only the DP one fires `execute_dummy_batch` when a step executed nothing (lines 2049-2055) — the call chain that makes a TT DP rank that idles while a peer runs reach that RPC. `TTWorker.execute_dummy_batch` (worker.py:335) now answers it by name, because `Executor.collective_rpc` dispatches by string and a missing method was an `AttributeError` from the worker loop. It refuses loudly rather than doing a dummy forward: the TT equivalent needs a validated device forward on the mesh, which is a paired tt-metal change. **Substituted only where the loop is DP-specific.**',
    steps: [['Poll', '_process_input_queue takes new and aborted requests.'], ['Step', 'scheduler.schedule() then the worker executor.'], ['Publish', 'Outputs go to the output queue; stats are drained.']],
    cond: ['run_busy_loop is the DP-only loop; the single-process path never reaches execute_dummy_batch.'] },

  { id: 'vSched', code: 'vS', name: 'Upstream scheduler', short: 'V-SCHEDULER', group: 'v', gx: 4, gy: 27, w: 3, d: 3, h: 46, kind: 'tall',
    one: 'Fills a token budget each step, with no notion of a prefill or decode phase.',
    what: 'Upstream asks one question every step: how many tokens can I afford? Whatever fits, fits — prefill and decode are just tokens. That simplicity is what the TT scheduler has to give up.',
    how: '`vllm/v1/core/sched/scheduler.py` at v0.26.0, 2823 lines. `schedule()` opens with `token_budget = self.max_num_scheduled_tokens`, spends it in a running pass, then admits waiting requests — under woosuk\'s explicit NOTE that "There\'s no \'decoding phase\' nor \'prefill phase\' in the scheduler." `TTScheduler` inverts this: it classifies the step first and hides the ineligible half of the state from the base class, which is why `throttle_prefills` is accepted and ignored on TT and the DP prefill-balancing cadence is unreachable there. **Substituted by** `TTScheduler` (`scheduler.py:170`): it classifies the step first, then hides the ineligible half of the state from this base class.',
    steps: [['Budget', 'One scalar for the whole step.'], ['Running pass', 'Decode and prefill chunks already resident, decrementing the budget.'], ['Waiting pass', 'Admit new work while tokens remain and KV is available.'], ['Preempt', 'PRIORITY: max(running, key=(priority, arrival_time)) at scheduler.py:576-581. FCFS: running.pop() at 601.']],
    cond: ['TT gives up the single-budget model because the device can only take work in fixed shapes.'] },

  { id: 'vKV', code: 'vK', name: 'KV cache manager', short: 'V-KVCACHE', group: 'v', gx: 8, gy: 27, w: 3, d: 3, h: 38, kind: 'store',
    one: 'Owns the block pool, the prefix cache, and the decision about what fits.',
    what: 'The memory bookkeeper. It hands out blocks, remembers which blocks hold which prompt prefixes, and is the reason a second request with the same prompt is cheap.',
    how: '`vllm/v1/core/kv_cache_manager.py` at v0.26.0, 760 lines. `get_computed_blocks(request)` (line 207) is what the scheduler consults before prefilling; `allocate_slots` (283) hands out the rest; `cache_blocks` (689) records the full blocks of a finished request. The pool itself is built by the coordinator, not here: `BlockPool(num_gpu_blocks=kv_cache_config.num_blocks, ...)` at `kv_cache_coordinator.py:90-97`. So the pool is sized by whatever the engine KV planner derived from the `determine_available_memory` result — which is why a synthetic byte budget returned by the TT worker is enough to control the count from another process. **Substituted by** `TTWorker.determine_available_memory` (`worker.py:401`), which returns a synthetic byte budget instead of profiling.',
    steps: [['Allocate', 'get_computed_blocks for the prefix hit, then allocate_slots for the rest.'], ['Hash', 'Block hashes come from cache_config.prefix_caching_hash_algo.'], ['Cache on free', 'A finished request\'s full blocks are recorded, not discarded.']],
    cond: ['The TT worker does not profile: it returns a synthetic byte budget so the planner in another process reconstructs the same block count.'] },

  { id: 'vExec', code: 'vX', name: 'Executor and collective RPC', short: 'V-EXECUTOR', group: 'v', gx: 0, gy: 31.5, w: 3, d: 3, h: 36, kind: 'job',
    one: 'Reaches the worker by method NAME, which is why a missing override is an AttributeError.',
    what: 'The layer between the engine and the worker process. It does not import the worker class; it looks methods up by string and calls whatever it finds. That is the whole reason an unimplemented optional feature fails so unhelpfully.',
    how: '`vllm/v1/executor/abstract.py` at v0.26.0. Every optional worker feature is dispatched by name: `initialize_from_config`, `compile_or_warm_up_model`, `determine_available_memory`, `get_kv_cache_spec`, `execute_dummy_batch`, `take_draft_token_ids`, `sleep`, `wake_up`. None of those names is declared on `WorkerBase`; the lookup lands on `WorkerWrapperBase.__getattr__` (`worker_base.py:333`). So a plugin that omits one gets `AttributeError` from inside a worker RPC loop, and only when the feature is actually used. **Kept as-is.** The executor is upstream code; a plugin only has to answer the names it wants, which is why a missing one is an AttributeError rather than a clean refusal.',
    steps: [['Select', 'Worker class resolved from parallel_config.worker_cls.'], ['RPC', 'collective_rpc("method") dispatches by string to each worker.'], ['Fold', 'Results reduce into compilation_config, KV counts and stats.']],
    cond: ['This string dispatch is why an unimplemented optional feature must fail loudly at config time, not be left to surface as an AttributeError.'] },

  { id: 'vWorker', code: 'vW', name: 'Reference Worker', short: 'V-WORKER', group: 'v', gx: 4, gy: 31.5, w: 3, d: 3, h: 34, kind: 'box',
    one: 'What a complete WorkerBase implementation looks like, and the bar TTWorker is measured against.',
    what: 'The GPU implementation. It shows which optional methods exist in practice and what they do — LoRA, sleep, dummy batches — so you can see which ones a plugin is declining rather than missing.',
    how: '`vllm/v1/worker/gpu_worker.py` at v0.26.0, 1453 lines. Implements `get_model` (929), the four LoRA methods (add_lora 1237, remove_lora 1240, list_loras 1243, pin_lora 1246, one-line delegations to the model runner), `execute_dummy_batch` (1233, sized by `model_runner.uniform_decode_query_len`), `sleep(level)` (192) and `wake_up(tags)` (227) — the last three absent from `WorkerBase` entirely. `TTWorker` mirrors the first group and omits only `sleep` and `wake_up`, which cannot be reached on TT because `ModelConfig.__post_init__` already refuses `--enable-sleep-mode` for any non-CUDA/ROCm/XPU platform; the LoRA quartet and `execute_dummy_batch` are present and refuse by name. **Substituted by** `TTWorker` (`worker.py:183`): a mesh device instead of a GPU, a tt-metal generator instead of vLLM layers, and no profiling run.',
    steps: [['init_device', 'Set the device up, then run the platform hook again per subprocess.'], ['load_model', 'Weights onto the device; the runner holds the model.'], ['execute_model', 'One step; returns None to defer sampling.']],
    cond: ['TT init_device re-runs TTPlatform.check_and_update_config because multiprocessing means platform class state is per process.'] },

  { id: 'vRunner', code: 'vR', name: 'GPU model runner', short: 'V-RUNNER', group: 'v', gx: 9, gy: 31.5, w: 4, d: 3, h: 62, kind: 'tall',
    one: 'The upstream engine room: step in as tensors, step out as ModelRunnerOutput.',
    what: 'The counterpart TTModelRunner replaces. It builds the batch tensors, runs the model, samples, and assembles the output structure the engine expects.',
    how: '`vllm/v1/worker/gpu_model_runner.py` at v0.26.0, 7846 lines. Owns `get_supported_generation_tasks` and `get_supported_pooling_tasks` (3309, 3327) — the members that decide what a model can be asked to do, and where `runner_type` becomes visible. It is also the file whose churn the plugin insulates itself from by reimplementing `InputBatch` rather than subclassing `gpu_input_batch.InputBatch`. **Substituted by** `TTModelRunner` (`model_runner.py:177`): a tt-metal generator instead of vLLM model layers, over a persistent `InputBatch` rather than a per-step rebuild.',
    steps: [['Build', 'InputBatch and the persistent state from a SchedulerOutput.'], ['Forward', 'Attention, MLP, sampling.'], ['Emit', 'ModelRunnerOutput with sampled tokens and logprobs.']],
    cond: ['The plugin reimplements its own InputBatch, so churn in gpu_input_batch is decoupled; the shared types it does import are not.'] },
];

export const FLOWS = [
  { id: 'lifecycle', name: 'Lifecycle', hops: [
    ['EP', 'MR', 'activate', { group: 'vllm.general_plugins', name: 'tt_model_registry' }, 'yx'],
    ['MR', 'PF', 'platform claimed', { cls: 'vllm_tt_plugin.platform.TTPlatform' }, 'xy'],
    ['PF', 'CF', 'check_and_update_config', { keys_written: ['_tt_output_tokens_per_step', '_tt_resolved_lane_count'] }, 'xy'],
    ['CF', 'SC', 'scheduler built', { mode: 'DEFAULT', max_num_seqs: 1 }, 'xy'],
    ['SC', 'IB', 'scheduled', { num_scheduled_tokens: { 'req-7': 128 } }, 'xy'],
    ['IB', 'RN', 'step input', { batch_size: 1, block_size: 64 }, 'xy'],
    ['RN', 'RN', 'forward + sample', { mesh: '1,1', num_tokens_out: 1 }, 'xy'],
    ['RN', 'SC', 'update_from_output', { finished: [], token_ids: [4821] }, 'yx'],
  ] },
  { id: 'config', name: 'Config', hops: [
    ['PF', 'MR', 'resolve class', { arch: 'TTGptOssForCausalLM', source: 'tt-metal generator' }, 'yx'],
    ['MR', 'PF', 'capabilities', { output_tokens_per_step: 1, supports_prefix_caching: true }, 'yx'],
    ['PF', 'FS', 'refuse unserved', { flag: '--kv-cache-dtype', value: 'fp8' }, 'xy'],
    ['PF', 'BO', 'block gate', { output_tokens_per_step: 1, block_contract: null }, 'xy'],
    ['BO', 'CF', 'fit max_model_len', { max_model_len: 8192, kv_blocks: 4096 }, 'xy'],
    ['CF', 'WK', 'config pickled', { additional_config: { _tt_output_tokens_per_step: 1 } }, 'yx'],
  ] },
  { id: 'lanedp', name: 'Lane-DP fold', hops: [
    ['SC', 'LC', 'lane plan', { lanes: 2, per_lane_waiting: [3, 1] }, 'xy'],
    ['LC', 'IB', 'per-lane batch', { lane_0: 3, lane_1: 1 }, 'xy'],
    ['IB', 'RN', 'gathered batch', { total_seqs: 4, gathered: true }, 'xy'],
    ['RN', 'LC', 'split outputs', { lane_0: 3, lane_1: 1 }, 'yx'],
    ['LC', 'SC', 'merged schedule', { num_scheduled_tokens: 512 }, 'yx'],
  ] },
  { id: 'stdp', name: 'Standard-DP discovery', hops: [
    ['DP', 'DP', 'publish visible set', { TT_VISIBLE_DEVICES: '0,1,2,3', ranks: 4 }, 'xy'],
    ['DP', 'DP', 'join with timeout', { groups: ['0', '1', '2,3'], timeout_s: 30 }, 'xy'],
    ['DP', 'WK', 'split result', { this_rank_group: '2,3', mesh_grid: [1, 2] }, 'yx'],
    ['WK', 'DP', 'cache group + grid', { group: '2,3', grid: [1, 2] }, 'yx'],
  ] },
  { id: 'blockout', name: 'Block output', hops: [
    ['PF', 'BO', 'capability read', { output_tokens_per_step: 4, adaptive: true }, 'yx'],
    ['BO', 'FS', 'relax five gates', { max_num_seqs: 1, dp: false, backend: 'mp', async: false }, 'xy'],
    ['BO', 'SC', 'reserve placeholder', { batch_size: 1, k_tokens: 4 }, 'xy'],
    ['SC', 'BO', 'fallback decision', { batch_size: 3, served_as: 'baseline width-1' }, 'yx'],
    ['BO', 'RN', 'canvas arithmetic', { prompt_len: 1300, output_tokens: 8, tile: 32 }, 'yx'],
  ] },
  { id: 'grammar', name: 'Optional', hops: [
    ['SC', 'SO', 'GrammarOutput', { req_ids: ['req-7'], masks: 1 }, 'xy'],
    ['SO', 'IB', 'reorder for device batch', { from: [0], to: [0] }, 'xy'],
    ['IB', 'RN', 'bitmask', { illegal_masked: true, disable_any_whitespace: true }, 'xy'],
    ['RN', 'SC', 'valid token only', { token_id: 1274, text: '{"' }, 'yx'],
    ['RN', 'AD', 'deferred readback', { pending: true, token_id: 4821 }, 'xy'],
    ['SC', 'SD', 'draft request', { num_draft: 4, lookahead: 4 }, 'xy'],
    ['SD', 'SC', 'accept walk', { accepted: 3, rejected_at: 3 }, 'yx'],
  ] },
  { id: 'coupling', name: 'TT vs upstream', hops: [
    ['UC', 'PF', 'Platform members', { overridden: 14, refused: 11 }, 'xy'],
    ['UC', 'WK', 'WorkerBase members', { overridden: 9, missing_loud: 6 }, 'xy'],
    ['UC', 'SC', 'scheduler overrides', { subclass: 'AsyncScheduler' }, 'xy'],
    ['UC', 'RN', 'runner types', { subclass: 'AsyncModelRunnerOutput' }, 'xy'],
    ['UC', 'FS', 'refusal surface', { config_time: 11, request_time: 4 }, 'yx'],
    ['SC', 'vSched', 'replaces', { change: 'classify the step, then hide the ineligible half' }, 'yx'],
    ['WK', 'vWorker', 'replaces', { change: 'mesh device, no profiling' }, 'yx'],
    ['RN', 'vRunner', 'replaces', { change: 'tt-metal generator, persistent batch' }, 'yx'],
  ] },

];



export const CH = [
  { id: 'entry', title: 'Entry and configuration', reveal: ['EP', 'PF'],
    lede: `Two hooks make vLLM notice Tenstorrent; one class answers every question about the hardware.`,
    story: `<p>Nothing in the plugin is reachable until <code>pyproject.toml:36-40</code> is read and <code>ttnn</code> imports. After that every question vLLM asks about the hardware funnels into <mark>one class</mark>, <code>TTPlatform</code>. It decides the device name, the visible-device variable, the worker class, and — critically — it is the single place an unsupported feature is refused.</p>`,
    flow: [['EP', 'PF', 'platform claimed', { cls: 'TTPlatform' }], ['PF', 'EP', 'config accepted', { worker_cls: 'TTWorker' }]] },

  { id: 'config', title: 'Configuration and model registration', reveal: ['CF', 'MR'],
    lede: `Settings are worked out once and carried across process boundaries; models are translated from names to generators.`,
    story: `<p>Two structures that both exist because the work is split across processes. <code>config.py</code> carries derived state through <code>additional_config</code> into the worker, keeping operator input and plugin-derived values in separate namespaces. <code>model_registry.py</code> turns a HuggingFace architecture name into <mark>the tt-metal generator that can run it</mark>, with three layers of specificity: operator override, tt-metal's own generators, then drop-in bundles under <code>EXTRA_MODELS_DIR</code>.</p>`,
    flow: [['MR', 'CF', 'capabilities', { output_tokens_per_step: 1 }], ['CF', 'MR', 'hashed config', { keys: 7 }]] },

  { id: 'sched', title: 'Scheduling and batching', reveal: ['SC', 'IB'],
    lede: `Upstream fills a token budget; TT can only fill what the device can actually take in one step.`,
    story: `<p><code>TTScheduler</code> inherits upstream\'s priority policy and skipped-waiting queue and adds the admission rules the hardware forces. <code>InputBatch</code> is then the flat, pre-baked view of that batch — <mark>rebuilt once and kept</mark>, because re-packing host tensors every step would cost more than the model forward does. It still borrows two upstream types, <code>CachedRequestState</code> and <code>MultiGroupBlockTable</code>, and those are the live coupling points.</p>`,
    flow: [['SC', 'IB', 'scheduled', { reqs: 2 }], ['IB', 'SC', 'batch ready', { cached: true }]] },

  { id: 'exec', title: 'Worker and runner execution', reveal: ['WK', 'RN', 'AD'],
    lede: `The engine room: open the mesh, turn one step into one device forward pass, and never wait on the device.`,
    story: `<p><code>TTWorker</code> binds <code>TT_VISIBLE_DEVICES</code> before anything builds a mesh — tt-metal latches the visible set once and never re-reads it — then sizes the KV pool and hands off to the runner. <code>TTModelRunner</code> is the largest file in the plugin at 2,753 lines and does the forward pass, the sampling handoff, and the output assembly. <code>async_decode.py</code> is what keeps the host off the critical path: the readback is deferred and <mark>drained exactly once</mark>, whichever caller gets there first.</p>`,
    flow: [['WK', 'RN', 'load + warm up', { kv_blocks: 4096 }], ['RN', 'AD', 'deferred readback', { pending: 1 }], ['AD', 'RN', 'drained', { token_id: 4821 }]] },

  { id: 'par', title: 'Parallelism: lane-DP, standard DP, and block output', reveal: ['LC', 'DP', 'BO'],
    lede: `Three different ways to use more than one device, each with its own admission rules.`,
    story: `<p>Lane-DP runs several independent KV replicas in <em>one</em> process and merges their work into a gathered batch. Standard DP runs one mesh per rank and has to agree on device ownership first, which is what the discovery job does. Block output is not parallelism but behaves like a gate: only models that commit several tokens at once are affected, and <mark>only on solo decode steps</mark> does the speedup apply. See <code>docs/SCHEDULING.md</code> for the full model.</p>`,
    flow: [['DP', 'LC', 'lanes resolved', { lanes: 2 }], ['LC', 'BO', 'admission', { output_tokens_per_step: 1 }], ['BO', 'DP', 'baseline fallback', { batch_size: 3 }]] },

  { id: 'opt', title: 'Optional features', reveal: ['SO', 'LP', 'SD'],
    lede: `Structured output, logprobs, and speculative decoding: two shipped, one not yet reachable.`,
    story: `<p>Structured output forces <code>disable_any_whitespace</code> for every model, not just block ones — under greedy decoding the model can pick a whitespace token as the argmax indefinitely and return truncated, unparseable JSON. Logprobs come off the device where they were computed, with a documented ceiling. Speculative decoding is the odd one: <mark>the contract and the accept walk are written and tested</mark>, but nothing imports them at runtime and the config still refuses <code>speculative_config</code>. The drafter runtime is open PR #131.</p>`,
    flow: [['SO', 'LP', 'masked sample', { token_id: 1274 }], ['LP', 'SD', 'draft requested', { num_draft: 4 }], ['SD', 'SO', 'unreachable today', { refused: true }]] },

  { id: 'up', title: 'Upstream coupling and the refusal surface', reveal: ['UC', 'FS'],
    lede: `Where the plugin binds to vLLM internals, and how it handles what it does not serve.`,
    story: `<p>The plugin binds to a lot of vLLM internals, so a version bump is a checklist rather than a surprise — <code>.github/prompts/vllm-upgrade-assessment.prompt.md</code> is the method, and every gap it finds is recorded per member in the questions on this map. On the other side, <code>feature_support.py</code> turns every gap into a loud refusal at config time, <mark>naming the flag the operator actually typed</mark>. Features the base class already refuses correctly get no validator at all: adding one would be a second message for a problem upstream already reports well.</p>`,
    flow: [['UC', 'FS', 'gap found', { member: 'cache_dtype' }], ['FS', 'UC', 'refused', { flag: '--kv-cache-dtype', value: 'fp8' }]] },

  { id: 'upengine', title: 'The upstream v1 engine', reveal: ['vPlat', 'vInput', 'vSamp'],
    lede: `The engine TT plugs into: who validates a request, who decides what runs, and who owns the memory.`,
    story: `<p>Upstream is not an empty scaffold — it is a working engine with a simple, strong idea: <mark>one token budget per step</mark>, filled in a running pass then a waiting pass, with no prefill/decode phase at all. TT inherits all of it and overrides the parts the hardware constrains. The two structures that decide almost everything downstream are the platform gate, which is unconditional and runs before a request becomes work, and the KV manager, which is the only reason a repeated prompt is cheap.</p>`,
    flow: [['vPlat', 'vInput', 'resolved platform', { device: 'tt' }], ['vInput', 'vSamp', 'validate + expand', { n: 4, children: 4 }]] },

  { id: 'upsched', title: 'The upstream scheduler and KV cache', reveal: ['vEngine', 'vSched', 'vKV'],
    lede: `Where the step is planned and where the blocks live — the two places TT had to change.`,
    story: `<p><code>Scheduler</code> spends one scalar per step, and the elegant part is that it never asks whether a token is a prefill or a decode. <code>TTScheduler</code> gives that up: the device takes work in fixed shapes, so the step is <em>classified</em> first and the ineligible half of the state is hidden from the base class. The price is that upstream's DP prefill-balancing throttle is unreachable on TT. <code>KVCacheManager</code> is upstream's block book; TT never profiles and instead returns a synthetic byte budget so the planner in a different process lands on the same count.</p>`,
    flow: [['vEngine', 'vSched', 'schedule', { token_budget: 8192 }], ['vSched', 'vKV', 'allocate', { blocks: 4 }], ['vKV', 'vEngine', 'cached prefix', { hit: 0 }]] },

  { id: 'upworker', title: 'The upstream worker and runner', reveal: ['vExec', 'vWorker', 'vRunner'],
    lede: `The executor reaches the worker by method name, and that detail explains the whole failure mode.`,
    story: `<p>Upstream's executor does not import the worker class — it dispatches <code>collective_rpc("sleep")</code> by string and lets <code>WorkerWrapperBase.__getattr__</code> find the method. A plugin that omits one therefore produces an <mark>AttributeError from inside a worker RPC loop</mark>, and only when the feature is used. That is the mechanical reason this atlas treats "refuse loudly at config time" as the shipped behaviour for every feature TT does not serve: silence is what the contract does by default.</p>`,
    flow: [['vExec', 'vWorker', 'collective_rpc("execute_model")', { args: 1 }], ['vWorker', 'vRunner', 'run step', { batch_size: 1 }], ['vRunner', 'vExec', 'output', { token_ids: [4821] }]] },

  { id: 'all', title: 'The whole system', reveal: [],
    lede: `Everything at once, for free exploration.`,
    story: `<p>Choose which flow runs (bottom left). Hover anything; click to pin; → goes inside. The <mark>Open questions</mark> tab lists every question by ID, and the decisions table in <code>SYSTEM.md</code> records the calls that were actually made rather than the ones that looked obvious.</p>`,
    flow: null },
];

export const HOW_HTML = `<div class="eyebrow">vllm-tt-plugin · vLLM 0.26.0</div><h1 class="t">How it's built</h1><div class="sub">the shape, and what sits around it</div>
<h3 class="sec">Baseline</h3><p>Upstream vLLM tag <code>v0.26.0</code> = <code>568afb3a13806beb53bb2e6bd518269357b237c0</code>, pinned by <code>docs/install-vllm-tt.sh:36</code>. Every gap claim in the research report is measured against that tag, not against upstream <code>main</code>. Upstream citations on this map are written <code>path:line</code> against that tag; the analysis checkout used locally is gitignored and is not part of this repository.</p>
<h3 class="sec">Filesystem</h3><pre>src/vllm_tt_plugin/
  entrypoints.py · model_registry.py · loader.py
  platform.py · config.py · logger.py
  scheduler.py · lane_scheduler.py · input_batch.py
  worker.py · model_runner.py · model_input.py
  async_decode.py · structured_output.py · logprobs.py
  spec_decode.py · spec_accept.py
  feature_support.py · utils/dp_discovery.py
(upstream v0.26.0)   read at the tag, not vendored here
system-atlas/atlas/         this atlas</pre>
<h3 class="sec">Size, largest first</h3><pre>model_runner.py 2753   platform.py  2290   input_batch.py  1505
scheduler.py     996   worker.py     977   async_decode.py   945
lane_scheduler.py 816  spec_accept.py 512    config.py        369
spec_decode.py   385   dp_discovery.py 270   model_input.py  177
structured_output.py 100  logprobs.py 81  loader.py 55</pre>
<h3 class="sec">Tests</h3><p>411 host test functions across 36 files under <code>tests/</code> (excluding <code>tests/tt</code>, which needs a live server and TT hardware). Zero skips and zero xfails in the host suite. <code>ci/host-stubs/ttnn/</code> supplies an import-only <code>ttnn</code> stand-in whose every device-reaching entry point raises, so a test that starts depending on real hardware fails loudly instead of passing against a fake device.</p>`;
