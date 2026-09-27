# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Config-time and request-time refusals for upstream features TT cannot serve.

Upstream vLLM accepts a long list of engine options and request parameters and
then discovers, deep inside execution, that the active platform cannot honour
them. On the TT backend those options are worse than useless: a launch that
"works" by accident returns wrong answers. This module turns each one into a
refusal raised during config resolution or request validation, with a message
that prints the offending value and names the CLI flag the operator typed.

Why a separate module rather than more of ``platform.py``: that file is already
2260 lines and AGENTS.md §7 says a new conditional branch there is a prompt to
split into a named neighbour. ``config.py`` is the accessor layer for values
handed across process boundaries, not a validator, so a purpose-named module is
the smaller change.

Two rules govern everything here:

* No ``assert``. ``python -O`` removes asserts, and a refusal that disappears
  under ``-O`` is not a refusal (AGENTS.md §2.4).
* Only read fields that the pinned vLLM actually defines. A validator that
  raises ``AttributeError`` is a crash, not a clean refusal. Every attribute
  read below was checked against the pinned tree in ``third_party/vllm-0.26.0``.

Deliberately absent: sleep mode and the nixl/KV-connector capability members.
``ModelConfig.__post_init__`` already refuses ``enable_sleep_mode`` because it
gates the flag on ``current_platform.is_sleep_mode_available()``
(``third_party/vllm-0.26.0/vllm/config/model.py:546``), which is ``False`` for
TT, and ``Platform.get_nixl_*`` already return empty defaults, so a second
message for a problem upstream reports well would be duplication rather than
coverage.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams

# The subset of vllm.config.cache.CacheDType the TT backend can allocate.
# Every other literal in that alias (fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla,
# the turboquant and per-token-head families, nvfp4) names a device-side
# encoding the tt-metal KV allocator does not implement. Reached through
# --kv-cache-dtype.
TT_SUPPORTED_CACHE_DTYPES = frozenset({"auto", "bfloat16", "float16"})

# The subset of vllm.config.model.RunnerOption the TT backend can serve.
# "pooling" and "draft" name runners the plugin does not implement; see
# TTModelRunner.get_supported_tasks. Reached through --runner.
TT_SUPPORTED_RUNNERS = frozenset({"auto", "generate"})


def validate_speculative_decoding(cls: type, vllm_config: "VllmConfig") -> None:
    """Refuse speculative decoding.

    The contract types (spec_decode.py) and the host accept walk
    (spec_accept.py) exist, but the model drafter that produces candidates is
    not wired into the runner, so there is nothing to accept yet. Upstream
    reads speculative_config in Scheduler.__init__ (scheduler.py:236-262) and
    would set up lookahead slots and draft-token bookkeeping that the TT
    scheduler never satisfies.
    """
    if vllm_config.speculative_config is not None:
        raise ValueError(
            f"Speculative decoding is not supported by the TT backend, but "
            f"vllm_config.speculative_config is set to "
            f"{vllm_config.speculative_config!r}. Unset --speculative-config "
            f"and --speculative-model, or use a build that includes the "
            f"speculative-decoding drafter work (issue #110)."
        )


def validate_tensor_and_pipeline_parallel(cls: type, vllm_config: "VllmConfig") -> None:
    """Refuse tensor- and pipeline-parallel launch.

    TT parallelism lives inside the tt-metal generator, which shards a model
    across the devices of one mesh. It is not exposed as vLLM tensor or
    pipeline parallelism, so a size above 1 would ask upstream to shard a model
    the plugin never hands it in pieces.
    """
    parallel_config = vllm_config.parallel_config
    tensor_parallel_size = parallel_config.tensor_parallel_size
    pipeline_parallel_size = parallel_config.pipeline_parallel_size
    if tensor_parallel_size != 1 or pipeline_parallel_size != 1:
        raise ValueError(
            f"Tensor/pipeline parallelism is not supported by the TT backend: "
            f"resolved tensor_parallel_size={tensor_parallel_size} and "
            f"pipeline_parallel_size={pipeline_parallel_size} (both must be 1). "
            f"TT shards a model across the devices of one mesh inside the "
            f"tt-metal generator rather than at the vLLM level, so do not pass "
            f"--tensor-parallel-size or --pipeline-parallel-size. Use "
            f"--data-parallel-size to run several independent meshes."
        )


def validate_context_parallel(cls: type, vllm_config: "VllmConfig") -> None:
    """Refuse decode- and prefill-context parallelism.

    Both are read by upstream's Scheduler.__init__ (scheduler.py:169-170) and
    shard KV state across ranks. The TT scheduler allocates KV per mesh without
    a CP axis, so honouring either would silently produce a wrong KV layout.
    """
    parallel_config = vllm_config.parallel_config
    decode_context_parallel_size = parallel_config.decode_context_parallel_size
    prefill_context_parallel_size = parallel_config.prefill_context_parallel_size
    if decode_context_parallel_size != 1 or prefill_context_parallel_size != 1:
        raise ValueError(
            f"Context parallelism is not supported by the TT backend: resolved "
            f"decode_context_parallel_size={decode_context_parallel_size} and "
            f"prefill_context_parallel_size={prefill_context_parallel_size} "
            f"(both must be 1). Unset --decode-context-parallel-size and "
            f"--prefill-context-parallel-size."
        )


def validate_expert_parallel(cls: type, vllm_config: "VllmConfig") -> None:
    """Refuse expert parallelism and expert-parallel load balancing.

    ``ParallelConfig.eplb_config`` is ``Field(default_factory=EPLBConfig)`` and
    is therefore never ``None``, so the gate is the ``enable_eplb`` boolean
    beside it. Expert parallelism shards MoE experts across ranks, which the TT
    mesh allocator does not do.
    """
    parallel_config = vllm_config.parallel_config
    enable_expert_parallel = parallel_config.enable_expert_parallel
    enable_eplb = parallel_config.enable_eplb
    if enable_expert_parallel or enable_eplb:
        raise ValueError(
            f"Expert parallelism is not supported by the TT backend: resolved "
            f"enable_expert_parallel={enable_expert_parallel} and "
            f"enable_eplb={enable_eplb} (both must be False). MoE expert "
            f"placement is decided by the tt-metal generator across one mesh, "
            f"so do not pass --enable-expert-parallel, --enable-eplb, or "
            f"--eplb-config."
        )


def validate_lora(cls: type, vllm_config: "VllmConfig") -> None:
    """Refuse LoRA.

    vLLM resolves a punica wrapper through
    ``Platform.get_punica_wrapper()`` and dispatches ``add_lora`` and friends to
    the worker. The TT model code is a tt-metal generator, so there is no
    punica layer to attach adapters to. The README states this as an
    operational constraint; this makes it a startup error instead of a late one.
    """
    if vllm_config.lora_config is not None:
        raise ValueError(
            f"LoRA is not supported by the TT backend, but vllm_config."
            f"lora_config is set to {vllm_config.lora_config!r}. Unset "
            f"--enable-lora, --lora-modules, --lora-extra-vocab-size, "
            f"--max-lora-rank and the other --lora-* options."
        )


def validate_kv_cache_dtype(cls: type, vllm_config: "VllmConfig") -> None:
    """Refuse a KV cache dtype the TT allocator cannot hold.

    ``TTWorker._build_default_kv_cache_spec`` maps the requested dtype through
    ``STR_DTYPE_TO_TORCH_DTYPE`` (worker.py:374-378), so an unsupported value
    would surface as a ``KeyError`` during worker startup rather than as a
    message naming the flag.
    """
    cache_dtype = vllm_config.cache_config.cache_dtype
    if cache_dtype is not None and cache_dtype not in TT_SUPPORTED_CACHE_DTYPES:
        supported = ", ".join(sorted(TT_SUPPORTED_CACHE_DTYPES))
        raise ValueError(
            f"KV cache dtype {cache_dtype!r} is not supported by the TT "
            f"backend: supported values are {supported}. Unset --kv-cache-dtype "
            f"to use the model dtype, or pass one of the supported values "
            f"explicitly. TT weight and KV quantization needs a tt-metal "
            f"generator that loads quantized weights, which no released "
            f"generator does yet."
        )


def validate_kv_transfer(cls: type, vllm_config: "VllmConfig") -> None:
    """Refuse P/D disaggregation and encoder-cache transfer.

    Upstream's ``Scheduler.__init__`` builds a KV connector from
    ``kv_transfer_config`` (scheduler.py:127-153) and an encoder-cache connector
    from ``ec_transfer_config`` (scheduler.py:159-163). ``TTLaneCoordinator``
    reports no connector, so a configured transfer would be accepted at startup
    and then never happen.
    """
    kv_transfer_config = vllm_config.kv_transfer_config
    ec_transfer_config = vllm_config.ec_transfer_config
    if kv_transfer_config is not None or ec_transfer_config is not None:
        raise ValueError(
            f"KV transfer is not supported by the TT backend: resolved "
            f"kv_transfer_config={kv_transfer_config!r} and "
            f"ec_transfer_config={ec_transfer_config!r} (both must be unset). "
            f"Unset --kv-transfer-config. The TT scheduler attaches no KV "
            f"connector, so a configured transfer would be accepted at startup "
            f"and never performed."
        )


def validate_runner(cls: type, vllm_config: "VllmConfig") -> None:
    """Refuse a non-generative model runner.

    ``ModelConfig.runner`` is the declared field behind ``--runner``; the
    resolved ``runner_type`` it becomes is an instance attribute set during
    ``__post_init__``. Gating on the declared field keeps this validator
    independent of construction order. ``TTModelRunner.get_supported_tasks``
    already returns only generation tasks, so this is the config-time half of
    the same rule and the pooling asserts in ``input_batch.py`` are its
    runtime backstop.
    """
    runner = vllm_config.model_config.runner
    if runner not in TT_SUPPORTED_RUNNERS:
        supported = ", ".join(sorted(TT_SUPPORTED_RUNNERS))
        raise ValueError(
            f"Model runner {runner!r} is not supported by the TT backend: "
            f"supported values are {supported}. Unset --runner, or pass one of "
            f"the supported values. The TT backend serves text generation only; "
            f"pooling, embedding and transcription runners are not implemented."
        )


def validate_routed_experts(cls: type, vllm_config: "VllmConfig") -> None:
    """Refuse routed-expert reporting.

    Upstream builds a ``RoutedExpertsManager`` from this flag
    (scheduler.py:329-340) and ``SamplingParams.routed_experts_prompt_start``
    indexes into its output. The TT scheduler collects neither, so the request
    would be accepted and the routing data silently absent.
    """
    enable_return_routed_experts = vllm_config.model_config.enable_return_routed_experts
    if enable_return_routed_experts:
        raise ValueError(
            f"Routed-expert reporting is not supported by the TT backend: "
            f"resolved enable_return_routed_experts="
            f"{enable_return_routed_experts}. Unset "
            f"--enable-return-routed-experts. The TT scheduler builds no "
            f"RoutedExpertsManager, so the routing data would be silently "
            f"missing from every response."
        )


def validate_all(cls: type, vllm_config: "VllmConfig") -> None:
    """Run every config-time refusal in a fixed order.

    Called from ``TTPlatform._apply_check_and_update_config`` at the point the
    three original ``assert`` refusals sat, so every unsupported feature is
    rejected at the same point in config resolution as before.
    """
    validate_speculative_decoding(cls, vllm_config)
    validate_tensor_and_pipeline_parallel(cls, vllm_config)
    validate_context_parallel(cls, vllm_config)
    validate_expert_parallel(cls, vllm_config)
    validate_lora(cls, vllm_config)
    validate_kv_cache_dtype(cls, vllm_config)
    validate_kv_transfer(cls, vllm_config)
    validate_runner(cls, vllm_config)
    validate_routed_experts(cls, vllm_config)


def verify_quantization(cls: type, quant: str) -> None:
    """``Platform.verify_quantization`` override for the TT backend.

    Upstream's implementation only raises when ``supported_quantization`` is a
    non-empty list, and its message names neither the flag nor the supported
    set (vllm/platforms/interface.py:948-956). The base list holds
    *quantization method names*, not dtypes, so the honest TT answer is "none
    of them" rather than a list of dtypes. ``TTPlatform`` therefore leaves
    ``supported_quantization`` empty and refuses here instead, where the
    message can say what the operator should do.

    Called by ``ModelConfig._verify_quantization`` (model.py:1121) during
    ``VllmConfig`` construction — that is, before ``check_and_update_config``
    runs at all, so this one refusal fires earlier than the rest.
    """
    raise ValueError(
        f"Weight quantization {quant!r} is not supported by the TT backend. "
        f"Unset --quantization to load the model in its own dtype. TT weight "
        f"quantization requires a tt-metal generator that loads quantized "
        f"weights, and no released generator does; serving a quantized model "
        f"without that would silently produce wrong output."
    )


def _summarize(value: object) -> str:
    """Render an offending request value for a refusal message.

    AGENTS.md §7 requires a refusal to print the offending value, not only the
    field name. ``repr`` on an arbitrary container can be arbitrarily long (and
    ``RepetitionDetectionParams`` has no useful ``repr``), so this caps the
    rendering and marks the truncation rather than letting a client-supplied
    value flood the error.
    """
    rendered = repr(value)
    limit = 120
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[:limit]}... (truncated from {len(rendered)} chars)"


def unsupported_request_params(params: "SamplingParams") -> list[str]:
    """Return one line per request control the TT backend silently drops.

    These three are accepted by vLLM and then ignored on the TT backend, so a
    client asking for them gets output that does not match the request. They
    have no CLI flag: they are per-request, so the message names the field and
    the accepted value instead.

    ``n`` is deliberately absent. It looks like a gap but is not one: upstream
    fans a multi-completion request out into independent single-sample child
    requests, and that happens strictly after this validation runs.
    ``AsyncLLM`` calls ``InputProcessor.process_inputs`` (which calls
    ``Platform.validate_request`` at ``input_processor.py:296``) on the parent
    params, and only afterwards expands the children at
    ``async_llm.py:388-397`` via ``ParentRequest.get_child_info``, each with
    ``n = 1`` (``parallel_sampling.py:73-74``). ``OutputProcessor`` re-joins
    them under one external id. The runner therefore only ever sees ``n == 1``,
    and refusing ``n > 1`` here would reject a request that already works.
    ``LLMEngine`` does the same fan-out at ``llm_engine.py:280``.

    Block-output models refuse a superset, including ``n``; a block-output
    model owns its sampling outright and never sees the child-request shape.
    """
    unsupported = []
    if params.thinking_token_budget is not None:
        unsupported.append(f"thinking_token_budget={params.thinking_token_budget!r}")
    if params.repetition_detection is not None:
        unsupported.append(
            f"repetition_detection={_summarize(params.repetition_detection)}"
        )
    if params.extra_args:
        unsupported.append(f"extra_args={_summarize(params.extra_args)}")
    return unsupported


def unsupported_block_output_params(params: "SamplingParams") -> list[str]:
    """Return one line per request control a block-output model cannot honour.

    A block-output model owns its Gumbel sampling outright, so it refuses the
    whole response-shaping set rather than routing part of it to host sampling.
    """
    unsupported = unsupported_request_params(params)
    # n is refused here and not in unsupported_request_params: a block-output
    # model owns its Gumbel sampling and never sees upstream's child-request
    # fan-out, so there is nothing to aggregate multiple completions into.
    if params.n != 1:
        unsupported.insert(0, f"n={params.n!r} (accepted: 1)")
    if params.logprobs is not None:
        unsupported.append(f"logprobs={params.logprobs!r} (accepted: None)")
    if params.logprob_token_ids is not None:
        unsupported.append("logprob_token_ids (accepted: omitted/None)")
    if params.flat_logprobs:
        unsupported.append("flat_logprobs=True (accepted: False)")
    if params.bad_words:
        unsupported.append("bad_words (accepted: omitted/empty)")
    if params.structured_outputs is not None:
        unsupported.append("structured_outputs (accepted: omitted/None)")
    if params.logit_bias is not None:
        unsupported.append("logit_bias (accepted: omitted/None)")
    if params.allowed_token_ids is not None:
        unsupported.append("allowed_token_ids (accepted: omitted/None)")
    if params.min_tokens != 0:
        unsupported.append(f"min_tokens={params.min_tokens!r} (accepted: 0)")
    return unsupported


def raise_for_unsupported_params(unsupported: list[str], reason: str, dev: str) -> None:
    """Raise the shared "does not support these request parameters" error.

    One message builder for both request paths, so the block-output refusal and
    the token-at-a-time refusal cannot drift apart in wording.
    """
    if unsupported:
        raise ValueError(
            f"{reason} on {dev}, and does not support these request "
            f"parameters: " + "; ".join(unsupported)
        )
