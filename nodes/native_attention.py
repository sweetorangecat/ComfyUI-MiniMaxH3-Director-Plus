"""Lazy ComfyUI native sparse attention for SelfLift's global-stream contract.

Verification here describes the installed model hooks, not GPU kernel dispatch.
No Torch, ComfyUI, or optional attention plugin is imported at module load time.
"""
from __future__ import annotations

import importlib
import inspect
import math
from pathlib import Path
import sys
from types import CodeType


_NATIVE_MODULE = "comfy_extras.nodes_sparse_attention"
_SOL_CONFIG = {
    "tau": 1.3, "start_percent": .2, "end_percent": 1., "min_tokens": 12288,
    "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
    "dense_blocks": "", "verbose": False,
}
NATIVE_ATTENTION_EVIDENCE_KEYS = (
    "sol_attn_provider", "native_sparse_verified", "native_sparse_blocks",
    "native_sparse_total_blocks", "native_sparse_override_verified",
    "native_sparse_reason", "native_sparse_verification",
    "native_sparse_start_sigma", "native_sparse_end_sigma",
    "selflift_attention_requested_mode", "selflift_attention_route",
    "selflift_global_stream_supported", "selflift_highres_tiling", "selflift_highres_tiling_mode",
    "selflift_highres_token_chunk", "selflift_highres_weight_cache_gb",
    "selflift_node_id", "selflift_node_variant", "selflift_speed_warning",
)
SOL_ATTN_EVIDENCE_KEYS = (
    "sol_attn_applied", "sol_attn_tau", "sol_attn_start_percent", "sol_attn_end_percent",
    "sol_attn_min_tokens", "sol_attn_extra_tokens_requested",
    "sol_attn_extra_tokens_applied", "sol_attn_extra_tokens",
    "sol_attn_sink_conditioning", "sol_attn_int8_qk", "sol_attn_int8_pv",
)


def native_sparse_node():
    """Resolve the core provider independently of the external Sol plugin."""
    module = importlib.import_module(_NATIVE_MODULE)
    node = getattr(module, "BlockSparseAttention", None)
    if not callable(getattr(node, "execute", None)):
        raise RuntimeError("ComfyUI native BlockSparseAttention is unavailable")
    return node


def _native_function(function, module, factory_name, leaf):
    """Require the actual loaded factory's code and globals, not its name alone."""
    if module is None or getattr(function, "__module__", None) != _NATIVE_MODULE:
        return False
    factory = getattr(module, factory_name, None)
    factory_code = getattr(factory, "__code__", None)
    code = getattr(function, "__code__", None)
    if factory_code is None or code is None:
        return False
    return (
        function.__globals__ is module.__dict__
        and Path(code.co_filename).name == "nodes_sparse_attention.py"
        and function.__qualname__ == f"{factory_name}.<locals>.{leaf}"
        and any(isinstance(item, CodeType) and item.co_name == leaf and code is item
                for item in factory_code.co_consts)
    )


def _nonlocals(function):
    try:
        return inspect.getclosurevars(function).nonlocals
    except (TypeError, ValueError):
        return {}


def _finite_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _forward_patch_conflict(model, options):
    patches = getattr(model, "object_patches", {})
    if isinstance(patches, dict):
        for name in patches:
            if not isinstance(name, str):
                continue
            parts = name.split(".")
            if parts[:2] != ["diffusion_model", "blocks"]:
                continue
            # get_model_object("diffusion_model") resolves an exact whole-model
            # replacement, but cannot expose deferred block/attention children.
            # They can stale our closures or change forwards the producer would
            # bypass. Keep MLP chunk hooks and ordinary weight LoRAs supported.
            if len(parts) <= 3 or parts[3] in {"attn", "forward"}:
                return f"unsupported deferred H3 attention/block object patch: {name}"
    if options.get("sol_take_forward") is not None:
        return "unsupported upstream sol_take_forward attention hook"
    return None


def reject_attention_conflicts(model):
    """Guard a replacement/fallback that would bypass upstream attention hooks."""
    options = getattr(model, "model_options", {}).get("transformer_options", {})
    dit = options.get("patches_replace", {}).get("dit", {})
    reason = _forward_patch_conflict(model, options)
    if options.get("optimized_attention_override") is not None:
        reason = reason or "an upstream attention override is installed"
    if any(isinstance(key, tuple) and key and key[0] == "double_block" for key in dit):
        reason = reason or "upstream H3 attention block replacements are installed"
    if reason:
        raise RuntimeError("Cannot replace upstream attention hooks: " + reason)


def _config_error(model, patch, config, extra_supported):
    for key in ("tau", "min_tokens"):
        actual = _finite_number(getattr(patch, key, None))
        if actual is None or not math.isclose(actual, config[key], rel_tol=1e-7, abs_tol=1e-7):
            return f"native sol-attn {key} does not match the requested contract"
    if getattr(patch, "sink_conditioning", None) != config["sink_conditioning"]:
        return "native sol-attn conditioning sink does not match the requested contract"
    if getattr(patch, "dense_blocks", None) != set():
        return "native sol-attn must cover every H3 block without dense-block exclusions"
    extra = config["extra_tokens"] if extra_supported else 0
    if getattr(patch, "extra_tokens", None) != extra:
        return "native sol-attn extra_tokens does not match the supported contract"
    try:
        sampling = model.get_model_object("model_sampling")
        for name, percent in (("sigma_start", "start_percent"), ("sigma_end", "end_percent")):
            actual = _finite_number(getattr(patch, name, None))
            expected = _finite_number(sampling.percent_to_sigma(config[percent]))
            if actual is None or expected is None or not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
                return "native sol-attn sigma window does not match the actual model sampling"
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return "native sol-attn sigma window cannot be checked against the actual model sampling"
    return None


def inspect_native_attention(model, *, config=None):
    """Inspect all current H3 block closures and their matching native override.

    Unknown hooks, including a native wrapper around an unknown previous hook,
    remain untouched and are incompatible. Sparse producers may bypass those
    previous hooks, so their callable presence alone is insufficient evidence.
    """
    config = {**_SOL_CONFIG, **(config or {})}
    evidence = dict(sol_attn_provider="none", native_sparse_verified=False,
                    native_sparse_blocks=0, native_sparse_total_blocks=0,
                    native_sparse_override_verified=False,
                    native_sparse_reason="native H3 transformer blocks are unavailable",
                    native_sparse_verification="model_patch_protocol",
                    native_sparse_start_sigma=None, native_sparse_end_sigma=None)
    options = getattr(model, "model_options", {})
    options = options.get("transformer_options", {}) if isinstance(options, dict) else {}
    if not isinstance(options, dict):
        evidence["native_sparse_reason"] = "invalid transformer_options"
        return evidence
    override = options.get("optimized_attention_override")
    if override is not None:
        evidence["sol_attn_provider"] = "generic_attention_override"
    try:
        blocks = model.get_model_object("diffusion_model").blocks
        evidence["native_sparse_total_blocks"] = len(blocks)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return evidence
    if not blocks:
        return evidence
    replace = options.get("patches_replace", {})
    dit = replace.get("dit", {}) if isinstance(replace, dict) else {}
    if not isinstance(dit, dict):
        evidence["native_sparse_reason"] = "invalid H3 block replacement table"
        return evidence
    module = sys.modules.get(_NATIVE_MODULE)
    patch_type = getattr(module, "SparseAttnPatch", ())
    patches = []
    conflict = _forward_patch_conflict(model, options)
    problems = [conflict] if conflict else []
    for index, block in enumerate(blocks):
        function = dit.get(("double_block", index))
        if function is None:
            problems.append(f"missing native H3 block patch at {index}")
            continue
        if not _native_function(function, module, "make_h3_block_patch", "block_patch"):
            problems.append(f"unsupported H3 block patch at {index}")
            continue
        closure = _nonlocals(function)
        patch = closure.get("patch")
        if (not isinstance(patch, patch_type) or closure.get("block") is not block
                or type(closure.get("block_index")) is not int or closure["block_index"] != index):
            problems.append(f"native sparse patch does not match current H3 block {index}")
            continue
        evidence["native_sparse_blocks"] += 1
        patches.append(patch)
    patch = patches[0] if patches else None
    if patches:
        evidence["sol_attn_provider"] = "comfy_native_block_sparse"
    if any(item is not patch for item in patches):
        problems.append("H3 blocks do not share the same native sparse patch")
    if not _native_function(override, module, "make_attention_override", "override"):
        problems.append("missing or unknown native attention override")
    else:
        closure = _nonlocals(override)
        if patch is None or closure.get("patch") is not patch:
            problems.append("native attention override does not match the H3 block patches")
        elif closure.get("previous") is not None:
            problems.append("native attention override wraps an unsupported previous attention override")
        else:
            evidence["native_sparse_override_verified"] = True
    if patch is not None:
        evidence["native_sparse_start_sigma"] = _finite_number(getattr(patch, "sigma_start", None))
        evidence["native_sparse_end_sigma"] = _finite_number(getattr(patch, "sigma_end", None))
        if getattr(patch, "vsa", None) is not False or getattr(patch, "topk_ratio", None) != 0.0:
            problems.append("native global streaming requires sol-attn, not SLA/VSA")
        try:
            extra_supported = "extra_tokens" in inspect.signature(native_sparse_node().execute).parameters
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            problems.append("native BlockSparseAttention provider is unavailable")
        else:
            problem = _config_error(model, patch, config, extra_supported)
            if problem:
                problems.append(problem)
            if not problems:
                evidence.update(
                    sol_attn_applied=True, sol_attn_tau=float(patch.tau),
                    sol_attn_start_percent=config["start_percent"], sol_attn_end_percent=config["end_percent"],
                    sol_attn_min_tokens=int(patch.min_tokens),
                    sol_attn_extra_tokens_requested=config["extra_tokens"],
                    sol_attn_extra_tokens_applied=extra_supported,
                    sol_attn_extra_tokens=int(patch.extra_tokens),
                    sol_attn_sink_conditioning=patch.sink_conditioning,
                )
    evidence["native_sparse_verified"] = not problems
    evidence["native_sparse_reason"] = problems[0] if problems else None
    return evidence


def apply_native_sol_attention(model, config):
    """Apply the requested core patch without erasing upstream attention hooks."""
    config = {**_SOL_CONFIG, **config}
    existing = inspect_native_attention(model, config=config)
    if existing["native_sparse_verified"]:
        return model, existing
    reject_attention_conflicts(model)
    node = native_sparse_node()
    parameters = inspect.signature(node.execute).parameters
    kwargs = dict(model=model, selection={"selection": "sol-attn", "tau": config["tau"]},
                  start_percent=config["start_percent"], end_percent=config["end_percent"],
                  dense_blocks=config["dense_blocks"], min_tokens=config["min_tokens"],
                  sink_conditioning=config["sink_conditioning"], verbose=config["verbose"])
    if "extra_tokens" in parameters:
        kwargs["extra_tokens"] = config["extra_tokens"]
    result = node.execute(**kwargs)
    result = getattr(result, "result", result)
    patched = result[0] if isinstance(result, (tuple, list)) else result
    evidence = inspect_native_attention(patched, config=config)
    if not evidence["native_sparse_verified"]:
        raise RuntimeError("Native Sol-Attn/global_stream compatibility failed: " + evidence["native_sparse_reason"])
    return patched, evidence
