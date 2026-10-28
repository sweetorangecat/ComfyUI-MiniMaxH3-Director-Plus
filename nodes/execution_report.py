"""Bounded execution evidence carried by actual outputs, never global job state.

Example::

    report = new_report(guide, latent_image)
    with stage_timer(report, "sampling", path="native", steps=len(sigmas) - 1):
        output = native_sample(...)
    attach_report(output, report)

Measurements are host wall-clock durations, not isolated CUDA kernel times. An
inherited measurement describes the source artifact's creation; ComfyUI may
have cached that artifact. Stages overlap when one encloses other work and
must not be summed into a claimed current-job total.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import logging
import math
import time

LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus.Execution")
REPORT_KEY = "_h3_execution_report"
MAX_STAGES = 64
_PLAN_KEYS = (
    "mode", "voice_mode", "performance_preset", "resolved_backend", "seed",
    "width", "height", "length", "target_width", "target_height", "steps",
    "sampler_name", "scheduler", "scheduler_name", "scheduler_sigma_count",
    "video_shift", "audio_shift", "two_stage_enabled",
    "resolved_two_stage_route", "first_lora_name", "second_lora_name", "vosr2_profile",
    "selflift_lora_kind",
    "first_lora_strength", "second_lora_strength", "postprocess_mode", "postprocess_path",
    "ref_image_size", "requested_ref_image_size",
    "sage_requested", "sage_applied", "sage_error", "sage_disabled_reason",
    "attention_backend", "minimax_head_chunks", "mlp_chunking_applied",
    "mlp_chunking_error", "minimax_mlp_chunks", "minimax_mlp_seq_threshold",
    "sol_attn_applied", "sol_attn_error", "sol_attn_tau", "sol_attn_min_tokens",
    "sol_attn_extra_tokens_requested", "sol_attn_extra_tokens_applied",
    "sol_attn_extra_tokens",
    "sol_attn_sink_conditioning", "sol_attn_int8_qk", "sol_attn_int8_pv",
    "selflift_highres_tiling", "selflift_highres_tiling_mode",
    "selflift_highres_token_chunk", "selflift_highres_weight_cache_gb",
    "selflift_node_id", "selflift_node_variant",
    "selflift_global_stream_supported", "selflift_speed_warning",
)
_RECIPE_KEYS = (
    "id", "key", "name", "preset", "route", "backend", "adapter", "adapter_name", "lora_name",
    "adapter_filename", "lora_strength", "strength", "steps", "sampler", "sampler_name",
    "scheduler", "video_shift", "audio_shift", "resolution", "native_resolution",
    "use_cache", "use_sage", "two_stage",
)
_DETAIL_KEYS = (
    "path", "steps", "first_steps", "second_steps", "frame_count", "width", "height",
    "includes_nested_stages", "profile", "attempt", "engine", "scope",
)
_SAMPLING_KEYS = (
    "evidence", "origin", "sampler", "sampler_source", "scheduler", "scheduler_source",
    "steps", "sigma_count", "video_shift", "audio_shift", "first_lora_name",
    "first_lora_strength", "second_lora_name", "second_lora_strength",
    "selflift_rho", "selflift_w_min", "selflift_w_max", "selflift_upscaler_model",
    "selflift_node_id", "selflift_node_variant",
    "selflift_global_stream_supported", "selflift_speed_warning",
)


def _scalars(source, keys):
    result = {}
    for key in keys:
        value = source.get(key)
        if isinstance(value, str):
            result[key] = value[:256]
        elif value is None or isinstance(value, (bool, int)):
            if key in source:
                result[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[key] = value
    return result


def execution_plan(guide):
    """Describe routing and endpoint semantics without serializing references."""
    guide = guide or {}
    plan = _scalars(guide, _PLAN_KEYS)
    preprocessing = guide.get("reference_preprocessing")
    if isinstance(preprocessing, (list, tuple)):
        plan["reference_preprocessing"] = [str(item)[:256] for item in preprocessing[:8]]
    if isinstance(guide.get("scheduler_name"), str):
        plan["scheduler"] = guide["scheduler_name"][:256]
    signature = guide.get("scheduler_model_sampling")
    if isinstance(signature, dict):
        plan["scheduler_model_sampling"] = _scalars(signature, ("video_shift", "audio_shift"))
        plan.update(plan["scheduler_model_sampling"])
    route = guide.get("resolved_backend", "unknown")
    plan["route"] = str(route)[:256]
    endpoint = "hard_anchor" if route == "fl2va_model" else "reference_only"
    plan["endpoints"] = {
        label: endpoint if guide.get(key) is not None or guide.get("requested_" + key) else "absent"
        for label, key in (("first", "first_frame"), ("last", "last_frame"))
    }
    plan["references"] = {
        name: len(guide[key]) if isinstance(guide.get(key), dict) else 0
        for name, key in (("images", "ref_images"), ("videos", "ref_videos"),
                          ("audios", "ref_audios"), ("video_audios", "ref_video_audios"))
    }
    recipe = guide.get("sampling_recipe")
    if isinstance(recipe, dict):
        plan["sampling_recipe"] = _scalars(recipe, _RECIPE_KEYS)
    return plan


def report_from(output):
    if isinstance(output, dict):
        result = output.get(REPORT_KEY)
        if result is None and "samples" in output:
            return report_from(output["samples"])
    else:
        result = getattr(output, REPORT_KEY, None)
    return result if isinstance(result, dict) and result.get("version") == 1 else None


def _plan_snapshot(source):
    if not isinstance(source, dict):
        return {}
    plan = _scalars(source, (*_PLAN_KEYS, "route"))
    preprocessing = source.get("reference_preprocessing")
    if isinstance(preprocessing, (list, tuple)):
        plan["reference_preprocessing"] = [str(item)[:256] for item in preprocessing[:8]]
    if isinstance(source.get("scheduler_model_sampling"), dict):
        plan["scheduler_model_sampling"] = _scalars(source["scheduler_model_sampling"], ("video_shift", "audio_shift"))
    if isinstance(source.get("sampling_recipe"), dict):
        plan["sampling_recipe"] = _scalars(source["sampling_recipe"], _RECIPE_KEYS)
    if isinstance(source.get("endpoints"), dict):
        plan["endpoints"] = {key: value for key, value in _scalars(source["endpoints"], ("first", "last")).items()
                             if value in {"hard_anchor", "reference_only", "absent"}}
    if isinstance(source.get("references"), dict):
        plan["references"] = _scalars(source["references"], ("images", "videos", "audios", "video_audios"))
    return plan


def _model_sampling_signature(model):
    try:
        value = model.get_model_object("model_sampling")
        signature = {"video_shift": float(value.shift)}
        if getattr(value, "audio_shift", None) is not None:
            signature["audio_shift"] = float(value.audio_shift)
        return _scalars(signature, ("video_shift", "audio_shift"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return {}


def sampling_settings(guide, *, model=None, sampler=None, sigmas=None, second_model=None):
    """Snapshot actual sampler inputs; call at the execution boundary.

    Recipe declarations are deliberately excluded. Completed/failed execution
    is evidenced by the associated stage record, not by these input settings.
    Object metadata takes priority over a mutable shared guide.
    """
    guide = guide or {}
    settings = _scalars(guide, ("first_lora_name", "first_lora_strength", "second_lora_name", "second_lora_strength"))
    settings.update(evidence="sampler_inputs", origin="current_execution")
    sampler_name = getattr(sampler, "_h3_sampler_name", None)
    sampler_source = "sampler_object"
    if not isinstance(sampler_name, str):
        function = getattr(sampler, "sampler_function", None)
        function = getattr(function, "func", function)
        function_name = getattr(function, "__name__", "")
        if function_name.startswith("sample_"):
            sampler_name = function_name.removeprefix("sample_")
        else:
            sampler_name = guide.get("sampler_name")
            sampler_source = "router_guide"
    if isinstance(sampler_name, str):
        settings.update(sampler=sampler_name[:256], sampler_source=sampler_source)
    scheduler = getattr(sigmas, "_h3_scheduler_name", None)
    scheduler_source = "sigmas_object"
    if not isinstance(scheduler, str):
        scheduler = guide.get("scheduler_name")
        scheduler_source = "router_guide"
    if isinstance(scheduler, str):
        settings.update(scheduler=scheduler[:256], scheduler_source=scheduler_source)
    if sigmas is not None:
        settings.update(steps=max(0, len(sigmas) - 1), sigma_count=len(sigmas))
        scheduled = getattr(sigmas, "_mmh3_sampling_signature", None)
        if isinstance(scheduled, dict):
            settings["scheduler_model_sampling"] = _scalars(scheduled, ("video_shift", "audio_shift"))
    settings.update(_model_sampling_signature(model))
    if second_model is not None:
        signature = _model_sampling_signature(second_model)
        if signature:
            settings["second_model_sampling"] = signature
    return settings


def _sampling_snapshot(source):
    result = _scalars(source, _SAMPLING_KEYS)
    for key in ("scheduler_model_sampling", "second_model_sampling"):
        if isinstance(source.get(key), dict):
            result[key] = _scalars(source[key], ("video_shift", "audio_shift"))
    return result


def new_report(guide=None, source=None):
    upstream = report_from(source)
    report = {
        "version": 1,
        "measurement": "host_wall_clock_seconds",
        "timing_scope": "artifact_lineage; upstream may be cached; overlapping stages are not additive",
        "plan": execution_plan(guide),
        "stages": [],
    }
    if upstream:
        report["plan"] = _plan_snapshot(upstream.get("plan"))
        if isinstance(upstream.get("current_plan"), dict):
            report["current_plan"] = _plan_snapshot(upstream["current_plan"])
        if guide:
            report["current_plan"] = execution_plan(guide)
        reconstruction = reconstruction_diagnostics(upstream.get("reconstruction"))
        if reconstruction:
            report["reconstruction"] = reconstruction
        if isinstance(upstream.get("sampling_settings"), dict):
            report["sampling_settings"] = _sampling_snapshot(upstream["sampling_settings"])
            report["sampling_settings"]["origin"] = "upstream_artifact"
        stages = upstream.get("stages")
        for stage in stages[-MAX_STAGES:] if isinstance(stages, list) else []:
            if not isinstance(stage, dict):
                continue
            inherited = _scalars(stage, (*_DETAIL_KEYS, "stage", "status", "seconds", "error_type"))
            inherited["origin"] = "upstream_artifact"
            report["stages"].append(inherited)
    return report


@contextmanager
def stage_timer(report, stage, **details):
    started = time.perf_counter()
    record = {"stage": str(stage)[:64], "status": "running", "origin": "current_execution",
              **_scalars(details, _DETAIL_KEYS)}
    try:
        yield record
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__[:64])
        raise
    else:
        record["status"] = "completed"
    finally:
        record["seconds"] = round(max(0.0, time.perf_counter() - started), 6)
        report["stages"].append(record)
        del report["stages"][:-MAX_STAGES]
        LOGGER.info("[H3 execution] %s", json.dumps(record, ensure_ascii=False, allow_nan=False))


def attach_report(output, report):
    """Attach a snapshot to a LATENT dict or IMAGE tensor and AV children."""
    snapshot = copy.deepcopy(report)
    if isinstance(output, dict):
        output[REPORT_KEY] = snapshot
        if "samples" in output:
            attach_report(output["samples"], snapshot)
    else:
        # Native outputs are tensors/NestedTensor. Non-tensor sentinel values
        # used by integrations need no metadata and remain untouched.
        if hasattr(output, "shape") or getattr(output, "is_nested", False):
            setattr(output, REPORT_KEY, snapshot)
            if getattr(output, "is_nested", False):
                for member in output.unbind():
                    attach_report(member, snapshot)
    return output


def build_execution_report(guide, source=None):
    """Build JSON-safe evidence; absence of an artifact yields no timings."""
    return new_report(guide, source)


def reconstruction_diagnostics(source):
    """Only numeric settings, memory evidence, shapes and status are exportable."""
    if not isinstance(source, dict):
        return {}
    result = _scalars(source, (
        "profile", "status", "seed", "scale", "backend_scale", "sizing_strategy", "elapsed_seconds", "wall_seconds",
        "timing_scope", "error_type", "frame_count", "seconds", "backend_implementation",
        "backend_released", "backend_release_error",
    ))
    for key in ("input_shape", "output_shape", "requested_size", "processing_input_size"):
        value = source.get(key)
        if isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value[:8]):
            result[key] = list(value[:8])
    for key in ("memory_before", "memory_after"):
        if isinstance(source.get(key), dict):
            result[key] = _scalars(source[key], (
                "available", "device", "device_name", "free_vram_gb", "total_vram_gb",
                "allocated_gb", "reserved_gb", "process_peak_allocated_gb",
                "process_peak_reserved_gb", "measurement_error",
            ))
    config_keys = ("quality_profile", "tile_strategy", "tile_size", "tile_overlap", "vae_tile_size",
                   "vae_tile_overlap", "image_batch", "frame_batch", "dino_batch", "temporal_cache",
                   "cache_threshold", "cache_refresh", "memory_policy", "color_alignment",
                   "torch_compile", "vae_encode_amp", "precision", "model_bundle")
    for key in ("requested_config", "settings", "upstream_settings", "effective_settings", "loader_settings", "video_settings"):
        if isinstance(source.get(key), dict):
            result[key] = _scalars(source[key], config_keys)
            for section in ("loader", "settings", "video"):
                if isinstance(source[key].get(section), dict):
                    result[key][section] = _scalars(source[key][section], config_keys)
    return result
