"""Native sparse protocol and SelfLift preflight, without GPU dispatch claims."""
import importlib
from pathlib import Path
import sys
import types

import pytest

from nodes import performance, sampling_experiments as experiments


NATIVE_SOURCE = '''
class SparseAttnPatch:
    def __init__(self, sampling, start_percent, end_percent, tau=1.3,
                 min_tokens=12288, extra_tokens=256,
                 sink_conditioning="exact_kv_and_rows"):
        self.tau = tau
        self.topk_ratio = 0.0
        self.vsa = False
        self.min_tokens = min_tokens
        self.extra_tokens = extra_tokens
        self.sink_conditioning = sink_conditioning
        self.dense_blocks = set()
        self.sigma_start = sampling.percent_to_sigma(start_percent)
        self.sigma_end = sampling.percent_to_sigma(end_percent)

def make_h3_block_patch(block, block_index, patch):
    def block_patch(args, extra):
        # All three values are real nonlocals in ComfyUI's installed hook.
        return block, block_index, patch, args, extra
    return block_patch

def make_attention_override(patch, previous):
    def override(func, *args, **kwargs):
        return patch, previous, func, args, kwargs
    return override

class BlockSparseAttention:
    @classmethod
    def execute(cls, model, selection, start_percent, end_percent,
                dense_blocks="", min_tokens=12288, extra_tokens=0,
                sink_conditioning="exact_kv_and_rows", verbose=False):
        calls.append(dict(selection=selection, start_percent=start_percent,
                          end_percent=end_percent, dense_blocks=dense_blocks,
                          min_tokens=min_tokens, extra_tokens=extra_tokens,
                          sink_conditioning=sink_conditioning, verbose=verbose))
        patch = SparseAttnPatch(model.get_model_object("model_sampling"),
                                start_percent, end_percent, selection["tau"],
                                min_tokens, extra_tokens, sink_conditioning)
        result = model.clone()
        options = result.model_options.setdefault("transformer_options", {})
        previous = options.get("optimized_attention_override")
        options["optimized_attention_override"] = make_attention_override(patch, previous)
        for i, block in enumerate(model.get_model_object("diffusion_model").blocks):
            result.set_model_patch_replace(make_h3_block_patch(block, i, patch),
                                            "dit", "double_block", i)
        return types.SimpleNamespace(result=(result,))
'''


def native_runtime(monkeypatch, *, extra_tokens=True):
    """The real core's factory/closure protocol, with no tensor kernels."""
    module = types.ModuleType("comfy_extras.nodes_sparse_attention")
    module.types = types
    module.calls = []
    source = NATIVE_SOURCE
    if not extra_tokens:
        source = source.replace('min_tokens=12288, extra_tokens=0,', 'min_tokens=12288,')
        source = source.replace('extra_tokens=extra_tokens,', 'extra_tokens=None,')
        source = source.replace('min_tokens, extra_tokens, sink_conditioning)',
                                'min_tokens, 0, sink_conditioning)')
    exec(compile(source, "/comfy_extras/nodes_sparse_attention.py", "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def copy_dicts(value):
    return {key: copy_dicts(item) if isinstance(item, dict) else item
            for key, item in value.items()}


class Model:
    def __init__(self, count=3):
        self.network = types.SimpleNamespace(blocks=[object() for _ in range(count)])
        self.sampling = types.SimpleNamespace(shift=12., audio_shift=3.)
        self.sampling.percent_to_sigma = lambda percent: self.sampling.shift * (1 - percent)
        self.model_options = {"transformer_options": {}}

    def get_model_object(self, name):
        # Comfy's getter resolves only exact object-patch keys. Deferred child
        # replacements stay invisible when reading the whole diffusion model.
        if name in getattr(self, "object_patches", {}):
            return self.object_patches[name]
        return {"diffusion_model": self.network, "model_sampling": self.sampling}[name]

    def clone(self):
        result = Model(0)
        result.network = self.network
        result.sampling = self.sampling
        result.model_options = copy_dicts(self.model_options)
        result.object_patches = getattr(self, "object_patches", {}).copy()
        return result

    def set_model_patch_replace(self, function, category, kind, index):
        self.model_options["transformer_options"].setdefault("patches_replace", {}).setdefault(category, {})[(kind, index)] = function


def native_model(runtime, model=None):
    return runtime.BlockSparseAttention.execute(
        model or Model(), {"selection": "sol-attn", "tau": 1.3}, .2, 1.,
        min_tokens=12288, extra_tokens=256,
        sink_conditioning="exact_kv_and_rows").result[0]


def legacy_override_model():
    model = Model()
    def make_override():
        def override(*args, **kwargs):
            return args, kwargs
        return override
    model.model_options["transformer_options"]["optimized_attention_override"] = make_override()
    return model


def native_api():
    assert (Path(__file__).resolve().parents[1] / "nodes/native_attention.py").is_file(), "Native sparse compatibility helper is missing"
    return importlib.import_module("nodes.native_attention")


def selflift_guide(**changes):
    result = dict(mode="REF2VA", performance_preset="quality_two_stage",
                  resolved_backend="ref2va_model", sampling_method="selflift",
                  total_vram_gb=32., free_vram_gb=26.,
                  selflift_upscaler_model="h3_fp16.safetensors",
                  attention_backend="sol_attn", sol_attn_applied=True)
    result.update(changes)
    return result


def run_short(model, guide):
    from test_sampling_experiments import AV, AvatarSamplerNode, euler_sampler
    node = AvatarSamplerNode()
    experiments.run_selflift(model, [], object(), euler_sampler(),
        [1, .95, .9, .8, .7, .6, .45, .3, .18, .08, 0],
        {"samples": AV()}, 42, guide,
        registry={"SelfLiftAvatarH3Sampler": lambda: node})
    return node


def test_legacy_generic_flag_cannot_claim_native_global_stream():
    guide = selflift_guide(native_sparse_verified=True, native_sparse_blocks=50,
                          native_sparse_total_blocks=50, selflift_attention_route="global_stream")
    node = run_short(legacy_override_model(), guide)
    assert node.received["highres_tiling_mode"] == "spatial"
    assert guide["native_sparse_verified"] is False
    assert guide["native_sparse_blocks"] == 0
    assert guide["selflift_attention_route"] == "spatial_compatibility"


def test_selflift_native_provider_is_independent_of_legacy_availability(monkeypatch):
    runtime = native_runtime(monkeypatch)
    monkeypatch.setattr(performance, "_sol_attn_class", lambda: pytest.fail("Native SelfLift must not load the legacy Sol plugin"))
    monkeypatch.setattr(performance, "_apply_minimax_ffn_chunking", lambda model, guide: model)
    guide = selflift_guide(sol_attn_available=False, sampling_attention_kernel="pytorch")
    assert performance.u22_attention_plan(guide)["backend"] == "sol_attn"
    patched = performance._apply_sol_attention(Model(), guide)
    assert guide["sol_attn_provider"] == "comfy_native_block_sparse"
    assert guide["native_sparse_verified"] is True
    assert guide["native_sparse_blocks"] == guide["native_sparse_total_blocks"] == 3
    assert guide["sol_attn_extra_tokens_requested"] == guide["sol_attn_extra_tokens"] == 256
    assert guide["sol_attn_extra_tokens_applied"] is True
    assert "sampling_attention_kernel" not in guide
    assert runtime.calls == [dict(selection={"selection": "sol-attn", "tau": 1.3},
        start_percent=.2, end_percent=1., dense_blocks="", min_tokens=12288,
        extra_tokens=256, sink_conditioning="exact_kv_and_rows", verbose=False)]
    assert native_api().inspect_native_attention(patched)["native_sparse_verified"] is True


def test_valid_native_closures_enable_stream_even_without_guide_claim(monkeypatch):
    runtime = native_runtime(monkeypatch)
    guide = selflift_guide(sol_attn_applied=False, attention_backend="stale")
    model = native_model(runtime)
    node = run_short(model, guide)
    assert node.received["highres_tiling_mode"] == "global_stream"
    assert node.received["rho"] == 0.
    assert node.received["low_res_model"] is node.received["high_res_model"] is model
    assert guide["native_sparse_verified"] is True
    assert guide["native_sparse_override_verified"] is True
    assert guide["selflift_attention_route"] == "global_stream"
    assert guide["attention_backend"] == "sol_attn"


@pytest.mark.parametrize("fault", ["partial", "stale", "wrong_index", "mixed_patch", "unknown_block", "spoofed_code", "unknown_override", "missing_override", "wrong_override", "unknown_previous", "sla", "vsa"])
def test_native_inspection_rejects_incompatible_block_and_override_protocol(monkeypatch, fault):
    runtime = native_runtime(monkeypatch)
    model = native_model(runtime)
    options = model.model_options["transformer_options"]
    dit = options["patches_replace"]["dit"]
    patch = __import__("inspect").getclosurevars(options["optimized_attention_override"]).nonlocals["patch"]
    other = runtime.SparseAttnPatch(model.sampling, .2, 1.)
    if fault == "partial":
        del dit[("double_block", 1)]
    elif fault == "stale":
        model.network = types.SimpleNamespace(blocks=[object() for _ in range(3)])
    elif fault == "wrong_index":
        dit[("double_block", 1)] = runtime.make_h3_block_patch(model.network.blocks[1], 2, patch)
    elif fault == "mixed_patch":
        dit[("double_block", 1)] = runtime.make_h3_block_patch(model.network.blocks[1], 1, other)
    elif fault == "unknown_block":
        dit[("double_block", 1)] = lambda args, extra: None
    elif fault == "spoofed_code":
        original = dit[("double_block", 1)]
        counterfeit = types.FunctionType(original.__code__.replace(), original.__globals__, closure=original.__closure__)
        counterfeit.__qualname__ = original.__qualname__
        dit[("double_block", 1)] = counterfeit
    elif fault == "unknown_override":
        options["optimized_attention_override"] = lambda *args, **kwargs: None
    elif fault == "missing_override":
        options.pop("optimized_attention_override")
    elif fault == "wrong_override":
        options["optimized_attention_override"] = runtime.make_attention_override(other, None)
    elif fault == "unknown_previous":
        options["optimized_attention_override"] = runtime.make_attention_override(patch, lambda *args: None)
    elif fault == "sla":
        patch.topk_ratio = .1
    elif fault == "vsa":
        patch.vsa = True
    evidence = native_api().inspect_native_attention(model)
    assert evidence["native_sparse_verified"] is False
    assert evidence["native_sparse_total_blocks"] == 3
    assert evidence["native_sparse_reason"]


def test_native_application_preserves_and_rejects_unknown_semantic_patches(monkeypatch):
    runtime = native_runtime(monkeypatch)
    model = legacy_override_model()
    original = model.model_options["transformer_options"]["optimized_attention_override"]
    with pytest.raises(RuntimeError, match="override|原生|attention"):
        performance._apply_sol_attention(model, selflift_guide())
    assert model.model_options["transformer_options"]["optimized_attention_override"] is original
    assert runtime.calls == []


@pytest.mark.parametrize("kind", ["author", "long"])
def test_strict_selflift_checks_actual_native_model_before_low_sampling(monkeypatch, kind):
    from test_sampling_experiments import AV, AvatarSamplerNode, euler_sampler
    node = AvatarSamplerNode()
    model = legacy_override_model()
    guide = selflift_guide(native_sparse_verified=True, selflift_global_stream_supported=True,
                          selflift_highres_tiling_mode="global_stream")
    sigmas = [1, .95, .9, .8, .7, .6, .45, .3, .18, .08, 0]
    latent = {"samples": AV()}
    if kind == "author":
        guide.update(community_lora_profile="combat", fps=24, motion_smoothing="off",
                     selflift_upscaler_model="h3_upscaler_lms_v0.1.safetensors")
        sigmas = [1, .96, .9, .85, .75, .65, .5, .4, .3, .2, .1, .04, 0]
    else:
        import torch
        class NativeAV:
            is_nested = True
            def __init__(self, values):
                self.values = values
            def unbind(self):
                return self.values
        guide["long_video_window"] = {"sample_frames": 328}
        latent = {"samples": NativeAV((torch.zeros(1, 4, 97, 2, 3), torch.zeros(1, 8, 2, 547))),
                  "noise_mask": NativeAV((torch.ones(1, 1, 97, 2, 3), torch.ones(1, 1, 2, 547)))}
    with pytest.raises(RuntimeError, match="global_stream|Sol-Attn|native"):
        experiments.run_selflift(model, [], object(), euler_sampler(), sigmas,
            latent, 42, guide, registry={"SelfLiftAvatarH3Sampler": lambda: node})
    assert not hasattr(node, "received")
    assert guide["native_sparse_verified"] is False
    assert guide["selflift_attention_route"] == "unavailable"
    assert "selflift_global_stream_supported" not in guide
    assert "selflift_highres_tiling_mode" not in guide


def test_restoring_and_falling_back_clears_stale_native_evidence():
    guide = selflift_guide(sol_attn_provider="comfy_native_block_sparse",
        native_sparse_verified=True, native_sparse_blocks=50, native_sparse_total_blocks=50,
        native_sparse_override_verified=True, native_sparse_reason=None,
        selflift_attention_requested_mode="global_stream", selflift_attention_route="global_stream")
    performance._clear_sol_attention_evidence(guide)
    assert "native_sparse_verified" not in guide
    assert "sol_attn_provider" not in guide
    assert "selflift_attention_route" not in guide


def test_standard_plan_and_application_keep_legacy_policy(monkeypatch):
    native_runtime(monkeypatch)
    received = []
    class Legacy:
        @staticmethod
        def execute(*args, **kwargs):
            received.append((args, kwargs))
            return (args[0],)
    monkeypatch.setattr(performance, "_sol_attn_class", lambda: Legacy)
    monkeypatch.setattr(performance, "_apply_minimax_ffn_chunking", lambda model, guide: model)
    guide = selflift_guide(sampling_method="standard", sol_attn_available=True)
    assert performance._apply_sol_attention("standard", guide) == "standard"
    assert received[0][0][1:5] == (1.3, .2, 1., 12288)
    guide["sol_attn_available"] = False
    assert performance.u22_attention_plan(guide)["backend"] == "sage"


def test_native_provider_reports_unsupported_extra_tokens_without_claiming_256(monkeypatch):
    runtime = native_runtime(monkeypatch, extra_tokens=False)
    monkeypatch.setattr(performance, "_apply_minimax_ffn_chunking", lambda model, guide: model)
    guide = selflift_guide()
    model = performance._apply_sol_attention(Model(), guide)
    assert guide["native_sparse_verified"] is True
    assert guide["sol_attn_extra_tokens_requested"] == 256
    assert guide["sol_attn_extra_tokens_applied"] is False
    assert guide["sol_attn_extra_tokens"] == 0
    assert runtime.calls[0]["extra_tokens"] is None
    assert native_api().inspect_native_attention(model)["native_sparse_verified"] is True


@pytest.mark.parametrize("setting,value", [("tau", 1.5), ("min_tokens", 0),
    ("sink_conditioning", "off"), ("dense_blocks", {1}), ("extra_tokens", 0),
    ("sigma_start", 7.)])
def test_native_inspector_rejects_changed_quality_contract(monkeypatch, setting, value):
    import inspect
    runtime = native_runtime(monkeypatch)
    model = native_model(runtime)
    patch = inspect.getclosurevars(model.model_options["transformer_options"]["optimized_attention_override"]).nonlocals["patch"]
    setattr(patch, setting, value)
    assert native_api().inspect_native_attention(model)["native_sparse_verified"] is False


def test_author_sparse_sigma_window_uses_actual_author_shift_before_patching(monkeypatch):
    runtime = native_runtime(monkeypatch)
    monkeypatch.setattr(performance, "resolve_registered_model_name", lambda category, name, **kwargs: "verified/" + name)
    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda model, *args, **kwargs: model)
    monkeypatch.setattr(performance, "_apply_minimax_ffn_chunking", lambda model, guide: model)
    class Shift:
        @staticmethod
        def execute(model, video, audio):
            shifted = model.clone()
            shifted.sampling = types.SimpleNamespace(shift=video, audio_shift=audio,
                percent_to_sigma=lambda percent: video * (1 - percent))
            return (shifted,)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_minimax_h3", types.SimpleNamespace(MiniMaxH3SigmaShift=Shift))
    model = Model()
    model.sampling.shift, model.sampling.audio_shift = 10., 5.
    guide = selflift_guide(community_lora_profile="combat", fps=24, motion_smoothing="off")
    first, _, ready, second = performance.MiniMaxH3AccelerationRouter().apply(model, guide)
    assert ready and first is second
    actual = native_api().inspect_native_attention(first)
    assert actual["native_sparse_verified"] is True
    assert actual["native_sparse_start_sigma"] == pytest.approx(12. * .8)
    assert len(runtime.calls) == 1


def test_execution_reports_carry_actual_protocol_evidence_and_compatible_route(monkeypatch):
    from nodes import execution_report
    runtime = native_runtime(monkeypatch)
    model = native_model(runtime)
    guide = selflift_guide(native_sparse_verified=False, native_sparse_blocks=99)
    run_short(model, guide)
    plan = execution_report.execution_plan(guide)
    settings = execution_report.sampling_settings(guide, model=model)
    for key in ("sol_attn_provider", "native_sparse_verified", "native_sparse_blocks",
                "native_sparse_total_blocks", "native_sparse_override_verified",
                "native_sparse_verification", "selflift_attention_requested_mode", "selflift_attention_route"):
        assert plan[key] == settings[key] == guide[key]
    assert settings["native_sparse_verification"] == "model_patch_protocol"
    stale = dict(guide, native_sparse_verified=True, native_sparse_blocks=99)
    actual = execution_report.sampling_settings(stale, model=Model())
    assert actual["native_sparse_verified"] is False and actual["native_sparse_blocks"] == 0
    report = execution_report.new_report(guide)
    report["sampling_settings"] = settings
    output = {}
    execution_report.attach_report(output, report)
    inherited = execution_report.new_report({}, output)["sampling_settings"]
    assert inherited["native_sparse_verified"] is True
    assert inherited["selflift_attention_route"] == "global_stream"


def test_short_exact_fallback_rejects_instead_of_dropping_an_upstream_override(monkeypatch):
    calls = []
    monkeypatch.setattr(performance, "_free_vram_bytes", lambda: None)
    monkeypatch.setattr(performance, "apply_h3_reuse_attention", lambda *args, **kwargs: calls.append(args) or args[0])
    model = legacy_override_model()
    original = model.model_options["transformer_options"]["optimized_attention_override"]
    with pytest.raises(RuntimeError, match="attention|override"):
        performance._apply_minimax_reuse_attention(model, selflift_guide())
    assert calls == []
    assert model.model_options["transformer_options"]["optimized_attention_override"] is original


@pytest.mark.parametrize("name", ["diffusion_model.blocks.0.forward", "diffusion_model.blocks.1.attn.forward"])
def test_native_inspector_rejects_unknown_deferred_attention_object_patch(monkeypatch, name):
    runtime = native_runtime(monkeypatch)
    model = native_model(runtime)
    model.object_patches = {name: lambda *args, **kwargs: None}
    evidence = native_api().inspect_native_attention(model)
    assert evidence["native_sparse_verified"] is False
    with pytest.raises(RuntimeError, match="attention|forward"):
        native_api().apply_native_sol_attention(model, {})


@pytest.mark.parametrize("name", ["diffusion_model.blocks", "diffusion_model.blocks.1",
    "diffusion_model.blocks.1.attn", "diffusion_model.blocks.1.attn.qkv_proj",
    "diffusion_model.blocks.1.attn.q_norm", "diffusion_model.blocks.1.attn.forward.custom"])
@pytest.mark.parametrize("action", ["inspect", "native_apply", "exact_fallback"])
def test_deferred_structural_attention_replacements_are_rejected_and_preserved(monkeypatch, name, action):
    runtime = native_runtime(monkeypatch)
    model = native_model(runtime) if action == "inspect" else Model()
    runtime.calls.clear()
    replacement = object()
    model.object_patches = {name: replacement}
    original_network = model.get_model_object("diffusion_model")
    original_options = copy_dicts(model.model_options)
    fallback_calls = []
    monkeypatch.setattr(performance, "_free_vram_bytes", lambda: None)
    monkeypatch.setattr(performance, "apply_h3_reuse_attention",
                        lambda *args, **kwargs: fallback_calls.append(args) or args[0])
    if action == "inspect":
        evidence = native_api().inspect_native_attention(model)
        assert evidence["native_sparse_verified"] is False
        assert name in evidence["native_sparse_reason"]
        with pytest.raises(RuntimeError, match="attention|blocks"):
            native_api().apply_native_sol_attention(model, {})
    elif action == "native_apply":
        with pytest.raises(RuntimeError, match="attention|blocks"):
            performance._apply_sol_attention(model, selflift_guide())
    else:
        with pytest.raises(RuntimeError, match="attention|blocks"):
            performance._apply_minimax_reuse_attention(model, selflift_guide())
    assert model.object_patches == {name: replacement}
    assert model.object_patches[name] is replacement
    assert model.get_model_object("diffusion_model") is original_network
    assert model.model_options == original_options
    assert runtime.calls == fallback_calls == []


def test_visible_diffusion_model_and_mlp_chunk_patches_keep_native_compatibility(monkeypatch):
    runtime = native_runtime(monkeypatch)
    model = Model()
    replacement = types.SimpleNamespace(blocks=[object() for _ in range(3)])
    mlp_chunk = lambda *args, **kwargs: None
    model.object_patches = {"diffusion_model": replacement,
                           "diffusion_model.blocks.1.mlp.forward": mlp_chunk}
    assert model.get_model_object("diffusion_model") is replacement
    patched, evidence = native_api().apply_native_sol_attention(model, {})
    assert evidence["native_sparse_verified"] is True
    assert patched.get_model_object("diffusion_model") is replacement
    assert patched.object_patches["diffusion_model.blocks.1.mlp.forward"] is mlp_chunk
    assert native_api().inspect_native_attention(patched)["native_sparse_verified"] is True
