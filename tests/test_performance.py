import pytest

from nodes import performance
from nodes.performance import (
    MiniMaxH3AccelerationRouter,
    MiniMaxH3MemoryAwareSampler,
    MiniMaxH3PerformancePreset,
    MiniMaxH3SamplerRouter,
    MiniMaxH3SchedulerRouter,
    acceleration_plan,
    memory_policy,
    preset_values,
    sampler_name_for_guide,
    sampler_route,
    scheduler_plan,
)
from nodes.two_stage_assets import FL_STAGE1_LORA, FL_STAGE2_LORA, REF_STAGE_LORA


def test_fast_preset_exposes_four_step_sampling_contract():
    values = preset_values("fast_4step")
    assert values["steps"] == 4
    assert values["use_cache"] is True
    assert values["use_sage"] is True


@pytest.mark.parametrize("in_place", [False, True], ids=["clone", "in_place"])
def test_official_lora_stacks_same_patch_key_and_logs_positive_delta(monkeypatch, caplog, in_place):
    class FolderPaths:
        @staticmethod
        def get_full_path(category, name):
            assert category == "loras"
            if name == "minimax/adapter.safetensors":
                return f"/models/{name}"
            return None

        @staticmethod
        def get_filename_list(category):
            assert category == "loras"
            return ["minimax/adapter.safetensors"]

    class Model:
        def __init__(self, patches):
            self.patches = patches

    model = Model({"shared.weight": [("old",)]})
    loaded_model = model if in_place else Model({"shared.weight": [("old",), ("new",)]})
    calls = []

    class CoreLoader:
        def load_lora_model_only(self, received_model, name, strength):
            calls.append((received_model, name, strength))
            if in_place:
                received_model.patches["shared.weight"].append(("new",))
            return (loaded_model,)

    import sys
    monkeypatch.setitem(sys.modules, "folder_paths", FolderPaths)
    monkeypatch.setitem(sys.modules, "nodes", type("Nodes", (), {"LoraLoaderModelOnly": CoreLoader})())
    monkeypatch.setattr(performance, "_turbo_class", lambda *_: pytest.fail("_turbo_class must not be called"))

    caplog.set_level("INFO", logger="MiniMaxH3.DirectorPlus")
    before_count = performance._patch_entry_count(model)
    result = performance._load_lightx2v_lora(
        model,
        "adapter.safetensors",
        strength=0.75,
    )

    assert result is loaded_model
    assert calls == [(model, "minimax/adapter.safetensors", 0.75)]
    assert set(result.patches) == {"shared.weight"}
    assert result.patches["shared.weight"] == [("old",), ("new",)]
    assert performance._patch_entry_count(result) - before_count == 1
    log_message = next(record.getMessage() for record in caplog.records if "[H3 LoRA]" in record.getMessage())
    assert "官方加载成功" in log_message
    assert "minimax/adapter.safetensors" in log_message
    assert "strength=0.75" in log_message
    assert "patch_delta=1" in log_message


def test_official_lora_rejects_zero_patch_delta(monkeypatch):
    class FolderPaths:
        @staticmethod
        def get_full_path(category, name):
            assert category == "loras"
            if name == "minimax/adapter.safetensors":
                return f"/models/{name}"
            return None

        @staticmethod
        def get_filename_list(category):
            assert category == "loras"
            return ["minimax/adapter.safetensors"]

    class Model:
        patches = {"adapter": [object()]}

    class CoreLoader:
        def load_lora_model_only(self, model, name, strength):
            return (model,)

    import sys
    monkeypatch.setitem(sys.modules, "folder_paths", FolderPaths)
    monkeypatch.setitem(sys.modules, "nodes", type("Nodes", (), {"LoraLoaderModelOnly": CoreLoader})())

    with pytest.raises(
        performance.H3LoRAApplicationError,
        match=r"未应用任何模型补丁.*minimax/adapter\.safetensors",
    ):
        performance._load_lightx2v_lora(Model(), "adapter.safetensors")



def test_official_lora_wraps_unexpected_core_loader_exception(monkeypatch):
    class FolderPaths:
        @staticmethod
        def get_full_path(category, name):
            if name == "minimax/adapter.safetensors":
                return f"/models/{name}"
            return None

        @staticmethod
        def get_filename_list(category):
            assert category == "loras"
            return ["minimax/adapter.safetensors"]

    class Model:
        patches = {}

    class LoaderFailure(Exception):
        pass

    class BrokenCoreLoader:
        def load_lora_model_only(self, *args):
            raise LoaderFailure("corrupt safetensors payload")

    import sys
    monkeypatch.setitem(sys.modules, "folder_paths", FolderPaths)
    monkeypatch.setitem(sys.modules, "nodes", type("Nodes", (), {"LoraLoaderModelOnly": BrokenCoreLoader})())

    with pytest.raises(
        performance.H3LoRAApplicationError,
        match=r"官方 H3 Turbo LoRA 加载失败.*minimax/adapter\.safetensors",
    ):
        performance._load_lightx2v_lora(Model(), "adapter.safetensors")


def test_quality_preset_keeps_conservative_sampling():
    values = preset_values("quality")
    assert values["steps"] >= 10
    assert values["use_cache"] is False


def test_performance_node_reads_guide_preset():
    result = MiniMaxH3PerformancePreset().apply({"performance_preset": "reference_fast"})
    assert result[0] == 6
    assert result[1] is True


def test_performance_node_safely_downgrades_invalid_t2va_reference_preset():
    result = MiniMaxH3PerformancePreset().apply({
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "reference_fast",
    })
    assert result[0] == 20
    assert result[1] is False
    assert result[2] is False
    assert "稳定质量" in result[3]


def test_acceleration_plan_downgrades_invalid_t2va_reference_preset():
    plan = acceleration_plan({
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "reference_fast",
        "resolved_backend": "fl2va_model",
    })
    assert plan["preset"] == "quality"
    assert plan["use_turbo_lora"] is False


def test_fast_preset_falls_back_to_safe_steps_after_acceleration_failure():
    result = MiniMaxH3PerformancePreset().apply(
        {"performance_preset": "fast_4step"},
        acceleration_ready=False,
    )
    assert result[0] == 8


def test_low_vram_description_matches_disabled_cache():
    result = MiniMaxH3PerformancePreset().apply({"performance_preset": "low_vram"})
    assert "EasyCache" not in result[3]


@pytest.mark.parametrize("mode", ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"])
@pytest.mark.parametrize("preset", ["quality", "quality_sage", "quality_two_stage", "fast_4step", "reference_fast", "low_vram", "low_vram_two_stage", "custom"])
def test_every_mode_has_a_defined_performance_contract(mode, preset):
    backend = "ref2va_model" if mode == "REF2VA" else "fl2va_model"
    values = performance._runtime_preset_values(
        {"mode": mode, "resolved_backend": backend},
        preset,
    )
    plan = acceleration_plan({
        "mode": mode,
        "performance_preset": preset,
        "resolved_backend": backend,
    })

    assert values["steps"] >= 4
    assert values["use_sage"] is (preset in {"quality_sage", "fast_4step", "reference_fast", "low_vram"})
    expected_cache = preset in {"fast_4step", "reference_fast"} and not (mode == "T2VA" and preset == "fast_4step")
    assert values["use_cache"] is expected_cache
    assert plan["backend"] == backend
    expected_preset = "quality" if mode == "T2VA" and preset == "reference_fast" else preset
    assert plan["use_turbo_lora"] is (expected_preset in {"fast_4step", "quality_two_stage", "low_vram_two_stage"})
    if expected_preset in {"quality_two_stage", "low_vram_two_stage"}:
        assert plan["first_lora_name"] == (
            REF_STAGE_LORA if backend == "ref2va_model" else FL_STAGE1_LORA
        )
    else:
        assert plan["lora_name"] == (
            performance.REF2VA_TURBO_LORA_NAME if backend == "ref2va_model" else performance.TURBO_LORA_NAME
        )


def test_low_vram_preset_uses_dynamic_safe_policy_without_cache():
    values = preset_values("low_vram")
    assert values["clip_device"] == "dynamic"
    assert values["vae_device"] == "dynamic"
    assert values["fish_device"] == "cpu"
    assert values["use_cache"] is False


def test_quality_priority_preset_uses_dynamic_safe_policy_without_quality_changes():
    values = preset_values("quality_sage")

    assert values["steps"] == 20
    assert values["use_sage"] is True
    assert values["use_cache"] is False
    assert values["clip_device"] == "dynamic"
    assert values["vae_device"] == "dynamic"
    assert values["fish_device"] == "cpu"


def test_quality_two_stage_uses_exact_head_chunking_without_sage_or_cache():
    values = preset_values("quality_two_stage")

    assert values["steps"] == 8
    assert values["two_stage_split_step"] == 4
    assert values["use_sage"] is False
    assert values["use_cache"] is False
    assert values["use_head_chunking"] is True
    assert values["minimax_head_chunks"] == 8


def test_low_vram_two_stage_uses_trained_route_and_low_vram_policy():
    guide = {
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "low_vram_two_stage",
        "resolved_backend": "fl2va_model",
    }

    values = preset_values("low_vram_two_stage")
    plan = acceleration_plan(guide)

    assert values["steps"] == 8
    assert values["two_stage_split_step"] == 4
    assert values["two_stage_scale"] == pytest.approx(1.5)
    assert values["use_head_chunking"] is True
    assert values["minimax_head_chunks"] == 16
    assert values["use_sage"] is False
    assert values["use_cache"] is False
    assert plan["route"] == "trained_latent_fl"
    assert plan["use_turbo_lora"] is True
    assert sampler_name_for_guide(guide, "res_multistep") == "euler"
    assert scheduler_plan(guide)["split_step"] == 4


def test_low_vram_two_stage_fails_instead_of_silently_generating_blurry_single_pass():
    guide = {
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "low_vram_two_stage",
    }

    with pytest.raises(RuntimeError, match="低显存二采.*未就绪"):
        MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=False)


def test_quality_two_stage_applies_only_exact_low_vram_attention(monkeypatch):
    calls = []

    monkeypatch.setattr(
        performance,
        "_load_lightx2v_lora",
        lambda model, name, strength=1.0, low_vram=False: calls.append(
            ("lora", model, name, strength)
        ) or f"{model}:{name}:{strength}",
    )

    monkeypatch.setattr(
        performance,
        "_apply_minimax_reuse_attention",
        lambda model, guide: calls.append(("chunks", model, guide["performance_preset"])) or f"{model}:head-chunks",
        raising=False,
    )
    monkeypatch.setattr(
        performance,
        "_apply_sage_attention",
        lambda *args: (_ for _ in ()).throw(AssertionError("二采不得启用 Sage")),
    )

    guide = {
        "mode": "T2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "fl2va_model",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert len(result) == 4
    assert result[0].endswith(":head-chunks")
    assert result[3].endswith(":head-chunks")
    assert result[2] is True
    assert calls == [
        ("lora", "model", FL_STAGE1_LORA, 0.75),
        ("lora", "model", FL_STAGE2_LORA, 0.70),
        ("chunks", f"model:{FL_STAGE1_LORA}:0.75", "quality_two_stage"),
        ("chunks", f"model:{FL_STAGE2_LORA}:0.7", "quality_two_stage"),
    ]
    assert guide["head_chunking_applied"] is True
    assert guide["sage_applied"] is False


def test_quality_two_stage_head_chunk_failure_bypasses_second_pass(monkeypatch):
    monkeypatch.setattr(
        performance,
        "_load_lightx2v_lora",
        lambda model, name, strength=1.0, low_vram=False: f"{model}:{name}:{strength}",
    )
    monkeypatch.setattr(
        performance,
        "_apply_minimax_reuse_attention",
        lambda model, guide: (_ for _ in ()).throw(RuntimeError("KJ patch missing")),
        raising=False,
    )
    guide = {
        "mode": "T2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "fl2va_model",
    }

    acceleration = MiniMaxH3AccelerationRouter().apply("model", guide)
    preset = MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=acceleration[2])

    assert acceleration[0] == "model"
    assert acceleration[3] == "model"
    assert acceleration[2] is False
    assert guide["head_chunking_applied"] is False
    assert guide["two_stage_enabled"] is False
    assert guide["two_stage_split_step"] == 0
    assert preset[0] == 8
    assert "单采" in preset[3]


def test_t2va_fast_uses_official_h3_turbo_contract():
    guide = {
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "fast_4step",
        "resolved_backend": "fl2va_model",
    }
    assert performance.allowed_performance_presets("T2VA", "none") == (
        "quality", "quality_sage", "quality_two_stage", "fast_4step", "low_vram", "low_vram_two_stage"
    )
    assert acceleration_plan(guide)["use_turbo_lora"] is True
    assert acceleration_plan(guide)["lora_name"] == performance.TURBO_LORA_NAME
    assert performance._runtime_preset_values(guide, "fast_4step")["use_cache"] is False


@pytest.mark.parametrize("mode", ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"])
@pytest.mark.parametrize("voice_mode", ["none", "h3_reference", "fish_lock"])
def test_quality_priority_acceleration_keeps_native_quality_contract(mode, voice_mode):
    guide = {
        "mode": mode,
        "voice_mode": voice_mode,
        "performance_preset": "quality_sage",
        "resolved_backend": "ref2va_model" if voice_mode != "none" or mode == "REF2VA" else "fl2va_model",
    }

    values = performance._runtime_preset_values(guide, "quality_sage")
    plan = acceleration_plan(guide)

    assert values["steps"] == 20
    assert values["use_sage"] is True
    assert values["use_cache"] is False
    assert plan["use_turbo_lora"] is False
    assert sampler_route(guide) == "native"


def test_quality_priority_acceleration_applies_sage_without_cache_or_lora(monkeypatch):
    calls = []
    monkeypatch.setattr(performance, "_apply_sage_attention", lambda model, guide: calls.append("sage") or f"{model}:sage")
    monkeypatch.setattr(performance, "_apply_easy_cache", lambda model, guide: calls.append("cache") or f"{model}:cache")
    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda *args, **kwargs: calls.append("lora"))

    guide = {"mode": "T2VA", "performance_preset": "quality_sage", "resolved_backend": "fl2va_model"}
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert result[0] == "model:sage"
    assert calls == ["sage"]
    assert result[2] is True


def test_quality_priority_uses_h3_memory_efficient_sage_patch(monkeypatch):
    calls = []

    class MemoryEfficientSage:
        @staticmethod
        def execute(model):
            calls.append(("sage", model))
            return (f"{model}:h3sage",)

    class HeadChunkPatch:
        @staticmethod
        def execute(model, head_chunks):
            calls.append(("chunks", model, head_chunks))
            return (f"{model}:chunks",)

    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: MemoryEfficientSage)
    monkeypatch.setattr(performance, "_kj_minimax_class", lambda name: HeadChunkPatch)

    result = performance._apply_sage_attention(
        "model",
        {"performance_preset": "quality_sage"},
    )

    assert result == "model:h3sage:chunks"
    assert calls == [
        ("sage", "model"),
        ("chunks", "model:h3sage", 8),
    ]


def test_exact_low_vram_attention_uses_internal_reuse_patch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        performance,
        "apply_h3_reuse_attention",
        lambda model, head_chunks: calls.append((model, head_chunks)) or f"{model}:reuse",
        raising=False,
    )

    result = performance._apply_minimax_reuse_attention(
        "model",
        {"minimax_head_chunks": 8},
    )

    assert result == "model:reuse"
    assert calls == [("model", 8)]


def test_quality_two_stage_without_verified_patch_fails_closed():
    guide = {
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "quality_two_stage",
    }

    result = MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=None)

    assert result[0] == 8
    assert guide["two_stage_enabled"] is False
    assert guide["two_stage_split_step"] == 0
    assert guide["two_stage_scale"] == 1.0


def test_two_stage_acceleration_fallback_reconciles_single_stage_output_geometry():
    guide = {
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "quality_two_stage",
        "two_stage_enabled": True,
        "resolved_two_stage_route": "trained_latent_fl",
        "first_stage_width": 896,
        "first_stage_height": 512,
        "second_stage_width": 1344,
        "second_stage_height": 768,
        "postprocess_source_width": 1344,
        "postprocess_source_height": 768,
        "target_width": 2560,
        "target_height": 1440,
        "postprocess_mode": "rtx_vsr",
        "final_upscale_scale_x": 2560 / 1344,
        "final_upscale_scale_y": 1440 / 768,
        "final_upscale_scale": 2560 / 1344,
        "quality_basis": "H3 神经 latent 二采",
    }

    MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=False)

    assert guide["two_stage_enabled"] is False
    assert guide["two_stage_fallback"] is True
    assert guide["resolved_two_stage_route"] == "bypass"
    assert (guide["second_stage_width"], guide["second_stage_height"]) == (896, 512)
    assert (guide["postprocess_source_width"], guide["postprocess_source_height"]) == (896, 512)
    assert guide["final_upscale_scale_x"] == pytest.approx(2560 / 896)
    assert guide["final_upscale_scale_y"] == pytest.approx(1440 / 512)
    assert guide["final_upscale_scale"] == pytest.approx(2560 / 896)
    assert guide["quality_basis"] == "H3 原生（训练型二采不可用，已回退）"
    assert scheduler_plan(guide) == {
        "scheduler": "simple",
        "steps": 8,
        "split_step": 0,
        "refine_reference_tail": False,
    }


def test_fl_two_stage_uses_u17_model_lora_contract():
    guide = {
        "performance_preset": "quality_two_stage",
        "resolved_backend": "fl2va_model",
        "voice_mode": "none",
    }

    plan = acceleration_plan(guide)

    assert plan["first_lora_name"] == FL_STAGE1_LORA
    assert plan["first_lora_strength"] == 0.75
    assert plan["second_lora_name"] == FL_STAGE2_LORA
    assert plan["second_lora_strength"] == 0.70
    assert scheduler_plan(guide) == {
        "scheduler": "beta",
        "steps": 8,
        "split_step": 4,
        "refine_reference_tail": False,
    }


def test_reference_two_stage_uses_u16_model_lora_contract():
    guide = {
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "h3_reference",
    }

    plan = acceleration_plan(guide)

    assert plan["first_lora_name"] == REF_STAGE_LORA
    assert plan["first_lora_strength"] == 0.75
    assert plan["second_lora_name"] == REF_STAGE_LORA
    assert plan["second_lora_strength"] == 0.75
    assert scheduler_plan(guide) == {
        "scheduler": "simple",
        "steps": 8,
        "split_step": 4,
        "refine_reference_tail": True,
    }


def test_every_trained_two_stage_route_forces_euler():
    assert sampler_name_for_guide(
        {"performance_preset": "quality_two_stage"},
        "res_multistep",
    ) == "euler"


def test_scheduler_router_applies_reference_tail_refiner(monkeypatch):
    import sys
    import types
    import torch

    base = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.0])
    refined = torch.cat((base[:-1], torch.tensor([0.1, 0.0])))
    calls = []

    class BasicScheduler:
        @classmethod
        def execute(cls, model, scheduler, steps, denoise):
            calls.append(("scheduler", model, scheduler, steps, denoise))
            return types.SimpleNamespace(result=(base,))

    monkeypatch.setitem(
        sys.modules,
        "comfy_extras.nodes_custom_sampler",
        types.SimpleNamespace(BasicScheduler=BasicScheduler),
    )
    monkeypatch.setattr(
        performance,
        "_apply_reference_sigma_refiner",
        lambda sigmas: calls.append(("refiner", sigmas)) or refined,
        raising=False,
    )

    result = MiniMaxH3SchedulerRouter().route(
        "model",
        8,
        {
            "performance_preset": "quality_two_stage",
            "resolved_backend": "ref2va_model",
            "voice_mode": "h3_reference",
        },
    )

    assert torch.equal(result[0], refined)
    assert calls[0] == ("scheduler", "model", "simple", 8, 1.0)
    assert calls[1][0] == "refiner"


def test_low_vram_uses_h3_memory_efficient_sage_patch_with_more_head_chunks(monkeypatch):
    calls = []

    class MemoryEfficientSage:
        @staticmethod
        def execute(model):
            calls.append(("sage", model))
            return (f"{model}:h3sage",)

    class HeadChunkPatch:
        @staticmethod
        def execute(model, head_chunks):
            calls.append(("chunks", model, head_chunks))
            return (f"{model}:chunks",)

    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: MemoryEfficientSage)
    monkeypatch.setattr(performance, "_kj_minimax_class", lambda name: HeadChunkPatch)

    result = performance._apply_sage_attention(
        "model",
        {"performance_preset": "low_vram"},
    )

    assert result == "model:h3sage:chunks"
    assert calls == [
        ("sage", "model"),
        ("chunks", "model:h3sage", 16),
    ]


def test_quality_priority_acceleration_failure_keeps_twenty_steps(monkeypatch):
    monkeypatch.setattr(
        performance,
        "_apply_sage_attention",
        lambda model, guide: (_ for _ in ()).throw(RuntimeError("sage unavailable")),
        raising=False,
    )
    guide = {"mode": "T2VA", "performance_preset": "quality_sage", "resolved_backend": "fl2va_model"}
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert result[0] == "model"
    assert result[2] is False
    assert MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=result[2])[0] == 20


def test_reference_fast_applies_sage_and_easycache_on_routed_model(monkeypatch):
    calls = []

    def apply_sage(model, guide):
        calls.append("sage")
        return f"{model}:sage"

    def apply_cache(model, guide):
        calls.append("cache")
        return f"{model}:cache"

    monkeypatch.setattr(performance, "_apply_sage_attention", apply_sage, raising=False)
    monkeypatch.setattr(performance, "_apply_easy_cache", apply_cache, raising=False)

    guide = {
        "mode": "REF2VA",
        "performance_preset": "reference_fast",
        "resolved_backend": "ref2va_model",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert result[0] == "model:sage:cache"
    assert calls == ["sage", "cache"]
    assert guide["sage_applied"] is True
    assert guide["easycache_applied"] is True
    assert result[2] is True


def test_reference_fast_uses_safe_steps_when_requested_acceleration_fails(monkeypatch):
    monkeypatch.setattr(
        performance,
        "_apply_sage_attention",
        lambda model, guide: (_ for _ in ()).throw(RuntimeError("sage unavailable")),
        raising=False,
    )
    guide = {
        "performance_preset": "reference_fast",
        "resolved_backend": "ref2va_model",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)
    steps = MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=result[2])[0]

    assert result[0] == "model"
    assert result[2] is False
    assert guide["sage_applied"] is False
    assert steps == 8


@pytest.mark.parametrize("preset", ["low_vram", "low_vram_two_stage", "quality_sage"])
def test_memory_policy_restores_comfy_state(monkeypatch, preset):
    class VramState:
        NORMAL_VRAM = "normal"
        LOW_VRAM = "low"

    mm = type("ModelManagement", (), {"vram_state": VramState.NORMAL_VRAM, "VRAMState": VramState})()
    import sys
    monkeypatch.setitem(sys.modules, "comfy", type("Comfy", (), {"model_management": mm})())
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    with memory_policy({"performance_preset": preset}):
        assert mm.vram_state == VramState.LOW_VRAM
    assert mm.vram_state == VramState.NORMAL_VRAM


def test_two_stage_memory_policy_keeps_comfy_native_vram_state(monkeypatch):
    class VramState:
        NORMAL_VRAM = "normal"
        LOW_VRAM = "low"

    mm = type("ModelManagement", (), {"vram_state": VramState.NORMAL_VRAM, "VRAMState": VramState})()
    import sys
    monkeypatch.setitem(sys.modules, "comfy", type("Comfy", (), {"model_management": mm})())
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    with memory_policy({"performance_preset": "quality_two_stage"}):
        assert mm.vram_state == VramState.NORMAL_VRAM


def test_non_two_stage_preset_explicitly_bypasses_refinement():
    guide = {"mode": "FL2VA", "voice_mode": "none", "performance_preset": "quality"}
    MiniMaxH3PerformancePreset().apply(guide)
    assert guide["two_stage_enabled"] is False
    assert guide["two_stage_split_step"] == 0
    assert guide["two_stage_scale"] == 1.0
    assert guide["two_stage_status"] == "旁路"


def test_memory_aware_sampler_exposes_guide_input():
    inputs = MiniMaxH3MemoryAwareSampler.INPUT_TYPES()
    assert "guide" in inputs["optional"]


def test_fl2va_fast_loads_registered_adapter(monkeypatch):
    model = object()
    accelerated = object()
    calls = []
    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda value, name, **kwargs: calls.append((value, name)) or accelerated)
    monkeypatch.setattr(performance, "_apply_sage_attention", lambda value, guide: value)
    monkeypatch.setattr(performance, "_apply_easy_cache", lambda value, guide: value)

    result = MiniMaxH3AccelerationRouter().apply(
        model,
        {"performance_preset": "fast_4step", "resolved_backend": "fl2va_model"},
    )

    assert result[0] is accelerated
    assert "已启用" in result[1]
    assert calls == [(model, performance.TURBO_LORA_NAME)]


def test_ref2va_fast_loads_official_ref2va_lora(monkeypatch):
    model = object()
    accelerated = object()
    calls = []
    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda value, name, **kwargs: calls.append((value, name)) or accelerated)
    monkeypatch.setattr(performance, "_apply_sage_attention", lambda value, guide: value)
    monkeypatch.setattr(performance, "_apply_easy_cache", lambda value, guide: value)

    result = MiniMaxH3AccelerationRouter().apply(
        model,
        {"performance_preset": "fast_4step", "resolved_backend": "ref2va_model"},
    )

    assert result[0] is accelerated
    assert "REF2VA" in result[1]
    assert calls == [(model, performance.REF2VA_TURBO_LORA_NAME)]


def test_fast_preset_uses_official_ref2va_four_step_contract():
    values = preset_values("fast_4step", backend="ref2va_model")
    assert values["steps"] == 4
    assert values["use_sage"] is True
    assert values["use_cache"] is True
    assert values["use_turbo_sampler"] is False


def test_ref2va_plan_uses_adapter_and_native_sampler():
    guide = {"performance_preset": "fast_4step", "resolved_backend": "ref2va_model"}
    plan = acceleration_plan(guide)
    assert plan["use_turbo_lora"] is True
    assert plan["use_turbo_sampler"] is False
    assert plan["lora_name"] == performance.REF2VA_TURBO_LORA_NAME
    assert sampler_route(guide) == "native"


def test_voice_reference_backend_gets_ref2va_acceleration():
    guide = {
        "mode": "FL2VA",
        "voice_mode": "h3_reference",
        "performance_preset": "fast_4step",
        "resolved_backend": "ref2va_model",
    }
    plan = acceleration_plan(guide)
    assert plan["lora_name"] == performance.REF2VA_TURBO_LORA_NAME
    assert plan["use_turbo_lora"] is True


def test_fast_fl2va_routes_to_h3_turbo_sampler():
    assert sampler_route({"performance_preset": "fast_4step", "resolved_backend": "fl2va_model"}) == "native"
    assert sampler_route({"performance_preset": "fast_4step", "resolved_backend": "ref2va_model"}) == "native"


@pytest.mark.parametrize("mode", ["I2VA", "FL2VA", "L2VA"])
def test_fast_fl2va_modes_share_the_same_checked_turbo_contract(mode):
    guide = {"mode": mode, "performance_preset": "fast_4step", "resolved_backend": "fl2va_model"}
    plan = acceleration_plan(guide)
    assert plan["use_turbo_lora"] is True
    assert plan["use_turbo_sampler"] is False
    assert sampler_route(guide) == "native"


def test_turbo_sampler_is_disabled_when_lora_falls_back():
    guide = {
        "mode": "I2VA",
        "performance_preset": "fast_4step",
        "resolved_backend": "fl2va_model",
        "turbo_lora_applied": False,
    }
    assert sampler_route(guide) == "native"


def test_sampler_router_falls_back_if_turbo_node_is_unavailable(monkeypatch):
    guide = {"mode": "FL2VA", "performance_preset": "fast_4step", "resolved_backend": "fl2va_model", "turbo_lora_applied": True}
    monkeypatch.setattr(performance, "_turbo_class", lambda _name: (_ for _ in ()).throw(RuntimeError("node unavailable")))
    import sys
    samplers = type("Samplers", (), {"sampler_object": staticmethod(lambda name: ("native", name))})()
    monkeypatch.setitem(sys.modules, "comfy", type("Comfy", (), {"samplers": samplers})())
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)

    result = MiniMaxH3SamplerRouter().route("res_multistep", guide)
    assert result == (("native", "euler"),)
    assert guide["turbo_sampler_applied"] is False


def test_acceleration_router_falls_back_without_leaving_turbo_sampler_enabled(monkeypatch):
    model = object()

    def fail(*_args, **_kwargs):
        raise AttributeError("incompatible H3 model wrapper")

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fail)
    guide = {
        "mode": "I2VA",
        "performance_preset": "fast_4step",
        "resolved_backend": "fl2va_model",
        "turbo_sampler_applied": True,
    }

    result = MiniMaxH3AccelerationRouter().apply(model, guide)
    assert result[0] is model
    assert guide["turbo_lora_applied"] is False
    assert guide["turbo_sampler_applied"] is False
    assert "incompatible H3 model wrapper" in guide["turbo_lora_error"]
    assert sampler_route(guide) == "native"
    assert "回退" in result[1]


def test_fast_route_stops_when_official_lora_has_no_verified_patches(monkeypatch):
    def fail(*_args, **_kwargs):
        raise performance.H3LoRAApplicationError("官方 LoRA 未应用任何模型补丁")

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fail)
    guide = {
        "mode": "FL2VA",
        "performance_preset": "fast_4step",
        "resolved_backend": "fl2va_model",
        "turbo_lora_error": "old failure",
        "turbo_sampler_applied": True,
        "two_stage_enabled": True,
        "two_stage_status": "待执行",
        "two_stage_split_step": 4,
        "two_stage_scale": 1.5,
        "first_lora_name": "old-first",
        "second_lora_name": "old-second",
        "resolved_two_stage_route": "old-route",
    }

    with pytest.raises(performance.H3LoRAApplicationError, match="未应用任何模型补丁"):
        MiniMaxH3AccelerationRouter().apply(object(), guide)

    assert guide["turbo_lora_applied"] is False
    assert guide["turbo_sampler_applied"] is False
    assert guide["two_stage_enabled"] is False
    assert guide["two_stage_status"] == "旁路"
    assert guide["two_stage_split_step"] == 0
    assert guide["two_stage_scale"] == 1.0
    assert "first_lora_name" not in guide
    assert "second_lora_name" not in guide
    assert guide["resolved_two_stage_route"] == "bypass"
    assert "未应用任何模型补丁" in guide["turbo_lora_error"]


def test_fast_route_retries_official_lora_after_a_previous_failure(monkeypatch):
    calls = []

    def fail(*_args, **_kwargs):
        calls.append("lora")
        raise performance.H3LoRAApplicationError("官方 LoRA 未应用任何模型补丁")

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fail)
    guide = {
        "mode": "FL2VA",
        "performance_preset": "fast_4step",
        "resolved_backend": "fl2va_model",
    }

    for _ in range(2):
        with pytest.raises(performance.H3LoRAApplicationError, match="未应用任何模型补丁"):
            MiniMaxH3AccelerationRouter().apply(object(), guide)

    assert calls == ["lora", "lora"]


@pytest.mark.parametrize("preset", ["quality", "quality_sage", "reference_fast", "low_vram"])
def test_non_turbo_routes_refresh_resolved_route_without_changing_request_fields(monkeypatch, preset):
    def unexpected_loader(*_args, **_kwargs):
        pytest.fail("non-Turbo presets must not load an official LoRA")

    monkeypatch.setattr(performance, "_load_lightx2v_lora", unexpected_loader)
    monkeypatch.setattr(performance, "_apply_sage_attention", lambda model, _guide: model)
    monkeypatch.setattr(performance, "_apply_easy_cache", lambda model, _guide: model)
    guide = {
        "mode": "FL2VA",
        "voice_mode": "none",
        "performance_preset": preset,
        "resolved_backend": "fl2va_model",
        "turbo_lora_error": "old failure",
        "resolved_two_stage_route": "old-route",
    }
    request_fields = {
        key: guide[key]
        for key in ("mode", "voice_mode", "performance_preset", "resolved_backend")
    }

    MiniMaxH3AccelerationRouter().apply(object(), guide)

    assert guide["resolved_two_stage_route"] == "bypass"
    assert {key: guide[key] for key in request_fields} == request_fields


def test_two_stage_route_stops_when_official_lora_has_no_verified_patches(monkeypatch):
    def fail(*_args, **_kwargs):
        raise performance.H3LoRAApplicationError("官方 LoRA 未应用任何模型补丁")

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fail)
    guide = {
        "mode": "T2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "fl2va_model",
        "turbo_lora_error": "old failure",
        "turbo_sampler_applied": True,
        "two_stage_enabled": True,
        "two_stage_status": "待执行",
        "two_stage_split_step": 4,
        "two_stage_scale": 1.5,
        "first_lora_name": "old-first",
        "second_lora_name": "old-second",
        "resolved_two_stage_route": "old-route",
    }

    with pytest.raises(performance.H3LoRAApplicationError, match="未应用任何模型补丁"):
        MiniMaxH3AccelerationRouter().apply(object(), guide)

    assert guide["turbo_lora_applied"] is False
    assert guide["turbo_sampler_applied"] is False
    assert guide["two_stage_enabled"] is False
    assert guide["two_stage_status"] == "旁路"
    assert guide["two_stage_split_step"] == 0
    assert guide["two_stage_scale"] == 1.0
    assert "first_lora_name" not in guide
    assert "second_lora_name" not in guide
    assert guide["resolved_two_stage_route"] == "trained_latent_fl"
    assert "未应用任何模型补丁" in guide["turbo_lora_error"]


def test_two_stage_route_retries_official_lora_and_refreshes_route(monkeypatch):
    calls = []

    def fail(*_args, **_kwargs):
        calls.append("lora")
        raise performance.H3LoRAApplicationError("官方 LoRA 未应用任何模型补丁")

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fail)
    guide = {
        "mode": "T2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "fl2va_model",
    }
    routes = []

    for _ in range(2):
        with pytest.raises(performance.H3LoRAApplicationError, match="未应用任何模型补丁"):
            MiniMaxH3AccelerationRouter().apply(object(), guide)
        routes.append(guide["resolved_two_stage_route"])

    assert calls == ["lora", "lora"]
    assert routes == ["trained_latent_fl", "trained_latent_fl"]


def test_sampler_router_exposes_valid_sampler_combo():
    sampler_spec = MiniMaxH3SamplerRouter.INPUT_TYPES()["required"]["sampler_name"]
    assert sampler_spec[0]
    assert sampler_spec[1]["default"] == "res_multistep"


def test_ref2va_fast_sampler_router_forces_official_euler(monkeypatch):
    guide = {"performance_preset": "fast_4step", "resolved_backend": "ref2va_model"}
    import sys
    samplers = type("Samplers", (), {"sampler_object": staticmethod(lambda name: name)})()
    monkeypatch.setitem(sys.modules, "comfy", type("Comfy", (), {"samplers": samplers})())
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    assert MiniMaxH3SamplerRouter().route("res_multistep", guide) == ("euler",)


def test_fl2va_fast_sampler_router_also_forces_official_euler(monkeypatch):
    guide = {"performance_preset": "fast_4step", "resolved_backend": "fl2va_model"}
    import sys
    samplers = type("Samplers", (), {"sampler_object": staticmethod(lambda name: name)})()
    monkeypatch.setitem(sys.modules, "comfy", type("Comfy", (), {"samplers": samplers})())
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    assert MiniMaxH3SamplerRouter().route("res_multistep", guide) == ("euler",)
