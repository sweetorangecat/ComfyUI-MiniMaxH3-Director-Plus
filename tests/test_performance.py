import re

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
from nodes.two_stage_assets import FL_STAGE1_LORA, FL_STAGE2_LORA, V4_TURBO_LORA, V4_TURBO_LORA_PRUNED, resolve_two_stage_route


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


def test_official_lora_uses_h3_loader_when_core_loader_matches_zero(monkeypatch):
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
        def __init__(self, patches=None, injections=None):
            self.patches = patches or {"adapter": [object()]}
            self.injections = injections or {}

    class CoreLoader:
        def load_lora_model_only(self, model, name, strength):
            return (model,)

    class H3Loader:
        def apply_lora(self, model, name, strength, low_vram=False):
            assert name == "minimax/adapter.safetensors"
            assert strength == 1.0
            assert low_vram is False
            return (Model(injections={"bypass_lora": [object()]}),)

    import sys
    monkeypatch.setitem(sys.modules, "folder_paths", FolderPaths)
    monkeypatch.setitem(sys.modules, "nodes", type("Nodes", (), {"LoraLoaderModelOnly": CoreLoader})())
    monkeypatch.setattr(performance, "_turbo_class", lambda name: H3Loader if name == "MiniMaxH3TurboLoRA" else pytest.fail(name))

    result = performance._load_lightx2v_lora(Model(), "adapter.safetensors")

    assert performance._lora_effect_count(result) > performance._lora_effect_count(Model())


def test_v4_lora_falls_back_to_pruned_comfyui_variant(monkeypatch):
    class FolderPaths:
        @staticmethod
        def get_full_path(category, name):
            assert category == "loras"
            if name == "minimax/" + V4_TURBO_LORA_PRUNED:
                return f"/models/{name}"
            return None

        @staticmethod
        def get_filename_list(category):
            assert category == "loras"
            return ["minimax/" + V4_TURBO_LORA_PRUNED]

    class Model:
        def __init__(self, patches):
            self.patches = patches

    model = Model({"shared.weight": [("old",)]})
    loaded_model = Model({"shared.weight": [("old",), ("new",)]})
    calls = []

    class CoreLoader:
        def load_lora_model_only(self, received_model, name, strength):
            calls.append((received_model, name, strength))
            return (loaded_model,)

    import sys
    monkeypatch.setitem(sys.modules, "folder_paths", FolderPaths)
    monkeypatch.setitem(sys.modules, "nodes", type("Nodes", (), {"LoraLoaderModelOnly": CoreLoader})())
    monkeypatch.setattr(performance, "_turbo_class", lambda *_: pytest.fail("_turbo_class must not be called"))

    result = performance._load_lightx2v_lora(model, V4_TURBO_LORA, strength=1.0)

    assert result is loaded_model
    assert calls == [(model, "minimax/" + V4_TURBO_LORA_PRUNED, 1.0)]


def test_v4_lora_prefers_pruned_variant_when_both_files_exist(monkeypatch):
    class FolderPaths:
        @staticmethod
        def get_full_path(category, name):
            assert category == "loras"
            return f"/models/{name}"

        @staticmethod
        def get_filename_list(category):
            assert category == "loras"
            return ["minimax/" + V4_TURBO_LORA, "minimax/" + V4_TURBO_LORA_PRUNED]

    class Model:
        def __init__(self, patches):
            self.patches = patches

    model = Model({"shared.weight": [("old",)]})
    loaded_model = Model({"shared.weight": [("old",), ("new",)]})
    calls = []

    class CoreLoader:
        def load_lora_model_only(self, received_model, name, strength):
            calls.append((received_model, name, strength))
            return (loaded_model,)

    import sys
    monkeypatch.setitem(sys.modules, "folder_paths", FolderPaths)
    monkeypatch.setitem(sys.modules, "nodes", type("Nodes", (), {"LoraLoaderModelOnly": CoreLoader})())
    monkeypatch.setattr(performance, "_turbo_class", lambda *_: pytest.fail("_turbo_class must not be called"))

    result = performance._load_lightx2v_lora(model, V4_TURBO_LORA, strength=1.0)

    # Native-path pruned conversion must win over the original file even
    # though the original resolves too; the original would drag in the slow
    # H3-specific runtime-injection loader.
    assert result is loaded_model
    assert calls == [(model, V4_TURBO_LORA_PRUNED, 1.0)]


def test_v4_lora_missing_error_lists_both_file_names(monkeypatch):
    class FolderPaths:
        @staticmethod
        def get_full_path(category, name):
            return None

        @staticmethod
        def get_filename_list(category):
            return []

    import sys
    monkeypatch.setitem(sys.modules, "folder_paths", FolderPaths)

    with pytest.raises(
        performance.H3LoRAApplicationError,
        match=rf"{re.escape(V4_TURBO_LORA_PRUNED)} 或 {re.escape(V4_TURBO_LORA)}",
    ):
        performance._load_lightx2v_lora(object(), V4_TURBO_LORA)



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


# Public legacy-preset migration happens in schema; this table covers only the
# defensive quality fallback for guides that bypass schema normalization.
PERFORMANCE_DEFENSIVE_FALLBACKS = {
    ("T2VA", "reference_fast"): "quality",
    ("I2VA", "reference_fast"): "quality",
    ("FL2VA", "reference_fast"): "quality",
    ("L2VA", "reference_fast"): "quality",
    ("REF2VA", "low_vram_two_stage"): "low_vram",
    ("REF2VA", "reference_fast"): "quality",
    ("REF2VA", "fl_quality_fast_v4"): "quality",
}


@pytest.mark.parametrize("mode", ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"])
@pytest.mark.parametrize("preset", ["quality", "quality_sage", "quality_two_stage", "fl_quality_fast_v4", "fast_4step", "reference_fast", "low_vram", "low_vram_two_stage", "custom"])
def test_every_mode_has_a_defined_performance_contract(mode, preset):
    backend = "ref2va_model" if mode == "REF2VA" else "fl2va_model"
    values = performance._runtime_preset_values(
        {"mode": mode, "resolved_backend": backend},
        preset,
    )
    guide = {
        "mode": mode,
        "performance_preset": preset,
        "resolved_backend": backend,
    }

    plan = acceleration_plan(guide)

    assert values["steps"] >= 4
    assert values["use_sage"] is (preset in {"quality_sage", "fast_4step", "reference_fast", "low_vram"})
    expected_cache = preset in {"fast_4step", "reference_fast"} and not (mode == "T2VA" and preset == "fast_4step")
    assert values["use_cache"] is expected_cache
    assert plan["backend"] == backend
    expected_preset = PERFORMANCE_DEFENSIVE_FALLBACKS.get((mode, preset), preset)
    assert plan["preset"] == expected_preset
    assert plan["use_turbo_lora"] is (
        expected_preset in {"fast_4step", "fl_quality_fast_v4", "quality_two_stage", "low_vram_two_stage"}
    )
    if expected_preset != preset:
        safe_result = MiniMaxH3PerformancePreset().apply({
            "mode": mode,
            "performance_preset": preset,
            "resolved_backend": backend,
        })
        assert plan["route"] == "bypass"
        assert safe_result[1] is (expected_preset in {"quality_sage", "low_vram"})
        assert safe_result[2] is False
    if expected_preset in {"quality_two_stage", "low_vram_two_stage"}:
        assert plan["first_lora_name"] == (
            V4_TURBO_LORA if backend == "ref2va_model" else FL_STAGE1_LORA
        )
    else:
        assert plan["lora_name"] == (
            performance.FL_V4_LORA_NAME
            if expected_preset == "fl_quality_fast_v4"
            else performance.REF2VA_TURBO_LORA_NAME if backend == "ref2va_model" else performance.TURBO_LORA_NAME
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


def test_fl_v4_quality_fast_is_an_eight_step_single_pass_contract():
    values = preset_values("fl_quality_fast_v4", backend="fl2va_model")
    plan = acceleration_plan({
        "mode": "FL2VA",
        "voice_mode": "none",
        "performance_preset": "fl_quality_fast_v4",
        "resolved_backend": "fl2va_model",
    })
    assert values["steps"] == 8
    assert values["use_sage"] is False
    assert values["use_cache"] is False
    assert values["use_turbo_sampler"] is False
    assert values.get("two_stage_split_step", 0) == 0
    assert plan["use_turbo_lora"] is True
    assert plan["lora_name"] == performance.FL_V4_LORA_NAME
    assert plan["lora_strength"] == pytest.approx(1.0)
    assert sampler_name_for_guide({**plan, "performance_preset": "fl_quality_fast_v4"}, "res_multistep") == "euler"


def test_reference_quality_native_is_twenty_step_sage_without_turbo():
    values = preset_values("ref_quality_native", backend="ref2va_model")
    plan = acceleration_plan({
        "mode": "REF2VA",
        "voice_mode": "h3_reference",
        "performance_preset": "ref_quality_native",
        "resolved_backend": "ref2va_model",
    })
    assert values["steps"] == 20
    assert values["use_sage"] is True
    assert values["use_cache"] is False
    assert plan["use_turbo_lora"] is False


def test_reference_fast_is_official_ref2va_four_step_contract():
    values = preset_values("ref_fast_4step", backend="ref2va_model")
    plan = acceleration_plan({
        "mode": "REF2VA",
        "voice_mode": "h3_reference",
        "performance_preset": "ref_fast_4step",
        "resolved_backend": "ref2va_model",
    })
    assert values["steps"] == 4
    assert values["use_sage"] is False
    assert values["use_cache"] is False
    assert plan["use_turbo_lora"] is True
    assert plan["lora_name"] == performance.REF2VA_TURBO_LORA_NAME
    assert sampler_name_for_guide({**plan, "performance_preset": "ref_fast_4step"}, "res_multistep") == "euler"


def test_new_presets_reject_the_wrong_backend():
    assert acceleration_plan({
        "mode": "REF2VA", "voice_mode": "none",
        "performance_preset": "fl_quality_fast_v4", "resolved_backend": "ref2va_model",
    })["preset"] == "quality"
    assert acceleration_plan({
        "mode": "FL2VA", "voice_mode": "none",
        "performance_preset": "ref_quality_native", "resolved_backend": "fl2va_model",
    })["preset"] == "quality"


def test_quality_two_stage_uses_exact_head_chunking_without_sage_or_cache():
    values = preset_values("quality_two_stage")

    assert values["steps"] == 12
    assert values["two_stage_split_step"] == 8
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

    assert values["steps"] == 12
    assert values["two_stage_split_step"] == 8
    assert values["two_stage_scale"] == pytest.approx(1.5)
    assert values["use_head_chunking"] is True
    assert values["minimax_head_chunks"] == 16
    assert values["use_sage"] is False
    assert values["use_cache"] is False
    assert plan["route"] == "trained_latent_fl"
    assert plan["use_turbo_lora"] is True
    assert sampler_name_for_guide(guide, "res_multistep") == "euler"
    assert scheduler_plan(guide)["split_step"] == 8


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

    # 宿主机若真实注册了社区细节 LoRA，FL 路线现在也会命中它们；这些用例只对
    # 注意力补丁行为断言，屏蔽细节链解析以保持宿主无关的确定性。
    monkeypatch.setattr(
        performance, "resolve_registered_model_name", lambda category, name, comfy_root=None: None
    )
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


def test_quality_two_stage_patches_sage_attention_when_available(monkeypatch):
    calls = []

    class FakeSageNode:
        def execute(self, model):
            calls.append(("sage", model))
            return (f"{model}:sage",)

    monkeypatch.delenv("MMH3_TWO_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_SECOND_STAGE_SAGE", raising=False)
    # 宿主机若真实注册了社区细节 LoRA，FL 路线现在也会命中它们；这些用例只对
    # 注意力补丁行为断言，屏蔽细节链解析以保持宿主无关的确定性。
    monkeypatch.setattr(
        performance, "resolve_registered_model_name", lambda category, name, comfy_root=None: None
    )
    monkeypatch.setattr(
        performance,
        "_load_lightx2v_lora",
        lambda model, name, strength=1.0, low_vram=False: calls.append(
            ("lora", model, name, strength)
        ) or f"{model}:{name}:{strength}",
    )
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)
    monkeypatch.setattr(
        performance,
        "_apply_minimax_reuse_attention",
        lambda *args: (_ for _ in ()).throw(AssertionError("Sage 可用时不得回退分块注意力")),
        raising=False,
    )

    guide = {
        "mode": "T2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "fl2va_model",
        "first_stage_width": 544,
        "second_stage_width": 1088,
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert len(result) == 4
    assert result[0].endswith(":sage")
    assert result[3].endswith(":sage")
    assert result[2] is True
    assert calls == [
        ("lora", "model", FL_STAGE1_LORA, 0.75),
        ("lora", "model", FL_STAGE2_LORA, 0.70),
        ("sage", f"model:{FL_STAGE1_LORA}:0.75"),
        ("sage", f"model:{FL_STAGE2_LORA}:0.7"),
    ]
    assert guide["sage_applied"] is True
    assert guide["second_stage_sage_applied"] is True
    assert guide["head_chunking_applied"] is False
    assert guide["two_stage_scale"] == pytest.approx(2.0)
    assert "SageAttention" in result[1]


def test_quality_two_stage_second_stage_sage_can_be_disabled(monkeypatch):
    calls = []

    class FakeSageNode:
        def execute(self, model):
            calls.append(("sage", model))
            return (f"{model}:sage",)

    monkeypatch.setenv("MMH3_SECOND_STAGE_SAGE", "0")
    # 宿主机若真实注册了社区细节 LoRA，FL 路线现在也会命中它们；这些用例只对
    # 注意力补丁行为断言，屏蔽细节链解析以保持宿主无关的确定性。
    monkeypatch.setattr(
        performance, "resolve_registered_model_name", lambda category, name, comfy_root=None: None
    )
    monkeypatch.setattr(
        performance,
        "_load_lightx2v_lora",
        lambda model, name, strength=1.0, low_vram=False: calls.append(
            ("lora", model, name, strength)
        ) or f"{model}:{name}:{strength}",
    )
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)
    monkeypatch.setattr(
        performance,
        "_apply_minimax_reuse_attention",
        lambda model, guide: calls.append(("chunks", model, guide["performance_preset"])) or f"{model}:head-chunks",
        raising=False,
    )

    guide = {
        "mode": "T2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "fl2va_model",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert result[0].endswith(":sage")
    assert result[3].endswith(":head-chunks")
    assert result[2] is True
    assert calls == [
        ("lora", "model", FL_STAGE1_LORA, 0.75),
        ("lora", "model", FL_STAGE2_LORA, 0.70),
        ("sage", f"model:{FL_STAGE1_LORA}:0.75"),
        ("chunks", f"model:{FL_STAGE2_LORA}:0.7", "quality_two_stage"),
    ]
    assert guide["sage_applied"] is True
    assert guide["second_stage_sage_applied"] is False
    assert guide["head_chunking_applied"] is True


def test_two_stage_sage_can_be_fully_disabled(monkeypatch):
    calls = []

    class FakeSageNode:
        def execute(self, model):
            calls.append(("sage", model))
            return (f"{model}:sage",)

    monkeypatch.setenv("MMH3_TWO_STAGE_SAGE", "0")
    # 宿主机若真实注册了社区细节 LoRA，FL 路线现在也会命中它们；这些用例只对
    # 注意力补丁行为断言，屏蔽细节链解析以保持宿主无关的确定性。
    monkeypatch.setattr(
        performance, "resolve_registered_model_name", lambda category, name, comfy_root=None: None
    )
    monkeypatch.setattr(
        performance,
        "_load_lightx2v_lora",
        lambda model, name, strength=1.0, low_vram=False: calls.append(
            ("lora", model, name, strength)
        ) or f"{model}:{name}:{strength}",
    )
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)
    monkeypatch.setattr(
        performance,
        "_apply_minimax_reuse_attention",
        lambda model, guide: calls.append(("chunks", model, guide["performance_preset"])) or f"{model}:head-chunks",
        raising=False,
    )

    guide = {
        "mode": "T2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "fl2va_model",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert result[0].endswith(":head-chunks")
    assert result[3].endswith(":head-chunks")
    assert result[2] is True
    assert calls == [
        ("lora", "model", FL_STAGE1_LORA, 0.75),
        ("lora", "model", FL_STAGE2_LORA, 0.70),
        ("chunks", f"model:{FL_STAGE1_LORA}:0.75", "quality_two_stage"),
        ("chunks", f"model:{FL_STAGE2_LORA}:0.7", "quality_two_stage"),
    ]
    assert guide["sage_applied"] is False
    assert guide["second_stage_sage_applied"] is False
    assert guide["head_chunking_applied"] is True


def test_ref_two_stage_loads_separate_clone_for_exact_second_stage(monkeypatch):
    calls = []

    class FakeSageNode:
        def execute(self, model):
            calls.append(("sage", model))
            return (f"{model}:sage",)

    monkeypatch.setenv("MMH3_SECOND_STAGE_SAGE", "0")
    monkeypatch.setattr(
        performance, "resolve_registered_model_name", lambda category, name, comfy_root=None: None
    )

    def fake_lora(model, name, strength=1.0, low_vram=False):
        calls.append(("lora", model, name, strength))
        return f"lora-clone-{len(calls)}"

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fake_lora)
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)
    monkeypatch.setattr(
        performance,
        "_apply_minimax_reuse_attention",
        lambda model, guide: calls.append(("chunks", model)) or f"{model}:head-chunks",
        raising=False,
    )

    guide = {
        "mode": "REF2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "h3_reference",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert calls == [
        ("lora", "model", V4_TURBO_LORA, 1.0),
        ("lora", "model", V4_TURBO_LORA, 1.0),
        ("sage", "lora-clone-1"),
        ("chunks", "lora-clone-2"),
    ]
    assert result[0] == "lora-clone-1:sage"
    assert result[3] == "lora-clone-2:head-chunks"
    assert result[2] is True
    assert guide["sage_applied"] is True
    assert guide["second_stage_sage_applied"] is False
    assert guide["head_chunking_applied"] is True
    assert guide["detail_loras_applied"] == []


def test_ref_two_stage_shares_one_model_clone_by_default(monkeypatch):
    calls = []

    class FakeSageNode:
        def execute(self, model):
            calls.append(("sage", model))
            return (f"{model}:sage",)

    monkeypatch.delenv("MMH3_TWO_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_SECOND_STAGE_SAGE", raising=False)
    monkeypatch.setattr(
        performance, "resolve_registered_model_name", lambda category, name, comfy_root=None: None
    )

    def fake_lora(model, name, strength=1.0, low_vram=False):
        calls.append(("lora", model, name, strength))
        return f"lora-clone-{len(calls)}"

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fake_lora)
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)
    monkeypatch.setattr(
        performance,
        "_apply_minimax_reuse_attention",
        lambda *args: (_ for _ in ()).throw(AssertionError("共享模型全速模式不得回退分块注意力")),
        raising=False,
    )

    guide = {
        "mode": "REF2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "h3_reference",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert calls == [
        ("lora", "model", V4_TURBO_LORA, 1.0),
        ("sage", "lora-clone-1"),
    ]
    assert result[0] == result[3] == "lora-clone-1:sage"
    assert result[2] is True
    assert guide["sage_applied"] is True
    assert guide["second_stage_sage_applied"] is True
    assert guide["head_chunking_applied"] is False
    assert guide["detail_loras_applied"] == []


def test_ref_two_stage_applies_u22_detail_loras_when_registered(monkeypatch):
    calls = []

    class FakeSageNode:
        def execute(self, model):
            calls.append(("sage", model))
            return (f"{model}:sage",)

    monkeypatch.delenv("MMH3_TWO_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_SECOND_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_DETAIL_LORAS", raising=False)
    monkeypatch.delenv("MMH3_EXTRA_LORAS", raising=False)
    monkeypatch.setattr(
        performance, "resolve_registered_model_name", lambda category, name, comfy_root=None: name
    )

    def fake_lora(model, name, strength=1.0, low_vram=False):
        calls.append(("lora", model, name, strength))
        return f"lora-clone-{len(calls)}"

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fake_lora)
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)
    monkeypatch.setattr(
        performance,
        "_apply_minimax_reuse_attention",
        lambda *args: (_ for _ in ()).throw(AssertionError("默认全速模式不得回退分块注意力")),
        raising=False,
    )

    guide = {
        "mode": "REF2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "h3_reference",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert calls == [
        ("lora", "model", V4_TURBO_LORA, 1.0),
        ("lora", "lora-clone-1", "wushu_spatial_physics_v2_1000_pruned.safetensors", 0.3),
        ("lora", "lora-clone-2", "MysticXXX_MMH3-V1.safetensors", 0.5),
        ("lora", "lora-clone-3", "动作i连续性修复LORA.safetensors", 0.4),
        ("lora", "lora-clone-4", "MinimaxH3真实电影质感V1.0.safetensors", 0.5),
        ("sage", "lora-clone-5"),
    ]
    assert result[0] == result[3] == "lora-clone-5:sage"
    assert result[2] is True
    assert guide["detail_loras_applied"] == [
        "wushu_spatial_physics_v2_1000_pruned.safetensors@0.3",
        "MysticXXX_MMH3-V1.safetensors@0.5",
        "动作i连续性修复LORA.safetensors@0.4",
        "MinimaxH3真实电影质感V1.0.safetensors@0.5",
    ]
    assert "4 个社区 LoRA" in result[1]


def test_fingerprint_base_model_hashes_adaln_t_table():
    import hashlib

    import numpy as np

    table = np.arange(1025 * 8, dtype="<f4").reshape(1025, 8)

    class FakeTensor:
        def __init__(self, array):
            self._array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return self._array

    class FakeInner:
        def state_dict(self):
            return {"adaln_t_table": FakeTensor(table), "unrelated": FakeTensor(table)}

    class FakeModel:
        model = FakeInner()

    expected = hashlib.sha256(table.tobytes()).hexdigest()
    assert performance._fingerprint_base_model(FakeModel()) == expected
    assert performance._fingerprint_base_model("model") is None
    assert performance._fingerprint_base_model(object()) is None

    class BrokenInner:
        def state_dict(self):
            raise RuntimeError("no weights")

    class BrokenModel:
        model = BrokenInner()

    assert performance._fingerprint_base_model(BrokenModel()) is None


def test_ref_two_stage_extra_loras_fall_back_to_alternate_file_names(monkeypatch):
    calls = []
    present = {
        "wushu_spatial_physics_v2_1000_pruned.safetensors",
        "MysticXXX_MMH3-V1.safetensors",
        "动作连续性修复LORA.safetensors",
        "MinimaxH3真实电影质感V0.1.safetensors",
    }

    class FakeSageNode:
        def execute(self, model):
            calls.append(("sage", model))
            return (f"{model}:sage",)

    monkeypatch.delenv("MMH3_TWO_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_SECOND_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_DETAIL_LORAS", raising=False)
    monkeypatch.delenv("MMH3_EXTRA_LORAS", raising=False)
    monkeypatch.setattr(
        performance,
        "resolve_registered_model_name",
        lambda category, name, comfy_root=None: name if name in present else None,
    )

    def fake_lora(model, name, strength=1.0, low_vram=False):
        calls.append(("lora", model, name, strength))
        return f"lora-clone-{len(calls)}"

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fake_lora)
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)

    guide = {
        "mode": "REF2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "h3_reference",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert calls == [
        ("lora", "model", V4_TURBO_LORA, 1.0),
        ("lora", "lora-clone-1", "wushu_spatial_physics_v2_1000_pruned.safetensors", 0.3),
        ("lora", "lora-clone-2", "MysticXXX_MMH3-V1.safetensors", 0.5),
        ("lora", "lora-clone-3", "动作连续性修复LORA.safetensors", 0.4),
        ("lora", "lora-clone-4", "MinimaxH3真实电影质感V0.1.safetensors", 0.5),
        ("sage", "lora-clone-5"),
    ]
    assert result[2] is True
    assert guide["detail_loras_applied"] == [
        "wushu_spatial_physics_v2_1000_pruned.safetensors@0.3",
        "MysticXXX_MMH3-V1.safetensors@0.5",
        "动作连续性修复LORA.safetensors@0.4",
        "MinimaxH3真实电影质感V0.1.safetensors@0.5",
    ]


def test_ref_two_stage_extra_loras_can_be_disabled_independently(monkeypatch):
    calls = []

    class FakeSageNode:
        def execute(self, model):
            calls.append(("sage", model))
            return (f"{model}:sage",)

    monkeypatch.delenv("MMH3_TWO_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_SECOND_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_DETAIL_LORAS", raising=False)
    monkeypatch.setenv("MMH3_EXTRA_LORAS", "0")
    monkeypatch.setattr(
        performance, "resolve_registered_model_name", lambda category, name, comfy_root=None: name
    )

    def fake_lora(model, name, strength=1.0, low_vram=False):
        calls.append(("lora", model, name, strength))
        return f"lora-clone-{len(calls)}"

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fake_lora)
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)

    guide = {
        "mode": "REF2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "h3_reference",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert calls == [
        ("lora", "model", V4_TURBO_LORA, 1.0),
        ("lora", "lora-clone-1", "wushu_spatial_physics_v2_1000_pruned.safetensors", 0.3),
        ("lora", "lora-clone-2", "MysticXXX_MMH3-V1.safetensors", 0.5),
        ("sage", "lora-clone-3"),
    ]
    assert result[2] is True
    assert guide["detail_loras_applied"] == [
        "wushu_spatial_physics_v2_1000_pruned.safetensors@0.3",
        "MysticXXX_MMH3-V1.safetensors@0.5",
    ]


def test_ref_two_stage_detail_loras_can_be_disabled(monkeypatch):
    calls = []

    class FakeSageNode:
        def execute(self, model):
            calls.append(("sage", model))
            return (f"{model}:sage",)

    monkeypatch.delenv("MMH3_TWO_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_SECOND_STAGE_SAGE", raising=False)
    monkeypatch.setenv("MMH3_DETAIL_LORAS", "0")
    monkeypatch.setenv("MMH3_EXTRA_LORAS", "0")
    monkeypatch.setattr(
        performance, "resolve_registered_model_name", lambda category, name, comfy_root=None: name
    )

    def fake_lora(model, name, strength=1.0, low_vram=False):
        calls.append(("lora", model, name, strength))
        return f"lora-clone-{len(calls)}"

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fake_lora)
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)

    guide = {
        "mode": "REF2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "h3_reference",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert calls == [
        ("lora", "model", V4_TURBO_LORA, 1.0),
        ("sage", "lora-clone-1"),
    ]
    assert result[2] is True
    assert guide["detail_loras_applied"] == []


def test_quality_two_stage_second_stage_exact_failure_degrades_to_sage(monkeypatch):
    class FakeSageNode:
        def execute(self, model):
            return (f"{model}:sage",)

    monkeypatch.setenv("MMH3_SECOND_STAGE_SAGE", "0")
    monkeypatch.setattr(
        performance,
        "_load_lightx2v_lora",
        lambda model, name, strength=1.0, low_vram=False: f"{model}:{name}:{strength}",
    )
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)
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
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert result[0].endswith(":sage")
    assert result[3].endswith(":sage")
    assert result[2] is True
    assert guide["sage_applied"] is True
    assert guide["second_stage_sage_applied"] is True
    assert guide["head_chunking_applied"] is False
    assert "KJ patch missing" in guide["head_chunking_error"]


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
    assert preset[0] == 12
    assert "单采" in preset[3]


def test_t2va_fast_uses_official_h3_turbo_contract():
    guide = {
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "fast_4step",
        "resolved_backend": "fl2va_model",
    }
    assert performance.allowed_performance_presets("T2VA", "none") == (
        "smart_free_1080p", "quality", "quality_sage", "quality_two_stage", "fl_quality_fast_v4", "fast_4step", "low_vram", "low_vram_two_stage"
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

    assert result[0] == 12
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
        "steps": 12,
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
        "steps": 12,
        "split_step": 8,
        "refine_reference_tail": False,
    }


def test_ref_two_stage_uses_u22_v4_lora_contract():
    guide = {
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "h3_reference",
    }

    plan = acceleration_plan(guide)

    assert plan["route"] == "trained_latent_ref"
    assert plan["first_lora_name"] == V4_TURBO_LORA
    assert plan["first_lora_strength"] == 1.0
    assert plan["second_lora_name"] == V4_TURBO_LORA
    assert plan["second_lora_strength"] == 1.0
    assert scheduler_plan(guide) == {
        "scheduler": "beta",
        "steps": 12,
        "split_step": 8,
        "refine_reference_tail": False,
    }


@pytest.mark.parametrize("preset", ["quality_two_stage", "low_vram_two_stage"])
@pytest.mark.parametrize(
    ("entry", "expects_error"),
    [
        (acceleration_plan, False),
        (lambda guide: MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=True), False),
        (resolve_two_stage_route, True),
    ],
    ids=["acceleration_plan", "performance_apply", "two_stage_route"],
)
def test_reference_backend_two_stage_behavior_at_performance_entries(preset, entry, expects_error):
    guide = {
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": preset,
        "resolved_backend": "ref2va_model",
    }

    if expects_error:
        # resolve_two_stage_route still refuses the unvalidated 8GB variant.
        if preset == "low_vram_two_stage":
            assert entry(guide) == "trained_latent_ref"
        return
    result = entry(guide)
    if preset == "quality_two_stage":
        # U22 recipe: REF2VA quality two-stage is a live route now.
        assert guide.get("resolved_performance_preset", preset) == "quality_two_stage"
        if isinstance(result, dict):
            assert result["preset"] == "quality_two_stage"
            assert result["route"] == "trained_latent_ref"
    else:
        expected = "low_vram"
        assert guide["resolved_performance_preset"] == expected
        assert guide["two_stage_enabled"] is False
        if isinstance(result, dict):
            assert result["preset"] == expected


@pytest.mark.parametrize(
    "entry",
    [
        lambda guide: preset_values(guide["performance_preset"]),
        acceleration_plan,
        lambda guide: MiniMaxH3PerformancePreset().apply(guide),
        scheduler_plan,
        resolve_two_stage_route,
        lambda guide: memory_policy(guide).__enter__(),
    ],
    ids=["preset_values", "acceleration_plan", "performance_apply", "scheduler_plan", "two_stage_route", "memory_policy"],
)
def test_unresolved_smart_preset_is_rejected_at_every_performance_entry(entry):
    guide = {
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "smart_free_1080p",
        "resolved_backend": "fl2va_model",
    }

    with pytest.raises(
        (ValueError, RuntimeError),
        match="免费智能 1080p 必须先由导演台解析为具体性能预设",
    ):
        entry(guide)


@pytest.mark.parametrize("preset", ["quality_two_stage", "low_vram_two_stage"])
def test_base_backend_keeps_trained_two_stage_routes(preset):
    guide = {
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": preset,
        "resolved_backend": "fl2va_model",
    }

    assert resolve_two_stage_route(guide) == "trained_latent_fl"
    assert acceleration_plan(guide)["route"] == "trained_latent_fl"
    result = MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=True)
    assert result[0] == 12
    assert guide["two_stage_enabled"] is True


def test_every_trained_two_stage_route_forces_euler():
    assert sampler_name_for_guide(
        {"performance_preset": "quality_two_stage"},
        "res_multistep",
    ) == "euler"


def test_scheduler_router_runs_reference_two_stage_u22_schedule(monkeypatch):
    import sys
    import types

    class BasicScheduler:
        @staticmethod
        def execute(model, scheduler, steps, denoise):
            assert (model, scheduler, steps, denoise) == ("model", "beta", 12, 1.0)
            return ([("sigma", steps)],)

    custom_sampler = types.ModuleType("comfy_extras.nodes_custom_sampler")
    custom_sampler.BasicScheduler = BasicScheduler
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", custom_sampler)

    guide = {
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "h3_reference",
    }

    assert MiniMaxH3SchedulerRouter().route("model", 8, guide) == ([("sigma", 12)],)
    assert guide["resolved_performance_preset"] == "quality_two_stage"
    assert guide["two_stage_split_step"] == 8


def test_scheduler_router_allows_quality_sage_on_reference_backend(monkeypatch):
    import sys
    import types

    class BasicScheduler:
        @staticmethod
        def execute(model, scheduler, steps, denoise):
            assert (model, scheduler, steps, denoise) == ("model", "simple", 20, 1.0)
            return (["sigma"],)

    custom_sampler = types.ModuleType("comfy_extras.nodes_custom_sampler")
    custom_sampler.BasicScheduler = BasicScheduler
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", custom_sampler)

    guide = {
        "performance_preset": "quality_sage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "none",
    }

    assert MiniMaxH3SchedulerRouter().route("model", 20, guide) == (["sigma"],)
    assert guide["scheduler_name"] == "simple"


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


def test_reference_fast_bypassing_schema_is_safely_rejected(monkeypatch):
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

    assert result[0] == "model"
    assert calls == []
    assert guide["sage_applied"] is False
    assert guide["easycache_applied"] is False
    assert result[2] is False


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


def test_fast_route_falls_back_when_official_lora_has_no_verified_patches(monkeypatch):
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

    result = MiniMaxH3AccelerationRouter().apply(object(), guide)
    assert result[0] is not None
    assert result[2] is False

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
        result = MiniMaxH3AccelerationRouter().apply(object(), guide)
        assert result[2] is False

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


def test_two_stage_route_falls_back_when_official_lora_has_no_verified_patches(monkeypatch):
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

    result = MiniMaxH3AccelerationRouter().apply(object(), guide)
    assert result[2] is False

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
        result = MiniMaxH3AccelerationRouter().apply(object(), guide)
        assert result[2] is False
        routes.append(guide["resolved_two_stage_route"])

    assert calls == ["lora", "lora"]
    assert routes == ["bypass", "bypass"]


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


def test_ref_two_stage_community_lora_widget_switch(monkeypatch):
    calls = []

    class FakeSageNode:
        def execute(self, model):
            calls.append(("sage", model))
            return (f"{model}:sage",)

    monkeypatch.delenv("MMH3_TWO_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_SECOND_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_DETAIL_LORAS", raising=False)
    monkeypatch.delenv("MMH3_EXTRA_LORAS", raising=False)
    monkeypatch.setattr(
        performance, "resolve_registered_model_name", lambda category, name, comfy_root=None: name
    )

    def fake_lora(model, name, strength=1.0, low_vram=False):
        calls.append(("lora", name, strength))
        return f"clone-{len(calls)}"

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fake_lora)
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)

    def run(mode):
        calls.clear()
        guide = {
            "mode": "REF2VA",
            "performance_preset": "quality_two_stage",
            "resolved_backend": "ref2va_model",
            "voice_mode": "h3_reference",
        }
        MiniMaxH3AccelerationRouter().apply("model", guide, community_loras=mode)
        return [entry[1] for entry in calls if entry[0] == "lora"]

    assert run("仅 U22 细节链") == [
        V4_TURBO_LORA,
        "wushu_spatial_physics_v2_1000_pruned.safetensors",
        "MysticXXX_MMH3-V1.safetensors",
    ]
    assert run("全部关闭") == [V4_TURBO_LORA]
    assert run("全部自动叠加") == [
        V4_TURBO_LORA,
        "wushu_spatial_physics_v2_1000_pruned.safetensors",
        "MysticXXX_MMH3-V1.safetensors",
        "动作i连续性修复LORA.safetensors",
        "MinimaxH3真实电影质感V1.0.safetensors",
    ]


def test_acceleration_router_records_second_stage_noise_mode(monkeypatch):
    class FakeSageNode:
        def execute(self, model):
            return (f"{model}:sage",)

    monkeypatch.delenv("MMH3_TWO_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_SECOND_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_DETAIL_LORAS", raising=False)
    monkeypatch.delenv("MMH3_EXTRA_LORAS", raising=False)
    monkeypatch.setattr(
        performance, "resolve_registered_model_name", lambda category, name, comfy_root=None: name
    )
    monkeypatch.setattr(
        performance, "_load_lightx2v_lora", lambda model, name, strength=1.0, low_vram=False: model
    )
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)

    guide = {
        "mode": "REF2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "h3_reference",
    }
    MiniMaxH3AccelerationRouter().apply("model", guide)
    assert guide["second_stage_noise_mode"] == "注入新噪声（U22 同配方）"

    guide_legacy = dict(guide)
    MiniMaxH3AccelerationRouter().apply(
        "model", guide_legacy, second_stage_noise="不注入（旧行为）"
    )
    assert guide_legacy["second_stage_noise_mode"] == "不注入（旧行为）"


def test_fl_two_stage_applies_detail_loras_when_registered(monkeypatch):
    calls = []

    class FakeSageNode:
        def execute(self, model):
            calls.append(("sage", model))
            return (f"{model}:sage",)

    monkeypatch.delenv("MMH3_TWO_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_SECOND_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_DETAIL_LORAS", raising=False)
    monkeypatch.delenv("MMH3_EXTRA_LORAS", raising=False)
    monkeypatch.setattr(
        performance, "resolve_registered_model_name", lambda category, name, comfy_root=None: name
    )

    def fake_lora(model, name, strength=1.0, low_vram=False):
        calls.append(("lora", model, name, strength))
        return f"lora-clone-{len(calls)}"

    monkeypatch.setattr(performance, "_load_lightx2v_lora", fake_lora)
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)
    monkeypatch.setattr(
        performance,
        "_apply_minimax_reuse_attention",
        lambda *args: (_ for _ in ()).throw(AssertionError("默认全速模式不得回退分块注意力")),
        raising=False,
    )

    guide = {
        "mode": "I2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "fl2va_model",
        "voice_mode": "none",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert calls == [
        ("lora", "model", FL_STAGE1_LORA, 0.75),
        ("lora", "lora-clone-1", "wushu_spatial_physics_v2_1000_pruned.safetensors", 0.3),
        ("lora", "lora-clone-2", "MysticXXX_MMH3-V1.safetensors", 0.5),
        ("lora", "lora-clone-3", "动作i连续性修复LORA.safetensors", 0.4),
        ("lora", "lora-clone-4", "MinimaxH3真实电影质感V1.0.safetensors", 0.5),
        ("lora", "model", FL_STAGE2_LORA, 0.7),
        ("lora", "lora-clone-6", "wushu_spatial_physics_v2_1000_pruned.safetensors", 0.3),
        ("lora", "lora-clone-7", "MysticXXX_MMH3-V1.safetensors", 0.5),
        ("lora", "lora-clone-8", "动作i连续性修复LORA.safetensors", 0.4),
        ("lora", "lora-clone-9", "MinimaxH3真实电影质感V1.0.safetensors", 0.5),
        ("sage", "lora-clone-5"),
        ("sage", "lora-clone-10"),
    ]
    assert result[0] == "lora-clone-5:sage"
    assert result[3] == "lora-clone-10:sage"
    assert result[2] is True
    assert guide["detail_loras_applied"] == [
        "wushu_spatial_physics_v2_1000_pruned.safetensors@0.3",
        "MysticXXX_MMH3-V1.safetensors@0.5",
        "动作i连续性修复LORA.safetensors@0.4",
        "MinimaxH3真实电影质感V1.0.safetensors@0.5",
    ]
