import sys
import types
from pathlib import Path

import pytest


from nodes.two_stage_assets import (
    FL_STAGE1_LORA,
    FL_STAGE2_LORA,
    LATENT_UPSCALER_MODEL,
    REF_STAGE_LORA,
    V4_TURBO_LORA,
    V4_TURBO_LORA_PRUNED,
    dependency_report,
    resolve_v4_turbo_lora_name,
    run_trained_latent_upscaler,
    resolve_two_stage_route,
)


def _touch(root: Path, relative: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_route_is_locked_to_backend_and_excludes_fish():
    assert resolve_two_stage_route(
        {"resolved_backend": "fl2va_model", "voice_mode": "none"}
    ) == "trained_latent_fl"
    assert resolve_two_stage_route(
        {"resolved_backend": "ref2va_model", "voice_mode": "h3_reference"}
    ) == "trained_latent_ref"
    assert resolve_two_stage_route(
        {"resolved_backend": "ref2va_model", "voice_mode": "fish_lock"}
    ) == "bypass"


def test_fl_dependency_report_requires_only_fl_assets(tmp_path):
    report = dependency_report(
        tmp_path,
        "trained_latent_fl",
        node_mappings={},
    )

    assert report["ready"] is False
    assert report["missing"] == [
        "MinimaxH3LatentUpscaler3D",
        LATENT_UPSCALER_MODEL,
        FL_STAGE1_LORA,
        FL_STAGE2_LORA,
    ]
    assert REF_STAGE_LORA not in report["missing"]
    assert "H3SigmaRefiner" not in report["missing"]


def test_reference_dependency_report_requires_v4_lora_without_refiner(tmp_path):
    report = dependency_report(
        tmp_path,
        "trained_latent_ref",
        node_mappings={},
    )

    assert report["ready"] is False
    assert report["missing"] == [
        "MinimaxH3LatentUpscaler3D",
        LATENT_UPSCALER_MODEL,
        f"{V4_TURBO_LORA} 或 {V4_TURBO_LORA_PRUNED}",
    ]
    assert FL_STAGE1_LORA not in report["missing"]
    assert FL_STAGE2_LORA not in report["missing"]
    assert REF_STAGE_LORA not in report["missing"]
    assert "H3SigmaRefiner" not in report["missing"]
    assert report["required_assets"] == [
        LATENT_UPSCALER_MODEL,
        V4_TURBO_LORA,
        V4_TURBO_LORA_PRUNED,
    ]


def test_pruned_v4_lora_variant_satisfies_reference_dependency(tmp_path):
    _touch(tmp_path, f"models/latent_upscale_models/{LATENT_UPSCALER_MODEL}")
    _touch(tmp_path, f"models/loras/{V4_TURBO_LORA_PRUNED}")

    report = dependency_report(
        tmp_path,
        "trained_latent_ref",
        node_mappings={"MinimaxH3LatentUpscaler3D": object()},
    )

    assert report["ready"] is True
    assert report["missing"] == []
    assert resolve_v4_turbo_lora_name(tmp_path) == V4_TURBO_LORA_PRUNED


def test_pruned_v4_lora_is_preferred_over_original_variant(tmp_path):
    _touch(tmp_path, f"models/loras/{V4_TURBO_LORA}")
    _touch(tmp_path, f"models/loras/{V4_TURBO_LORA_PRUNED}")

    # The pruned ComfyUI conversion loads through the native LoRA path
    # (no runtime-injection hooks), so it wins whenever both files exist.
    assert resolve_v4_turbo_lora_name(tmp_path) == V4_TURBO_LORA_PRUNED


def test_installed_legacy_upscaler_id_is_accepted(tmp_path):
    _touch(tmp_path, f"models/latent_upscale_models/{LATENT_UPSCALER_MODEL}")
    _touch(tmp_path, f"models/loras/{FL_STAGE1_LORA}")
    _touch(tmp_path, f"models/loras/{FL_STAGE2_LORA}")

    report = dependency_report(
        tmp_path,
        "trained_latent_fl",
        node_mappings={"MinimaxH3LatentUpscalerNode3D": object()},
    )

    assert report["ready"] is True
    assert report["missing"] == []
    assert report["upscaler_node_id"] == "MinimaxH3LatentUpscalerNode3D"


def test_registered_comfy_model_paths_satisfy_two_stage_dependencies(tmp_path, monkeypatch):
    shared_models = tmp_path.parent / f"{tmp_path.name}-shared-models"
    upscaler = shared_models / "latent_upscale_models" / LATENT_UPSCALER_MODEL
    stage1 = shared_models / "loras" / FL_STAGE1_LORA
    stage2 = shared_models / "loras" / FL_STAGE2_LORA
    for path in (upscaler, stage1, stage2):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    registered = {
        ("latent_upscale_models", LATENT_UPSCALER_MODEL): str(upscaler),
        ("loras", FL_STAGE1_LORA): str(stage1),
        ("loras", FL_STAGE2_LORA): str(stage2),
    }
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        types.SimpleNamespace(
            base_path=str(tmp_path),
            get_full_path=lambda category, name: registered.get((category, name)),
        ),
    )

    report = dependency_report(
        tmp_path,
        "trained_latent_fl",
        node_mappings={"MinimaxH3LatentUpscaler3D": object()},
    )

    assert report["ready"] is True
    assert report["missing"] == []


def test_registered_lora_subfolders_satisfy_two_stage_dependencies(tmp_path, monkeypatch):
    shared_loras = tmp_path.parent / f"{tmp_path.name}-shared-loras" / "minimax"
    stage1 = shared_loras / FL_STAGE1_LORA
    stage2 = shared_loras / FL_STAGE2_LORA
    for path in (stage1, stage2):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    _touch(tmp_path, f"models/latent_upscale_models/{LATENT_UPSCALER_MODEL}")

    candidates = [f"minimax/{FL_STAGE1_LORA}", f"minimax/{FL_STAGE2_LORA}"]

    def get_full_path(category, name):
        if category != "loras" or name not in candidates:
            return None
        return str(shared_loras / Path(name).name)

    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        types.SimpleNamespace(
            base_path=str(tmp_path),
            get_full_path=get_full_path,
            get_filename_list=lambda category: candidates if category == "loras" else [],
        ),
    )

    report = dependency_report(
        tmp_path,
        "trained_latent_fl",
        node_mappings={"MinimaxH3LatentUpscaler3D": object()},
    )

    assert report["ready"] is True
    assert report["missing"] == []


def test_bypass_route_has_no_required_assets(tmp_path):
    report = dependency_report(tmp_path, "bypass", node_mappings={})
    assert report["ready"] is True
    assert report["missing"] == []


def test_trained_upscaler_adapter_supports_current_temporal_api(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    calls = []

    class CurrentUpscaler:
        @classmethod
        def execute(
            cls,
            latent,
            model_name,
            mode,
            align,
            enable_temporal_chunking,
            force_unload,
            device,
            precision,
        ):
            calls.append(
                (
                    latent,
                    model_name,
                    mode,
                    align,
                    enable_temporal_chunking,
                    force_unload,
                    device,
                    precision,
                )
            )
            return types.SimpleNamespace(result=({"samples": torch.ones(1, 24, 2, 6, 6)},))

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": CurrentUpscaler},
        raising=False,
    )
    latent = {"samples": torch.zeros(1, 24, 2, 4, 4)}

    result = run_trained_latent_upscaler(latent, 1.5)

    assert result["samples"].shape == (1, 24, 2, 6, 6)
    assert calls == [
        (
            latent,
            LATENT_UPSCALER_MODEL,
            {"mode": "scale by multiplier", "scale": 1.5},
            32,
            True,
            True,
            "cuda",
            "bf16",
        )
    ]


def test_trained_upscaler_adapter_prefers_real_execute_over_comfy_v3_wrapper(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    calls = []

    class WrappedCurrentUpscaler:
        FUNCTION = "EXECUTE_NORMALIZED"

        @classmethod
        def execute(
            cls,
            latent,
            model_name,
            mode,
            align,
            enable_temporal_chunking,
            force_unload,
            device,
            precision,
        ):
            calls.append(
                (
                    latent,
                    model_name,
                    mode,
                    align,
                    enable_temporal_chunking,
                    force_unload,
                    device,
                    precision,
                )
            )
            return types.SimpleNamespace(result=({"samples": torch.ones(1, 24, 2, 6, 6)},))

        @classmethod
        def EXECUTE_NORMALIZED(cls, *args, **kwargs):
            raise AssertionError("the ComfyUI v3 wrapper must not be called")

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": WrappedCurrentUpscaler},
        raising=False,
    )
    latent = {"samples": torch.zeros(1, 24, 2, 4, 4)}

    result = run_trained_latent_upscaler(latent, 1.5)

    assert result["samples"].shape == (1, 24, 2, 6, 6)
    assert calls == [
        (
            latent,
            LATENT_UPSCALER_MODEL,
            {"mode": "scale by multiplier", "scale": 1.5},
            32,
            True,
            True,
            "cuda",
            "bf16",
        )
    ]


def test_trained_upscaler_adapter_binds_instance_execute_behind_comfy_v3_wrapper(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    calls = []

    class WrappedInstanceUpscaler:
        FUNCTION = "EXECUTE_NORMALIZED"

        def execute(
            self,
            latent,
            model_name,
            mode,
            align,
            enable_temporal_chunking,
            force_unload,
            device,
            precision,
        ):
            calls.append((enable_temporal_chunking, force_unload, mode))
            return types.SimpleNamespace(
                result=({"samples": torch.ones(1, 24, 2, 6, 6)},)
            )

        def EXECUTE_NORMALIZED(self, *args, **kwargs):
            raise AssertionError("the ComfyUI v3 wrapper must not be called")

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": WrappedInstanceUpscaler},
        raising=False,
    )

    result = run_trained_latent_upscaler(
        {"samples": torch.zeros(1, 24, 2, 4, 4)}, 1.5
    )

    assert result["samples"].shape == (1, 24, 2, 6, 6)
    assert calls == [
        (True, True, {"mode": "scale by multiplier", "scale": 1.5})
    ]


def test_trained_upscaler_adapter_uses_model_scale_cuda_bf16_and_chunking(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    calls = []

    class Upscaler:
        @classmethod
        def execute(cls, latent, model_name, mode, align, enable_chunking, device, precision):
            calls.append((latent, model_name, mode, align, enable_chunking, device, precision))
            return types.SimpleNamespace(result=({"samples": torch.ones(1, 24, 2, 6, 6)},))

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": Upscaler},
        raising=False,
    )
    latent = {"samples": torch.zeros(1, 24, 2, 4, 4)}

    result = run_trained_latent_upscaler(latent, 1.5)

    assert result["samples"].shape == (1, 24, 2, 6, 6)
    assert calls == [(
        latent,
        LATENT_UPSCALER_MODEL,
        {"mode": "scale by multiplier", "scale": 1.5},
        32,
        True,
        "cuda",
        "bf16",
    )]


def test_trained_upscaler_adapter_reads_execute_signature_behind_comfy_v3_wrapper(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    calls = []

    class WrappedUpscaler:
        FUNCTION = "EXECUTE_NORMALIZED"

        @classmethod
        def execute(cls, latent, model_name, mode, align, enable_chunking, device, precision):
            calls.append((latent, model_name, mode, align, enable_chunking, device, precision))
            return types.SimpleNamespace(result=({"samples": torch.ones(1, 24, 2, 6, 6)},))

        @classmethod
        def EXECUTE_NORMALIZED(cls, *args, **kwargs):
            return cls.execute(*args, **kwargs)

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": WrappedUpscaler},
        raising=False,
    )
    latent = {"samples": torch.zeros(1, 24, 2, 4, 4)}

    result = run_trained_latent_upscaler(latent, 1.5)

    assert result["samples"].shape == (1, 24, 2, 6, 6)
    assert calls == [(
        latent,
        LATENT_UPSCALER_MODEL,
        {"mode": "scale by multiplier", "scale": 1.5},
        32,
        True,
        "cuda",
        "bf16",
    )]


def test_trained_upscaler_adapter_supports_pre_chunking_keep_proportion_interface(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    calls = []

    class WrappedOldUpscaler:
        FUNCTION = "EXECUTE_NORMALIZED"

        @classmethod
        def execute(cls, latent, model_name, mode, align, keep_proportion, device, precision):
            calls.append((latent, model_name, mode, align, keep_proportion, device, precision))
            return types.SimpleNamespace(result=({"samples": torch.ones(1, 24, 2, 6, 6)},))

        @classmethod
        def EXECUTE_NORMALIZED(cls, *args, **kwargs):
            return cls.execute(*args, **kwargs)

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": WrappedOldUpscaler},
        raising=False,
    )
    latent = {"samples": torch.zeros(1, 24, 2, 4, 4)}

    result = run_trained_latent_upscaler(latent, 1.5)

    assert result["samples"].shape == (1, 24, 2, 6, 6)
    assert calls == [(
        latent,
        LATENT_UPSCALER_MODEL,
        {"mode": "scale by multiplier", "scale": 1.5},
        32,
        False,
        "cuda",
        "bf16",
    )]


def test_trained_upscaler_adapter_supports_legacy_flat_scale_interface(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    calls = []

    class FlatScaleUpscaler:
        @classmethod
        def execute(
            cls,
            latent,
            model_name,
            scale,
            align,
            enable_chunking,
            device,
            precision,
        ):
            calls.append(
                (latent, model_name, scale, align, enable_chunking, device, precision)
            )
            return types.SimpleNamespace(
                result=({"samples": torch.ones(1, 24, 2, 6, 6)},)
            )

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": FlatScaleUpscaler},
        raising=False,
    )
    latent = {"samples": torch.zeros(1, 24, 2, 4, 4)}

    result = run_trained_latent_upscaler(latent, 1.5)

    assert result["samples"].shape == (1, 24, 2, 6, 6)
    assert calls == [
        (
            latent,
            LATENT_UPSCALER_MODEL,
            1.5,
            32,
            True,
            "cuda",
            "bf16",
        )
    ]


def test_trained_upscaler_adapter_rejects_unknown_required_fields(monkeypatch):
    import torch
    import nodes.two_stage_assets as assets

    class UnknownUpscaler:
        @classmethod
        def execute(cls, latent, model_name, mode, required_future_flag):
            raise AssertionError("unsupported node must not execute")

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": UnknownUpscaler},
        raising=False,
    )

    with pytest.raises(RuntimeError, match="required_future_flag"):
        run_trained_latent_upscaler(
            {"samples": torch.zeros(1, 24, 2, 4, 4)}, 1.5
        )


def test_trained_upscaler_adapter_rejects_required_positional_only_fields(monkeypatch):
    import torch
    import nodes.two_stage_assets as assets

    class PositionalOnlyUpscaler:
        @classmethod
        def execute(cls, latent, /, model_name, mode, device, precision):
            raise AssertionError("unsupported node must not execute")

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": PositionalOnlyUpscaler},
        raising=False,
    )

    with pytest.raises(RuntimeError, match="latent"):
        run_trained_latent_upscaler(
            {"samples": torch.zeros(1, 24, 2, 4, 4)}, 1.5
        )


def test_trained_upscaler_rejects_invalid_input_before_external_execution(monkeypatch):
    import torch
    import nodes.two_stage_assets as assets

    class Upscaler:
        @classmethod
        def execute(cls, latent, model_name, mode, device, precision):
            raise AssertionError("invalid input must fail before external execution")

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": Upscaler},
        raising=False,
    )

    with pytest.raises(RuntimeError, match="输入无效"):
        run_trained_latent_upscaler({"samples": torch.zeros(24, 4, 4)}, 1.5)


def test_trained_upscaler_rejects_non_tensor_output(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    class FakeSamples:
        ndim = 5
        shape = (1, 24, 2, 6, 6)

    class BrokenUpscaler:
        @classmethod
        def execute(cls, latent, model_name, mode, device, precision):
            return types.SimpleNamespace(result=({"samples": FakeSamples()},))

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": BrokenUpscaler},
        raising=False,
    )

    with pytest.raises(RuntimeError, match="无效结果"):
        run_trained_latent_upscaler(
            {"samples": torch.zeros(1, 24, 2, 4, 4)}, 1.5
        )


def test_trained_upscaler_rejects_invalid_output_channel_count(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    class BrokenUpscaler:
        @classmethod
        def execute(cls, latent, model_name, mode, device, precision):
            return types.SimpleNamespace(
                result=({"samples": torch.ones(1, 16, 2, 6, 6)},)
            )

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": BrokenUpscaler},
        raising=False,
    )

    with pytest.raises(RuntimeError, match="通道数错误"):
        run_trained_latent_upscaler(
            {"samples": torch.zeros(1, 24, 2, 4, 4)}, 1.5
        )


def test_trained_upscaler_rejects_output_rank_change(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    class BrokenUpscaler:
        @classmethod
        def execute(cls, latent, model_name, mode, device, precision):
            return types.SimpleNamespace(
                result=({"samples": torch.ones(1, 24, 6, 6)},)
            )

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": BrokenUpscaler},
        raising=False,
    )

    with pytest.raises(RuntimeError, match="维度"):
        run_trained_latent_upscaler(
            {"samples": torch.zeros(1, 24, 2, 4, 4)}, 1.5
        )


@pytest.mark.parametrize(
    "output_shape",
    [
        (2, 24, 2, 6, 6),
        (1, 24, 3, 6, 6),
    ],
)
def test_trained_upscaler_rejects_changed_non_spatial_dimensions(
    monkeypatch, output_shape
):
    import types
    import torch
    import nodes.two_stage_assets as assets

    class BrokenUpscaler:
        @classmethod
        def execute(cls, latent, model_name, mode, device, precision):
            return types.SimpleNamespace(
                result=({"samples": torch.ones(output_shape)},)
            )

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": BrokenUpscaler},
        raising=False,
    )

    with pytest.raises(RuntimeError, match="非空间尺寸"):
        run_trained_latent_upscaler(
            {"samples": torch.zeros(1, 24, 2, 4, 4)}, 1.5
        )


def test_trained_upscaler_rejects_non_growing_spatial_output(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    class BrokenUpscaler:
        @classmethod
        def execute(cls, latent, model_name, mode, device, precision):
            return types.SimpleNamespace(
                result=({"samples": torch.ones(1, 24, 2, 4, 4)},)
            )

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": BrokenUpscaler},
        raising=False,
    )

    with pytest.raises(RuntimeError, match="空间尺寸没有增长"):
        run_trained_latent_upscaler(
            {"samples": torch.zeros(1, 24, 2, 4, 4)}, 1.5
        )
