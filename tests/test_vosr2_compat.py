import types

import pytest


def test_current_vosr2_contract_is_exposed_as_legacy_three_node_adapter(monkeypatch):
    from nodes import vosr2_compat

    class CurrentLoader:
        @classmethod
        def execute(cls, model, dtype):
            return types.SimpleNamespace(result=(('bundle', model, dtype),))

    class CurrentUpscale:
        pass

    current = types.SimpleNamespace(
        VOSR2ModelLoader=CurrentLoader,
        VOSR2Upscale=CurrentUpscale,
    )
    monkeypatch.setattr(vosr2_compat, "_load_current_vosr2", lambda: current)

    mappings = vosr2_compat.compatibility_mappings()

    assert set(mappings) == {
        "TESpeedVOSR2Loader",
        "TESpeedVOSR2Settings",
        "TESpeedVOSR2Video",
    }
    assert mappings["TESpeedVOSR2Loader"].FUNCTION == "load"
    assert mappings["TESpeedVOSR2Settings"].FUNCTION == "settings"
    assert mappings["TESpeedVOSR2Video"].FUNCTION == "upscale"


def test_core_mapping_gets_lazy_aliases_when_v3_extension_is_present(monkeypatch):
    import nodes

    from nodes import two_stage_assets, vosr2_compat

    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {"Existing": object()}, raising=False)
    monkeypatch.setattr(
        vosr2_compat,
        "compatibility_mappings",
        lambda: {"TESpeedVOSR2Loader": object(), "TESpeedVOSR2Settings": object(), "TESpeedVOSR2Video": object()},
    )

    mappings = two_stage_assets._comfy_node_mappings()

    assert set(mappings) == {
        "Existing",
        "TESpeedVOSR2Loader",
        "TESpeedVOSR2Settings",
        "TESpeedVOSR2Video",
    }


def test_loader_maps_legacy_precision_and_calls_current_v3_loader(monkeypatch):
    from nodes import vosr2_compat

    calls = []

    class CurrentLoader:
        @classmethod
        def execute(cls, model, dtype):
            calls.append((model, dtype))
            return types.SimpleNamespace(result=("bundle",))

    current = types.SimpleNamespace(VOSR2ModelLoader=CurrentLoader)
    monkeypatch.setattr(vosr2_compat, "_load_current_vosr2", lambda: current)

    result = vosr2_compat.TESpeedVOSR2Loader().load(
        model_bundle="VOSR2",
        precision="auto",
        memory_policy="auto",
        torch_compile=False,
        vae_encode_amp=False,
    )

    assert result == ("bundle",)
    assert calls == [("VOSR2", "default")]


def test_video_adapter_preserves_frames_and_forwards_bounded_batches(monkeypatch):
    torch = pytest.importorskip("torch")

    from nodes import vosr2_compat

    calls = []

    def run_vosr2(model, images, upscale, seed, color_alignment, tile_size,
                  tile_overlap, vae_tile_size, vae_tile_overlap):
        calls.append({
            "model": model,
            "shape": tuple(images.shape),
            "upscale": upscale,
            "seed": seed,
            "color_alignment": color_alignment,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "vae_tile_size": vae_tile_size,
            "vae_tile_overlap": vae_tile_overlap,
        })
        return torch.zeros(images.shape[0], images.shape[1] * upscale,
                           images.shape[2] * upscale, 3)

    current = types.SimpleNamespace(run_vosr2=run_vosr2)
    monkeypatch.setattr(vosr2_compat, "_load_current_vosr2", lambda: current)

    images = torch.zeros(5, 12, 20, 3)
    result = vosr2_compat.TESpeedVOSR2Video().upscale(
        model="model",
        images=images,
        scale=2,
        seed=41,
        settings={
            "color_alignment": "wavelet",
            "tile_size": 512,
            "tile_overlap": 32,
            "vae_tile_size": 1024,
            "vae_tile_overlap": 32,
        },
        frame_batch=2,
        temporal_cache=False,
        cache_threshold=0.003,
        cache_refresh=4,
    )

    assert result[0].shape == (5, 24, 40, 3)
    assert [item["shape"][0] for item in calls] == [2, 2, 1]
    assert [item["seed"] for item in calls] == [41, 43, 45]
    assert all(item["upscale"] == 2 for item in calls)
    assert all(item["model"] == "model" for item in calls)


def test_video_adapter_rejects_frame_count_or_shape_changes(monkeypatch):
    torch = pytest.importorskip("torch")

    from nodes import vosr2_compat

    current = types.SimpleNamespace(
        run_vosr2=lambda *args, **kwargs: torch.zeros(1, 12, 20, 3)
    )
    monkeypatch.setattr(vosr2_compat, "_load_current_vosr2", lambda: current)

    with pytest.raises(RuntimeError, match="帧数"):
        vosr2_compat.TESpeedVOSR2Video().upscale(
            model="model",
            images=torch.zeros(2, 12, 20, 3),
            scale=1,
            seed=1,
            settings={},
            frame_batch=2,
            temporal_cache=False,
            cache_threshold=0.003,
            cache_refresh=4,
        )
