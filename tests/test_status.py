from types import SimpleNamespace

import pytest

from nodes import rtx_vsr_stream
import nodes.status as status_module
from nodes.status import detect_capabilities, status_summary
from nodes.two_stage_assets import (
    FL_STAGE1_LORA,
    FL_STAGE2_LORA,
    LATENT_UPSCALER_MODEL,
    REF_STAGE_LORA,
)


class FakeVideoSuperResWithQualityLevel:
    QualityLevel = object()


def test_status_distinguishes_fish_node_from_model(tmp_path):
    (tmp_path / "custom_nodes" / "ComfyUI-fish-audio-s2").mkdir(parents=True)
    (tmp_path / "models" / "fishaudioS2").mkdir(parents=True)

    status = detect_capabilities(tmp_path)

    assert status["fish_s2"]["node_available"] is True
    assert status["fish_s2"]["model_available"] is False
    assert status["fish_s2"]["available"] is False


def test_status_detects_h3_models_and_acceleration_components(tmp_path):
    diffusion = tmp_path / "models" / "diffusion_models"
    loras = tmp_path / "models" / "loras"
    diffusion.mkdir(parents=True)
    loras.mkdir(parents=True)
    (diffusion / "MiniMax-H3-FL2VA-Q4_K_M.gguf").write_bytes(b"x")
    (diffusion / "MiniMax-H3-Ref2VA-Q4_K_M.gguf").write_bytes(b"x")
    (loras / "minimax_h3_fl2v_lightx2v_turbo_4step.safetensors").write_bytes(b"x")
    (tmp_path / "custom_nodes" / "ComfyUI-SolAttn_triton").mkdir(parents=True)

    status = detect_capabilities(tmp_path)

    assert status["models"]["fl2va"] is True
    assert status["models"]["ref2va"] is True
    assert status["acceleration"]["lightx2v_4step"] is True
    assert status["acceleration"]["solattn"] is True


def test_status_detects_sage_and_easycache_from_existing_node_bundles(tmp_path):
    custom = tmp_path / "custom_nodes"
    (custom / "ComfyUI-KJNodes").mkdir(parents=True)
    (custom / "ComfyUI-DaSiWa-Nodes").mkdir(parents=True)

    status = detect_capabilities(tmp_path)

    assert status["acceleration"]["sage_attention"] is True
    assert status["acceleration"]["easycache"] is True


def test_status_summary_is_chinese(tmp_path):
    (tmp_path / "custom_nodes" / "ComfyUI-fish-audio-s2").mkdir(parents=True)
    (tmp_path / "models" / "fishaudioS2").mkdir(parents=True)
    summary = status_summary(detect_capabilities(tmp_path))
    assert "Fish S2" in summary
    assert "模型未就绪" in summary


def test_status_reports_rtx_vsr_dependency_state(monkeypatch, tmp_path):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(status_module.importlib, "import_module", missing)
    status = detect_capabilities(tmp_path)
    assert status["postprocess"]["rtx_vsr"]["node_available"] is True
    assert status["postprocess"]["rtx_vsr"]["dependency_available"] is False
    assert "安装后重启" in status["postprocess"]["rtx_vsr"]["message"]

    monkeypatch.setattr(rtx_vsr_stream.sys, "platform", "linux")
    monkeypatch.setattr(status_module.importlib, "import_module", lambda name: SimpleNamespace(VideoSuperRes=object()))
    missing_quality = detect_capabilities(tmp_path)["postprocess"]["rtx_vsr"]
    assert missing_quality["dependency_available"] is False
    assert "QualityLevel" in missing_quality["message"]
    assert "570.190+" in missing_quality["message"]
    assert "Broadcast SDK" not in missing_quality["message"]


@pytest.mark.parametrize(
    "module",
    [
        SimpleNamespace(
            VideoSuperRes=object(),
            effects=SimpleNamespace(QualityLevel=object()),
        ),
        SimpleNamespace(VideoSuperRes=FakeVideoSuperResWithQualityLevel),
    ],
    ids=["module_effects", "video_super_res_class"],
)
def test_status_accepts_each_production_quality_level_source(monkeypatch, tmp_path, module):
    monkeypatch.setattr(status_module.importlib, "import_module", lambda name: module)

    ready = detect_capabilities(tmp_path)["postprocess"]["rtx_vsr"]

    assert ready["dependency_available"] is True
    assert ready["message"] == "RTX VSR 依赖已就绪"


def test_status_reports_rtx_vsr_when_windows_dll_is_missing(monkeypatch, tmp_path):
    def missing_dll(name):
        raise OSError("The specified module could not be found")

    monkeypatch.setattr(status_module.importlib, "import_module", missing_dll)

    status = detect_capabilities(tmp_path)

    assert status["postprocess"]["rtx_vsr"]["dependency_available"] is False
    assert "安装后重启" in status["postprocess"]["rtx_vsr"]["message"]


@pytest.mark.parametrize(
    ("platform", "expected_fragments", "unexpected_fragment"),
    [
        ("linux", ("570.190+", "580.82+", "590.44+", "nvidia-vfx"), "Broadcast SDK"),
        ("win32", ("NVIDIA Broadcast SDK/Video Effects",), None),
    ],
)
@pytest.mark.parametrize("module", [None, SimpleNamespace()])
def test_status_rtx_vsr_failures_use_platform_runtime_guidance(
    monkeypatch,
    tmp_path,
    platform,
    expected_fragments,
    unexpected_fragment,
    module,
):
    monkeypatch.setattr(rtx_vsr_stream.sys, "platform", platform)
    if module is None:
        def missing(name):
            raise ModuleNotFoundError(name)

        monkeypatch.setattr(status_module.importlib, "import_module", missing)
    else:
        monkeypatch.setattr(status_module.importlib, "import_module", lambda name: module)

    message = detect_capabilities(tmp_path)["postprocess"]["rtx_vsr"]["message"]

    for fragment in expected_fragments:
        assert fragment in message
    if unexpected_fragment is not None:
        assert unexpected_fragment not in message


def test_status_reports_trained_two_stage_assets_by_route(tmp_path):
    latent_models = tmp_path / "models" / "latent_upscale_models"
    loras = tmp_path / "models" / "loras"
    latent_models.mkdir(parents=True)
    loras.mkdir(parents=True)
    (latent_models / LATENT_UPSCALER_MODEL).write_bytes(b"x")
    (loras / FL_STAGE1_LORA).write_bytes(b"x")
    (loras / FL_STAGE2_LORA).write_bytes(b"x")
    (loras / REF_STAGE_LORA).write_bytes(b"x")

    status = detect_capabilities(
        tmp_path,
        node_mappings={
            "MinimaxH3LatentUpscaler3D": object(),
            "H3SigmaRefiner": object(),
        },
    )

    trained = status["acceleration"]["trained_latent_two_stage"]
    assert trained["fl"]["ready"] is True
    assert trained["reference"]["ready"] is True
    assert trained["model_directory"] == "models/latent_upscale_models"
    assert trained["model_name"] == LATENT_UPSCALER_MODEL
