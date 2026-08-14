from pathlib import Path

from nodes.status import detect_capabilities, status_summary


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
