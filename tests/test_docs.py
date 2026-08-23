from pathlib import Path

from nodes.schema import PUBLIC_API_KEYS


DOCS = Path("docs")


def test_docs_cover_required_workflows():
    usage = (DOCS / "使用说明.md").read_text(encoding="utf-8")
    for text in ("I2VA", "FL2VA", "REF2VA", "H3 原生音色参考", "Fish S2", "提示词约束", "硬首尾帧"):
        assert text in usage


def test_api_docs_cover_every_public_key():
    api = (DOCS / "API说明.md").read_text(encoding="utf-8")
    for key in PUBLIC_API_KEYS:
        assert f"`{key}`" in api


def test_docs_explain_runtime_status_and_output_retrieval():
    api = (DOCS / "API说明.md").read_text(encoding="utf-8")
    troubleshooting = (DOCS / "故障排查.md").read_text(encoding="utf-8")
    assert "/h3-director-plus/status" in api
    assert "history" in api
    assert "模型未就绪" in troubleshooting


def test_docs_describe_trained_two_stage_assets_and_routes():
    usage = (DOCS / "使用说明.md").read_text(encoding="utf-8")
    api = (DOCS / "API说明.md").read_text(encoding="utf-8")
    combined = usage + api

    for text in (
        "minimax_h3_latent_upscaler_3d_bf16.safetensors",
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "trained_latent_fl",
        "trained_latent_ref",
        "H3SigmaRefiner",
    ):
        assert text in combined

    assert "前 6 步定结构" not in usage
    assert "前 6 步定结构" not in api
    assert "双线性放大后的最后 2 步" not in usage
    assert "双线性放大后的最后 2 步" not in api


def test_docs_explain_vram_limits_true_2k_4k_and_install_commands():
    usage = (DOCS / "使用说明.md").read_text(encoding="utf-8")
    troubleshooting = (DOCS / "故障排查.md").read_text(encoding="utf-8")
    combined = usage + troubleshooting

    for text in (
        "8–12GB",
        "低显存不开放 4K",
        "2560×1440",
        "3840×2160",
        "4K 不是 H3 原生采样",
        "hf download LBH-123-AI/Minimax_h3_latent_Upscaler",
        "codeload.github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler",
    ):
        assert text in combined


def test_autodl_install_keeps_huggingface_hub_compatible_with_transformers():
    usage = (DOCS / "使用说明.md").read_text(encoding="utf-8")

    assert "\npython -m pip install -U huggingface_hub\n" not in usage
    assert '--index-url https://pypi.org/simple "huggingface-hub==0.36.0"' in usage
    assert "--force-reinstall --no-deps" in usage
    assert "https://files.pythonhosted.org/packages/cb/bd/1a875e0d592d447cbc02805fd3fe0f497714d6a2583f59d14fa9ebad96eb/huggingface_hub-0.36.0-py3-none-any.whl" in usage
