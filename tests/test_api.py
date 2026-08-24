from copy import deepcopy

import pytest

import asyncio

from api.routes import ALLOWED_EXTENSIONS, asset_destination, prepare_generation, validate_generation
from api.template import TemplateError, patch_template
from nodes.schema import public_schema


BASE_TEMPLATE = {
    "10": {
        "class_type": "MiniMaxH3DirectorPlus",
        "inputs": {"mode": "FL2VA", "prompt": "", "duration": 5, "voice_mode": "none"},
        "_meta": {"title": "快速设置 / API 入参"},
    },
    "11": {"class_type": "LoadImage", "inputs": {"image": "first.png"}, "_meta": {"title": "API 首帧图片"}},
    "12": {"class_type": "LoadImage", "inputs": {"image": "last.png"}, "_meta": {"title": "API 尾帧图片"}},
    "13": {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}, "_meta": {"title": "API 音色参考音频1"}},
    "14": {"class_type": "LoadAudio", "inputs": {"audio": "voice2.wav"}, "_meta": {"title": "API 音色参考音频2"}},
    "15": {"class_type": "LoadAudio", "inputs": {"audio": "voice3.wav"}, "_meta": {"title": "API 音色参考音频3"}},
    "31": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}, "_meta": {"title": "API 参考图1"}},
    **{
        str(30 + index): {"class_type": "LoadImage", "inputs": {"image": f"ref{index}.png"}, "_meta": {"title": f"API 参考图{index}"}}
        for index in range(2, 10)
    },
}


def test_schema_contains_chinese_names():
    schema = public_schema()
    assert schema["properties"]["voice_mode"]["中文名称"] == "音色模式"


def test_patch_template_routes_i2va_voice_to_ref_backend():
    request, prompt = prepare_generation(
        {
            "mode": "I2VA",
            "first_image": "first.png",
            "voice_mode": "h3_reference",
            "voice_reference_audio": "voice.wav",
        },
        deepcopy(BASE_TEMPLATE),
    )
    assert request["resolved_backend"] == "ref2va_model"
    assert prompt["10"]["inputs"]["mode"] == "I2VA"
    assert prompt["10"]["inputs"]["first_image"] == ["11", 0]
    assert prompt["10"]["inputs"]["voice_reference_audio"] == ["13", 0]


def test_patch_template_accepts_three_numbered_voice_references():
    prompt = patch_template(BASE_TEMPLATE, {
        "mode": "REF2VA",
        "voice_mode": "h3_reference",
        "voice_reference_audios": ["one.wav", "two.wav", "three.wav"],
        "voice_reference_names": ["角色甲", "角色乙", "旁白"],
        "references": ["person.png"],
    })
    assert prompt["10"]["inputs"]["voice_reference_audio"] == ["13", 0]
    assert prompt["10"]["inputs"]["voice_reference_audio_2"] == ["14", 0]
    assert prompt["10"]["inputs"]["voice_reference_audio_3"] == ["15", 0]
    assert prompt["13"]["inputs"]["audio"] == "one.wav"
    assert prompt["14"]["inputs"]["audio"] == "two.wav"
    assert prompt["15"]["inputs"]["audio"] == "three.wav"
    assert prompt["10"]["inputs"]["voice_reference_name_1"] == "角色甲"
    assert prompt["10"]["inputs"]["voice_reference_name_2"] == "角色乙"
    assert prompt["10"]["inputs"]["voice_reference_name_3"] == "旁白"


def test_api_template_routes_fish_generated_dialogue_into_h3_guide():
    from tools.build_u11_workflow import build_api_template

    prompt = patch_template(build_api_template(), {
        "mode": "I2VA",
        "first_image": "first.png",
        "voice_mode": "fish_lock",
        "voice_reference_audios": ["sample.wav"],
        "target_dialogue": "这是新的对白。",
        "reference_transcript": "这是样本原文。",
        "fish_model_path": "s2-pro-w4a16 (auto download)",
    })

    bridge_id, bridge = next((node_id, node) for node_id, node in prompt.items() if node["class_type"] == "MiniMaxH3FishVoiceBridge")
    guide = next(node for node in prompt.values() if node.get("_meta", {}).get("title") == "API H3 原生指南")
    assert bridge["inputs"]["guide"] == ["10", 0]
    assert bridge["inputs"]["reference_audio"] == ["13", 0]
    assert guide["inputs"]["generated_voice_audio"] == [bridge_id, 0]


def test_api_template_routes_acceleration_status_into_performance_preset():
    from tools.build_u11_workflow import build_api_template

    prompt = build_api_template()
    performance = next(node for node in prompt.values() if node["class_type"] == "MiniMaxH3PerformancePreset")
    scheduler = next(node for node in prompt.values() if node["class_type"] == "MiniMaxH3SchedulerRouter")

    assert performance["inputs"]["acceleration_ready"] == ["15", 2]
    assert scheduler["inputs"]["steps"] == ["28", 0]


def test_api_template_and_patcher_expose_final_postprocess_controls():
    from tools.build_u11_workflow import build_api_template

    template = build_api_template()
    assert template["10"]["inputs"]["postprocess_mode"] == "native"
    assert template["10"]["inputs"]["rtx_quality"] == "HIGH"
    assert template["10"]["inputs"]["ai_upscale_model"] == "auto"
    assert template["24"]["inputs"]["quality"] == 20

    prompt = patch_template(template, {
        "mode": "T2VA",
        "postprocess_mode": "rtx_vsr",
        "rtx_quality": "ULTRA",
        "ai_upscale_model": "OmniSR_X4_DIV2K.safetensors",
    })
    assert prompt["10"]["inputs"]["postprocess_mode"] == "rtx_vsr"
    assert prompt["10"]["inputs"]["rtx_quality"] == "ULTRA"
    assert prompt["10"]["inputs"]["ai_upscale_model"] == "OmniSR_X4_DIV2K.safetensors"


def test_api_template_removes_fish_bridge_for_native_h3_reference():
    from tools.build_u11_workflow import build_api_template

    prompt = patch_template(build_api_template(), {
        "mode": "I2VA",
        "first_image": "first.png",
        "voice_mode": "h3_reference",
        "voice_reference_audios": ["sample.wav"],
    })

    assert not any(node["class_type"] == "MiniMaxH3FishVoiceBridge" for node in prompt.values())
    guide = next(node for node in prompt.values() if node.get("_meta", {}).get("title") == "API H3 原生指南")
    assert "generated_voice_audio" not in guide["inputs"]


def test_prepare_generation_maps_internal_preset_back_to_node_label():
    request, prompt = prepare_generation(
        {"mode": "REF2VA", "voice_mode": "none", "performance_preset": "极速4步"},
        deepcopy(BASE_TEMPLATE),
    )
    assert request["performance_preset"] == "fast_4step"
    assert prompt["10"]["inputs"]["performance_preset"] == "极速4步"


def test_patch_template_exposes_seed_and_preset_sampling():
    template = {
        **deepcopy(BASE_TEMPLATE),
        "18": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}, "_meta": {"title": "API 随机种子"}},
        "20": {"class_type": "MiniMaxH3SchedulerRouter", "inputs": {"steps": 20}, "_meta": {"title": "API 调度器"}},
    }
    prompt = patch_template(template, {"mode": "T2VA", "voice_mode": "none", "performance_preset": "fast_4step", "seed": 123})
    assert prompt["18"]["inputs"]["noise_seed"] == 123
    assert prompt["20"]["inputs"]["steps"] == 4


def test_patch_template_accepts_reference_image_list():
    prompt = patch_template(BASE_TEMPLATE, {"mode": "REF2VA", "voice_mode": "none", "references": ["角色.png"]})
    assert prompt["31"]["inputs"]["image"] == "角色.png"
    assert prompt["10"]["inputs"]["reference_image_1"] == ["31", 0]


def test_patch_template_accepts_nine_reference_images():
    references = [f"角色{index}.png" for index in range(1, 10)]
    prompt = patch_template(BASE_TEMPLATE, {"mode": "REF2VA", "voice_mode": "none", "references": references})
    assert prompt["39"]["inputs"]["image"] == "角色9.png"
    assert prompt["10"]["inputs"]["reference_image_9"] == ["39", 0]


def test_patch_template_removes_unused_optional_loaders():
    prompt = patch_template(BASE_TEMPLATE, {"mode": "T2VA", "voice_mode": "none", "prompt": "test", "duration": 5})
    assert set(prompt) == {"10"}
    assert "first_image" not in prompt["10"]["inputs"]
    assert "last_image" not in prompt["10"]["inputs"]
    assert "voice_reference_audio" not in prompt["10"]["inputs"]


def test_patch_template_requires_controller_title():
    with pytest.raises(TemplateError, match="API 入参"):
        patch_template({}, {"mode": "T2VA"})


@pytest.mark.parametrize("filename", ["../outside.wav", "..\\outside.wav", "folder/voice.wav", "folder\\voice.wav"])
def test_asset_upload_rejects_parent_or_nested_paths(tmp_path, filename):
    with pytest.raises(ValueError, match="文件名"):
        asset_destination(tmp_path, filename)


def test_asset_upload_accepts_known_media_extension(tmp_path):
    destination = asset_destination(tmp_path, "参考音色.wav")
    assert destination.parent == tmp_path / "h3-director-plus"
    assert destination.suffix in ALLOWED_EXTENSIONS


def test_validate_generation_does_not_queue_models():
    calls = []

    async def validator(prompt_id, prompt, targets):
        calls.append((prompt_id, prompt, targets))
        return True, None, ["24"], {}

    result = asyncio.run(validate_generation(
        {"mode": "T2VA", "prompt": "测试", "voice_mode": "none"},
        deepcopy(BASE_TEMPLATE),
        validator,
    ))
    assert result["valid"] is True
    assert result["resolved_backend"] == "fl2va_model"
    assert len(calls) == 1
