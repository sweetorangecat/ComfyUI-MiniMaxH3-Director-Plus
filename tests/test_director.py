import json

import pytest

from nodes.director import MiniMaxH3DirectorPlus, align_frame_count, native_resolution_for_request
from nodes.resolution import calculate_resolution
from nodes.schema import RequestError
from nodes.upscale import MiniMaxH3VideoUpscale


def test_director_exposes_native_upload_widgets_for_each_media_role():
    inputs = MiniMaxH3DirectorPlus.INPUT_TYPES()
    optional = inputs["optional"]
    assert optional["first_image_file"][1]["image_upload"] is True
    assert optional["last_image_file"][1]["image_upload"] is True
    assert optional["reference_image_1_file"][1]["image_upload"] is True
    assert optional["voice_reference_audio_file"][1]["audio_upload"] is True
    assert optional["voice_reference_audio_2_file"][1]["audio_upload"] is True
    assert optional["voice_reference_audio_3_file"][1]["audio_upload"] is True
    assert optional["reference_image_9_file"][1]["image_upload"] is True


def test_resolution_preset_is_labeled_as_the_final_output_target():
    resolution_spec = MiniMaxH3DirectorPlus.INPUT_TYPES()["required"]["resolution_preset"]

    assert "最终输出目标" in resolution_spec[1]["tooltip"]
    assert "低显存" in resolution_spec[1]["tooltip"]


def test_director_exposes_postprocess_widgets():
    required = MiniMaxH3DirectorPlus.INPUT_TYPES()["required"]

    assert required["postprocess_mode"][0] == ["native", "rtx_vsr"]
    assert "AI 细节重建（RTX VSR）" in required["postprocess_mode"][1]["tooltip"]
    assert required["rtx_quality"][0] == ["HIGH", "ULTRA"]


def test_uploaded_filenames_are_loaded_without_external_connections(monkeypatch):
    first = object()
    voice = {"waveform": object(), "sample_rate": 32000}
    monkeypatch.setattr(
        "nodes.director.load_uploaded_image",
        lambda filename: first if filename == "first.png" else None,
    )
    monkeypatch.setattr(
        "nodes.director.load_uploaded_audio",
        lambda filename: voice if filename == "voice.wav" else None,
    )

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="I2VA",
        prompt="我的新对白。",
        duration=5,
        width=1344,
        height=768,
        voice_mode="h3_reference",
        ref_image_size="match",
        performance_preset="参考图加速",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
        first_image_file="first.png",
        voice_reference_audio_file="voice.wav",
    )

    assert guide["first_frame"] is None
    assert guide["ref_images"]["ref_image_1"] is first
    assert guide["ref_audios"]["ref_audio_1"] is voice


def test_native_fl2va_guide_keeps_hard_endpoint_images():
    first = object()
    last = object()

    result = MiniMaxH3DirectorPlus().build(
        mode="FL2VA",
        prompt="人物从房间走到门外。",
        duration=5,
        width=1344,
        height=768,
        voice_mode="none",
        ref_image_size="match",
        performance_preset="稳定质量",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
        first_image=first,
        last_image=last,
    )

    guide, length, resolved_prompt, backend, warnings, _, _, is_hard_endpoint = result[:8]
    assert guide["resolved_backend"] == "fl2va_model"
    assert guide["first_frame"] is first
    assert guide["last_frame"] is last
    assert guide["ref_images"] == {}
    assert length == 124
    assert resolved_prompt == "人物从房间走到门外。"
    assert backend == "fl2va_model"
    assert is_hard_endpoint is True
    assert warnings == ""


def test_h3_reference_fl2va_guide_converts_endpoints_to_references():
    first = object()
    last = object()
    voice = {"waveform": object(), "sample_rate": 32000}

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="FL2VA",
        prompt="人物说：我们回家。",
        duration=5,
        width=1344,
        height=768,
        voice_mode="h3_reference",
        ref_image_size="match",
        performance_preset="参考图加速",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
        first_image=first,
        last_image=last,
        voice_reference_audio=voice,
    )

    assert guide["resolved_backend"] == "ref2va_model"
    assert guide["first_frame"] is None
    assert guide["last_frame"] is None
    assert guide["ref_images"] == {"ref_image_1": first, "ref_image_2": last}
    assert guide["ref_audios"] == {"ref_audio_1": voice}
    assert "<Audio 1>" in guide["prompt"]
    assert "不是硬端点" in guide["warnings"][0]


def test_fish_lock_exports_voice_sample_and_dialogue_for_fish_node():
    voice = {"waveform": object(), "sample_rate": 32000}

    result = MiniMaxH3DirectorPlus().build(
        mode="I2VA",
        prompt="人物看向镜头。",
        duration=5,
        width=1344,
        height=768,
        voice_mode="fish_lock",
        ref_image_size="match",
        performance_preset="参考图加速",
        timeline_data=json.dumps({"version": 1, "items": []}),
        target_dialogue="欢迎回来。",
        reference_transcript="",
        first_image=object(),
        voice_reference_audio=voice,
    )

    guide, _, _, _, _, exported_voice, dialogue, is_hard_endpoint = result[:8]
    assert guide["voice_mode"] == "fish_lock"
    assert exported_voice is voice
    assert dialogue == "欢迎回来。"
    assert is_hard_endpoint is False


def test_voice_reference_disables_hard_endpoint_branch():
    result = MiniMaxH3DirectorPlus().build(
        mode="FL2VA",
        prompt="人物转身。",
        duration=5,
        width=1344,
        height=768,
        voice_mode="h3_reference",
        ref_image_size="match",
        performance_preset="稳定质量",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
        first_image=object(),
        voice_reference_audio={"waveform": object(), "sample_rate": 32000},
    )
    assert result[7] is False


def test_ref2va_accepts_five_additional_reference_images():
    images = [object() for _ in range(5)]
    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="REF2VA",
        prompt="保持五个角色身份一致。",
        duration=5,
        width=1344,
        height=768,
        voice_mode="none",
        ref_image_size="match",
        performance_preset="参考图加速",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
        reference_image_1=images[0],
        reference_image_2=images[1],
        reference_image_3=images[2],
        reference_image_4=images[3],
        reference_image_5=images[4],
    )
    assert list(guide["ref_images"].values()) == images


def test_ref2va_accepts_nine_total_reference_images_and_three_audio_samples():
    images = [object() for _ in range(9)]
    audios = [{"waveform": object(), "sample_rate": 32000} for _ in range(3)]
    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="REF2VA",
        prompt="三个角色依次说话。",
        duration=5,
        width=1344,
        height=768,
        voice_mode="h3_reference",
        ref_image_size="match",
        performance_preset="参考图加速",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
        reference_image_1=images[0], reference_image_2=images[1], reference_image_3=images[2],
        reference_image_4=images[3], reference_image_5=images[4], reference_image_6=images[5],
        reference_image_7=images[6], reference_image_8=images[7], reference_image_9=images[8],
        voice_reference_audio=audios[0], voice_reference_audio_2=audios[1], voice_reference_audio_3=audios[2],
    )

    assert list(guide["ref_images"].values()) == images
    assert list(guide["ref_audios"].values()) == audios
    assert "<Audio 1>" in guide["prompt"]
    assert "<Audio 2>" in guide["prompt"]
    assert "<Audio 3>" in guide["prompt"]


def test_numbered_voice_references_reject_gaps_instead_of_silently_renumbering():
    with pytest.raises(RequestError, match="音色参考必须从 1 开始连续上传"):
        MiniMaxH3DirectorPlus().build(
            mode="REF2VA",
            prompt="角色乙说话。",
            duration=5,
            width=1344,
            height=768,
            voice_mode="h3_reference",
            ref_image_size="match",
            performance_preset="参考图加速",
            timeline_data="{}",
            target_dialogue="",
            reference_transcript="",
            reference_image_1=object(),
            voice_reference_name_2="角色乙",
            voice_reference_audio_2={"waveform": object(), "sample_rate": 32000},
        )


def test_numbered_reference_images_reject_gaps_instead_of_silently_renumbering():
    with pytest.raises(RequestError, match="参考图必须从 1 开始连续上传"):
        MiniMaxH3DirectorPlus().build(
            mode="REF2VA",
            prompt="保持人物身份。",
            duration=5,
            width=1344,
            height=768,
            voice_mode="none",
            ref_image_size="match",
            performance_preset="参考图加速",
            timeline_data="{}",
            target_dialogue="",
            reference_transcript="",
            reference_image_2=object(),
        )


def test_ref2va_picture_slots_include_the_first_two_visible_slots_in_gap_validation():
    with pytest.raises(RequestError, match="参考图必须从 1 开始连续上传"):
        MiniMaxH3DirectorPlus().build(
            mode="REF2VA",
            prompt="结束构图参考。",
            duration=5,
            width=1344,
            height=768,
            voice_mode="none",
            ref_image_size="match",
            performance_preset="参考图加速",
            timeline_data="{}",
            target_dialogue="",
            reference_transcript="",
            last_image=object(),
        )


def test_frame_alignment_matches_native_h3_grid():
    assert align_frame_count(5 * 24) == 124
    assert align_frame_count(10 * 24) == 243


def test_director_seed_uses_comfy_control_after_generate_and_is_exported():
    seed_spec = MiniMaxH3DirectorPlus.INPUT_TYPES()["required"]["seed"]
    assert seed_spec[0] == "INT"
    assert seed_spec[1]["control_after_generate"] == "seed_mode"

    result = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=5,
        width=1344,
        height=768,
        seed=123456,
        voice_mode="none",
        ref_image_size="match",
        performance_preset="稳定质量",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert result[-1] == 123456


def test_low_vram_native_bypass_keeps_requested_dimensions_for_reporting_only():
    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="I2VA",
        prompt="镜头缓慢推进。",
        duration=12,
        width=864,
        height=1568,
        aspect_ratio="9:16",
        resolution_preset="1.30 MP",
        voice_mode="none",
        ref_image_size="match",
        performance_preset="低显存",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
        first_image=object(),
    )

    assert (guide["requested_width"], guide["requested_height"]) == (864, 1568)
    assert (guide["target_width"], guide["target_height"]) == (
        guide["native_width"],
        guide["native_height"],
    )
    assert guide["postprocess_path"] == "native_bypass"
    assert guide["upscale_required"] is False
    assert guide["upscale_method"] == "none"
    assert guide["native_width"] * guide["native_height"] <= 0.30 * 1024 * 1024
    assert guide["width"] == guide["native_width"]
    assert guide["height"] == guide["native_height"]

    decoded_frames = object()
    output_frames, _ = MiniMaxH3VideoUpscale().apply(decoded_frames, guide)
    assert output_frames is decoded_frames


@pytest.mark.parametrize(
    ("duration", "safe_preset"),
    [
        (4, "1.00 MP"),
        (8, "0.50 MP"),
        (12, "0.30 MP"),
        (15, "0.26 MP"),
    ],
)
def test_low_vram_native_resolution_scales_with_clip_duration(duration, safe_preset):
    native_width, native_height, upscale_required = native_resolution_for_request(
        1568,
        896,
        duration,
        "low_vram",
        "16:9",
    )
    expected_width, expected_height = calculate_resolution(safe_preset, "16:9")

    assert (native_width, native_height) == (expected_width, expected_height)
    assert upscale_required is True


def test_low_vram_keeps_small_target_without_unnecessary_upscale():
    native_width, native_height, upscale_required = native_resolution_for_request(
        416,
        736,
        12,
        "low_vram",
        "9:16",
    )

    assert (native_width, native_height) == (416, 736)
    assert upscale_required is False


def test_low_vram_rejects_target_above_duration_limit():
    with pytest.raises(RequestError, match="1920×1080"):
        MiniMaxH3DirectorPlus().build(
            mode="T2VA",
            prompt="镜头缓慢推进。",
            duration=12,
            width=3840,
            height=2160,
            aspect_ratio="16:9",
            resolution_preset="4K UHD",
            voice_mode="none",
            ref_image_size="match",
            performance_preset="低显存",
            timeline_data="{}",
            target_dialogue="",
            reference_transcript="",
        )


def test_duration_seven_accepts_portrait_target_within_rotated_limit():
    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=7,
        width=1056,
        height=1888,
        aspect_ratio="9:16",
        resolution_preset="1.90 MP",
        voice_mode="none",
        ref_image_size="match",
        performance_preset="低显存",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert (guide["requested_width"], guide["requested_height"]) == (1056, 1888)


def test_duration_seven_rejects_portrait_target_above_rotated_limit():
    with pytest.raises(RequestError, match="1080×1920"):
        MiniMaxH3DirectorPlus().build(
            mode="T2VA",
            prompt="镜头缓慢推进。",
            duration=7,
            width=1440,
            height=2560,
            aspect_ratio="9:16",
            resolution_preset="2K QHD",
            voice_mode="none",
            ref_image_size="match",
            performance_preset="低显存",
            timeline_data="{}",
            target_dialogue="",
            reference_transcript="",
        )


def test_same_size_rtx_request_uses_native_bypass():
    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=5,
        width=1344,
        height=768,
        voice_mode="none",
        ref_image_size="match",
        performance_preset="稳定质量",
        postprocess_mode="rtx_vsr",
        rtx_quality="ULTRA",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert guide["postprocess_mode"] == "rtx_vsr"
    assert guide["rtx_quality"] == "ULTRA"
    assert guide["postprocess_path"] == "native_bypass"
    assert guide["upscale_required"] is False
    assert guide["upscale_method"] == "none"


def test_low_vram_target_upscale_selects_rtx_vsr():
    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="I2VA",
        prompt="镜头缓慢推进。",
        duration=12,
        width=864,
        height=1568,
        aspect_ratio="9:16",
        resolution_preset="1.30 MP",
        voice_mode="none",
        ref_image_size="match",
        performance_preset="低显存",
        postprocess_mode="rtx_vsr",
        rtx_quality="HIGH",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
        first_image=object(),
    )

    assert guide["postprocess_path"] == "rtx_vsr"
    assert (guide["target_width"], guide["target_height"]) == (
        guide["requested_width"],
        guide["requested_height"],
    )
    assert guide["upscale_required"] is True
    assert guide["upscale_method"] == "rtx_vsr"


def test_smaller_target_selects_downscale(monkeypatch):
    monkeypatch.setattr(
        "nodes.director.native_resolution_for_request",
        lambda *args, **kwargs: (1920, 1080, False),
    )

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=5,
        width=1344,
        height=768,
        voice_mode="none",
        ref_image_size="match",
        performance_preset="稳定质量",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert guide["postprocess_path"] == "downscale"
    assert (guide["target_width"], guide["target_height"]) == (
        guide["requested_width"],
        guide["requested_height"],
    )
    assert guide["upscale_required"] is False
    assert guide["upscale_method"] == "cpu_bicubic"
