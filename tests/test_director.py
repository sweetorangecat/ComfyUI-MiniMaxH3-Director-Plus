import json

import pytest

from nodes.director import MiniMaxH3DirectorPlus, align_frame_count, native_resolution_for_request
from nodes.resolution import calculate_resolution, h3_native_canvas
from nodes.schema import RequestError


@pytest.fixture(autouse=True)
def trained_two_stage_test_environment(monkeypatch):
    monkeypatch.setattr(
        "nodes.director._cuda_memory_gb",
        lambda: (32.0, 29.0),
        raising=False,
    )
    monkeypatch.setattr(
        "nodes.director._trained_two_stage_dependency_report",
        lambda route: {
            "route": route,
            "ready": True,
            "missing": [],
            "required_assets": ["test-upscaler", "test-lora"],
        },
        raising=False,
    )


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


def test_director_raw_combo_exposes_both_low_vram_presets():
    values = MiniMaxH3DirectorPlus.INPUT_TYPES()["required"]["performance_preset"][0]

    assert "低显存" in values
    assert "低显存二采" in values


def test_resolution_preset_is_labeled_as_the_final_output_target():
    resolution_spec = MiniMaxH3DirectorPlus.INPUT_TYPES()["required"]["resolution_preset"]

    assert "最终输出目标" in resolution_spec[1]["tooltip"]
    assert "低显存" in resolution_spec[1]["tooltip"]


def test_director_exposes_postprocess_widgets():
    required = MiniMaxH3DirectorPlus.INPUT_TYPES()["required"]
    optional = MiniMaxH3DirectorPlus.INPUT_TYPES()["optional"]

    assert required["postprocess_mode"][0] == ["native", "lanczos", "ai_upscale", "rtx_vsr"]
    assert "AI 细节重建（RTX VSR）" in required["postprocess_mode"][1]["tooltip"]
    assert required["rtx_quality"][0] == ["HIGH", "ULTRA", "HIGHBITRATE_ULTRA"]
    assert required["ai_upscale_model"][0][0] == "auto"
    assert optional["motion_smoothing"][0] == ["off", "rife_x2"]
    assert optional["motion_smoothing"][1]["default"] == "off"
    assert "运动平滑" in optional["motion_smoothing"][1]["tooltip"]
    assert optional["audio_loudness"][0] == ["auto", "original"]
    assert optional["audio_loudness"][1]["default"] == "auto"


def test_legacy_auto_motion_smoothing_stays_off_without_loading_rife(monkeypatch):
    checked = []
    monkeypatch.setattr("nodes.director.probe_vsr_capability", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "nodes.director.probe_rife_capability",
        lambda model_name="rife_v4.26.safetensors": checked.append(model_name),
    )

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="一个连续的缓慢推进镜头。",
        duration=15,
        width=2560,
        height=1440,
        voice_mode="none",
        ref_image_size="match",
        performance_preset="质量优先二采样",
        postprocess_mode="rtx_vsr",
        motion_smoothing="auto",
        resolution_preset="2K QHD",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert checked == []
    assert guide["motion_smoothing"] == "off"
    assert guide["output_frame_multiplier"] == 1
    assert guide["audio_loudness"] == "auto"


@pytest.mark.parametrize("postprocess_mode, expected_path", [("lanczos", "lanczos"), ("ai_upscale", "ai_upscale")])
def test_director_routes_generic_final_upscale_modes(monkeypatch, postprocess_mode, expected_path):
    if postprocess_mode == "ai_upscale":
        monkeypatch.setattr("nodes.director.resolve_upscale_model_name", lambda *args, **kwargs: "fake.pth")

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=5,
        width=2560,
        height=1440,
        voice_mode="none",
        ref_image_size="match",
        performance_preset="稳定质量",
        postprocess_mode=postprocess_mode,
        resolution_preset="2K QHD",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert guide["postprocess_path"] == expected_path


def test_ai_upscale_missing_model_is_reported_before_generation(monkeypatch):
    monkeypatch.setattr(
        "nodes.director.resolve_upscale_model_name",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("通用 AI 超分模型不存在：missing.pth")),
    )

    with pytest.raises(RequestError, match="AI 超分.*不存在"):
        MiniMaxH3DirectorPlus().build(
            mode="T2VA",
            prompt="镜头缓慢推进。",
            duration=5,
            width=2560,
            height=1440,
            voice_mode="none",
            ref_image_size="match",
            performance_preset="稳定质量",
            postprocess_mode="ai_upscale",
            ai_upscale_model="missing.pth",
            resolution_preset="2K QHD",
            timeline_data="{}",
            target_dialogue="",
            reference_transcript="",
        )


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


def test_t2va_ignores_stale_uploaded_images_before_loading_them(monkeypatch, caplog):
    def fail_if_loaded(filename):
        raise AssertionError(f"T2VA 不应加载残留图片：{filename}")

    monkeypatch.setattr("nodes.director.load_uploaded_image", fail_if_loaded)
    caplog.set_level("INFO", logger="MiniMaxH3.DirectorPlus")

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="纯文本镜头。",
        duration=5,
        width=1344,
        height=768,
        voice_mode="none",
        ref_image_size="match",
        performance_preset="稳定质量",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
        first_image_file="stale-first.png",
        last_image_file="stale-last.png",
        reference_image_1_file="stale-reference.png",
    )

    assert guide["first_frame"] is None
    assert guide["last_frame"] is None
    assert guide["ref_images"] == {}
    assert any("T2VA" in warning and "不兼容" in warning for warning in guide["warnings"])
    assert "mode=T2VA backend=fl2va_model" in caplog.text
    assert "first_frame=False last_frame=False ref_images=0" in caplog.text


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


@pytest.mark.parametrize("preset", ["稳定质量", "质量优先加速", "极速4步", "参考图加速"])
def test_all_non_low_vram_presets_cap_long_h3_sampling_to_official_canvas(preset):
    native_width, native_height, upscale_required = native_resolution_for_request(
        1920,
        1088,
        15,
        preset,
        "16:9",
    )

    assert (native_width, native_height) == (1344, 768)
    assert upscale_required is True


def test_two_stage_build_reports_first_second_and_final_sizes(monkeypatch):
    normal_probes = []
    monkeypatch.setattr(
        "nodes.director.probe_vsr_capability",
        lambda quality, device_id=0: normal_probes.append((quality, device_id)) or True,
    )

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=15,
        width=2560,
        height=1440,
        aspect_ratio="16:9",
        resolution_preset="2K QHD",
        voice_mode="none",
        ref_image_size="match",
        performance_preset="质量优先二采样",
        postprocess_mode="rtx_vsr",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert (guide["first_stage_width"], guide["first_stage_height"]) == (1280, 704)
    assert (guide["second_stage_width"], guide["second_stage_height"]) == (1920, 1056)
    assert (guide["target_width"], guide["target_height"]) == (2560, 1440)
    assert guide["quality_basis"] == "H3 神经 latent 二采"
    assert guide["vram_safety_tier"] == "28gb_plus"
    assert guide["resolved_two_stage_route"] == "trained_latent_fl"
    assert guide["final_upscale_scale_x"] == pytest.approx(2560 / 1920)
    assert guide["final_upscale_scale_y"] == pytest.approx(1440 / 1056)
    assert guide["required_assets"] == ["test-upscaler", "test-lora"]
    assert guide["rtx_quality_requested"] == "HIGH"
    assert guide["rtx_quality"] == "HIGHBITRATE_ULTRA"
    assert guide["rtx_deblur_mode"] == "off"
    assert any("已关闭不稳定的 DEBLUR_LOW 双效果链" in warning for warning in guide["warnings"])
    assert normal_probes == [("HIGHBITRATE_ULTRA", 0)]


def test_quality_two_stage_fhd_uses_balanced_downscale_without_vsr_probe(monkeypatch):
    probes = []
    monkeypatch.setattr(
        "nodes.director.probe_vsr_capability",
        lambda *args, **kwargs: probes.append((args, kwargs)),
    )

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=15,
        width=1920,
        height=1080,
        aspect_ratio="16:9",
        resolution_preset="1080p FHD",
        voice_mode="none",
        ref_image_size="match",
        performance_preset="质量优先二采样",
        postprocess_mode="rtx_vsr",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert (guide["first_stage_width"], guide["first_stage_height"]) == (1344, 768)
    assert (guide["second_stage_width"], guide["second_stage_height"]) == (2016, 1152)
    assert (guide["target_width"], guide["target_height"]) == (1920, 1080)
    assert guide["postprocess_path"] == "balanced_fhd_downscale"
    assert guide["upscale_method"] == "aspect_lanczos_downscale"
    assert guide["upscale_required"] is False
    assert any("1080p 平衡二采" in warning for warning in guide["warnings"])
    assert probes == []


def test_quality_two_stage_portrait_fhd_uses_balanced_downscale(monkeypatch):
    monkeypatch.setattr(
        "nodes.director.probe_vsr_capability",
        lambda *args, **kwargs: pytest.fail("FHD balanced route must not probe RTX VSR"),
    )

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=15,
        width=1080,
        height=1920,
        aspect_ratio="9:16",
        resolution_preset="1080p FHD",
        voice_mode="none",
        ref_image_size="match",
        performance_preset="质量优先二采样",
        postprocess_mode="rtx_vsr",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert (guide["first_stage_width"], guide["first_stage_height"]) == (768, 1344)
    assert (guide["second_stage_width"], guide["second_stage_height"]) == (1152, 2016)
    assert guide["postprocess_path"] == "balanced_fhd_downscale"


def test_regular_rtx_vsr_uses_only_normal_probe_and_disables_deblur(monkeypatch):
    normal_probes = []
    monkeypatch.setattr(
        "nodes.director.probe_vsr_capability",
        lambda quality, device_id=0: normal_probes.append((quality, device_id)) or True,
    )

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=5,
        width=2560,
        height=1440,
        aspect_ratio="16:9",
        resolution_preset="2K QHD",
        voice_mode="none",
        ref_image_size="match",
        performance_preset="稳定质量",
        postprocess_mode="rtx_vsr",
        rtx_quality="ULTRA",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert guide["postprocess_path"] == "rtx_vsr"
    assert guide["rtx_deblur_mode"] == "off"
    assert normal_probes == [("ULTRA", 0)]


def test_quality_two_stage_uses_normal_probe_failure_and_preserves_error(monkeypatch):
    normal_probes = []
    source_error = RuntimeError("HIGHBITRATE_ULTRA 链路不可用")
    monkeypatch.setattr(
        "nodes.director.probe_vsr_capability",
        lambda *args, **kwargs: (_ for _ in ()).throw(source_error),
    )

    with pytest.raises(RequestError, match="HIGHBITRATE_ULTRA 链路不可用") as error:
        MiniMaxH3DirectorPlus().build(
            mode="T2VA",
            prompt="镜头缓慢推进。",
            duration=15,
            width=2560,
            height=1440,
            aspect_ratio="16:9",
            resolution_preset="2K QHD",
            voice_mode="none",
            ref_image_size="match",
            performance_preset="质量优先二采样",
            postprocess_mode="rtx_vsr",
            timeline_data="{}",
            target_dialogue="",
            reference_transcript="",
        )

    assert error.value.__cause__ is source_error
    assert normal_probes == []


def test_two_stage_4k_build_uses_1080p_neural_basis(monkeypatch):
    monkeypatch.setattr("nodes.director.probe_vsr_capability", lambda *args, **kwargs: True)

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=15,
        width=3840,
        height=2160,
        aspect_ratio="16:9",
        resolution_preset="4K UHD",
        voice_mode="none",
        ref_image_size="match",
        performance_preset="质量优先二采样",
        postprocess_mode="rtx_vsr",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert (guide["first_stage_width"], guide["first_stage_height"]) == (1280, 704)
    assert (guide["second_stage_width"], guide["second_stage_height"]) == (1920, 1056)
    assert guide["final_upscale_scale_x"] == pytest.approx(2.0)
    assert guide["final_upscale_scale_y"] == pytest.approx(2160 / 1056)


def test_small_two_stage_target_uses_neural_output_without_redundant_vsr(monkeypatch):
    checked = []
    monkeypatch.setattr(
        "nodes.director.probe_vsr_capability",
        lambda *args, **kwargs: checked.append(True),
    )

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=4,
        width=1152,
        height=768,
        aspect_ratio="3:2",
        resolution_preset="0.83 MP",
        voice_mode="none",
        ref_image_size="match",
        performance_preset="质量优先二采样",
        postprocess_mode="rtx_vsr",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert (guide["first_stage_width"], guide["first_stage_height"]) == (768, 512)
    assert (guide["second_stage_width"], guide["second_stage_height"]) == (1152, 768)
    assert (guide["target_width"], guide["target_height"]) == (1152, 768)
    assert guide["postprocess_path"] == "native_bypass"
    assert guide["upscale_required"] is False
    assert checked == []


def test_two_stage_rejects_busy_gpu_before_vsr_probe(monkeypatch):
    checked = []
    monkeypatch.setattr("nodes.director._cuda_memory_gb", lambda: (32.0, 8.0))
    monkeypatch.setattr(
        "nodes.director.probe_vsr_capability",
        lambda *args, **kwargs: checked.append(True),
    )

    with pytest.raises(RequestError, match="当前可用显存"):
        MiniMaxH3DirectorPlus().build(
            mode="T2VA",
            prompt="镜头缓慢推进。",
            duration=15,
            width=2560,
            height=1440,
            aspect_ratio="16:9",
            resolution_preset="2K QHD",
            voice_mode="none",
            ref_image_size="match",
            performance_preset="质量优先二采样",
            postprocess_mode="rtx_vsr",
            timeline_data="{}",
            target_dialogue="",
            reference_transcript="",
        )

    assert checked == []


def test_low_vram_two_stage_builds_safe_six_second_fhd_plan(monkeypatch):
    normal_probes = []
    resolved_models = []
    monkeypatch.setattr("nodes.director._cuda_memory_gb", lambda: (8.0, 7.0))
    monkeypatch.setattr(
        "nodes.director.probe_vsr_capability",
        lambda quality, device_id=0: normal_probes.append((quality, device_id)) or True,
    )
    monkeypatch.setattr(
        "nodes.director.resolve_upscale_model_name",
        lambda model, scale: resolved_models.append((model, scale)) or "RealESRGAN_x2plus.pth",
    )

    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=6,
        width=1920,
        height=1080,
        aspect_ratio="16:9",
        resolution_preset="1080p FHD",
        voice_mode="none",
        ref_image_size="match",
        performance_preset="低显存二采",
        postprocess_mode="ai_upscale",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert guide["performance_preset"] == "low_vram_two_stage"
    assert guide["vram_safety_tier"] == "8gb_low_vram_two_stage"
    assert 280_000 <= guide["first_stage_width"] * guide["first_stage_height"] <= 340_000
    assert 620_000 <= guide["second_stage_width"] * guide["second_stage_height"] <= 740_000
    assert 1.65 <= guide["final_upscale_scale"] <= 1.82
    assert guide["max_final_vsr_scale"] == pytest.approx(1.45 * (6 / 4) ** 0.5)
    assert (guide["target_width"], guide["target_height"]) == (1920, 1080)
    assert guide["ai_upscale_model"] == "RealESRGAN_x2plus.pth"
    assert guide["postprocess_path"] == "ai_upscale"
    assert guide["rtx_deblur_mode"] == "off"
    assert normal_probes == []
    assert len(resolved_models) == 1
    assert resolved_models[0][0] == "auto"
    assert resolved_models[0][1] == pytest.approx(1920 / guide["second_stage_width"])


def test_low_vram_two_stage_rejects_longer_video_before_vsr_probe(monkeypatch):
    checked = []
    monkeypatch.setattr("nodes.director._cuda_memory_gb", lambda: (8.0, 7.0))
    monkeypatch.setattr(
        "nodes.director.probe_vsr_capability",
        lambda *args, **kwargs: checked.append(True),
    )

    with pytest.raises(RequestError, match="只支持 4 到 6 秒"):
        MiniMaxH3DirectorPlus().build(
            mode="T2VA",
            prompt="镜头缓慢推进。",
            duration=7,
            width=1920,
            height=1080,
            aspect_ratio="16:9",
            resolution_preset="1080p FHD",
            voice_mode="none",
            ref_image_size="match",
            performance_preset="低显存二采",
            postprocess_mode="ai_upscale",
            timeline_data="{}",
            target_dialogue="",
            reference_transcript="",
        )

    assert checked == []


def test_low_vram_two_stage_rejects_x4_final_model_before_h3(monkeypatch):
    monkeypatch.setattr("nodes.director._cuda_memory_gb", lambda: (8.0, 7.0))
    monkeypatch.setattr(
        "nodes.director.resolve_upscale_model_name",
        lambda model, scale: "RealESRGAN_x4plus.pth",
    )

    with pytest.raises(RequestError, match="低显存二采.*X2"):
        MiniMaxH3DirectorPlus().build(
            mode="T2VA",
            prompt="镜头缓慢推进。",
            duration=4,
            width=1920,
            height=1080,
            aspect_ratio="16:9",
            resolution_preset="1080p FHD",
            voice_mode="none",
            ref_image_size="match",
            performance_preset="低显存二采",
            postprocess_mode="ai_upscale",
            ai_upscale_model="RealESRGAN_x4plus.pth",
            timeline_data="{}",
            target_dialogue="",
            reference_transcript="",
        )


def test_two_stage_rejects_missing_route_assets_before_vsr_probe(monkeypatch):
    checked = []
    monkeypatch.setattr(
        "nodes.director._trained_two_stage_dependency_report",
        lambda route: {
            "route": route,
            "ready": False,
            "missing": ["minimax_h3_latent_upscaler_3d_bf16.safetensors"],
            "required_assets": [],
        },
    )
    monkeypatch.setattr(
        "nodes.director.probe_vsr_capability",
        lambda *args, **kwargs: checked.append(True),
    )

    with pytest.raises(RequestError, match="训练型二采依赖缺失.*latent_upscaler"):
        MiniMaxH3DirectorPlus().build(
            mode="T2VA",
            prompt="镜头缓慢推进。",
            duration=15,
            width=2560,
            height=1440,
            aspect_ratio="16:9",
            resolution_preset="2K QHD",
            voice_mode="none",
            ref_image_size="match",
            performance_preset="质量优先二采样",
            postprocess_mode="rtx_vsr",
            timeline_data="{}",
            target_dialogue="",
            reference_transcript="",
        )

    assert checked == []


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


def test_same_size_rtx_request_uses_native_bypass_without_any_rtx_probe(monkeypatch):
    normal_probes = []
    monkeypatch.setattr(
        "nodes.director.probe_vsr_capability",
        lambda *args, **kwargs: normal_probes.append(True),
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
    assert guide["rtx_deblur_mode"] == "off"
    assert normal_probes == []


def test_low_vram_target_upscale_selects_rtx_vsr(monkeypatch):
    monkeypatch.setattr("nodes.director.probe_vsr_capability", lambda *args, **kwargs: True)

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


def test_rtx_vsr_capability_failure_is_reported_before_generation(monkeypatch):
    probe_error = RuntimeError(
        "RTX VSR 前置检查失败，尚未开始 H3 视频生成。\n"
        "详细错误：NvVFX_Load failed: The requested feature or capability was not found (code -14)"
    )
    monkeypatch.setattr(
        "nodes.director.probe_vsr_capability",
        lambda *args, **kwargs: (_ for _ in ()).throw(probe_error),
    )

    with pytest.raises(RequestError) as error:
        MiniMaxH3DirectorPlus().build(
            mode="T2VA",
            prompt="镜头缓慢推进。",
            duration=5,
            width=2560,
            height=1440,
            voice_mode="none",
            ref_image_size="match",
            performance_preset="稳定质量",
            postprocess_mode="rtx_vsr",
            rtx_quality="ULTRA",
            resolution_preset="2K QHD",
            timeline_data="{}",
            target_dialogue="",
            reference_transcript="",
        )

    assert str(error.value) == str(probe_error)
    assert str(error.value).count("RTX VSR 前置检查失败，尚未开始 H3 视频生成") == 1
    assert error.value.__cause__ is probe_error


def test_native_output_warns_when_large_final_target_is_not_upscaled():
    guide, *_ = MiniMaxH3DirectorPlus().build(
        mode="T2VA",
        prompt="镜头缓慢推进。",
        duration=5,
        width=2560,
        height=1440,
        voice_mode="none",
        ref_image_size="match",
        performance_preset="稳定质量",
        postprocess_mode="native",
        resolution_preset="2K QHD",
        timeline_data="{}",
        target_dialogue="",
        reference_transcript="",
    )

    assert guide["postprocess_path"] == "native_bypass"
    assert any("原生尺寸直出" in warning and "不会放大" in warning for warning in guide["warnings"])


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
