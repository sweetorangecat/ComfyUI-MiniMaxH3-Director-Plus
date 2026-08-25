import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import nodes.stream_output as stream_output


class _Folders:
    def get_output_directory(self):
        return "."

    def get_temp_directory(self):
        return "."

    def get_save_image_path(self, prefix, output_dir, width, height):
        return output_dir, prefix, 0, "", None


def _fake_dasiwa(captured, audio_value=None, encode_calls=None, audio_inputs=None):
    def audio_file(audio):
        if audio_inputs is None:
            assert audio is audio_value
        else:
            audio_inputs.append(audio)
        if audio_value is None:
            return None, None
        return ("audio.wav", 48000, 1), 2.0

    def encode(*args, **kwargs):
        if encode_calls is not None:
            encode_calls.append((args, kwargs))
        frame_chunks = args[6] if len(args) > 6 else kwargs["frame_generator"]
        captured.extend(list(frame_chunks()))
        return SimpleNamespace()

    return SimpleNamespace(
        _selected_bit_depth=lambda codec, bit_depth, source: 8,
        folder_paths=_Folders(),
        _format_filename_prefix=lambda value: value,
        find_ffmpeg=lambda: "ffmpeg",
        _encoded_frame_count=lambda source, pingpong: len(source) * (2 if pingpong else 1),
        _MAX_RAW_FRAME_CHUNK_BYTES=64 * 1024 * 1024,
        _frame_bytes=lambda chunk, bit_depth: chunk,
        _metadata_file=lambda prompt, extra: None,
        _audio_file=audio_file,
        _animated_image_settings=lambda container: None,
        _codec_candidates=lambda codec: ["H.264"],
        _auto_container_candidates=lambda codec, container: ["MP4"],
        _container_candidates=lambda codec, container: ["MP4"],
        _CONTAINER_EXTENSIONS={"MP4": ".mp4"},
        _output_filename=lambda filename, counter, extension, has_audio: f"{filename}{extension}",
        _encode_with_available_encoder=encode,
        _ANIMATED_IMAGE_SETTINGS={},
    )


def _combine(monkeypatch, guide, images, captured, audio=None, encode_calls=None, audio_inputs=None, **kwargs):
    monkeypatch.setattr(
        stream_output,
        "_dasiwa_video_module",
        lambda: _fake_dasiwa(captured, audio, encode_calls, audio_inputs),
    )
    monkeypatch.setattr(stream_output.os, "unlink", lambda path: None)
    return stream_output.MiniMaxH3StreamingVideoCombine().combine(
        images=images,
        guide=guide,
        frame_rate=24.0,
        codec="H.264",
        container="MP4",
        bit_depth="8-bit",
        quality=20,
        log_level="Standard",
        pingpong=False,
        save_metadata=False,
        filename_prefix="stream-test",
        save_output=True,
        pass_frames=kwargs.get("pass_frames", False),
        crop_to_audio=False,
        audio_codec="Auto",
        audio_bitrate="192k",
        save_first_frame=kwargs.get("save_first_frame", False),
        save_last_frame=kwargs.get("save_last_frame", False),
        audio=audio,
    )


def test_native_bypass_does_not_resize_or_require_vsr(monkeypatch):
    images = torch.arange(3 * 2 * 3 * 3, dtype=torch.float32).reshape(3, 2, 3, 3) / 100
    captured = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: (_ for _ in ()).throw(AssertionError("VSR loaded")))

    _combine(
        monkeypatch,
        {"target_width": 3, "target_height": 2, "postprocess_path": "native_bypass"},
        images,
        captured,
    )

    result = torch.cat(captured)
    assert result.shape == images.shape
    assert torch.equal(result, images)


def test_same_size_rtx_request_is_reported_as_native_bypass(monkeypatch):
    guide = {"native_width": 3, "native_height": 2, "target_width": 3, "target_height": 2,
             "postprocess_mode": "rtx_vsr", "upscale_required": False}
    assert stream_output._resolve_postprocess_path(guide, 3, 2) == "native_bypass"


def test_two_stage_actual_second_size_bypasses_redundant_vsr():
    guide = {
        "native_width": 896,
        "native_height": 512,
        "second_stage_width": 1344,
        "second_stage_height": 768,
        "target_width": 1344,
        "target_height": 768,
        "postprocess_path": "rtx_vsr",
    }
    assert stream_output._resolve_postprocess_path(guide, 1344, 768) == "native_bypass"


def test_prepare_postprocess_releases_h3_before_vsr(monkeypatch):
    calls = []
    monkeypatch.setattr(
        stream_output,
        "release_sampling_models",
        lambda: calls.append("released"),
        raising=False,
    )
    stream_output._prepare_postprocess_runtime(
        {"performance_preset": "quality_two_stage"},
        "rtx_vsr",
    )
    assert calls == ["released"]


def test_quality_two_stage_uses_crf_16_without_overriding_better_values():
    resolver = getattr(stream_output, "_resolved_encode_quality", None)
    assert callable(resolver), "缺少质量二采编码策略"
    guide = {"performance_preset": "quality_two_stage"}

    assert resolver(guide, "rtx_vsr", 20) == 16
    assert resolver(guide, "rtx_vsr", 14) == 14
    assert resolver({"performance_preset": "quality"}, "rtx_vsr", 20) == 20
    assert resolver(guide, "native_bypass", 20) == 16


def test_low_vram_two_stage_releases_h3_before_ai_x2_and_uses_crf_16():
    calls = []
    original = stream_output.release_sampling_models
    try:
        stream_output.release_sampling_models = lambda: calls.append("released")
        stream_output._prepare_postprocess_runtime(
            {"performance_preset": "low_vram_two_stage"},
            "ai_upscale",
        )
    finally:
        stream_output.release_sampling_models = original

    assert calls == ["released"]
    assert stream_output._resolved_encode_quality(
        {"performance_preset": "low_vram_two_stage"},
        "ai_upscale",
        20,
    ) == 16


def test_quality_two_stage_passes_crf_16_to_the_encoder(monkeypatch):
    images = torch.rand(2, 2, 3, 3)
    captured = []
    encode_calls = []
    tagged = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")
    monkeypatch.setattr(stream_output, "release_sampling_models", lambda: None)
    monkeypatch.setattr(
        stream_output,
        "_tag_h264_bt709",
        lambda ffmpeg, output_path: tagged.append((ffmpeg, output_path)) or True,
        raising=False,
    )

    class FakeProcessor:
        def __init__(self, api, quality, device_id, width, height):
            self.width, self.height = width, height

        def process(self, frame):
            return torch.zeros(self.height, self.width, 3)

        def close(self):
            pass

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeProcessor)
    _combine(
        monkeypatch,
        {
            "performance_preset": "quality_two_stage",
            "target_width": 6,
            "target_height": 4,
            "postprocess_path": "rtx_vsr",
            "rtx_quality": "HIGHBITRATE_ULTRA",
        },
        images,
        captured,
        encode_calls=encode_calls,
    )

    args, _ = encode_calls[0]
    assert args[9:11] == (16, 16)
    assert tagged == [("ffmpeg", ".\\stream-test.mp4")]


def test_h264_bt709_tagging_is_a_lossless_stream_copy(monkeypatch, tmp_path):
    tagger = getattr(stream_output, "_tag_h264_bt709", None)
    assert callable(tagger), "缺少 H.264 BT.709 无损标记"
    output_path = tmp_path / "video.mp4"
    output_path.write_bytes(b"encoded-video")
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        Path(command[-1]).write_bytes(output_path.read_bytes())
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(stream_output.subprocess, "run", fake_run)

    assert tagger("ffmpeg", str(output_path)) is True
    command, kwargs = commands[0]
    assert command[command.index("-c") + 1] == "copy"
    assert "-map_metadata" in command
    assert (
        command[command.index("-bsf:v") + 1]
        == "h264_metadata=video_full_range_flag=0:colour_primaries=1:"
        "transfer_characteristics=1:matrix_coefficients=1"
    )
    assert kwargs["check"] is True
    assert output_path.read_bytes() == b"encoded-video"


def test_h264_bt709_tagging_failure_preserves_original(monkeypatch, tmp_path, caplog):
    tagger = getattr(stream_output, "_tag_h264_bt709", None)
    assert callable(tagger), "缺少 H.264 BT.709 无损标记"
    output_path = tmp_path / "video.mp4"
    output_path.write_bytes(b"original-video")

    def fail_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial-remux")
        raise subprocess.CalledProcessError(1, command, stderr="metadata failed")

    monkeypatch.setattr(stream_output.subprocess, "run", fail_run)

    assert tagger("ffmpeg", str(output_path)) is False
    assert output_path.read_bytes() == b"original-video"
    assert list(tmp_path.glob("*.bt709-*.mp4")) == []
    assert "metadata failed" in caplog.text


def test_prepare_postprocess_does_not_unload_for_native_output(monkeypatch):
    calls = []
    monkeypatch.setattr(
        stream_output,
        "release_sampling_models",
        lambda: calls.append("released"),
        raising=False,
    )
    stream_output._prepare_postprocess_runtime(
        {"performance_preset": "quality_two_stage"},
        "native_bypass",
    )
    assert calls == []


@pytest.mark.parametrize("path", ["lanczos", "ai_upscale"])
def test_generic_upscale_paths_are_preserved_for_larger_targets(path):
    guide = {
        "native_width": 3,
        "native_height": 2,
        "target_width": 6,
        "target_height": 4,
        "postprocess_path": path,
    }

    assert stream_output._resolve_postprocess_path(guide, 3, 2) == path


def test_rtx_vsr_path_processes_frames_and_preserves_frame_count(monkeypatch):
    images = torch.rand(5, 2, 3, 3)
    captured = []
    processed = []
    instances = []
    encode_calls = []
    audio = object()

    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class FakeProcessor:
        def __init__(self, api, quality, device_id, width, height):
            instances.append(self)
            self.width, self.height = width, height

        def process(self, frame):
            processed.append(frame.clone())
            return torch.zeros(self.height, self.width, 3)

        def close(self):
            instances.remove(self)

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeProcessor)
    result = _combine(
        monkeypatch,
        {"target_width": 6, "target_height": 4, "postprocess_path": "rtx_vsr", "rtx_quality": "HIGH"},
        images,
        captured,
        audio=audio,
        encode_calls=encode_calls,
    )

    assert len(processed) == len(images)
    assert all(frame.shape == (3, 2, 3) for frame in processed)
    assert sum(len(chunk) for chunk in captured) == len(images)
    assert all(chunk.shape[1:] == (4, 6, 3) for chunk in captured)
    assert encode_calls[0][0][5] == 24.0
    assert encode_calls[0][0][12] == ("audio.wav", 48000, 1)
    assert result["ui"]["gifs"][0]["fps"] == 24.0
    assert instances == []


def test_rife_motion_smoothing_streams_seven_frames_at_48fps(monkeypatch):
    images = torch.stack([
        torch.full((2, 3, 3), float(index), dtype=torch.float32)
        for index in range(4)
    ])
    captured = []
    processed = []
    encode_calls = []
    rife_inputs = []
    probe_calls = []
    active_processors = []

    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")
    monkeypatch.setattr(
        stream_output,
        "probe_rife_capability",
        lambda model_name: probe_calls.append(model_name),
        raising=False,
    )
    def fake_rife_frames(source, model_name, pingpong=False):
        rife_inputs.append((source, model_name, pingpong))
        yield source[0]
        for index in range(len(source) - 1):
            yield (source[index] + source[index + 1]) / 2
            yield source[index + 1]

    monkeypatch.setattr(stream_output, "iter_rife_frames", fake_rife_frames, raising=False)

    class FakeVsrProcessor:
        def __init__(self, api, quality, device_id, width, height):
            active_processors.append(self)
            self.width, self.height = width, height

        def process(self, frame):
            processed.append(float(frame.mean()))
            return torch.full((self.height, self.width, 3), float(frame.mean()))

        def close(self):
            active_processors.remove(self)

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeVsrProcessor)

    result = _combine(
        monkeypatch,
        {
            "target_width": 6,
            "target_height": 4,
            "postprocess_path": "rtx_vsr",
            "rtx_quality": "ULTRA",
            "motion_smoothing": "rife_x2",
            "rife_model": "rife_v4.26.safetensors",
            "output_frame_multiplier": 2,
        },
        images,
        captured,
        encode_calls=encode_calls,
    )

    assert probe_calls == ["rife_v4.26.safetensors"]
    assert len(rife_inputs) == 1
    assert processed == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert sum(len(chunk) for chunk in captured) == 7
    assert all(len(chunk) <= 4 for chunk in captured)
    assert encode_calls[0][0][5] == 48.0
    assert result["ui"]["gifs"][0]["fps"] == 48.0
    assert result["ui"]["gifs"][0]["source_fps"] == 24.0
    assert result["ui"]["gifs"][0]["motion_smoothing"] == "rife_x2"
    assert active_processors == []


def test_auto_audio_loudness_is_applied_before_dasiwa_encoding(monkeypatch):
    waveform = torch.full((1, 2, 320), 10 ** (-31 / 20), dtype=torch.float32)
    audio = {"waveform": waveform, "sample_rate": 32000}
    audio_inputs = []

    _combine(
        monkeypatch,
        {
            "target_width": 3,
            "target_height": 2,
            "postprocess_path": "native_bypass",
            "audio_loudness": "auto",
        },
        torch.rand(3, 2, 3, 3),
        [],
        audio=audio,
        audio_inputs=audio_inputs,
    )

    assert len(audio_inputs) == 1
    normalized = audio_inputs[0]
    assert normalized is not audio
    assert normalized["sample_rate"] == 32000
    assert torch.max(torch.abs(normalized["waveform"])).item() == pytest.approx(
        10 ** (-1.5 / 20), rel=1e-5
    )
    assert torch.equal(audio["waveform"], waveform)


def test_original_audio_loudness_preserves_audio_object():
    audio = {"waveform": torch.rand(1, 2, 16), "sample_rate": 32000}
    assert stream_output._normalize_output_audio(audio, "original") is audio


def test_auto_audio_loudness_keeps_silence_and_caps_gain():
    silent = {"waveform": torch.zeros(1, 1, 16), "sample_rate": 32000}
    assert stream_output._normalize_output_audio(silent, "auto") is silent

    quiet = {"waveform": torch.full((1, 1, 16), 1e-6), "sample_rate": 32000}
    normalized = stream_output._normalize_output_audio(quiet, "auto")
    assert normalized["waveform"].abs().max().item() == pytest.approx(
        1e-6 * 10 ** (30 / 20), rel=1e-5
    )


def test_auto_audio_loudness_attenuates_clipping_source():
    audio = {"waveform": torch.ones(1, 1, 16), "sample_rate": 32000}
    normalized = stream_output._normalize_output_audio(audio, "auto")
    assert normalized["waveform"].abs().max().item() == pytest.approx(
        10 ** (-1.5 / 20), rel=1e-5
    )


def test_auto_audio_loudness_sanitizes_non_finite_samples():
    waveform = torch.tensor([[[float("nan"), float("inf"), -float("inf"), 0.1]]])
    audio = {"waveform": waveform, "sample_rate": 32000}

    normalized = stream_output._normalize_output_audio(audio, "auto")

    assert torch.isfinite(normalized["waveform"]).all()
    assert torch.equal(normalized["waveform"][..., :3], torch.zeros(1, 1, 3))
    assert normalized["waveform"].abs().max().item() == pytest.approx(
        10 ** (-1.5 / 20), rel=1e-5
    )


def test_motion_smoothing_off_never_loads_rife(monkeypatch):
    images = torch.rand(3, 2, 3, 3)
    captured = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")
    monkeypatch.setattr(
        stream_output,
        "probe_rife_capability",
        lambda *args: (_ for _ in ()).throw(AssertionError("RIFE probe")),
        raising=False,
    )
    monkeypatch.setattr(
        stream_output,
        "iter_rife_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("RIFE frames")),
        raising=False,
    )

    class FakeProcessor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            return frame.movedim(0, -1).contiguous()

        def close(self):
            pass

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeProcessor)

    result = _combine(
        monkeypatch,
        {
            "target_width": 3,
            "target_height": 2,
            "postprocess_path": "rtx_vsr",
            "motion_smoothing": "off",
        },
        images,
        captured,
    )

    assert sum(len(chunk) for chunk in captured) == 3
    assert result["ui"]["gifs"][0]["fps"] == 24.0


def test_quality_two_stage_rejects_rife_before_output_model_loading(monkeypatch):
    monkeypatch.setattr(
        stream_output,
        "load_vsr_api",
        lambda: (_ for _ in ()).throw(AssertionError("不得加载 VSR")),
    )

    with pytest.raises(ValueError, match="质量优先二采样.*RIFE"):
        _combine(
            monkeypatch,
            {
                "performance_preset": "quality_two_stage",
                "target_width": 6,
                "target_height": 4,
                "postprocess_path": "rtx_vsr",
                "motion_smoothing": "rife_x2",
            },
            torch.rand(3, 2, 3, 3),
            [],
        )


def test_rife_failure_does_not_retry_other_video_codecs(monkeypatch):
    images = torch.rand(3, 2, 3, 3)
    encode_attempts = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")
    monkeypatch.setattr(stream_output, "probe_rife_capability", lambda model: model)

    def failing_rife_frames(*args, **kwargs):
        yield images[0]
        raise stream_output.RifeProcessingError("RIFE 推理测试失败")

    monkeypatch.setattr(stream_output, "iter_rife_frames", failing_rife_frames)

    class FakeVsrProcessor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            return frame.movedim(0, -1).contiguous()

        def close(self):
            pass

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeVsrProcessor)
    dasiwa = _fake_dasiwa([])
    dasiwa._codec_candidates = lambda codec: ["H.264", "VP9"]

    def encode(*args, **kwargs):
        encode_attempts.append(args[1])
        list(args[6]())

    dasiwa._encode_with_available_encoder = encode
    monkeypatch.setattr(stream_output, "_dasiwa_video_module", lambda: dasiwa)
    monkeypatch.setattr(stream_output.os, "unlink", lambda path: None)

    with pytest.raises(RuntimeError, match="RIFE 推理测试失败"):
        stream_output.MiniMaxH3StreamingVideoCombine().combine(
            images=images,
            guide={
                "target_width": 6,
                "target_height": 4,
                "postprocess_path": "rtx_vsr",
                "motion_smoothing": "rife_x2",
            },
            frame_rate=24.0,
            codec="Auto",
            container="Auto",
            bit_depth="8-bit",
            quality=20,
            log_level="Standard",
            pingpong=False,
            save_metadata=False,
            filename_prefix="stream-test",
            save_output=True,
            pass_frames=False,
            crop_to_audio=False,
            audio_codec="Auto",
            audio_bitrate="192k",
            save_first_frame=False,
            save_last_frame=False,
        )

    assert encode_attempts == ["H.264"]


def test_output_path_never_materializes_full_target_batch(monkeypatch):
    images = torch.rand(9, 2, 3, 3)
    captured = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class FakeProcessor:
        def __init__(self, api, quality, device_id, width, height):
            self.width, self.height = width, height

        def process(self, frame):
            return torch.zeros(self.height, self.width, 3)

        def close(self):
            pass

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeProcessor)
    _combine(
        monkeypatch,
        {"target_width": 12, "target_height": 8, "postprocess_path": "rtx_vsr", "rtx_quality": "HIGH"},
        images,
        captured,
    )

    assert all(chunk.shape[0] <= 4 for chunk in captured)
    assert all(chunk.shape[1:] == (8, 12, 3) for chunk in captured)
    assert all(chunk.shape != (len(images), 8, 12, 3) for chunk in captured)


def test_vsr_iterator_preserves_pingpong_order(monkeypatch):
    images = torch.stack([torch.full((2, 3, 3), float(index)) for index in range(5)])
    processed = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class FakeProcessor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            processed.append(int(frame[0, 0, 0].item()))
            return frame.movedim(0, -1).contiguous()

        def close(self):
            pass

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeProcessor)

    chunks = list(stream_output._iter_vsr_frame_chunks(
        images, 3, 2, pingpong=True, max_chunk_bytes=2 * 3 * 3,
    ))

    assert processed == [0, 1, 2, 3, 4, 3, 2, 1]
    assert sum(len(chunk) for chunk in chunks) == 8


def test_vsr_stream_closes_upstream_generator_when_consumer_stops(monkeypatch):
    closed = []
    processor_closed = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class FakeProcessor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            return frame.movedim(0, -1).contiguous()

        def close(self):
            processor_closed.append(True)

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeProcessor)

    def frames():
        try:
            while True:
                yield torch.zeros(2, 3, 3)
        finally:
            closed.append(True)

    chunks = stream_output._iter_vsr_frame_stream(
        frames(), 3, 2, max_chunk_bytes=2 * 3 * 3,
    )
    next(chunks)
    chunks.close()

    assert closed == [True]
    assert processor_closed == [True]


def test_downscale_path_uses_cpu_resize_without_loading_vsr(monkeypatch):
    images = torch.rand(5, 8, 12, 3)
    captured = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: (_ for _ in ()).throw(AssertionError("VSR loaded")))

    result = _combine(
        monkeypatch,
        {"target_width": 6, "target_height": 4, "postprocess_path": "downscale"},
        images,
        captured,
    )

    output = torch.cat(captured)
    assert output.shape == (5, 4, 6, 3)
    assert result["ui"]["postprocess_path"] == "downscale"
    assert result["result"][0].shape[0] == 0


def test_discarded_frames_use_independent_zero_storage_cpu_image(monkeypatch):
    images = torch.rand(5, 8, 12, 4)
    captured = []

    result = _combine(
        monkeypatch,
        {"target_width": 6, "target_height": 4, "postprocess_path": "downscale"},
        images,
        captured,
    )

    output = result["result"][0]
    assert output.shape == (0, 4, 6, 3)
    assert output.device.type == "cpu"
    assert output.untyped_storage().nbytes() == 0
    assert output.data_ptr() != images.data_ptr()
    assert output._base is None
    assert not output.is_pinned()


def test_native_path_is_only_path_that_returns_pass_frames(monkeypatch):
    images = torch.rand(3, 2, 3, 3)
    captured = []

    result = _combine(
        monkeypatch,
        {"target_width": 3, "target_height": 2, "postprocess_path": "native_bypass"},
        images,
        captured,
        pass_frames=True,
    )

    assert torch.equal(result["result"][0], images)


def test_vsr_path_creates_one_processor_per_encoder_retry(monkeypatch):
    images = torch.rand(3, 2, 3, 3)
    captured = []
    encode_calls = []
    processor_instances = []
    processor_count = 0
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class FakeProcessor:
        def __init__(self, *args):
            nonlocal processor_count
            processor_count += 1
            processor_instances.append(self)

        def process(self, frame):
            return frame.movedim(0, -1).contiguous()

        def close(self):
            processor_instances.remove(self)

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeProcessor)

    def fail_encode(*args, **kwargs):
        encode_calls.append((args, kwargs))
        list(args[6]())
        if len(encode_calls) == 1:
            raise RuntimeError("first encoder failed after VSR")
        return SimpleNamespace()

    def fake_dasiwa(captured, audio_value=None, encode_calls=None):
        module = _fake_dasiwa(captured, audio_value, encode_calls)
        module._codec_candidates = lambda codec: ["H.264", "H.265 (HEVC)"]
        module._auto_container_candidates = lambda codec, container: ["MP4"]
        module._encode_with_available_encoder = fail_encode
        return module

    monkeypatch.setattr(stream_output, "_dasiwa_video_module", lambda: fake_dasiwa(captured, encode_calls=encode_calls))

    stream_output.MiniMaxH3StreamingVideoCombine().combine(
        images=images,
        guide={"target_width": 6, "target_height": 4, "postprocess_path": "rtx_vsr", "rtx_quality": "HIGH"},
        frame_rate=24.0, codec="Auto", container="MP4", bit_depth="8-bit", quality=20,
        log_level="Standard", pingpong=False, save_metadata=False, filename_prefix="stream-test",
        save_output=True, pass_frames=False, crop_to_audio=False, audio_codec="Auto",
        audio_bitrate="192k", save_first_frame=False, save_last_frame=False,
    )

    assert len(encode_calls) == 2
    assert processor_count == 2
    assert processor_instances == []


def test_vsr_cleanup_failure_stops_auto_codec_retry_after_partial_consumption(monkeypatch):
    images = torch.stack([torch.full((2, 3, 3), float(index)) for index in range(3)])
    processed = []
    encode_calls = []
    processor_count = 0
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class FakeProcessor:
        def __init__(self, *args):
            nonlocal processor_count
            processor_count += 1

        def process(self, frame):
            processed.append(int(frame[0, 0, 0].item()))
            return torch.zeros(4, 6, 3)

        def close(self):
            raise RuntimeError("CUDA cleanup failed")

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeProcessor)

    def consume_one_chunk(*args, **kwargs):
        encode_calls.append((args, kwargs))
        chunks = args[6]()
        next(chunks)
        chunks.close()
        return SimpleNamespace()

    dasiwa = _fake_dasiwa([], encode_calls=encode_calls)
    dasiwa._MAX_RAW_FRAME_CHUNK_BYTES = 6 * 4 * 3
    dasiwa._codec_candidates = lambda codec: ["H.264", "H.265 (HEVC)"]
    dasiwa._auto_container_candidates = lambda codec, container: ["MP4"]
    dasiwa._encode_with_available_encoder = consume_one_chunk
    monkeypatch.setattr(stream_output, "_dasiwa_video_module", lambda: dasiwa)

    with pytest.raises(stream_output._VsrProcessingError, match="CUDA cleanup failed"):
        stream_output.MiniMaxH3StreamingVideoCombine().combine(
            images=images,
            guide={"target_width": 6, "target_height": 4, "postprocess_path": "rtx_vsr"},
            frame_rate=24.0, codec="Auto", container="MP4", bit_depth="8-bit", quality=20,
            log_level="Standard", pingpong=False, save_metadata=False,
            filename_prefix="stream-test", save_output=True, pass_frames=False,
            crop_to_audio=False, audio_codec="Auto", audio_bitrate="192k",
            save_first_frame=False, save_last_frame=False,
        )

    assert len(encode_calls) == 1
    assert processor_count == 1
    assert processed == [0]


def test_vsr_cleanup_failure_does_not_mask_processing_error(monkeypatch):
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class FakeProcessor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            raise RuntimeError("VSR process failed")

        def close(self):
            raise RuntimeError("CUDA cleanup failed")

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeProcessor)

    with pytest.raises(stream_output._VsrProcessingError, match="VSR process failed"):
        next(stream_output._iter_vsr_frame_chunks(torch.rand(1, 2, 3, 3), 6, 4))


def test_vsr_dependency_error_is_not_replaced_by_cpu_resize(monkeypatch):
    images = torch.rand(3, 2, 3, 3)
    captured = []
    monkeypatch.setattr(
        stream_output,
        "load_vsr_api",
        lambda: (_ for _ in ()).throw(RuntimeError("nvidia-vfx missing")),
    )
    monkeypatch.setattr(
        stream_output,
        "_iter_resized_frame_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bicubic fallback")),
    )

    with pytest.raises(RuntimeError, match="nvidia-vfx missing"):
        _combine(
            monkeypatch,
            {"target_width": 6, "target_height": 4, "postprocess_path": "rtx_vsr", "rtx_quality": "HIGH"},
            images,
            captured,
        )


def test_vsr_save_frames_use_vsr_processor_and_metadata_path(monkeypatch):
    images = torch.rand(4, 2, 3, 3)
    captured = []
    saved = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")
    monkeypatch.setattr(
        stream_output,
        "_save_vsr_frame",
        lambda *args, **kwargs: saved.append((args[1], args[7])) or f"{args[7]}-{args[1]}.png",
    )

    class FakeProcessor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            return frame.movedim(0, -1).contiguous()

        def close(self):
            pass

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeProcessor)
    result = _combine(
        monkeypatch,
        {"target_width": 6, "target_height": 4, "postprocess_path": "rtx_vsr", "rtx_quality": "HIGH"},
        images,
        captured,
        save_first_frame=True,
        save_last_frame=True,
    )

    assert [index for index, _ in saved] == [0, 3]
    assert all(path == "first" or path == "last" for _, path in saved)
    assert result["ui"]["postprocess_path"] == "rtx_vsr"
    assert result["ui"]["images"][0]["postprocess_path"] == "rtx_vsr"


def test_vsr_save_frame_drops_alpha_before_chw_processing(monkeypatch, tmp_path):
    processed = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class FakeProcessor:
        def __init__(self, *args):
            self.width, self.height = args[-2:]

        def process(self, frame):
            processed.append(frame)
            return torch.zeros(self.height, self.width, 3)

        def close(self):
            pass

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", FakeProcessor)
    source = torch.rand(1, 2, 3, 4)
    stream_output._save_vsr_frame(
        source, 0, 6, 4, "HIGH", 0, str(tmp_path / "video.mp4"), "first"
    )

    assert processed[0].shape == (3, 2, 3)


def test_center_crop_to_target_aspect_preserves_center_even_dimensions_and_identity():
    wide = torch.arange(1056 * 1920 * 3, dtype=torch.float32).reshape(1056, 1920, 3)
    cropped = stream_output._center_crop_to_target_aspect(wide, 1920, 1080)

    assert cropped.shape == (1056, 1876, 3)
    assert cropped.data_ptr() == wide[:, 22:1898].data_ptr()
    assert torch.equal(cropped, wide[:, 22:1898])

    tall = torch.zeros(1200, 800, 4, dtype=torch.float16)
    vertical = stream_output._center_crop_to_target_aspect(tall, 16, 9)
    assert vertical.shape == (450, 800, 4)
    assert vertical.data_ptr() == tall[375:825].data_ptr()

    same = torch.zeros(1080, 1920, 3)
    assert stream_output._center_crop_to_target_aspect(same, 16, 9) is same
    with pytest.raises(ValueError):
        stream_output._center_crop_to_target_aspect(same, 0, 9)


@pytest.mark.parametrize("shape", [(9, 16, 3), (1, 16, 3), (0, 16, 3)])
def test_center_crop_rejects_non_h3_dual_chain_source_dimensions(shape):
    with pytest.raises(ValueError, match="偶数"):
        stream_output._center_crop_to_target_aspect(torch.zeros(shape), 16, 9)


@pytest.mark.parametrize(
    ("guide", "postprocess_path", "expected"),
    [
        ({"performance_preset": "quality_two_stage", "rtx_deblur_mode": "DEBLUR_LOW"}, "rtx_vsr", False),
        ({"performance_preset": "质量优先二采样", "rtx_deblur_mode": "DEBLUR_LOW"}, "rtx_vsr", False),
        ({"performance_preset": "low_vram_two_stage", "rtx_deblur_mode": "DEBLUR_LOW"}, "rtx_vsr", False),
        ({"performance_preset": "quality", "rtx_deblur_mode": "DEBLUR_LOW"}, "rtx_vsr", False),
        ({"performance_preset": "quality_two_stage"}, "rtx_vsr", False),
        ({"performance_preset": "quality_two_stage", "rtx_deblur_mode": "DEBLUR_LOW"}, "lanczos", False),
    ],
)
def test_deblur_route_is_narrowly_isolated(guide, postprocess_path, expected):
    assert stream_output._should_use_deblur_before_upscale(guide, postprocess_path) is expected


def test_dual_vsr_stream_crops_before_chw_and_closes_upstream(monkeypatch):
    inputs, created, closed = [], [], []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class OrdinaryProcessor:
        def __init__(self, *args):
            raise AssertionError("dual route must not construct ordinary VSR")

    class DualProcessor:
        def __init__(self, api, quality, device_id, input_width, input_height, output_width, output_height):
            created.append((api, quality, device_id, input_width, input_height, output_width, output_height))
            self.output_width = output_width
            self.output_height = output_height

        def process(self, frame):
            inputs.append(frame)
            return torch.full(
                (self.output_height, self.output_width, 3), float(frame[0, 0, 0])
            )

        def close(self):
            closed.append(True)

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", OrdinaryProcessor)
    monkeypatch.setattr(stream_output, "DeblurVsrFrameProcessor", DualProcessor)
    upstream_closed = []

    def frames():
        try:
            for value in range(5):
                yield torch.full((1056, 1920, 3), float(value))
        finally:
            upstream_closed.append(True)

    chunks = list(stream_output._iter_vsr_frame_stream(
        frames(), 1920, 1080, quality="ULTRA", deblur_before_upscale=True,
        max_chunk_bytes=1920 * 1080 * 3,
    ))

    assert created == [("api", "ULTRA", 0, 1876, 1056, 1920, 1080)]
    assert [frame.shape for frame in inputs] == [(3, 1056, 1876)] * 5
    assert sum(chunk.shape[0] for chunk in chunks) == 5
    assert all(chunk.shape[1:] == (1080, 1920, 3) for chunk in chunks)
    assert [float(chunk[0, 0, 0, 0]) for chunk in chunks] == [0, 1, 2, 3, 4]
    assert closed == [True]
    assert upstream_closed == [True]


def test_ordinary_vsr_stream_preserves_full_frame_without_aspect_crop(monkeypatch):
    processed = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class Processor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            processed.append(frame)
            return torch.zeros(1440, 2560, 3)

        def close(self):
            pass

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", Processor)
    list(stream_output._iter_vsr_frame_chunks(
        torch.zeros(1, 704, 1280, 3), 2560, 1440
    ))

    assert [frame.shape for frame in processed] == [(3, 704, 1280)]


def test_rife_vsr_stream_preserves_full_frame_without_aspect_crop(monkeypatch):
    processed = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class Processor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            processed.append(frame)
            return torch.zeros(1440, 2560, 3)

        def close(self):
            pass

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", Processor)
    list(stream_output._iter_vsr_frame_stream(
        iter([torch.zeros(704, 1280, 3)]), 2560, 1440,
        deblur_before_upscale=False,
    ))

    assert [frame.shape for frame in processed] == [(3, 704, 1280)]


def test_save_ordinary_vsr_frame_preserves_full_frame_without_aspect_crop(monkeypatch, tmp_path):
    processed = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class Processor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            processed.append(frame)
            return torch.zeros(1440, 2560, 3)

        def close(self):
            pass

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", Processor)
    stream_output._save_vsr_frame(
        torch.zeros(1, 704, 1280, 3), 0, 2560, 1440, "HIGH", 0,
        str(tmp_path / "video.mp4"), "first",
    )

    assert [frame.shape for frame in processed] == [(3, 704, 1280)]


def test_save_vsr_frame_preserves_processing_error_when_cleanup_also_fails(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class Processor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            raise RuntimeError("process failure")

        def close(self):
            raise RuntimeError("cleanup failure")

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", Processor)

    with pytest.raises(stream_output._VsrProcessingError, match="process failure"):
        stream_output._save_vsr_frame(
            torch.zeros(1, 2, 2, 3), 0, 2, 2, "HIGH", 0,
            str(tmp_path / "video.mp4"), "first",
        )

    assert "cleanup failure" in caplog.text


def test_save_vsr_frame_surfaces_cleanup_error_without_processing_error(monkeypatch, tmp_path):
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")

    class Processor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            return torch.zeros(2, 2, 3)

        def close(self):
            raise RuntimeError("cleanup failure")

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", Processor)

    with pytest.raises(stream_output._VsrProcessingError, match="cleanup failure"):
        stream_output._save_vsr_frame(
            torch.zeros(1, 2, 2, 3), 0, 2, 2, "HIGH", 0,
            str(tmp_path / "video.mp4"), "first",
        )


def test_quality_two_stage_combine_and_exports_use_ordinary_processor(monkeypatch):
    images = torch.rand(3, 1056, 1920, 3)
    captured, created, saved = [], [], []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")
    monkeypatch.setattr(stream_output, "release_sampling_models", lambda: None)
    monkeypatch.setattr(stream_output, "_tag_h264_bt709", lambda *args: False)

    class OrdinaryProcessor:
        def __init__(self, api, quality, device_id, output_width, output_height):
            created.append((api, quality, device_id, output_width, output_height))
            self.height, self.width = output_height, output_width

        def process(self, frame):
            assert frame.shape == (3, 1056, 1920)
            return torch.zeros(self.height, self.width, 3)

        def close(self):
            pass

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", OrdinaryProcessor)
    monkeypatch.setattr(
        stream_output, "_save_vsr_frame",
        lambda *args, **kwargs: saved.append((args[1], kwargs.get("deblur_before_upscale"))) or "frame.png",
    )

    _combine(
        monkeypatch,
        {"performance_preset": "quality_two_stage", "rtx_deblur_mode": "off",
         "target_width": 1920, "target_height": 1080, "postprocess_path": "rtx_vsr"},
        images, captured, save_first_frame=True, save_last_frame=True,
    )

    assert len(created) == 1
    assert created[0] == ("api", "HIGH", 0, 1920, 1080)
    assert saved == [(0, False), (2, False)]


def test_quality_vsr_failure_does_not_retry_other_codecs(monkeypatch):
    images = torch.rand(2, 1056, 1920, 3)
    attempts = []
    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")
    monkeypatch.setattr(stream_output, "release_sampling_models", lambda: None)

    class OrdinaryProcessor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            raise RuntimeError("dual failure")

        def close(self):
            pass

    monkeypatch.setattr(stream_output, "VsrFrameProcessor", OrdinaryProcessor)
    dasiwa = _fake_dasiwa([])
    dasiwa._codec_candidates = lambda codec: ["H.264", "VP9"]

    def encode(*args, **kwargs):
        attempts.append(args[1])
        list(args[6]())

    dasiwa._encode_with_available_encoder = encode
    monkeypatch.setattr(stream_output, "_dasiwa_video_module", lambda: dasiwa)
    monkeypatch.setattr(stream_output.os, "unlink", lambda path: None)

    with pytest.raises(stream_output._VsrProcessingError, match="dual failure"):
        stream_output.MiniMaxH3StreamingVideoCombine().combine(
            images=images,
            guide={"performance_preset": "quality_two_stage", "rtx_deblur_mode": "off",
                   "target_width": 1920, "target_height": 1080, "postprocess_path": "rtx_vsr"},
            frame_rate=24.0, codec="Auto", container="MP4", bit_depth="8-bit", quality=20,
            log_level="Standard", pingpong=False, save_metadata=False, filename_prefix="stream-test",
            save_output=True, pass_frames=False, crop_to_audio=False, audio_codec="Auto",
            audio_bitrate="192k", save_first_frame=False, save_last_frame=False,
        )

    assert attempts == ["H.264"]
