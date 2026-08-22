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
        lambda *args: saved.append((args[1], args[7])) or f"{args[7]}-{args[1]}.png",
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
