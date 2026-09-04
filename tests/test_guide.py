from nodes import guide
from nodes import fish
import torch


def test_h3_reference_audio_normalizes_mono_to_stereo_before_native_vae():
    audio = {"waveform": torch.zeros(1, 1, 8), "sample_rate": 24000}

    normalized = guide.normalize_h3_reference_audio(audio)

    assert normalized["waveform"].shape == (1, 2, 8)
    assert torch.equal(normalized["waveform"][:, 0], normalized["waveform"][:, 1])
    assert normalized["sample_rate"] == 24000


def test_h3_reference_audio_keeps_stereo_channels_unchanged():
    waveform = torch.stack((torch.zeros(8), torch.ones(8))).unsqueeze(0)
    audio = {"waveform": waveform, "sample_rate": 32000}

    normalized = guide.normalize_h3_reference_audio(audio)

    assert torch.equal(normalized["waveform"], waveform)


def test_fl_backend_calls_native_image_to_video(monkeypatch):
    calls = []

    class NativeImageToVideo:
        @staticmethod
        def execute(*args):
            calls.append(args)
            return "conditioning", "latent"

    monkeypatch.setattr(guide, "native_node", lambda name: NativeImageToVideo)
    state = {
        "resolved_backend": "fl2va_model",
        "prompt": "prompt",
        "width": 1344,
        "height": 768,
        "length": 124,
        "first_frame": "first",
        "last_frame": "last",
    }

    result = guide.MiniMaxH3DirectorPlusGuide().apply(
        clip="clip",
        video_vae="video_vae",
        audio_vae="audio_vae",
        guide=state,
    )

    assert result == ("conditioning", "latent")
    assert calls == [("clip", "video_vae", "prompt", 1344, 768, 124, "first", "last")]


def test_ref_backend_calls_native_reference_to_video(monkeypatch):
    calls = []

    class NativeReferenceToVideo:
        @staticmethod
        def execute(*args):
            calls.append(args)
            return "conditioning", "latent"

    monkeypatch.setattr(guide, "native_node", lambda name: NativeReferenceToVideo)
    state = {
        "resolved_backend": "ref2va_model",
        "prompt": "prompt",
        "width": 1344,
        "height": 768,
        "length": 124,
        "ref_image_size": "match",
        "ref_images": {"ref_image_1": "image"},
        "ref_videos": {},
        "ref_video_audios": {},
        "ref_audios": {"ref_audio_1": "audio"},
    }

    result = guide.MiniMaxH3DirectorPlusGuide().apply(
        clip="clip",
        video_vae="video_vae",
        audio_vae="audio_vae",
        guide=state,
    )

    assert result == ("conditioning", "latent")
    assert calls == [(
        "clip", "video_vae", "audio_vae", "prompt", 1344, 768, 124, "match",
        {"ref_image_1": "image"}, {}, {}, {"ref_audio_1": "audio"},
    )]


def test_ref_backend_passes_stereo_reference_audio_to_native_node(monkeypatch):
    calls = []

    class NativeReferenceToVideo:
        @staticmethod
        def execute(*args):
            calls.append(args)
            return "conditioning", "latent"

    monkeypatch.setattr(guide, "native_node", lambda name: NativeReferenceToVideo)
    audio = {"waveform": torch.zeros(1, 1, 8), "sample_rate": 24000}
    state = {
        "resolved_backend": "ref2va_model",
        "prompt": "prompt",
        "width": 1344,
        "height": 768,
        "length": 124,
        "ref_image_size": "match",
        "ref_images": {"ref_image_1": "image"},
        "ref_videos": {},
        "ref_video_audios": {},
        "ref_audios": {"ref_audio_1": audio},
    }

    guide.MiniMaxH3DirectorPlusGuide().apply(
        clip="clip",
        video_vae="video_vae",
        audio_vae="audio_vae",
        guide=state,
    )

    forwarded_audio = calls[0][-1]["ref_audio_1"]
    assert forwarded_audio["waveform"].shape == (1, 2, 8)
    assert audio["waveform"].shape == (1, 1, 8)


def test_low_vram_preserves_native_dynamic_routes():
    """LOW_VRAM must use ComfyUI's dynamic patcher instead of CPU execution."""
    class Patcher:
        load_device = "cuda"
        offload_device = "cpu"

    clip = type("Clip", (), {"patcher": Patcher()})()
    video_vae = type("Vae", (), {"patcher": Patcher()})()
    audio_vae = type("Vae", (), {"patcher": Patcher()})()

    routed = guide._route_low_vram_inputs(
        clip, video_vae, audio_vae, {"performance_preset": "low_vram"}
    )

    assert routed == (clip, video_vae, audio_vae)


def test_model_router_requests_only_selected_model():
    node = guide.MiniMaxH3ModelRouter()

    assert node.check_lazy_status({"resolved_backend": "fl2va_model"}) == ["fl2va_model"]
    assert node.check_lazy_status({"resolved_backend": "ref2va_model"}) == ["ref2va_model"]


def test_model_router_outputs_only_the_selected_model():
    node = guide.MiniMaxH3ModelRouter()

    assert node.select({"resolved_backend": "fl2va_model"}, fl2va_model="fl", ref2va_model="ref") == ("fl",)
    assert node.select({"resolved_backend": "ref2va_model"}, fl2va_model="fl", ref2va_model="ref") == ("ref",)


def test_fish_bridge_bypasses_external_model_outside_fish_mode(monkeypatch):
    monkeypatch.setattr(fish, "fish_voice_clone_node", lambda: (_ for _ in ()).throw(AssertionError("不应加载 Fish")))

    result = fish.MiniMaxH3FishVoiceBridge().generate(
        guide={"voice_mode": "h3_reference"},
        reference_audio=None,
    )

    assert result == (None,)


def test_fish_bridge_generates_new_dialogue_only_in_fish_mode(monkeypatch):
    calls = []

    class VoiceClone:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return ("generated-audio",)

    monkeypatch.setattr(fish, "fish_voice_clone_node", VoiceClone)

    result = fish.MiniMaxH3FishVoiceBridge().generate(
        guide={
            "voice_mode": "fish_lock",
            "fish_model_path": "s2-pro-w4a16 (auto download)",
            "target_dialogue": "新的目标对白",
            "reference_transcript": "样本原文",
        },
        reference_audio="voice-sample",
    )

    assert result == ("generated-audio",)
    assert calls == [{
        "model_path": "s2-pro-w4a16 (auto download)",
        "text": "新的目标对白",
        "reference_audio": "voice-sample",
        "language": "auto",
        "device": "auto",
        "precision": "auto",
        "attention": "auto",
        "max_new_tokens": 0,
        "chunk_length": 200,
        "temperature": 0.8,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "seed": 0,
        "keep_model_loaded": True,
        "compile_model": False,
        "reference_text": "样本原文",
    }]


def test_fish_bridge_uses_cpu_device_in_low_vram_mode(monkeypatch):
    calls = []

    class VoiceClone:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return ("generated-audio",)

    monkeypatch.setattr(fish, "fish_voice_clone_node", VoiceClone)
    result = fish.MiniMaxH3FishVoiceBridge().generate(
        guide={
            "voice_mode": "fish_lock",
            "performance_preset": "low_vram",
            "fish_model_path": "s2-pro-w4a16 (auto download)",
            "target_dialogue": "对白",
        },
        reference_audio="voice-sample",
    )

    assert result == ("generated-audio",)
    assert calls[0]["device"] == "cpu"


def test_attach_stage2_keyframe_latents_reencodes_on_second_stage_grid():
    class FakeVae:
        def __init__(self):
            self.calls = []

        def encode(self, image):
            self.calls.append(image)
            return torch.zeros(1, 24, 1, image.shape[1] // 16, image.shape[2] // 16)

    vae = FakeVae()
    keyframes = [
        {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 60, 34)},
        {"resolved_frame_index": 123, "latent": torch.zeros(1, 24, 1, 60, 34)},
    ]
    cond = [[torch.zeros(1), {"minimax_keyframes": keyframes}]]
    state = {
        "two_stage_enabled": True,
        "width": 544,
        "height": 960,
        "second_stage_width": 1088,
        "second_stage_height": 1920,
        "first_frame": torch.rand(1, 100, 80, 3),
        "last_frame": torch.rand(1, 120, 90, 3),
    }

    guide.attach_stage2_keyframe_latents(cond, vae, state)

    assert [tuple(call.shape) for call in vae.calls] == [(1, 1920, 1088, 3), (1, 1920, 1088, 3)]
    assert keyframes[0]["latent_stage2"].shape == (1, 24, 1, 120, 68)
    assert keyframes[1]["latent_stage2"].shape == (1, 24, 1, 120, 68)
    assert keyframes[0]["latent"].shape == (1, 24, 1, 60, 34)


def test_attach_stage2_keyframe_latents_skips_when_two_stage_disabled():
    class ForbiddenVae:
        def encode(self, image):
            raise AssertionError("二采关闭时不得重编码")

    keyframes = [{"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 60, 34)}]
    cond = [[torch.zeros(1), {"minimax_keyframes": keyframes}]]
    state = {
        "two_stage_enabled": False,
        "width": 544,
        "height": 960,
        "second_stage_width": 1088,
        "second_stage_height": 1920,
        "first_frame": torch.rand(1, 100, 80, 3),
    }

    guide.attach_stage2_keyframe_latents(cond, ForbiddenVae(), state)

    assert "latent_stage2" not in keyframes[0]
