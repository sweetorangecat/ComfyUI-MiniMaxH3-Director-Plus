from nodes import guide
from nodes import fish


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


def test_low_vram_routes_clip_and_vaes_to_cpu(monkeypatch):
    class Device:
        def __init__(self, device_type):
            self.type = device_type

    monkeypatch.setitem(__import__("sys").modules, "torch", type("Torch", (), {"device": Device})())

    class Patcher:
        def __init__(self):
            self.load_device = "cuda"
            self.offload_device = "cuda"
            self.model = "model"

        def clone(self):
            clone = Patcher()
            clone.model = self.model
            return clone

    class Clip:
        def __init__(self):
            self.patcher = Patcher()

        def clone(self):
            clone = Clip()
            clone.patcher = self.patcher.clone()
            return clone

    class Vae:
        def __init__(self):
            self.patcher = Patcher()
            self.device = "cuda"
            self.output_device = "cuda"
            self.first_stage_model = "model"

    clip, video_vae, audio_vae = guide._route_low_vram_inputs(
        Clip(), Vae(), Vae(), {"performance_preset": "low_vram"}
    )

    assert clip.patcher.load_device.type == "cpu"
    assert video_vae.patcher.load_device.type == "cpu"
    assert audio_vae.patcher.load_device.type == "cpu"


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
