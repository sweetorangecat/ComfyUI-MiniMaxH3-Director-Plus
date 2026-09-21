from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


def face_module():
    assert importlib.util.find_spec("nodes.face_refine") is not None, "FaceRefine is not integrated"
    from nodes import face_refine
    return face_refine


def test_detector_discovery_without_impact_subpack(tmp_path, monkeypatch):
    module = face_module()
    import folder_paths
    detector = tmp_path / "ultralytics" / "bbox" / "face_yolov8m.pt"
    detector.parent.mkdir(parents=True)
    detector.touch()
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda name: [])
    assert "bbox/face_yolov8m.pt" in module.detector_names()
    assert module.detector_path("bbox/face_yolov8m.pt") == str(detector)
    with pytest.raises(FileNotFoundError):
        module.detector_path("bbox/missing.pt")


class FakeAudioVAE:
    audio_sample_rate = 32000

    def encode(self, waveform):
        return waveform.movedim(-1, 1)


class CloneableModel:
    def __init__(self):
        self.model_options = {"transformer_options": {"existing": True}}

    def clone(self):
        return deepcopy(self)


@pytest.mark.parametrize("audio_length", [2, 9])
def test_audio_lock_preserves_source_and_only_masks_audio(audio_length):
    module = face_module()
    from comfy.nested_tensor import NestedTensor
    video = torch.randn(1, 2, 3, 2, 2)
    audio_template = torch.zeros(1, 2, 5)
    reference = torch.ones(1, 2, 5)
    latent = {"samples": NestedTensor([video, audio_template, reference]), "metadata": "kept"}
    audio = {"waveform": torch.ones(1, 2, audio_length), "sample_rate": 32000}
    original_model = CloneableModel()
    model, result, original_audio = module.MiniMaxH3FaceAudioLock().run(
        model=original_model, av_latent=latent, audio_vae=FakeAudioVAE(), audio=audio)
    members = result["samples"].unbind()
    masks = result["noise_mask"].unbind()
    assert members[0] is video
    assert members[2] is reference
    assert members[1].shape == audio_template.shape
    assert members[1][..., :min(5, audio_length)].eq(1).all()
    if audio_length < 5:
        assert members[1][..., audio_length:].eq(0).all()
    assert masks[0].eq(1).all() and masks[1].eq(0).all()
    assert masks[2].eq(0).all()
    assert result["metadata"] == "kept"
    assert original_audio is audio
    assert latent["samples"].unbind()[1] is audio_template
    assert original_model.model_options == {"transformer_options": {"existing": True}}
    assert model.model_options["transformer_options"]["minimax_h3_lock_audio_clean"] is True


def test_audio_lock_rejects_non_av_latent():
    module = face_module()
    with pytest.raises(ValueError, match="AV"):
        module.MiniMaxH3FaceAudioLock().run(
            model=CloneableModel(), av_latent={"samples": torch.zeros(1)},
            audio_vae=FakeAudioVAE(), audio={})


def test_stitch_keeps_unselected_frames_and_background(monkeypatch):
    module = face_module()
    import comfy
    management = types.ModuleType("comfy.model_management")
    management.get_torch_device = lambda: torch.device("cpu")
    management.throw_exception_if_processing_interrupted = lambda: None
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setattr(comfy, "model_management", management, raising=False)
    base = torch.zeros(3, 32, 32, 3)
    crops = torch.ones(1, 16, 16, 3)
    transform = {"boxes": [(8, 8, 16, 16)], "canvas": (16, 16),
                 "src_size": (32, 32), "source": [1], "detected": [True], "weights": [1]}
    result, = module.MiniMaxH3FaceStitch().run(
        base_images=base, refined_crops=crops, transform=transform, paste_region="face_only",
        mask_dilation=0, feather=0, colour_match=0, blend=1, undetected_frames="skip")
    assert result.shape == base.shape
    assert result[0].eq(0).all() and result[2].eq(0).all()
    assert result[1, :6].eq(0).all()
    assert result[1, 16, 16].min() > 0.9
    assert base.eq(0).all()


def test_stitch_rejects_wrong_source_resolution():
    module = face_module()
    with pytest.raises(ValueError, match="source|原片"):
        module.MiniMaxH3FaceStitch().run(
            base_images=torch.zeros(1, 32, 32, 3), refined_crops=torch.ones(1, 16, 16, 3),
            transform={"src_size": (64, 64), "boxes": [(0, 0, 16, 16)], "canvas": (16, 16)},
            paste_region="face_only", mask_dilation=0, feather=0, colour_match=0, blend=1)


def test_tracking_uses_selected_face_and_preserves_frame_mapping(monkeypatch):
    module = face_module()
    import comfy
    management = types.ModuleType("comfy.model_management")
    management.get_torch_device = lambda: torch.device("cpu")
    management.throw_exception_if_processing_interrupted = lambda: None
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setattr(comfy, "model_management", management, raising=False)
    images = torch.zeros(5, 64, 128, 3)
    pick = {"frames": 5, "src_size": (128, 64),
            "boxes": [[[10, 20, 26, 40], [90, 20, 106, 40]]] * 5,
            "confs": [[0.9, 0.9]] * 5, "segments": [(0, 5)], "picks": []}
    result = module.MiniMaxH3FaceTrackCrop().run(
        images=images, detector="not-needed.pt", confidence=0.35, crop_factor=2.0,
        canvas_width=128, canvas_height=128, canvas_mode="manual", smooth_window=1,
        size_smooth_window=1, smooth_method="moving_average", size_mode="per_frame",
        select="right_most", identity_track=False, face_pick=pick)
    crops, transform, preview, report, width, height, count = result
    assert crops.shape == (5, 128, 128, 3)
    assert preview.shape[0] > 0
    assert (width, height, count) == (128, 128, 5)
    assert transform["source"] == list(range(5))
    assert all(box[0] > 60 for box in transform["boxes"])


def test_per_frame_strength_keeps_locked_audio_mask():
    module = face_module()
    from comfy.nested_tensor import NestedTensor
    video = torch.zeros(1, 2, 3, 2, 2)
    audio = torch.zeros(1, 2, 4)
    latent = {"samples": NestedTensor([video, audio]),
              "noise_mask": NestedTensor([torch.ones_like(video), torch.zeros_like(audio)])}
    result, report, model = module.MiniMaxH3FacePerFrameDenoise().run(
        model=CloneableModel(), av_latent=latent,
        transform={"boxes": [(0, 0, 60, 60), (0, 0, 150, 150), (0, 0, 240, 240)], "crop_factor": 2},
        denoise_multiplier_small_face=1, denoise_multiplier_large_face=0.35,
        face_px_small=30, face_px_large=120, gamma=1, smooth_frames=1)
    mask, audio_mask = result["noise_mask"].unbind()
    assert mask[0, 0, 0, 0, 0].item() == pytest.approx(1)
    assert mask[0, 0, -1, 0, 0].item() == pytest.approx(0.35)
    assert audio_mask.eq(0).all()
    assert latent["noise_mask"].unbind()[0].eq(1).all()


def test_face_refine_switch_requests_only_original_frames_when_disabled():
    module = face_module()
    switch = module.MiniMaxH3FaceRefineSwitch()
    guide = {"face_refine_mode": "off"}

    assert switch.check_lazy_status(
        guide, original_images=None, refined_images=None
    ) == ["original_images"]
    original = object()
    assert switch.select(
        guide, original_images=original, refined_images=None
    ) == (original,)


def test_face_refine_switch_requests_only_refined_frames_when_enabled():
    module = face_module()
    switch = module.MiniMaxH3FaceRefineSwitch()
    guide = {"face_refine_mode": "auto"}

    assert switch.check_lazy_status(
        guide, original_images=None, refined_images=None
    ) == ["refined_images"]
    refined = object()
    assert switch.select(
        guide, original_images=None, refined_images=refined
    ) == (refined,)
